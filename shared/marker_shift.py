"""Per-recording estimation of the amplifier's marker-to-sample time shift.

The NeoRec/NVX outlet publishes apparent Unix-epoch sample timestamps that are
back-dated by roughly five seconds: a sample carrying stamp ``S`` was in fact
acquired at about ``S + lag``. PsychoPy's markers are flip-locked and stamped
with ``time.time()``, so they are correct. Epoching a recording on its raw
marker timestamps therefore cuts windows about five seconds too late, and the
error is not the same in every file (4.8--5.7 s across the recordings measured
so far), so no single constant fixes a whole study.

The lag is measured, not assumed. LabRecorder writes stream chunks to the XDF
in arrival order, so the newest sample already committed to the file when a
marker chunk is written is the newest sample that had arrived by that marker's
wall-clock instant. The gap between the marker's timestamp and that sample's
timestamp is the amplifier's timestamp lag, sampled once per marker:

    lag = marker_unix_time - newest_already_written_sample_timestamp

Sixty-odd markers per recording give a median with a median absolute deviation
of 0.03--0.15 s. What the median still contains, and cannot separate, is the
real transport latency (one chunk is about 0.45 s, so at most a few tenths of a
second) and half a chunk of write granularity. The estimate is therefore good
to roughly +/- 0.3 s, which is what "plausible intervals" needs and is an order
of magnitude better than leaving the five seconds uncorrected.

Validation

The stimulus video shows a hand pictogram -- the movement cue -- two seconds
after every ``*_microrepeat`` marker, alternating with a neutral dot. In the
overt-movement recordings whose EMG is strong enough to cross-correlate against
the microrepeat schedule, applying the shift measured here puts the EMG burst
at +1.9 to +3.0 s after the corrected marker, i.e. on the hand pictogram, in
every case. The occipital evoked response to the same pictogram lands in the
same place. Uncorrected, both fall about 2.5 s *before* the marker, which is
before the block instruction is on screen at all.

This is an estimate of the outlet's timestamp back-dating, cross-checked
against physiology. It is not a photodiode measurement of the physical
acquisition offset, and it does not claim to be one.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_DATA_STREAM = "NVX136_Data"
DEFAULT_MARKER_STREAM = "PsychoPyMarkers"

# Sanity bounds on the measured lag. The window is deliberately wide: it is
# there to catch a mis-paired stream or a clock-domain mix-up, not to enforce an
# expectation about how large the lag ought to be.
MIN_PLAUSIBLE_LAG_S = -1.0
MAX_PLAUSIBLE_LAG_S = 60.0
MIN_MARKERS = 5
# Above this spread the per-marker measurements disagree enough that the median
# is no longer a tight summary; the run continues but says so.
SPREAD_WARN_S = 0.5
# Chunked delivery makes a slow drift normal. This threshold flags a recording
# whose lag walked far enough during the run for one shift to fit it poorly.
DRIFT_WARN_MS_PER_S = 3.0

_ITEM_BYTES = {"int8": 1, "int16": 2, "int32": 4, "int64": 8,
               "float32": 4, "double64": 8}

_TAG_STREAM_HEADER = 2
_TAG_SAMPLES = 3


class MarkerShiftError(ValueError):
    """The marker time shift could not be measured from this recording."""


@dataclass(frozen=True)
class MarkerShiftEstimate:
    """One recording's measured marker-to-sample correction.

    ``shift_s`` is the number to add to every marker onset, and is the negated
    lag: the data is stamped early, so the markers have to move earlier to meet
    it.
    """

    shift_s: float
    lag_s: float
    spread_s: float               # median absolute deviation across markers
    n_markers: int
    drift_ms_per_s: float
    data_stream: str
    marker_stream: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        text = (f"marker time shift {self.shift_s:+.3f} s "
                f"(amplifier timestamp lag {self.lag_s:.3f} s "
                f"+/- {self.spread_s:.3f} s over {self.n_markers} markers)")
        for warning in self.warnings:
            text += f"; {warning}"
        return text


def _read_varlen(handle) -> int | None:
    """Read XDF's variable-length integer: one count byte, then that many."""
    raw = handle.read(1)
    if not raw:
        return None
    width = raw[0]
    if width == 0:
        return 0
    return int.from_bytes(handle.read(width), "little")


def _stream_metadata(handle, chunk_start: int, chunk_len: int) -> tuple[int, dict]:
    stream_id = struct.unpack("<I", handle.read(4))[0]
    raw = handle.read(chunk_start + chunk_len - handle.tell())
    root = ET.fromstring(raw.decode("utf-8", "replace"))

    def number(tag: str) -> float:
        try:
            return float(root.findtext(tag) or 0.0)
        except ValueError:
            return 0.0

    return stream_id, {
        "name": root.findtext("name") or "",
        "format": (root.findtext("channel_format") or "").strip().lower(),
        "channels": int(number("channel_count")),
        "srate": number("nominal_srate"),
    }


def _skip_sample_values(handle, fmt: str, channels: int) -> None:
    if fmt == "string":
        for _ in range(channels):
            length = _read_varlen(handle) or 0
            handle.read(length)
    else:
        handle.read(_ITEM_BYTES.get(fmt, 8) * channels)


def _read_stream_headers(path: Path) -> dict[int, dict]:
    """Stream id -> metadata, skipping every payload.

    A separate pass, because which stream is the data stream can depend on all
    of the headers -- the fallback below picks the widest continuous one -- and
    that has to be settled before the first sample chunk is interpreted.
    """
    meta: dict[int, dict] = {}
    with open(path, "rb") as handle:
        if handle.read(4) != b"XDF:":
            raise MarkerShiftError(f"{path.name} is not an XDF file")
        while True:
            chunk_len = _read_varlen(handle)
            if chunk_len is None:
                break
            chunk_start = handle.tell()
            tag = struct.unpack("<H", handle.read(2))[0]
            if tag == _TAG_STREAM_HEADER:
                stream_id, info = _stream_metadata(handle, chunk_start, chunk_len)
                meta[stream_id] = info
            handle.seek(chunk_start + chunk_len)
    return meta


def _select_streams(meta: dict[int, dict], path: Path,
                    data_stream: str, marker_stream: str) -> tuple[int, int]:
    """Resolve the data and marker stream ids the way the app's loader does.

    The data stream falls back to the continuous stream with the most channels
    -- the amplifier, never a one-channel marker or event stream -- so that a
    recording the app can epoch is also one this can measure. The marker stream
    gets no such fallback: nothing about an irregular stream's shape separates
    markers from an events channel, and guessing wrong would measure the lag
    against the wrong clock without saying so.
    """
    named = [sid for sid, info in meta.items() if info["name"] == data_stream]
    if named:
        data_id = named[0]
    else:
        continuous = [sid for sid, info in meta.items() if info["srate"] > 0]
        if not continuous:
            raise MarkerShiftError(
                f"{path.name} has no {data_stream!r} stream and no other "
                f"continuous-data stream to fall back on")
        data_id = max(continuous, key=lambda sid: meta[sid]["channels"])

    markers = [sid for sid, info in meta.items()
               if info["name"] == marker_stream]
    if not markers:
        raise MarkerShiftError(
            f"{path.name} has no {marker_stream!r} stream, so the marker time "
            f"shift cannot be measured from it")
    if meta[data_id]["srate"] <= 0:
        raise MarkerShiftError(
            f"{path.name} reports a non-positive rate for "
            f"{meta[data_id]['name']!r}; the position of the newest written "
            f"sample cannot be derived")
    return data_id, markers[0]


def _collect_lags(path: Path, data_stream: str, marker_stream: str) \
        -> tuple[np.ndarray, np.ndarray, dict]:
    """Walk the XDF in write order and pair every marker with the newest sample.

    Only each data chunk's header is parsed -- its first timestamp and its
    sample count -- and its payload is skipped, so a 46 MB recording is measured
    in about ten milliseconds. Marker chunks are walked sample by sample because
    an irregular stream stamps every sample individually.

    LabRecorder omits the explicit timestamp on a chunk that continues the
    previous one at the nominal rate; those chunks advance a running clock that
    the next explicit stamp resets, so a wrong nominal rate cannot accumulate
    across the recording.
    """
    meta = _read_stream_headers(path)
    data_id, marker_id = _select_streams(meta, path, data_stream, marker_stream)
    srate = meta[data_id]["srate"]
    marker_info = meta[marker_id]

    newest_sample_ts: float | None = None
    lags: list[float] = []
    marker_times: list[float] = []
    seen = {"data_chunks": 0, "marker_samples": 0,
            "data_stream": meta[data_id]["name"],
            "marker_stream": marker_info["name"]}

    with open(path, "rb") as handle:
        handle.read(4)
        while True:
            chunk_len = _read_varlen(handle)
            if chunk_len is None:
                break
            chunk_start = handle.tell()
            tag = struct.unpack("<H", handle.read(2))[0]

            if tag == _TAG_SAMPLES:
                stream_id = struct.unpack("<I", handle.read(4))[0]
                n_samples = _read_varlen(handle) or 0
                if stream_id == data_id and n_samples:
                    stamped = handle.read(1)[0]
                    if stamped == 8:
                        first = struct.unpack("<d", handle.read(8))[0]
                    elif newest_sample_ts is not None:
                        first = newest_sample_ts + 1.0 / srate
                    else:
                        first = float("nan")
                    newest_sample_ts = first + (n_samples - 1) / srate
                    seen["data_chunks"] += 1
                elif stream_id == marker_id and n_samples:
                    for _ in range(n_samples):
                        stamped = handle.read(1)[0]
                        stamp = (struct.unpack("<d", handle.read(8))[0]
                                 if stamped == 8 else float("nan"))
                        _skip_sample_values(handle, marker_info["format"],
                                            marker_info["channels"])
                        seen["marker_samples"] += 1
                        # Markers written before the first data chunk have no
                        # sample to be measured against and are simply skipped.
                        if (newest_sample_ts is not None
                                and np.isfinite(stamp)
                                and np.isfinite(newest_sample_ts)):
                            lags.append(stamp - newest_sample_ts)
                            marker_times.append(stamp)

            handle.seek(chunk_start + chunk_len)

    return np.asarray(lags, float), np.asarray(marker_times, float), seen


def estimate_marker_shift(
        path,
        data_stream: str = DEFAULT_DATA_STREAM,
        marker_stream: str = DEFAULT_MARKER_STREAM) -> MarkerShiftEstimate:
    """Measure the marker time shift this recording needs.

    Raises :class:`MarkerShiftError` rather than returning a fallback: a silent
    default here would be indistinguishable, in the output, from a measurement,
    and the shift is worth five seconds of epoch position. The caller is
    expected to report the failure and let the operator enter a shift by hand.
    """
    path = Path(path)
    lags, marker_times, seen = _collect_lags(path, data_stream, marker_stream)

    if lags.size < MIN_MARKERS:
        raise MarkerShiftError(
            f"{path.name} has only {lags.size} marker(s) that follow a data "
            f"chunk ({seen['marker_samples']} marker samples, "
            f"{seen['data_chunks']} data chunks); at least {MIN_MARKERS} are "
            f"needed to measure the shift")

    lag = float(np.median(lags))
    spread = float(np.median(np.abs(lags - lag)))
    if not np.isfinite(lag) or not MIN_PLAUSIBLE_LAG_S <= lag <= MAX_PLAUSIBLE_LAG_S:
        raise MarkerShiftError(
            f"{path.name} gives an implausible amplifier timestamp lag of "
            f"{lag:.3f} s; expected {MIN_PLAUSIBLE_LAG_S:g}..{MAX_PLAUSIBLE_LAG_S:g} s. "
            f"Check that {data_stream!r} and {marker_stream!r} really are this "
            f"recording's data and marker streams")

    elapsed = marker_times - marker_times[0]
    drift = (float(np.polyfit(elapsed, lags, 1)[0] * 1000.0)
             if elapsed[-1] > 0 else 0.0)

    warnings: list[str] = []
    if spread > SPREAD_WARN_S:
        warnings.append(
            f"per-marker lag varies by +/-{spread:.2f} s, so one shift fits "
            f"this recording only loosely")
    if abs(drift) > DRIFT_WARN_MS_PER_S:
        span = drift * (elapsed[-1] if elapsed.size else 0.0) / 1000.0
        warnings.append(
            f"the lag drifts {drift:+.1f} ms/s ({span:+.2f} s across the "
            f"recording); the median fits the middle better than the ends")

    return MarkerShiftEstimate(
        shift_s=-lag,
        lag_s=lag,
        spread_s=spread,
        n_markers=int(lags.size),
        drift_ms_per_s=drift,
        data_stream=seen["data_stream"],
        marker_stream=seen["marker_stream"],
        warnings=tuple(warnings),
    )


def resolve_marker_shift(
        path,
        auto: bool,
        manual_shift_s: float,
        data_stream: str = DEFAULT_DATA_STREAM,
        marker_stream: str = DEFAULT_MARKER_STREAM) \
        -> tuple[float, str, MarkerShiftEstimate | None]:
    """Return ``(shift_s, note, estimate)`` for one recording.

    With ``auto`` off the manual value is used unchanged, but the measurement is
    still attempted and reported beside it, so an operator who has overridden
    the detector can see what the recording itself says. A failure to measure is
    never allowed to stop a manual run.
    """
    if auto:
        try:
            estimate = estimate_marker_shift(path, data_stream, marker_stream)
        except MarkerShiftError as exc:
            raise MarkerShiftError(
                f"{exc}. Untick automatic detection and enter a marker time "
                f"shift by hand to process this recording anyway.") from exc
        return estimate.shift_s, f"detected {estimate.summary()}", estimate

    try:
        estimate = estimate_marker_shift(path, data_stream, marker_stream)
    except (MarkerShiftError, OSError, struct.error, ET.ParseError):
        return (float(manual_shift_s),
                f"marker time shift {float(manual_shift_s):+.3f} s "
                f"(entered by hand; auto-detection unavailable)", None)
    note = (f"marker time shift {float(manual_shift_s):+.3f} s "
            f"(entered by hand; this recording measures "
            f"{estimate.shift_s:+.3f} s)")
    return float(manual_shift_s), note, estimate
