"""Loading, timing, EMG preprocessing, epoching, and plotting.

XDF timing comes from embedded event markers and its clock correction is
resolved per recording. For the reviewed BrainVision session, timing is
instead reconstructed from the sample-locked first S3 marker, ``cond seq.txt``,
and the matching video microrepeat CSV; no XDF clock shift is applied. The
signal pipeline uses a bipolar difference, volts to millivolts, a causal 50 Hz
notch, a causal 20--95 Hz Butterworth band-pass (valid at 250 Hz),
Teager--Kaiser energy, and a 100 ms window / 20 ms step envelope.

Analysis and output are separate steps. :func:`analyze_batch` produces the
epochs, the per-trial verdicts and everything the figures are drawn from, and
writes nothing; :func:`save_batch_outputs` turns that same result into the file
tree. The desktop application analyses once and draws from memory, so opening a
participant does not first cost several hundred PNG files, and it writes only
when asked to. The command line still does both in one call.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, tf2sos

from shared import brainvision
from shared.marker_shift import resolve_marker_shift


PROJECT_DIR = Path(__file__).resolve().parent.parent
# Where saved figures and tables go, as a path *relative to the application*.
#
# It is deliberately not an absolute path. The root is written to the settings
# file, and the settings file travels with the application: a root recorded on
# the machine an analysis was set up on would send a copied installation's
# results back to the original folder, on a drive the copy may not even have.
# A relative root therefore hangs off the application directory -- not off
# whatever directory the program happened to be started from, which is what
# a bare ``Path.resolve()`` would have used.
DEFAULT_OUTPUT_ROOT = "output"
EXTENSIONS = {"xdf": ".xdf", "brainvision": ".vhdr"}
MOTOR_CUE_DELAY_S = 2.0


@dataclass
class ProcessingConfig:
    data_dir: str
    recordings: list[str]
    participant: str = ""
    file_type: str = "xdf"
    target_fs: float | None = None
    data_stream: str = "NVX136_Data"
    marker_stream: str = "PsychoPyMarkers"
    left_channels: tuple[str, str] | None = ("Aux 1.1", "Aux 1.2")
    right_channels: tuple[str, str] | None = ("Aux 2.1", "Aux 2.2")
    left_condition: str = "left_microrepeat"
    right_condition: str = "right_microrepeat"
    rest_condition: str = "rest"
    envelope: str = "tkeo"             # tkeo | rms
    notch_hz: float = 50.0
    notch_width_hz: float = 1.0
    band_low_hz: float = 20.0
    band_high_hz: float = 95.0
    filter_order: int = 4
    window_ms: float = 100.0
    step_ms: float = 20.0
    pre_s: float = 2.0
    post_s: float = 2.0
    trial_tail_s: float = 0.2           # classification tolerance after trial end
    rest_trim_start_s: float = 1.0
    rest_trim_end_s: float = 2.0
    peak_multiplier: float = 7.0        # height a peak must reach, x background
    background_multiplier: float = 6.0  # shoulder bar the width is measured at
    pre_reference_start_s: float = 0.5  # quiet cue interval used by branch 2
    pre_reference_end_s: float = 1.8
    secondary_pre_multiplier: float = 3.0
    secondary_width_multiplier: float = 3.0
    min_burst_ms: float = 50.0          # width required of a qualifying peak
    rest_fpr_warn: float = 0.05         # own-rest hit rate that voids a run
    rest_pseudotrial_step_s: float = 0.5
    auto_marker_shift: bool = True      # measure the shift from each recording
    marker_shift_s: float = -4.0       # added to all embedded marker onsets
    trial_figures: bool = True
    input_unit: str = "V"               # V | mV
    output_root: str = DEFAULT_OUTPUT_ROOT
    remove_mri_artifacts: bool = False
    ecg_channel: str = "ECG"
    mri_tr_s: float = 2.5
    video: int | None = None
    single_file_output: bool = False

    def validate(self) -> None:
        if not self.recordings:
            raise ValueError("at least one recording is required")
        self.file_type = str(self.file_type).strip().lower()
        if self.file_type not in EXTENSIONS:
            raise ValueError("EMG file_type must be 'xdf' or 'brainvision'")
        if self.remove_mri_artifacts and self.file_type != "brainvision":
            raise ValueError("MNE MRI correction currently requires a BrainVision .vhdr file")
        if self.remove_mri_artifacts and self.input_unit != "V":
            raise ValueError("BrainVision MRI inputs are calibrated to volts; input_unit must be V")
        if not np.isfinite(self.mri_tr_s) or self.mri_tr_s <= 0:
            raise ValueError("MRI TR must be positive and finite")
        if self.video not in (None, 1, 2, 3):
            raise ValueError("video must be 1, 2, 3 or null (automatic)")
        if self.rest_trim_start_s < 0 or self.rest_trim_end_s < 0:
            raise ValueError("rest trims must not be negative")
        if self.trial_tail_s < 0:
            raise ValueError("trial classification tail must not be negative")
        if self.background_multiplier <= 0:
            raise ValueError("background multiplier must be positive")
        if self.peak_multiplier <= 0:
            raise ValueError("peak multiplier must be positive")
        if self.peak_multiplier < self.background_multiplier:
            raise ValueError(
                "the peak height bar must not sit below the shoulder bar its "
                "width is measured at, or the width test could never be met")
        if self.pre_reference_start_s < 0:
            raise ValueError("pre-movement reference start must not be negative")
        if self.pre_reference_end_s <= self.pre_reference_start_s:
            raise ValueError(
                "pre-movement reference end must be after its start")
        if self.secondary_pre_multiplier <= 0:
            raise ValueError("secondary pre-movement multiplier must be positive")
        if self.secondary_width_multiplier <= 0:
            raise ValueError("secondary width multiplier must be positive")
        if self.secondary_width_multiplier > self.secondary_pre_multiplier:
            raise ValueError(
                "secondary width bar must not exceed the secondary "
                "pre-movement peak height bar")
        if not 0 < self.rest_fpr_warn <= 1:
            raise ValueError("rest false-positive warning level must be in (0, 1]")
        if self.min_burst_ms < 0:
            raise ValueError("minimum burst duration must not be negative")
        if self.rest_pseudotrial_step_s <= 0:
            raise ValueError("rest pseudo-trial step must be positive")
        if not np.isfinite(self.marker_shift_s):
            raise ValueError("marker time shift must be finite")
        enabled_pairs = [pair for pair in
                         (self.left_channels, self.right_channels) if pair]
        if not enabled_pairs:
            raise ValueError("at least one hand needs a bipolar EMG pair")
        if any(len(pair) != 2 for pair in enabled_pairs):
            raise ValueError(
                "each enabled hand needs exactly two channels for bipolar EMG")
        if self.envelope not in {"tkeo", "rms"}:
            raise ValueError("envelope must be 'tkeo' or 'rms'")
        if self.window_ms <= 0 or self.step_ms <= 0:
            raise ValueError("envelope window and step must be positive")
        if self.band_low_hz <= 0 or self.band_high_hz <= self.band_low_hz:
            raise ValueError("invalid EMG band-pass limits")


@dataclass
class EventSet:
    onsets: np.ndarray
    durations: np.ndarray


@dataclass
class LoadedRecording:
    signals: dict[str, np.ndarray]
    fs: float
    duration_s: float
    marker_events: dict[str, EventSet] = field(default_factory=dict)
    video_onset_s: float | None = None
    event_source: str = "xdf"
    provenance: dict = field(default_factory=dict)


@dataclass
class HandAnalysis:
    """Everything one hand's figures and verdicts are drawn from.

    Held in memory so any figure can be redrawn later without re-reading the
    recording, and so the application and the file writer draw the same trial
    from the same arrays the classifier was given.
    """

    grid: np.ndarray                 # seconds from the corrected event onset
    epochs: np.ndarray               # (n_trials, grid.size) envelope
    onsets: np.ndarray               # corrected marker onsets, seconds
    duration: float                  # scheduled motor-trial duration
    classification_duration: float   # that duration plus the tolerance tail
    rest_baseline: float
    metrics: TrialMetricSet | None = None
    metrics_error: str = ""          # why there are no metrics, when there are none


@dataclass
class HandResult:
    hand: str
    condition: str
    channels: tuple[str, str]
    n_trials: int
    movement_duration_s: float
    rest_baseline: float
    movement_mean: float
    movement_rest_ratio: float
    figure_path: str
    trial_dir: str = ""
    trial_metrics_csv: str = ""
    n_high_trials: int | None = None
    background: float = np.nan
    peak_threshold: float = np.nan
    burst_threshold: float = np.nan
    min_burst_ms: float = np.nan
    trial_tail_ms: float = np.nan
    rest_fpr: float = np.nan
    primary_rest_fpr: float = np.nan
    secondary_rest_fpr: float = np.nan
    threshold_note: str = ""
    analysis: HandAnalysis | None = None

    @property
    def high_percent(self) -> float:
        if self.n_high_trials is None or not self.n_trials:
            return np.nan
        return 100.0 * self.n_high_trials / self.n_trials


@dataclass
class RecordingResult:
    recording: str
    source_path: str
    fs: float
    timing_note: str
    hands: list[HandResult]
    marker_shift_s: float = np.nan     # what was actually applied to this file
    detected_shift_s: float = np.nan   # what the recording itself measures
    provenance: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    participant: str
    output_dir: str
    recordings: list[RecordingResult]
    summary_csv: str


@dataclass
class TrialMetricSet:
    """Peak detection for one hand's trials, referenced to its own rest."""

    high: np.ndarray                 # either detection branch qualifies
    primary_high: np.ndarray         # 7x rest + 6x-rest width branch
    secondary_high: np.ndarray       # 3x preparation + 3x-prep width branch
    peak_ratio: np.ndarray           # tallest sample in the trial, x background
    longest_burst_ms: np.ndarray     # widest above-shoulder stretch per trial
    burst_energy: np.ndarray         # descriptive only; not part of the rule
    active_fraction: np.ndarray      # descriptive only; not part of the rule
    background: float                # median clean-rest envelope
    peak_threshold: float            # peak_multiplier x background
    burst_threshold: float           # background_multiplier x background
    pre_movement_background: np.ndarray  # median from +0.5 through +1.8 s
    secondary_peak_ratio: np.ndarray     # post-preparation peak, x rest
    peak_pre_ratio: np.ndarray           # post-preparation peak, x preparation
    secondary_longest_burst_ms: np.ndarray
    secondary_burst_threshold: np.ndarray
    min_burst_ms: float
    rest_fpr: float                  # same detector run on this run's own rest
    primary_rest_fpr: float
    secondary_rest_fpr: float
    n_rest_windows: int
    rest_unreliable: bool            # rest_fpr above config.rest_fpr_warn


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unnamed"


def parse_pair(text: str) -> tuple[str, str]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("enter two comma-separated channels")
    return parts[0], parts[1]


def parse_optional_pair(text: str) -> tuple[str, str] | None:
    """Parse one pair, or disable that hand when the field is blank."""
    return None if not str(text).strip() else parse_pair(text)


def hand_specs(config: ProcessingConfig):
    """Configured hands only, preserving the established left/right order."""
    return [
        (hand, condition, pair)
        for hand, condition, pair in (
            ("left", config.left_condition, config.left_channels),
            ("right", config.right_condition, config.right_channels),
        )
        if pair is not None
    ]


def recording_path(data_dir: str, name: str, file_type: str = "xdf") -> Path:
    candidate = Path(name.strip().strip('"'))
    file_type = str(file_type).strip().lower()
    if file_type not in EXTENSIONS:
        raise ValueError(f"unsupported EMG file type {file_type!r}")
    extension = EXTENSIONS[file_type]
    if not candidate.is_absolute():
        candidate = Path(data_dir) / candidate
    if not candidate.suffix:
        candidate = candidate.with_suffix(extension)
    if candidate.is_file():
        return candidate

    # Helpful error for transposed tokens such as mi_finished1 vs mi1_finished.
    import difflib
    folder = candidate.parent
    choices = list(folder.glob(f"*{extension}")) if folder.is_dir() else []
    close = difflib.get_close_matches(candidate.stem, [p.stem for p in choices], n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise FileNotFoundError(f"recording not found: {candidate}.{hint}")


def _xdf_stream(streams, name: str, continuous: bool):
    for stream in streams:
        if stream["info"].get("name") and stream["info"]["name"][0] == name:
            return stream
    if continuous:
        candidates = [s for s in streams
                      if float(s["info"]["nominal_srate"][0]) > 0]
        if candidates:
            return max(candidates,
                       key=lambda s: int(s["info"]["channel_count"][0]))
    raise ValueError(f"XDF stream {name!r} not found")


def _xdf_labels(stream) -> list[str]:
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        return [channel["label"][0] for channel in channels]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("XDF data stream has no channel labels") from exc


def _extract_signals(data: np.ndarray, labels: list[str], requested: list[str],
                     source_name: str) -> dict[str, np.ndarray]:
    lut = {_norm(label): i for i, label in enumerate(labels)}
    result = {}
    for token in requested:
        key = _norm(token)
        if key in lut:
            index = lut[key]
        elif token.strip().isdigit():
            index = int(token) - 1  # numeric channel specifications are 1-based
            if not 0 <= index < data.shape[1]:
                raise ValueError(f"channel {token} is outside 1..{data.shape[1]}")
        else:
            aux = [label for label in labels if "aux" in label.lower()
                   or "emg" in label.lower()]
            raise ValueError(
                f"channel {token!r} not found in {source_name}; available EMG/Aux: {aux}")
        result[token] = np.asarray(data[:, index], dtype=np.float64)
    return result


def _marker_event_sets(marker, data_start: float, duration_s: float,
                       wanted: set[str]) -> tuple[dict[str, EventSet], float | None]:
    rows = marker.get("time_series", [])
    stamps = np.asarray(marker.get("time_stamps", []), dtype=np.float64)
    values = [str(row[0] if isinstance(row, (list, tuple, np.ndarray)) else row)
              for row in rows]
    relative = stamps - data_start
    if stamps.size and (np.nanmax(relative) < -1.0 or np.nanmin(relative) > duration_s + 1.0):
        raise ValueError(
            "marker and data timestamps are in incompatible clocks; the "
            "recording cannot be epoched from its own markers")

    video = next((float(t) for value, t in zip(values, relative)
                  if value.startswith("video_onset")), None)
    unique_times = np.unique(relative[np.isfinite(relative)])
    events: dict[str, EventSet] = {}
    for condition in wanted:
        onsets = relative[np.array([value == condition for value in values])]
        durations = []
        for onset in onsets:
            later = unique_times[unique_times > onset + 1e-6]
            durations.append(float(later[0] - onset) if later.size else np.nan)
        events[condition] = EventSet(
            np.asarray(onsets, dtype=np.float64),
            np.asarray(durations, dtype=np.float64))
    return events, video


def load_xdf(path: Path, requested: list[str], config: ProcessingConfig) \
        -> LoadedRecording:
    import pyxdf
    streams, _ = pyxdf.load_xdf(
        str(path), synchronize_clocks=False, dejitter_timestamps=False)
    data_stream = _xdf_stream(streams, config.data_stream, continuous=True)
    data = np.asarray(data_stream["time_series"])
    labels = _xdf_labels(data_stream)
    fs = float(data_stream["info"]["nominal_srate"][0])
    stamps = np.asarray(data_stream["time_stamps"], dtype=np.float64)
    duration = float(stamps[-1] - stamps[0]) if stamps.size > 1 else data.shape[0] / fs
    signals = _extract_signals(data, labels, requested, path.name)

    marker = _xdf_stream(streams, config.marker_stream, continuous=False)
    wanted = {config.left_condition, config.right_condition, config.rest_condition}
    events, video = _marker_event_sets(marker, float(stamps[0]), duration, wanted)
    return LoadedRecording(signals, fs, duration, events, video, "xdf")


def load_brainvision(path: Path, requested: list[str],
                     config: ProcessingConfig) -> LoadedRecording:
    """Load EMG inputs and scheduled events from a BrainVision run."""
    wanted = {config.rest_condition}
    wanted.update(condition for _, condition, _ in hand_specs(config))
    data, raw_events, video, info = brainvision.load(
        path, wanted, pick_labels=requested, video=config.video, mri_config=config)
    signals = {
        label: np.asarray(data[:, index], dtype=np.float64)
        for index, label in enumerate(requested)
    }
    events = {
        name: EventSet(np.asarray(onsets, dtype=np.float64),
                       np.asarray(durations, dtype=np.float64))
        for name, (onsets, durations) in raw_events.items()
    }
    duration = info["duration_s"]
    return LoadedRecording(
        signals, float(info["processing_fs"]), duration, events, video,
        "brainvision", dict(info))


def _resample_loaded(recording: LoadedRecording, target_fs: float | None) \
        -> LoadedRecording:
    native_fs = float(recording.fs)
    if target_fs is not None and (
            not np.isfinite(target_fs) or float(target_fs) <= 0):
        raise ValueError("EMG target sampling rate must be positive")
    provenance = recording.provenance
    provenance.setdefault("recording_fs", native_fs)
    if not target_fs or abs(native_fs - target_fs) < 1e-9:
        provenance.update({
            "processing_fs": native_fs,
            "resampled": provenance.get("resampled", False),
            "resampling_method": provenance.get("resampling_method", "none"),
        })
        return recording
    ratio = Fraction(float(target_fs) / native_fs).limit_denominator(1000)
    recording.signals = {
        name: resample_poly(values, ratio.numerator, ratio.denominator)
        for name, values in recording.signals.items()
    }
    recording.fs = float(target_fs)
    provenance.update({
        "processing_fs": float(target_fs),
        "resampled": True,
        "resampling_method": "scipy.signal.resample_poly",
    })
    return recording


def filter_emg(values: np.ndarray, fs: float, config: ProcessingConfig) -> np.ndarray:
    if config.band_high_hz >= fs / 2:
        raise ValueError(
            f"band-pass high edge {config.band_high_hz:g} Hz must be below "
            f"Nyquist ({fs / 2:g} Hz)")
    scale = 1000.0 if config.input_unit.lower() == "v" else 1.0
    notch_q = config.notch_hz / config.notch_width_hz
    notch = tf2sos(*iirnotch(config.notch_hz, notch_q, fs=fs))
    band = butter(config.filter_order,
                  (config.band_low_hz, config.band_high_hz),
                  btype="bandpass", output="sos", fs=fs)
    filtered = sosfilt(notch, np.asarray(values, dtype=float) * scale)
    return sosfilt(band, filtered)


def tkeo(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    if values.size >= 3:
        out[1:-1] = values[1:-1] ** 2 - values[:-2] * values[2:]
        out[0], out[-1] = out[1], out[-2]
    return out


def envelope_trace(values_mv: np.ndarray, fs: float, config: ProcessingConfig) \
        -> tuple[np.ndarray, np.ndarray]:
    win = int(round(config.window_ms * fs / 1000.0))
    step = int(round(config.step_ms * fs / 1000.0))
    if win < 1 or step < 1 or values_mv.size < win:
        return np.array([]), np.array([])
    starts = np.arange(0, values_mv.size - win + 1, step)
    source = tkeo(values_mv) if config.envelope == "tkeo" else values_mv
    windows = np.lib.stride_tricks.sliding_window_view(source, win)[starts]
    if config.envelope == "tkeo":
        envelope = np.mean(windows, axis=1)
    else:
        envelope = np.sqrt(np.mean(windows ** 2, axis=1))
    centers = (starts + win / 2.0) / fs
    return envelope, centers


def epoch_trace(values: np.ndarray, centers: np.ndarray, onsets: np.ndarray,
                grid: np.ndarray) -> np.ndarray:
    epochs = np.full((onsets.size, grid.size), np.nan)
    for index, onset in enumerate(onsets):
        epochs[index] = np.interp(onset + grid, centers, values,
                                  left=np.nan, right=np.nan)
    return epochs


def _rest_baseline(values: np.ndarray, centers: np.ndarray, events: EventSet,
                   trim_end_s: float) -> float:
    """Resting reference taken from the clean part of each rest block.

    Two departures from a plain pooled mean. The tail of every rest window
    already carries the next block: movement starts around a second before
    the task-block cue, and TKEO is an energy, so those few samples otherwise
    lift the reference by one to two orders of magnitude. ``trim_end_s`` drops
    them. The median across blocks is then taken instead of the mean, because
    a mean is set by whichever single block holds the largest artefact.
    """
    block_means = []
    for onset, duration in zip(events.onsets, events.durations):
        if not np.isfinite(duration):
            continue
        usable = float(duration) - trim_end_s
        if usable <= 0:
            continue
        chunk = values[(centers >= onset) & (centers < onset + usable)]
        if chunk.size:
            block_means.append(float(np.mean(chunk)))
    return float(np.median(block_means)) if block_means else np.nan


def _active_fraction(values: np.ndarray, centers: np.ndarray, onset: float,
                     duration: float, amplitude_threshold: float) -> float:
    """Fraction of envelope samples above threshold inside one motor trial."""
    chunk = values[_motor_segment(centers, onset, duration)]
    chunk = chunk[np.isfinite(chunk)]
    if not chunk.size:
        return np.nan
    return float(np.mean(chunk > amplitude_threshold))


def _trial_peak(values: np.ndarray, centers: np.ndarray, onset: float,
                duration: float) -> float:
    """Tallest envelope sample inside one motor trial, in raw envelope units.

    Height is what separates muscle activity from background here. On this
    study's 40 recordings the tallest sample of an ordinary rest window sits at
    2x its own background (median), while an overt-movement trial reaches
    1700x. Two orders of magnitude of headroom is why the verdict is anchored
    on height first and shape second.
    """
    chunk = values[_motor_segment(centers, onset, duration)]
    chunk = chunk[np.isfinite(chunk)]
    return float(np.max(chunk)) if chunk.size else np.nan


def _longest_active_run(values: np.ndarray, centers: np.ndarray, onset: float,
                        duration: float, amplitude_threshold: float,
                        bridge: int = 0) -> float:
    """Length in samples of the widest above-threshold burst.

    The active fraction alone cannot separate one sustained contraction from
    several unrelated transients that happen to sum to the same total. On this
    envelope a single 120 ms artefact already spans six samples, which is a
    third of a 5% threshold over a 4 s trial, so the fraction on its own flags
    twitches and cable knocks as movement.

    A burst is delimited the same way ``_peak_burst_energy`` delimits one, so
    the width and the energy reported for a trial describe the *same* event:
    dips shorter than ``bridge`` samples do not split it, because the envelope
    is a moving window that wide and cannot resolve them as separate events in
    the first place. Passing ``bridge=0`` recovers the strictly contiguous run.

    Samples rather than seconds, because the caller compares against a sample
    count: converting both sides to milliseconds first lets floating-point
    error in the envelope spacing decide a trial that sits exactly on the
    minimum burst duration.

    Only samples inside the motor trial count. A burst that straddles the
    onset contributes just the part that falls within it, and one entirely in
    the pre-cue lead-in or the post-trial tail contributes nothing.
    """
    chunk = values[_motor_segment(centers, onset, duration)]
    chunk = chunk[np.isfinite(chunk)]
    if not chunk.size:
        return np.nan
    spans = _burst_spans(chunk > amplitude_threshold, bridge)
    return float(max((stop - start for start, stop in spans), default=0))


def _burst_spans(active: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """Above-threshold stretches, with gaps of at most ``bridge`` samples joined.

    The envelope is a moving average ``window_ms`` wide, so two excursions
    closer together than one window are not resolvable as separate events: the
    same muscle activity is smeared across the dip between them. Bridging gaps
    up to that width therefore measures one physiological burst as one burst,
    rather than splitting a doublet into two halves that each fall short.
    """
    edges = np.diff(np.concatenate(([0], np.asarray(active, dtype=int), [0])))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, stop in spans[1:]:
        if start - merged[-1][1] <= bridge:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return [(int(a), int(b)) for a, b in merged]


def _peak_burst_energy(values: np.ndarray, centers: np.ndarray, onset: float,
                       duration: float, amplitude_threshold: float,
                       background: float, spacing_ms: float,
                       bridge: int) -> float:
    """Energy of the strongest burst inside one motor trial.

    The width test alone weighs only how long the envelope stays above the
    threshold, and so scores a 45x background twitch of 140 ms exactly as it
    scores nothing at all. Burst energy -- the area between the envelope and
    its own rest background, over one burst, in background-units x ms --
    restores height to the decision and trades it against width the way the
    contraction itself does.

    It is measured in background units so that it is comparable between runs
    on the same terms as the amplitude threshold, and only over samples inside
    the motor trial, on the same span as ``_longest_active_run``. Samples
    bridged across a within-burst dip contribute their own excess over
    background and never less than zero, so joining a doublet cannot lower the
    energy of either half.
    """
    chunk = values[_motor_segment(centers, onset, duration)]
    chunk = chunk[np.isfinite(chunk)]
    if not chunk.size or not np.isfinite(background) or background <= 0:
        return np.nan
    scaled = chunk / background
    spans = _burst_spans(scaled > amplitude_threshold / background, bridge)
    if not spans:
        return 0.0
    return float(max(
        np.sum(np.clip(scaled[start:stop] - 1.0, 0.0, None)) * spacing_ms
        for start, stop in spans))


def trial_emg_metrics(values: np.ndarray, centers: np.ndarray,
                      trial_onsets: np.ndarray, trial_duration: float,
                      rest_events: EventSet, config: ProcessingConfig) \
        -> TrialMetricSet:
    """Detect motor-task EMG peaks against this recording's own background.

    A trial is high when either of two branches qualifies. The primary branch
    requires a peak of at least ``peak_multiplier`` times this recording's
    clean-rest median, with ``min_burst_ms`` width at the lower
    ``background_multiplier`` rest shoulder.

    The secondary branch adapts to an unusually quiet trial. It takes the
    median envelope from ``pre_reference_start_s`` through
    ``pre_reference_end_s`` (the quiet part of the cue), then searches only
    after that interval. Its peak must reach ``secondary_pre_multiplier``
    times that trial's own preparation level. It must also be
    ``min_burst_ms`` wide at
    ``secondary_width_multiplier`` times that same preparation reference.
    This branch is independent of recording rest: satisfying the preparation
    comparison is sufficient even when the peak is below the rest-based bar.

    The +0.5 to +1.8 s default deliberately excludes both the initial cue
    response and the final 200 ms before the movement cue, where premature
    reactions can occur. A short classification-only tail prevents the
    scheduled trial boundary from cutting one resolvable burst into two
    sub-threshold fragments. Cohort trade-offs are recorded in the README.

    ``min_burst_ms`` is not a physiological width, and is not meant to be one.
    The envelope is a moving window ``window_ms`` wide, so a single
    instantaneous spike already lands in every window containing it. The
    default cutoff is deliberately shorter than that window because this task
    contains genuinely brief repeated micro-movements. Width is evaluated on
    the envelope grid, so a 50 ms cutoff with the default 20 ms step requires
    three above-threshold samples (an observed width of 60 ms or more).

    The threshold is deliberately anchored on the *median* rather
    than on a high percentile of rest: the median tracks the background level,
    which legitimately varies between runs, whereas a high percentile tracks
    rest's upper tail and so is dragged up by any peaks rest happens to
    contain -- the very thing being detected. On 002TEST the rest background
    moves by 1.7x across six runs while the 99th percentile of the same rest
    moves by 10x.

    Nothing is shared between recordings. The multipliers are study constants,
    so a run with elevated background becomes less sensitive by design, and
    each run is scored purely against itself.

    The same detector is then run over matched rest windows. That measured
    false-positive rate is the floor any trial percentage has to be read
    against, and a run whose own rest fires above ``rest_fpr_warn`` is marked
    unreliable rather than quietly reported as clean: its threshold has been
    lifted by bursts inside rest, so its trial count is an underestimate.
    """
    clean_rest = []
    pseudo_starts = []
    for onset, duration in zip(rest_events.onsets, rest_events.durations):
        if not np.isfinite(duration) or duration <= 0:
            continue
        start = float(onset) + config.rest_trim_start_s
        stop = float(onset + duration) - config.rest_trim_end_s
        if stop <= start:
            continue
        chunk = values[(centers >= start) & (centers < stop)]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            clean_rest.append(chunk)
        last_start = stop - trial_duration
        if last_start >= start:
            pseudo_starts.extend(np.arange(
                start, last_start + config.rest_pseudotrial_step_s / 2.0,
                config.rest_pseudotrial_step_s))

    if not clean_rest:
        raise ValueError("no clean rest samples are available for trial metrics")
    if not pseudo_starts:
        raise ValueError(
            "clean rest intervals are shorter than the motor trial; cannot "
            "form matched rest pseudo-trials")

    background = float(np.median(np.concatenate(clean_rest)))
    if not np.isfinite(background) or background <= 0:
        raise ValueError(
            "the clean rest envelope has no positive median, so no background "
            "reference can be formed for this hand")
    burst_threshold = config.background_multiplier * background
    peak_threshold = config.peak_multiplier * background

    spacing_s = (float(np.median(np.diff(centers)))
                 if centers.size > 1 else trial_duration)
    # Both sides of the width test are sample counts. The epsilon absorbs the
    # float error in `spacing_s` only; it is far below one sample.
    spacing_ms = spacing_s * 1000.0
    min_run = (int(np.ceil(config.min_burst_ms / spacing_ms - 1e-6))
               if spacing_ms > 0 else 0)
    # Gaps shorter than one envelope window are not resolvable as separate
    # events, so they are not treated as separating two bursts.
    bridge = (int(np.floor(config.window_ms / spacing_ms + 1e-6))
              if spacing_ms > 0 else 0)

    def widest(onset: float) -> float:
        return _longest_active_run(values, centers, onset, trial_duration,
                                   burst_threshold, bridge)

    def energy(onset: float) -> float:
        return _peak_burst_energy(values, centers, onset, trial_duration,
                                  burst_threshold, background, spacing_ms,
                                  bridge)

    def height(onset: float) -> float:
        return _trial_peak(values, centers, onset, trial_duration)

    def primary_qualifies(peaks: np.ndarray, runs: np.ndarray) -> np.ndarray:
        tall = np.isfinite(peaks) & (np.nan_to_num(peaks) >= peak_threshold)
        wide = np.isfinite(runs) & (np.nan_to_num(runs) >= min_run)
        return tall & wide

    def preparation_background(onset: float) -> float:
        start = onset + config.pre_reference_start_s
        stop = onset + config.pre_reference_end_s
        chunk = values[(centers >= start) & (centers < stop)]
        chunk = chunk[np.isfinite(chunk)]
        return float(np.median(chunk)) if chunk.size else np.nan

    def secondary_metrics(onset: float) -> tuple[float, float, float]:
        """Preparation median, later peak, and later width in samples."""
        preparation = preparation_background(onset)
        search_duration = trial_duration - config.pre_reference_end_s
        if search_duration <= 0 or not np.isfinite(preparation):
            return preparation, np.nan, np.nan
        search_onset = onset + config.pre_reference_end_s
        peak = _trial_peak(values, centers, search_onset, search_duration)
        shoulder = config.secondary_width_multiplier * preparation
        run = _longest_active_run(
            values, centers, search_onset, search_duration, shoulder, bridge)
        return preparation, peak, run

    def secondary_qualifies(preparations: np.ndarray, peaks: np.ndarray,
                            runs: np.ndarray) -> np.ndarray:
        valid = (np.isfinite(preparations) & (preparations > 0) &
                 np.isfinite(peaks) & np.isfinite(runs))
        tall_vs_preparation = np.nan_to_num(peaks) >= (
            config.secondary_pre_multiplier * np.nan_to_num(preparations))
        wide = np.nan_to_num(runs) >= min_run
        return valid & tall_vs_preparation & wide

    rest_runs = np.asarray([widest(start) for start in pseudo_starts],
                           dtype=float)
    rest_peaks = np.asarray([height(start) for start in pseudo_starts],
                            dtype=float)
    rest_secondary = np.asarray(
        [secondary_metrics(float(start)) for start in pseudo_starts],
        dtype=float)
    usable = np.isfinite(rest_runs)
    rest_runs, rest_peaks = rest_runs[usable], rest_peaks[usable]
    rest_secondary = rest_secondary[usable]
    if not rest_runs.size:
        raise ValueError("matched rest windows contain no envelope samples")
    primary_rest_high = primary_qualifies(rest_peaks, rest_runs)
    secondary_rest_high = secondary_qualifies(
        rest_secondary[:, 0], rest_secondary[:, 1], rest_secondary[:, 2])
    primary_rest_fpr = float(np.mean(primary_rest_high))
    secondary_rest_fpr = float(np.mean(secondary_rest_high))
    rest_fpr = float(np.mean(primary_rest_high | secondary_rest_high))

    trial_runs = np.asarray([widest(float(onset)) for onset in trial_onsets],
                            dtype=float)
    trial_peaks = np.asarray([height(float(onset)) for onset in trial_onsets],
                             dtype=float)
    trial_energies = np.asarray([energy(float(onset)) for onset in trial_onsets],
                                dtype=float)
    trial_fractions = np.asarray([
        _active_fraction(values, centers, float(onset), trial_duration,
                         burst_threshold)
        for onset in trial_onsets
    ], dtype=float)
    trial_secondary = np.asarray(
        [secondary_metrics(float(onset)) for onset in trial_onsets],
        dtype=float).reshape((-1, 3))
    preparations = trial_secondary[:, 0]
    secondary_peaks = trial_secondary[:, 1]
    secondary_runs = trial_secondary[:, 2]
    primary_high = primary_qualifies(trial_peaks, trial_runs)
    secondary_high = secondary_qualifies(
        preparations, secondary_peaks, secondary_runs)
    high = primary_high | secondary_high
    secondary_peak_ratio = np.divide(
        secondary_peaks, background,
        out=np.full(secondary_peaks.shape, np.nan),
        where=np.isfinite(secondary_peaks))
    peak_pre_ratio = np.divide(
        secondary_peaks, preparations,
        out=np.full(secondary_peaks.shape, np.nan),
        where=np.isfinite(secondary_peaks) & np.isfinite(preparations) &
              (preparations > 0))
    secondary_burst_threshold = (
        config.secondary_width_multiplier * np.maximum(
            background, preparations))
    return TrialMetricSet(
        high=high,
        primary_high=primary_high,
        secondary_high=secondary_high,
        peak_ratio=trial_peaks / background,
        longest_burst_ms=trial_runs * spacing_ms,
        burst_energy=trial_energies,
        active_fraction=trial_fractions,
        background=background,
        peak_threshold=float(peak_threshold),
        burst_threshold=float(burst_threshold),
        pre_movement_background=preparations,
        secondary_peak_ratio=secondary_peak_ratio,
        peak_pre_ratio=peak_pre_ratio,
        secondary_longest_burst_ms=secondary_runs * spacing_ms,
        secondary_burst_threshold=secondary_burst_threshold,
        min_burst_ms=float(config.min_burst_ms),
        rest_fpr=rest_fpr,
        primary_rest_fpr=primary_rest_fpr,
        secondary_rest_fpr=secondary_rest_fpr,
        n_rest_windows=int(rest_runs.size),
        rest_unreliable=bool(rest_fpr > config.rest_fpr_warn),
    )


def _events_for(recording: LoadedRecording, config: ProcessingConfig,
                shift: float, shift_note: str) \
        -> tuple[dict[str, EventSet], str]:
    shift = float(shift)
    events = {
        condition: EventSet(
            np.asarray(event.onsets, dtype=np.float64) + shift,
            np.asarray(event.durations, dtype=np.float64).copy(),
        )
        for condition, event in recording.marker_events.items()
    }
    if recording.event_source == "brainvision":
        note = (
            "BrainVision trials reconstructed from first S3 video start and "
            f"cond-seq video schedule; {shift_note}")
    else:
        note = f"embedded XDF marker timestamps shifted {shift:+.3f} s; {shift_note}"
    if recording.video_onset_s is not None:
        # Informational only, but report it on the same corrected time axis.
        corrected_video = recording.video_onset_s + shift
        note += f" (corrected video_onset at {corrected_video:+.3f} s)"
    required = [config.rest_condition]
    required.extend(condition for _, condition, _ in hand_specs(config))
    for condition in required:
        if condition not in events or not events[condition].onsets.size:
            raise KeyError(f"condition {condition!r} has no events")
    return events, note


def _motor_segment(times: np.ndarray, onset: float, duration: float) \
        -> np.ndarray:
    """Samples inside one motor trial: the half-open span ``[onset, onset+dur)``.

    The single definition of "inside the trial". Both the high/low decision and
    the y-scaling of the per-trial figures go through it, so what the figure is
    scaled to is exactly what the classifier looked at.
    """
    return (times >= onset) & (times < onset + duration)


def _shade_motor_trial(ax, duration: float, high_emg: bool = False,
                       onset: float = 0.0, zorder: float | None = None,
                       tail_s: float = 0.0) -> None:
    """Shade the scheduled trial and its optional classification-only tail."""
    cue_onset = onset + min(MOTOR_CUE_DELAY_S, duration)
    delay_color = "#edb6c2" if high_emg else "0.82"
    cue_color = "#f7cfd8" if high_emg else "0.9"
    style = {} if zorder is None else {"zorder": zorder}
    ax.axvspan(onset, cue_onset, color=delay_color, **style)
    if duration > MOTOR_CUE_DELAY_S:
        ax.axvspan(cue_onset, onset + duration, color=cue_color, **style)
    if tail_s > 0:
        tail_color = "#f5dce2" if high_emg else "0.96"
        ax.axvspan(onset + duration, onset + duration + tail_s,
                   facecolor=tail_color, edgecolor="0.65", hatch="////",
                   linewidth=0.0, **style)


def envelope_label(config: ProcessingConfig) -> str:
    """Y-axis name of whatever envelope this configuration produces."""
    return "TKEO energy [mV²]" if config.envelope == "tkeo" else "RMS [mV]"


def draw_hand_mean(ax, recording_name: str, hand: str, grid: np.ndarray,
                   epochs: np.ndarray, duration: float, rest_baseline: float,
                   config: ProcessingConfig, compact: bool = False) -> None:
    """Draw one hand's across-trial envelope onto *ax*.

    The single implementation behind both the saved figure and the panel the
    application shows, so what is on screen is what a saved file would contain.
    ``compact`` drops the axis furniture for a small panel; the data, the
    shading and the y-scaling are identical either way.
    """
    ax.clear()
    mean = np.nanmean(epochs, axis=0)
    median = np.nanmedian(epochs, axis=0)
    q1, q3 = np.nanpercentile(epochs, [25, 75], axis=0)

    _shade_motor_trial(ax, duration, tail_s=config.trial_tail_s)
    if np.isfinite(rest_baseline):
        ax.axhline(rest_baseline, color="tab:green", ls="--", lw=1.2)
    # Spread across trials, not uncertainty of an estimate: on this strongly
    # skewed energy the quartiles show how many repeats actually carry the
    # response, which a mean-centred interval hides.
    ax.fill_between(grid, q1, q3, color="tab:red", alpha=0.25)
    ax.plot(grid, median, color="tab:red", lw=2)
    ax.plot(grid, mean, color="black", lw=1, ls=":")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(grid[0], grid[-1])
    # Scale to the quartile band. A mean pulled up by one artefactual trial is
    # allowed to run off the top: keeping the band readable matters more than
    # keeping that excursion in frame.
    band = np.concatenate((q1[np.isfinite(q1)], q3[np.isfinite(q3)]))
    if band.size:
        bottom, top = min(float(band.min()), 0.0), float(band.max())
        span = top - bottom
        if span > 0:
            ax.set_ylim(bottom - 0.02 * span, top + 0.05 * span)
    ax.grid(alpha=0.15)
    if compact:
        ax.set_title(hand, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("s from onset", fontsize=7, labelpad=1)
        ax.set_ylabel(envelope_label(config), fontsize=7, labelpad=1)
        return
    ax.set_xlabel("time from event onset [s]")
    ax.set_ylabel(envelope_label(config))
    ax.set_title(f"{recording_name} — {hand}")


def _plot_hand(recording_name: str, hand: str, condition: str,
               pair: tuple[str, str], grid: np.ndarray, epochs: np.ndarray,
               duration: float, rest_baseline: float, config: ProcessingConfig,
               output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    draw_hand_mean(ax, recording_name, hand, grid, epochs, duration,
                   rest_baseline, config)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def decision_source(metrics: TrialMetricSet, index: int) -> str:
    """Which detection branch, if any, called this trial high."""
    primary = bool(metrics.primary_high[index])
    secondary = bool(metrics.secondary_high[index])
    if primary and secondary:
        return "both"
    if primary:
        return "primary"
    if secondary:
        return "secondary"
    return "none"


def _write_trial_metrics(path: Path, onsets: np.ndarray,
                         scheduled_duration: float,
                         metrics: TrialMetricSet,
                         config: ProcessingConfig) -> None:
    """Write auditable per-trial inputs and the branch behind each verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial", "onset_s", "high_emg", "decision_source",
        "primary_high", "secondary_high", "full_peak_x_rest",
        "primary_width_ms", "pre_movement_background",
        "post_pre_peak_x_rest", "post_pre_peak_x_pre",
        "secondary_width_ms", "rest_background", "primary_peak_threshold",
        "primary_width_threshold", "secondary_width_threshold",
        "scheduled_duration_s", "classification_tail_ms",
        "pre_reference_start_ms", "pre_reference_end_ms",
        "primary_rest_multiplier", "primary_width_multiplier",
        "secondary_pre_multiplier", "secondary_width_multiplier",
        "minimum_width_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, onset in enumerate(onsets):
            writer.writerow({
                "trial": index + 1,
                "onset_s": f"{float(onset):.6f}",
                "high_emg": bool(metrics.high[index]),
                "decision_source": decision_source(metrics, index),
                "primary_high": bool(metrics.primary_high[index]),
                "secondary_high": bool(metrics.secondary_high[index]),
                "full_peak_x_rest": f"{metrics.peak_ratio[index]:.6g}",
                "primary_width_ms":
                    f"{metrics.longest_burst_ms[index]:.6g}",
                "pre_movement_background":
                    f"{metrics.pre_movement_background[index]:.12g}",
                "post_pre_peak_x_rest":
                    f"{metrics.secondary_peak_ratio[index]:.6g}",
                "post_pre_peak_x_pre":
                    f"{metrics.peak_pre_ratio[index]:.6g}",
                "secondary_width_ms":
                    f"{metrics.secondary_longest_burst_ms[index]:.6g}",
                "rest_background": f"{metrics.background:.12g}",
                "primary_peak_threshold": f"{metrics.peak_threshold:.12g}",
                "primary_width_threshold": f"{metrics.burst_threshold:.12g}",
                "secondary_width_threshold":
                    f"{metrics.secondary_burst_threshold[index]:.12g}",
                "scheduled_duration_s": f"{scheduled_duration:.6g}",
                "classification_tail_ms": f"{1000 * config.trial_tail_s:.6g}",
                "pre_reference_start_ms":
                    f"{1000 * config.pre_reference_start_s:.6g}",
                "pre_reference_end_ms":
                    f"{1000 * config.pre_reference_end_s:.6g}",
                "primary_rest_multiplier": f"{config.peak_multiplier:.6g}",
                "primary_width_multiplier":
                    f"{config.background_multiplier:.6g}",
                "secondary_pre_multiplier":
                    f"{config.secondary_pre_multiplier:.6g}",
                "secondary_width_multiplier":
                    f"{config.secondary_width_multiplier:.6g}",
                "minimum_width_ms": f"{metrics.min_burst_ms:.6g}",
            })


def draw_trial(ax, index: int, grid: np.ndarray, epochs: np.ndarray,
               duration: float, rest_baseline: float,
               metrics: TrialMetricSet | None, config: ProcessingConfig,
               title: str = "", compact: bool = False) -> bool:
    """Draw trial *index* onto *ax* and return its high/low verdict.

    Each trial is scaled to the scheduled motor trial plus its short hatched
    classification tail. This is the same span the high/low decision used, so
    the y-axis shows the evidence the classifier actually considered. Activity
    before onset or after that tail is drawn but deliberately allowed to run
    off the top: letting it set the scale flattens the trial itself into a line.

    Amplitudes are therefore *not* comparable between trials by eye. The peak
    bar is drawn, so a trial that fell short can be read at a glance; it is off
    the top of the axes on trials whose peak never approached it. A light pink
    background marks trials classified as high EMG.

    ``compact`` strips the labels for a thumbnail in a grid of trials. It
    changes nothing that is drawn or scaled, so a thumbnail and its enlarged
    view cannot disagree about the trial.
    """
    ax.clear()
    trace = epochs[index]
    is_high = bool(metrics.high[index]) if metrics is not None else False
    background = "#ffe8ed" if is_high else "white"
    ax.set_facecolor(background)
    _shade_motor_trial(ax, duration, high_emg=is_high,
                       tail_s=config.trial_tail_s)
    if np.isfinite(rest_baseline):
        ax.axhline(rest_baseline, color="tab:green", ls="--", lw=1.0)
    if metrics is not None:
        if np.isfinite(metrics.peak_threshold):
            ax.axhline(metrics.peak_threshold, color="tab:purple", ls="-.",
                       lw=1.0, alpha=0.8)
        secondary_threshold = metrics.secondary_burst_threshold[index]
        if np.isfinite(secondary_threshold):
            ax.axhline(secondary_threshold, color="tab:blue", ls=":",
                       lw=1.0, alpha=0.8)
    ax.axvspan(config.pre_reference_start_s, config.pre_reference_end_s,
               color="tab:blue", alpha=0.06)
    ax.axvline(config.pre_reference_end_s, color="tab:blue", ls=":",
               lw=0.8, alpha=0.7)
    ax.plot(grid, np.nanmedian(epochs, axis=0), color="0.55", lw=1.0, ls=":")
    ax.plot(grid, trace, color="tab:red", lw=1.4 if not compact else 1.0)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(grid[0], grid[-1])
    segment = _motor_segment(grid, 0.0, duration + config.trial_tail_s)
    inside = trace[segment]
    inside = inside[np.isfinite(inside)]
    if inside.size:
        top, bottom = float(inside.max()), min(float(inside.min()), 0.0)
        span = top - bottom
        if span <= 0:
            span = abs(top) if top else 1.0
        ax.set_ylim(bottom - 0.05 * span, top + 0.12 * span)
    ax.grid(alpha=0.15)
    if compact:
        ax.set_title(f"{index + 1}", fontsize=8, pad=2,
                     color="crimson" if is_high else "0.3",
                     fontweight="bold" if is_high else "normal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=2)
        return is_high
    ax.set_xlabel("time from event onset [s]")
    ax.set_ylabel(envelope_label(config))
    ax.set_title(title or f"trial {index + 1}",
                 color="crimson" if is_high else "black",
                 fontweight="bold" if is_high else "normal")
    return is_high


def _plot_trials(recording_name: str, hand: str, condition: str,
                 grid: np.ndarray, epochs: np.ndarray, onsets: np.ndarray,
                 duration: float, rest_baseline: float,
                 metrics: TrialMetricSet, config: ProcessingConfig,
                 output_dir: Path) -> None:
    """One figure per trial, written under ``trials/<recording>/<hand>/``.

    Detailed classifier values remain available in ``trial_metrics.csv``; see
    :func:`draw_trial` for what each figure shows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    n_trials = epochs.shape[0]
    width = len(str(n_trials))
    for index in range(n_trials):
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        is_high = draw_trial(
            ax, index, grid, epochs, duration, rest_baseline, metrics, config,
            title=f"{recording_name} — {hand} — trial {index + 1}")
        # The saved figure keeps the pink page the application shows on the
        # panel itself, so a high trial is recognisable as a thumbnail on disk.
        fig.set_facecolor("#ffe8ed" if is_high else "white")
        fig.tight_layout()
        fig.savefig(output_dir / f"trial_{index + 1:0{width}d}.png", dpi=110)
        plt.close(fig)


def analyze_recording(path: Path, name: str, config: ProcessingConfig,
                      require_metrics: bool | None = None) -> RecordingResult:
    """Load, filter, epoch and classify one recording, writing nothing.

    ``require_metrics`` decides what happens when a hand has no usable rest to
    calibrate against: raising makes an unscorable recording stop a batch that
    was asked for per-trial output, while recording the reason lets the
    application show every other hand it did manage to score. It defaults to
    the strictness the requested output implies.
    """
    if require_metrics is None:
        require_metrics = config.trial_figures
    specs = hand_specs(config)
    requested = list(dict.fromkeys(
        channel for _, _, pair in specs for channel in pair))
    if config.file_type == "brainvision":
        loaded = load_brainvision(path, requested, config)
    else:
        loaded = load_xdf(path, requested, config)
    loaded = _resample_loaded(loaded, config.target_fs)
    # Measured before epoching and per recording: the amplifier's timestamp lag
    # is a property of the file, not of the study.
    if config.file_type == "brainvision":
        shift, shift_note, estimate = (
            0.0, "sample-locked S3 anchor; no XDF clock correction", None)
    else:
        shift, shift_note, estimate = resolve_marker_shift(
            path, config.auto_marker_shift, config.marker_shift_s,
            config.data_stream, config.marker_stream)
    events, timing_note = _events_for(loaded, config, shift, shift_note)

    results = []
    for hand, condition, pair in specs:
        bipolar = loaded.signals[pair[0]] - loaded.signals[pair[1]]
        filtered = filter_emg(bipolar, loaded.fs, config)
        envelope, centers = envelope_trace(filtered, loaded.fs, config)
        # Do not count uncorrected tails/filter edges as rest or contractions.
        # Expand exclusions by half an envelope window to exclude overlap.
        for start, end in loaded.provenance.get("mri_correction", {}).get(
                "bad_intervals_s", []):
            half = config.window_ms / 2000.0
            envelope[(centers >= start - half) & (centers <= end + half)] = np.nan
        event = events[condition]
        finite_durations = event.durations[np.isfinite(event.durations) &
                                          (event.durations > 0)]
        if not finite_durations.size:
            raise ValueError(f"no usable durations for {condition!r}")
        duration = float(np.median(finite_durations))
        classification_duration = duration + config.trial_tail_s
        step_s = config.step_ms / 1000.0
        grid = np.arange(
            -config.pre_s,
            classification_duration + config.post_s + step_s / 2,
            step_s)
        epochs = epoch_trace(envelope, centers, event.onsets, grid)
        rest = _rest_baseline(envelope, centers, events[config.rest_condition],
                              config.rest_trim_end_s)
        movement_mask = (grid >= 0) & (grid <= duration)
        movement = float(np.nanmean(epochs[:, movement_mask]))
        ratio = movement / rest if np.isfinite(rest) and rest != 0 else np.nan
        # The split is summarised for every run, so that the high/low counts
        # are available without also writing one figure per trial.
        metrics_error = ""
        try:
            metrics = trial_emg_metrics(
                envelope, centers, event.onsets, classification_duration,
                events[config.rest_condition], config)
        except ValueError as exc:
            if require_metrics:
                raise
            metrics, metrics_error = None, str(exc)
        note = ""
        if metrics is not None and metrics.rest_unreliable:
            note = (
                f"UNRELIABLE: the detector also fired in "
                f"{metrics.rest_fpr:.0%} of {metrics.n_rest_windows} rest "
                "windows; do not interpret the raw high-trial count as EMG "
                "contamination")
        elif metrics_error:
            note = f"not scored: {metrics_error}"
        results.append(HandResult(
            hand, condition, pair, int(event.onsets.size), duration, rest,
            movement, float(ratio), "",
            n_high_trials=None if metrics is None else int(np.sum(metrics.high)),
            background=np.nan if metrics is None else metrics.background,
            peak_threshold=(np.nan if metrics is None
                            else metrics.peak_threshold),
            burst_threshold=(np.nan if metrics is None
                             else metrics.burst_threshold),
            min_burst_ms=np.nan if metrics is None else metrics.min_burst_ms,
            trial_tail_ms=1000.0 * config.trial_tail_s,
            rest_fpr=np.nan if metrics is None else metrics.rest_fpr,
            primary_rest_fpr=(np.nan if metrics is None
                              else metrics.primary_rest_fpr),
            secondary_rest_fpr=(np.nan if metrics is None
                                else metrics.secondary_rest_fpr),
            threshold_note=note,
            analysis=HandAnalysis(
                grid=grid, epochs=epochs, onsets=event.onsets,
                duration=duration,
                classification_duration=classification_duration,
                rest_baseline=rest, metrics=metrics,
                metrics_error=metrics_error)))
    provenance = dict(getattr(loaded, "provenance", {}) or {})
    provenance.update({
        "source_path": str(path.resolve()),
        "event_source": loaded.event_source,
        "marker_time_shift_s": float(shift),
    })
    return RecordingResult(
        name, str(path), loaded.fs, timing_note, results,
        marker_shift_s=float(shift),
        detected_shift_s=(np.nan if estimate is None else estimate.shift_s),
        provenance=provenance)


def write_recording_outputs(result: RecordingResult, config: ProcessingConfig,
                            output_dir: Path) -> None:
    """Write one recording's figures and per-trial metrics into *output_dir*.

    Fills in the paths on the result, so a batch that has been saved reports
    where its files are and one that has not reports empty paths rather than
    names that were never written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = result.recording
    for hand in result.hands:
        analysis = hand.analysis
        if analysis is None:
            continue
        figure = output_dir / (
            f"{_safe_name(name)}_{hand.hand}_hand_{config.envelope}.png")
        _plot_hand(name, hand.hand, hand.condition, hand.channels,
                   analysis.grid, analysis.epochs, analysis.duration,
                   analysis.rest_baseline, config, figure)
        hand.figure_path = str(figure)
        trial_dir = output_dir / "trials" / _safe_name(name) / hand.hand
        if config.trial_figures and analysis.metrics is not None:
            _plot_trials(name, hand.hand, hand.condition, analysis.grid,
                         analysis.epochs, analysis.onsets, analysis.duration,
                         analysis.rest_baseline, analysis.metrics, config,
                         trial_dir)
            _write_trial_metrics(
                trial_dir / "trial_metrics.csv", analysis.onsets,
                analysis.duration, analysis.metrics, config)
            hand.trial_dir = str(trial_dir)
            hand.trial_metrics_csv = str(trial_dir / "trial_metrics.csv")
        elif config.trial_figures:
            # An unscored hand has no verdicts to draw; say so instead of
            # leaving a directory that looks like an empty result.
            hand.trial_dir = ""


def process_recording(path: Path, name: str, config: ProcessingConfig,
                      output_dir: Path) -> RecordingResult:
    """Analyse one recording and write its figures, in one call."""
    result = analyze_recording(path, name, config)
    write_recording_outputs(result, config, output_dir)
    return result


def _write_summary(path: Path, recordings: list[RecordingResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "recording", "hand", "condition", "channel_positive",
            "channel_negative", "sample_rate_hz", "trials",
            "recording_rate_hz", "resampled", "resampling_method",
            "movement_duration_s", "classification_tail_ms",
            "rest_baseline", "movement_mean",
            "movement_rest_ratio", "high_emg_trials", "high_emg_percent",
            "rest_false_positive_percent", "primary_rest_false_positive_percent",
            "secondary_rest_false_positive_percent", "rest_background",
            "peak_threshold", "burst_threshold", "min_burst_ms",
            "rest_warning",
            "marker_shift_s", "detected_marker_shift_s",
            "timing", "source_path", "event_source", "session_label",
            "session_condition", "video", "video_start_s",
            "video_start_sample", "session_order_file",
            "session_order_sha256", "schedule_file", "schedule_sha256",
            "figure", "trial_figures", "trial_metrics_csv",
        ])
        for recording in recordings:
            provenance = recording.provenance or {}
            for hand in recording.hands:
                writer.writerow([
                    recording.recording, hand.hand, hand.condition,
                    hand.channels[0], hand.channels[1], recording.fs,
                    hand.n_trials, provenance.get("recording_fs", recording.fs),
                    provenance.get("resampled", False),
                    provenance.get("resampling_method", "none"),
                    hand.movement_duration_s, hand.trial_tail_ms,
                    hand.rest_baseline,
                    hand.movement_mean, hand.movement_rest_ratio,
                    "" if hand.n_high_trials is None else hand.n_high_trials,
                    "" if hand.n_high_trials is None else f"{hand.high_percent:.1f}",
                    "" if hand.n_high_trials is None else f"{100 * hand.rest_fpr:.1f}",
                    ("" if hand.n_high_trials is None
                     else f"{100 * hand.primary_rest_fpr:.1f}"),
                    ("" if hand.n_high_trials is None
                     else f"{100 * hand.secondary_rest_fpr:.1f}"),
                    hand.background, hand.peak_threshold, hand.burst_threshold,
                    hand.min_burst_ms,
                    hand.threshold_note,
                    f"{recording.marker_shift_s:.3f}",
                    ("" if not np.isfinite(recording.detected_shift_s)
                     else f"{recording.detected_shift_s:.3f}"),
                    recording.timing_note,
                    provenance.get("source_path", recording.source_path),
                    provenance.get("event_source", ""),
                    provenance.get("session_label", ""),
                    provenance.get("condition", ""),
                    provenance.get("video", ""),
                    provenance.get("video_start_s", ""),
                    provenance.get("video_start_sample", ""),
                    provenance.get("session_order_file", ""),
                    provenance.get("session_order_sha256", ""),
                    provenance.get("schedule_file", ""),
                    provenance.get("schedule_sha256", ""),
                    hand.figure_path, hand.trial_dir,
                    hand.trial_metrics_csv,
                ])


def _screening_report(recordings: list[RecordingResult]) -> list[str]:
    """Per-recording contamination table for the day-one include/exclude call.

    Every trial percentage is printed next to the false-positive rate the same
    detector produced on that run's own rest, because the second is the floor
    the first has to be read against.
    """
    lines = [
        "",
        "Contaminated trials per recording (read against each run's rest floor):",
        f"  {'recording':16s}{'hand':6s}{'high EMG':>12s}{'rest floor':>12s}",
    ]
    for recording in recordings:
        for hand in recording.hands:
            if hand.n_high_trials is None:
                lines.append(f"  {recording.recording:16s}{hand.hand:6s}"
                             f"{'not scored':>12s}")
                continue
            metrics = getattr(getattr(hand, "analysis", None), "metrics", None)
            if bool(getattr(metrics, "rest_unreliable", False)):
                share = f"UNRELIABLE raw {hand.n_high_trials}/{hand.n_trials}"
            else:
                share = (f"{hand.n_high_trials}/{hand.n_trials} = "
                         f"{hand.high_percent:.0f}%")
            lines.append(f"  {recording.recording:16s}{hand.hand:6s}{share:>12s}"
                         f"{100 * hand.rest_fpr:11.0f}%")
            if hand.threshold_note:
                lines.append(f"      ! {hand.threshold_note}")
    return lines


def resolve_output_root(output_root: str | Path) -> Path:
    """The absolute root a configured output root names.

    An absolute root is obeyed as given. A relative one -- including the
    default, and including a box someone has emptied -- is taken from the
    application directory, so an installation writes inside itself wherever it
    has been copied to. See :data:`DEFAULT_OUTPUT_ROOT`.
    """
    root = Path(str(output_root).strip() or DEFAULT_OUTPUT_ROOT)
    return (root if root.is_absolute() else PROJECT_DIR / root).resolve()


def planned_output_dir(config: ProcessingConfig) -> Path:
    """Where this configuration's files would go, without creating anything."""
    participant = _safe_name(
        config.participant or Path(config.data_dir).resolve().name)
    output = resolve_output_root(config.output_root) / participant
    if config.single_file_output:
        if len(config.recordings) != 1:
            raise ValueError("Single-file mode requires exactly one filename")
        mode = "mne_aas_obs" if config.remove_mri_artifacts else "direct"
        output = output / _safe_name(Path(config.recordings[0]).stem) / mode
    return output


def analyze_batch(config: ProcessingConfig,
                  progress: Callable[[str], None] | None = None,
                  require_metrics: bool | None = None) -> BatchResult:
    """Analyse every recording in *config* without writing anything.

    The returned :class:`BatchResult` carries each hand's epochs and verdicts,
    so its figures can be drawn later -- on screen, or into files through
    :func:`save_batch_outputs` -- without reading the recordings again. Its
    ``summary_csv`` stays empty until something has actually been written.
    """
    config.validate()
    emit = progress or (lambda _message: None)
    output_dir = planned_output_dir(config)

    results = []
    for index, name in enumerate(config.recordings, start=1):
        path = recording_path(config.data_dir, name, config.file_type)
        display_name = path.stem
        emit(f"[{index}/{len(config.recordings)}] Processing {path.name} …")
        result = analyze_recording(path, display_name, config, require_metrics)
        results.append(result)
        emit(f"[{index}/{len(config.recordings)}] {result.timing_note}")
        ratios = ", ".join(
            f"{hand.hand} {hand.movement_rest_ratio:.2f}x"
            + ("" if hand.n_high_trials is None
               else f" ({hand.n_high_trials}/{hand.n_trials} high)")
            for hand in result.hands)
        emit(f"[{index}/{len(config.recordings)}] Analysed ({ratios})")

    for line in _screening_report(results):
        emit(line)
    return BatchResult(
        _safe_name(config.participant or Path(config.data_dir).resolve().name),
        str(output_dir), results, "")


def save_batch_outputs(batch: BatchResult, config: ProcessingConfig,
                       progress: Callable[[str], None] | None = None) \
        -> BatchResult:
    """Write an analysed batch's figures, per-trial metrics and summary."""
    emit = progress or (lambda _message: None)
    output_dir = Path(batch.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emit(f"Output: {output_dir}")
    for index, result in enumerate(batch.recordings, start=1):
        write_recording_outputs(result, config, output_dir)
        emit(f"[{index}/{len(batch.recordings)}] Saved {result.recording}")
    summary = output_dir / "emg_summary.csv"
    _write_summary(summary, batch.recordings)
    batch.summary_csv = str(summary)
    provenance = {"config": asdict(config), "recordings": [
        {"recording": result.recording, **result.provenance}
        for result in batch.recordings]}
    (output_dir / "processing.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    emit(f"Summary: {summary}")
    return batch


def process_batch(config: ProcessingConfig,
                  progress: Callable[[str], None] | None = None) -> BatchResult:
    """Analyse every recording and write the whole output tree."""
    batch = analyze_batch(config, progress)
    return save_batch_outputs(batch, config, progress)
