"""Bounded AAS-only streaming EMG, matching the no-cardiac replay experiment.

Only complete, received cycles are visible to GradientRemover. Emitted output
is immutable. No ECG, QRS detection or cardiac OBS is used here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import mne
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from emgcasting.mri import MNE_VERSION

FS = 250
HISTORY = 8


@dataclass(frozen=True)
class OnlineConfig:
    left: tuple[str, str] | None = None
    right: tuple[str, str] | None = ("F1", "F2")
    tr_s: float = 2.5
    buffer_cycles: int = 4
    start_marker: str = ""

    def __post_init__(self):
        if not np.isfinite(self.tr_s) or not .5 <= self.tr_s <= 10:
            raise ValueError("Cycle length must be between 0.5 and 10 seconds")
        if self.buffer_cycles not in (2, 3, 4, 8):
            raise ValueError("Conditioning buffer must contain 2, 3, 4 or 8 cycles")
        if not self.pairs:
            raise ValueError("Enter at least one hand's channel pair")
        for pair in self.pairs.values():
            if len(pair) != 2 or any(not isinstance(c, str) or not c.strip() for c in pair) or pair[0] == pair[1]:
                raise ValueError("Each hand needs two different channel names: positive, negative")
        if not np.isclose(self.tr_s * FS, round(self.tr_s * FS), atol=1e-8, rtol=0):
            raise ValueError("Cycle length must contain a whole number of 250-Hz samples")

    @property
    def pairs(self):
        return {hand: pair for hand, pair in (("left", self.left), ("right", self.right)) if pair}

    @property
    def channels(self):
        return tuple(dict.fromkeys(c for pair in self.pairs.values() for c in pair))

    @property
    def startup_s(self):
        return (HISTORY + self.buffer_cycles) * self.tr_s


@dataclass(frozen=True)
class OnlineFrame:
    start_s: float
    acquisition_end_s: float
    conditioned_v: np.ndarray
    filtered_mv: dict[str, np.ndarray]
    envelope_times_s: np.ndarray
    envelope_mv2: dict[str, np.ndarray]
    compute_s: float

    @property
    def times_s(self):
        return self.start_s + np.arange(self.conditioned_v.shape[1]) / FS


class StreamingEnvelope:
    """100-ms TKEO means every 20 ms, retaining both neighbors at joins.

    The first undefined TKEO sample repeats the next sample, as in core.tkeo.
    The last sample remains pending until its right neighbor arrives.
    """
    def __init__(self, channels):
        self.channels = channels
        self.data = np.empty((channels, 0))
        self.base = 0
        self.next_start = 0
        self.origin_s = None

    def feed(self, data, start_s):
        if self.origin_s is None:
            self.origin_s = start_s
        expected = self.origin_s + (self.base + self.data.shape[1]) / FS
        if not np.isclose(start_s, expected, atol=1e-8, rtol=0):
            raise ValueError("Envelope input is not contiguous")
        self.data = np.concatenate((self.data, data), axis=1)
        stop = self.base + self.data.shape[1]
        starts = np.arange(self.next_start, stop - 25, 5, dtype=int)
        if not len(starts):
            return np.empty(0), np.empty((self.channels, 0))
        centers = starts[:, None] + np.arange(25)
        # Only the very first energy sample lacks a left neighbor.
        centers = np.maximum(centers, 1) - self.base
        energy = (self.data[:, centers] ** 2
                  - self.data[:, centers - 1] * self.data[:, centers + 1])
        means = energy.mean(axis=-1)
        times = self.origin_s + (starts + 12.5) / FS
        self.next_start = int(starts[-1] + 5)
        keep = max(0, self.next_start - 1)
        self.data = self.data[:, keep - self.base:].copy()
        self.base = keep
        return times, means


class OnlineProcessor:
    def __init__(self, config, sfreq):
        self.config = config
        self.sfreq = float(sfreq)
        if not np.isfinite(self.sfreq) or not 250 <= self.sfreq <= 20000:
            raise ValueError("Online EMG supports 250–20000 Hz; use native 5000 Hz for BrainAmp MR")
        self.cycle = round(config.tr_s * self.sfreq)
        if not np.isclose(config.tr_s * self.sfreq, self.cycle, atol=1e-7, rtol=0):
            raise ValueError("Cycle length must contain a whole number of native samples")
        if mne.__version__ != MNE_VERSION or not hasattr(mne.preprocessing, "GradientRemover"):
            raise RuntimeError(f"Install this app's requirements: online correction requires MNE {MNE_VERSION}")
        self.pending = np.empty((len(config.channels), self.cycle))
        self.pending_size = 0
        self.native = deque(maxlen=HISTORY + 1)
        self.corrected = deque(maxlen=config.buffer_cycles)
        self.samples_received = 0
        self.cycles_received = 0
        self.frames_emitted = 0
        self.hands = tuple(config.pairs)
        self.pair_indices = [(config.channels.index(a), config.channels.index(b))
                             for a, b in config.pairs.values()]
        self.notch = tf2sos(*iirnotch(50, 50, fs=FS))
        self.band = butter(4, (20, 95), btype="bandpass", fs=FS, output="sos")
        self.notch_state = np.zeros((len(self.notch), len(self.hands), 2))
        self.band_state = np.zeros((len(self.band), len(self.hands), 2))
        self.envelope = StreamingEnvelope(len(self.hands))

    def feed(self, data):
        """Yield at most one frame per completed cycle, for any packet sizes."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] != len(self.config.channels) or not np.all(np.isfinite(data)):
            raise ValueError("Expected finite selected-channel data in volts")
        offset = 0
        while offset < data.shape[1]:
            count = min(self.cycle - self.pending_size, data.shape[1] - offset)
            self.pending[:, self.pending_size:self.pending_size + count] = data[:, offset:offset + count]
            offset += count
            self.pending_size += count
            self.samples_received += count
            if self.pending_size == self.cycle:
                self.native.append(self.pending.copy())
                self.pending_size = 0
                self.cycles_received += 1
                if len(self.native) == HISTORY + 1:
                    frame = self._complete_cycle()
                    if frame is not None:
                        self.frames_emitted += 1
                        yield frame

    def _complete_cycle(self):
        tick = time.perf_counter()
        buffer = np.concatenate(tuple(self.native), axis=1)
        remover = mne.preprocessing.GradientRemover(
            buffer, np.arange(HISTORY + 1) * self.cycle, window=(HISTORY, 0))
        self.corrected.append(remover.get_tr_corrected(HISTORY).copy())
        if len(self.corrected) < self.config.buffer_cycles:
            return None
        trailing = np.concatenate(tuple(self.corrected), axis=1)
        small = trailing if self.sfreq == FS else mne.filter.resample(
            trailing, down=self.sfreq / FS, method="polyphase", verbose="error")
        raw = mne.io.RawArray(small, mne.create_info(
            list(self.config.channels), FS, ["emg"] * len(self.config.channels)), verbose="error")
        raw.filter(.5, 100, picks="all", phase="zero", fir_design="firwin", verbose="error")
        raw.notch_filter([50], picks="all", notch_widths=1, phase="zero", verbose="error")
        chunk = round(self.config.tr_s * FS)
        # Hold one cycle after the committed chunk for the zero-phase filters.
        conditioned = raw.get_data()[:, -2 * chunk:-chunk].copy()
        end_s = self.cycles_received * self.config.tr_s
        start_s = end_s - 2 * self.config.tr_s
        bipolar = np.stack([conditioned[a] - conditioned[b] for a, b in self.pair_indices]) * 1000
        filtered, self.notch_state = sosfilt(self.notch, bipolar, axis=-1, zi=self.notch_state)
        filtered, self.band_state = sosfilt(self.band, filtered, axis=-1, zi=self.band_state)
        env_times, env = self.envelope.feed(filtered, start_s)
        return OnlineFrame(start_s, end_s, conditioned,
                           dict(zip(self.hands, filtered)), env_times,
                           dict(zip(self.hands, env)), time.perf_counter() - tick)
