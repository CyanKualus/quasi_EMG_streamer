"""Native-rate AAS and independent monopolar PCA-OBS from the reviewed report."""
from __future__ import annotations

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

MNE_COMMIT = "64473254ed0c2c64627a5864a666686a43ef8be8"
MNE_VERSION = "1.13.0.dev334+g64473254e"
OBS_DELAY_S = 0.212


def gradient_channel(x, fs, tr=2.5, half_window=4):
    """Exclude each target from its template; use eight neighbors at edges."""
    import mne
    from mne.preprocessing import GradientRemover, remove_fmri_gradient_artifact

    cycle = round(fs * tr)
    if cycle < 2 or abs(cycle - fs * tr) > 1e-6:
        raise ValueError("MRI TR must span an integer number of native samples")
    events = np.arange(0, len(x) - cycle + 1, cycle, dtype=int)
    if len(events) < 2 * half_window + 1:
        raise ValueError("MRI correction needs at least nine complete scanner cycles")
    raw = mne.io.RawArray(np.asarray(x)[None],
                          mne.create_info(["EMG"], fs, ["emg"]), verbose="error")
    clean = remove_fmri_gradient_artifact(
        raw, events, picks=["EMG"], window=(half_window, half_window),
        tr_tol=0, copy=True, verbose="error")
    future = GradientRemover(x[None], events, window=(0, 2 * half_window))
    past = GradientRemover(x[None], events, window=(2 * half_window, 0))
    data = clean.get_data()[0]
    for j in range(half_window):
        data[events[j]:events[j] + cycle] = future.get_tr_corrected(j)[0]
    for j in range(len(events) - half_window, len(events)):
        data[events[j]:events[j] + cycle] = past.get_tr_corrected(j)[0]
    return data, int(events[-1] + cycle)


def detect_qrs(ecg, fs=250.0):
    """Detect ECG R peaks afresh for this file; no prior EEG results required."""
    import mne

    raw = mne.io.RawArray(ecg[None], mne.create_info(["ECG"], fs, ["ecg"]),
                          verbose="error")
    events, _, hr = mne.preprocessing.find_ecg_events(
        raw, ch_name="ECG", l_freq=10, h_freq=25,
        qrs_threshold="auto", verbose="error")
    detector = mne.filter.filter_data(ecg, fs, 10, 25, verbose="error")
    selected, duplicates = [], []
    for sample in events[:, 0]:
        sample = int(sample)
        if selected and sample - selected[-1] < round(.12 * fs):
            if abs(detector[sample]) > abs(detector[selected[-1]]):
                duplicates.append(selected.pop() / fs)
                selected.append(sample)
            else:
                duplicates.append(sample / fs)
        else:
            selected.append(sample)
    qrs = np.asarray(selected, dtype=float) / fs
    if len(qrs) < 6:
        raise ValueError("Too few ECG R peaks for cardiac OBS; check the ECG channel")
    margin = max(.6, float(np.median(np.diff(qrs)) / 2 + .02))
    qrs = qrs[(qrs > margin) & (qrs < raw.times[-1] - margin)]
    if len(qrs) < 6 or np.any(np.diff(qrs) < .35):
        raise ValueError("ECG detection failed plausibility checks; inspect ECG before OBS")
    return qrs, {
        "method": "mne.preprocessing.find_ecg_events", "band_hz": [10, 25],
        "qrs_threshold": "auto", "n_detected": len(events), "n_used": len(qrs),
        "estimated_bpm": float(hr), "duplicate_times_removed_s": duplicates,
        "qrs_times_s": qrs.tolist(),
    }


def correct_raw(source, channels, *, ecg_channel="ECG", tr_s=2.5):
    """Read only selected EMG/ECG channels, correct, then return EMG in volts."""
    import mne

    if mne.__version__ != MNE_VERSION:
        raise RuntimeError(
            f"MRI correction requires the reviewed MNE {MNE_VERSION}; found "
            f"{mne.__version__}. Run setup_environment.bat and use the launcher.")
    if ecg_channel not in source.ch_names:
        raise ValueError(f"Cardiac OBS requires ECG channel {ecg_channel!r}")
    if ecg_channel in channels:
        raise ValueError("The ECG channel must be separate from the EMG electrodes")
    fs = float(source.info["sfreq"])
    if fs < 250:
        raise ValueError("The reviewed MRI pipeline requires input at 250 Hz or above")
    with threadpool_limits(limits=1):
        corrected = []
        for channel in [*channels, ecg_channel]:
            native = source.get_data(picks=[channel])[0]
            if not np.all(np.isfinite(native)):
                raise ValueError(f"{channel}: non-finite samples in the recording")
            clean, stop = gradient_channel(native, fs, tr_s)
            corrected.append(mne.filter.resample(
                clean, down=fs / 250, method="polyphase", verbose="error"))
        qrs, ecg_info = detect_qrs(corrected.pop())
        gradient = mne.io.RawArray(np.asarray(corrected),
                                  mne.create_info(channels, 250, ["emg"] * len(channels)),
                                  verbose="error")
        gradient.set_meas_date(source.info["meas_date"])
        gradient.set_annotations(source.annotations.copy())
        gradient.filter(.5, 100, picks="all", phase="zero",
                        fir_design="firwin", verbose="error")
        gradient.notch_filter([50], picks="all", notch_widths=1,
                              phase="zero", verbose="error")
        centers = qrs + OBS_DELAY_S
        final = mne.preprocessing.apply_pca_obs(
            gradient, picks=channels, qrs_times=centers.astype(float),
            n_components=4, n_jobs=1, copy=True, verbose="error")
    duration = final.n_times / 250
    bad = [(0.0, min(5.0, duration)), (max(0.0, duration - 5.0), duration)]
    tail = stop / fs
    if tail < duration:
        bad.append((tail, duration))
    for start, end in bad:
        final.annotations.append(start, end - start, "BAD_MRI_processing")
    final.annotations.append(qrs, np.zeros(len(qrs)), ["MNE_QRS"] * len(qrs))
    if not np.all(np.isfinite(final.get_data())):
        raise ValueError("MRI correction produced non-finite samples")
    note = ("Experimental EMG correction: AAS can create burst copies at neighboring "
            "scanner cycles; cardiac OBS had mixed effects in the reviewed results.")
    warnings.warn(note, UserWarning, stacklevel=2)
    return final, {
        "enabled": True, "method": "MNE AAS + cardiac PCA-OBS",
        "mne_version": mne.__version__, "mne_commit": MNE_COMMIT,
        "fit_channels": channels, "ecg_channel": ecg_channel,
        "bipolar_derivation": "after independent monopolar correction",
        "gradient_window": [4, 4], "edge_windows": [[0, 8], [8, 0]],
        "tr_s": tr_s, "grid_origin_sample": 0,
        "grid_note": "Computational periodic grid; assumes stable TR, not hardware triggers",
        "native_fs": fs, "final_fs": 250, "native_samples": int(source.n_times),
        "final_samples": int(final.n_times), "incomplete_tail_s": [tail, float(source.n_times / fs)],
        "conditioning": "polyphase to 250 Hz; zero-phase 0.5-100 Hz FIR; 50 Hz notch, width 1 Hz",
        "obs_n_components": 4, "obs_delay_s": OBS_DELAY_S, "ecg_detection": ecg_info,
        "bad_intervals_s": bad, "limitations": note,
    }
