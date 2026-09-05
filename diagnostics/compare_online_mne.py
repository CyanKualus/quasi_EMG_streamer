"""Replay the reviewed recordings through bounded MNE MRI cleaning buffers.

This is an experiment, not an acquisition driver. Raw recordings are read only.
GradientRemover receives only completed, available TRs. Downstream MNE calls
receive only a trailing buffer of already available gradient-corrected data.
Output chunks are committed once; later buffers cannot revise earlier output.
Run with the application's pinned .venv (see --help).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
from scipy.signal import butter, iirnotch, sosfilt, tf2sos, welch
from threadpoolctl import threadpool_limits

from emgcasting.core import ProcessingConfig, envelope_trace, filter_emg
from emgcasting.mri import (MNE_VERSION, OBS_DELAY_S, correct_raw, detect_qrs,
                            gradient_channel)
from shared.brainvision import read_schedule, video_start

FS = 250
TR = 2.5
CHANNELS = ["F1", "F2", "ECG"]
DEFAULT_DATA = Path(r"D:\ExpData\MEG\Quasi fMRI\data\Other\fMRI_test\Unfitered")
EMG_CONFIG = ProcessingConfig(".", ["replay"], left_channels=None)


@dataclass(frozen=True)
class ReplaySpec:
    before: int
    after: int
    buffer_s: float
    lag_s: float = 2.5

    @property
    def label(self):
        return f"aas{self.before}_{self.after}_buffer{self.buffer_s:g}_lag{self.lag_s:g}"


def gradient_replay(data, fs, before, after=0, tr=TR):
    """Emit complete cycles with a bounded buffer; unavailable samples are NaN.

    A target j is released at the END of cycle j+after. Even with after=0,
    linear detrending needs the complete target cycle. No offline edge repair
    or final flushing using future/unavailable data is performed.
    """
    cycle = round(fs * tr)
    if before + after < 1 or min(before, after) < 0:
        raise ValueError("At least one nonnegative template neighbor is required")
    n_cycles = data.shape[1] // cycle
    result = np.full_like(data, np.nan, dtype=float)
    timings = []
    span = before + after + 1
    for arrived in range(span, n_cycles + 1):
        start = (arrived - span) * cycle
        stop = arrived * cycle
        tick = time.perf_counter()
        # A copy enforces the data boundary: the MNE object cannot see future data.
        buffer = data[:, start:stop].copy()
        remover = mne.preprocessing.GradientRemover(
            buffer, np.arange(span) * cycle, window=(before, after))
        clean = remover.get_tr_corrected(before)
        target = start + before * cycle
        result[:, target:target + cycle] = clean
        timings.append(time.perf_counter() - tick)
    return result, np.asarray(timings)


def condition(data):
    raw = mne.io.RawArray(data, mne.create_info(["F1", "F2"], FS, ["emg"] * 2),
                          verbose="error")
    raw.filter(.5, 100, picks="all", phase="zero", fir_design="firwin", verbose="error")
    raw.notch_filter([50], picks="all", notch_widths=1, phase="zero", verbose="error")
    return raw


def full_reference(source):
    """Use the production entry point and separately recover its AAS stage."""
    tick = time.perf_counter()
    with warnings.catch_warnings(record=True):
        final, metadata = correct_raw(source, ["F1", "F2"])
    elapsed = time.perf_counter() - tick
    native = source.get_data(picks=CHANNELS)
    fs = source.info["sfreq"]
    gradient = np.stack([gradient_channel(x, fs)[0] for x in native])
    small = mne.filter.resample(gradient, down=fs / FS, method="polyphase", verbose="error")
    aas = condition(small[:2]).get_data()
    uncorrected = condition(mne.filter.resample(
        native[:2], down=fs / FS, method="polyphase", verbose="error")).get_data()
    return native, gradient, aas, uncorrected, final.get_data(), metadata, elapsed


def replay_downstream(gradient, native_fs, spec):
    """Trailing-window resampling, FIR, ECG detection and four-component OBS.

    MNE's batch ECG detector and OBS are recomputed inside the bounded buffer.
    They are NOT incremental APIs. If production ECG checks reject a window,
    explicitly fall back to AAS+conditioning and record that output as no-OBS.
    """
    cycle = round(TR * native_fs)
    small_cycle = round(TR * FS)
    n_small = round(gradient.shape[1] / native_fs * FS)
    final = np.full((2, n_small), np.nan)
    aas_only = np.full_like(final, np.nan)
    obs_applied = np.zeros(n_small, dtype=bool)
    qrs_committed = []
    updates = []
    warning_counts = Counter()
    fail_counts = Counter()
    width = round(spec.buffer_s * native_fs)
    lag = round(spec.lag_s * FS)
    if width < cycle or spec.lag_s + TR > spec.buffer_s:
        raise ValueError("Buffer must hold the emitted cycle plus its requested lag")
    # At acquisition_end, the latest available corrected cycle ends after
    # subtracting the future-neighbor delay from the acquisition position.
    for acquisition_cycle in range(1, gradient.shape[1] // cycle + 1):
        available_end = (acquisition_cycle - spec.after) * cycle
        start = available_end - width
        if start < 0 or available_end <= 0:
            continue
        buffer = gradient[:, start:available_end]
        if not np.all(np.isfinite(buffer)):
            continue
        emit_stop = round(available_end / native_fs * FS) - lag
        emit_start = emit_stop - small_cycle
        local_start = round(emit_start - start / native_fs * FS)
        local_stop = local_start + small_cycle
        tick = time.perf_counter()
        failure = ""
        qrs = np.array([])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            small = (buffer.copy() if native_fs == FS else mne.filter.resample(
                buffer, down=native_fs / FS, method="polyphase", verbose="error"))
            prepared = condition(small[:2])
            prepared_data = prepared.get_data()
            try:
                qrs, _ = detect_qrs(small[2])
                corrected = mne.preprocessing.apply_pca_obs(
                    prepared, picks=["F1", "F2"], qrs_times=qrs + OBS_DELAY_S,
                    n_components=4, n_jobs=1, copy=True, verbose="error").get_data()
                if not np.all(np.isfinite(corrected)):
                    raise ValueError("OBS produced non-finite samples")
            except (ValueError, IndexError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                fail_counts[failure] += 1
                corrected = prepared_data
        elapsed = time.perf_counter() - tick
        for warning in caught:
            warning_counts[str(warning.message)] += 1
        if np.any(np.isfinite(final[:, emit_start:emit_stop])):
            raise AssertionError("An earlier emitted chunk must never be revised")
        final[:, emit_start:emit_stop] = corrected[:, local_start:local_stop]
        aas_only[:, emit_start:emit_stop] = prepared_data[:, local_start:local_stop]
        obs_applied[emit_start:emit_stop] = not failure
        global_qrs = qrs + start / native_fs
        qrs_committed.extend(global_qrs[(global_qrs >= emit_start / FS)
                                        & (global_qrs < emit_stop / FS)].tolist())
        updates.append({"acquisition_end_s": acquisition_cycle * TR,
                        "emit_start_s": emit_start / FS, "emit_stop_s": emit_stop / FS,
                        "buffer_start_s": start / native_fs,
                        "buffer_stop_s": available_end / native_fs,
                        "compute_s": elapsed, "qrs_count": len(qrs),
                        "obs_applied": not failure, "failure": failure})
    return final, aas_only, obs_applied, np.asarray(qrs_committed), updates, warning_counts, fail_counts


def finite_filter(x):
    """Keep causal EMG filter state across emitted chunks, reset only on gaps."""
    good = np.isfinite(x)
    edges = np.diff(np.r_[False, good, False].astype(int))
    result = np.full_like(x, np.nan)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        result[start:stop] = filter_emg(x[start:stop], FS, EMG_CONFIG) * 1000  # uV
    return result


def correlation(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def similarity(candidate, reference, mask):
    valid = mask & np.isfinite(candidate) & np.isfinite(reference)
    x, y = candidate[valid], reference[valid]
    if not len(x):
        return {"n_samples": 0}
    rms_y = np.sqrt(np.mean(y ** 2))
    return {"n_samples": len(x), "correlation": correlation(x, y),
            "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "nrmse": float(np.sqrt(np.mean((x - y) ** 2)) / rms_y),
            "rms_ratio": float(np.sqrt(np.mean(x ** 2)) / rms_y)}


def scanner_excess(x, mask):
    """Welch on separate contiguous rest spans; never concatenate time gaps."""
    good = mask & np.isfinite(x)
    edges = np.diff(np.r_[False, good, False].astype(int))
    spectra, weights = [], []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= 4 * FS:
            freq, power = welch(x[start:stop], FS, nperseg=4 * FS, noverlap=2 * FS)
            spectra.append(power)
            weights.append(stop - start)
    if not spectra:
        return None
    power = np.average(spectra, axis=0, weights=weights)
    excess = 0.0
    for harmonic in np.arange(18.4, 100, 18.4):
        line = np.abs(freq - harmonic) <= .5
        flank = (np.abs(freq - harmonic) >= .75) & (np.abs(freq - harmonic) <= 2)
        excess += np.maximum(power[line] - np.median(power[flank]), 0).sum() * (freq[1] - freq[0])
    return float(excess)


def evaluate(candidate, reference, uncorrected, common, rest, qrs):
    result = {}
    for name, x, y, z in zip(["F1", "F2", "F1-F2"],
                             [*candidate, candidate[0] - candidate[1]],
                             [*reference, reference[0] - reference[1]],
                             [*uncorrected, uncorrected[0] - uncorrected[1]]):
        xf, yf, zf = (finite_filter(v) for v in (x, y, z))
        raw_metrics = similarity(x * 1e6, y * 1e6, common)
        band_metrics = similarity(xf, yf, common)
        envx, centers = envelope_trace(xf / 1000, FS, EMG_CONFIG)
        envy, _ = envelope_trace(yf / 1000, FS, EMG_CONFIG)
        # Require the whole 100-ms envelope plus TKEO neighbors to lie in common.
        center_indices = np.round(centers * FS).astype(int)
        lo, hi = center_indices - 14, center_indices + 15
        cumulative = np.r_[0, np.cumsum(~common)]
        envmask = (lo >= 0) & (hi <= len(common))
        valid_indices = np.flatnonzero(envmask)
        envmask[valid_indices] &= (cumulative[hi[valid_indices]] - cumulative[lo[valid_indices]]) == 0
        envelope_metrics = similarity(envx, envy, envmask)
        valid_rest = common & rest & np.isfinite(xf)
        residual = scanner_excess(x * 1e6, valid_rest)
        input_excess = scanner_excess(z * 1e6, valid_rest)
        heartbeat_epochs = []
        radius = round(.35 * FS)
        for t in qrs:
            p = round((t + OBS_DELAY_S) * FS)
            lo, hi = p - radius, p + radius + 1
            if lo >= 0 and hi <= len(common) and np.all(common[lo:hi]) and np.all(np.isfinite(xf[lo:hi])):
                epoch = xf[lo:hi]
                heartbeat_epochs.append(epoch - np.mean(epoch))
        heartbeat = (float(np.sqrt(np.mean(np.mean(heartbeat_epochs, axis=0) ** 2)))
                     if heartbeat_epochs else None)
        result[name] = {"conditioned_waveform_uv": raw_metrics,
                        "emg_20_95hz_uv": band_metrics, "tkeo_envelope": envelope_metrics,
                        "rest_samples": int(valid_rest.sum()),
                        "rest_rms_uv": float(np.sqrt(np.mean(xf[valid_rest] ** 2))) if valid_rest.any() else None,
                        "reference_rest_rms_uv": float(np.sqrt(np.mean(yf[valid_rest] ** 2))) if valid_rest.any() else None,
                        "input_rest_rms_uv": float(np.sqrt(np.mean(zf[valid_rest] ** 2))) if valid_rest.any() else None,
                        "scanner_excess_uv2": residual,
                        "scanner_excess_reduction_pct": 100 * (1 - residual / input_excess) if input_excess else None,
                        "heartbeat_average_rms_uv": heartbeat, "heartbeat_epochs": len(heartbeat_epochs)}
    return result


def qrs_agreement(candidate, reference, mask, tolerance=.04):
    def inside(times):
        idx = np.round(times * FS).astype(int)
        ok = (idx >= 0) & (idx < len(mask))
        return times[ok][mask[idx[ok]]]
    a, b = inside(candidate), inside(reference)
    matched = set()
    errors = []
    for t in a:
        if not len(b):
            continue
        j = int(np.argmin(np.abs(b - t)))
        if j not in matched and abs(b[j] - t) <= tolerance:
            matched.add(j)
            errors.append(abs(b[j] - t))
    return {"candidate_peaks": len(a), "reference_peaks": len(b), "matches": len(matched),
            "precision": len(matched) / len(a) if len(a) else None,
            "recall": len(matched) / len(b) if len(b) else None,
            "median_absolute_error_ms": float(np.median(errors) * 1000) if errors else None,
            "tolerance_ms": tolerance * 1000}


def stateful_filter_check(values, chunk_seconds=(2, 5, 10)):
    """The existing SciPy EMG filters already support streaming with saved zi."""
    notch = tf2sos(*iirnotch(50, 50, fs=FS))
    band = butter(4, (20, 95), btype="bandpass", output="sos", fs=FS)
    expected = filter_emg(values, FS, EMG_CONFIG)
    results = []
    for seconds in chunk_seconds:
        zi_n = np.zeros((len(notch), 2))
        zi_b = np.zeros((len(band), 2))
        actual, reset = [], []
        for start in range(0, len(values), round(seconds * FS)):
            block = values[start:start + round(seconds * FS)]
            y, zi_n = sosfilt(notch, block * 1000, zi=zi_n)
            y, zi_b = sosfilt(band, y, zi=zi_b)
            actual.append(y)
            reset.append(filter_emg(block, FS, EMG_CONFIG))
        actual, reset = np.concatenate(actual), np.concatenate(reset)
        results.append({"chunk_s": seconds,
                        "stateful_max_difference_mv": float(np.max(np.abs(actual - expected))),
                        "reset_each_chunk_nrmse": float(np.linalg.norm(reset - expected) / np.linalg.norm(expected))})
    return results


def specifications():
    # Five seconds cannot generally supply six usable heartbeats. Test rather
    # than silently shrink n_components or borrow a full-recording R-peak list.
    specs = [ReplaySpec(8, 0, w) for w in (5, 7.5, 10, 15, 20, 30, 60)]
    specs += [ReplaySpec(4, 4, w) for w in (10, 20, 30, 60)]
    specs += [ReplaySpec(n, 0, 20) for n in (2, 4, 12)]
    specs += [ReplaySpec(8, 0, 20, lag) for lag in (0, 5)]
    return specs


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def plot_recording(outdir, run, traces, reference, common):
    selected = ["aas8_0_buffer5_lag2.5", "aas8_0_buffer20_lag2.5", "aas8_0_buffer60_lag2.5",
                "aas4_4_buffer30_lag2.5"]
    ref = finite_filter(reference[0] - reference[1])
    ref_env, centers = envelope_trace(ref / 1000, FS, EMG_CONFIG)
    display = (centers >= 170) & (centers <= 220)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, label in zip(axes, selected):
        if label not in traces:
            continue
        x = finite_filter(traces[label][0] - traces[label][1])
        env, _ = envelope_trace(x / 1000, FS, EMG_CONFIG)
        ax.plot(centers[display], ref_env[display], color="black", lw=1, label="Current offline")
        ax.plot(centers[display], env[display], alpha=.8, lw=1, label=label)
        ax.set_ylabel("TKEO (mV²)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Recording time (s); acquisition delay is reported separately")
    fig.suptitle(f"p{run}: emitted online replay vs current offline envelope")
    fig.tight_layout()
    fig.savefig(outdir / f"p{run}_envelopes.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "output/online_mne_comparison")
    parser.add_argument("--runs", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--quick", action="store_true", help="Only 10/20-second causal and 20-second symmetric buffers")
    args = parser.parse_args()
    if mne.__version__ != MNE_VERSION:
        raise RuntimeError(f"Use pinned MNE {MNE_VERSION}; got {mne.__version__}")
    args.output.mkdir(parents=True, exist_ok=True)
    specs = specifications() if not args.quick else [ReplaySpec(8, 0, 10), ReplaySpec(8, 0, 20), ReplaySpec(4, 4, 20)]
    report = {"mne_version": mne.__version__, "numpy_version": np.__version__,
              "scipy_version": scipy.__version__, "python": sys.version, "platform": platform.platform(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "channels": CHANNELS, "tr_s": TR, "update_interval_s": TR,
              "common_interval_s": [100, 320], "runs": [],
              "scope": "Bounded numerical replay; no LSL transport or GUI benchmark. No future R-peak list in replay."}
    table = []
    with threadpool_limits(limits=1):
        for run in args.runs:
            path = args.data_dir / f"in1948_block{run:02d}.vhdr"
            print(f"p{run}: production reference", flush=True)
            source = mne.io.read_raw_brainvision(path, preload=False, verbose="error")
            native, baseline_gradient, baseline_aas, uncorrected, reference, metadata, offline_s = full_reference(source)
            native_fs = float(source.info["sfreq"])
            qrs = np.asarray(metadata["ecg_detection"]["qrs_times_s"])
            t = np.arange(reference.shape[1]) / FS
            common = (t >= 100) & (t < 320)
            rest = np.zeros(len(t), dtype=bool)
            schedule = read_schedule(ROOT / "schedules" / f"mri_eeg_order_{4-run:02d}_conn_microrepeats.csv", ["rest"])
            anchor = video_start(source, path.name)
            for onset, duration in zip(*schedule.events["rest"]):
                rest |= (t >= anchor + onset + 1) & (t < anchor + onset + duration - 2)
            run_report = {"run": run, "source": str(path), "native_fs": native_fs,
                          "native_samples": source.n_times, "video": 4-run, "video_anchor_s": anchor,
                          "source_files": [{"path": str(path.with_suffix(ext)),
                                            "size": path.with_suffix(ext).stat().st_size,
                                            "mtime_ns": path.with_suffix(ext).stat().st_mtime_ns}
                                           for ext in (".vhdr", ".vmrk", ".eeg")],
                          "offline_compute_s": offline_s, "reference_qrs_count": len(qrs),
                          "offline_metrics": evaluate(reference, reference, uncorrected, common, rest, qrs),
                          "offline_aas_metrics": evaluate(baseline_aas, reference, uncorrected, common, rest, qrs),
                          "stateful_emg_filter": stateful_filter_check(reference[0] - reference[1]),
                          "gradient": [], "variants": []}
            gradient_cache = {}
            for before, after in sorted({(s.before, s.after) for s in specs} | {(1, 0), (2, 0), (4, 0), (8, 0), (12, 0), (1, 1), (2, 2), (4, 4)}):
                print(f"p{run}: gradient ({before}, {after})", flush=True)
                grad, elapsed = gradient_replay(native, native_fs, before, after)
                if (before, after) in {(s.before, s.after) for s in specs}:
                    gradient_cache[before, after] = grad
                native_common = (np.arange(native.shape[1]) / native_fs >= 100) & (np.arange(native.shape[1]) / native_fs < 320)
                x, y = grad[0] - grad[1], baseline_gradient[0] - baseline_gradient[1]
                matched = similarity(x * 1e6, y * 1e6, native_common)
                valid = native_common & np.isfinite(x)
                max_difference = float(np.max(np.abs(x[valid] - y[valid])) * 1e6)
                if (before, after) == (4, 4):
                    np.testing.assert_allclose(grad[:, native_common], baseline_gradient[:, native_common], rtol=1e-12, atol=1e-14)
                run_report["gradient"].append({"before": before, "after": after,
                    "template_seconds": (before + after) * TR, "buffer_s": (before + after + 1) * TR,
                    "sample_age_range_s": [after * TR, (after + 1) * TR],
                    "p95_compute_s": float(np.percentile(elapsed, 95)),
                    "bipolar_native_similarity": matched, "bipolar_max_difference_uv": max_difference})
            traces = {}
            for spec in specs:
                print(f"p{run}: {spec.label}", flush=True)
                final, aas, obs_mask, peaks, updates, warns, failures = replay_downstream(
                    gradient_cache[spec.before, spec.after], native_fs, spec)
                metrics = evaluate(final, reference, uncorrected, common, rest, qrs)
                durations = [u["compute_s"] for u in updates]
                valid = common & np.all(np.isfinite(final), axis=0)
                summary = {**asdict(spec), "label": spec.label, "updates": len(updates),
                           "first_acquisition_output_s": updates[0]["acquisition_end_s"] if updates else None,
                           "sample_age_range_s": [spec.after * TR + spec.lag_s, (spec.after + 1) * TR + spec.lag_s],
                           "compute_p50_s": float(np.median(durations)),
                           "compute_p95_s": float(np.percentile(durations, 95)),
                           "compute_max_s": float(max(durations)),
                           "compute_total_s": sum(durations), "deadline_misses": sum(d > TR for d in durations),
                           "common_coverage_pct": 100 * valid.sum() / common.sum(),
                           "common_obs_applied_pct": 100 * (valid & obs_mask).sum() / common.sum(),
                           "failed_obs_updates": sum(failures.values()), "failure_reasons": dict(failures),
                           "warning_counts": dict(warns), "metrics": metrics,
                           "aas_only_metrics": evaluate(aas, baseline_aas, uncorrected, common, rest, qrs),
                           "qrs_agreement_with_offline": qrs_agreement(peaks, qrs, common & obs_mask)}
                run_report["variants"].append(summary)
                write_csv(args.output / f"p{run}_{spec.label}_updates.csv", updates)
                traces[spec.label] = final
                b = metrics["F1-F2"]
                table.append({"run": run, "variant": spec.label, **asdict(spec),
                              "coverage_pct": summary["common_coverage_pct"],
                              "obs_applied_pct": summary["common_obs_applied_pct"],
                              "waveform_r": b["emg_20_95hz_uv"].get("correlation"),
                              "waveform_nrmse": b["emg_20_95hz_uv"].get("nrmse"),
                              "envelope_r": b["tkeo_envelope"].get("correlation"),
                              "envelope_nrmse": b["tkeo_envelope"].get("nrmse"),
                              "rest_rms_uv": b["rest_rms_uv"],
                              "offline_rest_rms_uv": b["reference_rest_rms_uv"],
                              "scanner_peak_reduction_pct": b["scanner_excess_reduction_pct"],
                              "heartbeat_average_rms_uv": b["heartbeat_average_rms_uv"],
                              "min_delay_s": summary["sample_age_range_s"][0],
                              "max_delay_s": summary["sample_age_range_s"][1],
                              "p95_update_compute_s": summary["compute_p95_s"]})
            plot_recording(args.output, run, traces, reference, common)
            # Low-rate traces permit independent plots/recalculation without
            # reading or duplicating the 5000-Hz input recordings.
            np.savez_compressed(args.output / f"p{run}_traces.npz", reference=reference,
                                offline_aas=baseline_aas, uncorrected=uncorrected,
                                common_mask=common, rest_mask=rest, qrs=qrs, **traces)
            report["runs"].append(run_report)
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2, default=json_default, allow_nan=False), encoding="utf-8")
            write_csv(args.output / "summary.csv", table)
            print(f"p{run}: saved", flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
