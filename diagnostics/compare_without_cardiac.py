"""Compare the EMG pipeline with and without cardiac PCA-OBS.

Reuses verified 250-Hz offline stages, runs fresh two-channel AAS-only replay,
and evaluates both against identical rest/trial/heartbeat timings. ECG and
OBS are never called in the no-cardiac replay. The application is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from threadpoolctl import threadpool_limits

from diagnostics.compare_online_mne import (
    FS, TR, EMG_CONFIG, ReplaySpec, condition, correlation, finite_filter,
    gradient_replay, json_default, scanner_excess, similarity, write_csv,
)
from emgcasting.core import EventSet, envelope_trace, trial_emg_metrics
from emgcasting.mri import MNE_VERSION, OBS_DELAY_S
from shared.brainvision import read_schedule, video_start


def replay_without_cardiac(gradient, native_fs, spec):
    """Use only F1 and F2; omit ECG detection and OBS entirely."""
    if gradient.shape[0] != 2:
        raise ValueError("The AAS-only replay expects exactly two EMG channels")
    if spec.buffer_s < spec.lag_s + TR or spec.lag_s < 0:
        raise ValueError("Buffer must contain the output chunk and requested lag")
    cycle = round(TR * native_fs)
    width = round(spec.buffer_s * native_fs)
    chunk = round(TR * FS)
    output = np.full((2, round(gradient.shape[1] * FS / native_fs)), np.nan)
    updates = []
    for acquisition_cycle in range(1, gradient.shape[1] // cycle + 1):
        stop = (acquisition_cycle - spec.after) * cycle
        start = stop - width
        if start < 0 or stop <= 0:
            continue
        buffer = gradient[:, start:stop]
        if not np.all(np.isfinite(buffer)):
            continue
        emit_stop = round(stop * FS / native_fs - spec.lag_s * FS)
        emit_start = emit_stop - chunk
        local_start = round(emit_start - start * FS / native_fs)
        tick = time.perf_counter()
        small = (buffer.copy() if native_fs == FS else mne.filter.resample(
            buffer, down=native_fs / FS, method="polyphase", verbose="error"))
        prepared = condition(small).get_data()
        if np.any(np.isfinite(output[:, emit_start:emit_stop])):
            raise AssertionError("An emitted chunk cannot be overwritten")
        output[:, emit_start:emit_stop] = prepared[:, local_start:local_start + chunk]
        updates.append({"acquisition_end_s": acquisition_cycle * TR,
                        "emit_start_s": emit_start / FS, "emit_stop_s": emit_stop / FS,
                        "buffer_start_s": start / native_fs, "buffer_stop_s": stop / native_fs,
                        "compute_s": time.perf_counter() - tick})
    return output, updates


def runs_of_true(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def heartbeat_epochs(signal, qrs, allowed, delay=OBS_DELAY_S, radius_s=.35):
    radius = round(radius_s * FS)
    epochs, indices = [], []
    for t in qrs:
        center = round((t + delay) * FS)
        lo, hi = center - radius, center + radius + 1
        if lo >= 0 and hi <= len(signal) and np.all(allowed[lo:hi]) and np.all(np.isfinite(signal[lo:hi])):
            epoch = signal[lo:hi]
            epochs.append(epoch - np.mean(epoch))
            indices.append(center)
    return np.asarray(epochs), np.asarray(indices, dtype=int)


def heartbeat_measure(signal, qrs, allowed):
    epochs, indices = heartbeat_epochs(signal, qrs, allowed)
    if not len(epochs):
        return {"epochs": 0}, np.full(2 * round(.35 * FS) + 1, np.nan)
    mean = np.mean(epochs, axis=0)
    # Cross-half product is a descriptive coherence check; unlike RMS of a
    # finite-sample average it need not be positive in the absence of a signal.
    a, b = np.mean(epochs[::2], axis=0), np.mean(epochs[1::2], axis=0)
    return {"epochs": len(epochs), "average_rms_uv": float(np.sqrt(np.mean(mean ** 2))),
            "average_peak_to_peak_uv": float(np.ptp(mean)),
            "odd_even_template_r": correlation(a, b),
            "cross_half_power_uv2": float(np.mean(a * b))}, mean


def rest_timing_control(signal, qrs, rest, repetitions=200, seed=24):
    """Descriptive timing-shuffle control with the same number of rest epochs.

    Each R-centered epoch is relocated within its own trimmed rest span, with
    a >=200 ms center displacement. Samples and all epoch widths are retained;
    this is not an independent physiological ground truth or a formal test.
    """
    epochs, indices = heartbeat_epochs(signal, qrs, rest)
    if not len(epochs):
        return {"epochs": 0}
    radius = round(.35 * FS)
    intervals = runs_of_true(rest)
    limits = []
    for idx in indices:
        start, stop = next((a, b) for a, b in intervals if a <= idx < b)
        limits.append((start + radius, stop - radius))
    rng = np.random.default_rng(seed)
    null = []
    offsets = np.arange(-radius, radius + 1)
    for _ in range(repetitions):
        centers = []
        for idx, (low, high) in zip(indices, limits):
            candidate = int(rng.integers(low, high))
            for attempt in range(100):
                if abs(candidate - idx) >= .2 * FS:
                    break
                candidate = int(rng.integers(low, high))
            centers.append(candidate)
        shuffled = signal[np.asarray(centers)[:, None] + offsets]
        shuffled -= np.mean(shuffled, axis=1, keepdims=True)
        null.append(np.sqrt(np.mean(np.mean(shuffled, axis=0) ** 2)))
    observed = float(np.sqrt(np.mean(np.mean(epochs, axis=0) ** 2)))
    return {"epochs": len(epochs), "observed_rms_uv": observed,
            "shuffled_median_rms_uv": float(np.median(null)),
            "shuffled_95th_rms_uv": float(np.percentile(null, 95)),
            "observed_over_shuffled_median": observed / float(np.median(null)),
            "repetitions": repetitions, "seed": seed}


def trial_contrast(signal, onsets, allowed):
    rows = []
    for onset in onsets:
        lo, hi = round(onset * FS), round((onset + 4.2) * FS)
        if lo < 0 or hi > len(signal) or not np.all(allowed[lo:hi]):
            continue
        preparation = signal[round((onset + .2) * FS):round((onset + 1.8) * FS)]
        movement = signal[round((onset + 2.2) * FS):round((onset + 3.8) * FS)]
        prep_rms = float(np.sqrt(np.mean(preparation ** 2)))
        move_rms = float(np.sqrt(np.mean(movement ** 2)))
        rows.append({"onset_s": float(onset), "preparation_rms_uv": prep_rms,
                     "movement_rms_uv": move_rms, "ratio": move_rms / prep_rms})
    return {"n_trials": len(rows), "median_ratio": float(np.median([r["ratio"] for r in rows])),
            "trials": rows}


def evaluate_no_cardiac(data, allowed, rest, qrs_bv, qrs_mne, schedule, controls=False):
    t = np.arange(data.shape[1]) / FS
    metrics = {}
    hb_traces = {}
    for name, x in [("F1", data[0]), ("F2", data[1]), ("F1-F2", data[0] - data[1])]:
        filtered = finite_filter(x)
        valid = allowed & np.isfinite(filtered)
        rest_valid = valid & rest
        env, env_t = envelope_trace(filtered / 1000, FS, EMG_CONFIG)
        rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(filtered, 25)[::5] ** 2, axis=1))
        # Whole envelope window and its TKEO neighbors must be valid/rest.
        env_valid = np.array([np.all(rest_valid[max(0, start - 1):start + 26])
                              for start in range(0, len(filtered) - 24, 5)])
        hb_all, mean_all = heartbeat_measure(filtered, qrs_bv, valid)
        hb_rest, mean_rest = heartbeat_measure(filtered, qrs_bv, rest_valid)
        hb_mne, _ = heartbeat_measure(filtered, qrs_mne, valid)
        hb_traces[name] = {"all": mean_all, "rest": mean_rest}
        metrics[name] = {
            "seconds": float(valid.sum() / FS), "rest_seconds": float(rest_valid.sum() / FS),
            "rest_rms_uv": float(np.sqrt(np.mean(filtered[rest_valid] ** 2))),
            "rest_rms_envelope_median_uv": float(np.median(rms[env_valid])),
            "rest_rms_envelope_p95_uv": float(np.percentile(rms[env_valid], 95)),
            "rest_tkeo_median_mv2": float(np.median(env[env_valid])),
            "scanner_excess_uv2": scanner_excess(x * 1e6, rest_valid),
            "heartbeat_all_bv": hb_all, "heartbeat_rest_bv": hb_rest,
            "heartbeat_all_mne": hb_mne,
        }
        if name == "F1-F2":
            for hand in ("left", "right"):
                metrics[name][hand + "_contrast"] = trial_contrast(filtered, schedule[hand + "_microrepeat"][0], valid)
            # Match the detector's rest calibration and trial set to the same
            # allowed interval; keep the original detector parameters.
            rest_spans = runs_of_true(rest_valid)
            rest_events = EventSet(np.array([a / FS - 1 for a, b in rest_spans]),
                                   np.array([(b - a) / FS + 3 for a, b in rest_spans]))
            onsets = np.array([r["onset_s"] for r in metrics[name]["right_contrast"]["trials"]])
            restricted_env = env.copy()
            env_centers = np.minimum(np.round(env_t * FS).astype(int), len(valid) - 1)
            restricted_env[~valid[env_centers]] = np.nan
            detector = trial_emg_metrics(restricted_env, env_t, onsets, 4.2, rest_events, EMG_CONFIG)
            metrics[name]["classifier"] = {"high_trials": int(detector.high.sum()), "trials": len(detector.high),
                "rest_fpr": float(detector.rest_fpr), "rest_unreliable": detector.rest_unreliable,
                "n_rest_windows": detector.n_rest_windows}
            if controls:
                metrics[name]["rest_timing_control"] = rest_timing_control(filtered, qrs_bv, rest_valid)
    return metrics, hb_traces


def obs_injection_control(aas, reference, qrs, rest_intervals):
    """Probe just OBS with known conditioned EMG increments, keeping ECG fixed.

    Four 600-ms independent 25–85 Hz bursts are added one at a time after AAS
    and conditioning: two near R+212 ms and two midway between such centers.
    This isolates the cost of OBS, rather than re-testing AAS burst echoes.
    """
    def apply(data):
        raw = mne.io.RawArray(data, mne.create_info(["F1", "F2"], FS, ["emg"] * 2), verbose="error")
        return mne.preprocessing.apply_pca_obs(raw, picks=["F1", "F2"],
                    qrs_times=qrs + OBS_DELAY_S, n_components=4, n_jobs=1, verbose="error").get_data()
    base = apply(aas)
    np.testing.assert_allclose(base, reference, rtol=1e-12, atol=1e-14)
    rng = np.random.default_rng(91)
    rows = []
    targets = rest_intervals[1:5]
    for index, (start, stop) in enumerate(targets):
        candidates = qrs[(qrs > start + 2) & (qrs < stop - 2)]
        center = float(candidates[len(candidates) // 2] + OBS_DELAY_S)
        alignment = "near_heartbeat" if index % 2 == 0 else "between_heartbeats"
        if index % 2:
            later = qrs[qrs > center - OBS_DELAY_S + .01][0] + OBS_DELAY_S
            center = (center + later) / 2
        source = rng.normal(size=1500)
        noise = mne.filter.filter_data(source, FS, 25, 85, verbose="error")[650:800] * np.hanning(150)
        noise /= np.sqrt(np.mean(noise ** 2))
        for amplitude_uv in (10, 100):
            signal = np.zeros(aas.shape[1])
            left = round(center * FS) - 75
            signal[left:left + 150] = noise * amplitude_uv * 1e-6
            modified = aas.copy()
            modified[0] += signal / 2
            modified[1] -= signal / 2
            delta = apply(modified) - base
            recovered = finite_filter(delta[0] - delta[1])
            expected = finite_filter(signal)
            mask = np.zeros(len(signal), dtype=bool)
            mask[left - 25:left + 200] = True
            comp = similarity(recovered, expected, mask)
            gain = float(np.dot(recovered[mask], expected[mask]) / np.dot(expected[mask], expected[mask]))
            elsewhere = ~mask & (np.arange(len(signal)) / FS > 5) & (np.arange(len(signal)) / FS < len(signal) / FS - 5)
            rows.append({"center_s": center, "alignment": alignment, "injected_rms_uv": amplitude_uv,
                         "obs_projection_gain": gain, "obs_recovered": comp,
                         "off_target_difference_rms_uv": float(np.sqrt(np.mean((recovered[elsewhere] - expected[elsewhere]) ** 2))),
                         "no_obs_projection_gain": 1.0})
    return rows


def plot_results(output, reports, plot_data):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    names = ["Offline AAS", "+ cardiac OBS", "Online AAS 10 s", "Online AAS 20 s"]
    keys = ["offline_aas", "offline_obs", "aas8_0_buffer10_lag2.5", "aas8_0_buffer20_lag2.5"]
    colors = ["#2670af", "#d77718", "#239063", "#985bb3"]
    times = np.arange(-round(.35 * FS), round(.35 * FS) + 1) / FS
    for col, report in enumerate(reports):
        for key, name, color in zip(keys, names, colors):
            axes[0, col].plot(times, plot_data[report["run"]][key], label=name, color=color, lw=1)
        variants = {v["label"]: v for v in report["variants"]}
        metrics = [report["common_offline"]["offline_aas"], report["common_offline"]["offline_obs"],
                   variants[keys[2]]["metrics"], variants[keys[3]]["metrics"]]
        axes[1, col].bar(np.arange(4), [m["F1-F2"]["rest_rms_uv"] for m in metrics], color=colors)
        axes[1, col].set_xticks(np.arange(4), names, rotation=25, ha="right", fontsize=8)
        axes[0, col].set_title(f"p{report['run']}: heartbeat average")
        axes[0, col].set_xlabel("Seconds from R + 212 ms")
        axes[0, col].grid(alpha=.2)
        axes[1, col].grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("20–95 Hz bipolar EMG (µV)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_ylabel("Rest RMS (µV)")
    fig.suptitle("Cardiac OBS ablation: same 100–320 s interval; supplied BrainVision R markers\n"
                 "Heartbeat averages include all activity; lower rest RMS is not proof of cleaner physiology.", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "cardiac_ablation.png", dpi=160)
    fig.savefig(output / "cardiac_ablation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=ROOT / "output/online_mne_comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "output/no_cardiac_comparison")
    parser.add_argument("--runs", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    args = parser.parse_args()
    if mne.__version__ != MNE_VERSION:
        raise RuntimeError(f"Use pinned MNE {MNE_VERSION}")
    previous = json.loads((args.previous / "results.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"mne_version": mne.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "previous_results_sha256": hashlib.sha256((args.previous / "results.json").read_bytes()).hexdigest(),
              "common_interval_s": [100, 320], "offline_interval": "5 s to duration minus 5 s",
              "online_channels": ["F1", "F2"], "online_ecg_required": False, "runs": []}
    table, full_table, plot_data = [], [], {}
    with threadpool_limits(limits=1):
        for old in previous["runs"]:
            run = old["run"]
            if run not in args.runs:
                continue
            print(f"p{run}: offline cardiac ablation", flush=True)
            archive = np.load(args.previous / f"p{run}_traces.npz")
            raw = mne.io.read_raw_brainvision(old["source"], preload=False, verbose="error")
            native = raw.get_data(picks=["F1", "F2"])
            fs = float(raw.info["sfreq"])
            bv_path = Path(old["source"]).parent.parent / f"in1948_MRbvCBbv_p{run}_EMG.vhdr"
            bv = mne.io.read_raw_brainvision(bv_path, preload=False, verbose="error")
            offset = video_start(raw) - video_start(bv)
            qrs_bv = np.asarray([a["onset"] + offset for a in bv.annotations if a["description"].endswith("Pulse Artifact/R")])
            if len(qrs_bv) < 100:
                raise AssertionError("Missing supplied BrainVision R markers")
            schedule_file = ROOT / "schedules" / f"mri_eeg_order_{old['video']:02d}_conn_microrepeats.csv"
            schedule = {k: (v[0] + old["video_anchor_s"], v[1]) for k, v in read_schedule(schedule_file).events.items()}
            qrs_mne = archive["qrs"]
            t = np.arange(archive["reference"].shape[1]) / FS
            full = (t >= 5) & (t < t[-1] + 1 / FS - 5)
            common, rest = archive["common_mask"], archive["rest_mask"]
            run_report = {"run": run, "source": old["source"], "source_files": old["source_files"],
                          "bv_marker_source": str(bv_path.with_suffix('.vmrk')),
                          "bv_to_raw_offset_s": offset, "bv_r_count": len(qrs_bv),
                          "full_offline": {}, "common_offline": {}, "variants": []}
            plot_data[run] = {}
            for label, data in [("input", archive["uncorrected"]), ("offline_aas", archive["offline_aas"]), ("offline_obs", archive["reference"])]:
                metrics, _ = evaluate_no_cardiac(data, full, rest, qrs_bv, qrs_mne, schedule, controls=label != "input")
                run_report["full_offline"][label] = metrics
                metrics_common, means = evaluate_no_cardiac(data, common, rest, qrs_bv, qrs_mne, schedule)
                run_report["common_offline"][label] = metrics_common
                plot_data[run][label] = means["F1-F2"]["all"]
                b = metrics["F1-F2"]
                full_table.append({"run": run, "stage": label, "rest_rms_uv": b["rest_rms_uv"],
                    "rest_envelope_median_uv": b["rest_rms_envelope_median_uv"], "rest_envelope_p95_uv": b["rest_rms_envelope_p95_uv"],
                    "heartbeat_all_rms_uv": b["heartbeat_all_bv"]["average_rms_uv"],
                    "heartbeat_rest_rms_uv": b["heartbeat_rest_bv"]["average_rms_uv"],
                    "right_movement_preparation_ratio": b["right_contrast"]["median_ratio"],
                    "left_movement_preparation_ratio": b["left_contrast"]["median_ratio"],
                    "high_trials": b["classifier"]["high_trials"], "rest_fpr": b["classifier"]["rest_fpr"],
                    "scanner_excess_uv2": b["scanner_excess_uv2"]})
            specs = [ReplaySpec(v["before"], v["after"], v["buffer_s"], v["lag_s"]) for v in old["variants"]]
            gradient_cache = {}
            saved = {}
            for spec in specs:
                key = spec.before, spec.after
                if key not in gradient_cache:
                    print(f"p{run}: two-channel gradient {key}", flush=True)
                    gradient_cache[key] = gradient_replay(native, fs, *key)
                gradient, grad_timings = gradient_cache[key]
                data, updates = replay_without_cardiac(gradient, fs, spec)
                coverage = 100 * np.mean(np.all(np.isfinite(data[:, common]), axis=0))
                assert coverage == 100
                metrics, means = evaluate_no_cardiac(data, common, rest, qrs_bv, qrs_mne, schedule)
                # Same gradient/FIR settings as the earlier benchmark; verify
                # the fresh no-ECG path against its pre-OBS measurements.
                old_variant = next(v for v in old["variants"] if v["label"] == spec.label)
                for name in ("F1", "F2", "F1-F2"):
                    np.testing.assert_allclose(metrics[name]["rest_rms_uv"],
                        old_variant["aas_only_metrics"][name]["rest_rms_uv"], rtol=1e-11, atol=1e-10)
                paired, _ = evaluate_no_cardiac(archive[spec.label], common, rest, qrs_bv, qrs_mne, schedule)
                x = finite_filter(data[0] - data[1])
                y = finite_filter(archive["offline_aas"][0] - archive["offline_aas"][1])
                envx, env_t = envelope_trace(x / 1000, FS, EMG_CONFIG)
                envy, _ = envelope_trace(y / 1000, FS, EMG_CONFIG)
                envelope = similarity(envx, envy, (env_t >= 100.1) & (env_t < 319.9))
                compute = [u["compute_s"] for u in updates]
                entry = {"label": spec.label, **asdict(spec), "updates": len(updates),
                    "common_coverage_pct": coverage, "metrics": metrics, "matched_with_obs_metrics": paired,
                    "aas_envelope_similarity": envelope,
                    "compute_p95_s": float(np.percentile(compute, 95)), "compute_max_s": max(compute),
                    "gradient_compute_p95_s": float(np.percentile(grad_timings, 95)),
                    "first_output_acquisition_s": updates[0]["acquisition_end_s"],
                    "sample_age_s": [spec.after * TR + spec.lag_s, (spec.after + 1) * TR + spec.lag_s],
                    "prior_obs_accepted_pct": old_variant["common_obs_applied_pct"]}
                run_report["variants"].append(entry)
                b, c = metrics["F1-F2"], paired["F1-F2"]
                table.append({"run": run, "variant": spec.label, "buffer_s": spec.buffer_s,
                    "no_obs_rest_rms_uv": b["rest_rms_uv"], "with_obs_rest_rms_uv": c["rest_rms_uv"],
                    "no_obs_heartbeat_rms_uv": b["heartbeat_all_bv"]["average_rms_uv"],
                    "with_obs_heartbeat_rms_uv": c["heartbeat_all_bv"]["average_rms_uv"],
                    "no_obs_right_contrast": b["right_contrast"]["median_ratio"],
                    "with_obs_right_contrast": c["right_contrast"]["median_ratio"],
                    "no_obs_rest_fpr": b["classifier"]["rest_fpr"], "with_obs_rest_fpr": c["classifier"]["rest_fpr"],
                    "envelope_r_vs_offline_aas": envelope["correlation"], "envelope_nrmse_vs_offline_aas": envelope["nrmse"],
                    "compute_p95_ms": entry["compute_p95_s"] * 1000,
                    "prior_obs_accepted_pct": old_variant["common_obs_applied_pct"]})
                saved[spec.label] = data
                plot_data[run][spec.label] = means["F1-F2"]["all"]
                write_csv(args.output / f"p{run}_{spec.label}_updates.csv", updates)
                print(f"p{run}: {spec.label}, no-OBS envelope r={envelope['correlation']:.4f}", flush=True)
            print(f"p{run}: known EMG burst control for OBS", flush=True)
            run_report["obs_injection_control"] = obs_injection_control(
                archive["offline_aas"], archive["reference"], qrs_mne,
                [(a / FS, b / FS) for a, b in runs_of_true(rest)])
            np.savez_compressed(args.output / f"p{run}_no_obs_traces.npz", **saved)
            report["runs"].append(run_report)
            (args.output / "results.json").write_text(json.dumps(report, indent=2, default=json_default, allow_nan=False), encoding="utf-8")
            write_csv(args.output / "offline_summary.csv", full_table)
            write_csv(args.output / "online_summary.csv", table)
    if len(report["runs"]) == 3:
        plot_results(args.output, report["runs"], plot_data)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
