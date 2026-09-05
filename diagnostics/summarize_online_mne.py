"""Make comparison tables, figures and a synthetic AAS burst control."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend
from threadpoolctl import threadpool_limits

from diagnostics.compare_online_mne import gradient_replay, write_csv


def burst_probe():
    """AAS is linear: an isolated increment shows deterministic burst echoes.

    This tests the gradient stage only, with exact stable TR alignment; it
    does not establish physiological preservation by cardiac OBS.
    """
    fs = 5000
    cycle = 12500
    data = np.zeros((1, 40 * cycle))
    t = np.arange(cycle) / fs
    burst = np.sin(2 * np.pi * 63 * t) * np.exp(-.5 * ((t - 1.25) / .12) ** 2)
    data[0, 20 * cycle:21 * cycle] = burst
    target = detrend(burst)
    denominator = np.dot(target, target)
    result = []
    with threadpool_limits(limits=1):
        for before, after in [(2, 0), (4, 0), (8, 0), (12, 0), (4, 4)]:
            clean, _ = gradient_replay(data, fs, before, after)
            gains = {}
            for offset in range(-4, 13):
                chunk = clean[0, (20 + offset) * cycle:(21 + offset) * cycle]
                gains[str(offset)] = float(np.dot(chunk, target) / denominator)
            result.append({"before": before, "after": after, "burst_projection_by_cycle_offset": gains})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/online_mne_comparison")
    args = parser.parse_args()
    report = json.loads((args.output / "results.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#2670af", "#d77718", "#239063"]
    for run, color in zip(report["runs"], colors):
        for before, after, style in [(8, 0, "-"), (4, 4, "--")]:
            variants = sorted((v for v in run["variants"] if v["before"] == before
                               and v["after"] == after and v["lag_s"] == 2.5),
                              key=lambda v: v["buffer_s"])
            x = [v["buffer_s"] for v in variants]
            label = f"p{run['run']} {'past 8' if after == 0 else 'past 4 / future 4'}"
            bipolar = [v["metrics"]["F1-F2"] for v in variants]
            values = [[b["tkeo_envelope"]["correlation"] for b in bipolar],
                      [b["emg_20_95hz_uv"]["nrmse"] for b in bipolar],
                      [b["rest_rms_uv"] / b["reference_rest_rms_uv"] for b in bipolar],
                      [v["common_obs_applied_pct"] for v in variants]]
            for ax, y in zip(axes.flat, values):
                ax.plot(x, y, style, color=color, marker="o", ms=4, label=label)
    for ax, title in zip(axes.flat, ["TKEO envelope correlation with offline",
                                    "20–95 Hz waveform normalized RMS error",
                                    "Rest RMS / offline rest RMS", "Time from accepted OBS windows (%)"]):
        ax.set_title(title)
        ax.set_xlabel("Downstream buffer (seconds)")
        ax.grid(alpha=.25)
        ax.set_xticks([5, 10, 20, 30, 60])
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[1, 0].axhline(1, color="gray", lw=.8)
    axes[1, 1].set_ylim(-3, 103)
    fig.suptitle("F1−F2: bounded MNE replay, same 100–320 s evaluation interval\n"
                 "Solid: 2.5–5 s output age; dashed: 12.5–15 s. Failed OBS windows use AAS only.", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / "window_comparison.png", dpi=170)
    fig.savefig(args.output / "window_comparison.pdf")
    plt.close(fig)

    tables = []
    labels = [v["label"] for v in report["runs"][0]["variants"]]
    for label in labels:
        variants = [v for run in report["runs"] for v in run["variants"] if v["label"] == label]
        row = {"variant": label}
        metrics = {
            "waveform_r": [v["metrics"]["F1-F2"]["emg_20_95hz_uv"]["correlation"] for v in variants],
            "waveform_nrmse": [v["metrics"]["F1-F2"]["emg_20_95hz_uv"]["nrmse"] for v in variants],
            "envelope_r": [v["metrics"]["F1-F2"]["tkeo_envelope"]["correlation"] for v in variants],
            "envelope_nrmse": [v["metrics"]["F1-F2"]["tkeo_envelope"]["nrmse"] for v in variants],
            "rest_ratio": [v["metrics"]["F1-F2"]["rest_rms_uv"] / v["metrics"]["F1-F2"]["reference_rest_rms_uv"] for v in variants],
            "obs_applied_pct": [v["common_obs_applied_pct"] for v in variants],
            "p95_compute_ms": [v["compute_p95_s"] * 1000 for v in variants],
        }
        for key, values in metrics.items():
            row[key + "_median"] = float(np.median(values))
            row[key + "_min"] = float(min(values))
            row[key + "_max"] = float(max(values))
        tables.append(row)
    write_csv(args.output / "aggregated.csv", tables)
    burst = burst_probe()
    (args.output / "synthetic_gradient_burst.json").write_text(json.dumps(burst, indent=2))
    print("variant | wave r | wave NRMSE | env r | env NRMSE | rest ratio | OBS % | p95 ms (medians)")
    for row in tables:
        print(row["variant"], " | ", " | ".join(f"{row[k + '_median']:.3f}" for k in metrics))
    for run in report["runs"]:
        print("\np", run["run"], "offline", run["offline_metrics"]["F1-F2"], flush=True)
        print("gradient", [(g["before"], g["after"], g["bipolar_native_similarity"]["correlation"],
                            g["bipolar_native_similarity"]["nrmse"], g["bipolar_max_difference_uv"])
                           for g in run["gradient"]])


if __name__ == "__main__":
    main()
