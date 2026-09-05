"""Summarize completed cardiac-ablation runs and plot their EMG envelopes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.compare_online_mne import FS, EMG_CONFIG, finite_filter, write_csv
from emgcasting.core import envelope_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/no_cardiac_comparison")
    parser.add_argument("--previous", type=Path, default=ROOT / "output/online_mne_comparison")
    args = parser.parse_args()
    report = json.loads((args.output / "results.json").read_text())
    aggregated = []
    for width in (5, 7.5, 10, 15, 20, 30, 60):
        label = f"aas8_0_buffer{width:g}_lag2.5"
        variants = [next(v for v in r["variants"] if v["label"] == label) for r in report["runs"]]
        row = {"buffer_s": width}
        for key, values in {
            "envelope_r": [v["aas_envelope_similarity"]["correlation"] for v in variants],
            "envelope_nrmse": [v["aas_envelope_similarity"]["nrmse"] for v in variants],
            "p95_ms": [v["compute_p95_s"] * 1000 for v in variants],
        }.items():
            row[key + "_min"] = min(values)
            row[key + "_max"] = max(values)
        aggregated.append(row)
    write_csv(args.output / "window_summary.csv", aggregated)
    fig, axes = plt.subplots(len(report["runs"]), 2, figsize=(13, 3 * len(report["runs"])), squeeze=False)
    for row, result in enumerate(report["runs"]):
        run = result["run"]
        previous = np.load(args.previous / f"p{run}_traces.npz")
        replay = np.load(args.output / f"p{run}_no_obs_traces.npz")
        for key, name, color in [("offline_aas", "Offline AAS only", "#2670af"),
                                  ("reference", "Offline AAS + OBS", "#d77718"),
                                  ("aas8_0_buffer10_lag2.5", "Online AAS, 10 s buffer", "#239063")]:
            data = replay[key] if key.startswith("aas8") else previous[key]
            bipolar = finite_filter(data[0] - data[1]) / 1000
            env, t = envelope_trace(bipolar, FS, EMG_CONFIG)
            for ax, limits in zip(axes[row], [(100, 320), (135, 140)]):
                mask = (t >= limits[0]) & (t < limits[1])
                ax.plot(t[mask], env[mask], lw=1, alpha=.8, color=color, label=name)
                ax.grid(alpha=.2)
                ax.set_xlabel("Recording time (s)")
                ax.set_ylabel("TKEO (mV²)")
        axes[row, 0].set_title(f"p{run}: common comparison interval")
        axes[row, 1].set_title(f"p{run}: fixed scheduled-rest excerpt")
        axes[row, 0].legend(fontsize=7)
    fig.suptitle("EMG pipeline with cardiac OBS omitted\n"
                 "Output aligned by sample time; online AAS has 2.5–5 s release delay", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / "no_cardiac_envelopes.png", dpi=160)
    fig.savefig(args.output / "no_cardiac_envelopes.pdf")
    plt.close(fig)
    print("WINDOWS", json.dumps(aggregated))
    for result in report["runs"]:
        no, yes = (result["full_offline"][k]["F1-F2"] for k in ("offline_aas", "offline_obs"))
        print("RUN", result["run"], json.dumps({
            "rest_no_yes": [no["rest_rms_uv"], yes["rest_rms_uv"]],
            "rest_reduction_pct_omitting_obs": 100 * (1 - no["rest_rms_uv"] / yes["rest_rms_uv"]),
            "median_rms_envelope_no_yes": [no["rest_rms_envelope_median_uv"], yes["rest_rms_envelope_median_uv"]],
            "heartbeat_all_no_yes": [no["heartbeat_all_bv"]["average_rms_uv"], yes["heartbeat_all_bv"]["average_rms_uv"]],
            "heartbeat_rest_no_yes": [no["heartbeat_rest_bv"]["average_rms_uv"], yes["heartbeat_rest_bv"]["average_rms_uv"]],
            "right_contrast_no_yes": [no["right_contrast"]["median_ratio"], yes["right_contrast"]["median_ratio"]],
            "classifier_no_yes": [no["classifier"], yes["classifier"]],
            "timing_no_yes": [no["rest_timing_control"], yes["rest_timing_control"]],
            "injection_gain_range": [min(x["obs_projection_gain"] for x in result["obs_injection_control"]),
                                     max(x["obs_projection_gain"] for x in result["obs_injection_control"])],
        }))


if __name__ == "__main__":
    main()
