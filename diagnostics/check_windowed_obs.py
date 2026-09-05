"""Diagnostic oracle control: vary only the PCA-OBS fitting window.

Uses offline-conditioned AAS and the FULL-recording R-peak list, so this is
explicitly NOT an online candidate. It separates window-local PCA fitting
from changes in gradient templates, ECG detections and FIR edge behavior.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mne
import numpy as np
from threadpoolctl import threadpool_limits

from diagnostics.compare_online_mne import FS, TR, evaluate, json_default
from emgcasting.mri import OBS_DELAY_S


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/online_mne_comparison")
    args = parser.parse_args()
    report = json.loads((args.output / "results.json").read_text())
    results = []
    with threadpool_limits(limits=1):
        for run in report["runs"]:
            data = np.load(args.output / f"p{run['run']}_traces.npz")
            aas = data["offline_aas"]
            qrs = data["qrs"]
            for width in (10, 20, 60):
                output = np.full_like(aas, np.nan)
                boundary_peaks_trimmed = 0
                for stop in range(round(width * FS), aas.shape[1] + 1, round(TR * FS)):
                    start = stop - round(width * FS)
                    local_qrs = qrs[(qrs > start / FS + .6) & (qrs < stop / FS - .6)] - start / FS
                    if len(local_qrs) < 6:
                        continue
                    raw = mne.io.RawArray(aas[:, start:stop],
                        mne.create_info(["F1", "F2"], FS, ["emg", "emg"]), verbose="error")
                    # Guard the separately reproduced MNE inclusive-end bug.
                    # This diagnostic trims a boundary peak instead of counting
                    # fallback-to-AAS as an OBS result.
                    while len(local_qrs) >= 6:
                        indices = raw.time_as_index(local_qrs + OBS_DELAY_S)
                        half_rr = round(np.median(np.diff(indices)) / 2)
                        if indices[-1] + half_rr < raw.n_times:
                            break
                        local_qrs = local_qrs[:-1]
                        boundary_peaks_trimmed += 1
                    if len(local_qrs) < 6:
                        continue
                    clean = mne.preprocessing.apply_pca_obs(raw, picks=["F1", "F2"],
                        qrs_times=local_qrs + OBS_DELAY_S, n_components=4, n_jobs=1, verbose="error")
                    emit_stop = stop - round(TR * FS)
                    emit_start = emit_stop - round(TR * FS)
                    output[:, emit_start:emit_stop] = clean.get_data()[:, -round(2 * TR * FS):-round(TR * FS)]
                metrics = evaluate(output, data["reference"], data["uncorrected"],
                                   data["common_mask"], data["rest_mask"], qrs)
                results.append({"run": run["run"], "buffer_s": width,
                                "boundary_peaks_trimmed": boundary_peaks_trimmed,
                                "common_coverage_pct": 100 * np.mean(np.all(np.isfinite(output[:, data['common_mask']]), axis=0)),
                                "metrics": metrics})
                b = metrics["F1-F2"]
                print(f"p{run['run']} oracle OBS {width}s: waveform r={b['emg_20_95hz_uv']['correlation']:.3f}, "
                      f"envelope r={b['tkeo_envelope']['correlation']:.3f}", flush=True)
    (args.output / "oracle_obs_control.json").write_text(json.dumps({
        "scope": __doc__, "mne_version": mne.__version__, "results": results},
        indent=2, default=json_default, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
