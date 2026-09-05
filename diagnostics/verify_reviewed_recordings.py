"""Compare this copy with the original code and the report's saved EMG outputs."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mne
import numpy as np

from emgcasting import core
from emgcasting.single_file import analyze_file
from settings import file_config

ORIGINAL = ROOT.parent / "CapTest" / "NeuroCasting_QuasifMRI"
DATA = Path(r"D:\ExpData\MEG\Quasi fMRI\data\Other\fMRI_test")
LEGACY = '''
from pathlib import Path
import sys
import numpy as np
from emgcasting.core import ProcessingConfig, analyze_batch
p = Path(sys.argv[1])
c = ProcessingConfig(str(p.parent), [p.name], participant="in1948",
                     file_type="brainvision", left_channels=None,
                     right_channels=("F1", "F2"), trial_figures=False)
r = analyze_batch(c, require_metrics=False).recordings[0].hands[0]
np.savez(sys.argv[2], epochs=r.analysis.epochs, grid=r.analysis.grid,
         onsets=r.analysis.onsets, high=r.analysis.metrics.high,
         rest_fpr=r.rest_fpr, movement_rest_ratio=r.movement_rest_ratio)
'''


def main():
    reports = []
    with tempfile.TemporaryDirectory() as temporary:
        for run in (1, 2, 3):
            direct_path = DATA / f"in1948_MRbvCBbv_p{run}_EMG.vhdr"
            reference_path = Path(temporary) / f"direct{run}.npz"
            subprocess.run([sys.executable, "-c", LEGACY, str(direct_path),
                            str(reference_path)], cwd=ORIGINAL, check=True)
            direct_config = file_config(direct_path, overrides={"trial_figures": False})
            direct_batch = analyze_file(direct_config)
            actual = direct_batch.recordings[0].hands[0]
            with np.load(reference_path) as expected:
                for key in ("epochs", "grid", "onsets"):
                    np.testing.assert_array_equal(getattr(actual.analysis, key), expected[key])
                np.testing.assert_array_equal(actual.analysis.metrics.high, expected["high"])
                assert actual.rest_fpr == expected["rest_fpr"]
                assert actual.movement_rest_ratio == expected["movement_rest_ratio"]
            core.save_batch_outputs(direct_batch, direct_config)
            raw_path = DATA / "Unfitered" / f"in1948_block{run:02}.vhdr"
            config = file_config(raw_path, mri=True, overrides={"trial_figures": False})
            loaded = core.load_brainvision(raw_path, ["F1", "F2"], config)
            reference_dir = ORIGINAL / "output/in1948/mne_emg_comparison" / f"block{run:02}"
            reference = mne.io.read_raw_fif(
                reference_dir / f"in1948_mne_AAS_OBS_EMG_p{run}_raw.fif",
                preload=False, verbose="error")
            before = reference.get_data(picks=["F1", "F2"])
            now = np.vstack([loaded.signals["F1"], loaded.signals["F2"]])
            np.testing.assert_allclose(now, before, rtol=2e-6, atol=2e-11)
            original_metadata = json.loads((reference_dir / "processing.json").read_text())
            new_metadata = loaded.provenance["mri_correction"]
            np.testing.assert_array_equal(new_metadata["ecg_detection"]["qrs_times_s"],
                                          original_metadata["qrs_times_s"])
            with patch.object(core, "load_brainvision", return_value=loaded):
                batch = analyze_file(config)
            core.save_batch_outputs(batch, config)
            assert batch.recordings[0].provenance["resampled"]
            reports.append({
                "run": run, "direct_matches_original_exactly": True,
                "mri_matches_report_within_fif_precision": True,
                "mri_max_absolute_difference_v": float(np.max(np.abs(now - before))),
                "mri_samples": now.shape[1],
                "qrs_count": len(new_metadata["ecg_detection"]["qrs_times_s"]),
                "qrs_times_match_report_exactly": True,
                "right_hand_trials": batch.recordings[0].hands[0].n_trials,
                "direct_summary": direct_batch.summary_csv,
                "mri_summary": batch.summary_csv,
            })
            print(f"p{run}: direct matches original; corrected EMG and ECG times match report", flush=True)
    destination = ROOT / "output/verification"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reviewed_recordings.json").write_text(
        json.dumps({"mne_version": mne.__version__, "recordings": reports}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
