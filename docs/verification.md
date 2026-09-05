# Verification — 5 September 2026

The standalone copy passed **56 tests** (`.venv/Scripts/python.exe -m pytest -q`).
These include the retained EMG classifier regressions and new tests for
single-file BrainVision loading, calibration, schedule lookup, MRI selection,
file readiness, output separation, artifact subtraction and the two-tab GUI.

`diagnostics/verify_reviewed_recordings.py` additionally checked all three
reviewed in1948 recordings using the local pinned environment.

| Run | Direct EMG vs original software | MNE F1/F2 vs report FIF | Maximum sample difference | ECG R detections |
|---|---|---|---:|---:|
| p1 / block01 | Exact array equality | Matches within FIF storage precision | 0.000853 µV | 397, identical times |
| p2 / block02 | Exact array equality | Matches within FIF storage precision | 0.000694 µV | 383, identical times |
| p3 / block03 | Exact array equality | Matches within FIF storage precision | 0.000464 µV | 382, identical times |

Direct comparisons include every envelope epoch, time grid, trial onset,
high-trial verdict, rest false-positive rate and movement/rest ratio. The
original application was executed in a separate process for these comparisons.
The correction comparison covers all F1/F2 samples before the retained EMG
filtering: each corrected recording has **85,644 samples at 250 Hz**. The new
software independently redetected ECG timing and reproduced the report's saved
detections, including duplicate suppression in block03.

Both processing modes produced one right-hand result with 20 scheduled trials.
The existing classifier flags these recordings as unreliable because it fires
on too many rest windows; successful numerical verification does not validate
the physiological interpretation of that count. The report's EMG correction
limitations remain applicable.

The Windows Qt desktop was also exercised with hidden windows:

- Processed p1 through the background worker and displayed its single EMG card.
- Opened the trial browser with all 20 trials.
- Saved the mean plot, 20 trial plots, trial metrics, summary CSV and JSON.
- Captured and inspected the Start, Analysis and Trials views.

See [machine-readable comparisons](../output/verification/reviewed_recordings.json),
[Start screen](../output/verification/start.png),
[Analysis screen](../output/verification/analysis.png), and
[Trial browser](../output/verification/trials.png).

The new copy stores outputs separately under `output/in1948/<filename>/<mode>/`.
Verification read the source recordings and original software; it did not
modify them. The installed `.venv` contains the pinned MNE and SciPy versions
and uses the existing Python 3.12 installation's other installed packages.
