# quasi_EMG_streamer

Standalone EMG copy of NeuroCasting. It processes **one filename at a time**
and retains the existing mean-envelope plots, high-EMG trial screening, trial
browser, figures and CSV output. The EEG processing package, component fitting,
time-frequency analysis and EEG score tabs are absent. A separate **Online
Monitoring** tab receives BrainAmp MR data through BrainVision Recorder RDA
and displays the TKEO envelope of AAS-corrected EMG in the El Artem plot style.

## Online BrainAmp MR monitoring

Open **Online Monitoring**, enter the Recorder computer's IP address (or keep
`127.0.0.1` on the same computer), leave port **51244**, set the hand channel
pairs, and press **Connect**. In BrainVision Recorder, enable **Remote Data
Access** and start monitoring. The default right-hand pair is **F1, F2**;
the left hand is disabled until its two channel names are entered.

The default pipeline uses **AAS only, with no cardiac processing or ECG
requirement**: 20 seconds of past template history, a 10-second filter buffer,
first corrected display after approximately **30 seconds**, then an update
every **2.5 seconds** with **2.5–5 seconds of delay plus computation**.
Both graphs display TKEO. Each has independent time-window, integer Ymin/Ymax,
signed magnitude N (`10^N`), threshold and freeze controls on its left. Scaling
is manual and updates immediately, including while frozen. Stream options fold
away to the right; Connect, Disconnect and the progress bar stay below the
bottom graph's options on the left.

Selecting Online Monitoring opens the window full screen. Press Esc or switch
tabs to restore the previous window size.

See [the connection and monitoring guide](docs/online_monitoring.md) for
Recorder setup, timing assumptions, stream interruption behavior and validation.

## Open and process a file

With Python 3.12 installed, run **`setup_environment.bat`** once to create the
local environment and install dependencies. Then double-click
**`launch_quasi_EMG_processing.bat`** to open the application.

1. Browse to one **`.vhdr`** BrainVision header, keeping its referenced
   `.eeg` and `.vmrk` companions in place. Ordinary XDF EMG input is also supported.
2. Leave the left-hand pair blank and the right-hand pair as **F1, F2** for the
   reviewed recordings. Change these fields for other electrode arrangements.
3. Choose the video schedule, or leave **Automatic** selected. Automatic reads
   the recording's `pN` or `blockNN` ordinal and `cond seq.txt` beside the file
   or in its parent folder. For an arbitrary filename, select Video 1, 2 or 3.
   The three original stimulus CSVs are bundled in `schedules/`.
4. Set **Remove MRI artifacts using MNE (AAS + cardiac OBS)** as appropriate:

   | Checkbox | Processing |
   |---|---|
   | Clear (default) | Existing bipolar EMG pipeline, with no added artifact correction; use for already corrected BrainVision files. |
   | Ticked | Full MNE AAS + cardiac OBS from the reviewed report, followed by the same EMG pipeline; use for uncorrected BrainVision files. An ECG channel is required. |

5. Press **Process file**. Review **EMG Analysis**, open the hand's trial
   browser, then use **Save figures + CSV** to save the displayed result.

For a recording that is still being acquired, tick **Wait for this recording
to stop before processing**, then press **Process file**. This handles the
selected recording once: it waits for the header and its companion files to
exist, close, and remain unchanged for two seconds. On Windows, it checks that
no process still has the files open. This checks file readiness; it does not
connect to the recorder's Stop button. The wait times out after one hour.
Closing the application while waiting requests cancellation; close again when
the worker has stopped. During analysis or saving, let the operation finish.

## Timing and correction

BrainVision uses the earliest **S3** marker and the selected video's schedule.
Marker times are sample locked; there is no XDF marker-time shift. The recording
must contain the scheduled events through their ends. XDF retains the original
embedded-trial marker and automatic clock-lag handling.

The MNE option follows the full pipeline selected by the user from
`EEG_EMG_MNE_vs_BrainVision_results.md`:

- Correct each EMG monopole and ECG at the native sample rate. Use a stable
  **2.5 s** computational cycle grid starting at sample zero, four preceding
  and four following cycles, excluding the target. Use eight real neighbors
  on one side for edge cycles. At least nine complete cycles are required.
- Resample to **250 Hz**, apply **0.5–100 Hz** zero-phase FIR filtering and a
  **50 Hz** notch of 1 Hz width to EMG.
- Detect ECG R peaks afresh from this file using **10–25 Hz**, automatic
  threshold and duplicate suppression within 120 ms. No prior EEG run or
  saved heartbeat list is required.
- Apply independent monopolar PCA-OBS fits with **four components** at
  **R + 212 ms**, then form the requested bipolar difference.
- Apply the original EMG pipeline: causal 50 Hz notch, causal 20–95 Hz
  Butterworth band-pass, TKEO envelope, 100 ms window / 20 ms step and the
  original rest/preparation-based trial classifier.

MRI correction uses pinned MNE **1.13.0.dev334+g64473254e**, commit
`64473254ed0c2c64627a5864a666686a43ef8be8`, and the scientific package versions
used in the report. Other MNE versions are rejected for correction. Advanced
parameters, including ECG channel and scanner TR, live in
`quasi_emg_settings.json`. GUI preferences use a separate application profile.

The report identified inverse burst copies at neighboring scanner cycles and
mixed effects of cardiac correction on EMG. This remains an experimental
correction, and the resulting bandwidth is limited to 100 Hz. The periodic grid
assumes a stable scanner cycle; it does not recover physical scanner triggers.
An incomplete final cycle is not corrected. The incomplete tail and first/last
five seconds are excluded from envelope-based scoring, including overlapping
envelope windows. These exclusions are recorded in `processing.json`.

## Outputs and recording-stop integration

Saved outputs go into this application's folder:

```text
output/<participant>/<filename>/direct/
output/<participant>/<filename>/mne_aas_obs/
```

Each contains `emg_summary.csv`, mean figures, requested trial figures and
metrics, and `processing.json` with parameters, source file identity, timing,
correction details, ECG detections and exclusions. Different files and modes
have separate folders. Repeating the same file and mode replaces its previous
saved output. Source recordings are opened for reading only. Results marked
**UNRELIABLE** retain the original rest false-positive qualification.

The CLI processes and saves a single file immediately, so a recorder can call
it after closing its output files:

```powershell
& .\.venv\Scripts\python.exe run_emg_cli.py "D:\recordings\run.vhdr" --video 3
& .\.venv\Scripts\python.exe run_emg_cli.py "D:\recordings\raw.vhdr" --video 3 --mne-mri
```

Add `--wait-for-stop` if the file may still be open. Add `--no-trial-figures`
to save a mean plot and summary without the per-trial output folder.
Run `--help` for the electrode, ECG, TR, participant and output-folder options.
The GUI can also open and process a supplied file:

```powershell
& .\.venv\Scripts\python.exe app.py "D:\recordings\raw.vhdr" --video 3 --mne-mri --process
```

## Verification

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pytest -q
```

The tests cover the retained EMG detector, voltage scaling, single-file input,
video lookup, the correction switch, periodic artifact subtraction, file
readiness, isolated outputs and the desktop controls. Reviewed-recording
checks and numerical comparisons are recorded in `docs/verification.md`.

Original EMG algorithm and interface copied from NeuroCasting_QuasifMRI.
MRI settings are from the 5 September 2026 in1948 report, preserved in
`docs/EEG_EMG_MNE_vs_BrainVision_results.md` for provenance.

## Online artifact-cleaning comparison

The three reviewed recordings were replayed using MNE `GradientRemover` and
5–60-second downstream buffers. Gradient correction can reproduce the offline
step exactly with delayed symmetric templates, and past-only templates preserve
the AAS EMG envelope closely. Windowed cardiac PCA-OBS differs substantially
from the full-recording result. See [the comparison report](docs/online_mne_comparison.md)
for measurements, figures, delays, a reproduced MNE boundary error and commands
to repeat the experiment. This adds diagnostics; the application's processing
and GUI behavior are unchanged.

A follow-up [comparison with cardiac OBS omitted](docs/no_cardiac_comparison.md)
tests all three recordings with no ECG dependency in the replay, including
movement contrast, independent heartbeat timing, short buffers and known EMG
burst probes. For the tested envelope display, the results support keeping AAS
and making cardiac OBS optional.
