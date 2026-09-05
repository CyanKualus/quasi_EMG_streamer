**MNE artifact cleaning for an online EMG display — 5 September 2026**

The experiment compares the existing application with bounded replay using the
same pinned MNE version, `1.13.0.dev334+g64473254e`. The production application,
its settings and the participant recordings were not changed by that experiment.
The subsequent [Online Monitoring implementation](online_monitoring.md) uses
the AAS-only configuration selected in the follow-up comparison below.

The follow-up [test with cardiac OBS omitted](no_cardiac_comparison.md) directly
evaluates whether cardiac correction benefits this EMG display, including fresh
two-channel replay, movement contrast and known-burst controls.

**Result: the gradient step transfers well; the full AAS + OBS pipeline does
not become equivalent just by running it in short windows.**

For an initial online **EMG envelope display**, eight past scanner cycles
(20 seconds of template history) are a reasonable starting point from these
recordings. Compare this AAS-only output with the existing pipeline’s AAS stage:
the bipolar envelope correlations were **0.9982, 0.9989 and 0.9985**, with
normalized RMS errors of **6.0%, 4.7% and 5.5%**. The corresponding 20–95 Hz
waveform correlations were **0.9385, 0.9417 and 0.9414**. These values belong to
the tested 20-second downstream buffer with 2.5–5 seconds of output age.
They do **not** compare AAS-only output with the full offline cardiac correction.

Treat cardiac OBS as a separate experimental option during integration. The
tested complete pipelines preserved many large bursts, but their waveforms
and envelope amplitudes were not interchangeable with the current full-recording
result. Longer buffers did not reliably improve every recording. No configuration
here validates artifact-free or physiologically faithful EMG.

**Gradient function comparison**

The lower-level class and existing wrapper matched to floating-point precision
when both used four preceding and four following cycles. Maximum bipolar
sample difference was **8.7 × 10⁻¹² µV** on the common interior interval across
all three recordings. This direct function-equivalence check also compared
F1, F2 and ECG independently with strict numerical assertions.

| Gradient neighbors | Template history/span | Native bipolar correlation with existing AAS, across p1–p3 | Sample age at cycle release |
|---|---:|---:|---:|
| 1 past | 2.5 s | 0.731–0.746 | 0–2.5 s |
| 2 past | 5 s | 0.855–0.865 | 0–2.5 s |
| 4 past | 10 s | 0.942–0.950 | 0–2.5 s |
| 8 past | 20 s | 0.933–0.945 | 0–2.5 s |
| 12 past | 30 s | 0.926–0.941 | 0–2.5 s |
| 1 past + 1 future | 5 s | 0.855–0.865 | 2.5–5 s |
| 2 past + 2 future | 10 s | 0.945–0.948 | 5–7.5 s |
| 4 past + 4 future | 20 s | 1.000 | 10–12.5 s |

These native-rate correlations differ from the band-limited display measurements
above. The target cycle is additional to the template: eight past cycles need
22.5 seconds in the native AAS buffer. Even past-only use of this MNE function
waits for a complete target cycle because it linearly detrends that cycle.

Ten seconds of past history had slightly closer native waveforms than twenty,
but twenty gave closer AAS envelope amplitudes and fewer ECG plausibility
failures in the full pipeline. With four past cycles and the same 20-second
downstream buffer, AAS-only envelope correlations were 0.9937–0.9951 and
normalized envelope errors were 10.0–11.2%.

**Full cleaning with different buffer lengths**

The following are **medians across the three recordings**, using eight past
gradient cycles and a fixed 2.5-second downstream lag. Each value compares the
emitted pipeline, including any explicitly logged AAS-only fallback, with the
current **full AAS + OBS** output.

| Downstream buffer | 20–95 Hz waveform r | Waveform normalized RMS error | TKEO envelope r | Envelope normalized RMS error | Time from accepted OBS windows |
|---|---:|---:|---:|---:|---:|
| 5 s | 0.838 | 55.5% | 0.989 | 15.1% | **0%** |
| 7.5 s | 0.490 | 96.7% | 0.938 | 41.7% | 100% |
| 10 s | 0.552 | 91.3% | 0.948 | 36.0% | 100% |
| 15 s | 0.573 | 91.1% | 0.938 | 42.4% | 100% |
| 20 s | 0.609 | 87.7% | 0.930 | 42.1% | 100% |
| 30 s | 0.621 | 86.1% | 0.940 | 41.1% | 100% |
| 60 s | 0.684 | 78.8% | 0.949 | 40.6% | 100% |

The 5-second case is predominantly **AAS only**, since the cardiac step requires
six usable R peaks after edge and plausibility exclusions. Only 0–1.14% of the
common interval came from accepted OBS calls, depending on recording. Its high
envelope correlation must not be interpreted as successful full cleaning.
For the other past-eight configurations, OBS acceptance over the common interval
was 96.59–100%; a median of 100% can hide failures in one recording.

Across recordings, the 20-second configuration had waveform r **0.491–0.610**,
envelope r **0.897–0.939** and envelope normalized error **41.6–53.9%**. At
60 seconds these ranges were **0.495–0.708**, **0.884–0.956** and **34.9–59.7%**.
The 30-second buffer was especially poor for p3’s envelope (r = 0.804,
normalized error = 112.6%). Thus a single pooled correlation would obscure
material recording-to-recording variation.

Retaining the exact four-past/four-future gradient settings did not solve the
cardiac discrepancy. With a 60-second downstream buffer, waveform correlations
were **0.767, 0.753 and 0.620**, with **69.7%, 69.4% and 83.0%** normalized errors.
Its envelope correlations were 0.955–0.962, but sample age increased to 12.5–15 s.
The AAS-only output of that symmetric replay was essentially identical in the
20–95 Hz band; the major remaining difference arose downstream of AAS.

![Window comparison](../output/online_mne_comparison/window_comparison.png)

**Artifact measures and diagnostic controls**

For eight-past AAS and a 20-second downstream buffer, the full emitted output
gave the following results on the matched evaluation interval:

| Recording | Offline rest RMS, µV | Replay rest RMS, µV | Offline scanner-peak reduction vs input | Replay scanner-peak reduction vs input |
|---|---:|---:|---:|---:|
| p1 | 154.60 | 158.92 | 99.4336% | 99.3189% |
| p2 | 146.60 | 132.26 | 99.7712% | 99.6016% |
| p3 | 65.93 | 75.82 | 99.9678% | 99.6704% |

Scanner-frequency suppression remained substantial, but replay residual excess
power was about **1.2, 1.7 and 10.2 times** the offline residual, respectively.
Near-100% reductions can hide meaningful residual differences. Similar total
rest RMS also does not imply equal waveforms. Heartbeat-average RMS was
11.32/11.50/10.58 µV in replay versus 10.64/11.88/11.10 µV offline. These averages
include noncardiac signal and depend on the chosen band and epochs.

ECG detections in accepted 20-second windows had precision 92.5–98.4% and recall
93.6–98.8% against the offline list, using a ±40 ms match tolerance. This is
agreement with the application’s detector, not validation against independent
ECG ground truth.

The **oracle OBS control** held full-recording AAS, conditioning and R detections
fixed, and varied only the data supplied to the OBS fit. With 20-second windows,
waveform correlations were still **0.663, 0.660 and 0.581**; with 60-second windows,
they were **0.758, 0.743 and 0.627**. Local OBS fitting alone therefore explains
a substantial discrepancy; changing ECG detection is not the sole cause.
This diagnostic trims peaks whose inclusive fitting endpoint would fall outside
the buffer, to avoid the independently reproduced MNE boundary error.

A synthetic isolated 63 Hz burst confirmed the known AAS echo mechanism. With
eight past cycles, its detrended target-cycle projection gain was 1, but copies
with gain **−0.125** appeared in each of the following eight cycles (2.5–20 s
later). Four-past/four-future AAS instead produced −0.125 copies in four earlier
and four later cycles. Past-only AAS avoids anticipatory copies but retains
subsequent copies. This linear gradient-stage control does not test cardiac OBS
or establish preservation of real muscle activity.

**Practical integration findings**

- Use `GradientRemover` with an explicit scanner-cycle buffer and an explicit
  output timestamp. The tested past-eight, 20-second downstream configuration
  is a useful starting point for an AAS envelope display. Allow 40 seconds for
  startup and 2.5–5 seconds of steady-state sample age, plus processing time.
  A faster design that replaces the FIR conditioning with streaming IIR filters
  would need its own comparison; it is not represented by these numbers.
- Keep the existing causal EMG filter states between chunks. Processing in
  2-, 5- and 10-second chunks with saved states matched the application’s
  whole-recording filters **exactly** on all three recordings. Resetting states
  introduced normalized waveform errors of 9.7–12.0%, 5.6–8.5% and 3.0–3.8%,
  respectively. Preserve the TKEO neighbor and overlapping envelope history too.
- Handle failed cardiac updates explicitly. Besides ECG checks, the pinned MNE
  OBS code can raise a one-sample epoch-length error at the buffer boundary.
  [`reproduce_mne_obs_boundary.py`](../diagnostics/reproduce_mne_obs_boundary.py)
  reproduces it with synthetic data; it intentionally exits with the exception.
  When `last_peak + half_RR == n_times`, MNE’s inclusive slice needs one missing
  sample. This experiment leaves the pinned library unchanged and records
  failures instead of silently treating them as full cleaning.
- A zero-extra-lag control emitted samples aged 0–2.5 seconds. Its median
  waveform/envelope correlations were 0.625/0.938, versus 0.609/0.930 with the
  2.5-second lag and 0.618/0.933 with a 5-second lag. It did not restore
  equivalence. Its latest samples are at the FIR/OBS buffer edge, where padding
  and incomplete heartbeat fits can influence the apparent agreement.
- Computation was comfortably faster than the 2.5-second update cadence:
  20-second downstream buffers took about **73–85 ms at the 95th percentile**
  for the main past-eight configuration; 60-second buffers took 157–170 ms.
  The largest downstream update across all 5,817 calls was **246 ms**. Native
  AAS’s worst configuration-level 95th percentile was **20.5 ms**. No downstream
  update exceeded its 2.5-second deadline. This does not measure end-to-end GUI
  latency, acquisition jitter or sustained operation on another computer.

All **61 tests passed**, including new checks for delayed AAS equivalence,
future-data independence, nonoverlapping output commitment and filter-state
continuity. All 48 variants covered the entire common evaluation interval.
Source-file sizes and modification times were unchanged after the experiment.
Evidence is limited to three recordings from one participant with the existing
stable computational scanner grid. Real scanner triggers, drift, dropped data,
other participants and preservation of weak EMG still need evaluation before
using this as a dependable online measurement.

**Functions and what “online” means here**

The MRI option in [`emgcasting/mri.py`](../emgcasting/mri.py) uses:

| Stage | Existing application | Replay experiment |
|---|---|---|
| Scanner gradient artifact | `remove_fmri_gradient_artifact`, plus `GradientRemover` for recording edges | `GradientRemover.get_tr_corrected` on a bounded array of completed scanner cycles |
| Downsampling | `mne.filter.resample(method="polyphase")`, 5000 to 250 Hz | Same function on the trailing buffer |
| EMG conditioning | MNE zero-phase 0.5–100 Hz FIR and 50 Hz FIR notch | Same filters inside each trailing buffer |
| ECG detection | `find_ecg_events`, 10–25 Hz, automatic threshold; application duplicate and plausibility checks | The same application `detect_qrs` function, independently in every trailing buffer |
| Cardiac artifact | `apply_pca_obs`, four components, R + 212 ms, each monopole separately | Same batch function refitted inside each buffer, followed by F1−F2 |
| Display signal | Existing causal SciPy notch and 20–95 Hz band-pass; TKEO, 100 ms / 20 ms step | Same filters with continuous state across emitted chunks and the same envelope definition |

MNE explicitly describes [`GradientRemover`](https://mne.tools/dev/generated/mne.preprocessing.GradientRemover.html)
as a building block for real-time correction. The
[`Raw` wrapper](https://mne.tools/dev/generated/mne.preprocessing.remove_fmri_gradient_artifact.html)
uses that same class internally. Switching APIs alone therefore does not change
the AAS algorithm. The template’s past/future neighbors determine its data needs.

[`apply_pca_obs`](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html)
has no separate incremental fit/apply interface in the installed version. It
estimates the mean heartbeat template, PCA basis and median RR from its supplied
array. Here, “online OBS” means repeated bounded calls to this batch function.
[MNE-LSL](https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.stream.StreamLSL.html)
provides stream acquisition and causal IIR filters; those filters alone do not
replace scanner AAS or cardiac OBS. LSL acquisition, network timing and GUI
rendering were outside this numerical experiment.

**Replay protocol**

Inputs were the three original `in1948_block01`–`block03` BrainVision recordings
in `D:\ExpData\MEG\Quasi fMRI\data\Other\fMRI_test\Unfitered`: 342.576 seconds
each, F1/F2/ECG at 5000 Hz. Each offline reference was recomputed by the actual
`correct_raw` entry point. Video schedules were 3, 2 and 1, aligned to each file’s
earliest S3 marker.

- Use the same exact 2.5-second computational scanner grid as the application.
  Native gradient tests use past-only templates of 1, 2, 4, 8 or 12 cycles and
  symmetric templates of 1+1, 2+2 or 4+4 cycles.
- The main downstream sweep uses eight past cycles (20 seconds of template
  history) and buffers of 5, 7.5, 10, 15, 20, 30 and 60 seconds. Controls retain
  four-past/four-future AAS, change the past history, or change output lag.
  There are 16 end-to-end configurations per recording, 48 in total.
- Every update releases a new 2.5-second chunk, once. Each MNE call sees only
  data available by its simulated acquisition time. There is no full-recording
  ECG list, future-data access, output revision or retrospective final flush in
  an online candidate. Gradient startup and incomplete tail samples are withheld.
- Unless stated otherwise, emit the chunk ending 2.5 seconds before the latest
  available gradient-corrected sample. With past-only AAS, its samples are
  2.5–5 seconds old at release; four future cycles add another 10 seconds.
  Buffer length is history length, not steady-state delay. Compute time is
  additional. TKEO needs one subsequent sample, and the 100 ms envelope needs
  its complete window; a display must account for this small additional delay.
- Require a full buffer before starting. For eight-past AAS with a 20-second
  downstream buffer, first output arrives at acquisition time 40 seconds.
  A 60-second downstream buffer raises startup to 80 seconds.
- Failed ECG/OBS updates explicitly fall back to AAS plus conditioning and are
  logged. `common_obs_applied_pct` in the JSON measures time emitted from
  accepted OBS calls, not the fraction of samples on which an artifact was
  actually subtracted. OBS can leave buffer-edge samples outside its fits.

All comparisons use the **same 100–320 second interval** (220 seconds per
recording), after every configuration’s startup and before its unavailable end.
No lag optimization or waveform time-shifting is used. F1, F2 and F1−F2 metrics
are saved; the tables focus on the bipolar signal used by the application.
Normalized RMS error is `RMS(replay − offline) / RMS(offline)`; correlation alone
does not establish equal amplitudes or physiological preservation.

Rest measurements use the scheduled rest periods with the application’s 1-second
start and 2-second end trims, intersected with that common interval. Scanner
excess is power within ±0.5 Hz of 18.4 Hz and its harmonics below 100 Hz, above
the median local 0.75–2 Hz spectral flank, using separate 4-second Welch segments
inside contiguous rest spans. The denominator is equally conditioned input on
the same spans. This is a narrow spectral measurement, not total artifact power.
Heartbeat averages use the offline R times **only for evaluation**, with ±350 ms
epochs centered at R + 212 ms after 20–95 Hz EMG filtering.

**Reproduction and output files**

Run from the application folder:

```powershell
& .\.venv\Scripts\python.exe diagnostics/compare_online_mne.py
& .\.venv\Scripts\python.exe diagnostics/summarize_online_mne.py
& .\.venv\Scripts\python.exe diagnostics/check_windowed_obs.py
& .\.venv\Scripts\python.exe -m pytest -q
```

`--data-dir` and `--output` change the input/output paths. `--runs 1` selects one
recording; `--quick` selects three configurations for a smoke run. The separate
`check_windowed_obs.py` experiment intentionally uses offline AAS and the offline
R-peak list to isolate OBS fitting-window effects. Its results are labeled an
**oracle diagnostic**, and are not online performance claims.

- [Detailed results and environment provenance](../output/online_mne_comparison/results.json)
- [All 48 bipolar comparisons](../output/online_mne_comparison/summary.csv)
- [Across-recording aggregates](../output/online_mne_comparison/aggregated.csv)
- [Window comparison figure](../output/online_mne_comparison/window_comparison.png)
  and [PDF](../output/online_mne_comparison/window_comparison.pdf)
- Example envelopes: [p1](../output/online_mne_comparison/p1_envelopes.png),
  [p2](../output/online_mne_comparison/p2_envelopes.png),
  [p3](../output/online_mne_comparison/p3_envelopes.png)
- [Synthetic gradient burst control](../output/online_mne_comparison/synthetic_gradient_burst.json)
  and [oracle OBS control](../output/online_mne_comparison/oracle_obs_control.json)

The output folder also contains a CSV for every replay’s updates and compressed
250-Hz traces for independent inspection. The JSON records source file identity,
MNE/NumPy/SciPy/Python versions and the benchmark script’s SHA-256. Timing is for
this machine with one numerical-library thread, includes downstream resampling,
filtering, ECG and OBS per update, and excludes file I/O, transport and plotting.
Native AAS compute time is recorded separately.
