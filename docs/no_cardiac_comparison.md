**Testing the EMG pipeline without cardiac artifact removal — 5 September 2026**

**For the intended F1−F2 EMG envelope display, these recordings support leaving
cardiac PCA-OBS off by default.** MRI gradient subtraction (AAS) is still needed.
Removing OBS retained the main activity pattern, lowered resting amplitude,
improved the measured movement/preparation contrast, and allowed consistent
processing with short buffers and no ECG channel. This is a finding about this
20–95 Hz EMG pipeline and these three recordings, not a conclusion that cardiac
artifacts never matter.

OBS did reduce the whole-recording heartbeat-aligned average, particularly in
p3. It also preserved the injected test bursts well. The evidence therefore
supports making it optional for this display, rather than declaring OBS harmful
or the remaining signal artifact-free. The existing trial classifier remained
unreliable with and without OBS.

**Matched offline comparison**

Both versions retain the existing native-rate AAS, polyphase downsampling to
250 Hz, 0.5–100 Hz FIR and 50 Hz notch, independent F1/F2 monopoles followed by
their difference, causal EMG notch and 20–95 Hz band-pass, and 100 ms / 20 ms
TKEO envelope. The only omitted correction is cardiac PCA-OBS. The stored AAS
and OBS stages from the verified previous experiment were reused; an independent
OBS reapplication reproduced the stored full output to numerical precision.

Each comparison includes 332.576 seconds after removing the first and last five
seconds, all 99 seconds of trimmed scheduled rest, and 20 right-hand trials.

| Recording | Rest RMS with OBS, µV | Without OBS, µV | Rest RMS change when omitting OBS | Movement/preparation ratio with OBS | Without OBS |
|---|---:|---:|---:|---:|---:|
| p1 | 182.07 | 170.44 | −6.39% | 1.00 | 1.05 |
| p2 | 216.01 | 202.19 | −6.40% | 1.89 | 2.26 |
| p3 | 119.53 | 110.50 | −7.55% | 2.13 | 2.76 |

Movement/preparation values are medians of trial RMS ratios: seconds 2.2–3.8
versus 0.2–1.8 after the four-second trial onset. Larger ratios mean stronger
measured contrast, but cannot distinguish genuine EMG from all motion or
correction effects. p1 still has weak median contrast in either version.

Typical rest amplitude was also lower without OBS. Median 100 ms RMS envelopes
were **27.26, 28.00 and 32.38 µV**, versus **34.46, 37.68 and 41.34 µV** with OBS.
This is roughly a 21–26% reduction in the median envelope. AAS alone retained
**99.53%, 99.68% and 99.71% suppression of scanner-harmonic excess power** relative
to equally conditioned input. These are selected spectral peaks, not total
artifact power, and lower amplitude alone does not prove cleaner physiology.

These full-interval values differ from the prior online report’s 100–320-second
comparison and the earlier report’s separate filtering/measurement setup. The
comparisons in this report always use matched samples and the application’s
actual causal EMG filters.

**What remains of the heartbeat artifact?**

Evaluation used the supplied BrainVision `Pulse Artifact/R` markers, translated
to raw-recording time by the difference between the two S3 markers (1.4, 1.8 and
1.4 ms). Thus all versions were evaluated against the same timing list, rather
than each version’s own newly detected peaks. MNE R detections were also evaluated
as a secondary check and are saved in the JSON. Neither list is physiological
ground truth.

The table is the RMS of the demeaned average of ±350 ms waveform epochs centered
at R + 212 ms, **after the 20–95 Hz EMG filter**:

| Recording | Whole-recording heartbeat average without OBS, µV | With OBS, µV | Reduction from OBS | Rest-only average without OBS, µV | With OBS, µV |
|---|---:|---:|---:|---:|---:|
| p1 | 9.57 | 9.04 | 5.6% | 14.62 | 16.01 |
| p2 | 13.12 | 11.93 | 9.1% | 25.11 | 25.87 |
| p3 | 12.54 | 8.97 | 28.5% | 8.76 | 9.21 |

OBS has a measurable effect on the whole-recording average, especially p3,
but it did not reduce these rest-only averages. Averaging a finite number of
noisy EMG epochs leaves noncardiac variation, so these values should not be
read as isolated cardiac amplitudes.

A descriptive control relocated the same number of heartbeat epochs to random
times within each epoch’s own trimmed rest span, at least 200 ms from its original
center, for 200 repetitions. Without OBS, observed rest-average RMS was
**0.92, 1.20 and 0.79 times** the shuffled median; all were below the shuffled
95th percentile. The same conclusion held with OBS. This control did not show a
large, robust extra heartbeat-aligned rest average in the display band. It is
not a formal absence-of-artifact test; overlapping epochs, large intermittent
events and the limited rest duration constrain interpretation. Odd/even-beat
template consistency is also saved for inspection.

For context, MNE’s [PCA-OBS example](https://mne.tools/stable/auto_examples/preprocessing/esg_rm_heart_artefact_pcaobs.html)
describes its origins in simultaneous EEG-fMRI and an adaptation for EEG/ESG.
That does not establish a requirement to apply the same settings to this bipolar
EMG display. The installed [OBS function](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html)
fits separately to each picked channel; in this application those independent
monopolar fits are subtracted before forming F1−F2.

**Fresh online replay with OBS completely omitted**

All 16 previous configurations were rerun for each recording: **48 no-cardiac
replays and 5,817 updates**. The online processing path accepts **only F1 and F2**.
It does not read ECG data, detect heartbeats, fit OBS or fall back from failed
cardiac fits. The saved ECG information is used only by the evaluation and
separate injection control.

The primary sweep uses eight preceding 2.5-second scanner cycles (20 seconds of
template history), followed by the listed downstream buffer. It emits a new
2.5-second chunk with sample ages of 2.5–5 seconds. Every candidate was compared
on the same 100–320-second interval, with no time-shifting or retrospective
revision. All candidates covered that interval completely and their no-OBS
rest measurements reproduced the earlier experiment’s pre-OBS stage.

| Downstream buffer | Envelope correlation with offline AAS, range across p1–p3 | Normalized envelope RMS error | Downstream compute, 95th percentile |
|---|---:|---:|---:|
| 5 s | 0.99814–0.99886 | 4.75–6.08% | 11.4–12.2 ms |
| 7.5 s | 0.99824–0.99890 | 4.67–5.97% | 12.5–13.6 ms |
| 10 s | 0.99824–0.99890 | 4.67–5.97% | 12.6–15.8 ms |
| 15 s | 0.99824–0.99890 | 4.67–5.97% | 13.7–14.5 ms |
| 20 s | 0.99824–0.99890 | 4.67–5.97% | 14.9–16.5 ms |
| 30 s | 0.99824–0.99890 | 4.67–5.97% | 16.7–20.4 ms |
| 60 s | 0.99824–0.99890 | 4.67–5.97% | 23.3–25.2 ms |

These correlations compare **online AAS only with offline AAS only**. They do
not claim equivalence to the full OBS output. The five-second buffer now works
because the six-heartbeat requirement is absent. From about 7.5–10 seconds,
increasing the buffer made negligible difference to the measured EMG envelope.

For an initial implementation, use **20 seconds of past scanner history and a
10-second downstream buffer**, keeping the current filter and envelope settings.
This tested choice starts producing corrected output at acquisition time
**30 seconds**, then releases samples **2.5–5 seconds old**, plus computation.
Its two-channel native AAS cost was 8.3–11.1 ms at the 95th percentile, additional
to the 12.6–15.8 ms downstream figure. A five-second buffer starts at 25 seconds.
Buffer history is not the same as steady-state delay. The largest downstream
update across the whole experiment was 127 ms; none exceeded the 2.5-second
cadence. Transport and GUI rendering were not benchmarked.

The offline GUI’s MRI checkbox controls AAS and OBS together. Clearing that
checkbox does not select the AAS-only pipeline tested here. The subsequently
implemented [Online Monitoring tab](online_monitoring.md) now uses this
AAS-only pipeline, with no cardiac processing.

The earlier windowed OBS output was also compared with each matched no-OBS
replay. Effects were not uniformly favorable to either version. For example,
with 10-second downstream buffers, no-OBS rest RMS was **146.77, 122.94 and
67.00 µV**, versus **159.68, 117.87 and 70.00 µV** with windowed OBS; OBS lowered
rest RMS in p2. The corresponding movement/preparation ratios were
**1.13, 1.65 and 3.49** without OBS, versus **1.04, 1.45 and 1.89** with it. These
online trial counts and intervals are smaller than the full-recording table,
because all configurations share the same post-startup comparison interval.
Detailed results include the earlier OBS acceptance/fallback rate for each pair.

![Envelopes with and without cardiac OBS](../output/no_cardiac_comparison/no_cardiac_envelopes.png)

The fixed rest excerpt also shows an important limitation: high full-interval
correlation can coexist with visible small transient differences, particularly
for p3’s online AAS. Changing gradient templates can create delayed burst copies,
as demonstrated in the [previous comparison](online_mne_comparison.md). Omitting
cardiac OBS does not remove that AAS behavior.

**Known-burst control and classifier behavior**

Four independent 600 ms, 25–85 Hz bursts were injected one at a time per recording,
each at 10 and 100 µV bipolar RMS: **24 probes** in total. Two positions were near
an OBS heartbeat center and two between centers, all during scheduled rest.
The increments were added with opposite half-amplitudes to F1/F2 after AAS and
conditioning, so this tests the cardiac step alone. ECG timing remained fixed,
and each OBS basis was refitted after injection. The recovered increment was
measured after the actual causal EMG filters.

Full-recording OBS retained projection gains of **0.99649–1.00044** and waveform
correlations of **0.99980–0.99999** for these probes. Without OBS, the cardiac-stage
increment is unchanged by definition. Thus the probes **do not support saying
that full-recording OBS necessarily suppresses weak EMG**. They also do not
validate every kind of contraction or the short-window OBS fits from the previous
experiment; those are different fitting conditions.

The actual trial detector marked all 20 right-hand trials high in both offline
versions. Its own matched-rest false-positive rates were:

| Recording | Without OBS | With OBS |
|---|---:|---:|
| p1 | 95.6% | 90.4% |
| p2 | 88.9% | 86.7% |
| p3 | 97.0% | 94.8% |

All exceed the application’s 5% reliability threshold. OBS slightly reduced
these rates but did not make the classifier reliable. The recommendation to
omit OBS concerns an exploratory envelope display, not dependable automatic
detection of very weak contractions. The unusually high rest rates and AAS
burst copies remain unresolved.

**Reproduction and artifacts**

```powershell
& .\.venv\Scripts\python.exe diagnostics/compare_without_cardiac.py
& .\.venv\Scripts\python.exe diagnostics/summarize_without_cardiac.py
& .\.venv\Scripts\python.exe -m pytest -q
```

The default inputs are the preceding experiment in `output/online_mne_comparison`
and its three original BrainVision recording paths. `--previous`, `--output`
and `--runs` are available on the comparison script. MNE remains pinned to
`1.13.0.dev334+g64473254e`. Rest was trimmed by one second at its start and two
seconds at its end, and all displayed EMG measurements use the existing causal
20–95 Hz filtering. ECG epochs are independently selected identically across
each comparison; the JSON retains F1 and F2 results as well as F1−F2.

- [Full offline comparison CSV](../output/no_cardiac_comparison/offline_summary.csv)
- [All 48 online comparisons](../output/no_cardiac_comparison/online_summary.csv)
- [Window comparison CSV](../output/no_cardiac_comparison/window_summary.csv)
- [Detailed metrics, per-trial values, controls and provenance](../output/no_cardiac_comparison/results.json)
- [Heartbeat and resting-amplitude figure](../output/no_cardiac_comparison/cardiac_ablation.png)
  / [PDF](../output/no_cardiac_comparison/cardiac_ablation.pdf)
- [Envelope figure](../output/no_cardiac_comparison/no_cardiac_envelopes.png)
  / [PDF](../output/no_cardiac_comparison/no_cardiac_envelopes.pdf)
- [Reproducible benchmark](../diagnostics/compare_without_cardiac.py)

The output folder also contains update timing CSVs and compressed no-OBS traces
for each recording. **All 64 tests passed**, including tests that forbid ECG/OBS
calls in the new replay, verify independence from future samples, check the
trial contrast calculation, and recover a known heartbeat-aligned waveform.
All 48 common comparison intervals were complete. Source recording sizes and
modification times were unchanged. The application’s GUI, defaults and
production correction function were not changed by this experiment.
