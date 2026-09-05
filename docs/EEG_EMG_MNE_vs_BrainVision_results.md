**EEG and EMG: MNE fMRI-artifact correction compared with BrainVision**

Participant: in1948. Recordings: p1–p3, overt movement. Analysis date: 5 September 2026.

MNE substantially reduced the periodic scanner artifact in both EEG and EMG. For EEG, MNE left smaller scanner-frequency peaks than BrainVision, while BrainVision left less heartbeat-locked activity. For EMG, transferring the same EEG settings produced higher rest amplitudes than BrainVision and introduced smaller copies of an isolated test burst at neighboring scanner cycles. These results do not establish artifact-free signals or overall superiority of either method.

**Data and processing**

The inputs were `in1948_block01.vhdr`, `in1948_block02.vhdr` and `in1948_block03.vhdr` in `D:\ExpData\MEG\Quasi fMRI\data\Other\fMRI_test\Unfitered`. They were compared with the corresponding parent-folder files `in1948_MRbvCBbv_p1_EMG.vhdr` through `in1948_MRbvCBbv_p3_EMG.vhdr`. The recording order corresponds to overt-movement videos 3, 2 and 1. Each recording lasts 342.576 seconds. The original files were left unchanged.

EEG analysis used 31 scalp channels, excluding F1/F2 EMG, ECG and C1/C2 auxiliary inputs. ECG was processed separately to detect heartbeats. EMG processing treated F1 and F2 as independent monopolar channels and calculated F1−F2 after correcting them. The EMG run reused the finalized EEG parameters and saved heartbeat detections without further tuning.

The common MNE pipeline was:

1. **Scanner correction:** average artifact subtraction (AAS) at the native 5,000 Hz rate, using four preceding and four following 2.5-second cycles. The target cycle was excluded from its template. At the first/last four complete cycles, eight available neighbors on one side were used. MNE also linearly detrends each complete cycle.
2. **Conditioning:** polyphase resampling to 250 Hz, zero-phase FIR filtering at 0.5–100 Hz, and a 50 Hz notch with 1 Hz width.
3. **Cardiac correction:** PCA optimal basis set subtraction (OBS), using four components plus the mean template, centered 212 ms after the detected ECG R peak. The finalized ECG detector used a 10–25 Hz band, automatic threshold and removal of one duplicate detection in p3.

Processing used MNE `1.13.0.dev334+g64473254e`, pinned commit `64473254ed0c2c64627a5864a666686a43ef8be8`, in the separate `.venv-mne-fmri` environment. The functions were [`remove_fmri_gradient_artifact`](https://mne.tools/dev/generated/mne.preprocessing.remove_fmri_gradient_artifact.html) and [`apply_pca_obs`](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html). The cardiac delay follows the approximately 210 ms delay in the [original FMRIB implementation](https://github.com/sccn/fMRIb/blob/master/fmrib_pas.m), rounded to a 250 Hz sample.

For comparison, matching filters were applied to copies of the input and BrainVision data. Each run contributed 99 seconds of scheduled rest, drawn from nine 14-second rest periods with one second trimmed from the start and two from the end. Signals were aligned using the earliest S3 video marker; raw-minus-reference offsets were 1.4, 1.8 and 1.4 ms. RMS and spectra were calculated on each version's own 250 Hz samples. The first and last five seconds were excluded.

**EEG results**

The following values are the **median across 31 scalp channels of 20–95 Hz rest RMS**, in microvolts. “Input” includes the matched conditioning but no scanner or cardiac correction.

| Recording | Input | MNE AAS only | MNE AAS + OBS | BrainVision |
|---|---:|---:|---:|---:|
| p1 | 204.04 | 5.55 | 5.47 | 4.15 |
| p2 | 204.44 | 7.23 | 7.54 | 5.48 |
| p3 | 202.98 | 6.15 | 6.17 | 4.73 |

The complete MNE pipeline reduced this band-limited amplitude by approximately **96–97%** relative to input. BrainVision retained lower 20–95 Hz rest RMS in all three recordings. Amplitude reduction is not a direct percentage of artifact removed because physiological activity also contributes to this band.

In the broader EEG band, **median 1–40 Hz rest RMS** was similar between methods:

| Recording | MNE AAS + OBS (µV) | BrainVision (µV) |
|---|---:|---:|
| p1 | 20.34 | 19.80 |
| p2 | 28.18 | 29.62 |
| p3 | 27.11 | 30.76 |

MNE suppressed the scanner-frequency peaks more strongly. The table gives **median scalp-channel excess power at 18.4 Hz and its harmonics**, above the local spectral floor, in µV².

| Recording | Input | MNE AAS + OBS | BrainVision |
|---|---:|---:|---:|
| p1 | 54,076 | 0.86 | 5.13 |
| p2 | 49,745 | 3.45 | 7.00 |
| p3 | 50,180 | 1.99 | 6.18 |

MNE therefore left approximately **2–6 times less excess power at these scanner frequencies**. This narrow spectral result can coexist with higher total 20–95 Hz amplitude because other frequencies and residual sources contribute to RMS.

BrainVision suppressed the heartbeat-locked average more strongly. All stages were evaluated using the same supplied BrainVision R markers. Values below are **median scalp-channel RMS of the demeaned 1–40 Hz heartbeat average**, in microvolts.

| Recording | MNE AAS only | MNE AAS + OBS | MNE OBS using supplied BV R markers | BrainVision |
|---|---:|---:|---:|---:|
| p1 | 21.54 | 3.81 | 3.56 | 0.72 |
| p2 | 23.96 | 3.88 | 3.71 | 1.26 |
| p3 | 24.26 | 3.94 | 3.38 | 0.94 |

The extra MNE control using supplied BrainVision R markers did not close the gap, suggesting that detector differences alone do not explain it. Adding the 212 ms delay improved the initial zero-delay MNE results, whose heartbeat-average RMS values were 6.90, 6.97 and 6.50 µV.

Median channel correlations with BrainVision in 1–40 Hz were **0.565, 0.592 and 0.586**. Median waveform RMS differences were **17.55, 22.14 and 20.94 µV**. The two processed versions are therefore not interchangeable. The task-related C3/C4 alpha check generally showed reduced alpha during movement in both versions, with differences in some directions and magnitudes; it does not establish preserved motor physiology.

A separate synthetic EEG check added an independent 5 µV RMS, 8–30 Hz signal to Cz before processing. The recovered increment had gain **0.965–0.972**, correlation **0.922–0.925**, and normalized RMS error **0.401–0.409** relative to the expected signal. Near-unity gain therefore does not justify saying that 97% of real EEG was preserved: the pipeline also introduced additional variation.

**EMG results**

The table gives **20–95 Hz rest RMS in microvolts** for each monopole and their derived difference. These are individual-channel measurements, whereas the EEG values above are medians across scalp channels.

| Recording | Channel | Input | MNE AAS only | MNE AAS + OBS | BrainVision |
|---|---|---:|---:|---:|---:|
| p1 | F1 | 536.1 | 209.7 | 210.9 | 168.2 |
| p1 | F2 | 428.8 | 202.7 | 206.1 | 160.8 |
| p1 | F1−F2 | 391.5 | 159.7 | 169.3 | 121.8 |
| p2 | F1 | 553.1 | 238.1 | 235.8 | 191.8 |
| p2 | F2 | 429.1 | 213.7 | 213.0 | 170.6 |
| p2 | F1−F2 | 424.2 | 184.0 | 197.7 | 143.4 |
| p3 | F1 | 528.5 | 119.7 | 123.6 | 93.4 |
| p3 | F2 | 406.4 | 113.0 | 117.4 | 87.6 |
| p3 | F1−F2 | 373.5 | 101.7 | 108.7 | 79.5 |

For F1−F2, full MNE correction reduced rest RMS by **56.8%, 53.4% and 70.9%** from input, but the result remained **39.0%, 37.8% and 36.7% higher than BrainVision**. Periodic scanner contamination nevertheless fell sharply: scanner-harmonic excess power decreased by **99.84%, 99.79% and 99.74%**. This percentage concerns the selected scanner spectral peaks, not all artifact power.

Typical rest amplitude was closer between methods than whole-rest RMS suggests. The median 100 ms F1−F2 RMS envelope was **31.4, 34.7 and 36.7 µV** with MNE versus **27.8, 28.0 and 30.2 µV** with BrainVision. The corresponding 95th percentiles were **267.7, 269.8 and 175.9 µV** versus **123.6, 120.3 and 87.0 µV**. Large intermittent events therefore contribute strongly to the whole-rest difference. F1−F2 envelope correlations with BrainVision were **0.942, 0.913 and 0.916**, reflecting agreement in major bursts without proving preservation of weaker EMG.

The cardiac step had mixed effects. It reduced F1−F2 heartbeat-average RMS from **26.1 to 17.9**, **27.6 to 16.7** and **39.2 to 18.1 µV**, but increased 20–95 Hz rest RMS by **6.1%, 7.5% and 6.8%** relative to AAS alone. Independent monopolar cardiac fits need not cancel in their difference. The complete EEG settings therefore did not yield an unambiguous improvement for EMG.

The software schedule contains 20 left-cue and 20 right-cue four-second microrepeats per recording. F1/F2 are the right-hand electrode pair. The table shows **median trial F1−F2 movement/preparation RMS ratios**, comparing seconds 2.2–3.8 with seconds 0.2–1.8 of each microrepeat; 1 means equal amplitude.

| Recording | Left cue: MNE | Left cue: BrainVision | Right cue: MNE | Right cue: BrainVision |
|---|---:|---:|---:|---:|
| p1 | 1.01 | 1.02 | 0.98 | 1.14 |
| p2 | 1.04 | 1.01 | 1.91 | 2.26 |
| p3 | 1.08 | 0.98 | 2.15 | 2.90 |

Right-cue increases remained apparent in p2 and p3, but with weaker movement/preparation contrast after full MNE correction. The contrast in p1 was weak with both methods. These differences cannot be assigned uniquely to true EMG, motion or correction artifacts from this comparison alone.

The synthetic EMG test revealed a specific source of temporal contamination. A 0.6-second, 25–90 Hz burst was injected during rest with opposite signs in F1 and F2, giving 100 µV RMS in their difference. Its gain after full correction was **1.0005, 0.9987 and 0.9970**, with waveform correlations **0.9999, 0.9992 and 0.9973**. However, AAS created **eight inverse copies at approximately 12.5% amplitude**, at offsets **±2.5, ±5, ±7.5 and ±10 seconds**, because the burst entered neighboring scanner templates. OBS largely retained these copies and added further distributed changes. Such copies can resemble small contractions or inflate preparation/rest activity. This one-probe-per-recording test does not quantify preservation of all physiological EMG.

**Interpretation and limitations**

The EEG pipeline is effective for the periodic scanner artifact and substantially reduces cardiac activity, but BrainVision retains an advantage on the measured heartbeat residual. The unchanged transfer to EMG also removes the dominant periodic artifact, yet produces higher rest amplitude and demonstrable burst copies. It should be treated as an experimental EMG correction rather than an optimized replacement for the BrainVision output. Both AAS-only and full-correction stages were retained.

BrainVision is a processed comparator, not clean ground truth. Its original scanner-window and cardiac-correction settings were unavailable, and its processing environment could not be subjected to the same injection tests. Lower RMS, smaller signed heartbeat averages and higher intermethod correlation are each incomplete indicators of signal quality.

Original scanner triggers at 5,000 Hz were not supplied. Correction used a stable 12,500-sample computational grid supported by the waveform check and the approximately 2.5-second TR. This does not recover physical scanner-trigger timestamps. The incomplete final 76 ms remains uncorrected and is marked BAD; the first and last five seconds are also marked BAD in final outputs. The final 250 Hz, 0.5–100 Hz data do not retain the full native EMG bandwidth. Native 5,000 Hz AAS outputs are available separately.

The findings apply to these three overt-movement recordings. Export read-back checks verified channel names, sample counts and voltage scale. The earlier EEG implementation checks passed, and the EMG exports were checked across all samples. Each final recording has 85,644 samples at 250 Hz. Per-run processing files record the settings, ECG detections, source paths and provenance.

**Supporting files**

| Material | EEG | EMG |
|---|---|---|
| Full modality report | [EEG report](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_fmri_comparison/report.md) | [EMG report](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_emg_comparison/report.md) |
| Figures, PDF | [EEG plots](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_fmri_comparison/comparison.pdf) | [EMG plots](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_emg_comparison/comparison.pdf) |
| Browsable figures | [EEG HTML](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_fmri_comparison/comparison.html) | [EMG HTML](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_emg_comparison/comparison.html) |
| Channel measurements | [EEG CSV](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_fmri_comparison/comparison_channels.csv) | [EMG CSV](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_emg_comparison/comparison_channels.csv) |
| Injection checks | [EEG JSON](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_fmri_comparison/signal_injection_check.json) | [EMG JSON](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_emg_comparison/emg_burst_injection_check.json) |
| Final BrainVision exports and loading notes | [EEG files](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_fmri_comparison/cleaned_brainvision/README.md) | [EMG files](../../CapTest/NeuroCasting_QuasifMRI/output/in1948/mne_emg_comparison/cleaned_brainvision/README.md) |

Intermediate and final FIF recordings and `processing.json` files are in each modality's `block01`, `block02` and `block03` directories. Reproduction commands are included in the full modality reports linked above.
