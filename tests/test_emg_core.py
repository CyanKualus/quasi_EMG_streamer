import csv
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emgcasting import core
from emgcasting.core import (EventSet, LoadedRecording, ProcessingConfig,
                             _events_for, _marker_event_sets, _rest_baseline,
                             envelope_trace,
                             parse_pair, process_batch, tkeo,
                             trial_emg_metrics)


def test_parse_pair():
    assert parse_pair("Aux 1.1, Aux 1.2") == ("Aux 1.1", "Aux 1.2")


# ---------------------------------------------------------------------------
# Where the output goes.
#
# The root is written to the settings file and the settings file travels with
# the application, so a root that resolved against the machine an analysis was
# set up on -- or against whatever directory the program was started from --
# would send a copied installation's results somewhere else entirely.
# ---------------------------------------------------------------------------
def test_a_relative_output_root_is_taken_from_the_application_folder(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # started from somewhere else entirely
    config = ProcessingConfig(data_dir=str(tmp_path), recordings=["run1"],
                              participant="P01", output_root="output")
    assert core.planned_output_dir(config) == core.PROJECT_DIR / "output" / "P01"


def test_the_default_output_root_is_relative_and_lands_beside_the_app():
    assert not os.path.isabs(core.DEFAULT_OUTPUT_ROOT)
    assert ProcessingConfig(".", ["x"]).output_root == core.DEFAULT_OUTPUT_ROOT
    assert (core.resolve_output_root(core.DEFAULT_OUTPUT_ROOT).parent
            == core.PROJECT_DIR)


def test_an_emptied_output_root_falls_back_to_the_default(tmp_path, monkeypatch):
    """An operator who clears the box must not scatter folders at the root."""
    monkeypatch.chdir(tmp_path)
    assert core.resolve_output_root("  ") == core.PROJECT_DIR / "output"


def test_an_absolute_output_root_is_still_obeyed_as_given(tmp_path):
    config = ProcessingConfig(data_dir=str(tmp_path), recordings=["run1"],
                              participant="P01",
                              output_root=str(tmp_path / "elsewhere"))
    assert core.planned_output_dir(config) == tmp_path / "elsewhere" / "P01"


def test_default_classification_thresholds():
    config = ProcessingConfig(".", ["x"])
    assert config.peak_multiplier == 7.0
    assert config.min_burst_ms == 50.0
    assert config.trial_tail_s == 0.2
    assert (config.pre_reference_start_s, config.pre_reference_end_s) == (0.5, 1.8)
    assert config.secondary_pre_multiplier == 3.0
    assert config.secondary_width_multiplier == 3.0


def test_negative_classification_tail_is_rejected():
    config = ProcessingConfig(".", ["x"], trial_tail_s=-0.001)
    with pytest.raises(ValueError, match="classification tail"):
        config.validate()


@pytest.mark.parametrize("changes, message", [
    ({"pre_reference_start_s": -0.1}, "reference start"),
    ({"pre_reference_start_s": 1.8, "pre_reference_end_s": 1.8},
     "reference end"),
    ({"secondary_pre_multiplier": 0.0}, "secondary pre-movement"),
    ({"secondary_width_multiplier": 4.0}, "secondary width bar"),
])
def test_invalid_secondary_classifier_settings_are_rejected(changes, message):
    config = ProcessingConfig(".", ["x"], **changes)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_tkeo_constant_and_sinusoid():
    assert np.allclose(tkeo(np.ones(10)), 0)
    time = np.arange(1000) / 500
    signal = np.sin(2 * np.pi * 40 * time)
    assert np.mean(tkeo(signal)[10:-10]) > 0


def test_marker_events_use_data_clock_and_next_event_duration():
    marker = {
        "time_series": [["video_onset;x"], ["left"], ["left"], ["rest"]],
        "time_stamps": np.array([105.0, 107.0, 111.0, 115.0]),
    }
    events, video = _marker_event_sets(marker, 100.0, 20.0, {"left", "rest"})
    assert video == 5.0
    assert events["left"].onsets.tolist() == [7.0, 11.0]
    assert events["left"].durations.tolist() == [4.0, 4.0]


def test_marker_shift_moves_all_onsets_without_changing_durations():
    raw = {
        "left": EventSet(np.array([7.0, 11.0]), np.array([4.0, 4.0])),
        "right": EventSet(np.array([15.0]), np.array([4.0])),
        "rest": EventSet(np.array([19.0]), np.array([8.0])),
    }
    recording = LoadedRecording({}, 500.0, 30.0, raw, 5.0)
    config = ProcessingConfig(
        ".", ["x"], left_condition="left", right_condition="right",
        rest_condition="rest")

    shifted, note = _events_for(recording, config, -4.0, "entered by hand")

    assert shifted["left"].onsets.tolist() == [3.0, 7.0]
    assert shifted["right"].onsets.tolist() == [11.0]
    assert shifted["rest"].onsets.tolist() == [15.0]
    assert shifted["left"].durations.tolist() == [4.0, 4.0]
    assert raw["left"].onsets.tolist() == [7.0, 11.0]
    assert "shifted -4.000 s" in note
    assert "entered by hand" in note
    assert "corrected video_onset at +1.000 s" in note


def test_rms_envelope_shape():
    cfg = ProcessingConfig(".", ["x"], envelope="rms", window_ms=100, step_ms=20)
    values, centers = envelope_trace(np.ones(500), 500, cfg)
    assert values.shape == centers.shape
    assert np.allclose(values, 1.0)


def test_rest_baseline_drops_the_contaminated_tail():
    # Two 14 s rest blocks at a floor of 1.0, each with the next block's
    # movement already running during its final 2 s.
    centers = np.arange(0, 40, 0.02)
    values = np.ones_like(centers)
    for onset in (0.0, 20.0):
        values[(centers >= onset + 12) & (centers < onset + 14)] = 500.0
    events = EventSet(np.array([0.0, 20.0]), np.array([14.0, 14.0]))

    assert _rest_baseline(values, centers, events, 2.0) == 1.0
    # Without the trim the tail dominates, which is the bug this guards.
    assert _rest_baseline(values, centers, events, 0.0) > 50.0


def test_rest_baseline_is_robust_to_one_bad_block():
    centers = np.arange(0, 60, 0.02)
    values = np.ones_like(centers)
    values[(centers >= 40) & (centers < 52)] = 1000.0   # one artefactual block
    events = EventSet(np.array([0.0, 20.0, 40.0]), np.full(3, 14.0))

    assert _rest_baseline(values, centers, events, 2.0) == 1.0


def test_rest_baseline_skips_unusable_blocks():
    centers = np.arange(0, 20, 0.02)
    values = np.ones_like(centers)
    events = EventSet(np.array([0.0, 10.0]), np.array([np.nan, 1.0]))

    assert np.isnan(_rest_baseline(values, centers, events, 2.0))


def _peak_fixture(spacing=0.02, background=1.0):
    """Three 10 s rest blocks at a flat background, then two trial slots."""
    centers = np.arange(0, 60, spacing)
    values = np.full_like(centers, background)
    rest = EventSet(np.array([0.0, 12.0, 24.0]), np.full(3, 10.0))
    return centers, values, rest


def test_both_thresholds_are_multiples_of_this_recordings_own_background():
    centers, values, rest = _peak_fixture(background=2.0)
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=100.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.background == 2.0
    assert metrics.peak_threshold == 40.0
    assert metrics.burst_threshold == 7.0
    assert metrics.rest_fpr == 0.0


def test_quiet_pre_movement_reference_can_admit_a_3x_preparation_peak():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 40.0))  # +2.0 s, after preparation
    values[start:start + 6] = 5.5              # 120 ms at 5.5x preparation
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 4.1, rest, config)

    assert metrics.primary_high.tolist() == [False]
    assert metrics.secondary_high.tolist() == [True]
    assert metrics.high.tolist() == [True]
    assert metrics.secondary_peak_ratio[0] == pytest.approx(5.5)
    assert metrics.peak_pre_ratio[0] == pytest.approx(5.5)
    assert metrics.secondary_longest_burst_ms[0] == pytest.approx(120.0)


def test_secondary_branch_still_requires_3x_preparation():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 40.0))
    values[start:start + 6] = 2.9
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 4.1, rest, config)

    assert metrics.peak_pre_ratio[0] == pytest.approx(2.9)
    assert metrics.secondary_high.tolist() == [False]
    assert metrics.high.tolist() == [False]


def test_secondary_branch_is_independent_of_recording_rest():
    centers, values, rest = _peak_fixture(background=2.0)
    preparation = ((centers >= 38.5) & (centers < 39.8))
    values[preparation] = 1.0
    start = np.argmin(np.abs(centers - 40.0))
    values[start:start + 6] = 3.2  # 3.2x prep, but only 1.6x rest
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 4.1, rest, config)

    assert metrics.secondary_peak_ratio[0] == pytest.approx(1.6)
    assert metrics.peak_pre_ratio[0] == pytest.approx(3.2)
    assert metrics.primary_high.tolist() == [False]
    assert metrics.secondary_high.tolist() == [True]
    assert metrics.high.tolist() == [True]


def test_secondary_branch_does_not_classify_activity_inside_preparation():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 39.0))  # +1.0 s, inside reference
    values[start:start + 6] = 6.0
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 4.1, rest, config)

    assert metrics.primary_high.tolist() == [False]
    assert metrics.secondary_peak_ratio[0] == pytest.approx(1.0)
    assert metrics.secondary_high.tolist() == [False]
    assert metrics.high.tolist() == [False]


def test_elevated_background_lowers_sensitivity_by_design():
    # The same absolute peak, against two different rest backgrounds. The run
    # with the higher background is deliberately allowed to miss it.
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=4.0, background_multiplier=3.5, min_burst_ms=200.0,
        rest_pseudotrial_step_s=2.0)
    verdicts = []
    for background in (1.0, 3.0):
        centers, values, rest = _peak_fixture(background=background)
        start = np.argmin(np.abs(centers - 38.0))
        values[start:start + 15] = 8.0
        metrics = trial_emg_metrics(
            values, centers, np.array([38.0]), 2.0, rest, config)
        verdicts.append(bool(metrics.high[0]))

    assert verdicts == [True, False]


def test_a_wide_but_short_peak_is_not_high():
    # The rule's whole point: height carries the decision. A burst that is
    # amply wide but only reaches 4x its own background is background, and the
    # old width-or-energy rule called it movement.
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 40] = 4.0                      # 800 ms at 4x
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=100.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.longest_burst_ms[0] == pytest.approx(800.0)
    assert metrics.peak_ratio[0] == pytest.approx(4.0)
    assert metrics.high.tolist() == [False]


def test_a_tall_peak_narrower_than_one_window_is_not_high():
    # The other half of the conjunction: a lone sample far above the bar is
    # not an event the envelope can resolve, so height alone cannot carry it.
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 2] = 60.0                      # 40 ms at 60x
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=100.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.peak_ratio[0] == pytest.approx(60.0)
    assert metrics.longest_burst_ms[0] == pytest.approx(40.0)
    assert metrics.high.tolist() == [False]


def test_default_50_ms_width_accepts_three_20_ms_samples():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 3] = 60.0                      # 60 ms at 60x
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.min_burst_ms == 50.0
    assert metrics.longest_burst_ms[0] == pytest.approx(60.0)
    assert metrics.primary_high.tolist() == [True]


def test_rest_peaks_are_reported_rather_than_silently_raising_the_bar():
    # A burst inside every rest block. The detector fires on rest, so the run
    # must be marked unreliable instead of passing as clean.
    centers, values, rest = _peak_fixture()
    for onset in rest.onsets:
        start = np.argmin(np.abs(centers - (onset + 5.0)))
        values[start:start + 25] = 9.0
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=5.0, background_multiplier=3.5, min_burst_ms=200.0,
        rest_pseudotrial_step_s=2.0, rest_fpr_warn=0.05)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.rest_fpr > 0.05
    assert metrics.rest_unreliable
    # The background is still the rest median, so the bursts did not move it.
    assert metrics.background == 1.0


def test_peak_exactly_at_the_minimum_width_still_counts():
    # 20 ms envelope spacing is not exactly representable, so a 300 ms peak
    # against a 300 ms minimum is decided by how the comparison is made.
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 15] = 9.0                      # exactly 15 x 20 ms
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=5.0, background_multiplier=3.5, min_burst_ms=300.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.longest_burst_ms[0] == pytest.approx(300.0)
    assert metrics.high.tolist() == [True]


def test_peaks_outside_the_motor_trial_are_ignored():
    # A wide, strong burst just before onset and another just after the end.
    # Neither is inside the shaded trial, so neither may flag it.
    centers, values, rest = _peak_fixture()
    for offset in (37.0, 40.5):                   # trial is [38.0, 40.0)
        start = np.argmin(np.abs(centers - offset))
        values[start:start + 40] = 9.0            # 800 ms each
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=5.0, background_multiplier=3.5, min_burst_ms=150.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.longest_burst_ms[0] == 0.0
    assert metrics.active_fraction[0] == 0.0
    assert metrics.high.tolist() == [False]


def test_a_burst_straddling_onset_counts_only_its_inside_part():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 37.8))     # 400 ms, half before onset
    values[start:start + 20] = 9.0
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=5.0, background_multiplier=3.5, min_burst_ms=150.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.longest_burst_ms[0] == pytest.approx(200.0)


def test_motor_segment_matches_the_shaded_span():
    grid = np.arange(-2.0, 6.0 + 1e-9, 0.02)
    segment = core._motor_segment(grid, 0.0, 4.0)

    assert grid[segment].min() == pytest.approx(0.0)
    assert grid[segment].max() < 4.0
    assert not segment[grid < 0.0].any()
    assert not segment[grid >= 4.0].any()


def test_trial_figures_scale_to_the_motor_segment(monkeypatch, tmp_path):
    # A huge spike in the post-trial tail must not squash the trial itself.
    fs = 500.0
    time = np.arange(0, 40, 1 / fs)
    rng = np.random.default_rng(3)
    signals = {name: rng.normal(scale=1e-6, size=time.size)
               for name in ("L+", "L-", "R+", "R-")}
    for channel, onset in (("L+", 4.0), ("R+", 20.0)):
        window = (time >= onset) & (time < onset + 4)
        signals[channel][window] += 2e-4 * np.sin(2 * np.pi * 70 * time[window])
        tail = (time >= onset + 5.0) & (time < onset + 5.5)
        signals[channel][tail] += 5e-2 * np.sin(2 * np.pi * 70 * time[tail])
    events = {
        "left": EventSet(np.array([4.0]), np.array([4.0])),
        "right": EventSet(np.array([20.0]), np.array([4.0])),
        "rest": EventSet(np.array([30.0]), np.array([9.0])),
    }
    monkeypatch.setattr(core, "load_xdf", lambda path, requested, config:
                        LoadedRecording(signals, fs, time[-1], events, 1.0))

    captured = {}
    real = core._plot_trials

    def spy(*args, **kwargs):
        grid, epochs, _onsets, duration = args[3], args[4], args[5], args[6]
        segment = core._motor_segment(grid, 0.0, duration)
        captured["inside_max"] = float(np.nanmax(epochs[0][segment]))
        captured["outside_max"] = float(np.nanmax(epochs[0][~segment]))
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "_plot_trials", spy)
    open(tmp_path / "run1.xdf", "wb").close()
    core.process_batch(ProcessingConfig(
        data_dir=str(tmp_path), recordings=["run1"], participant="P03",
        left_channels=("L+", "L-"), right_channels=("R+", "R-"),
        left_condition="left", right_condition="right", rest_condition="rest",
        output_root=str(tmp_path / "out"), pre_s=1.0, post_s=2.0,
        auto_marker_shift=False, marker_shift_s=0.0))

    # The fixture is only meaningful if the tail really does dwarf the trial.
    assert captured["outside_max"] > 10 * captured["inside_max"]


def test_narrow_peak_fails_a_width_requirement_above_one_window():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 5] = 25.0                      # 100 ms, one window
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5,
        rest_pseudotrial_step_s=2.0)

    config.min_burst_ms = 100.0
    inert = trial_emg_metrics(values, centers, np.array([38.0]), 2.0, rest, config)
    config.min_burst_ms = 200.0
    strict = trial_emg_metrics(values, centers, np.array([38.0]), 2.0, rest, config)

    # 100 ms is one envelope window, so it imposes nothing on a single spike;
    # anything above it demands real width the spike does not have.
    assert inert.high.tolist() == [True]
    assert strict.high.tolist() == [False]


def test_brief_but_strong_peak_still_qualifies_on_height():
    # The 002TEST quasi2/right case: too brief for any width test worth the
    # name, far too strong to be background. Height is what carries it.
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 7] = 30.0                      # 140 ms at 30x
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=100.0,
        rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    assert metrics.peak_ratio[0] == pytest.approx(30.0)
    assert metrics.longest_burst_ms[0] == pytest.approx(140.0)
    assert metrics.high.tolist() == [True]
    assert metrics.rest_fpr == 0.0


def test_neither_condition_can_carry_a_trial_alone():
    # Width without height, and height without width, against the same
    # background. Joined by AND, so each is rejected; together they pass.
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=200.0,
        rest_pseudotrial_step_s=2.0)
    verdicts = []
    for height, width in ((5.0, 30), (60.0, 5), (60.0, 30)):
        centers, values, rest = _peak_fixture()
        start = np.argmin(np.abs(centers - 38.0))
        values[start:start + width] = height
        metrics = trial_emg_metrics(
            values, centers, np.array([38.0]), 2.0, rest, config)
        verdicts.append(bool(metrics.high[0]))

    assert verdicts == [False, False, True]


def test_a_dip_shorter_than_the_envelope_window_does_not_split_a_burst():
    # Two halves of one contraction, 60 ms apart. The envelope is a 100 ms
    # moving window, so it cannot resolve them as separate events; measuring
    # them separately halves the energy of a burst that is physically one.
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 5] = 25.0
    values[start + 8:start + 13] = 25.0                 # 60 ms gap between
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=150.0,
        window_ms=100.0, rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    # Bridged: 10 samples at 24 above background, and 3 bridged samples at 0.
    assert metrics.burst_energy[0] == pytest.approx(24 * 200.0)
    # Width is delimited the same way, so it describes that same one burst.
    assert metrics.longest_burst_ms[0] == pytest.approx(260.0)
    assert metrics.high.tolist() == [True]


def test_a_dip_longer_than_the_envelope_window_leaves_two_bursts():
    centers, values, rest = _peak_fixture()
    start = np.argmin(np.abs(centers - 38.0))
    values[start:start + 5] = 25.0
    values[start + 11:start + 16] = 25.0                # 120 ms gap between
    config = ProcessingConfig(
        ".", ["x"], rest_trim_start_s=0.0, rest_trim_end_s=0.0,
        peak_multiplier=20.0, background_multiplier=3.5, min_burst_ms=150.0,
        window_ms=100.0, rest_pseudotrial_step_s=2.0)

    metrics = trial_emg_metrics(
        values, centers, np.array([38.0]), 2.0, rest, config)

    # One burst, not their sum: 5 samples at 24 above background.
    assert metrics.burst_energy[0] == pytest.approx(24 * 100.0)
    assert metrics.longest_burst_ms[0] == pytest.approx(100.0)
    assert metrics.high.tolist() == [False]


def test_batch_creates_two_figures_and_summary(monkeypatch):
    fs = 500.0
    time = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(4)
    signals = {name: rng.normal(scale=1e-6, size=time.size)
               for name in ("L+", "L-", "R+", "R-")}
    onsets = {"left": np.array([10.0, 14.0]), "right": np.array([30.0, 34.0])}
    for condition, channel in (("left", "L+"), ("right", "R+")):
        for onset in onsets[condition]:
            window = (time >= onset) & (time < onset + 4)
            signals[channel][window] += 2e-4 * np.sin(
                2 * np.pi * 70 * time[window])

    events = {
        "left": EventSet(onsets["left"], np.full(2, 4.0)),
        "right": EventSet(onsets["right"], np.full(2, 4.0)),
        "rest": EventSet(np.array([18.0, 42.0]), np.full(2, 12.0)),
    }
    monkeypatch.setattr(core, "load_xdf", lambda path, requested, config:
                        LoadedRecording(signals, fs, time[-1], events, 5.0))
    metric_durations = []
    real_metrics = core.trial_emg_metrics

    def metric_spy(values, centers, trial_onsets, trial_duration,
                   rest_events, config):
        metric_durations.append(trial_duration)
        return real_metrics(values, centers, trial_onsets, trial_duration,
                            rest_events, config)

    monkeypatch.setattr(core, "trial_emg_metrics", metric_spy)

    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "run1.xdf"), "wb").close()
        config = ProcessingConfig(
            data_dir=tmp, recordings=["run1"], participant="P01",
            left_channels=("L+", "L-"), right_channels=("R+", "R-"),
            left_condition="left", right_condition="right",
            rest_condition="rest", output_root=os.path.join(tmp, "out"),
            pre_s=0.5, post_s=0.5, auto_marker_shift=False, marker_shift_s=0.0)
        result = process_batch(config)

        assert result.participant == "P01"
        assert os.path.isfile(result.summary_csv)
        hands = [hand for rec in result.recordings for hand in rec.hands]
        assert len(hands) == 2
        assert all(os.path.isfile(hand.figure_path) for hand in hands)
        assert all(hand.movement_rest_ratio > 1 for hand in hands)
        assert all(hand.trial_tail_ms == 200.0 for hand in hands)
        assert metric_durations == pytest.approx([4.2, 4.2])

        # trials/<recording>/<hand>/trial_N.png, one figure per marker onset.
        for hand in hands:
            expected = os.path.join(result.output_dir, "trials", "run1", hand.hand)
            assert hand.trial_dir == expected
            assert sorted(os.listdir(expected)) == [
                "trial_1.png", "trial_2.png", "trial_metrics.csv"]
            assert hand.trial_metrics_csv == os.path.join(
                expected, "trial_metrics.csv")
            with open(hand.trial_metrics_csv, encoding="utf-8") as handle:
                contents = handle.read()
            assert "decision_source" in contents
            assert "secondary_high" in contents


def test_trial_figures_can_be_switched_off(monkeypatch, tmp_path):
    fs = 500.0
    time = np.arange(0, 40, 1 / fs)
    rng = np.random.default_rng(7)
    signals = {name: rng.normal(scale=1e-6, size=time.size)
               for name in ("L+", "L-", "R+", "R-")}
    events = {
        "left": EventSet(np.array([4.0]), np.array([4.0])),
        "right": EventSet(np.array([20.0]), np.array([4.0])),
        "rest": EventSet(np.array([28.0]), np.array([8.0])),
    }
    monkeypatch.setattr(core, "load_xdf", lambda path, requested, config:
                        LoadedRecording(signals, fs, time[-1], events, 1.0))

    open(tmp_path / "run1.xdf", "wb").close()
    config = ProcessingConfig(
        data_dir=str(tmp_path), recordings=["run1"], participant="P02",
        left_channels=("L+", "L-"), right_channels=("R+", "R-"),
        left_condition="left", right_condition="right", rest_condition="rest",
        output_root=str(tmp_path / "out"), pre_s=0.5, post_s=0.5,
        trial_figures=False, auto_marker_shift=False, marker_shift_s=0.0)
    result = process_batch(config)

    assert not os.path.exists(os.path.join(result.output_dir, "trials"))
    assert all(hand.trial_dir == ""
               for rec in result.recordings for hand in rec.hands)
    assert all(hand.trial_metrics_csv == ""
               for rec in result.recordings for hand in rec.hands)


def _short_rest_batch(monkeypatch, tmp_path):
    """A recording whose rest is shorter than one trial, so nothing calibrates."""
    fs = 500.0
    time = np.arange(0, 40, 1 / fs)
    rng = np.random.default_rng(9)
    signals = {name: rng.normal(scale=1e-6, size=time.size)
               for name in ("L+", "L-", "R+", "R-")}
    events = {
        "left": EventSet(np.array([4.0]), np.array([4.0])),
        "right": EventSet(np.array([20.0]), np.array([4.0])),
        "rest": EventSet(np.array([30.0]), np.array([4.0])),
    }
    monkeypatch.setattr(core, "load_xdf", lambda path, requested, config:
                        LoadedRecording(signals, fs, time[-1], events, 1.0))
    open(tmp_path / "run1.xdf", "wb").close()
    return ProcessingConfig(
        data_dir=str(tmp_path), recordings=["run1"], participant="P04",
        left_channels=("L+", "L-"), right_channels=("R+", "R-"),
        left_condition="left", right_condition="right", rest_condition="rest",
        output_root=str(tmp_path / "out"), pre_s=0.5, post_s=0.5,
        auto_marker_shift=False, marker_shift_s=0.0)


def test_an_unscorable_hand_stops_a_batch_that_asked_for_trial_output(
        monkeypatch, tmp_path):
    config = _short_rest_batch(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="rest"):
        process_batch(config)


def test_an_unscorable_hand_is_reported_when_the_batch_can_go_on(
        monkeypatch, tmp_path):
    # What the application asks for: one hand without a usable rest reference
    # must not cost the operator every other hand in the session.
    config = _short_rest_batch(monkeypatch, tmp_path)
    result = core.analyze_batch(config, require_metrics=False)

    hands = [hand for rec in result.recordings for hand in rec.hands]
    assert [hand.n_high_trials for hand in hands] == [None, None]
    assert all(hand.threshold_note.startswith("not scored:") for hand in hands)
    assert all(hand.analysis.metrics is None for hand in hands)
    # The epochs are still there, so the mean figures can still be drawn.
    assert all(hand.analysis.epochs.shape[0] == 1 for hand in hands)


def test_analysis_writes_nothing_and_saving_writes_everything(
        monkeypatch, tmp_path):
    fs = 500.0
    time = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(12)
    signals = {name: rng.normal(scale=1e-6, size=time.size)
               for name in ("L+", "L-", "R+", "R-")}
    events = {
        "left": EventSet(np.array([10.0, 14.0]), np.full(2, 4.0)),
        "right": EventSet(np.array([30.0, 34.0]), np.full(2, 4.0)),
        "rest": EventSet(np.array([18.0, 42.0]), np.full(2, 12.0)),
    }
    monkeypatch.setattr(core, "load_xdf", lambda path, requested, config:
                        LoadedRecording(signals, fs, time[-1], events, 5.0))
    open(tmp_path / "run1.xdf", "wb").close()
    config = ProcessingConfig(
        data_dir=str(tmp_path), recordings=["run1"], participant="P05",
        left_channels=("L+", "L-"), right_channels=("R+", "R-"),
        left_condition="left", right_condition="right", rest_condition="rest",
        output_root=str(tmp_path / "out"), pre_s=0.5, post_s=0.5,
        auto_marker_shift=False, marker_shift_s=0.0)

    analysed = core.analyze_batch(config)
    assert not os.path.exists(analysed.output_dir)
    assert analysed.summary_csv == ""

    saved = core.save_batch_outputs(analysed, config)
    assert saved is analysed                 # the same result, now on disk
    assert os.path.isfile(saved.summary_csv)
    with open(saved.summary_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_path"].endswith("run1.xdf")
    assert rows[0]["event_source"] == "xdf"
    assert rows[0]["recording_rate_hz"] == "500.0"
    assert rows[0]["resampled"] == "False"
    for hand in (hand for rec in saved.recordings for hand in rec.hands):
        assert os.path.isfile(hand.figure_path)
        assert os.path.isfile(hand.trial_metrics_csv)
