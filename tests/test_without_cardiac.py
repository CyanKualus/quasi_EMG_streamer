"""No-cardiac replay must be independent of ECG and future samples."""
import mne
import numpy as np
from threadpoolctl import threadpool_limits

from diagnostics.compare_online_mne import ReplaySpec
from diagnostics.compare_without_cardiac import (
    heartbeat_measure, replay_without_cardiac, trial_contrast,
)


def test_no_cardiac_replay_needs_no_ecg_and_never_calls_obs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No cardiac processing should be called")
    monkeypatch.setattr(mne.preprocessing, "find_ecg_events", forbidden)
    monkeypatch.setattr(mne.preprocessing, "apply_pca_obs", forbidden)
    data = np.random.default_rng(71).normal(size=(2, 7500)) * 1e-4
    later_changed = data.copy()
    later_changed[:, 5000:] *= 10000
    with threadpool_limits(limits=1):
        first, updates = replay_without_cardiac(data, 250, ReplaySpec(8, 0, 10))
        second, _ = replay_without_cardiac(later_changed, 250, ReplaySpec(8, 0, 10))
    np.testing.assert_array_equal(first[:, :4375], second[:, :4375])
    assert np.isfinite(first[:, 1250:6875]).all()
    assert all(a["emit_stop_s"] == b["emit_start_s"] for a, b in zip(updates, updates[1:]))


def test_trial_contrast_uses_preparation_and_movement_and_excludes_missing_trials():
    signal = np.ones(3000)
    signal[round(3.2 * 250):round(4.8 * 250)] = 3
    mask = np.ones(3000, dtype=bool)
    mask[2000:] = False
    result = trial_contrast(signal, np.array([1., 8.]), mask)
    assert result["n_trials"] == 1
    assert result["median_ratio"] == 3


def test_heartbeat_measure_detects_known_repeated_waveform():
    signal = np.zeros(5000)
    qrs = np.arange(2., 18., 2.)
    template = np.sin(np.linspace(-4 * np.pi, 4 * np.pi, 177))
    for q in qrs:
        center = round((q + .212) * 250)
        signal[center - 88:center + 89] = template
    result, average = heartbeat_measure(signal, qrs, np.ones(5000, dtype=bool))
    np.testing.assert_allclose(average, template, atol=1e-14)
    assert result["epochs"] == len(qrs)
    assert result["odd_even_template_r"] > .999999
