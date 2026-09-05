"""Guard the replay experiment against future-data leakage and fake equivalence."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from diagnostics.compare_online_mne import (
    ReplaySpec, gradient_replay, replay_downstream, stateful_filter_check,
)
from emgcasting.mri import gradient_channel


def test_delayed_gradient_matches_production_interior():
    rng = np.random.default_rng(42)
    fs = 100
    cycle = 250
    data = rng.normal(size=(2, 14 * cycle + 17)) * 1e-4
    with threadpool_limits(limits=1):
        actual, _ = gradient_replay(data, fs, 4, 4)
        expected = np.stack([gradient_channel(x, fs)[0] for x in data])
    np.testing.assert_allclose(actual[:, 4 * cycle:10 * cycle],
                               expected[:, 4 * cycle:10 * cycle], rtol=1e-12, atol=1e-14)
    assert np.isnan(actual[:, :4 * cycle]).all()
    assert np.isnan(actual[:, 10 * cycle:]).all()


@pytest.mark.parametrize("before,after", [(2, 0), (4, 4)])
def test_gradient_future_changes_cannot_alter_released_cycles(before, after):
    rng = np.random.default_rng(31)
    fs = 100
    data = rng.normal(size=(3, 20 * 250))
    cutoff = 12 * 250
    changed = data.copy()
    changed[:, cutoff:] = rng.normal(size=changed[:, cutoff:].shape) * 1e6
    with threadpool_limits(limits=1):
        first, _ = gradient_replay(data, fs, before, after)
        second, _ = gradient_replay(changed, fs, before, after)
        prefix, _ = gradient_replay(data[:, :cutoff], fs, before, after)
    released_end = cutoff - after * 250
    np.testing.assert_array_equal(first[:, :released_end], second[:, :released_end])
    np.testing.assert_array_equal(first[:, :released_end], prefix[:, :released_end])


def test_downstream_only_sees_available_buffer(monkeypatch):
    # Mock cardiac fitting to isolate time slicing; real MNE resampling and
    # conditioning remain in the path. Production ECG is tested on recordings.
    from diagnostics import compare_online_mne as module
    monkeypatch.setattr(module, "detect_qrs", lambda ecg: (np.array([1., 2., 3.]), {}))
    monkeypatch.setattr(module.mne.preprocessing, "apply_pca_obs", lambda raw, **kwargs: raw.copy())
    rng = np.random.default_rng(17)
    gradient = rng.normal(size=(3, 30 * 250)) * 1e-4
    changed = gradient.copy()
    changed[:, 20 * 250:] *= 1e5
    spec = ReplaySpec(2, 0, 10, 2.5)
    with threadpool_limits(limits=1):
        first = replay_downstream(gradient, 250, spec)
        second = replay_downstream(changed, 250, spec)
    np.testing.assert_array_equal(first[0][:, :int(17.5 * 250)], second[0][:, :int(17.5 * 250)])
    updates = first[4]
    assert updates[0]["acquisition_end_s"] == 10
    for update in updates:
        assert update["buffer_stop_s"] <= update["acquisition_end_s"]
        assert update["emit_stop_s"] + spec.lag_s == update["buffer_stop_s"]
        assert update["buffer_start_s"] <= update["emit_start_s"]
    assert all(a["emit_stop_s"] == b["emit_start_s"] for a, b in zip(updates, updates[1:]))


def test_emg_filter_state_matches_whole_recording():
    values = np.random.default_rng(2).normal(size=11003) * 1e-4
    for result in stateful_filter_check(values):
        assert result["stateful_max_difference_mv"] == 0
        assert result["reset_each_chunk_nrmse"] > 0
