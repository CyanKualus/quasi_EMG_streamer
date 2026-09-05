import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from emgcasting import core, mri
from emgcasting.single_file import analyze_file
from settings import file_config
from shared import brainvision, file_ready


@pytest.fixture
def recording(tmp_path):
    fs = 250
    n = fs * 340
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 2.0, (n, 3)).astype("<f4")
    name = "participant_block01"
    samples.tofile(tmp_path / f"{name}.eeg")
    path = tmp_path / f"{name}.vhdr"
    path.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nCodepage=UTF-8\n"
        f"DataFile={name}.eeg\nMarkerFile={name}.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=3\nSamplingInterval=4000\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\nCh1=F1,,1,µV\nCh2=F2,,1,µV\nCh3=ECG,,1,µV\n",
        encoding="utf-8")
    (tmp_path / f"{name}.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        f"[Common Infos]\nDataFile={name}.eeg\n[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0\nMk2=Stimulus,S  3,501,1,0\n"
        "Mk3=Stimulus,S  4,84001,1,0\n", encoding="utf-8")
    (tmp_path / "cond seq.txt").write_text("om1 video3\n", encoding="utf-8")
    return path, samples


def test_direct_file_retains_calibration_and_trials(recording, monkeypatch):
    path, samples = recording
    monkeypatch.setattr(mri, "correct_raw", lambda *a, **kw: pytest.fail("MRI must stay off"))
    config = file_config(path, overrides={"trial_figures": False})
    loaded = core.load_brainvision(path, ["F1", "F2"], config)
    np.testing.assert_allclose(loaded.signals["F1"], samples[:, 0] * 1e-6, rtol=2e-7)
    assert loaded.fs == 250
    assert loaded.provenance["video"] == 3
    batch = analyze_file(config)
    assert len(batch.recordings) == 1
    result = batch.recordings[0]
    assert result.recording == path.stem
    assert [(hand.hand, hand.n_trials) for hand in result.hands] == [("right", 20)]
    assert result.provenance["mri_correction"] == {"enabled": False}


def test_mri_route_uses_corrected_monopoles_before_bipolar(recording, monkeypatch):
    import mne
    path, samples = recording
    seen = []
    def correct(raw, channels, **kwargs):
        seen.append((channels, kwargs, raw.preload))
        data = raw.get_data(picks=channels) * np.array([[2.0], [3.0]])
        result = mne.io.RawArray(data, mne.create_info(channels, 250, ["emg"] * 2),
                                 verbose="error")
        return result, {"enabled": True, "bad_intervals_s": []}
    monkeypatch.setattr(mri, "correct_raw", correct)
    loaded = core.load_brainvision(path, ["F1", "F2"], file_config(path, mri=True))
    np.testing.assert_allclose(loaded.signals["F1"] - loaded.signals["F2"],
                               (2 * samples[:, 0].astype(float) - 3 * samples[:, 1]) * 1e-6,
                               atol=3e-12)
    assert seen == [(["F1", "F2"], {"ecg_channel": "ECG", "tr_s": 2.5}, False)]


def test_arbitrary_filename_can_use_explicit_video_without_manifest(recording):
    path, _ = recording
    renamed = path.with_name("new recording.vhdr")
    path.rename(renamed)
    (path.parent / "cond seq.txt").unlink()
    loaded = core.load_brainvision(renamed, ["F1", "F2"], file_config(renamed, video=2))
    assert loaded.provenance["video"] == 2
    assert loaded.video_onset_s == 2
    with pytest.raises(FileNotFoundError, match="select Video"):
        core.load_brainvision(renamed, ["F1", "F2"], file_config(renamed))


def test_block_ordinal_uses_parent_manifest(tmp_path):
    (tmp_path / "cond seq.txt").write_text("om1 video3\nqm1 video2\n", encoding="utf-8")
    path = tmp_path / "Unfitered" / "person_block02.vhdr"
    assert brainvision.session_entry(path).video == 2


def test_saved_runs_and_correction_modes_are_separate(recording, tmp_path):
    path, _ = recording
    config = file_config(path, overrides={"output_root": str(tmp_path / "output"),
                                          "trial_figures": False})
    direct = core.planned_output_dir(config)
    config.remove_mri_artifacts = True
    corrected = core.planned_output_dir(config)
    assert direct != corrected and direct.parent == corrected.parent
    config.remove_mri_artifacts = False
    batch = core.save_batch_outputs(analyze_file(config), config)
    assert Path(batch.summary_csv).is_file()
    metadata = json.loads((Path(batch.output_dir) / "processing.json").read_text())
    assert len(metadata["recordings"][0]["source_files"]) == 3
    assert metadata["config"]["recordings"] == [path.name]


def test_incomplete_trio_and_open_writer_are_not_ready(recording):
    path, _ = recording
    assert len(file_ready.snapshot(path)) == 3
    data = path.with_suffix(".eeg")
    if os.name == "nt":
        with data.open("ab"):
            with pytest.raises(OSError, match="still open"):
                file_ready.snapshot(path)
    data.unlink()
    with pytest.raises(OSError):
        file_ready.snapshot(path)


def test_wait_resets_when_files_change(monkeypatch):
    elapsed = [0.0]
    states = iter([("a",), ("a",), ("b",), ("b",), ("b",), ("b",)])
    monkeypatch.setattr(file_ready.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(file_ready.time, "sleep", lambda dt: elapsed.__setitem__(0, elapsed[0] + dt))
    monkeypatch.setattr(file_ready, "snapshot", lambda path: next(states, ("b",)))
    file_ready.wait_for_recording("example.vhdr", stable_s=.5)
    assert elapsed[0] >= 1.0


def test_wait_can_be_cancelled():
    with pytest.raises(InterruptedError):
        file_ready.wait_for_recording("example.vhdr", cancelled=lambda: True)


def test_changing_file_during_analysis_rejects_result(recording, monkeypatch):
    path, _ = recording
    original = core.analyze_batch
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        with path.with_suffix(".vmrk").open("a") as handle:
            handle.write("\n; changed\n")
        return result
    monkeypatch.setattr(core, "analyze_batch", changing)
    with pytest.raises(RuntimeError, match="changed during"):
        analyze_file(file_config(path))


def test_aas_removes_periodic_signal_and_leaves_incomplete_tail():
    fs, tr = 250, 2.5
    period = np.sin(np.arange(625) * 2 * np.pi * 46 / 625)
    x = np.r_[np.tile(period, 20), np.array([10.0, 11.0])]
    clean, stop = mri.gradient_channel(x, fs, tr)
    assert stop == 12500
    assert np.std(clean[:stop]) < .01 * np.std(x[:stop])
    np.testing.assert_array_equal(clean[stop:], x[stop:])


def test_aas_rejects_short_recording():
    with pytest.raises(ValueError, match="nine complete"):
        mri.gradient_channel(np.ones(1000), 250)


def test_mri_missing_ecg_is_explicit():
    import mne
    raw = mne.io.RawArray(np.ones((2, 1000)),
                          mne.create_info(["F1", "F2"], 250, ["emg"] * 2), verbose="error")
    with pytest.raises(ValueError, match="requires ECG"):
        mri.correct_raw(raw, ["F1", "F2"])


def test_gui_has_single_filename_and_mri_switch(tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from app import QuasiEMGApp
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "gui.ini"), QtCore.QSettings.Format.IniFormat)
    window = QuasiEMGApp(settings)
    try:
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == ["Start", "EMG Analysis", "Online Monitoring"]
        window.ed_filename.setText(str(tmp_path / "run.vhdr"))
        assert window._config().remove_mri_artifacts is False
        window.chk_mri.setChecked(True)
        assert window._config().remove_mri_artifacts is True
        assert len(window._config().recordings) == 1
        window.ed_filename.setText(str(tmp_path / "run.xdf"))
        assert not window.chk_mri.isEnabled()
        assert not window.chk_mri.isChecked()
        assert "klh.pipeline" not in __import__("sys").modules
    finally:
        window.close()
        qt.processEvents()
