"""Protocol, numerical continuity and desktop lifecycle checks for live EMG."""
import socket
import struct
import threading
import time

import mne
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from emgcasting.core import ProcessingConfig, envelope_trace, filter_emg
from emgcasting.online import OnlineConfig, OnlineFrame, OnlineProcessor, StreamingEnvelope
from shared.rda import BlockSequence, RDAError, parse_data, parse_start, receive_packet


def packet(kind, body=b""):
    return bytes.fromhex("8e45584396c9864caf4a98bbf6c91450") + struct.pack("<II", 24 + len(body), kind) + body


def start_body(fs=250, names=("F1", "F2"), resolutions=(.5, 2.)):
    return (struct.pack("<Id", len(names), 1e6 / fs) + np.asarray(resolutions, "<f8").tobytes()
            + b"".join(n.encode() + b"\0" for n in names))


def data_body(values, block=1, kind=4, markers=()):
    """Values are wire units, sample-major, deliberately NOT physical volts."""
    body = struct.pack("<III", block, len(values), len(markers))
    body += np.asarray(values, "<f4" if kind == 4 else "<i2").tobytes()
    for position, description in markers:
        text = b"Stimulus\0" + description.encode() + b"\0"
        body += struct.pack("<IIIi", 16 + len(text), position, 1, -1) + text
    return body


@pytest.mark.parametrize("kind", [2, 4])
def test_rda_calibrates_both_wire_formats_and_reads_markers(kind):
    info = parse_start(start_body(fs=5000))
    assert info.names == ("F1", "F2") and info.sfreq == 5000
    block = parse_data(data_body([[4, -3], [8, 7]], kind=kind,
                                  markers=((1, "S  3"),)), kind, info)
    np.testing.assert_allclose(block.volts, [[2e-6, 4e-6], [-6e-6, 14e-6]], rtol=1e-14)
    assert block.markers[0].position == 1
    assert block.markers[0].description == "S  3"
    assert block.markers[0].channel == -1


def test_rda_fragmented_tcp_frame_survives_socket_timeouts():
    first, second = socket.socketpair()
    first.settimeout(.005)
    source = packet(1, start_body())
    def send():
        for fragment in (source[:3], source[3:25], source[25:]):
            second.sendall(fragment)
            time.sleep(.025)
        second.close()
    thread = threading.Thread(target=send)
    thread.start()
    try:
        kind, body = receive_packet(first)
        assert kind == 1 and body == start_body()
        with pytest.raises(EOFError):
            receive_packet(first)
    finally:
        first.close()
        thread.join(1)


def test_rda_rejects_wrong_header_truncation_and_nonfinite_data():
    info = parse_start(start_body())
    for body in (b"", start_body()[:-1]):
        with pytest.raises(RDAError):
            parse_start(body)
    for body in (data_body([[1, 2]])[:-1], data_body([[np.nan, 2]]),
                 data_body([[1, 2]], markers=((3, "S1"),))):
        with pytest.raises(RDAError):
            parse_data(body, 4, info)
    a, b = socket.socketpair()
    try:
        b.sendall(b"x" * 16 + struct.pack("<II", 24, 1))
        with pytest.raises(RDAError, match="header"):
            receive_packet(a)
    finally:
        a.close()
        b.close()


def test_block_counter_wrap_is_valid_but_missing_or_duplicate_blocks_stop():
    counter = BlockSequence()
    for n in (2**32 - 2, 2**32 - 1, 0, 1):
        counter.check(n)
    for n in (1, 3):
        with pytest.raises(RDAError, match="discontinuity"):
            counter.check(n)


def test_envelope_matches_whole_signal_across_irregular_boundaries():
    source = np.random.default_rng(3).normal(size=(2, 5003))
    streaming = StreamingEnvelope(2)
    all_t, all_env = [], []
    start = 0
    for count in (1, 2, 20, 1, 1, 6, 599, 625, 77, 3000, 671):
        t, env = streaming.feed(source[:, start:start + count], 25 + start / 250)
        all_t.append(t)
        all_env.append(env)
        start += count
        assert streaming.data.shape[1] <= 26
    assert start == source.shape[1]
    config = ProcessingConfig(".", ["test"])
    actual = np.concatenate(all_env, axis=1)
    for index in range(2):
        expected, t = envelope_trace(source[index], 250, config)
        np.testing.assert_allclose(actual[index], expected[:actual.shape[1]], atol=1e-14)
    np.testing.assert_allclose(np.concatenate(all_t), 25 + t[:actual.shape[1]])


@pytest.mark.parametrize("buffer_cycles", [2, 3, 4, 8])
def test_processor_matches_replay_has_bounded_buffers_and_no_cardiac(monkeypatch, buffer_cycles):
    from diagnostics.compare_online_mne import ReplaySpec, gradient_replay
    from diagnostics.compare_without_cardiac import replay_without_cardiac
    def forbidden(*args, **kwargs):
        raise AssertionError("Online monitoring must not call cardiac processing")
    monkeypatch.setattr(mne.preprocessing, "find_ecg_events", forbidden)
    monkeypatch.setattr(mne.preprocessing, "apply_pca_obs", forbidden)
    source = np.random.default_rng(22).normal(size=(2, 12017)) * 1e-4
    source += np.sin(np.arange(source.shape[1]) * 2 * np.pi * 46 / 625) * .01
    config = OnlineConfig(left=("F2", "F1"), buffer_cycles=buffer_cycles)
    processor = OnlineProcessor(config, 250)
    with threadpool_limits(limits=1):
        gradient, _ = gradient_replay(source, 250, 8)
        reference, _ = replay_without_cardiac(gradient, 250, ReplaySpec(8, 0, buffer_cycles * 2.5))
        frames = []
        for start in range(0, source.shape[1], 137):
            frames.extend(processor.feed(source[[1, 0], start:start + 137]))
    assert frames[0].acquisition_end_s == config.startup_s
    assert frames[0].start_s == config.startup_s - 5
    assert processor.pending_size == 142
    assert len(processor.native) == 9
    assert len(processor.corrected) == buffer_cycles
    for frame in frames:
        start = round(frame.start_s * 250)
        np.testing.assert_allclose(frame.conditioned_v[[1, 0]], reference[:, start:start + 625], atol=1e-14)
    right = np.concatenate([f.filtered_mv["right"] for f in frames])
    start, end = round(frames[0].start_s * 250), round(frames[-1].start_s * 250) + 625
    expected = filter_emg(reference[0, start:end] - reference[1, start:end], 250, ProcessingConfig(".", ["test"]))
    np.testing.assert_allclose(right, expected, atol=1e-14)
    np.testing.assert_allclose(np.concatenate([f.filtered_mv["left"] for f in frames]), -right, atol=1e-14)
    env, times = envelope_trace(right, 250, ProcessingConfig(".", ["test"]))
    actual = np.concatenate([f.envelope_mv2["right"] for f in frames])
    np.testing.assert_allclose(actual, env[:len(actual)], atol=1e-14)
    np.testing.assert_allclose(np.concatenate([f.envelope_times_s for f in frames]),
                               frames[0].start_s + times[:len(actual)])


def test_config_rejects_invalid_pairs_and_fractional_cycles():
    for kwargs in ({"right": None}, {"right": ("F1", "F1")}, {"tr_s": 2.501}, {"buffer_cycles": 1}):
        with pytest.raises(ValueError):
            OnlineConfig(**kwargs)
    with pytest.raises(ValueError, match="native samples"):
        OnlineProcessor(OnlineConfig(), 333)


class LocalRecorder:
    """A real TCP peer with synthetic RDA samples for lifecycle integration."""
    def __init__(self, mode="live"):
        self.mode = mode
        self.stop_event = threading.Event()
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            with self.listener.accept()[0] as conn:
                if self.mode == "partial":
                    conn.sendall(packet(1, start_body())[:7])
                    self.stop_event.wait(10)
                    return
                conn.sendall(packet(1, start_body()))
                for index in range(36):
                    if self.stop_event.is_set():
                        return
                    samples = np.random.default_rng(index).normal(size=(250, 2)) * 100
                    markers = ((125, "S  3"),) if index == 0 else ()
                    block = index + (2 if self.mode == "gap" and index >= 31 else 1)
                    conn.sendall(packet(4, data_body(samples, block, markers=markers)))
                    if self.stop_event.wait(.03):
                        return
                if self.mode == "restart":
                    conn.sendall(packet(3))
                    conn.sendall(packet(1, start_body()))
                    conn.sendall(packet(4, data_body(np.ones((250, 2)), 1)))
                self.stop_event.wait(10)
        except OSError:
            pass
        finally:
            self.listener.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(2)


def pump_until(qt, predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qt.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for desktop/stream state")


@pytest.mark.parametrize("mode", ["live", "gap", "partial", "restart"])
def test_monitor_gui_network_stop_reconnect_and_freeze(tmp_path, mode):
    from PyQt6 import QtCore, QtWidgets
    from online_view import OnlineTab
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "monitor.ini"), QtCore.QSettings.Format.IniFormat)
    tab = OnlineTab(settings)
    server = LocalRecorder(mode)
    tab.port.setCurrentText(str(server.port))
    if mode == "live":
        tab.marker.setText("S 3")
        tab.left.setText("F2, F1")
    try:
        tab.start()
        if mode == "partial":
            pump_until(qt, lambda: tab.is_running())
        elif mode == "gap":
            pump_until(qt, lambda: not tab.is_running())
            assert "discontinuity" in tab.status.text()
            assert "above" not in tab.sensors["right"].text()
        elif mode == "restart":
            pump_until(qt, lambda: tab.worker.snapshot.get("received_s") == 1)
            tab._poll()
            assert tab.latest_frame is None
            assert tab.worker.snapshot["state"] == "Warming up"
        else:
            pump_until(qt, lambda: tab.latest_frame is not None)
            assert tab.latest_frame.acquisition_end_s >= 30
            assert tab.curves["right"].xData.size > 0
            tab.window_options["right"].pause.setChecked(True)
            frozen = tab.curves["right"].xData.copy()
            pump_until(qt, lambda: tab.worker.snapshot.get("received_s", 0) >= 35.5)
            tab._poll()
            np.testing.assert_array_equal(tab.curves["right"].xData, frozen)
            assert "frozen" in tab.sensors["right"].text()
            assert tab.curves["left"].xData[-1] > frozen[-1]
            assert "frozen" not in tab.sensors["left"].text()
            tab.window_options["right"].pause.setChecked(False)
            tab.window_options["left"].threshold.setValue(0)
            tab.window_options["right"].threshold.setValue(10000)
            tab._poll()
            assert tab.curves["right"].xData[-1] > frozen[-1]
            assert "above" in tab.sensors["left"].text()
            assert "below" in tab.sensors["right"].text()
        tick = time.monotonic()
        tab.stop()
        pump_until(qt, lambda: not tab.is_running(), seconds=4)
        qt.processEvents()
        assert time.monotonic() - tick < 4
        assert tab.connect_button.isEnabled()
    finally:
        tab.stop()
        if tab.worker:
            tab.worker.wait(4000)
        server.close()
        tab.close()
        qt.processEvents()


def test_tkeo_graphs_have_independent_scales_windows_and_thresholds(tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from online_view import OnlineTab
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "plots.ini"), QtCore.QSettings.Format.IniFormat)
    settings.setValue("online/mode", 1)  # A saved EMG preference must not select EMG again.
    settings.setValue("online/auto_scale", True)
    settings.setValue("online/windows/left/auto_scale", True)
    tab = OnlineTab(settings)
    try:
        for hand, options in tab.window_options.items():
            assert options.y_min.isEnabled() and options.y_max.isEnabled()
            assert not tab.plots[hand].getViewBox().autoRangeEnabled()[1]
            np.testing.assert_allclose(tab.plots[hand].viewRange()[1], [-1, 10])
        tab.config = OnlineConfig(left=("F2", "F1"))
        for start in (0., 2.5, 5., 7.5):
            frame = OnlineFrame(start, start + 5, np.zeros((2, 625)),
                                {"left": np.full(625, -99.), "right": np.full(625, 99.)},
                                start + np.arange(125) * .02,
                                {"left": np.full(125, .002), "right": np.full(125, .0004)}, 0.)
            tab.frames.append(frame)
        tab.latest_frame = frame
        left, right = (tab.window_options[hand] for hand in ("left", "right"))
        left.y_min.setValue(-1)
        left.y_max.setValue(4)
        left.magnitude.setValue(-3)
        left.window_s.setValue(5)
        left.threshold.setValue(.003)
        right.y_min.setValue(0)
        right.y_max.setValue(8)
        right.magnitude.setValue(-4)
        right.window_s.setValue(10)
        right.threshold.setValue(.0005)
        # No polling, new samples or event-loop ticks are needed to rescale.
        np.testing.assert_allclose(tab.curves["left"].yData, 2)
        np.testing.assert_allclose(tab.curves["right"].yData, 4)
        np.testing.assert_allclose(tab.plots["left"].viewRange()[1], [-1, 4])
        np.testing.assert_allclose(tab.plots["right"].viewRange()[1], [0, 8])
        assert np.ptp(tab.plots["left"].viewRange()[0]) == pytest.approx(5)
        assert np.ptp(tab.plots["right"].viewRange()[0]) == pytest.approx(10)
        assert len(tab.curves["left"].xData) < len(tab.curves["right"].xData)
        assert tab.threshold_lines["left"].value() == pytest.approx(3)
        assert tab.threshold_lines["right"].value() == pytest.approx(5)
        assert "TKEO" in tab.plots["left"].getAxis("left").labelText

        # Changing the displayed magnitude scales the trace and threshold
        # together; stored energy and the physical threshold remain unchanged.
        left.magnitude.setValue(-4)
        np.testing.assert_allclose(tab.curves["left"].yData, 20)
        np.testing.assert_allclose(tab.curves["right"].yData, 4)
        assert tab.threshold_lines["left"].value() == pytest.approx(30)
        assert left.threshold.value() == .003
        np.testing.assert_array_equal(frame.envelope_mv2["left"], .002)
        assert not tab.plots["left"].getViewBox().autoRangeEnabled()[1]
        assert not tab.plots["right"].getViewBox().autoRangeEnabled()[1]
        assert left.y_min.isEnabled() and right.y_min.isEnabled()
    finally:
        tab.close()
        qt.processEvents()


def test_graph_preferences_restore_per_hand_and_migrate_shared_settings(tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from online_view import OnlineTab
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.Format.IniFormat)
    for name, value in (("window_s", 35), ("y_range", .02), ("threshold", .002), ("auto_scale", False)):
        settings.setValue("online/" + name, value)
    tab = OnlineTab(settings)
    restored = None
    try:
        for options in tab.window_options.values():
            assert options.window_s.value() == 35
            assert options.y_min.value() * options.unit_mv2 == pytest.approx(-.001)
            assert options.y_max.value() * options.unit_mv2 == pytest.approx(.02)
            assert options.threshold.value() == .002
            assert options.y_min.isEnabled() and options.y_max.isEnabled()
        left, right = (tab.window_options[hand] for hand in ("left", "right"))
        left.y_min.setValue(-2)
        left.y_max.setValue(9)
        left.magnitude.setValue(-5)
        left.window_s.setValue(15)
        left.threshold.setValue(.004)
        left.pause.setChecked(True)
        right.y_max.setValue(12)
        tab.save_settings()
        restored = OnlineTab(settings)
        left, right = (restored.window_options[hand] for hand in ("left", "right"))
        assert (left.y_min.value(), left.y_max.value(), left.magnitude.value()) == (-2, 9, -5)
        assert left.window_s.value() == 15 and right.window_s.value() == 35
        assert left.threshold.value() == .004 and right.threshold.value() == .002
        assert right.y_max.value() == 12
        for plot in restored.plots.values():
            assert not plot.getViewBox().autoRangeEnabled()[1]
        assert not left.pause.isChecked()  # Reopening must show incoming data.
        left.y_min.setValue(10)
        assert left.y_max.value() > left.y_min.value()
        left.y_max.setValue(-3)
        assert left.y_min.value() < left.y_max.value()
        lo, hi = restored.plots["left"].viewRange()[1]
        assert lo < hi
    finally:
        tab.close()
        if restored is not None:
            restored.close()
        qt.processEvents()


@pytest.mark.parametrize("old_magnitude", [-2, 3, 12])
def test_signed_magnitude_migration_and_integer_limits(tmp_path, old_magnitude):
    from PyQt6 import QtCore, QtWidgets
    from online_view import GraphWindowOptions
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "exponent.ini"), QtCore.QSettings.Format.IniFormat)
    prefix = "online/windows/left/"
    settings.setValue(prefix + "magnitude", old_magnitude)
    settings.setValue(prefix + "y_min", -.5)
    settings.setValue(prefix + "y_max", 10.25)
    options = GraphWindowOptions("left")
    restored = GraphWindowOptions("left")
    try:
        options.restore(settings, "left")
        assert options.magnitude.value() == -old_magnitude
        assert options.unit_mv2 == pytest.approx(10. ** -old_magnitude, rel=1e-12, abs=0)
        assert isinstance(options.y_min, QtWidgets.QSpinBox)
        assert isinstance(options.y_max, QtWidgets.QSpinBox)
        assert (options.y_min.value(), options.y_max.value()) == (-1, 11)
        options.save(settings, "left")
        restored.restore(settings, "left")
        assert restored.magnitude.value() == -old_magnitude  # No second sign flip.
        assert (restored.y_min.value(), restored.y_max.value()) == (-1, 11)
        restored.y_min.setValue(11)
        assert restored.y_max.value() == 12
        restored.y_max.setValue(-2)
        assert restored.y_min.value() == -3
    finally:
        options.close()
        restored.close()
        qt.processEvents()


def test_manual_scaling_is_immediate_before_samples_and_while_frozen(tmp_path):
    from PyQt6 import QtCore, QtTest, QtWidgets
    from online_view import OnlineTab
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "immediate.ini"), QtCore.QSettings.Format.IniFormat)
    tab = OnlineTab(settings)
    tab.timer.stop()  # Edits must work without even the display polling timer.
    try:
        right = tab.window_options["right"]
        right.y_min.setValue(0)
        right.y_max.setValue(5)
        np.testing.assert_allclose(tab.plots["right"].viewRange()[1], [0, 5])
        # Typing a valid value applies it before Enter/focus loss as well.
        right.y_max.lineEdit().selectAll()
        QtTest.QTest.keyClicks(right.y_max.lineEdit(), "7")
        np.testing.assert_allclose(tab.plots["right"].viewRange()[1], [0, 7])

        first = OnlineFrame(0, 5, np.zeros((2, 625)), {"right": np.zeros(625)},
                            np.array([.05, .07, .09]), {"right": np.array([.001, .002, .003])}, 0.)
        tab.frames.append(first)
        tab.latest_frame = first
        tab._draw()
        right.pause.setChecked(True)
        frozen_x = tab.curves["right"].xData.copy()
        later = OnlineFrame(2.5, 7.5, np.zeros((2, 625)), {"right": np.zeros(625)},
                            np.array([2.55, 2.57, 2.59]), {"right": np.array([.009, .009, .009])}, 0.)
        tab.frames.append(later)
        tab.latest_frame = later
        tab._draw()
        right.magnitude.setValue(-4)
        right.y_max.setValue(40)
        np.testing.assert_array_equal(tab.curves["right"].xData, frozen_x)
        np.testing.assert_allclose(tab.curves["right"].yData, [10, 20, 30])
        np.testing.assert_allclose(tab.plots["right"].viewRange()[1], [0, 40])
        np.testing.assert_allclose(tab.plots["left"].viewRange()[1], [-1, 10])
        assert tab.threshold_lines["right"].value() == pytest.approx(10)
        right.pause.setChecked(False)
        assert tab.curves["right"].xData[-1] == later.envelope_times_s[-1]
        np.testing.assert_allclose(tab.plots["right"].viewRange()[1], [0, 40])
        assert not tab.plots["right"].getViewBox().autoRangeEnabled()[1]
    finally:
        tab.close()
        qt.processEvents()


def test_stream_panel_folds_and_leaves_connection_controls_below_left_options(tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from app import QuasiEMGApp
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "fold.ini"), QtCore.QSettings.Format.IniFormat)
    window = QuasiEMGApp(settings)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    tab = window.online_tab
    restored = None
    try:
        window.tabs.setCurrentWidget(tab)
        window.show()
        qt.processEvents()
        def position(widget):
            return widget.mapTo(tab, QtCore.QPoint())
        right_options = tab.window_options["right"]
        assert position(tab.connect_button).y() > position(right_options).y() + right_options.height()
        assert position(tab.progress).y() > position(tab.connect_button).y()
        assert position(tab.connect_button).x() < position(tab.plots["right"]).x()
        assert position(tab.progress).x() < position(tab.plots["right"]).x()
        expanded_width = tab.plots["right"].width()
        tab.stream_toggle.click()
        qt.processEvents()
        assert not tab.stream_panel.isVisible()
        assert tab.connect_button.isVisible() and tab.disconnect_button.isVisible() and tab.progress.isVisible()
        assert tab.plots["right"].width() > expanded_width + 250
        assert tab.stream_toggle.arrowType() == QtCore.Qt.ArrowType.LeftArrow
        tab.save_settings()
        restored = QuasiEMGApp(settings)
        assert not restored.online_tab.stream_toggle.isChecked()
        assert restored.online_tab.stream_panel.isHidden()
        tab.stream_toggle.click()
        qt.processEvents()
        assert tab.stream_panel.isVisible()
        assert tab.plots["right"].width() == expanded_width
    finally:
        window.close()
        if restored is not None:
            restored.close()
        qt.processEvents()


def test_monitoring_enters_full_screen_and_restores_window_and_saved_geometry(tmp_path):
    from PyQt6 import QtCore, QtGui, QtWidgets
    from app import QuasiEMGApp
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "fullscreen.ini"), QtCore.QSettings.Format.IniFormat)
    window = QuasiEMGApp(settings)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    restored = None
    try:
        window.show()
        qt.processEvents()
        # Normalize oversized geometry to the test platform's virtual screen
        # before checking an exact round trip through full-screen mode.
        window.restoreGeometry(window.saveGeometry())
        qt.processEvents()
        original_size = window.size()
        assert not window.isFullScreen()
        window.tabs.setCurrentWidget(window.online_tab)
        qt.processEvents()
        assert window.isFullScreen()
        window.tabs.setCurrentWidget(window.start_tab)
        qt.processEvents()
        assert not window.isFullScreen()
        assert window.size() == original_size
        window.tabs.setCurrentWidget(window.online_tab)
        qt.processEvents()
        # A deliberately hidden test window cannot receive keyboard focus.
        # Check the registered key and activate its normal signal instead.
        assert window._exit_full_screen.key() == QtGui.QKeySequence("Escape")
        window._exit_full_screen.activated.emit()
        qt.processEvents()
        assert not window.isFullScreen()
        assert window.tabs.currentWidget() is window.online_tab
        assert window.size() == original_size
        window.tabs.setCurrentWidget(window.start_tab)
        window.showMaximized()
        qt.processEvents()
        window.tabs.setCurrentWidget(window.online_tab)
        assert window.isFullScreen()
        window.tabs.setCurrentWidget(window.start_tab)
        qt.processEvents()
        assert window.isMaximized() and not window.isFullScreen()
        window.showNormal()
        qt.processEvents()
        window.tabs.setCurrentWidget(window.online_tab)
        assert window.isFullScreen()
        window.close()
        restored = QuasiEMGApp(settings)
        assert not restored.isFullScreen()
        assert restored.tabs.currentWidget() is restored.start_tab
    finally:
        window.close()
        if restored is not None:
            restored.close()
        qt.processEvents()


def test_main_window_can_reconnect_then_close_during_partial_packet(tmp_path):
    from PyQt6 import QtCore, QtWidgets
    from app import QuasiEMGApp
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "close.ini"), QtCore.QSettings.Format.IniFormat)
    window = QuasiEMGApp(settings)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    tab = window.online_tab
    first, second = LocalRecorder("partial"), LocalRecorder("partial")
    try:
        tab.port.setCurrentText(str(first.port))
        tab.start()
        pump_until(qt, lambda: tab.worker.sock is not None)
        window._refresh_busy()
        assert not window.btn_start.isEnabled()
        assert not tab.connect_button.isEnabled()
        tab.stop()
        pump_until(qt, lambda: not tab.is_running())
        qt.processEvents()
        old = tab.worker
        tab.port.setCurrentText(str(second.port))
        tab.start()
        assert tab.worker is not old
        pump_until(qt, lambda: tab.worker.sock is not None)
        window.close()
        pump_until(qt, lambda: not window.isVisible() and not tab.is_running(), seconds=4)
        assert settings.value("online/port") == str(second.port)
    finally:
        tab.stop()
        if tab.worker:
            tab.worker.wait(4000)
        first.close()
        second.close()
        window.close()
        qt.processEvents()
