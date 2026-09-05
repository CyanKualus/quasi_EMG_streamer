"""BrainAmp MR monitoring tab; El Artem's black/white, two-hand plot style."""
from __future__ import annotations

from collections import deque
import queue
import socket
import time

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
from threadpoolctl import threadpool_limits

from emgcasting.core import parse_optional_pair
from emgcasting.online import HISTORY, OnlineConfig, OnlineProcessor
from shared.rda import BlockSequence, RDAError, parse_data, parse_start, receive_packet


class MonitorWorker(QtCore.QThread):
    """One socket/processing owner; a bounded queue keeps Qt event traffic small."""
    def __init__(self, host, port, config, parent=None):
        super().__init__(parent)
        self.host, self.port, self.config = host, port, config
        self.events = queue.Queue(maxsize=16)
        self.snapshot = {"state": "Connecting", "message": f"Connecting to {host}:{port}…"}
        self.sock = None

    def stop(self):
        self.requestInterruption()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def publish(self, kind, value):
        try:
            self.events.put_nowait((kind, value))
        except queue.Full:
            raise RDAError("Display fell behind the stream. Reconnect to rebuild the template.") from None

    def run(self):
        try:
            with threadpool_limits(limits=1):
                self._receive()
        except Exception as exc:
            if self.isInterruptionRequested():
                self.snapshot = {"state": "Disconnected", "message": "Disconnected."}
            else:
                self.snapshot = {"state": "Stopped", "message": str(exc)}
        finally:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            if self.isInterruptionRequested():
                self.snapshot = {"state": "Disconnected", "message": "Disconnected."}

    def _receive(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=3)
        self.sock.settimeout(.2)
        self.snapshot = {"state": "Waiting", "message": "Connected. Start monitoring in BrainVision Recorder."}
        info = processor = None
        indices = []
        sequence = BlockSequence()
        origin_clock = None
        last_marker = "—"
        last_compute = 0.
        gated = bool(self.config.start_marker)
        last_data_clock = time.monotonic()
        while not self.isInterruptionRequested():
            kind, body = receive_packet(self.sock, self.isInterruptionRequested)
            if kind == 1:
                info = parse_start(body)
                missing = set(self.config.channels) - set(info.names)
                if missing:
                    raise RDAError(f"Missing channels: {', '.join(sorted(missing))}. "
                                   f"Recorder offers: {', '.join(info.names)}")
                indices = [info.names.index(name) for name in self.config.channels]
                processor = OnlineProcessor(self.config, info.sfreq)
                sequence = BlockSequence()
                origin_clock = None
                last_marker = "—"
                last_compute = 0.
                gated = bool(self.config.start_marker)
                last_data_clock = time.monotonic()
                self.publish("reset", info)
                self.snapshot = {"state": "Waiting" if gated else "Warming up",
                                 "message": "Waiting for samples"}
            elif kind in (2, 4):
                if processor is None or info is None:
                    raise RDAError("RDA data arrived without a start header")
                block = parse_data(body, kind, info)
                sequence.check(block.number)
                now = time.monotonic()
                last_data_clock = now
                offset = 0
                if block.markers:
                    last_marker = f"{block.markers[-1].kind}: {block.markers[-1].description}"
                if gated:
                    wanted = " ".join(self.config.start_marker.split())
                    hits = [m.position for m in block.markers if " ".join(m.description.split()) == wanted]
                    if not hits:
                        self.snapshot = {"state": "Waiting", "message": f"Waiting for start marker: {wanted}",
                                         "last_data_clock": now, "marker": last_marker}
                        continue
                    offset = min(hits)
                    gated = False
                selected = block.volts[indices, offset:]
                if origin_clock is None:
                    # RDA has sample counters, but no source wall-clock timestamps.
                    # Anchor the FIRST received block only; later delays remain visible.
                    origin_clock = now - selected.shape[1] / info.sfreq
                for frame in processor.feed(selected):
                    if self.isInterruptionRequested():
                        return
                    self.publish("frame", frame)
                    last_compute = frame.compute_s
                received_s = processor.samples_received / info.sfreq
                lag = max(0., time.monotonic() - origin_clock - received_s)
                lag_limit = max(10., 4 * self.config.tr_s)
                if lag > lag_limit:
                    raise RDAError(f"Stream is over {lag_limit:g} seconds behind its initial timing. "
                                   "Check the network/CPU, then reconnect.")
                self.snapshot = {"state": "Live" if processor.frames_emitted else "Warming up",
                                 "message": "", "received_s": received_s,
                                 "origin_clock": origin_clock, "last_data_clock": now,
                                 "marker": last_marker, "compute_s": last_compute,
                                 "sfreq": info.sfreq, "block": block.number}
            elif kind in (3, 6):
                processor = info = None
                self.publish("reset", None)
                self.snapshot = {"state": "Waiting", "message": "Recorder stopped monitoring. Waiting for a new start."}
            elif kind not in (5, 7, 8, 9, 10000):
                raise RDAError(f"Unsupported RDA message type: {kind}")
            # NEWSTATE, INFO and keepalives are not samples. Recording can start
            # while monitoring continues; its state notification must not reset AAS.
            if processor is not None and time.monotonic() - last_data_clock > 10:
                raise TimeoutError("Recorder sent no EEG samples for 10 seconds. Reconnect when monitoring resumes.")


class GraphWindowOptions(QtWidgets.QGroupBox):
    """One hand's display preferences; signal values remain in physical mV²."""

    changed = QtCore.pyqtSignal()

    def __init__(self, hand, parent=None):
        super().__init__(f"{hand.upper()} window options", parent)
        self.setObjectName("windowOptions")
        self.setStyleSheet("QGroupBox#windowOptions { padding: 30px 8px 6px 8px; }")
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(6, 0, 6, 6)
        form.setVerticalSpacing(2)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.window_s = QtWidgets.QSpinBox()
        self.window_s.setRange(5, 120)
        self.window_s.setValue(20)
        self.window_s.setSuffix(" s")
        self.y_min = QtWidgets.QSpinBox()
        self.y_max = QtWidgets.QSpinBox()
        for widget in (self.y_min, self.y_max):
            widget.setRange(-10000, 10000)
            widget.setToolTip("Integer axis value in units of 10ᴺ mV²; set N with Magnitude N")
        self.y_min.setMaximum(9999)
        self.y_max.setMinimum(-9999)
        self.y_min.setValue(-1)
        self.y_max.setValue(10)
        self.magnitude = QtWidgets.QSpinBox()
        self.magnitude.setRange(-12, 6)
        self.magnitude.setValue(-3)
        self.magnitude.setToolTip("Scientific notation: one axis unit = 10ᴺ mV². N = −3 means 0.001 mV².")
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(6)
        self.threshold.setRange(0, 10000)
        self.threshold.setValue(.001)
        self.threshold.setSingleStep(.001)
        self.threshold.setToolTip("Manual TKEO amplitude threshold in mV², independent of N")
        self.pause = QtWidgets.QCheckBox("Freeze display")
        self.pause.setToolTip("Freeze this graph's data; its scale can still be changed while acquisition continues")
        for label, widget in (("Time window", self.window_s), ("Ymin", self.y_min),
                              ("Ymax", self.y_max), ("Magnitude N", self.magnitude)):
            form.addRow(label, widget)
        form.addRow("Threshold (mV²)", self.threshold)
        form.addRow(self.pause)
        for widget in (self.window_s, self.threshold):
            widget.setKeyboardTracking(False)
        self.y_min.valueChanged.connect(lambda: self._limits_changed(self.y_min))
        self.y_max.valueChanged.connect(lambda: self._limits_changed(self.y_max))
        for signal in (self.window_s.valueChanged, self.y_min.valueChanged,
                       self.y_max.valueChanged, self.magnitude.valueChanged,
                       self.threshold.valueChanged, self.pause.toggled):
            signal.connect(lambda *_: self.changed.emit())

    @property
    def unit_mv2(self):
        return 10. ** self.magnitude.value()

    def _limits_changed(self, edited):
        # Keep a nonzero, ordered range, including when a typed limit crosses
        # the other one. Move the opposite limit by one representable step.
        if self.y_min.value() >= self.y_max.value():
            other = self.y_max if edited is self.y_min else self.y_min
            with QtCore.QSignalBlocker(other):
                other.setValue(edited.value() + (1 if edited is self.y_min else -1))

    def restore(self, settings, hand):
        prefix = f"online/windows/{hand}/"
        legacy_extent = settings.value("online/y_range", .01, type=float)
        # A previously selected EMG trace used mV, so its limits cannot be
        # reused for the energy display. Both hands now always show TKEO.
        if settings.value("online/mode", 0, type=int) != 0:
            legacy_extent = .01
        if not np.isfinite(legacy_extent) or legacy_extent <= 0:
            legacy_extent = .01
        legacy_magnitude = max(-12, min(6, int(np.floor(np.log10(legacy_extent))) - 1))
        legacy_unit = 10. ** legacy_magnitude
        defaults = {
            "window_s": settings.value("online/window_s", 20, type=int),
            "magnitude": legacy_magnitude, "y_min": -.05 * legacy_extent / legacy_unit,
            "y_max": legacy_extent / legacy_unit,
            "threshold": settings.value("online/threshold", .001, type=float),
        }
        for name, default in defaults.items():
            widget = getattr(self, name)
            value = settings.value(prefix + name, default,
                                   type=int if name in ("window_s", "magnitude") else float)
            if (name == "magnitude" and settings.contains(prefix + name)
                    and settings.value(prefix + "scale_version", 1, type=int) < 2):
                value = -value  # Convert the old 10⁻ᴺ convention exactly once.
            # Round old fractional limits outward to keep their data in view.
            if name == "y_min":
                value = int(np.floor(value))
            elif name == "y_max":
                value = int(np.ceil(value))
            with QtCore.QSignalBlocker(widget):
                widget.setValue(value)
        self._limits_changed(self.y_min)
        # Old automatic-scale preferences no longer override manual limits.
        self.changed.emit()

    def save(self, settings, hand):
        prefix = f"online/windows/{hand}/"
        for name in ("window_s", "y_min", "y_max", "magnitude", "threshold"):
            settings.setValue(prefix + name, getattr(self, name).value())
        settings.setValue(prefix + "scale_version", 2)


class OnlineTab(QtWidgets.QWidget):
    def __init__(self, settings, parent=None, can_start=lambda: True):
        super().__init__(parent)
        self.settings = settings
        self.can_start = can_start
        self.worker = None
        self.config = OnlineConfig()
        self.frames = deque()
        self.display_data = {hand: (np.empty(0), np.empty(0), 0.) for hand in ("left", "right")}
        self.latest_frame = None
        self.stream_info = None
        self.last_snapshot = {}
        self._dirty = True
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 12, 18, 12)
        title = QtWidgets.QLabel("BrainAmp MR · online TKEO")
        title.setStyleSheet("font-size: 21px; font-weight: 600;")
        outer.addWidget(title)
        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, 1)
        plots_layout = QtWidgets.QVBoxLayout()
        body.addLayout(plots_layout, 1)
        sidebar = QtWidgets.QWidget()
        sidebar.setMaximumWidth(340)
        controls = QtWidgets.QVBoxLayout(sidebar)
        controls.setContentsMargins(0, 0, 0, 0)
        self.connection_box = QtWidgets.QGroupBox("Stream options")
        form = QtWidgets.QFormLayout(self.connection_box)
        self.host = QtWidgets.QLineEdit("127.0.0.1")
        self.host.setToolTip("IP address or hostname of the computer running BrainVision Recorder")
        self.port = QtWidgets.QComboBox()
        self.port.setEditable(True)
        self.port.addItems(["51244", "51234"])
        self.port.setToolTip("51244 = float32 (recommended); 51234 = signed int16")
        self.left = QtWidgets.QLineEdit("")
        self.right = QtWidgets.QLineEdit("F1, F2")
        for edit in (self.left, self.right):
            edit.setPlaceholderText("+, − channels; blank = disabled")
        self.tr = QtWidgets.QDoubleSpinBox()
        self.tr.setRange(.5, 10)
        self.tr.setDecimals(3)
        self.tr.setSingleStep(.5)
        self.tr.setSuffix(" s")
        self.tr.setValue(2.5)
        self.tr.setToolTip("Stable artifact repetition period; 2.5 s matches the tested recordings")
        self.buffer = QtWidgets.QComboBox()
        for n in (2, 3, 4, 8):
            self.buffer.addItem("", n)
        self.buffer.setCurrentIndex(2)
        self.marker = QtWidgets.QLineEdit()
        self.marker.setPlaceholderText("Optional; blank = first sample")
        self.marker.setToolTip("Wait for this RDA marker description before starting the fixed cycle grid")
        for label, widget in (("Recorder host", self.host), ("RDA port", self.port),
                              ("LEFT (+, −)", self.left), ("RIGHT (+, −)", self.right),
                              ("Cycle / TR", self.tr), ("Filter buffer", self.buffer),
                              ("Start marker", self.marker)):
            form.addRow(label, widget)
        controls.addWidget(self.connection_box)
        self.connection_controls = QtWidgets.QWidget()
        connection_layout = QtWidgets.QVBoxLayout(self.connection_controls)
        connection_layout.setContentsMargins(0, 0, 0, 0)
        buttons = QtWidgets.QHBoxLayout()
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.setObjectName("primaryButton")
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.connect_button.clicked.connect(self.start)
        self.disconnect_button.clicked.connect(self.stop)
        buttons.addWidget(self.connect_button)
        buttons.addWidget(self.disconnect_button)
        connection_layout.addLayout(buttons)
        self.timing = QtWidgets.QLabel()
        self.timing.setWordWrap(True)
        controls.addWidget(self.timing)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("Waiting for stream")
        connection_layout.addWidget(self.progress)
        self.sensors = {}
        for hand in ("left", "right"):
            label = QtWidgets.QLabel(f"{hand.upper()} · waiting")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(46)
            self.sensors[hand] = label
        controls.addStretch()
        hint = QtWidgets.QLabel("In Recorder: enable Remote Data Access and start monitoring. "
                               "Record the source data in Recorder as usual.")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(sidebar)
        self.stream_panel = QtWidgets.QWidget()
        self.stream_panel.setMinimumWidth(310)
        self.stream_panel.setMaximumWidth(355)
        column_layout = QtWidgets.QVBoxLayout(self.stream_panel)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.addWidget(scroll, 1)
        body.addWidget(self.stream_panel)
        self.stream_toggle = QtWidgets.QToolButton()
        self.stream_toggle.setCheckable(True)
        self.stream_toggle.setFixedWidth(24)
        self.stream_toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                        QtWidgets.QSizePolicy.Policy.Expanding)
        self.stream_toggle.toggled.connect(self._set_stream_expanded)
        body.addWidget(self.stream_toggle)
        self.window_options = {}
        self.plots, self.curves, self.threshold_lines = {}, {}, {}
        for hand in ("left", "right"):
            row = QtWidgets.QHBoxLayout()
            plots_layout.addLayout(row, 1)
            options = GraphWindowOptions(hand)
            options_scroll = QtWidgets.QScrollArea()
            options_scroll.setWidgetResizable(True)
            options_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            options_scroll.setWidget(options)
            options_column = QtWidgets.QWidget()
            options_column.setMinimumWidth(280)
            options_column.setMaximumWidth(290)
            options_layout = QtWidgets.QVBoxLayout(options_column)
            options_layout.setContentsMargins(0, 0, 0, 0)
            options_layout.addWidget(options_scroll, 1)
            if hand == "right":
                options_layout.addWidget(self.connection_controls)
            row.addWidget(options_column)
            self.window_options[hand] = options
            graph_column = QtWidgets.QVBoxLayout()
            row.addLayout(graph_column, 1)
            plot = pg.PlotWidget(background="k")
            plot.showGrid(x=True, y=True, alpha=.25)
            plot.setLabel("bottom", "Time from monitoring start", units="s")
            plot.setTitle(f"{hand.upper()} · TKEO", color="w", size="14pt")
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.disableAutoRange()
            plot.getPlotItem().hideButtons()
            plot.setMinimumHeight(190)
            for axis in ("left", "bottom"):
                plot.getAxis(axis).setTextPen("w")
                plot.getAxis(axis).setPen(pg.mkPen("#8d939c"))
                # A prefix on the already squared mV² unit would be ambiguous.
                plot.getAxis(axis).enableAutoSIPrefix(False)
            curve = plot.plot(pen=pg.mkPen("w", width=1))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#b68a42", style=QtCore.Qt.PenStyle.DashLine))
            plot.addItem(line)
            graph_column.addWidget(plot, 1)
            graph_column.addWidget(self.sensors[hand])
            self.plots[hand], self.curves[hand], self.threshold_lines[hand] = plot, curve, line
        self.status = QtWidgets.QLabel("Disconnected. Enter the Recorder host and channel pairs, then connect.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.status)
        self.details = QtWidgets.QLabel("RDA input · AAS with 8 past cycles · output 250 Hz")
        self.details.setWordWrap(True)
        outer.addWidget(self.details)
        self._restore()
        self.tr.valueChanged.connect(self._timing_changed)
        self.buffer.currentIndexChanged.connect(self._timing_changed)
        self._timing_changed()
        for hand, options in self.window_options.items():
            options.changed.connect(lambda hand=hand: self._display_changed(hand))
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._set_sensors("Waiting")
        self._draw()

    def is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def _timing_changed(self):
        tr = self.tr.value()
        for index in range(self.buffer.count()):
            cycles = self.buffer.itemData(index)
            self.buffer.setItemText(index, f"{cycles * tr:g} s ({cycles} cycles)")
        startup = (HISTORY + self.buffer.currentData()) * tr
        self.timing.setText(f"{HISTORY * tr:g} s template history · first display at ≈{startup:g} s\n"
                            f"Delay {tr:g}–{2 * tr:g} s + computation")

    def _set_stream_expanded(self, expanded):
        self.stream_panel.setVisible(expanded)
        self.stream_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow if expanded else QtCore.Qt.ArrowType.LeftArrow)
        label = "Fold stream options to the right" if expanded else "Show stream options"
        self.stream_toggle.setToolTip(label)
        self.stream_toggle.setAccessibleName(label)

    def _display_changed(self, hand):
        # Apply edits synchronously, even without a worker or while frozen.
        self._draw(hand)

    def start(self):
        if self.is_running():
            return
        if not self.can_start():
            self.status.setText("Finish file analysis or saving before starting online monitoring.")
            return
        try:
            config = OnlineConfig(parse_optional_pair(self.left.text()),
                                  parse_optional_pair(self.right.text()), self.tr.value(),
                                  self.buffer.currentData(), self.marker.text().strip())
            port = int(self.port.currentText())
            if not 1 <= port <= 65535 or not self.host.text().strip():
                raise ValueError("Enter a Recorder host and a port between 1 and 65535")
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.save_settings()
        self.config = config
        self._reset(None)
        for options in self.window_options.values():
            options.pause.setChecked(False)
        self.connection_box.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = MonitorWorker(self.host.text().strip(), port, config, self)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def stop(self):
        if self.is_running():
            self.worker.stop()
            self.disconnect_button.setEnabled(False)
            self.status.setText("Disconnecting…")

    def _finished(self):
        self._poll()
        self.connection_box.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

    def _reset(self, info):
        self.stream_info = info
        self.frames.clear()
        self.latest_frame = None
        self.progress.setValue(0)
        self.progress.setFormat("Waiting for stream")
        for hand, curve in self.curves.items():
            self.display_data[hand] = (np.empty(0), np.empty(0), 0.)
            curve.setData([], [])
            pair = self.config.pairs.get(hand)
            self.plots[hand].setTitle(
                f"{hand.upper()} · TKEO · {pair[0]} − {pair[1]}" if pair else f"{hand.upper()} · disabled",
                color="w", size="14pt")
        if info is not None:
            self.details.setText(f"{info.sfreq:g} Hz input → 250 Hz EMG · "
                                 f"Channels: {', '.join(info.names)}")
        else:
            self.details.setText("RDA input · AAS with 8 past cycles · output 250 Hz")
        self._set_sensors("Waiting")
        self._dirty = True

    def _poll(self):
        if self.worker is None:
            if self._dirty:
                self._draw()
                self._dirty = False
            return
        while True:
            try:
                kind, value = self.worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "reset":
                self._reset(value)
            else:
                self.frames.append(value)
                self.latest_frame = value
                # Retain a fixed 120-second maximum even while the plot is frozen.
                while self.frames and self.frames[0].start_s < value.start_s - 120:
                    self.frames.popleft()
                self._dirty = True
        snapshot = self.worker.snapshot
        self.last_snapshot = snapshot
        state = snapshot["state"]
        now = time.monotonic()
        silence = now - snapshot.get("last_data_clock", now)
        if state in ("Live", "Warming up") and silence > 1:
            state = "Stalled"
        received = snapshot.get("received_s", 0.)
        self.progress.setValue(min(1000, round(1000 * received / self.config.startup_s)))
        self.progress.setFormat("Template ready" if received >= self.config.startup_s else
                                f"Warm-up: {received:.1f} / {self.config.startup_s:g} s")
        if snapshot.get("message"):
            message = snapshot["message"]
        elif state == "Warming up":
            message = f"Warming up · {max(0, self.config.startup_s - received):.1f} s of data until first display"
        elif state == "Stalled":
            message = f"Stream stalled · no samples for {silence:.1f} s"
        else:
            message = state
        if self.latest_frame is not None and "origin_clock" in snapshot:
            times = self.latest_frame.envelope_times_s
            if len(times):
                age = max(0., now - snapshot["origin_clock"] - times[-1])
                message += (f" · newest TKEO age ≈{age:.1f} s"
                            f" · processing {snapshot.get('compute_s', 0) * 1000:.0f} ms/update")
        if state == "Live":
            frozen = [hand.upper() for hand in self.config.pairs if self.window_options[hand].pause.isChecked()]
            if frozen:
                message += f" · {', '.join(frozen)} frozen"
        message += f" · marker {snapshot.get('marker', '—')}"
        self.status.setText(message)
        if self._dirty:
            self._draw()
            self._dirty = False
        if state == "Live" and self.latest_frame is not None:
            for hand, label in self.sensors.items():
                if hand not in self.latest_frame.envelope_mv2:
                    self._set_sensor(hand, "Waiting")
                    continue
                options = self.window_options[hand]
                if options.pause.isChecked():
                    self._set_sensor(hand, "Frozen")
                    continue
                env = self.latest_frame.envelope_mv2[hand]
                value = float(env[-1]) if len(env) else 0
                above = value >= options.threshold.value()
                label.setText(f"{hand.upper()} · {'above' if above else 'below'} threshold\n{value:.5g} mV²")
                label.setStyleSheet("color: white; font-weight: 600; border-radius: 5px; "
                                   f"background: {'#ba403e' if above else '#277b53'};")
        else:
            self._set_sensors(state)

    def _set_sensors(self, state):
        for hand in self.sensors:
            self._set_sensor(hand, state)

    def _set_sensor(self, hand, state):
        label = self.sensors[hand]
        label.setText(f"{hand.upper()} · {state.lower() if hand in self.config.pairs else 'disabled'}")
        label.setStyleSheet("color: white; background: #515b68; border-radius: 5px;")

    def _draw(self, selected_hand=None):
        frames = tuple(self.frames)
        for hand, plot in self.plots.items():
            if selected_hand is not None and hand != selected_hand:
                continue
            options = self.window_options[hand]
            unit = options.unit_mv2
            plot.setLabel("left", f"TKEO (×10<sup>{options.magnitude.value()}</sup> mV²)")
            line = self.threshold_lines[hand]
            line.setVisible(hand in self.config.pairs)
            line.setValue(options.threshold.value() / unit)
            if not options.pause.isChecked():
                end = frames[-1].times_s[-1] if frames else 0
                if frames and hand in self.config.pairs:
                    times = np.concatenate([f.envelope_times_s for f in frames])
                    values = np.concatenate([f.envelope_mv2[hand] for f in frames])
                else:
                    times, values = np.empty(0), np.empty(0)
                self.display_data[hand] = times, values, end
            times, values, end = self.display_data[hand]
            duration = options.window_s.value()
            keep = times >= end - duration
            self.curves[hand].setData(times[keep], values[keep] / unit)
            plot.setXRange(max(0, end - duration), max(duration, end), padding=0)
            plot.setYRange(options.y_min.value(), options.y_max.value(), padding=0)

    def _restore(self):
        for name in ("host", "left", "right", "marker"):
            widget = getattr(self, name)
            widget.setText(self.settings.value("online/" + name, widget.text(), type=str))
        self.port.setCurrentText(self.settings.value("online/port", "51244", type=str))
        self.tr.setValue(self.settings.value("online/tr", self.tr.value(), type=float))
        index = self.settings.value("online/buffer", self.buffer.currentIndex(), type=int)
        self.buffer.setCurrentIndex(max(0, min(index, self.buffer.count() - 1)))
        for hand, options in self.window_options.items():
            options.restore(self.settings, hand)
        expanded = self.settings.value("online/stream_options_expanded", True, type=bool)
        self.stream_toggle.setChecked(expanded)
        self._set_stream_expanded(expanded)

    def save_settings(self):
        for name in ("host", "left", "right", "marker"):
            self.settings.setValue("online/" + name, getattr(self, name).text())
        self.settings.setValue("online/port", self.port.currentText())
        self.settings.setValue("online/tr", self.tr.value())
        self.settings.setValue("online/buffer", self.buffer.currentIndex())
        for hand, options in self.window_options.items():
            options.save(self.settings, hand)
        self.settings.setValue("online/stream_options_expanded", self.stream_toggle.isChecked())
        self.settings.sync()
