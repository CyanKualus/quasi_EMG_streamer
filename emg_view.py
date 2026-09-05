"""The EMG analysis tab and its per-recording trial browser.

Two levels of detail, because the two questions are different ones. The tab
answers "which recordings are contaminated" at a glance: every recording shows
its mean left- and right-hand envelope side by side, and under each the share
of trials the detector called high, against the false-positive floor the same
detector produced on that recording's own rest. The trial browser answers "and
which trials, exactly": all of a hand's trials at once as a grid of thumbnails,
any of which enlarges into a pannable, zoomable panel beside them.

Nothing here re-analyses anything. Both levels draw from the arrays
:func:`emgcasting.core.analyze_batch` already produced, through the same
drawing functions that write the saved figures, so the screen and a saved file
cannot disagree.
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets

from emgcasting import core
from emgcasting.single_file import analyze_file
from shared import theme

PALETTE = theme.PALETTE

# Thumbnails wide enough to show a burst, narrow enough that a 30-trial hand
# fits into a handful of rows. Six columns made a twelve-trial hand two short
# rows adrift in an otherwise empty pane; four fills the pane better and gives
# each thumbnail enough width to read the burst off.
TRIAL_COLUMNS = 4
TRIAL_THUMBNAIL_PX = 165
MEAN_PANEL_PX = 210


class StaticCanvas(FigureCanvas):
    """A figure that lets the mouse wheel through to whatever contains it.

    Matplotlib's canvas overrides ``wheelEvent`` to emit its own
    ``scroll_event`` and never ignores the Qt event, so a canvas inside a
    scroll area silently swallows every wheel turn made over it -- which, on a
    page that is mostly figures, means the page does not scroll at all. None of
    these figures react to the wheel, so it is handed back to the parent.
    """

    def wheelEvent(self, event):    # noqa: N802 - Qt's own spelling
        event.ignore()


def _hand_title(hand: str) -> str:
    return f"{hand} hand"


def _percent_text(hand_result) -> tuple[str, str]:
    """Headline share of high-EMG trials, and any caveat on that number.

    The detector's hit rate on the recording's own rest is deliberately not
    shown. It is a property of the analysis rather than of the participant,
    and printed beside the share it was read as a second result -- a number
    the card was not answering a question about. It is still measured, and
    still written to ``emg_summary.csv``, for a post-hoc look.

    The caveat is not part of that: it is what the share on the card is worth,
    so a share that is known to be an underestimate, or a hand that could not
    be scored at all, still says so here.
    """
    if hand_result.n_high_trials is None:
        return ("not scored",
                hand_result.threshold_note or "this hand could not be scored")
    analysis = getattr(hand_result, "analysis", None)
    metrics = getattr(analysis, "metrics", None)
    if bool(getattr(metrics, "rest_unreliable", False)):
        return (
            f"UNRELIABLE: raw detector {hand_result.n_high_trials}/"
            f"{hand_result.n_trials} high",
            hand_result.threshold_note
            or "the detector also fired on too much rest; do not interpret "
               "the raw trial count",
        )
    return (f"{hand_result.n_high_trials}/{hand_result.n_trials} high EMG "
            f"= {hand_result.high_percent:.0f}%"), hand_result.threshold_note


class AnalysisWorker(QtCore.QThread):
    """Runs the EMG batch off the GUI thread."""

    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config, wait_for_stop=False):
        super().__init__()
        self.config = config
        self.wait_for_stop = wait_for_stop

    def run(self):
        try:
            # A hand whose rest cannot calibrate a threshold is reported as
            # unscored rather than taking the whole participant down with it.
            self.done.emit(analyze_file(
                self.config, self.progress.emit, wait_for_stop=self.wait_for_stop,
                cancelled=self.isInterruptionRequested))
        except Exception:
            self.failed.emit(traceback.format_exc())


class SaveWorker(QtCore.QThread):
    """Writes an already-analysed batch to disk off the GUI thread."""

    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, batch, config):
        super().__init__()
        self.batch, self.config = batch, config

    def run(self):
        try:
            self.done.emit(core.save_batch_outputs(
                self.batch, self.config, self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


class TrialsWindow(QtWidgets.QDialog):
    """Every trial of one recording and hand: overview grid plus detail panel.

    The grid is one figure rather than one canvas per trial, so a 40-trial hand
    opens in a single draw. Clicking a thumbnail -- or moving through them with
    the arrow keys -- enlarges that trial on the right, where matplotlib's own
    toolbar provides the zoom and pan; the numbers behind its verdict are
    listed underneath. The recording and hand are named in the title bar, so
    nothing above the trials repeats them.
    """

    def __init__(self, recording_name, hand_result, config, parent=None):
        super().__init__(parent)
        self.hand_result = hand_result
        self.analysis = hand_result.analysis
        self.config = config
        self.recording_name = recording_name
        self.selected = 0
        self._axes = {}
        self.setWindowTitle(
            f"{recording_name} — {_hand_title(hand_result.hand)} — trials")
        self.resize(1220, 780)
        self.setSizeGripEnabled(True)
        # Several hands can be open at once for comparison, so a closed window
        # must release its figures rather than wait for the card to die.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._build()
        self._install_arrow_keys()
        self._draw_overview()
        self._select(0)

    # ---- construction ----
    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # ---- left: all trials at once ----
        self.overview = StaticCanvas(Figure(figsize=(7.0, 6.0)))
        self.overview.mpl_connect("button_press_event", self._clicked)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setWidget(self.overview)
        # A hand with two rows of trials must not leave the other two thirds of
        # the pane as an empty sunken box, so the scroll area is capped at the
        # height of the grid it holds and a stretch takes the rest.
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.scroll, 1)
        left_layout.addStretch(0)
        splitter.addWidget(left)

        # ---- right: the selected trial, zoomable ----
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.detail = FigureCanvas(Figure(figsize=(5.4, 4.4)))
        self.detail_ax = self.detail.figure.add_subplot(111)
        toolbar = NavigationToolbar2QT(self.detail, right)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        self.btn_prev = QtWidgets.QPushButton("← previous")
        self.btn_next = QtWidgets.QPushButton("next →")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        buttons.addWidget(self.btn_prev)
        buttons.addWidget(self.btn_next)
        buttons.addStretch(1)
        self.metrics_text = QtWidgets.QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(240)
        # A column of measurements against a column of thresholds, so the
        # figures have to line up: the one place in the window with a fixed
        # pitch. Set as a font rather than a style sheet, which would drop the
        # border and padding the rest of the fields are drawn with.
        self.metrics_text.setFont(QtGui.QFont("Consolas", 9))
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.detail, 1)
        right_layout.addLayout(buttons)
        right_layout.addWidget(self.metrics_text)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close)
        outer.addLayout(footer)

    # ---- overview ----
    def _draw_overview(self):
        epochs = self.analysis.epochs
        n_trials = epochs.shape[0]
        columns = min(TRIAL_COLUMNS, max(1, n_trials))
        rows = int(np.ceil(n_trials / columns)) if n_trials else 1
        figure = self.overview.figure
        figure.clear()
        self._axes = {}
        for index in range(n_trials):
            ax = figure.add_subplot(rows, columns, index + 1)
            core.draw_trial(
                ax, index, self.analysis.grid, epochs, self.analysis.duration,
                self.analysis.rest_baseline, self.analysis.metrics,
                self.config, compact=True)
            self._axes[ax] = index
        figure.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.04,
                               wspace=0.18, hspace=0.34)
        # One row is worth a fixed amount of screen, so a hand with forty
        # trials scrolls instead of shrinking, and a hand with six does not
        # stretch six thumbnails into six full-height panels. Capping the
        # viewport to match keeps the shortfall out of the scroll area.
        height = rows * TRIAL_THUMBNAIL_PX
        self.overview.setFixedHeight(height)
        self.scroll.setMaximumHeight(height)
        self.overview.draw_idle()

    def _clicked(self, event):
        index = self._axes.get(event.inaxes)
        if index is not None:
            self._select(index)

    def _step(self, delta):
        count = self.analysis.epochs.shape[0]
        if count:
            self._select((self.selected + delta) % count)

    def _install_arrow_keys(self):
        """Walk the trials with the arrow keys, wherever the focus sits.

        Shortcuts rather than ``keyPressEvent``: an arrow key reaching a
        focused text box or scroll area first would move a cursor or the view
        instead of the selection, which is not what an arrow means here.
        """
        for keys, delta in ((("Left", "Up"), -1), (("Right", "Down"), 1)):
            for key in keys:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
                shortcut.setContext(
                    QtCore.Qt.ShortcutContext.WindowShortcut)
                shortcut.activated.connect(
                    lambda step=delta: self._step(step))

    def _select(self, index):
        count = self.analysis.epochs.shape[0]
        if not count:
            return
        self.selected = int(np.clip(index, 0, count - 1))
        for ax, position in self._axes.items():
            chosen = position == self.selected
            for spine in ax.spines.values():
                spine.set_color(PALETTE["accent"] if chosen else PALETTE["line"])
                spine.set_linewidth(2.4 if chosen else 0.8)
        self.overview.draw_idle()
        core.draw_trial(
            self.detail_ax, self.selected, self.analysis.grid,
            self.analysis.epochs, self.analysis.duration,
            self.analysis.rest_baseline, self.analysis.metrics, self.config,
            title=(f"{self.recording_name} — {_hand_title(self.hand_result.hand)} "
                   f"— trial {self.selected + 1}"))
        self.detail.figure.tight_layout()
        self.detail.draw_idle()
        self.metrics_text.setPlainText(self._metrics_report(self.selected))

    def _metrics_report(self, index):
        metrics = self.analysis.metrics
        onset = float(self.analysis.onsets[index])
        lines = [f"trial {index + 1} of {self.analysis.epochs.shape[0]}, "
                 f"marker onset {onset:.3f} s (corrected)"]
        if metrics is None:
            lines.append(self.analysis.metrics_error
                         or "this hand could not be scored")
            return "\n".join(lines)
        verdict = "HIGH EMG" if metrics.high[index] else "low"
        lines.append(f"verdict: {verdict} "
                     f"(branch: {core.decision_source(metrics, index)})")
        lines.append("")
        lines.append("primary branch, against this recording's own rest")
        lines.append(f"  peak      {metrics.peak_ratio[index]:8.2f}× rest "
                     f"(needs {self.config.peak_multiplier:g}×)")
        lines.append(f"  width     {metrics.longest_burst_ms[index]:8.0f} ms "
                     f"at {self.config.background_multiplier:g}× rest "
                     f"(needs {metrics.min_burst_ms:g} ms)")
        lines.append("")
        lines.append("adaptive branch, against this trial's own preparation")
        lines.append(f"  reference {metrics.pre_movement_background[index]:8.3g} "
                     f"({1000 * self.config.pre_reference_start_s:.0f}–"
                     f"{1000 * self.config.pre_reference_end_s:.0f} ms)")
        lines.append(f"  peak      {metrics.peak_pre_ratio[index]:8.2f}× "
                     f"preparation (needs "
                     f"{self.config.secondary_pre_multiplier:g}×), "
                     f"{metrics.secondary_peak_ratio[index]:.2f}× rest")
        lines.append(f"  width     {metrics.secondary_longest_burst_ms[index]:8.0f} "
                     f"ms at {self.config.secondary_width_multiplier:g}× "
                     f"preparation (needs {metrics.min_burst_ms:g} ms)")
        return "\n".join(lines)


class RecordingCard(QtWidgets.QGroupBox):
    """One recording: both mean envelopes, both shares, both trial buttons."""

    def __init__(self, result, config, parent=None):
        super().__init__(parent)
        self.result = result
        self.config = config
        condition = result.provenance.get("condition")
        title = result.recording
        if condition:
            title += f"  ·  {condition}"
        title += f"  ·  {result.fs:g} Hz"
        if np.isfinite(result.marker_shift_s):
            title += f"  ·  marker shift {result.marker_shift_s:+.3f} s"
        self.setTitle(title)
        self._build()

    def _build(self):
        grid = QtWidgets.QGridLayout(self)
        for column, hand in enumerate(self.result.hands):
            canvas = StaticCanvas(Figure(figsize=(4.2, 2.4)))
            canvas.setMinimumHeight(MEAN_PANEL_PX)
            ax = canvas.figure.add_subplot(111)
            analysis = hand.analysis
            if analysis is not None:
                core.draw_hand_mean(
                    ax, self.result.recording, _hand_title(hand.hand),
                    analysis.grid, analysis.epochs, analysis.duration,
                    analysis.rest_baseline, self.config, compact=True)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "no analysis", ha="center", va="center",
                        color=PALETTE["faint"], transform=ax.transAxes)
            canvas.figure.tight_layout()
            canvas.draw_idle()

            # Share and qualifier in one wrapping label rather than two rows:
            # the vertical space a card does not spend on furniture is space
            # the next recording's mean figures can use without scrolling, and
            # a single label imposes no minimum width on the column.
            headline, note = _percent_text(hand)
            text = f"<span style='font-size:14px; font-weight:600'>{headline}</span>"
            if note:
                text += (f" <span style='color:{PALETTE['warn']}'>· {note}"
                         f"</span>")
            share = QtWidgets.QLabel(text)
            share.setWordWrap(True)
            button = QtWidgets.QPushButton(
                f"{hand.hand}-hand trials ({hand.n_trials})…")
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setEnabled(hand.analysis is not None
                              and hand.analysis.epochs.shape[0] > 0)
            button.clicked.connect(
                lambda _checked=False, h=hand: self._open_trials(h))
            # Half a card wide for four words of label reads as a banner, so
            # the button keeps its own width under the figure it belongs to.
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                 QtWidgets.QSizePolicy.Policy.Fixed)
            button_row = QtWidgets.QHBoxLayout()
            button_row.setContentsMargins(0, 0, 0, 0)
            button_row.addWidget(button)
            button_row.addStretch(1)

            grid.addWidget(canvas, 0, column)
            grid.addWidget(share, 1, column)
            grid.addLayout(button_row, 2, column)
            grid.setColumnStretch(column, 1)

    def _open_trials(self, hand):
        window = TrialsWindow(self.result.recording, hand, self.config, self)
        window.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        window.show()


class EMGTab(QtWidgets.QWidget):
    """The EMG analysis tab: one card per recording, plus the run log.

    The tab owns no input fields. It reads the filename and correction choice
    from the Start tab and the processing parameters from the settings file.
    """

    # Mirrored onto the start tab, so a run started there can be followed
    # without changing tabs to find out whether the EMG half is still going.
    status_changed = QtCore.pyqtSignal(str)
    # True when the recording was analysed, False when it failed.
    analysis_finished = QtCore.pyqtSignal(bool)

    def __init__(self, config_provider, parent=None):
        super().__init__(parent)
        self._config_provider = config_provider
        self.batch = None
        self.config = None
        self.worker = None
        self.saver = None
        self._build()

    def _set_status(self, text):
        self.lbl_status.setText(text)
        self.status_changed.emit(text)

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 14)
        outer.setSpacing(10)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        self.lbl_status = QtWidgets.QLabel(
            "Choose a filename on the Start tab and press Process file.")
        self.lbl_status.setObjectName("statusLine")
        self.lbl_status.setWordWrap(True)
        self.btn_save = QtWidgets.QPushButton("Save figures + CSV")
        self.btn_save.setObjectName("primaryButton")
        self.btn_save.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_save.setToolTip(
            "Write the mean figures, one figure per trial, trial_metrics.csv "
            "and emg_summary.csv under the output root. The analysis on screen "
            "is already complete; this only puts it on disk.")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        self.btn_open = QtWidgets.QPushButton("Open output folder")
        self.btn_open.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open_output)
        head.addWidget(self.lbl_status, 1)
        head.addWidget(self.btn_save)
        head.addWidget(self.btn_open)
        outer.addLayout(head)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Cards are laid out down the page; a sideways scrollbar would mean a
        # recording whose numbers are off screen to the right.
        self.scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards = QtWidgets.QWidget()
        # The cards carry the white; the strip behind them is the page, so a
        # run with two recordings does not end in a tall panel of nothing.
        self.cards.setObjectName("scrollBody")
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards)
        self.cards_layout.setContentsMargins(0, 0, 6, 0)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.cards)
        outer.addWidget(self.scroll, 1)

    # ---- running ----
    def start(self, wait_for_stop=False):
        """Analyse the recordings the start tab currently describes."""
        try:
            config = self._config_provider()
            config.validate()
        except Exception as exc:
            self._set_status(f"EMG not started: {exc}")
            return False
        self.config = config
        self._clear_cards()
        self.batch = None
        self.btn_save.setEnabled(False)
        self.btn_open.setEnabled(False)
        self._set_status("analysing EMG…")
        self.worker = AnalysisWorker(config, wait_for_stop)
        # Progress replaces the status line as it arrives; the final word is
        # written by whichever of done/failed follows it.
        self.worker.progress.connect(self._set_status)
        self.worker.done.connect(self._analysis_done)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.start()
        return True

    def skip(self, reason):
        """Report why this run carries no EMG analysis, and clear the old one.

        A stale result from the previous participant left on screen would be
        read as this one's, which is worse than an empty tab.
        """
        self._clear_cards()
        self.batch = None
        self.btn_save.setEnabled(False)
        self.btn_open.setEnabled(False)
        self._set_status(reason)

    def _analysis_done(self, batch):
        self.batch = batch
        self._show(batch)
        scored = sum(1 for rec in batch.recordings for hand in rec.hands
                     if hand.n_high_trials is not None)
        total = sum(len(rec.hands) for rec in batch.recordings)
        self._set_status(
            f"{len(batch.recordings)} recording(s), {scored} of "
            f"{total} configured hand channel(s) scored. Nothing written yet.")
        self.btn_save.setEnabled(True)
        self.analysis_finished.emit(True)

    def _analysis_failed(self, details):
        self._set_status("EMG analysis failed.")
        self._report_failure("EMG analysis failed", details)
        self.analysis_finished.emit(False)

    def _report_failure(self, title, details):
        """Show a traceback the tab no longer has a log to keep it in."""
        QtWidgets.QMessageBox.critical(self, title, details)

    def _save(self):
        if self.batch is None or self.config is None:
            return
        self.btn_save.setEnabled(False)
        self._set_status("writing EMG figures and tables…")
        self.saver = SaveWorker(self.batch, self.config)
        self.saver.progress.connect(self._set_status)
        self.saver.done.connect(self._save_done)
        self.saver.failed.connect(self._save_failed)
        self.saver.start()

    def _save_done(self, batch):
        self.btn_save.setEnabled(True)
        self.btn_open.setEnabled(os.path.isdir(batch.output_dir))
        self._set_status(f"written to {batch.output_dir}")

    def _save_failed(self, details):
        self.btn_save.setEnabled(True)
        self._set_status("saving failed.")
        self._report_failure("EMG save failed", details)

    # ---- display ----
    def _clear_cards(self):
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _show(self, batch):
        self._clear_cards()
        for result in batch.recordings:
            card = RecordingCard(result, self.config, self.cards)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _open_output(self):
        if self.batch is None:
            return
        path = self.batch.output_dir
        if not os.path.isdir(path):
            return
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - the operator asked for it
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
