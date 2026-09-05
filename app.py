"""Standalone quasi-fMRI hand EMG processing; one filename per run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from emgcasting.core import parse_optional_pair
from emg_view import EMGTab
from online_view import OnlineTab
from settings import file_config, load_settings
from shared import theme


class QuasiEMGApp(QtWidgets.QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings if settings is not None else QtCore.QSettings(
            "NeuroCasting", "quasi_EMG_processing")
        self.setWindowTitle("quasi_EMG_processing")
        self.resize(1120, 780)
        self._monitor_geometry = None
        self._monitor_window_state = QtCore.Qt.WindowState.WindowNoState
        theme.apply(self)
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.start_tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.start_tab)
        outer.setContentsMargins(28, 24, 28, 24)
        title = QtWidgets.QLabel("Quasi-fMRI · hand EMG")
        title.setStyleSheet("font-size: 23px; font-weight: 600;")
        outer.addWidget(title)
        intro = QtWidgets.QLabel("Process one recording as soon as acquisition has finished.")
        outer.addWidget(intro)
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(16)
        self.ed_filename = QtWidgets.QLineEdit()
        self.ed_filename.setPlaceholderText("Full path to a .vhdr or .xdf file")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(self.ed_filename, 1)
        file_row.addWidget(self.btn_browse)
        form.addRow("filename", file_row)
        self.ed_participant = QtWidgets.QLineEdit()
        self.ed_participant.setPlaceholderText("Optional; inferred from filename when blank")
        form.addRow("participant", self.ed_participant)
        self.cmb_video = QtWidgets.QComboBox()
        self.cmb_video.addItem("Automatic — cond seq.txt", None)
        for video in (1, 2, 3):
            self.cmb_video.addItem(f"Video {video}", video)
        self.cmb_video.setToolTip(
            "BrainVision trials use the first S3 marker and the selected video schedule. "
            "Automatic mode reads pN or blockNN from the filename and cond seq.txt. "
            "Select a video explicitly for other filenames. XDF uses its own trial markers.")
        form.addRow("video schedule", self.cmb_video)
        defaults = load_settings()
        self.ed_left = QtWidgets.QLineEdit(
            ", ".join(defaults.get("left_channels") or ()))
        self.ed_right = QtWidgets.QLineEdit(
            ", ".join(defaults.get("right_channels") or ()))
        self.ed_left.setPlaceholderText("Leave blank to disable")
        form.addRow("left hand EMG (+, −)", self.ed_left)
        form.addRow("right hand EMG (+, −)", self.ed_right)
        self.chk_mri = QtWidgets.QCheckBox("Remove MRI artifacts using MNE (AAS + cardiac OBS)")
        self.chk_mri.setToolTip(
            "Use on uncorrected BrainVision recordings. Requires an ECG channel. "
            "Applies the full pipeline from the reviewed EEG/EMG report before EMG analysis.")
        form.addRow("", self.chk_mri)
        self.lbl_mri = QtWidgets.QLabel()
        self.lbl_mri.setWordWrap(True)
        form.addRow("", self.lbl_mri)
        self.chk_wait = QtWidgets.QCheckBox("Wait for this recording to stop before processing")
        self.chk_wait.setToolTip(
            "Waits for the selected file (and BrainVision companion files) to exist, "
            "close and remain unchanged for two seconds. Processes this file once.")
        form.addRow("", self.chk_wait)
        outer.addLayout(form)
        outer.addStretch()
        self.lbl_status = QtWidgets.QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        outer.addWidget(self.lbl_status)
        self.btn_start = QtWidgets.QPushButton("Process file")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.clicked.connect(self.on_start)
        outer.addWidget(self.btn_start)
        self.tabs.addTab(self.start_tab, "Start")
        self.emg_tab = EMGTab(self._config)
        self.tabs.addTab(self.emg_tab, "EMG Analysis")
        self.online_tab = OnlineTab(self.settings, can_start=lambda: not self._busy())
        self.tabs.addTab(self.online_tab, "Online Monitoring")
        self.tabs.currentChanged.connect(self._tab_changed)
        self._exit_full_screen = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        self._exit_full_screen.activated.connect(self._leave_monitor_full_screen)
        self.tabs.setTabToolTip(self.tabs.indexOf(self.online_tab), "Opens full screen; press Esc to return to the window")
        self.emg_tab.status_changed.connect(self.lbl_status.setText)
        self.emg_tab.analysis_finished.connect(self._analysis_finished)
        self._busy_timer = QtCore.QTimer(self)
        self._busy_timer.setInterval(150)
        self._busy_timer.timeout.connect(self._refresh_busy)
        self._busy_timer.start()
        self.chk_mri.toggled.connect(self._mri_changed)
        self.ed_filename.textChanged.connect(self._filename_changed)
        self._restore()
        self._filename_changed()

    def _config(self):
        return file_config(
            self.ed_filename.text(), participant=self.ed_participant.text(),
            mri=self.chk_mri.isChecked(), video=self.cmb_video.currentData(),
            overrides={"left_channels": parse_optional_pair(self.ed_left.text()),
                       "right_channels": parse_optional_pair(self.ed_right.text())})

    def _tab_changed(self, _index):
        if self.tabs.currentWidget() is self.online_tab:
            if not self.isFullScreen():
                self._monitor_geometry = self.saveGeometry()
                self._monitor_window_state = self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized
                self.showFullScreen()
        elif self._monitor_geometry is not None:
            self._leave_monitor_full_screen()

    def _leave_monitor_full_screen(self):
        if self._monitor_geometry is not None:
            geometry = self._monitor_geometry
            self._monitor_geometry = None
            self.restoreGeometry(geometry)
            self.setWindowState(self._monitor_window_state)
        elif self.isFullScreen():
            self.showNormal()

    def _browse(self):
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose one recording", self.ed_filename.text(),
            "Recordings (*.vhdr *.xdf);;BrainVision header (*.vhdr);;XDF (*.xdf)")
        if name:
            self.ed_filename.setText(name)

    def _filename_changed(self):
        brainvision = Path(self.ed_filename.text().strip().strip('"')).suffix.lower() != ".xdf"
        self.chk_mri.setEnabled(brainvision)
        self.cmb_video.setEnabled(brainvision)
        if not brainvision:
            self.chk_mri.setChecked(False)
        self._mri_changed()

    def _mri_changed(self):
        self.lbl_mri.setVisible(self.chk_mri.isChecked())
        self.lbl_mri.setText(
            "Full correction: native-rate AAS → 250 Hz conditioning → cardiac OBS → EMG. "
            "Experimental for EMG: correction can introduce small burst copies and alter rest activity."
            if self.chk_mri.isChecked() else "")

    def _busy(self):
        return any(worker is not None and worker.isRunning()
                   for worker in (self.emg_tab.worker, self.emg_tab.saver))

    def _refresh_busy(self):
        busy = self._busy()
        monitoring = self.online_tab.is_running()
        self.btn_start.setEnabled(not busy and not monitoring)
        self.start_tab.setEnabled(not busy and not monitoring)
        self.online_tab.connect_button.setEnabled(not busy and not monitoring)
        self.emg_tab.btn_save.setEnabled(self.emg_tab.batch is not None and not busy and not monitoring)

    def on_start(self):
        if self._busy() or self.online_tab.is_running():
            return
        self._save_settings()
        if self.emg_tab.start(wait_for_stop=self.chk_wait.isChecked()):
            self._refresh_busy()
            self.tabs.setCurrentWidget(self.emg_tab)

    def _analysis_finished(self, success):
        if success:
            self.tabs.setCurrentWidget(self.emg_tab)

    def _restore(self):
        for key, widget in self._text_fields().items():
            widget.setText(self.settings.value(key, widget.text(), type=str))
        self.chk_mri.setChecked(self.settings.value("mri", False, type=bool))
        self.chk_wait.setChecked(self.settings.value("wait", False, type=bool))
        self.cmb_video.setCurrentIndex(max(0, min(3, self.settings.value("video", 0, type=int))))
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _text_fields(self):
        return {"filename": self.ed_filename, "participant": self.ed_participant,
                "left_channels": self.ed_left, "right_channels": self.ed_right}

    def _save_settings(self):
        self.online_tab.save_settings()
        for key, widget in self._text_fields().items():
            self.settings.setValue(key, widget.text())
        self.settings.setValue("mri", self.chk_mri.isChecked())
        self.settings.setValue("wait", self.chk_wait.isChecked())
        self.settings.setValue("video", self.cmb_video.currentIndex())
        self.settings.setValue("geometry", self._monitor_geometry
                               if self._monitor_geometry is not None else self.saveGeometry())
        self.settings.sync()

    def closeEvent(self, event):
        if self.online_tab.is_running():
            self.online_tab.stop()
            event.ignore()
            QtCore.QTimer.singleShot(200, self.close)
            return
        if self._busy():
            self.lbl_status.setText("Processing or saving is still running. Close when it finishes.")
            if self.emg_tab.worker is not None:
                self.emg_tab.worker.requestInterruption()
            event.ignore()
            return
        self._save_settings()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?")
    parser.add_argument("--mne-mri", action="store_true")
    parser.add_argument("--process", action="store_true", help="Process the supplied file at launch")
    parser.add_argument("--video", type=int, choices=(1, 2, 3))
    args = parser.parse_args(argv)
    qt = QtWidgets.QApplication(sys.argv[:1])
    window = QuasiEMGApp()
    if args.filename:
        window.ed_filename.setText(str(Path(args.filename).resolve()))
        window.chk_mri.setChecked(args.mne_mri)
        window.cmb_video.setCurrentIndex(args.video or 0)
    window.show()
    if args.process:
        QtCore.QTimer.singleShot(0, window.on_start)
    return qt.exec()


if __name__ == "__main__":
    raise SystemExit(main())
