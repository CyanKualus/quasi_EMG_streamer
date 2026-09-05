"""Exercise GUI worker, plots, trial browser and saving on the reviewed p1 file."""
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6 import QtCore, QtWidgets
from app import QuasiEMGApp
from emg_view import TrialsWindow


def main():
    output = ROOT / "output/verification"
    output.mkdir(parents=True, exist_ok=True)
    qt = QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(output / "desktop_check.ini"), QtCore.QSettings.Format.IniFormat)
    window = QuasiEMGApp(settings)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    failures = []
    window.emg_tab._report_failure = lambda title, details: failures.append((title, details))
    window.ed_filename.setText(
        r"D:\ExpData\MEG\Quasi fMRI\data\Other\fMRI_test\in1948_MRbvCBbv_p1_EMG.vhdr")
    window.chk_mri.setChecked(False)
    window.chk_wait.setChecked(False)
    window.show()
    qt.processEvents()
    window.grab().save(str(output / "start.png"))
    window.on_start()
    def wait_until(predicate):
        deadline = time.monotonic() + 90
        while not predicate():
            qt.processEvents()
            if failures:
                raise AssertionError(failures)
            if time.monotonic() > deadline:
                raise TimeoutError("Desktop operation did not finish")
            time.sleep(.02)
        qt.processEvents()
    wait_until(lambda: window.emg_tab.batch is not None and not window._busy())
    assert window.emg_tab.cards_layout.count() == 2
    assert window.emg_tab.batch.recordings[0].hands[0].n_trials == 20
    window.grab().save(str(output / "analysis.png"))
    card = window.emg_tab.cards_layout.itemAt(0).widget()
    trial_window = TrialsWindow(card.result.recording, card.result.hands[0],
                                card.config, window)
    trial_window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    trial_window.show()
    qt.processEvents()
    dialogs = [w for w in qt.topLevelWidgets() if isinstance(w, TrialsWindow)]
    assert len(dialogs) == 1
    dialogs[0].grab().save(str(output / "trials.png"))
    dialogs[0].close()
    window.emg_tab._save()
    wait_until(lambda: not window._busy() and window.emg_tab.btn_save.isEnabled())
    batch = window.emg_tab.batch
    assert Path(batch.summary_csv).is_file()
    assert (Path(batch.output_dir) / "processing.json").is_file()
    hand = batch.recordings[0].hands[0]
    assert Path(hand.trial_metrics_csv).is_file()
    assert len(list(Path(hand.trial_dir).glob("trial_*.png"))) == 20
    assert window.emg_tab.btn_open.isEnabled()
    window.close()
    qt.processEvents()
    print("Desktop: worker completed, one EMG card, 20 browsable trials, figures/CSV/JSON saved")


if __name__ == "__main__":
    main()
