"""Validate production monitoring against saved replay results and real TCP RDA.

The server replays existing recordings locally; it does not contact an amplifier.
Run with the app's .venv. Writes numerical results and an offscreen GUI capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mne
import numpy as np
from PyQt6 import QtCore, QtWidgets
from threadpoolctl import threadpool_limits

from app import QuasiEMGApp
from emgcasting.core import ProcessingConfig, envelope_trace, filter_emg
from emgcasting.online import OnlineConfig, OnlineProcessor

DATA = Path(r"D:\ExpData\MEG\Quasi fMRI\data\Other\fMRI_test\Unfitered")


def packet(kind, body=b""):
    return bytes.fromhex("8e45584396c9864caf4a98bbf6c91450") + struct.pack("<II", 24 + len(body), kind) + body


def replay_tcp(listener, source, fs, finished, release):
    """Sample-major float32 wire units with unequal resolutions to test scaling."""
    try:
        with listener.accept()[0] as conn:
            res = np.array([.5, 2.])
            body = struct.pack("<Id", 2, 1e6 / fs) + res.astype("<f8").tobytes() + b"F1\0F2\0"
            conn.sendall(packet(1, body))
            chunk = round(.1 * fs)
            for index, start in enumerate(range(0, source.shape[1], chunk)):
                values = (source[:, start:start + chunk].T / (res * 1e-6)).astype("<f4")
                body = struct.pack("<III", index + 1, len(values), 0) + values.tobytes()
                conn.sendall(packet(4, body))
                if release.wait(.005):
                    return
            finished.set()
            release.wait(10)
    except OSError:
        pass
    finally:
        listener.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "online_monitor_validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    demo = None
    with threadpool_limits(limits=1):
        for number in (1, 2, 3):
            path = args.data_dir / f"in1948_block{number:02d}.vhdr"
            raw = mne.io.read_raw_brainvision(path, preload=False, verbose="error")
            source = raw.get_data(picks=["F1", "F2"])
            fs = raw.info["sfreq"]
            raw.close()
            processor = OnlineProcessor(OnlineConfig(), fs)
            frames = []
            tick = time.perf_counter()
            # Deliberately avoid packet boundaries aligned to native TRs.
            for start in range(0, source.shape[1], 733):
                frames.extend(processor.feed(source[:, start:start + 733]))
            elapsed = time.perf_counter() - tick
            with np.load(ROOT / "output" / "no_cardiac_comparison" / f"p{number}_no_obs_traces.npz") as stored:
                reference = stored["aas8_0_buffer10_lag2.5"]
            first = round(frames[0].start_s * 250)
            actual = np.concatenate([f.conditioned_v for f in frames], axis=1)
            expected = reference[:, first:first + actual.shape[1]]
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-11)
            emg = np.concatenate([f.filtered_mv["right"] for f in frames])
            expected_emg = filter_emg(expected[0] - expected[1], 250, ProcessingConfig(".", ["verify"]))
            np.testing.assert_allclose(emg, expected_emg, atol=1e-12, rtol=1e-10)
            env = np.concatenate([f.envelope_mv2["right"] for f in frames])
            expected_env, _ = envelope_trace(emg, 250, ProcessingConfig(".", ["verify"]))
            np.testing.assert_allclose(env, expected_env[:len(env)], atol=1e-12, rtol=1e-10)
            row = {"recording": path.name, "native_hz": fs, "frames": len(frames),
                   "first_output_acquisition_s": frames[0].acquisition_end_s,
                   "first_output_start_s": frames[0].start_s,
                   "conditioned_max_abs_error_v": float(np.max(np.abs(actual - expected))),
                   "emg_max_abs_error_mv": float(np.max(np.abs(emg - expected_emg))),
                   "envelope_max_abs_error_mv2": float(np.max(np.abs(env - expected_env[:len(env)]))),
                   "processing_median_ms": float(np.median([f.compute_s for f in frames]) * 1000),
                   "processing_p95_ms": float(np.percentile([f.compute_s for f in frames], 95) * 1000),
                   "replay_wall_s": elapsed}
            print(json.dumps(row), flush=True)
            rows.append(row)
            if number == 1:
                demo = source[:, :round(90 * fs)].copy()
                demo_fs = fs
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(args.output / "gui.ini"), QtCore.QSettings.Format.IniFormat)
    settings.clear()
    window = QuasiEMGApp(settings)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(1280, 920)
    window.show()
    tab = window.online_tab
    tab.left.setText("F1, F2")
    tab.right.setText("F2, F1")
    window.tabs.setCurrentWidget(tab)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    tab.port.setCurrentText(str(listener.getsockname()[1]))
    finished, release = threading.Event(), threading.Event()
    server = threading.Thread(target=replay_tcp, args=(listener, demo, demo_fs, finished, release), daemon=True)
    server.start()
    tab.start()
    try:
        deadline = time.monotonic() + 50
        while time.monotonic() < deadline:
            qt.processEvents()
            if tab.latest_frame is not None and tab.latest_frame.acquisition_end_s >= 90:
                break
            if not tab.is_running():
                raise RuntimeError(tab.status.text())
            time.sleep(.01)
        else:
            raise RuntimeError("Timed out waiting for 90 s TCP replay")
        tab._poll()
        assert tab.stream_info.sfreq == 5000
        assert tab.curves["left"].xData.size > 500
        assert tab.curves["right"].xData.size > 500
        # Compare the actual float32 TCP path to the stored double-precision replay.
        with np.load(ROOT / "output" / "no_cardiac_comparison" / "p1_no_obs_traces.npz") as stored:
            reference = stored["aas8_0_buffer10_lag2.5"]
        actual = np.concatenate([f.conditioned_v for f in tab.frames], axis=1)
        first = round(tab.frames[0].start_s * 250)
        error = float(np.max(np.abs(actual - reference[:, first:first + actual.shape[1]])))
        assert error < 1e-8, error  # Float32 wire quantization; <=0.01 microvolt.
        screenshot = args.output / "online_monitor.png"
        # Explicitly label the review capture; the product status is still tested above.
        tab.status.setText("LOCAL RECORDED-DATA REPLAY · 90 s received · BrainAmp hardware not connected")
        qt.processEvents()
        window.grab().save(str(screenshot))
        report = {"mne": mne.__version__, "physical_amplifier_tested": False,
                  "recordings": rows, "tcp_float32_max_abs_error_v": error,
                  "tcp_replayed_s": 90, "screenshot": str(screenshot)}
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"tcp_float32_max_abs_error_v": error, "screenshot": str(screenshot)}), flush=True)
    finally:
        tab.stop()
        release.set()
        if tab.worker:
            tab.worker.wait(5000)
        server.join(2)
        qt.processEvents()
        window.close()


if __name__ == "__main__":
    main()
