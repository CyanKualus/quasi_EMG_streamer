"""Wait for one recording's files to settle and its writer to close them."""
from __future__ import annotations

import configparser
import os
from pathlib import Path
import time


def recording_files(path):
    path = Path(path)
    if path.suffix.lower() != ".vhdr":
        return [path]
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(text[text.index("[Common Infos]"):])
    common = cfg["Common Infos"]
    return [path, path.parent / common["DataFile"], path.parent / common["MarkerFile"]]


def writer_closed(path):
    if os.name != "nt":
        return True  # Outside Windows, only the stable-size check is available.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create(str(Path(path).resolve()), 0x80000000, 0, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        return False
    close(handle)
    return True


def snapshot(path):
    files = recording_files(path)
    states = [(str(item.resolve()), item.stat().st_size, item.stat().st_mtime_ns)
              for item in files]
    if any(size <= 0 for _, size, _ in states):
        raise OSError("Recording files are still empty")
    if not all(writer_closed(item) for item in files):
        raise OSError("The recording is still open in another application")
    return tuple(states)


def wait_for_recording(path, progress=None, cancelled=None, *,
                       stable_s=2.0, timeout_s=3600.0):
    emit = progress or (lambda message: None)
    emit("Waiting for recording files to stop changing and close…")
    start = time.monotonic()
    last, since = None, start
    while time.monotonic() - start < timeout_s:
        if cancelled and cancelled():
            raise InterruptedError("Waiting for the recording was cancelled")
        now = time.monotonic()
        try:
            current = snapshot(path)
        except (OSError, ValueError, KeyError, configparser.Error):
            current = None
        if current is None or current != last:
            since, last = now, current
        elif now - since >= stable_s:
            emit("Recording closed; starting EMG processing.")
            return
        time.sleep(.25)
    raise TimeoutError("The recording did not finish within one hour")
