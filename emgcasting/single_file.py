"""One finalized recording per invocation, shared by GUI and CLI."""
from pathlib import Path

from . import core
from shared.file_ready import snapshot, wait_for_recording


def analyze_file(config, progress=None, *, wait_for_stop=False, cancelled=None):
    config.validate()
    if len(config.recordings) != 1:
        raise ValueError("Choose exactly one filename")
    path = Path(config.data_dir) / config.recordings[0]
    if wait_for_stop:
        wait_for_recording(path, progress, cancelled)
    before = snapshot(path)
    if progress:
        progress("MNE AAS + cardiac OBS, then EMG…" if config.remove_mri_artifacts
                 else "Processing EMG…")
    batch = core.analyze_batch(config, progress, require_metrics=False)
    if snapshot(path) != before:
        raise RuntimeError("Recording files changed during analysis; rerun after recording stops")
    batch.recordings[0].provenance["source_files"] = [
        {"path": name, "size_bytes": size, "mtime_ns": modified}
        for name, size, modified in before]
    return batch
