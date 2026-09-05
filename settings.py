"""Shared configuration for the desktop and single-file command line."""
from dataclasses import fields
import json
from pathlib import Path

from emgcasting.core import ProcessingConfig

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "quasi_emg_settings.json"


def load_settings(path=SETTINGS_PATH):
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    allowed = {item.name for item in fields(ProcessingConfig)} - {
        "data_dir", "recordings", "participant", "file_type", "video",
        "remove_mri_artifacts", "single_file_output"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown EMG settings in {path}: {sorted(unknown)}")
    return values


def file_config(filename, *, participant="", mri=False, video=None,
                overrides=None, settings_path=SETTINGS_PATH):
    text = str(filename).strip().strip('"')
    if not text:
        raise ValueError("Choose one recording filename")
    path = Path(text).expanduser().resolve()
    file_type = {".vhdr": "brainvision", ".xdf": "xdf"}.get(path.suffix.lower())
    if file_type is None:
        raise ValueError("Choose a BrainVision .vhdr header or an XDF .xdf file")
    values = load_settings(settings_path)
    values.update(overrides or {})
    for key in ("left_channels", "right_channels"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    config = ProcessingConfig(
        data_dir=str(path.parent), recordings=[path.name],
        participant=participant.strip() or path.stem.split("_")[0],
        file_type=file_type, video=video, remove_mri_artifacts=mri,
        single_file_output=True, **values)
    config.validate()
    return config
