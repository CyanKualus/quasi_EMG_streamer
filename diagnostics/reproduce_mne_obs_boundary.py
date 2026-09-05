"""Minimal synthetic reproducer of the pinned MNE PCA-OBS end-index error.

Expected on MNE 1.13.0.dev334+g64473254e: ValueError, shape (200,) into (201,).
The last peak index (2400) plus half the RR interval (100) equals n_times
(2500). The epoch's inclusive right endpoint needs sample 2500, which does
not exist. The current MNE trimming loop checks '>' instead of '>='.

This deliberately lets the exception propagate. It uses no participant data
and changes neither MNE nor the production application's behavior.
"""
import mne
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    print(f"MNE {mne.__version__}", flush=True)
    data = np.random.default_rng(1).normal(size=(1, 2500)) * 1e-4
    raw = mne.io.RawArray(data, mne.create_info(["F1"], 250, ["emg"]), verbose="error")
    qrs = np.arange(200, 2401, 200, dtype=float) / 250
    with threadpool_limits(limits=1):
        mne.preprocessing.apply_pca_obs(raw, picks=["F1"], qrs_times=qrs, verbose="error")


if __name__ == "__main__":
    main()
