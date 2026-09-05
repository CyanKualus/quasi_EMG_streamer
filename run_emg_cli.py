"""Process one closed recording, optionally removing MRI artifacts with MNE."""
import argparse
import sys

from emgcasting.core import parse_optional_pair, save_batch_outputs
from emgcasting.single_file import analyze_file
from settings import file_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", help="One .vhdr or .xdf file, including its extension")
    parser.add_argument("--mne-mri", action="store_true", help="Full MNE AAS + cardiac OBS")
    parser.add_argument("--video", type=int, choices=(1, 2, 3), help="Override automatic video lookup")
    parser.add_argument("--participant", default="")
    parser.add_argument("--left-emg", help="Positive,negative electrodes; empty disables this hand")
    parser.add_argument("--right-emg")
    parser.add_argument("--ecg", help="ECG channel for cardiac OBS (default: ECG)")
    parser.add_argument("--tr", type=float, help="Stable scanner period in seconds (default: 2.5)")
    parser.add_argument("--wait-for-stop", action="store_true", help="Wait for this file to close")
    parser.add_argument("--no-trial-figures", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    overrides = {}
    for key, value in (("left_channels", args.left_emg), ("right_channels", args.right_emg)):
        if value is not None:
            overrides[key] = parse_optional_pair(value)
    for key, value in (("ecg_channel", args.ecg), ("mri_tr_s", args.tr),
                       ("output_root", args.output_root)):
        if value is not None:
            overrides[key] = value
    if args.no_trial_figures:
        overrides["trial_figures"] = False
    try:
        config = file_config(args.filename, participant=args.participant,
                             mri=args.mne_mri, video=args.video, overrides=overrides)
        batch = analyze_file(config, print, wait_for_stop=args.wait_for_stop)
        save_batch_outputs(batch, config, print)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"EMG processing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
