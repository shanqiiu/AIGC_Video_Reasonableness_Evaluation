# -*- coding: utf-8 -*-
"""Script entry for single-video blur detection."""

import os
import sys
import argparse


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    # Ensure project root on path
    sys.path.insert(0, _project_root())

    from src.perceptual_quality.blur import BlurDetector, BlurDetectionConfig

    parser = argparse.ArgumentParser(description="Run blur detection on a single video")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--subject_noun", type=str, default="person")
    args = parser.parse_args()

    config = BlurDetectionConfig()
    config.update_device_config("device", args.device)

    detector = BlurDetector(config)
    result = detector.detect(args.video_path, subject_noun=args.subject_noun)

    print("Detection completed.")
    print(result)


if __name__ == "__main__":
    main()


