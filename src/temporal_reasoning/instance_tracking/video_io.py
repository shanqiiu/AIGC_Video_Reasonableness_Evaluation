from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image


def extract_frames_from_video(video_path: str, jpeg_quality: int = 95) -> Tuple[List[Image.Image], List[str]]:
    """
    Read frames from `video_path` and return them as RGB PIL images.

    Args:
        video_path: Path to the input video.
        jpeg_quality: JPEG encode quality used to mimic the original script.

    Returns:
        frames: List of PIL RGB frames.
        frame_names: Zero-padded frame index strings.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    frame_names: List[str] = []
    frame_count = 0

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        frame_names.append(f"{frame_count:04d}")

    cap.release()
    return frames, frame_names

