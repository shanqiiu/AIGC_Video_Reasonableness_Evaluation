from typing import List
from PIL import Image


def load_video_sliding_window(video_file: str, window_size: int = 5) -> List[List[Image.Image]]:
    """Read a video and return frames grouped by a sliding window.

    Each element is a list of PIL Images of length window_size, centered on each frame
    (with padding at the boundaries by repeating edge frames).
    """
    # Local import to avoid hard dependency at module import time
    from decord import VideoReader  # type: ignore

    vr = VideoReader(video_file)
    total_frames = len(vr)
    frames_by_group: List[List[Image.Image]] = []

    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend

    for current_frame in range(total_frames):
        start_frame = max(0, current_frame - left_extend)
        end_frame = min(total_frames, current_frame + right_extend + 1)

        frame_indices = list(range(start_frame, end_frame))

        # Pad to window_size by repeating edge frames
        while len(frame_indices) < window_size:
            if start_frame == 0:
                frame_indices.append(frame_indices[-1])
            else:
                frame_indices.insert(0, frame_indices[0])

        frames_np = vr.get_batch(frame_indices).asnumpy()

        if current_frame < left_extend:
            frames_by_group.append([Image.fromarray(frames_np[0])] * window_size)
        else:
            frames_by_group.append([Image.fromarray(frame) for frame in frames_np])

    return frames_by_group


