# -*- coding: utf-8 -*-
"""
视频处理工具函数
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


def load_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """
    加载视频帧
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数，None表示加载所有帧
        target_size: 目标尺寸 (width, height)，None表示保持原尺寸
    
    Returns:
        视频帧列表，每帧为RGB图像 (H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR转RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        if target_size is not None:
            frame_rgb = cv2.resize(frame_rgb, target_size)
        
        frames.append(frame_rgb)
        frame_count += 1
        
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    return frames


def resize_frames(
    frames: List[np.ndarray],
    target_size: Tuple[int, int]
) -> List[np.ndarray]:
    """
    调整帧大小
    
    Args:
        frames: 视频帧列表
        target_size: 目标尺寸 (width, height)
    
    Returns:
        调整后的帧列表
    """
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        resized_frames.append(resized)
    return resized_frames


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        视频信息字典，包含fps, width, height, frame_count
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': 0.0  # 将在下面计算
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info


def frame_to_timestamp(frame_id: int, fps: float) -> str:
    """
    将帧ID转换为时间戳字符串
    
    Args:
        frame_id: 帧ID
        fps: 视频帧率
    
    Returns:
        时间戳字符串，格式为 "XX.XXs"
    """
    timestamp = frame_id / fps if fps > 0 else 0.0
    return f"{timestamp:.2f}s"


def save_frame(frame: np.ndarray, output_path: str):
    """
    保存单帧图像
    
    Args:
        frame: 图像帧 (H, W, 3) RGB
        output_path: 输出路径
    """
    # RGB转BGR用于OpenCV保存
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, frame_bgr)

