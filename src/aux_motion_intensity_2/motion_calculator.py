"""
运动幅度计算工具
"""

import torch
import numpy as np


def calculate_motion_degree(keypoints, video_width, video_height):
    """
    Calculate the normalized motion amplitude for each batch sample
    
    Parameters:
    keypoints: torch.Tensor, shape [batch_size, 49, 792, 2]
    video_width: int, width of the video
    video_height: int, height of the video
    
    Returns:
    motion_amplitudes: torch.Tensor, shape [batch_size], containing the normalized motion amplitude for each batch sample
    """

    # Calculate the length of the video diagonal
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2, dtype=torch.float32))
    
    # Compute the Euclidean distance between adjacent frames
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)  # shape [batch_size, 48, 792]
    
    # Normalize the distances by the diagonal length to eliminate resolution effects
    normalized_distances = distances / diagonal
    
    # Sum the normalized distances to get the total normalized motion distance for each keypoint
    total_normalized_distances = torch.sum(normalized_distances, dim=1)  # shape [batch_size, 792]
    
    # Compute the normalized motion amplitude for each batch sample (mean of total normalized motion distance for all points)
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)  # shape [batch_size]
    
    return motion_amplitudes


def is_mask_suitable_for_tracking(mask, video_width, video_height, grid_size, min_area_ratio=0.001):
    """
    判断分割掩码是否适合进行点跟踪。
    
    Parameters:
    mask: torch.Tensor, 掩码
    video_width: int, 视频宽度
    video_height: int, 视频高度
    grid_size: int, 规则网格大小
    min_area_ratio: float, 掩码覆盖的最小面积占比阈值
    
    Returns:
    bool: 是否适合进行跟踪
    """
    mask_area = torch.sum(mask > 0).item()
    total_area = video_width * video_height
    area_ratio = mask_area / total_area
    
    # 掩码面积过小则不适合
    if area_ratio < min_area_ratio:
        return False
    
    # 掩码像素数量需达到一定规模，避免稀疏导致无法稳定跟踪
    # 至少需要约 (grid_size/2)^2 个像素
    min_pixels_needed = (grid_size // 2) ** 2
    if mask_area < min_pixels_needed:
        return False
    
    return True

