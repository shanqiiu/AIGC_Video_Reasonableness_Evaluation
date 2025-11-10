"""
运动幅度计算工具
"""

import torch
import numpy as np


def calculate_motion_degree(keypoints, video_width, video_height):
    """
    计算每个样本的归一化运动幅度。
    
    Args:
        keypoints: torch.Tensor，形状 [batch_size, 49, 792, 2]，跟踪点坐标
        video_width: int，视频宽度
        video_height: int，视频高度
    
    Returns:
        torch.Tensor: 形状 [batch_size]，表示各样本的归一化运动幅度
    """

    # 计算视频对角线长度，用于归一化
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2, dtype=torch.float32))
    
    # 计算相邻帧之间的欧氏距离
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)  # shape [batch_size, 48, 792]
    
    # 用对角线长度归一化距离，消除分辨率影响
    normalized_distances = distances / diagonal
    
    # 求和得到每个轨迹点的归一化运动距离
    total_normalized_distances = torch.sum(normalized_distances, dim=1)  # shape [batch_size, 792]
    
    # 对所有轨迹点取平均，得到每个样本的运动幅度
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

