"""
�˶����ȼ��㹤��
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
    �жϷָ������Ƿ��ʺϽ��е���١�
    
    Parameters:
    mask: torch.Tensor, ����
    video_width: int, ��Ƶ���
    video_height: int, ��Ƶ�߶�
    grid_size: int, ���������С
    min_area_ratio: float, ���븲�ǵ���С���ռ����ֵ
    
    Returns:
    bool: �Ƿ��ʺϽ��и���
    """
    mask_area = torch.sum(mask > 0).item()
    total_area = video_width * video_height
    area_ratio = mask_area / total_area
    
    # ���������С���ʺ�
    if area_ratio < min_area_ratio:
        return False
    
    # ��������������ﵽһ����ģ������ϡ�赼���޷��ȶ�����
    # ������ҪԼ (grid_size/2)^2 ������
    min_pixels_needed = (grid_size // 2) ** 2
    if mask_area < min_pixels_needed:
        return False
    
    return True

