# -*- coding: utf-8 -*-
"""
运动平滑度计算
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_flow_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    计算光流幅值
    
    Args:
        u: x方向光流 (H, W)
        v: y方向光流 (H, W)
    
    Returns:
        光流幅值 (H, W)
    """
    return np.sqrt(u**2 + v**2)


def compute_flow_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    计算光流方向
    
    Args:
        u: x方向光流 (H, W)
        v: y方向光流 (H, W)
    
    Returns:
        光流方向（弧度） (H, W)
    """
    return np.arctan2(v, u)


def compute_motion_smoothness(
    optical_flows: List[Tuple[np.ndarray, np.ndarray]]
) -> List[float]:
    """
    计算运动平滑度
    
    Args:
        optical_flows: 光流序列，每个元素为(u, v)元组
    
    Returns:
        平滑度分数列表，每个元素对应相邻帧对的平滑度 (0-1)
    """
    if len(optical_flows) < 2:
        return []
    
    smoothness_scores = []
    
    for i in range(len(optical_flows) - 1):
        u1, v1 = optical_flows[i]
        u2, v2 = optical_flows[i+1]
        
        # 计算光流差异
        du = u2 - u1
        dv = v2 - v1
        flow_diff = np.sqrt(du**2 + dv**2)
        
        # 归一化为平滑度分数 (0-1)
        # 使用95分位数作为归一化基准，避免异常值影响
        max_diff = np.percentile(flow_diff, 95)
        if max_diff > 0:
            smoothness = 1.0 - np.clip(flow_diff / max_diff, 0, 1)
        else:
            smoothness = np.ones_like(flow_diff)
        
        # 计算平均平滑度
        smoothness_scores.append(float(np.mean(smoothness)))
    
    return smoothness_scores


def compute_flow_statistics(
    optical_flows: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, float]:
    """
    计算光流统计信息
    
    Args:
        optical_flows: 光流序列
    
    Returns:
        统计信息字典，包含：
        - mean_magnitude: 平均光流幅值
        - max_magnitude: 最大光流幅值
        - std_magnitude: 光流幅值标准差
        - mean_direction: 平均光流方向
    """
    if not optical_flows:
        return {
            'mean_magnitude': 0.0,
            'max_magnitude': 0.0,
            'std_magnitude': 0.0,
            'mean_direction': 0.0
        }
    
    all_magnitudes = []
    all_directions = []
    
    for u, v in optical_flows:
        mag = compute_flow_magnitude(u, v)
        direction = compute_flow_direction(u, v)
        
        all_magnitudes.append(mag.flatten())
        all_directions.append(direction.flatten())
    
    all_magnitudes = np.concatenate(all_magnitudes)
    all_directions = np.concatenate(all_directions)
    
    return {
        'mean_magnitude': float(np.mean(all_magnitudes)),
        'max_magnitude': float(np.max(all_magnitudes)),
        'std_magnitude': float(np.std(all_magnitudes)),
        'mean_direction': float(np.mean(all_directions))
    }

