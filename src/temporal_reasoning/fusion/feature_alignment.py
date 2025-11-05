# -*- coding: utf-8 -*-
"""
特征对齐模块
"""

from typing import List, Dict
import numpy as np


def align_anomalies_spatially_and_temporally(
    motion_anomalies: List[Dict],
    structure_anomalies: List[Dict],
    physiological_anomalies: List[Dict],
    spatial_threshold: float = 0.1,  # 空间阈值（归一化坐标）
    temporal_threshold: float = 0.5   # 时间阈值（秒）
) -> List[List[Dict]]:
    """
    在空间和时间上对齐异常
    
    Args:
        motion_anomalies: 光流异常列表
        structure_anomalies: 结构异常列表
        physiological_anomalies: 生理异常列表
        spatial_threshold: 空间对齐阈值
        temporal_threshold: 时间对齐阈值（秒）
    
    Returns:
        对齐后的异常组列表，每个组包含多个模态的异常
    """
    # 将所有异常合并并标记来源
    all_anomalies = []
    
    for anomaly in motion_anomalies:
        anomaly['modality'] = 'motion'
        all_anomalies.append(anomaly)
    
    for anomaly in structure_anomalies:
        anomaly['modality'] = 'structure'
        all_anomalies.append(anomaly)
    
    for anomaly in physiological_anomalies:
        anomaly['modality'] = 'physiological'
        all_anomalies.append(anomaly)
    
    if not all_anomalies:
        return []
    
    # 按时间戳排序
    all_anomalies.sort(key=lambda x: x.get('frame_id', 0))
    
    # 分组：时间相近的异常归为一组
    groups = []
    current_group = []
    
    for anomaly in all_anomalies:
        if not current_group:
            current_group = [anomaly]
        else:
            # 检查时间是否相近
            last_frame = current_group[-1].get('frame_id', 0)
            current_frame = anomaly.get('frame_id', 0)
            
            # 简化：如果帧ID相差小于threshold，归为一组
            frame_threshold = int(temporal_threshold * 30)  # 假设30fps
            
            if abs(current_frame - last_frame) <= frame_threshold:
                current_group.append(anomaly)
            else:
                groups.append(current_group)
                current_group = [anomaly]
    
    if current_group:
        groups.append(current_group)
    
    return groups

