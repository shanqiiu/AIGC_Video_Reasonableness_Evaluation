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


def compute_flow_divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    计算光流散度（用于检测局部运动异常）
    
    Args:
        u: x方向光流 (H, W)
        v: y方向光流 (H, W)
    
    Returns:
        光流散度 (H, W)
    """
    # 计算梯度
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    
    # 散度 = du/dx + dv/dy
    divergence = du_dx + dv_dy
    return divergence


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


def detect_motion_discontinuities(
    optical_flows: List[Tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
    fps: float = 30.0,
    masks: Optional[List[Optional[np.ndarray]]] = None
) -> List[Dict]:
    """
    检测运动突变
    
    Args:
        optical_flows: 光流序列
        threshold: 突变阈值（光流幅值变化率），默认0.5（使用mask时推荐0.5-1.0）
        fps: 视频帧率，用于计算时间戳
        masks: 可选的mask序列，每帧一个mask（bool或0/1数组），用于只计算mask区域内的光流变化率
               如果为None，则计算整个视频的光流变化率
    
    Returns:
        异常列表，每个元素包含：
        - type: 异常类型
        - frame_id: 帧ID
        - timestamp: 时间戳字符串
        - confidence: 置信度
        - description: 描述
    """
    if len(optical_flows) < 2:
        return []
    
    anomalies = []
    
    # 最小光流幅值阈值，用于过滤静止区域（避免分母过小导致数值爆炸）
    min_magnitude_threshold = 0.5  # 像素单位
    
    for i in range(len(optical_flows) - 1):
        u1, v1 = optical_flows[i]
        u2, v2 = optical_flows[i+1]
        
        # 计算光流幅值
        flow1_mag = compute_flow_magnitude(u1, v1)
        flow2_mag = compute_flow_magnitude(u2, v2)
        
        # 如果提供了mask，只计算mask区域内的光流
        if masks is not None and i < len(masks):
            mask1 = masks[i]
            mask2 = masks[i + 1] if (i + 1) < len(masks) else None
            
            # 使用mask1（前一帧的mask）来过滤光流
            if mask1 is not None:
                # 确保mask是bool类型
                if mask1.dtype != bool:
                    mask1 = mask1 > 0
                
                # 确保mask形状与光流一致
                if mask1.shape != flow1_mag.shape:
                    # 如果形状不匹配，尝试调整
                    import cv2
                    h, w = flow1_mag.shape
                    mask1 = cv2.resize(mask1.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # 只考虑mask区域内的光流
                valid_flow1 = flow1_mag[mask1]
                valid_flow2 = flow2_mag[mask1]
                
                if len(valid_flow1) == 0:
                    # 如果mask区域为空，跳过
                    continue
                
                # 计算mask区域内的平均光流幅值
                mean_flow1 = float(np.mean(valid_flow1))
                mean_flow2 = float(np.mean(valid_flow2))
                
                # 计算变化率（使用平均幅值，更稳定）
                if mean_flow1 > min_magnitude_threshold:
                    # 当平均幅值足够大时，使用相对变化率
                    mean_change_rate = abs(mean_flow2 - mean_flow1) / mean_flow1
                else:
                    # 当平均幅值很小时，使用绝对差值归一化
                    mean_change_rate = abs(mean_flow2 - mean_flow1) / min_magnitude_threshold
                
                # 计算有效运动区域的变化率（排除静止区域）
                valid_mask = valid_flow1 > min_magnitude_threshold
                if np.any(valid_mask):
                    valid_flow1_filtered = valid_flow1[valid_mask]
                    valid_flow2_filtered = valid_flow2[valid_mask]
                    valid_change_rate = np.abs(valid_flow2_filtered - valid_flow1_filtered) / valid_flow1_filtered
                    # 使用95分位数而不是最大值，避免异常值影响
                    percentile_change = float(np.percentile(valid_change_rate, 95))
                    mean_valid_change = float(np.mean(valid_change_rate))
                else:
                    # 如果所有区域都接近静止，使用平均变化率
                    percentile_change = mean_change_rate
                    mean_valid_change = mean_change_rate
                
                # 综合判断：使用平均变化率和95分位数的较大值
                overall_change = max(mean_change_rate, percentile_change)
                
            else:
                # 如果mask为None，跳过该帧
                continue
        else:
            # 没有提供mask，使用整个视频的光流（保持原有逻辑但改进计算方式）
            mean_flow1 = float(np.mean(flow1_mag))
            mean_flow2 = float(np.mean(flow2_mag))
            
            # 方法1：使用平均幅值的变化率
            if mean_flow1 > min_magnitude_threshold:
                mean_change_rate = abs(mean_flow2 - mean_flow1) / mean_flow1
            else:
                mean_change_rate = abs(mean_flow2 - mean_flow1) / min_magnitude_threshold
            
            # 方法2：计算有效运动区域的变化率
            valid_mask = flow1_mag > min_magnitude_threshold
            if np.any(valid_mask):
                valid_flow1 = flow1_mag[valid_mask]
                valid_flow2 = flow2_mag[valid_mask]
                valid_change_rate = np.abs(valid_flow2 - valid_flow1) / valid_flow1
                percentile_change = float(np.percentile(valid_change_rate, 95))
                mean_valid_change = float(np.mean(valid_change_rate))
            else:
                abs_diff = abs(mean_flow2 - mean_flow1)
                percentile_change = abs_diff / min_magnitude_threshold
                mean_valid_change = percentile_change
            
            overall_change = max(mean_change_rate, percentile_change)
            mean_valid_change = mean_valid_change
        
        # 检测突变
        if overall_change > threshold:
            # 计算置信度（限制在合理范围内）
            confidence = min(1.0, overall_change / threshold)
            
            # 计算时间戳
            timestamp = f"{i / fps:.2f}s"
            
            anomalies.append({
                'type': 'motion_discontinuity',
                'frame_id': i,
                'timestamp': timestamp,
                'confidence': confidence,
                'description': f"第{i}帧检测到运动突变，变化率: {overall_change:.3f}",
                'max_change_rate': overall_change,  # 保持字段名兼容，但实际是综合变化率
                'mean_change_rate': mean_valid_change
            })
    
    return anomalies


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

