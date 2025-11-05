# -*- coding: utf-8 -*-
"""
异常融合模块
"""

from typing import List, Dict


def fuse_multimodal_anomalies(
    aligned_anomalies: List[List[Dict]],
    multimodal_confidence_boost: float = 1.2,
    single_modality_threshold: float = 0.8
) -> List[Dict]:
    """
    融合多模态异常
    
    Args:
        aligned_anomalies: 对齐后的异常组列表
        multimodal_confidence_boost: 多模态一致时的置信度提升倍数
        single_modality_threshold: 单模态异常的最小置信度阈值
    
    Returns:
        融合后的异常列表
    """
    fused_anomalies = []
    
    for anomaly_group in aligned_anomalies:
        if not anomaly_group:
            continue
        
        # 统计各模态的异常
        modalities = [a.get('modality', 'unknown') for a in anomaly_group]
        modality_set = set(modalities)
        
        # 计算融合置信度
        confidences = [a.get('confidence', 0.0) for a in anomaly_group]
        base_confidence = max(confidences) if confidences else 0.0
        
        # 多模态一致时提升置信度
        if len(modality_set) >= 2:
            fused_confidence = min(1.0, base_confidence * multimodal_confidence_boost)
        else:
            # 单模态异常需要达到阈值
            if base_confidence < single_modality_threshold:
                continue  # 过滤低置信度单模态异常
            fused_confidence = base_confidence
        
        # 确定异常类型
        anomaly_type = _determine_anomaly_type(anomaly_group)
        
        # 确定严重程度
        severity = _determine_severity(fused_confidence, len(modality_set))
        
        # 使用第一个异常的时间戳和位置信息
        primary_anomaly = anomaly_group[0]
        
        fused_anomaly = {
            'type': anomaly_type,
            'timestamp': primary_anomaly.get('timestamp', ''),
            'frame_id': primary_anomaly.get('frame_id', 0),
            'confidence': fused_confidence,
            'description': _generate_description(anomaly_group),
            'modalities': list(modality_set),
            'severity': severity,
            'location': primary_anomaly.get('location', {})
        }
        
        fused_anomalies.append(fused_anomaly)
    
    return fused_anomalies


def _determine_anomaly_type(anomaly_group: List[Dict]) -> str:
    """
    确定异常类型
    
    Args:
        anomaly_group: 异常组
    
    Returns:
        异常类型字符串
    """
    # 统计各类型异常
    type_counts = {}
    for anomaly in anomaly_group:
        anomaly_type = anomaly.get('type', 'unknown')
        type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
    
    # 返回最常见的类型
    if type_counts:
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    return 'unknown'


def _determine_severity(confidence: float, modality_count: int) -> str:
    """
    确定严重程度
    
    Args:
        confidence: 置信度
        modality_count: 模态数量
    
    Returns:
        严重程度字符串
    """
    if confidence >= 0.9 or modality_count >= 3:
        return 'Critical'
    elif confidence >= 0.7 or modality_count >= 2:
        return 'Moderate'
    else:
        return 'Minor'


def _generate_description(anomaly_group: List[Dict]) -> str:
    """
    生成异常描述
    
    Args:
        anomaly_group: 异常组
    
    Returns:
        描述字符串
    """
    if len(anomaly_group) == 1:
        return anomaly_group[0].get('description', '检测到异常')
    
    # 多模态异常
    modalities = set(a.get('modality', 'unknown') for a in anomaly_group)
    modality_names = {
        'motion': '运动',
        'structure': '结构',
        'physiological': '生理'
    }
    
    modality_str = '、'.join([modality_names.get(m, m) for m in modalities])
    return f"多模态异常检测：{modality_str}"

