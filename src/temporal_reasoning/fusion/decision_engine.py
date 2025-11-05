# -*- coding: utf-8 -*-
"""
融合决策引擎
"""

from typing import List, Dict, Tuple
import numpy as np

from .feature_alignment import align_anomalies_spatially_and_temporally
from .anomaly_fusion import fuse_multimodal_anomalies
from .anomaly_filter import AnomalyFilter
from ..core.config import FusionConfig


class FusionDecisionEngine:
    """融合决策引擎"""
    
    def __init__(
        self,
        config: FusionConfig,
        cotracker_validator = None
    ):
        """
        初始化融合决策引擎
        
        Args:
            config: FusionConfig配置对象
            cotracker_validator: Co-Tracker验证器实例（可选）
        """
        self.config = config
        self.cotracker_validator = cotracker_validator
        self.anomaly_filter = AnomalyFilter(
            enable_cotracker_validation=(cotracker_validator is not None),
            cotracker_validator=cotracker_validator
        )
    
    def fuse(
        self,
        motion_anomalies: List[Dict],
        structure_anomalies: List[Dict],
        physiological_anomalies: List[Dict]
    ) -> List[Dict]:
        """
        融合多模态异常
        
        Args:
            motion_anomalies: 光流异常列表
            structure_anomalies: 结构异常列表
            physiological_anomalies: 生理异常列表
        
        Returns:
            融合后的异常列表
        """
        # 1. 异常对齐
        aligned_anomalies = align_anomalies_spatially_and_temporally(
            motion_anomalies,
            structure_anomalies,
            physiological_anomalies
        )
        
        # 2. 多模态融合
        fused_anomalies = fuse_multimodal_anomalies(
            aligned_anomalies,
            multimodal_confidence_boost=self.config.multimodal_confidence_boost,
            single_modality_threshold=self.config.single_modality_confidence_threshold
        )
        
        # 3. 时序验证
        validated_anomalies = self._validate_temporal_consistency(fused_anomalies)
        
        # 4. 过滤假阳性（使用Co-Tracker验证）
        # 注意：这里需要video_frames或video_tensor，但当前接口没有提供
        # 可以在上层调用时进行过滤
        # filtered_anomalies = self.anomaly_filter.filter_anomalies(
        #     validated_anomalies,
        #     video_frames=video_frames,
        #     video_tensor=video_tensor
        # )
        
        return validated_anomalies
    
    def _validate_temporal_consistency(self, anomalies: List[Dict]) -> List[Dict]:
        """
        验证时序一致性
        
        过滤持续时间过短的异常
        
        Args:
            anomalies: 异常列表
        
        Returns:
            验证后的异常列表
        """
        if not anomalies:
            return []
        
        # 按帧ID分组
        frame_groups = {}
        for anomaly in anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in frame_groups:
                frame_groups[frame_id] = []
            frame_groups[frame_id].append(anomaly)
        
        # 检查连续帧
        validated = []
        frame_ids = sorted(frame_groups.keys())
        
        for i, frame_id in enumerate(frame_ids):
            # 检查前后是否有连续帧
            has_prev = i > 0 and frame_ids[i-1] == frame_id - 1
            has_next = i < len(frame_ids) - 1 and frame_ids[i+1] == frame_id + 1
            
            # 如果异常持续至少min_duration帧，则保留
            if has_prev or has_next or len(frame_groups[frame_id]) >= self.config.min_anomaly_duration_frames:
                validated.extend(frame_groups[frame_id])
        
        return validated
    
    def compute_final_scores(
        self,
        motion_score: float,
        structure_score: float,
        physiological_score: float,
        fused_anomalies: List[Dict]
    ) -> Tuple[float, float]:
        """
        计算最终得分
        
        Args:
            motion_score: 运动得分
            structure_score: 结构得分
            physiological_score: 生理得分
            fused_anomalies: 融合后的异常列表
        
        Returns:
            (motion_reasonableness_score, structure_stability_score)
        """
        # 计算异常惩罚
        motion_anomaly_count = len([
            a for a in fused_anomalies 
            if 'motion' in a.get('modalities', [])
        ])
        
        structure_anomaly_count = len([
            a for a in fused_anomalies 
            if 'structure' in a.get('modalities', [])
        ])
        
        # 应用惩罚
        motion_penalty = min(0.5, motion_anomaly_count * 0.1)
        structure_penalty = min(0.5, structure_anomaly_count * 0.1)
        
        motion_reasonableness = max(0.0, motion_score * (1.0 - motion_penalty))
        structure_stability = max(0.0, structure_score * (1.0 - structure_penalty))
        
        return motion_reasonableness, structure_stability

