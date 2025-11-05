# -*- coding: utf-8 -*-
"""
异常过滤模块
用于过滤假阳性异常，包括边缘消失、小尺寸消失等
"""

from typing import List, Dict, Optional
import numpy as np
import torch


class AnomalyFilter:
    """异常过滤器"""
    
    def __init__(
        self,
        enable_cotracker_validation: bool = True,
        cotracker_validator = None
    ):
        """
        初始化异常过滤器
        
        Args:
            enable_cotracker_validation: 是否启用Co-Tracker验证
            cotracker_validator: Co-Tracker验证器实例
        """
        self.enable_cotracker_validation = enable_cotracker_validation
        self.cotracker_validator = cotracker_validator
    
    def filter_anomalies(
        self,
        anomalies: List[Dict],
        video_frames: Optional[List[np.ndarray]] = None,
        video_tensor: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        过滤异常列表，移除假阳性
        
        Args:
            anomalies: 异常列表
            video_frames: 视频帧列表（numpy数组）
            video_tensor: 视频tensor (1, T, C, H, W)（如果提供，用于Co-Tracker验证）
        
        Returns:
            过滤后的异常列表
        """
        filtered_anomalies = []
        
        for anomaly in anomalies:
            # 检查是否需要验证
            if not self._needs_validation(anomaly):
                filtered_anomalies.append(anomaly)
                continue
            
            # 使用Co-Tracker验证
            if self.enable_cotracker_validation and self.cotracker_validator is not None:
                is_valid = self._validate_with_cotracker(
                    anomaly, video_frames, video_tensor
                )
                if is_valid:
                    filtered_anomalies.append(anomaly)
                else:
                    # 添加过滤信息
                    anomaly['filtered'] = True
                    anomaly['filter_reason'] = 'cotracker_validation_failed'
            else:
                # 如果没有Co-Tracker，抛出异常
                raise RuntimeError(
                    "Co-Tracker验证器未初始化\n"
                    "请确保Co-Tracker验证器已正确初始化，或设置 enable_cotracker_validation=False 禁用验证"
                )
        
        return filtered_anomalies
    
    def _needs_validation(self, anomaly: Dict) -> bool:
        """检查异常是否需要验证"""
        anomaly_type = anomaly.get('type', '')
        
        # 结构相关异常需要验证
        structure_types = [
            'structural_disappearance',
            'structural_appearance',
            'structure'
        ]
        
        return any(st in anomaly_type.lower() for st in structure_types)
    
    def _validate_with_cotracker(
        self,
        anomaly: Dict,
        video_frames: Optional[List[np.ndarray]] = None,
        video_tensor: Optional[torch.Tensor] = None
    ) -> bool:
        """使用Co-Tracker验证异常"""
        if self.cotracker_validator is None:
            raise RuntimeError(
                "Co-Tracker验证器未初始化\n"
                "请确保Co-Tracker验证器已正确初始化"
            )
        
        # 获取视频tensor
        if video_tensor is None and video_frames is not None:
            video_tensor = self._frames_to_tensor(video_frames)
        
        if video_tensor is None:
            raise ValueError(
                "无法获取视频tensor进行Co-Tracker验证\n"
                "请提供 video_frames 或 video_tensor 参数"
            )
        
        # 获取异常信息
        anomaly_type = anomaly.get('type', '').lower()
        frame_id = anomaly.get('frame_id', 0)
        location = anomaly.get('location', {})
        mask = location.get('mask')
        
        if mask is None:
            raise ValueError(
                f"异常 {anomaly.get('type', 'unknown')} 缺少掩码信息\n"
                f"请确保异常包含 'location.mask' 字段"
            )
        
        # 获取视频尺寸
        _, _, _, video_height, video_width = video_tensor.shape
        
        # 根据异常类型进行验证
        if 'disappear' in anomaly_type or 'vanish' in anomaly_type:
            is_valid, validation_info = self.cotracker_validator.validate_disappearance(
                video_tensor,
                mask,
                frame_id,
                video_width,
                video_height
            )
        elif 'appear' in anomaly_type or 'emerge' in anomaly_type:
            is_valid, validation_info = self.cotracker_validator.validate_appearance(
                video_tensor,
                mask,
                frame_id,
                video_width,
                video_height
            )
        else:
            # 其他类型也需要验证
            raise ValueError(
                f"未知的异常类型: {anomaly_type}\n"
                f"支持的异常类型: structural_disappearance, structural_appearance"
            )
        
        # 保存验证信息
        anomaly['validation_info'] = validation_info
        
        return is_valid
    
    
    def _frames_to_tensor(self, video_frames: List[np.ndarray]) -> torch.Tensor:
        """将视频帧列表转换为tensor"""
        if not video_frames:
            raise ValueError("视频帧列表为空")
        
        try:
            # 转换为tensor
            frames_array = np.stack(video_frames)  # (T, H, W, 3)
            frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
            
            # 归一化到[0, 1]
            frames_tensor = frames_tensor / 255.0
            
            return frames_tensor
        except Exception as e:
            raise RuntimeError(
                f"无法转换视频帧为tensor: {e}\n"
                f"请确保视频帧格式正确"
            )


def filter_false_positives(
    anomalies: List[Dict],
    cotracker_validator = None,
    video_frames: Optional[List[np.ndarray]] = None,
    video_tensor: Optional[torch.Tensor] = None
) -> List[Dict]:
    """
    过滤假阳性异常的便捷函数
    
    Args:
        anomalies: 异常列表
        cotracker_validator: Co-Tracker验证器
        video_frames: 视频帧列表
        video_tensor: 视频tensor
    
    Returns:
        过滤后的异常列表
    """
    filter_instance = AnomalyFilter(
        enable_cotracker_validation=(cotracker_validator is not None),
        cotracker_validator=cotracker_validator
    )
    
    return filter_instance.filter_anomalies(
        anomalies,
        video_frames=video_frames,
        video_tensor=video_tensor
    )

