# -*- coding: utf-8 -*-
"""
实例追踪分析器
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from ..core.config import GroundingDINOConfig, SAMConfig, TrackerConfig
from .cotracker_validator import CoTrackerValidator


class InstanceTrackingAnalyzer:
    """实例追踪分析器"""
    
    def __init__(
        self,
        gdino_config: GroundingDINOConfig,
        sam_config: SAMConfig,
        tracker_config: TrackerConfig
    ):
        """
        初始化实例追踪分析器
        
        Args:
            gdino_config: Grounding DINO配置
            sam_config: SAM配置
            tracker_config: 追踪器配置
        """
        self.gdino_config = gdino_config
        self.sam_config = sam_config
        self.tracker_config = tracker_config
        
        self.grounding_dino = None
        self.sam_model = None
        self.tracker = None
        self.cotracker_validator = None
    
    def initialize(self):
        """初始化模型"""
        print("正在初始化实例追踪分析器...")
        try:
            # 这里可以添加实际的模型初始化代码
            # 由于Grounded-SAM-2的集成较复杂，这里提供一个占位实现
            print("警告: Grounded-SAM-2模型初始化需要根据实际实现调整")
            print("实例追踪分析器使用简化实现")
            
            # 初始化Co-Tracker验证器（如果启用）
            if self.tracker_config.enable_cotracker_validation:
                try:
                    device = self.tracker_config.use_gpu if self.tracker_config.use_gpu else "cpu"
                    checkpoint_path = self.tracker_config.cotracker_checkpoint or self.tracker_config.model_path
                    self.cotracker_validator = CoTrackerValidator(
                        checkpoint_path=checkpoint_path,
                        device=device,
                        grid_size=self.tracker_config.grid_size
                    )
                    if self.cotracker_validator.cotracker_model is not None:
                        print("Co-Tracker验证器初始化成功")
                    else:
                        print("警告: Co-Tracker验证器初始化失败，将不使用Co-Tracker验证")
                        self.cotracker_validator = None
                except Exception as e:
                    print(f"警告: Co-Tracker验证器初始化失败: {e}")
                    print("将不使用Co-Tracker验证")
                    self.cotracker_validator = None
            else:
                print("Co-Tracker验证已禁用")
                self.cotracker_validator = None
                
        except Exception as e:
            print(f"警告: 实例追踪分析器初始化失败: {e}")
            print("将使用简化实现")
    
    def detect_instances(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        检测和分割实例
        
        Args:
            image: 输入图像 (H, W, 3) RGB
            text_prompts: 文本提示列表
        
        Returns:
            掩码列表，每个元素为(mask, confidence)元组
        """
        # 简化实现：返回空列表
        # 实际实现需要调用Grounded DINO + SAM
        return []
    
    def track_instances(
        self,
        video_frames: List[np.ndarray],
        detections: List[List[Tuple[np.ndarray, float]]]
    ) -> Dict[int, Dict]:
        """
        追踪实例
        
        Args:
            video_frames: 视频帧序列
            detections: 每帧的检测结果
        
        Returns:
            追踪结果字典，key为实例ID，value为追踪信息
        """
        # 简化实现：返回空字典
        # 实际实现需要调用DeAOT或Co-Tracker
        return {}
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        text_prompts: Optional[List[str]] = None,
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """
        分析视频结构稳定性
        
        Args:
            video_frames: 视频帧序列
            text_prompts: 可选文本提示列表
            fps: 视频帧率
        
        Returns:
            (structure_score, anomalies):
            - structure_score: 结构稳定性得分 (0-1)
            - anomalies: 结构异常列表
        """
        print("正在分析结构稳定性...")
        
        if text_prompts is None:
            text_prompts = []
        
        # 简化实现：如果没有文本提示，返回默认得分
        if not text_prompts:
            print("警告: 未提供文本提示，无法进行实例检测")
            return 1.0, []
        
        # 1. 检测实例
        print("正在检测实例...")
        detections = []
        for i, frame in enumerate(tqdm(video_frames, desc="检测实例")):
            masks = self.detect_instances(frame, text_prompts)
            detections.append(masks)
        
        # 2. 追踪实例
        print("正在追踪实例...")
        tracked_instances = self.track_instances(video_frames, detections)
        
        # 3. 分析结构稳定性
        print("正在分析结构完整性...")
        structure_score, anomalies = self._analyze_structure_stability(
            tracked_instances,
            fps=fps
        )
        
        # 4. 使用Co-Tracker验证异常（如果可用）
        if self.cotracker_validator is not None and anomalies:
            print("正在使用Co-Tracker验证异常...")
            anomalies = self._validate_anomalies_with_cotracker(
                anomalies,
                video_frames
            )
        
        print(f"结构稳定性得分: {structure_score:.3f}")
        print(f"检测到 {len(anomalies)} 个结构异常（已过滤假阳性）")
        
        return structure_score, anomalies
    
    def _analyze_structure_stability(
        self,
        tracked_instances: Dict[int, Dict],
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """
        分析结构稳定性
        
        Args:
            tracked_instances: 追踪结果
            fps: 视频帧率
        
        Returns:
            (structure_score, anomalies)
        """
        if not tracked_instances:
            return 1.0, []
        
        anomalies = []
        structure_scores = []
        
        for instance_id, track_info in tracked_instances.items():
            # 分析掩码面积变化
            # 分析掩码形状变化
            # 检测消失异常
            
            # 简化实现：假设所有实例都正常
            structure_scores.append(1.0)
        
        base_score = float(np.mean(structure_scores)) if structure_scores else 1.0
        
        return base_score, anomalies
    
    def _validate_anomalies_with_cotracker(
        self,
        anomalies: List[Dict],
        video_frames: List[np.ndarray]
    ) -> List[Dict]:
        """
        使用Co-Tracker验证异常
        
        Args:
            anomalies: 异常列表
            video_frames: 视频帧列表
        
        Returns:
            验证后的异常列表
        """
        if self.cotracker_validator is None or not video_frames:
            return anomalies
        
        # 转换视频为tensor
        try:
            frames_array = np.stack(video_frames)  # (T, H, W, 3)
            video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            video_tensor = video_tensor.unsqueeze(0) / 255.0  # (1, T, C, H, W)
            
            # 获取视频尺寸
            _, _, _, video_height, video_width = video_tensor.shape
            
            validated_anomalies = []
            for anomaly in anomalies:
                # 获取异常信息
                anomaly_type = anomaly.get('type', '').lower()
                frame_id = anomaly.get('frame_id', 0)
                location = anomaly.get('location', {})
                mask = location.get('mask')
                
                if mask is None:
                    validated_anomalies.append(anomaly)
                    continue
                
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
                    validated_anomalies.append(anomaly)
                    continue
                
                # 保存验证信息
                anomaly['validation_info'] = validation_info
                
                # 只保留有效的异常
                if is_valid:
                    validated_anomalies.append(anomaly)
                else:
                    anomaly['filtered'] = True
                    anomaly['filter_reason'] = validation_info.get('reason', 'unknown')
            
            return validated_anomalies
            
        except Exception as e:
            raise RuntimeError(
                f"Co-Tracker验证失败: {e}\n"
                f"请检查Co-Tracker模型是否正确初始化"
            )

