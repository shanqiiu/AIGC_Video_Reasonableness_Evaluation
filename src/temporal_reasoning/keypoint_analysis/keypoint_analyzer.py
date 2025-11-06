# -*- coding: utf-8 -*-
"""
关键点分析器
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path

from .keypoint_extractor import MediaPipeKeypointExtractor
from ..core.config import KeypointConfig


class KeypointAnalyzer:
    """关键点分析器"""
    
    def __init__(self, config: KeypointConfig):
        """
        初始化关键点分析�?
        
        Args:
            config: KeypointConfig配置对象
        """
        self.config = config
        self.extractor = None
    
    def initialize(self):
        """初始化关键点提取�?"""
        print("正在初始化关键点分析�?...")
        try:
            if self.config.model_type == "mediapipe":
                # 使用.cache作为缓存目录
                import os
                from pathlib import Path
                
                # 获取项目根目�?
                project_root = Path(__file__).parent.parent.parent.parent
                cache_dir = project_root / ".cache"
                cache_dir = str(cache_dir.absolute())
                
                model_path = self.config.model_path if self.config.model_path else None
                
                print(f"缓存目录: {cache_dir}")
                if model_path:
                    print(f"模型路径: {model_path}")
                else:
                    print("使用默认模型（MediaPipe将自动下载）")
                
                self.extractor = MediaPipeKeypointExtractor(
                    model_path=model_path,
                    cache_dir=cache_dir
                )
            else:
                print(f"警告: 不支持的关键点模型类�?: {self.config.model_type}")
                print("使用MediaPipe作为默认")
                self.extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
            
            print("关键点分析器初始化完成！")
        except ImportError as e:
            print(f"错误: MediaPipe导入失败: {e}")
            print("请确保已安装MediaPipe: pip install mediapipe>=0.10.0")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"错误: 关键点分析器初始化失�?: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """
        分析视频生理动作自然�?
        
        Args:
            video_frames: 视频帧序�?
            fps: 视频帧率
        
        Returns:
            (physiological_score, anomalies):
            - physiological_score: 生理动作自然性得�? (0-1)
            - anomalies: 生理异常列表
        """
        if self.extractor is None:
            self.initialize()
        
        # 重置timestamp计数器（兼容性调用，IMAGE模式不需要timestamp）
        # 注意：使用IMAGE模式时，不需要timestamp，但保留此调用以保持兼容性
        if hasattr(self.extractor, 'reset_timestamp'):
            self.extractor.reset_timestamp()
    
        print("正在分析生理动作自然�?...")
        
        # 1. 提取关键点序�?
        print("正在提取关键�?...")
        keypoint_sequences = []
        for frame in tqdm(video_frames, desc="提取关键�?"):
            keypoints = self.extractor.extract_keypoints(frame, fps=fps)
            keypoint_sequences.append(keypoints)
        
        # 2. 分析生理动作
        print("正在分析生理动作...")
        physiological_score, anomalies = self._analyze_physiological_actions(
            keypoint_sequences,
            fps=fps
        )
        
        print(f"生理动作自然性得�?: {physiological_score:.3f}")
        print(f"检测到 {len(anomalies)} 个生理异�?")
        
        return physiological_score, anomalies
    
    def _analyze_physiological_actions(
        self,
        keypoint_sequences: List[Dict],
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """
        分析生理动作
        
        Args:
            keypoint_sequences: 关键点序�?
            fps: 视频帧率
        
        Returns:
            (physiological_score, anomalies)
        """
        if not keypoint_sequences:
            return 1.0, []
        
        anomalies = []
        action_scores = []
        
        # 分析眨眼
        blink_score, blink_anomalies = self._analyze_blink_pattern(keypoint_sequences, fps)
        action_scores.append(blink_score)
        anomalies.extend(blink_anomalies)
        
        # 分析嘴型
        mouth_score, mouth_anomalies = self._analyze_mouth_pattern(keypoint_sequences, fps)
        action_scores.append(mouth_score)
        anomalies.extend(mouth_anomalies)
        
        # 分析手势
        gesture_score, gesture_anomalies = self._analyze_hand_gesture(keypoint_sequences, fps)
        action_scores.append(gesture_score)
        anomalies.extend(gesture_anomalies)
        
        # 计算综合得分
        physiological_score = float(np.mean(action_scores)) if action_scores else 1.0
        
        return physiological_score, anomalies
    
    def _analyze_blink_pattern(
        self,
        keypoint_sequences: List[Dict],
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """分析眨眼模式"""
        # 简化实现：返回默认得分
        # 实际实现需要分析眼部关键点距离变化
        return 1.0, []
    
    def _analyze_mouth_pattern(
        self,
        keypoint_sequences: List[Dict],
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """分析嘴型模式"""
        # 简化实现：返回默认得分
        # 实际实现需要分析嘴部关键点位置变化
        return 1.0, []
    
    def _analyze_hand_gesture(
        self,
        keypoint_sequences: List[Dict],
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """分析手势模式"""
        # 简化实现：返回默认得分
        # 实际实现需要分析手部关键点角度变化
        return 1.0, []
