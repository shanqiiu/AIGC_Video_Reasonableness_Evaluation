"""
场景分类器 - 自动判断视频场景类型（静态/动态），并提供置信度。
"""

import numpy as np


class SceneClassifier:
    """
    场景分类器 - 自动判断视频场景类型（静态/动态），并提供置信度。
    """
    
    def __init__(self,
                 static_threshold: float = 0.1,
                 low_dynamic_threshold: float = 0.3,
                 medium_dynamic_threshold: float = 0.6,
                 high_dynamic_threshold: float = 1.0,
                 motion_ratio_threshold: float = 1.5):
        """
        初始化场景分类器
        
        Args:
            static_threshold: 静态场景阈值（主体动态度）
            low_dynamic_threshold: 低动态场景阈值
            medium_dynamic_threshold: 中等动态场景阈值
            high_dynamic_threshold: 高动态场景阈值
            motion_ratio_threshold: 运动比率阈值（主体/背景），用于区分相机运动与物体运动
        """
        self.static_threshold = static_threshold
        self.low_dynamic_threshold = low_dynamic_threshold
        self.medium_dynamic_threshold = medium_dynamic_threshold
        self.high_dynamic_threshold = high_dynamic_threshold
        self.motion_ratio_threshold = motion_ratio_threshold
    
    def classify_scene(self,
                     background_motion: float,
                     subject_motion: float,
                     pure_subject_motion: float,
                     motion_ratio: float) -> dict:
        """
        分类场景类型
        
        Args:
            background_motion: 背景动态度
            subject_motion: 主体动态度
            pure_subject_motion: 纯主体动态度（主体减去背景）
            motion_ratio: 运动比率（主体/背景）
            
        Returns:
            包含场景类型、强度等级与置信度的结果字典
        """
        # 1) 判断相机运动主导还是物体运动主导
        motion_dominant_type = self._determine_motion_dominant(
            background_motion, subject_motion, motion_ratio
        )
        
        # 2) 基于纯主体动态度判断强度等级
        intensity_level = self._determine_intensity_level(pure_subject_motion)
        
        # 3) 综合判断得到场景类型与描述
        scene_type, scene_description = self._determine_scene_type(
            motion_dominant_type, intensity_level, 
            background_motion, subject_motion, motion_ratio
        )
        
        return {
            'scene_type': scene_type,
            'scene_description': scene_description,
            'motion_dominant': motion_dominant_type,
            'intensity_level': intensity_level,
            'background_motion': float(background_motion),
            'subject_motion': float(subject_motion),
            'pure_subject_motion': float(pure_subject_motion),
            'motion_ratio': float(motion_ratio),
            'confidence': self._calculate_confidence(
                background_motion, subject_motion, motion_ratio
            )
        }
    
    def _determine_motion_dominant(self,
                                   background_motion: float,
                                   subject_motion: float,
                                   motion_ratio: float) -> str:
        """判断运动主导类型"""
        
        # 如果运动比率小于阈值，说明主体并不明显强于背景
        if motion_ratio < self.motion_ratio_threshold:
            return 'camera_motion'  # 相机运动主导
        # 如果主体运动明显大于背景运动
        elif motion_ratio >= self.motion_ratio_threshold:
            return 'object_motion'  # 物体运动主导
        else:
            return 'mixed_motion'  # 混合运动
    
    def _determine_intensity_level(self, pure_subject_motion: float) -> str:
        """判断运动强度等级"""
        
        if pure_subject_motion < self.static_threshold:
            return 'static'
        elif pure_subject_motion < self.low_dynamic_threshold:
            return 'low_dynamic'
        elif pure_subject_motion < self.medium_dynamic_threshold:
            return 'medium_dynamic'
        elif pure_subject_motion < self.high_dynamic_threshold:
            return 'high_dynamic'
        else:
            return 'extreme_dynamic'
    
    def _determine_scene_type(self,
                              motion_dominant: str,
                              intensity_level: str,
                              background_motion: float,
                              subject_motion: float,
                              motion_ratio: float) -> tuple:
        """综合判断场景类型并给出中文描述"""
        
        # 相机运动主导的场景
        if motion_dominant == 'camera_motion':
            if intensity_level == 'static':
                return 'static_camera', '静态相机运动场景（相机平移、缩放等）'
            elif intensity_level == 'low_dynamic':
                return 'low_dynamic_camera', '低动态相机运动场景'
            else:
                return 'dynamic_camera', f'{intensity_level}级动态相机运动场景'
        
        # 物体运动主导的场景
        elif motion_dominant == 'object_motion':
            if intensity_level == 'static':
                return 'static_object', '静态物体场景（物体基本静止）'
            elif intensity_level == 'low_dynamic':
                return 'low_dynamic_object', '低动态物体运动场景'
            elif intensity_level == 'medium_dynamic':
                return 'medium_dynamic_object', '中等动态物体运动场景'
            elif intensity_level == 'high_dynamic':
                return 'high_dynamic_object', '高动态物体运动场景'
            else:
                return 'extreme_dynamic_object', '极高动态物体运动场景'
        
        # 混合运动场景
        else:
            return 'mixed_scene', f'混合运动场景（相机运动：{background_motion:.4f}，物体运动：{subject_motion:.4f}）'
    
    def _calculate_confidence(self,
                             background_motion: float,
                             subject_motion: float,
                             motion_ratio: float) -> float:
        """计算分类结果的置信度（0~1）"""
        
        # 基于运动比率的明确程度计算置信度
        if motion_ratio < 0.5 or motion_ratio > 2.0:
            # 运动比率极端（明显偏向一方），置信度高
            ratio_confidence = 0.9
        elif 0.8 < motion_ratio < 1.2:
            # 运动比率接近1（难以区分），置信度低
            ratio_confidence = 0.3
        else:
            # 运动比率有一定差异，置信度中等
            ratio_confidence = 0.6
        
        # 基于运动强度计算置信度
        total_motion = background_motion + subject_motion
        if 0.01 < total_motion < 5.0:
            intensity_confidence = 0.8  # 处于合理的运动范围
        else:
            intensity_confidence = 0.4  # 运动过小或过大
        
        # 综合置信度
        confidence = (ratio_confidence + intensity_confidence) / 2
        
        return float(np.clip(confidence, 0.0, 1.0))

