# -*- coding: utf-8 -*-
"""
关键点提取器
"""

import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class MediaPipeKeypointExtractor:
    """基于MediaPipe的关键点提取器"""
    
    def __init__(self):
        """初始化MediaPipe"""
        self.holistic = None
        self.mp_holistic = None
        self.mp_hands = None
        self.mp_face_mesh = None
        
        try:
            import mediapipe as mp
            self.mp_holistic = mp.solutions.holistic
            self.mp_hands = mp.solutions.hands
            self.mp_face_mesh = mp.solutions.face_mesh
            
            # 初始化模型
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                refine_face_landmarks=True
            )
            print("MediaPipe模型初始化成功")
        except ImportError:
            print("警告: MediaPipe未安装，请运行: pip install mediapipe")
            print("关键点提取功能将不可用")
        except Exception as e:
            print(f"警告: MediaPipe初始化失败: {e}")
    
    def extract_keypoints(self, image: np.ndarray) -> Dict:
        """
        提取关键点
        
        Args:
            image: 输入图像 (H, W, 3) RGB，范围[0, 255]
        
        Returns:
            关键点字典，包含：
            - body: 身体关键点 (N, 3) 或 None
            - left_hand: 左手关键点 (N, 3) 或 None
            - right_hand: 右手关键点 (N, 3) 或 None
            - face: 面部关键点 (N, 3) 或 None
        """
        if self.holistic is None:
            return {
                'body': None,
                'left_hand': None,
                'right_hand': None,
                'face': None
            }
        
        try:
            results = self.holistic.process(image)
            
            keypoints = {
                'body': None,
                'left_hand': None,
                'right_hand': None,
                'face': None
            }
            
            # 提取身体关键点
            if results.pose_landmarks:
                keypoints['body'] = self._landmarks_to_array(results.pose_landmarks.landmark)
            
            # 提取手部关键点
            if results.left_hand_landmarks:
                keypoints['left_hand'] = self._landmarks_to_array(results.left_hand_landmarks.landmark)
            if results.right_hand_landmarks:
                keypoints['right_hand'] = self._landmarks_to_array(results.right_hand_landmarks.landmark)
            
            # 提取面部关键点
            if results.face_landmarks:
                keypoints['face'] = self._landmarks_to_array(results.face_landmarks.landmark)
            
            return keypoints
            
        except Exception as e:
            print(f"警告: 关键点提取失败: {e}")
            return {
                'body': None,
                'left_hand': None,
                'right_hand': None,
                'face': None
            }
    
    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """
        将landmarks转换为numpy数组
        
        Args:
            landmarks: MediaPipe landmarks对象
        
        Returns:
            关键点数组 (N, 3)，每行为(x, y, z)
        """
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

