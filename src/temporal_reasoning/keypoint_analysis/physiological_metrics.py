# -*- coding: utf-8 -*-
"""
生理动作指标计算模块
提供眨眼、嘴型、手势等生理动作的具体计算方法
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class PhysiologicalMetrics:
    """生理动作指标计算类"""
    
    # MediaPipe 面部关键点索引
    FACE_LANDMARKS = {
        # 左眼
        'left_eye': {
            'vertical1': (159, 145),  # 上下眼睑中心
            'vertical2': (158, 144),  # 左侧
            'horizontal': (33, 133)    # 左右眼角
        },
        # 右眼
        'right_eye': {
            'vertical1': (386, 374),
            'vertical2': (385, 373),
            'horizontal': (362, 263)
        },
        # 嘴部
        'mouth': {
            'top': 13,      # 上唇中心
            'bottom': 14,   # 下唇中心
            'left': 78,     # 左嘴角
            'right': 308    # 右嘴角
        }
    }
    
    # 分析参数
    PARAMS = {
        # 眨眼参数
        'ear_threshold': 0.25,         # EAR阈值
        'min_blink_frames': 2,         # 最少闭眼帧数
        'max_blink_frames_ratio': 0.5, # 最多闭眼帧数比例（秒）
        'normal_blink_rate_min': 5/60, # 最小眨眼频率（次/秒）
        'normal_blink_rate_max': 30/60,# 最大眨眼频率（次/秒）
        
        # 嘴型参数
        'mar_threshold': 0.5,          # 嘴部开合阈值
        'mar_jump_threshold': 0.3,     # MAR跳跃阈值
        'max_open_duration_s': 3.0,    # 最大持续张嘴时间（秒）
        
        # 手势参数
        'velocity_threshold': 0.3,     # 速度突变阈值
        'jitter_threshold': 0.05,      # 抖动阈值
        'window_size': 5,              # 滑动窗口大小
    }
    
    @staticmethod
    def compute_eye_aspect_ratio(face_landmarks: np.ndarray, eye_indices: dict) -> Optional[float]:
        """
        计算眼睛纵横比（EAR - Eye Aspect Ratio）
        
        Args:
            face_landmarks: 面部关键点数组 (468, 3)
            eye_indices: 眼睛关键点索引字典
        
        Returns:
            EAR值，范围约[0, 0.4]，值越小眼睛越闭合
        """
        try:
            v1_top, v1_bottom = eye_indices['vertical1']
            v2_top, v2_bottom = eye_indices['vertical2']
            h_left, h_right = eye_indices['horizontal']
            
            # 计算垂直距离
            vertical1 = np.linalg.norm(face_landmarks[v1_top] - face_landmarks[v1_bottom])
            vertical2 = np.linalg.norm(face_landmarks[v2_top] - face_landmarks[v2_bottom])
            
            # 计算水平距离
            horizontal = np.linalg.norm(face_landmarks[h_left] - face_landmarks[h_right])
            
            # 计算EAR
            if horizontal > 0:
                ear = (vertical1 + vertical2) / (2.0 * horizontal)
                return float(ear)
            return 0.0
        except Exception:
            return None
    
    @staticmethod
    def compute_mouth_aspect_ratio(face_landmarks: np.ndarray, mouth_indices: dict) -> Optional[float]:
        """
        计算嘴部纵横比（MAR - Mouth Aspect Ratio）
        
        Args:
            face_landmarks: 面部关键点数组 (468, 3)
            mouth_indices: 嘴部关键点索引字典
        
        Returns:
            MAR值，值越大嘴巴张开越大
        """
        try:
            top = mouth_indices['top']
            bottom = mouth_indices['bottom']
            left = mouth_indices['left']
            right = mouth_indices['right']
            
            # 计算垂直和水平距离
            vertical = np.linalg.norm(face_landmarks[top] - face_landmarks[bottom])
            horizontal = np.linalg.norm(face_landmarks[left] - face_landmarks[right])
            
            # 计算MAR
            if horizontal > 0:
                mar = vertical / horizontal
                return float(mar)
            return 0.0
        except Exception:
            return None
    
    @staticmethod
    def detect_blinks(
        ear_sequence: List[Optional[float]],
        fps: float,
        params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        检测眨眼事件
        
        Args:
            ear_sequence: EAR值序列
            fps: 视频帧率
            params: 参数字典（可选）
        
        Returns:
            眨眼事件列表
        """
        if params is None:
            params = PhysiologicalMetrics.PARAMS
        
        ear_threshold = params['ear_threshold']
        min_blink_frames = params['min_blink_frames']
        max_blink_frames = int(fps * params['max_blink_frames_ratio'])
        
        blinks = []
        is_closed = False
        blink_start = -1
        
        for frame_idx, ear in enumerate(ear_sequence):
            if ear is None:
                continue
            
            if not is_closed and ear < ear_threshold:
                # 开始闭眼
                is_closed = True
                blink_start = frame_idx
            elif is_closed and ear >= ear_threshold:
                # 睁眼
                is_closed = False
                blink_duration = frame_idx - blink_start
                
                # 有效眨眼
                if min_blink_frames <= blink_duration <= max_blink_frames:
                    blinks.append({
                        'start': blink_start,
                        'end': frame_idx,
                        'duration_frames': blink_duration,
                        'duration_ms': (blink_duration / fps) * 1000
                    })
        
        return blinks
    
    @staticmethod
    def detect_sequence_jumps(
        sequence: List[Optional[float]],
        threshold: float
    ) -> List[int]:
        """
        检测序列中的跳跃（不连续点）
        
        Args:
            sequence: 数值序列
            threshold: 跳跃阈值
        
        Returns:
            跳跃点的索引列表
        """
        jumps = []
        
        for i in range(1, len(sequence)):
            if sequence[i] is None or sequence[i-1] is None:
                continue
            
            diff = abs(sequence[i] - sequence[i-1])
            if diff > threshold:
                jumps.append(i)
        
        return jumps
    
    @staticmethod
    def compute_hand_velocity(
        hand_positions: List[Tuple[int, Optional[np.ndarray]]],
        fps: float
    ) -> List[Tuple[int, Optional[float]]]:
        """
        计算手部运动速度
        
        Args:
            hand_positions: 手部位置序列 [(frame_idx, position), ...]
            fps: 视频帧率
        
        Returns:
            速度序列 [(frame_idx, velocity), ...]
        """
        velocities = []
        
        for i in range(1, len(hand_positions)):
            frame_idx1, pos1 = hand_positions[i-1]
            frame_idx2, pos2 = hand_positions[i]
            
            if pos1 is not None and pos2 is not None:
                displacement = np.linalg.norm(pos2 - pos1)
                time_delta = (frame_idx2 - frame_idx1) / fps
                velocity = displacement / time_delta if time_delta > 0 else 0.0
                velocities.append((frame_idx2, velocity))
            else:
                velocities.append((frame_idx2, None))
        
        return velocities
    
    @staticmethod
    def compute_jitter(
        positions: List[np.ndarray],
        window_size: int = 5
    ) -> List[float]:
        """
        计算位置序列的抖动程度
        
        Args:
            positions: 位置序列
            window_size: 滑动窗口大小
        
        Returns:
            抖动值序列
        """
        jitters = []
        
        for i in range(window_size, len(positions)):
            window = positions[i-window_size:i]
            
            if len(window) > 0:
                positions_array = np.array(window)
                std = np.std(positions_array, axis=0).mean()
                jitters.append(float(std))
            else:
                jitters.append(0.0)
        
        return jitters


class AnomalyBuilder:
    """异常事件构建器"""
    
    @staticmethod
    def build_blink_anomaly(
        anomaly_type: str,
        frame_id: int,
        fps: float,
        **kwargs
    ) -> Dict:
        """构建眨眼相关异常"""
        severity_map = {
            'abnormal_blink_duration': 'medium',
            'low_blink_rate': 'low',
            'high_blink_rate': 'medium'
        }
        
        confidence_map = {
            'abnormal_blink_duration': 0.8,
            'low_blink_rate': 0.7,
            'high_blink_rate': 0.8
        }
        
        return {
            'type': anomaly_type,
            'frame_id': frame_id,
            'timestamp': f"{frame_id / fps:.2f}s",
            'severity': severity_map.get(anomaly_type, 'low'),
            'confidence': confidence_map.get(anomaly_type, 0.7),
            **kwargs
        }
    
    @staticmethod
    def build_mouth_anomaly(
        anomaly_type: str,
        frame_id: int,
        fps: float,
        **kwargs
    ) -> Dict:
        """构建嘴型相关异常"""
        severity_map = {
            'mouth_discontinuity': 'medium',
            'prolonged_mouth_opening': 'low'
        }
        
        confidence_map = {
            'mouth_discontinuity': 0.8,
            'prolonged_mouth_opening': 0.7
        }
        
        return {
            'type': anomaly_type,
            'frame_id': frame_id,
            'timestamp': f"{frame_id / fps:.2f}s",
            'severity': severity_map.get(anomaly_type, 'low'),
            'confidence': confidence_map.get(anomaly_type, 0.7),
            **kwargs
        }
    
    @staticmethod
    def build_hand_anomaly(
        anomaly_type: str,
        hand_type: str,
        frame_id: int,
        fps: float,
        **kwargs
    ) -> Dict:
        """构建手势相关异常"""
        severity_map = {
            'hand_velocity_jump': 'medium',
            'hand_jitter': 'low',
            'hand_disappear': 'low',
            'hand_appear': 'low'
        }
        
        confidence_map = {
            'hand_velocity_jump': 0.75,
            'hand_jitter': 0.7,
            'hand_disappear': 0.6,
            'hand_appear': 0.6
        }
        
        return {
            'type': anomaly_type,
            'hand': hand_type,
            'frame_id': frame_id,
            'timestamp': f"{frame_id / fps:.2f}s",
            'severity': severity_map.get(anomaly_type, 'low'),
            'confidence': confidence_map.get(anomaly_type, 0.7),
            **kwargs
        }

