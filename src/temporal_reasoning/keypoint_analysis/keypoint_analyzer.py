# -*- coding: utf-8 -*-
"""
关键点分析器完整实现示例
展示如何在 KeypointAnalyzer 中实现眨眼、嘴型、手势分析
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path

from .keypoint_extractor import MediaPipeKeypointExtractor
from .physiological_metrics import PhysiologicalMetrics, AnomalyBuilder
from ..core.config import KeypointConfig


class KeypointAnalyzer:
    """关键点分析器（包含完整指标实现）"""
    
    def __init__(self, config: KeypointConfig):
        """
        初始化关键点分析器
        
        Args:
            config: KeypointConfig配置对象
        """
        self.config = config
        self.extractor = None
        self.visualizer = None
        self.metrics = PhysiologicalMetrics()
        
        # 初始化可视化器（如果启用）
        if self.config.enable_visualization:
            from .keypoint_visualizer import KeypointVisualizer
            self.visualizer = KeypointVisualizer(
                show_face=self.config.show_face,
                show_face_mesh=self.config.show_face_mesh,
                point_radius=self.config.point_radius,
                line_thickness=self.config.line_thickness
            )
    
    def initialize(self):
        """初始化关键点提取器"""
        print("正在初始化关键点分析器...")
        try:
            if self.config.model_type == "mediapipe":
                import os
                from pathlib import Path
                
                project_root = Path(__file__).parent.parent.parent.parent
                cache_dir = project_root / ".cache"
                cache_dir = str(cache_dir.absolute())
                
                print(f"缓存目录: {cache_dir}")
                print(f"MediaPipe模型缓存目录: {cache_dir}/mediapipe/")
                
                self.extractor = MediaPipeKeypointExtractor(
                    model_path=None,
                    cache_dir=cache_dir
                )
            else:
                print(f"警告: 不支持的关键点模型类型: {self.config.model_type}")
                self.extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
            
            print("关键点分析器初始化完成！")
        except Exception as e:
            print(f"错误: 关键点分析器初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        fps: float = 30.0,
        video_path: str = None
    ) -> Tuple[float, List[Dict]]:
        """
        分析视频生理动作自然性
        
        Args:
            video_frames: 视频帧序列
            fps: 视频帧率
            video_path: 视频路径（用于可视化输出，可选）
        
        Returns:
            (physiological_score, anomalies)
        """
        if self.extractor is None:
            self.initialize()
        
        if hasattr(self.extractor, 'reset_timestamp'):
            self.extractor.reset_timestamp()
        
        print("正在分析生理动作自然性...")
        
        # 1. 提取关键点序列
        print("正在提取关键点...")
        keypoint_sequences = []
        visualized_frames = []
        
        for frame_idx, frame in enumerate(tqdm(video_frames, desc="提取关键点")):
            keypoints = self.extractor.extract_keypoints(frame, fps=fps)
            keypoint_sequences.append(keypoints)
            
            # 如果启用可视化，处理可视化
            if self.config.enable_visualization and self.visualizer is not None:
                import cv2
                frame_bgr = frame.copy()
                if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
                    if frame_bgr.dtype == np.uint8:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                
                vis_frame = self.visualizer.visualize(
                    image=frame_bgr,
                    keypoints=keypoints,
                    output_path=None,
                    show=False
                )
                visualized_frames.append(vis_frame)
        
        # 保存可视化结果（如果启用）
        if self.config.enable_visualization and self.visualizer is not None and visualized_frames:
            self._save_visualization(visualized_frames, fps, video_path)
        
        # 2. 分析生理动作
        print("正在分析生理动作...")
        physiological_score, anomalies = self._analyze_physiological_actions(
            keypoint_sequences,
            fps=fps
        )
        
        print(f"生理动作自然性得分: {physiological_score:.3f}")
        print(f"检测到 {len(anomalies)} 个生理异常")
        
        return physiological_score, anomalies
    
    def _analyze_physiological_actions(
        self,
        keypoint_sequences: List[Dict],
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """分析生理动作"""
        if not keypoint_sequences:
            return 1.0, []
        
        anomalies = []
        action_scores = []
        
        # 分析眨眼
        blink_score, blink_anomalies = self._analyze_blink_pattern(keypoint_sequences, fps)
        action_scores.append(blink_score)
        anomalies.extend(blink_anomalies)
        print(f"  - 眨眼分析: 得分={blink_score:.3f}, 异常={len(blink_anomalies)}")
        
        # 分析嘴型
        mouth_score, mouth_anomalies = self._analyze_mouth_pattern(keypoint_sequences, fps)
        action_scores.append(mouth_score)
        anomalies.extend(mouth_anomalies)
        print(f"  - 嘴型分析: 得分={mouth_score:.3f}, 异常={len(mouth_anomalies)}")
        
        # 分析手势
        gesture_score, gesture_anomalies = self._analyze_hand_gesture(keypoint_sequences, fps)
        action_scores.append(gesture_score)
        anomalies.extend(gesture_anomalies)
        print(f"  - 手势分析: 得分={gesture_score:.3f}, 异常={len(gesture_anomalies)}")
        
        # 计算综合得分
        physiological_score = float(np.mean(action_scores)) if action_scores else 1.0
        
        return physiological_score, anomalies
    
    def _analyze_blink_pattern(
        self,
        keypoint_sequences: List[Dict],
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """分析眨眼模式"""
        if not keypoint_sequences:
            return 1.0, []
        
        params = self.metrics.PARAMS
        anomalies = []
        ear_sequence = []
        
        # 计算每帧的EAR
        for frame_idx, keypoints in enumerate(keypoint_sequences):
            if keypoints['face'] is None:
                ear_sequence.append(None)
                continue
            
            face = keypoints['face']
            
            # 计算左眼和右眼的EAR
            left_ear = self.metrics.compute_eye_aspect_ratio(
                face, self.metrics.FACE_LANDMARKS['left_eye']
            )
            right_ear = self.metrics.compute_eye_aspect_ratio(
                face, self.metrics.FACE_LANDMARKS['right_eye']
            )
            
            # 平均EAR
            if left_ear is not None and right_ear is not None:
                avg_ear = (left_ear + right_ear) / 2.0
            else:
                avg_ear = left_ear or right_ear
            
            ear_sequence.append(avg_ear)
        
        # 检测眨眼事件
        blinks = self.metrics.detect_blinks(ear_sequence, fps, params)
        
        # 计算眨眼频率
        video_duration = len(keypoint_sequences) / fps
        blink_rate = len(blinks) / video_duration if video_duration > 0 else 0
        
        # 检测异常眨眼频率
        if blink_rate < params['normal_blink_rate_min']:
            anomaly = AnomalyBuilder.build_blink_anomaly(
                'low_blink_rate',
                0,
                fps,
                blink_rate=blink_rate * 60,
                description=f"眨眼频率过低: {blink_rate * 60:.1f}次/分钟"
            )
            anomalies.append(anomaly)
        elif blink_rate > params['normal_blink_rate_max']:
            anomaly = AnomalyBuilder.build_blink_anomaly(
                'high_blink_rate',
                0,
                fps,
                blink_rate=blink_rate * 60,
                description=f"眨眼频率过高: {blink_rate * 60:.1f}次/分钟"
            )
            anomalies.append(anomaly)
        
        # 检测眨眼持续时间过长
        for blink in blinks:
            if blink['duration_ms'] > params['max_blink_frames_ratio'] * 1000:
                anomaly = AnomalyBuilder.build_blink_anomaly(
                    'abnormal_blink_duration',
                    blink['start'],
                    fps,
                    duration_ms=blink['duration_ms'],
                    description=f"眨眼持续时间过长: {blink['duration_ms']:.0f}ms"
                )
                anomalies.append(anomaly)
        
        # 计算得分
        if not anomalies:
            score = 1.0
        else:
            penalty = sum(0.1 if a['severity'] == 'low' else 0.2 for a in anomalies)
            score = max(0.0, 1.0 - penalty)
        
        return score, anomalies
    
    def _analyze_mouth_pattern(
        self,
        keypoint_sequences: List[Dict],
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """分析嘴型模式"""
        if not keypoint_sequences:
            return 1.0, []
        
        params = self.metrics.PARAMS
        anomalies = []
        mar_sequence = []
        
        # 计算每帧的MAR
        for frame_idx, keypoints in enumerate(keypoint_sequences):
            if keypoints['face'] is None:
                mar_sequence.append(None)
                continue
            
            face = keypoints['face']
            mar = self.metrics.compute_mouth_aspect_ratio(
                face, self.metrics.FACE_LANDMARKS['mouth']
            )
            mar_sequence.append(mar)
        
        # 检测MAR跳跃（不连续）
        jumps = self.metrics.detect_sequence_jumps(
            mar_sequence,
            params['mar_jump_threshold']
        )
        
        for frame_idx in jumps:
            mar_diff = abs(mar_sequence[frame_idx] - mar_sequence[frame_idx-1])
            anomaly = AnomalyBuilder.build_mouth_anomaly(
                'mouth_discontinuity',
                frame_idx,
                fps,
                mar_jump=mar_diff,
                description=f"嘴型变化不连续: MAR跳跃 {mar_diff:.2f}"
            )
            anomalies.append(anomaly)
        
        # 检测持续张嘴
        open_start = -1
        for frame_idx, mar in enumerate(mar_sequence):
            if mar is None:
                continue
            
            if mar > params['mar_threshold']:
                if open_start == -1:
                    open_start = frame_idx
            else:
                if open_start != -1:
                    open_duration = frame_idx - open_start
                    duration_s = open_duration / fps
                    if duration_s > params['max_open_duration_s']:
                        anomaly = AnomalyBuilder.build_mouth_anomaly(
                            'prolonged_mouth_opening',
                            open_start,
                            fps,
                            duration_s=duration_s,
                            description=f"嘴部持续张开: {duration_s:.1f}秒"
                        )
                        anomalies.append(anomaly)
                    open_start = -1
        
        # 计算得分
        if not anomalies:
            score = 1.0
        else:
            penalty = sum(0.1 if a['severity'] == 'low' else 0.15 for a in anomalies)
            score = max(0.0, 1.0 - penalty)
        
        return score, anomalies
    
    def _analyze_hand_gesture(
        self,
        keypoint_sequences: List[Dict],
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """分析手势模式"""
        if not keypoint_sequences:
            return 1.0, []
        
        params = self.metrics.PARAMS
        anomalies = []
        
        # 分析左手和右手
        for hand_type in ['left_hand', 'right_hand']:
            hand_positions = []
            
            # 提取手腕位置（索引0）
            for frame_idx, keypoints in enumerate(keypoint_sequences):
                if keypoints[hand_type] is not None:
                    wrist_pos = keypoints[hand_type][0][:2]  # 只取x, y
                    hand_positions.append((frame_idx, wrist_pos))
                else:
                    hand_positions.append((frame_idx, None))
            
            # 计算速度
            velocities = self.metrics.compute_hand_velocity(hand_positions, fps)
            
            # 检测速度跳跃
            vel_jumps = []
            for i in range(1, len(velocities)):
                frame_idx1, vel1 = velocities[i-1]
                frame_idx2, vel2 = velocities[i]
                
                if vel1 is not None and vel2 is not None:
                    vel_diff = abs(vel2 - vel1)
                    if vel_diff > params['velocity_threshold']:
                        vel_jumps.append((frame_idx2, vel_diff))
            
            for frame_idx, vel_diff in vel_jumps:
                anomaly = AnomalyBuilder.build_hand_anomaly(
                    'hand_velocity_jump',
                    hand_type,
                    frame_idx,
                    fps,
                    velocity_jump=vel_diff,
                    description=f"{hand_type}运动速度突变: {vel_diff:.2f}"
                )
                anomalies.append(anomaly)
            
            # 检测手部抖动
            valid_positions = [pos for _, pos in hand_positions if pos is not None]
            if len(valid_positions) > params['window_size']:
                jitters = self.metrics.compute_jitter(
                    valid_positions,
                    params['window_size']
                )
                
                for i, jitter_val in enumerate(jitters):
                    if jitter_val > params['jitter_threshold']:
                        frame_idx = i + params['window_size']
                        if frame_idx < len(keypoint_sequences):
                            anomaly = AnomalyBuilder.build_hand_anomaly(
                                'hand_jitter',
                                hand_type,
                                frame_idx,
                                fps,
                                jitter_std=jitter_val,
                                description=f"{hand_type}抖动过大: std={jitter_val:.3f}"
                            )
                            anomalies.append(anomaly)
        
        # 计算得分
        if not anomalies:
            score = 1.0
        else:
            penalty = sum(0.05 if a['severity'] == 'low' else 0.1 for a in anomalies)
            score = max(0.0, 1.0 - penalty)
        
        return score, anomalies
    
    def _save_visualization(
        self,
        visualized_frames: List[np.ndarray],
        fps: float,
        video_path: str = None
    ):
        """保存可视化结果"""
        if not self.config.save_visualization:
            return
        
        # 确定输出目录
        if self.config.visualization_output_dir:
            output_dir = Path(self.config.visualization_output_dir)
        else:
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = project_root / "outputs" / "keypoint_visualization"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        if video_path:
            video_name = Path(video_path).stem
            output_path = output_dir / f"{video_name}_keypoints.mp4"
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"keypoints_{timestamp}.mp4"
        
        # 保存视频
        if visualized_frames:
            import cv2
            h, w = visualized_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            
            for frame in tqdm(visualized_frames, desc="保存可视化结果"):
                out.write(frame)
            
            out.release()
            print(f"可视化结果已保存到: {output_path}")

