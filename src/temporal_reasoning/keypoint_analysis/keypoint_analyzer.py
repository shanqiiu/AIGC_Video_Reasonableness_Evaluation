# -*- coding: utf-8 -*-
"""
关键点分析器
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import os

from .keypoint_extractor import MediaPipeKeypointExtractor
from .keypoint_visualizer import KeypointVisualizer
from ..core.config import KeypointConfig


class KeypointAnalyzer:
    """关键点分析器"""
    
    def __init__(self, config: KeypointConfig):
        """
        初始化关键点分析器
        
        Args:
            config: KeypointConfig配置对象
        """
        self.config = config
        self.extractor = None
        self.visualizer = None
        
        # 初始化可视化器（如果启用）
        if self.config.enable_visualization:
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
                # 使用.cache作为缓存目录
                import os
                from pathlib import Path
                
                # 获取项目根目录
                project_root = Path(__file__).parent.parent.parent.parent
                cache_dir = project_root / ".cache"
                cache_dir = str(cache_dir.absolute())
                
                print(f"缓存目录: {cache_dir}")
                print(f"MediaPipe模型缓存目录: {cache_dir}/mediapipe/")
                print("离线模式：仅从缓存目录加载模型（不会自动下载）")
                print("注意：模型文件应直接放在 mediapipe/ 目录中（不在 models/ 子目录中）")
                
                self.extractor = MediaPipeKeypointExtractor(
                    model_path=None,  # 旧API不使用model_path
                    cache_dir=cache_dir
                )
            else:
                print(f"警告: 不支持的关键点模型类型: {self.config.model_type}")
                print("使用MediaPipe作为默认")
                self.extractor = MediaPipeKeypointExtractor(
                    cache_dir=".cache"
                )
            
            print("关键点分析器初始化完成！")
        except ImportError as e:
            print(f"错误: MediaPipe导入失败: {e}")
            print("请确保已安装MediaPipe: pip install mediapipe>=0.10.0")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"错误: 关键点分析器初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        fps: float = 30.0,
        video_path: Optional[str] = None
    ) -> Tuple[float, List[Dict]]:
        """
        分析视频生理动作自然性
        
        Args:
            video_frames: 视频帧序列
            fps: 视频帧率
            video_path: 视频路径（用于可视化输出，可选）
        
        Returns:
            (physiological_score, anomalies):
            - physiological_score: 生理动作自然性得分 (0-1)
            - anomalies: 生理异常列表
        """
        if self.extractor is None:
            self.initialize()
        
        # 重置timestamp计数器（兼容性调用，IMAGE模式不需要timestamp）
        # 注意：使用IMAGE模式时，不需要timestamp，但保留此调用以保持兼容性
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
                # 转换为BGR格式（OpenCV使用BGR）
                frame_bgr = frame.copy()
                if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
                    # 如果是RGB，转换为BGR
                    if frame_bgr.dtype == np.uint8:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                
                # 可视化关键点
                vis_frame = self.visualizer.visualize(
                    image=frame_bgr,
                    keypoints=keypoints,
                    output_path=None,  # 稍后统一保存
                    show=False  # 不单独显示每一帧
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
        """
        分析生理动作
        
        Args:
            keypoint_sequences: 关键点序列
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
    
    def _save_visualization(
        self,
        visualized_frames: List[np.ndarray],
        fps: float,
        video_path: Optional[str] = None
    ):
        """
        保存可视化结果
        
        Args:
            visualized_frames: 可视化后的帧列表
            fps: 视频帧率
            video_path: 原始视频路径（用于生成输出文件名）
        """
        if not self.config.save_visualization:
            return
        
        # 确定输出目录
        if self.config.visualization_output_dir:
            output_dir = Path(self.config.visualization_output_dir)
        else:
            # 使用默认输出目录
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
        
        # 如果启用显示，显示第一帧
        if self.config.show_visualization and visualized_frames:
            import cv2
            cv2.imshow('Keypoint Visualization (First Frame)', visualized_frames[0])
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
