# -*- coding: utf-8 -*-
"""
光流分析�?
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch
import cv2

from .raft_wrapper import RAFTWrapper
from .motion_smoothness import (
    compute_motion_smoothness,
    detect_motion_discontinuities,
    compute_flow_statistics
)
from ..core.config import RAFTConfig


class MotionFlowAnalyzer:
    """光流分析�?"""
    
    def __init__(self, config: RAFTConfig):
        """
        初始化光流分析器
        
        Args:
            config: RAFTConfig配置对象
        """
        self.config = config
        self.raft_model = None
        # 根据use_gpu配置正确设置设备字符�?
        if config.use_gpu and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
    
    def initialize(self):
        """初始化RAFT模型"""
        print("正在初始化光流分析器...")
        try:
            self.raft_model = RAFTWrapper(
                model_path=self.config.model_path,
                model_type=self.config.model_type,
                device=self.device
            )
            print("光流分析器初始化完成�?")
        except Exception as e:
            print(f"警告: 光流分析器初始化失败: {e}")
            print("将使用简化实�?")
            # 初始化失败时，raft_model保持为None
            # analyze()方法会检查并抛出异常
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """
        分析视频运动平滑�?
        
        Args:
            video_frames: 视频帧序列，每帧为RGB图像 (H, W, 3)
            fps: 视频帧率，用于计算时间戳
        
        Returns:
            (motion_score, anomalies): 
            - motion_score: 运动合理性得�? (0-1)
            - anomalies: 运动异常列表
        """
        if len(video_frames) < 2:
            return 1.0, []
        
        if self.raft_model is None:
            raise RuntimeError(
                "RAFT模型未初始化\n"
                "请先调用 initialize() 方法初始化模�?"
            )
        
        # 1. 计算光流序列
        print("正在计算光流...")
        optical_flows = []
        for i in tqdm(range(len(video_frames) - 1), desc="计算光流"):
            try:
                u, v = self.raft_model.compute_flow(video_frames[i], video_frames[i+1])
                optical_flows.append((u, v))
            except Exception as e:
                raise RuntimeError(
                    f"第{i}帧光流计算失�?: {e}\n"
                    f"请检查RAFT模型是否正确初始�?"
                )
        
        if not optical_flows:
            return 1.0, []
        
        # 2. 计算运动平滑�?
        print("正在分析运动平滑�?...")
        motion_smoothness = compute_motion_smoothness(optical_flows)
        
        if not motion_smoothness:
            return 1.0, []
        
        # 3. 检测运动突�?
        print("正在检测运动突�?...")
        # 从配置中获取阈�?
        threshold = getattr(self.config, 'motion_discontinuity_threshold', 0.3)
        motion_anomalies = detect_motion_discontinuities(
            optical_flows,
            threshold=threshold,
            fps=fps
        )
        
        # 4. 计算得分
        base_score = float(np.mean(motion_smoothness))
        
        # 异常惩罚：每个异常扣�?
        anomaly_penalty = min(0.5, len(motion_anomalies) * 0.1)
        final_score = max(0.0, base_score * (1.0 - anomaly_penalty))
        
        # 5. 计算统计信息
        flow_stats = compute_flow_statistics(optical_flows)
        
        print(f"运动合理性得�?: {final_score:.3f}")
        print(f"检测到 {len(motion_anomalies)} 个运动异�?")

        if getattr(self.config, "enable_visualization", False):
            self._save_visualizations(video_frames, optical_flows, motion_anomalies)
        
        return final_score, motion_anomalies

    @staticmethod
    def _flow_to_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=False)
        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _select_frames_for_visualization(
        self,
        num_flows: int,
        anomalies: List[Dict],
    ) -> List[int]:
        candidate_frames = sorted(
            {a.get("frame_id") for a in anomalies if isinstance(a, dict) and "frame_id" in a}
        )
        candidate_frames = [idx for idx in candidate_frames if isinstance(idx, int)]
        max_frames = max(0, getattr(self.config, "visualization_max_frames", 0))
        if not candidate_frames:
            if max_frames <= 0 or num_flows == 0:
                return []
            stride = max(1, num_flows // max_frames) if max_frames > 0 else 1
            candidate_frames = list(range(0, num_flows, stride))
        if max_frames > 0:
            candidate_frames = candidate_frames[:max_frames]
        return candidate_frames

    def _save_visualizations(
        self,
        video_frames: List[np.ndarray],
        optical_flows: List[Tuple[np.ndarray, np.ndarray]],
        motion_anomalies: List[Dict],
    ) -> None:
        frame_indices = self._select_frames_for_visualization(len(optical_flows), motion_anomalies)
        if not frame_indices:
            return

        if self.config.visualization_output_dir:
            output_dir = Path(self.config.visualization_output_dir).expanduser().resolve()
        else:
            output_dir = Path(__file__).resolve().parents[3] / "outputs" / "motion_flow"
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx in frame_indices:
            if idx >= len(optical_flows) or idx >= len(video_frames):
                continue
            u, v = optical_flows[idx]
            flow_color = self._flow_to_color(u, v)

            frame_rgb = video_frames[idx]
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            if frame_rgb.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_rgb

            blended = cv2.addWeighted(flow_color, 0.7, frame_bgr, 0.3, 0.0)
            save_path = output_dir / f"flow_{idx:04d}.png"
            cv2.imwrite(str(save_path), blended)