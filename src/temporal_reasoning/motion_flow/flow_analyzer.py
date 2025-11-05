# -*- coding: utf-8 -*-
"""
光流分析器
"""

import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch

from .raft_wrapper import RAFTWrapper
from .motion_smoothness import (
    compute_motion_smoothness,
    detect_motion_discontinuities,
    compute_flow_statistics
)
from ..core.config import RAFTConfig


class MotionFlowAnalyzer:
    """光流分析器"""
    
    def __init__(self, config: RAFTConfig):
        """
        初始化光流分析器
        
        Args:
            config: RAFTConfig配置对象
        """
        self.config = config
        self.raft_model = None
        # 根据use_gpu配置正确设置设备字符串
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
            print("光流分析器初始化完成！")
        except Exception as e:
            print(f"警告: 光流分析器初始化失败: {e}")
            print("将使用简化实现")
            # 初始化失败时，raft_model保持为None
            # analyze()方法会检查并抛出异常
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        fps: float = 30.0
    ) -> Tuple[float, List[Dict]]:
        """
        分析视频运动平滑度
        
        Args:
            video_frames: 视频帧序列，每帧为RGB图像 (H, W, 3)
            fps: 视频帧率，用于计算时间戳
        
        Returns:
            (motion_score, anomalies): 
            - motion_score: 运动合理性得分 (0-1)
            - anomalies: 运动异常列表
        """
        if len(video_frames) < 2:
            return 1.0, []
        
        if self.raft_model is None:
            raise RuntimeError(
                "RAFT模型未初始化\n"
                "请先调用 initialize() 方法初始化模型"
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
                    f"第{i}帧光流计算失败: {e}\n"
                    f"请检查RAFT模型是否正确初始化"
                )
        
        if not optical_flows:
            return 1.0, []
        
        # 2. 计算运动平滑度
        print("正在分析运动平滑度...")
        motion_smoothness = compute_motion_smoothness(optical_flows)
        
        if not motion_smoothness:
            return 1.0, []
        
        # 3. 检测运动突变
        print("正在检测运动突变...")
        # 从配置中获取阈值
        threshold = getattr(self.config, 'motion_discontinuity_threshold', 0.3)
        motion_anomalies = detect_motion_discontinuities(
            optical_flows,
            threshold=threshold,
            fps=fps
        )
        
        # 4. 计算得分
        base_score = float(np.mean(motion_smoothness))
        
        # 异常惩罚：每个异常扣分
        anomaly_penalty = min(0.5, len(motion_anomalies) * 0.1)
        final_score = max(0.0, base_score * (1.0 - anomaly_penalty))
        
        # 5. 计算统计信息
        flow_stats = compute_flow_statistics(optical_flows)
        
        print(f"运动合理性得分: {final_score:.3f}")
        print(f"检测到 {len(motion_anomalies)} 个运动异常")
        
        return final_score, motion_anomalies