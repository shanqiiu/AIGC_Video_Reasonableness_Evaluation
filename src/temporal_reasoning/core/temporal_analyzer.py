# -*- coding: utf-8 -*-
"""
时序合理性分析器主类
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from .config import TemporalReasoningConfig
from ..motion_flow.flow_analyzer import MotionFlowAnalyzer
from ..instance_tracking.instance_analyzer import InstanceTrackingAnalyzer
from ..keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from ..fusion.decision_engine import FusionDecisionEngine
from ..utils.video_utils import get_video_info

# 修复导入路径
import sys
from pathlib import Path


class TemporalReasoningAnalyzer:
    """
    时序合理性分析器
    """
    
    def __init__(self, config: TemporalReasoningConfig):
        """
        初始化分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.motion_analyzer = None
        self.instance_analyzer = None
        self.keypoint_analyzer = None
        self.fusion_engine = None
        self._initialized = False
    
    def initialize(self):
        """初始化所有子模块"""
        if self._initialized:
            print("分析器已初始�?")
            return
        
        print("=" * 50)
        print("正在初始化时序合理性分析器...")
        print("=" * 50)
        
        try:
            # 初始化光流分析器
            print("\n[1/4] 初始化光流分析器...")
            self.motion_analyzer = MotionFlowAnalyzer(self.config.raft)
            self.motion_analyzer.initialize()
            
            # 初始化实例追踪分析器
            print("\n[2/4] 初始化实例追踪分析器...")
            self.instance_analyzer = InstanceTrackingAnalyzer(
                self.config.grounding_dino,
                self.config.sam,
                self.config.tracker
            )
            self.instance_analyzer.initialize()
            
            # 初始化关键点分析�?
            print("\n[3/4] 初始化关键点分析�?...")
            self.keypoint_analyzer = KeypointAnalyzer(self.config.keypoint)
            self.keypoint_analyzer.initialize()
            
            # 初始化融合决策引�?
            print("\n[4/4] 初始化融合决策引�?...")
            # 获取Co-Tracker验证器（如果可用�?
            cotracker_validator = None
            if hasattr(self.instance_analyzer, 'cotracker_validator'):
                cotracker_validator = self.instance_analyzer.cotracker_validator
            self.fusion_engine = FusionDecisionEngine(
                self.config.fusion,
                cotracker_validator=cotracker_validator
            )
            
            self._initialized = True
            print("\n" + "=" * 50)
            print("时序合理性分析器初始化完成！")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n错误: 初始化失�?: {e}")
            raise
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        text_prompts: Optional[List[str]] = None,
        fps: Optional[float] = None,
        video_path: Optional[str] = None
    ) -> Dict:
        """
        分析视频时序合理�?
        
        Args:
            video_frames: 视频帧序列，每帧为RGB图像 (H, W, 3)
            text_prompts: 可选文本提示列表（如["tongue", "finger"]�?
            fps: 视频帧率，如果为None则从视频推断
        
        Returns:
            dict: {
                'motion_reasonableness_score': float,  # 0-1
                'structure_stability_score': float,    # 0-1
                'anomalies': List[dict],               # 异常实例列表
            }
        """
        if not self._initialized:
            self.initialize()
        
        if not video_frames:
            raise ValueError("视频帧序列为�?")
        
        if fps is None:
            fps = 30.0  # 默认帧率
        
        print("\n" + "=" * 50)
        print("开始分析视频时序合理�?...")
        print(f"视频帧数: {len(video_frames)}")
        print(f"视频帧率: {fps:.2f} fps")
        if text_prompts:
            print(f"文本提示: {', '.join(text_prompts)}")
        print("=" * 50)
        
        # 1. 光流分析
        print("\n>>> 步骤1: 光流分析")
        # 传递阈值配�?
        if hasattr(self.config, 'thresholds'):
            self.motion_analyzer.config.motion_discontinuity_threshold = self.config.thresholds.motion_discontinuity_threshold
        motion_score, motion_anomalies = self.motion_analyzer.analyze(video_frames, fps=fps)
        
        # 2. 实例追踪分析
        print("\n>>> 步骤2: 实例追踪分析")
        structure_score, structure_anomalies = self.instance_analyzer.analyze(
            video_frames, text_prompts=text_prompts, fps=fps
        )
        
        # 3. 关键点分�?
        print("\n>>> 步骤3: 关键点分�?")
        physiological_score, physiological_anomalies = self.keypoint_analyzer.analyze(
            video_frames, fps=fps, video_path=video_path
        )
        
        # 4. 多模态融�?
        print("\n>>> 步骤4: 多模态融�?")
        
        # 获取Co-Tracker验证器（如果可用�?
        cotracker_validator = None
        if hasattr(self.instance_analyzer, 'cotracker_validator'):
            cotracker_validator = self.instance_analyzer.cotracker_validator
        
        # 更新融合引擎的验证器
        if cotracker_validator is not None:
            self.fusion_engine.cotracker_validator = cotracker_validator
            self.fusion_engine.anomaly_filter.cotracker_validator = cotracker_validator
        
        fused_anomalies = self.fusion_engine.fuse(
            motion_anomalies,
            structure_anomalies,
            physiological_anomalies
        )
        
        # 5. 过滤假阳性（使用Co-Tracker验证�?
        if cotracker_validator is not None:
            print("\n>>> 步骤5: 过滤假阳性异�?")
            try:
                # 转换视频帧为tensor
                import torch
                frames_array = np.stack(video_frames)  # (T, H, W, 3)
                video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
                video_tensor = video_tensor.unsqueeze(0) / 255.0  # (1, T, C, H, W)
                
                filtered_anomalies = self.fusion_engine.anomaly_filter.filter_anomalies(
                    fused_anomalies,
                    video_tensor=video_tensor
                )
                
                print(f"过滤前异常数�?: {len(fused_anomalies)}")
                print(f"过滤后异常数�?: {len(filtered_anomalies)}")
                fused_anomalies = filtered_anomalies
            except Exception as e:
                raise RuntimeError(
                    f"假阳性过滤失�?: {e}\n"
                    f"请检查Co-Tracker模型是否正确初始�?"
                )
        
        # 6. 计算最终得�?
        print("\n>>> 步骤6: 计算最终得�?")
        final_motion_score, final_structure_score = self.fusion_engine.compute_final_scores(
            motion_score,
            structure_score,
            physiological_score,
            fused_anomalies
        )
        
        # 构建结果
        result = {
            'motion_reasonableness_score': float(final_motion_score),
            'structure_stability_score': float(final_structure_score),
            'anomalies': fused_anomalies,
            'sub_scores': {
                'motion_score': float(motion_score),
                'structure_score': float(structure_score),
                'physiological_score': float(physiological_score)
            },
            'anomaly_counts': {
                'motion': len(motion_anomalies),
                'structure': len(structure_anomalies),
                'physiological': len(physiological_anomalies),
                'fused': len(fused_anomalies)
            }
        }
        
        print("\n" + "=" * 50)
        print("分析完成�?")
        print("=" * 50)
        print(f"运动合理性得�?: {final_motion_score:.3f}")
        print(f"结构稳定性得�?: {final_structure_score:.3f}")
        print(f"检测到 {len(fused_anomalies)} 个异�?")
        print("=" * 50)
        
        return result

