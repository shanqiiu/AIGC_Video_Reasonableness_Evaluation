# -*- coding: utf-8 -*-
"""
时序合理性评估主入口，负责统筹光流、结构、关键点及融合流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import TemporalReasoningConfig
from ..fusion.decision_engine import FusionDecisionEngine
from ..instance_tracking import TemporalCoherenceConfig, TemporalCoherencePipeline
from ..keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from ..motion_flow.flow_analyzer import MotionFlowAnalyzer


@dataclass
class StructureAnalysisOutput:
    score: float
    vanish_score: float
    emerge_score: float
    anomalies: List[Dict]
    metadata: Dict[str, object]


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
        self.motion_analyzer: Optional[MotionFlowAnalyzer] = None
        self.structure_pipeline: Optional[TemporalCoherencePipeline] = None
        self.keypoint_analyzer: Optional[KeypointAnalyzer] = None
        self.fusion_engine: Optional[FusionDecisionEngine] = None
        self._initialized = False

    def initialize(self):
        """初始化所有子模块"""
        if self._initialized:
            print("提示：分析器已处于初始化状态。")
            return

        print("=" * 50)
        print("正在初始化时序合理性分析器...")
        print("=" * 50)

        try:
            # 1. 光流分析模块
            print("\n[1/4] 初始化光流分析器...")
            self.motion_analyzer = MotionFlowAnalyzer(self.config.raft)
            self.motion_analyzer.initialize()

            # 2. 结构一致性分析管线
            print("\n[2/4] 初始化实例追踪 / 结构分析管线...")
            coherence_config = self._build_temporal_coherence_config()
            self.structure_pipeline = TemporalCoherencePipeline(coherence_config)
            self.structure_pipeline.initialize()

            # 3. 关键点分析器（可选，如果配置中禁用了则跳过）
            if self.config.keypoint is not None:
                print("\n[3/4] 初始化关键点分析器...")
                self.keypoint_analyzer = KeypointAnalyzer(self.config.keypoint)
                self.keypoint_analyzer.initialize()
            else:
                print("\n[3/4] 跳过关键点分析器（已禁用，通用物体检测模式）...")
                self.keypoint_analyzer = None

            # 4. 融合决策引擎
            print("\n[4/4] 初始化融合决策引擎...")
            self.fusion_engine = FusionDecisionEngine(self.config.fusion, cotracker_validator=None)

            self._initialized = True
            print("\n" + "=" * 50)
            print("时序合理性分析器初始化完成！")
            print("=" * 50)

        except Exception as exc:
            print(f"\n错误：初始化失败：{exc}")
            raise

    def _build_temporal_coherence_config(self) -> TemporalCoherenceConfig:
        """构造结构分析管线的配置字典"""
        meta_info_path = Path(self.config.output_dir) / "temporal_coherence_meta.json"
        cotracker_checkpoint = (
            self.config.tracker.cotracker_checkpoint
            or self.config.tracker.model_path
            or ".cache/scaled_offline.pth"
        )
        device = (
            "cuda"
            if "cuda" in str(self.config.device).lower() and torch.cuda.is_available()
            else "cpu"
        )

        prompts = self.config.structure_prompts or ["object"]
        prompts = [p.strip() for p in prompts if p and p.strip()]
        text_prompt = ". ".join(prompts) if prompts else "object"
        if not text_prompt.endswith("."):
            text_prompt = f"{text_prompt}."

        return TemporalCoherenceConfig(
            meta_info_path=str(meta_info_path),
            text_prompt=text_prompt,
            grounding_config_path=self.config.grounding_dino.config_path,
            grounding_checkpoint_path=self.config.grounding_dino.model_path,
            bert_path=self.config.grounding_dino.bert_path,
            sam2_config_path=self.config.sam.config_path,
            sam2_checkpoint_path=self.config.sam.model_path,
            cotracker_checkpoint_path=cotracker_checkpoint,
            device=device,
            box_threshold=self.config.grounding_dino.box_threshold,
            text_threshold=self.config.grounding_dino.text_threshold,
            grid_size=self.config.tracker.grid_size,
            iou_threshold=0.75,
            enable_visualization=self.config.structure_visualization_enable,
            visualization_output_dir=self.config.structure_visualization_output_dir,
            visualization_max_frames=self.config.structure_visualization_max_frames,
            enable_cotracker=getattr(self.config, 'enable_cotracker', False),
            cotracker_visualization_enable=self.config.cotracker_visualization_enable,
            cotracker_visualization_output_dir=self.config.cotracker_visualization_output_dir,
            cotracker_visualization_fps=self.config.cotracker_visualization_fps,
            cotracker_visualization_mode=self.config.cotracker_visualization_mode,
            cotracker_visualization_full_video=getattr(self.config, 'cotracker_visualization_full_video', False),
            enable_region_temporal_analysis=getattr(self.config, 'enable_region_temporal_analysis', True),  # 默认开启
            region_temporal_config=getattr(self.config, 'region_temporal_config', None),
        )

    def analyze(
        self,
        video_frames: List[np.ndarray],
        text_prompts: Optional[Sequence[str]] = None,
        fps: Optional[float] = None,
        video_path: Optional[str] = None,
    ) -> Dict:
        """
        分析视频时序合理性

        Args:
            video_frames: 视频帧序列，每帧为RGB图像 (H, W, 3)
            text_prompts: 文本提示列表
            fps: 视频帧率
            video_path: 原始视频路径（结构分析需要）

        Returns:
            分析结果字典
        """
        if not self._initialized:
            self.initialize()

        if not video_frames:
            raise ValueError("视频帧序列为空")

        fps = fps or 30.0

        print("\n" + "=" * 50)
        print("开始分析视频时序合理性...")
        print(f"视频帧数：{len(video_frames)}")
        print(f"视频帧率：{fps:.2f} fps")
        if text_prompts:
            print(f"文本提示：{', '.join(text_prompts)}")
        print("=" * 50)

        # 1. 光流分析
        print("\n>>> 步骤1: 光流分析")
        if hasattr(self.config, "thresholds"):
            self.motion_analyzer.config.motion_discontinuity_threshold = (
                self.config.thresholds.motion_discontinuity_threshold
            )
        motion_score, motion_anomalies = self.motion_analyzer.analyze(video_frames, fps=fps)

        # 2. 结构分析
        print("\n>>> 步骤2: 实例追踪 / 结构分析")
        structure_output = self._analyze_structure(video_path, text_prompts)

        # 3. 关键点分析（可选）
        if self.keypoint_analyzer is not None:
            print("\n>>> 步骤3: 关键点分析")
            physiological_score, physiological_anomalies = self.keypoint_analyzer.analyze(
                video_frames, fps=fps, video_path=video_path
            )
        else:
            print("\n>>> 步骤3: 关键点分析（已跳过）")
            physiological_score = 1.0
            physiological_anomalies = []

        # 4. 多模态融合
        print("\n>>> 步骤4: 多模态融合")
        fused_anomalies = self.fusion_engine.fuse(
            motion_anomalies,
            structure_output.anomalies,
            physiological_anomalies,
            structure_context={
                "vanish_score": structure_output.vanish_score,
                "emerge_score": structure_output.emerge_score,
                **structure_output.metadata,
            },
        )

        # 5. 计算最终得分
        print("\n>>> 步骤5: 计算最终得分")
        final_motion_score, final_structure_score = self.fusion_engine.compute_final_scores(
            motion_score,
            structure_output.score,
            physiological_score,
            fused_anomalies,
            structure_context={
                "vanish_score": structure_output.vanish_score,
                "emerge_score": structure_output.emerge_score,
                **structure_output.metadata,
            },
        )

        # 构建 sub_scores（如果关键点模块未启用，不包含 physiological_score）
        sub_scores = {
            "motion_score": float(motion_score),
            "structure_score": float(structure_output.score),
        }
        if self.keypoint_analyzer is not None:
            sub_scores["physiological_score"] = float(physiological_score)
        
        # 构建 anomaly_counts（如果关键点模块未启用，不包含 physiological）
        anomaly_counts = {
            "motion": len(motion_anomalies),
            "structure": len(structure_output.anomalies),
            "fused": len(fused_anomalies),
        }
        if self.keypoint_analyzer is not None:
            anomaly_counts["physiological"] = len(physiological_anomalies)
        
        # 收集每帧异常检测信息
        per_frame_anomaly_detection = self._collect_per_frame_anomaly_info(
            video_frames,
            motion_anomalies,
            structure_output,
            physiological_anomalies if self.keypoint_analyzer is not None else [],
            fps
        )
        
        # 收集所有阈值配置
        thresholds_info = self._collect_thresholds_info()
        
        result = {
            "motion_reasonableness_score": float(final_motion_score),
            "structure_stability_score": float(final_structure_score),
            "anomalies": fused_anomalies,
            "sub_scores": sub_scores,
            "anomaly_counts": anomaly_counts,
            "structure_metrics": {
                "coherence_score": float(structure_output.score),
                "vanish_score": float(structure_output.vanish_score),
                "emerge_score": float(structure_output.emerge_score),
                **structure_output.metadata,
            },
            "per_frame_anomaly_detection": per_frame_anomaly_detection,
            "thresholds": thresholds_info,
        }

        print("\n" + "=" * 50)
        print("分析完成")
        print("=" * 50)
        print(f"运动合理性得分：{final_motion_score:.3f}")
        print(f"结构稳定性得分：{final_structure_score:.3f}")
        print(f"检测到 {len(fused_anomalies)} 个融合异常。")
        print("=" * 50)

        return result

    def _collect_per_frame_anomaly_info(
        self,
        video_frames: List[np.ndarray],
        motion_anomalies: List[Dict],
        structure_output: StructureAnalysisOutput,
        physiological_anomalies: List[Dict],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        收集每帧的异常检测信息
        
        Returns:
            每帧的异常检测信息列表
        """
        total_frames = len(video_frames)
        per_frame_info = []
        
        # 按帧ID组织异常
        motion_by_frame = {}
        for anomaly in motion_anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in motion_by_frame:
                motion_by_frame[frame_id] = []
            motion_by_frame[frame_id].append(anomaly)
        
        structure_by_frame = {}
        for anomaly in structure_output.anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in structure_by_frame:
                structure_by_frame[frame_id] = []
            structure_by_frame[frame_id].append(anomaly)
        
        physiological_by_frame = {}
        for anomaly in physiological_anomalies:
            frame_id = anomaly.get('frame_id', 0)
            if frame_id not in physiological_by_frame:
                physiological_by_frame[frame_id] = []
            physiological_by_frame[frame_id].append(anomaly)
        
        # 获取结构分析的每帧信息
        structure_frame_states = structure_output.metadata.get('frame_states', [])
        structure_frame_dict = {state.get('frame_index', -1): state for state in structure_frame_states}
        
        # 为每一帧创建信息
        for frame_idx in range(total_frames):
            frame_info = {
                "frame_index": frame_idx,
                "timestamp": frame_idx / max(fps, 1),
                "motion": {
                    "anomalies": motion_by_frame.get(frame_idx, []),
                    "anomaly_count": len(motion_by_frame.get(frame_idx, [])),
                },
                "structure": {
                    "anomalies": structure_by_frame.get(frame_idx, []),
                    "anomaly_count": len(structure_by_frame.get(frame_idx, [])),
                    "frame_state": structure_frame_dict.get(frame_idx, {}),
                },
                "physiological": {
                    "anomalies": physiological_by_frame.get(frame_idx, []),
                    "anomaly_count": len(physiological_by_frame.get(frame_idx, [])),
                },
            }
            
            # 添加结构分析的详细值（如果可用）
            if frame_idx in structure_frame_dict:
                frame_state = structure_frame_dict[frame_idx]
                frame_info["structure"].update({
                    "object_count": frame_state.get("object_count", 0),
                    "frame_type": frame_state.get("frame_type", "unknown"),
                    "detected": frame_state.get("detected", False),
                })
            
            per_frame_info.append(frame_info)
        
        return per_frame_info
    
    def _collect_thresholds_info(self) -> Dict[str, Any]:
        """
        收集所有阈值配置信息
        
        Returns:
            阈值配置字典
        """
        thresholds_info = {}
        
        # 运动分析阈值
        if hasattr(self.config, 'thresholds'):
            thresholds_info['motion'] = {
                "motion_discontinuity_threshold": self.config.thresholds.motion_discontinuity_threshold,
            }
        
        # 结构分析阈值
        if self.structure_pipeline:
            thresholds_info['structure'] = {
                "size_change_area_ratio_threshold": self.structure_pipeline.config.size_change_area_ratio_threshold,
                "size_change_height_ratio_threshold": self.structure_pipeline.config.size_change_height_ratio_threshold,
                "size_change_min_area": self.structure_pipeline.config.size_change_min_area,
                "iou_threshold": self.structure_pipeline.config.iou_threshold,
            }
        
        # 融合阈值
        if hasattr(self.config, 'fusion'):
            thresholds_info['fusion'] = {
                "multimodal_confidence_boost": self.config.fusion.multimodal_confidence_boost,
                "min_anomaly_duration_frames": self.config.fusion.min_anomaly_duration_frames,
                "single_modality_confidence_threshold": self.config.fusion.single_modality_confidence_threshold,
            }
        
        # 关键点分析阈值
        if hasattr(self.config, 'thresholds') and hasattr(self.config.thresholds, 'keypoint_displacement_threshold'):
            thresholds_info['keypoint'] = {
                "keypoint_displacement_threshold": self.config.thresholds.keypoint_displacement_threshold,
            }
        
        # 区域时序变化检测阈值
        if self.structure_pipeline and self.structure_pipeline.config.region_temporal_config:
            region_config = self.structure_pipeline.config.region_temporal_config
            thresholds_info['region_temporal'] = {
                "motion_threshold": region_config.motion_threshold,
                "similarity_threshold": region_config.similarity_threshold,
                "consecutive_frames": region_config.consecutive_frames,
                "baseline_window": region_config.baseline_window,
                "min_roi_size": region_config.min_roi_size,
                "hist_diff_threshold": region_config.hist_diff_threshold,
            }
        
        return thresholds_info

    def _analyze_structure(
        self,
        video_path: Optional[str],
        text_prompts: Optional[Sequence[str]],
    ) -> StructureAnalysisOutput:
        if self.structure_pipeline is None:
            print("警告：结构分析管线未初始化，返回默认结果。")
            return StructureAnalysisOutput(
                score=1.0,
                vanish_score=1.0,
                emerge_score=1.0,
                anomalies=[],
                metadata={},
            )

        if not video_path:
            print("警告：未提供视频路径，无法执行结构一致性分析。")
            return StructureAnalysisOutput(
                score=1.0,
                vanish_score=1.0,
                emerge_score=1.0,
                anomalies=[],
                metadata={},
            )

        try:
            result = self.structure_pipeline.evaluate_video(video_path, text_prompts)
            return StructureAnalysisOutput(
                score=float(result.coherence_score),
                vanish_score=float(result.vanish_score),
                emerge_score=float(result.emerge_score),
                anomalies=result.anomalies,
                metadata=result.metadata,
            )
        except Exception as exc:
            print(f"警告：结构分析失败，将使用默认得分。详情：{exc}")
            return StructureAnalysisOutput(
                score=1.0,
                vanish_score=1.0,
                emerge_score=1.0,
                anomalies=[],
                metadata={"error": str(exc)},
            )

