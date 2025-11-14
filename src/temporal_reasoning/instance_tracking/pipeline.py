from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
COTRACKER_ROOT = _ROOT / "third_party" / "co-tracker"
if COTRACKER_ROOT.exists() and str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore
from cotracker.utils.visualizer import read_video_from_path, Visualizer  # type: ignore

from .detection import DetectionConfig, Sam2DetectionEngine
from .evaluation import TemporalEventEvaluator
from .mask_manager import MaskDictionary
from .tcs_utils import get_appear_objects, get_disappear_objects
from .types import ObjectInfo
from .video_io import extract_frames_from_video

# 导入区域时序变化检测器
try:
    from src.temporal_reasoning.region_analysis.region_temporal_change_detector import (
        RegionTemporalChangeDetector,
        RegionTemporalChangeConfig,
    )
    REGION_ANALYSIS_AVAILABLE = True
except ImportError:
    REGION_ANALYSIS_AVAILABLE = False


@dataclass
class TemporalCoherenceConfig:
    meta_info_path: str
    text_prompt: str
    grounding_config_path: str = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    grounding_checkpoint_path: str = ".cache/groundingdino_swinb_cogcoor.pth"
    bert_path: str = ".cache/google-bert/bert-base-uncased"
    sam2_config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint_path: str = ".cache/sam2.1_hiera_large.pt"
    cotracker_checkpoint_path: str = ".cache/scaled_offline.pth"
    enable_cotracker: bool = False  # 是否启用CoTracker验证（默认不启用，仅使用SAM2评估）
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.35
    grid_size: int = 30
    iou_threshold: float = 0.75
    enable_visualization: bool = False
    visualization_output_dir: Optional[str] = None
    visualization_max_frames: int = 50
    cotracker_visualization_enable: bool = False
    cotracker_visualization_output_dir: Optional[str] = None
    cotracker_visualization_fps: int = 12
    cotracker_visualization_mode: str = "grayscale"
    cotracker_visualization_full_video: bool = False  # 是否生成整体视频的追踪可视化（默认不生成）
    size_change_area_ratio_threshold: float = 3.0
    size_change_height_ratio_threshold: float = 2.5
    size_change_min_area: int = 200
    # 区域时序变化检测配置（复用区域分析逻辑）
    enable_region_temporal_analysis: bool = True  # 是否启用区域时序变化检测（默认开启）
    region_temporal_config: Optional[RegionTemporalChangeConfig] = None  # 区域时序变化检测配置


@dataclass
class TemporalCoherenceResult:
    coherence_score: float
    vanish_score: float
    emerge_score: float
    anomalies: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class TemporalCoherencePipeline:
    """
    End-to-end pipeline that mirrors the logic in `temporal_coherence_score.py`
    while offering a reusable and testable interface.
    """

    _PALETTE = np.array([
        [60, 60, 60],
        [90, 90, 90],
        [120, 120, 120],
        [150, 150, 150],
        [45, 45, 45],
        [30, 30, 30],
        [0, 80, 80],
        [80, 0, 0],
        [0, 0, 90],
        [100, 60, 0],
    ], dtype=np.uint8)

    def __init__(self, config: TemporalCoherenceConfig):
        self.config = config
        self.detection_engine = Sam2DetectionEngine(
            DetectionConfig(
                grounding_config_path=config.grounding_config_path,
                grounding_checkpoint_path=config.grounding_checkpoint_path,
                bert_path=config.bert_path,
                sam2_config_path=config.sam2_config_path,
                sam2_checkpoint_path=config.sam2_checkpoint_path,
                box_threshold=config.box_threshold,
                text_threshold=config.text_threshold,
                device=config.device,
            )
        )
        self.cotracker_model: Optional[CoTrackerPredictor] = None
        self.event_evaluator: Optional[TemporalEventEvaluator] = None

    def initialize(self) -> None:
        self.detection_engine.initialize()
        # CoTracker是可选的，默认不启用
        if self.config.enable_cotracker:
            device = self.config.device if torch.cuda.is_available() else "cpu"
            self.cotracker_model = CoTrackerPredictor(
                checkpoint=self.config.cotracker_checkpoint_path,
                v2=False,
                offline=True,
                window_len=60,
            ).to(device)
            self.event_evaluator = TemporalEventEvaluator(
                self.cotracker_model,
                grid_size=self.config.grid_size,
            )
            print("[配置] 已启用 CoTracker 验证（将验证消失/出现的合理性）")
        else:
            self.cotracker_model = None
            self.event_evaluator = None
            print("[配置] CoTracker 已禁用（仅使用 SAM2 进行评估）")

    def _compose_text_prompt(
        self,
        text_prompts: Optional[Sequence[str]],
        fallback: Optional[str] = None,
    ) -> str:
        prompts = [p.strip() for p in (text_prompts or []) if p and p.strip()]
        prompt = ". ".join(prompts) if prompts else (fallback or self.config.text_prompt).strip()
        if not prompt.endswith("."):
            prompt = f"{prompt}."
        return prompt

    def _prepare_tracking_result(self, video_object_data: List[Dict], step: int) -> List[Dict]:
        if not video_object_data:
            return []
        filtered = [item for idx, item in enumerate(video_object_data, start=1) if idx % (step + 1) != 0]
        return filtered[:: max(step, 1)]

    def _object_info_from_mask(self, mask, class_name: str, instance_id: int) -> ObjectInfo:
        mask_cpu = mask.detach().to("cpu").bool()
        obj = ObjectInfo(instance_id=instance_id, mask=mask_cpu, class_name=class_name)
        obj.update_box()
        return obj

    def _compute_sam2_only_scores(
        self,
        disappear_objects: List[Dict],
        appear_objects: List[Dict],
        tracking_result: List[Dict],
        objects_count: int,
        fps: int,
    ) -> Tuple[float, float]:
        """
        仅使用SAM2评估时计算消失/出现分数
        
        逻辑：
        1. 如果没有消失/出现对象，分数为1.0
        2. 如果有消失/出现对象，根据对象数量和持续时间计算分数
        3. 考虑对象的持续时间：持续时间越长，突然消失/出现越不合理
        
        Args:
            disappear_objects: 消失对象列表
            appear_objects: 出现对象列表
            tracking_result: 追踪结果
            objects_count: 对象总数
            fps: 帧率
        
        Returns:
            (vanish_score, emerge_score): 消失分数和出现分数
        """
        # 计算消失分数
        if not disappear_objects:
            vanish_score = 1.0
        else:
            if objects_count == 0:
                vanish_score = 1.0
            else:
                # 计算消失对象的平均持续时间
                total_duration = 0.0
                for obj in disappear_objects:
                    first_appearance = obj.get("first_appearance", 0)
                    last_frame = obj.get("last_frame", first_appearance)
                    duration = (last_frame - first_appearance + 1) / fps  # 转换为秒
                    total_duration += duration
                
                avg_duration = total_duration / len(disappear_objects) if disappear_objects else 0.0
                
                # 计算消失比例
                disappear_ratio = len(disappear_objects) / objects_count if objects_count > 0 else 0.0
                
                # 分数计算：
                # - 消失比例越高，分数越低
                # - 持续时间越长，突然消失越不合理，分数越低
                # - 最小分数为0.0
                vanish_score = max(0.0, 1.0 - disappear_ratio * (1.0 + min(avg_duration / 2.0, 1.0)))
        
        # 计算出现分数
        if not appear_objects:
            emerge_score = 1.0
        else:
            if objects_count == 0:
                # 如果之前没有对象，新出现对象是合理的
                emerge_score = 0.8
            else:
                # 计算出现对象的数量比例
                appear_ratio = len(appear_objects) / objects_count if objects_count > 0 else 0.0
                
                # 分数计算：
                # - 出现比例越高，分数越低
                # - 最小分数为0.0
                emerge_score = max(0.0, 1.0 - appear_ratio * 0.8)
        
        return vanish_score, emerge_score

    def evaluate_video(
        self,
        video_path: str,
        text_prompts: Optional[Sequence[str]] = None,
    ) -> TemporalCoherenceResult:
        # CoTracker是可选的，如果没有启用则跳过验证
        if self.config.enable_cotracker and self.event_evaluator is None:
            raise RuntimeError("CoTracker已启用但TemporalEventEvaluator未初始化。")
        
        # 初始化检测失败列表（用于记录采样帧未检测到但之前有传播对象的情况）
        detection_failures: List[Dict] = []

        vis_dir: Optional[Path] = None
        vis_counter = 0
        max_visualizations = max(0, self.config.visualization_max_frames)
        if self.config.enable_visualization and max_visualizations > 0:
            vis_dir = self._get_visualization_dir(
                self.config.visualization_output_dir,
                video_path,
            )

        frames, _ = extract_frames_from_video(video_path)
        fps_value = read_video_fps(video_path)
        fps = max(1, int(fps_value) if fps_value else 24)
        step = max(1, fps - 1)
        text_prompt = self._compose_text_prompt(text_prompts)
        print(f"[Structure] 使用文本 prompt: \"{text_prompt}\"")

        inference_state = self.detection_engine.init_video_state(video_path)
        sam2_masks = MaskDictionary()
        objects_count = 0
        video_object_data: List[Dict] = []
        frame_states: List[Dict[str, Any]] = []  # 记录每帧的状态信息
        
        for start_frame_idx in range(0, len(frames), step):
            image = frames[start_frame_idx]
            mask_dict = self.detection_engine.detect(image, text_prompt)
            detected = bool(mask_dict.labels)
            has_propagated_objects = bool(sam2_masks.labels) if sam2_masks else False
            
            print(
                f"[Structure] 帧 {start_frame_idx:04d} {'检测到' if detected else '未检测到'} "
                f"(目标数: {len(mask_dict.labels)}, 传播对象: {len(sam2_masks.labels) if sam2_masks else 0})"
            )

            if not mask_dict.labels:
                # 如果之前有传播的对象，继续传播它们，但记录检测失败异常
                if has_propagated_objects:
                    # 记录检测失败异常：采样帧应该检测到对象，但没有检测到
                    detection_failures.append({
                        'frame': start_frame_idx,
                        'propagated_objects': list(sam2_masks.labels.keys()) if sam2_masks else [],
                        'description': f'采样帧{start_frame_idx}未检测到对象，但之前有{len(sam2_masks.labels)}个传播对象，说明对象突然消失或畸变'
                    })
                    print(f"[警告] 帧 {start_frame_idx:04d} 未检测到对象，但之前有传播对象，继续传播并记录异常")
                    
                    # 继续传播之前的对象
                    # 需要将sam2_masks中的对象添加到inference_state以便继续传播
                    if hasattr(self.detection_engine.video_predictor, "reset_state"):
                        self.detection_engine.video_predictor.reset_state(inference_state)
                    
                    # 将之前传播的对象添加到inference_state
                    for obj_id, obj in sam2_masks.labels.items():
                        # 创建一个临时的mask_dict用于添加到状态
                        temp_mask_dict = MaskDictionary()
                        temp_mask_dict.labels[obj_id] = obj
                        self.detection_engine.add_masks_to_video_state(inference_state, start_frame_idx, temp_mask_dict)
                    
                    # 传播之前的对象
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.detection_engine.propagate(
                        inference_state,
                        step,
                        start_frame_idx,
                    ):
                        frame_masks = MaskDictionary()
                        frame_data: Dict[int, Dict] = {}
                        for i, out_obj_id in enumerate(out_obj_ids):
                            out_mask = out_mask_logits[i] > 0.0
                            # 从sam2_masks获取class_name
                            class_name = sam2_masks.get_class_name(out_obj_id) if sam2_masks and out_obj_id in sam2_masks.labels else ""
                            mask_2d = out_mask[0] if out_mask.ndim == 3 else out_mask
                            obj = self._object_info_from_mask(mask_2d, class_name, out_obj_id)
                            frame_masks.labels[out_obj_id] = obj
                            frame_data[out_obj_id] = obj.to_serializable()
                        sam2_masks = frame_masks.clone()
                        video_object_data.append(frame_data)
                        
                        # 记录传播帧的状态
                        frame_states.append({
                            "frame_index": out_frame_idx,
                            "frame_type": "propagated",  # 传播帧
                            "detected": False,  # 传播帧不是检测得到的
                            "object_count": len(frame_data),
                            "object_ids": list(frame_data.keys()),
                            "source_frame": start_frame_idx,  # 来源采样帧
                            "timestamp": out_frame_idx / max(fps, 1),
                        })

                        if vis_dir is not None and vis_counter < max_visualizations:
                            if 0 <= out_frame_idx < len(frames):
                                self._save_structure_visualization(
                                    frames[out_frame_idx],
                                    frame_masks,
                                    vis_dir,
                                    out_frame_idx,
                                )
                                vis_counter += 1
                else:
                    # 既没有检测到，也没有传播对象，填充空数据
                    for empty_idx in range(step + 1):
                        video_object_data.append({})
                        # 记录空帧的状态
                        frame_states.append({
                            "frame_index": start_frame_idx + empty_idx,
                            "frame_type": "empty",  # 空帧
                            "detected": False,
                            "object_count": 0,
                            "object_ids": [],
                            "timestamp": (start_frame_idx + empty_idx) / max(fps, 1),
                        })
                continue

            objects_count, updated_dict = mask_dict.update_with_tracker(
                sam2_masks,
                iou_threshold=self.config.iou_threshold,
                objects_count=objects_count,
            )
            mask_dict = updated_dict

            if hasattr(self.detection_engine.video_predictor, "reset_state"):
                self.detection_engine.video_predictor.reset_state(inference_state)
            self.detection_engine.add_masks_to_video_state(inference_state, start_frame_idx, mask_dict)

            # 将采样帧的数据添加到video_object_data中，确保采样帧也被包含在异常检测中
            sampling_frame_data: Dict[int, Dict] = {}
            for obj_id, obj in mask_dict.labels.items():
                sampling_frame_data[obj_id] = obj.to_serializable()
            video_object_data.append(sampling_frame_data)
            
            # 记录采样帧的状态
            frame_states.append({
                "frame_index": start_frame_idx,
                "frame_type": "sampling",  # 采样帧
                "detected": True,
                "object_count": len(mask_dict.labels),
                "object_ids": list(mask_dict.labels.keys()),
                "timestamp": start_frame_idx / max(fps, 1),
            })

            if (
                vis_dir is not None
                and vis_counter < max_visualizations
                and mask_dict.labels
            ):
                self._save_structure_visualization(
                    image,
                    mask_dict,
                    vis_dir,
                    start_frame_idx,
                )
                vis_counter += 1

            for out_frame_idx, out_obj_ids, out_mask_logits in self.detection_engine.propagate(
                inference_state,
                step,
                start_frame_idx,
            ):
                frame_masks = MaskDictionary()
                frame_data: Dict[int, Dict] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = out_mask_logits[i] > 0.0
                    class_name = mask_dict.get_class_name(out_obj_id) if out_obj_id in mask_dict.labels else ""
                    mask_2d = out_mask[0] if out_mask.ndim == 3 else out_mask
                    obj = self._object_info_from_mask(mask_2d, class_name, out_obj_id)
                    frame_masks.labels[out_obj_id] = obj
                    frame_data[out_obj_id] = obj.to_serializable()
                sam2_masks = frame_masks.clone()
                video_object_data.append(frame_data)
                
                # 记录传播帧的状态
                frame_states.append({
                    "frame_index": out_frame_idx,
                    "frame_type": "propagated",  # 传播帧
                    "detected": False,  # 传播帧不是检测得到的
                    "object_count": len(frame_data),
                    "object_ids": list(frame_data.keys()),
                    "source_frame": start_frame_idx,  # 来源采样帧
                    "timestamp": out_frame_idx / max(fps, 1),
                })

                if vis_dir is not None and vis_counter < max_visualizations:
                    if 0 <= out_frame_idx < len(frames):
                        self._save_structure_visualization(
                            frames[out_frame_idx],
                            frame_masks,
                            vis_dir,
                            out_frame_idx,
                        )
                        vis_counter += 1

        video_array = read_video_from_path(video_path)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)[None].float()
        # 使用所有帧进行评估（包含采样帧和传播帧）
        # 这样可以更准确地检测消失/出现事件
        tracking_result = video_object_data  # 使用完整的video_object_data，包含所有帧
        disappear_objects = get_disappear_objects(tracking_result)
        appear_objects = get_appear_objects(tracking_result)
        
        # CoTracker验证（可选）
        if self.config.enable_cotracker and self.event_evaluator is not None:
            vanish_score, emerge_score = self.event_evaluator.score(
                video_tensor,
                tracking_result,
                objects_count,
            )
            print(f"[评估] CoTracker验证结果: vanish_score={vanish_score:.3f}, emerge_score={emerge_score:.3f}")
        else:
            # 仅使用SAM2评估，基于消失/出现对象的数量和持续时间计算分数
            vanish_score, emerge_score = self._compute_sam2_only_scores(
                disappear_objects,
                appear_objects,
                tracking_result,
                objects_count,
                fps,
            )
            # 检测失败会严重影响一致性分数
            if detection_failures:
                # 每个检测失败都会降低分数
                failure_penalty = len(detection_failures) * 0.1
                vanish_score = max(0.0, vanish_score - failure_penalty)
                emerge_score = max(0.0, emerge_score - failure_penalty)
                print(f"[评估] 检测到{len(detection_failures)}个检测失败异常，已降低一致性分数")
            print(f"[评估] 仅使用SAM2评估（CoTracker已禁用）: vanish_score={vanish_score:.3f}, emerge_score={emerge_score:.3f}")
        
        coherence_score = (vanish_score + emerge_score) / 2

        anomalies = self._build_structure_anomalies(
            disappear_objects,
            appear_objects,
            vanish_score,
            emerge_score,
            fps=fps,
        )
        anomalies.extend(
            self._detect_shape_anomalies(
                tracking_result,
                fps=fps,
            )
        )
        # 添加检测失败异常（采样帧未检测到但之前有传播对象）
        anomalies.extend(
            self._build_detection_failure_anomalies(
                detection_failures,
                fps=fps,
            )
        )
        # 区域时序变化检测（复用区域分析逻辑，使用SAM2的mask）
        if self.config.enable_region_temporal_analysis and REGION_ANALYSIS_AVAILABLE:
            region_anomalies = self._analyze_region_temporal_changes(
                frames,
                video_object_data,  # 使用完整的video_object_data，而不是过滤后的tracking_result
                fps=fps,
            )
            anomalies.extend(region_anomalies)

        if self.config.cotracker_visualization_enable:
            self._save_cotracker_visualization(
                video_tensor=video_tensor,
                video_path=video_path,
                fps=fps,
            )

        # 统计检测失败信息
        detection_failure_stats = {
            "detection_failure_count": len(detection_failures),
            "detection_failure_frames": [f.get('frame', 0) for f in detection_failures],
            "detection_failure_details": [
                {
                    "frame": f.get('frame', 0),
                    "propagated_objects": f.get('propagated_objects', []),
                    "propagated_objects_count": len(f.get('propagated_objects', [])),
                    "description": f.get('description', ''),
                }
                for f in detection_failures
            ],
        }
        
        # 收集阈值信息
        thresholds_info = {
            "size_change_area_ratio_threshold": self.config.size_change_area_ratio_threshold,
            "size_change_height_ratio_threshold": self.config.size_change_height_ratio_threshold,
            "size_change_min_area": self.config.size_change_min_area,
            "iou_threshold": self.config.iou_threshold,
        }
        
        metadata = {
            "objects_count": objects_count,
            "tracking_result_length": len(tracking_result),  # 所有帧的数量（采样帧 + 传播帧）
            "total_frames": len(frames),  # 视频总帧数
            "step": step,
            "detection_failures": detection_failure_stats,
            "frame_states": frame_states,  # 每帧的状态信息
            "thresholds": thresholds_info,  # 阈值配置
            "video_object_data": video_object_data,  # 完整的视频对象数据（包含每帧的mask信息），用于光流分析
        }
        return TemporalCoherenceResult(
            coherence_score=coherence_score,
            vanish_score=vanish_score,
            emerge_score=emerge_score,
            anomalies=anomalies,
            metadata=metadata,
        )

    @staticmethod
    def _get_visualization_dir(base_dir: Optional[str], video_path: str) -> Path:
        if base_dir:
            root = Path(base_dir).expanduser().resolve()
        else:
            root = Path(__file__).resolve().parents[3] / "outputs" / "structure_visualization"
        root.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem if video_path else "video"
        target = root / video_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _get_cotracker_visualization_dir(base_dir: Optional[str], video_path: str) -> Path:
        if base_dir:
            root = Path(base_dir).expanduser().resolve()
        else:
            root = Path(__file__).resolve().parents[3] / "outputs" / "cotracker_visualization"
        root.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem if video_path else "video"
        target = root / video_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _save_cotracker_visualization(
        self,
        video_tensor: torch.Tensor,
        video_path: str,
        fps: int,
    ) -> None:
        if not self.config.cotracker_visualization_enable:
            return
        if not self.config.enable_cotracker:
            print("警告：CoTracker 验证未启用，无法生成 CoTracker 可视化结果。请使用 --enable_cotracker 启用。")
            return
        if self.cotracker_model is None:
            print("警告：CoTracker 模型未初始化，无法生成可视化结果。")
            return

        output_dir = self._get_cotracker_visualization_dir(
            self.config.cotracker_visualization_output_dir,
            video_path,
        )

        try:
            device = next(self.cotracker_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        video_on_device = video_tensor.to(device)
        fps_value = self.config.cotracker_visualization_fps or fps
        fps_value = max(1, int(fps_value))
        
        # 根据配置决定是否生成整体视频的追踪可视化
        if self.config.cotracker_visualization_full_video:
            try:
                tracks, visibility = self.cotracker_model(
                    video_on_device,
                    grid_size=self.config.grid_size,
                    grid_query_frame=0,
                    backward_tracking=True,
                )
            except Exception as exc:
                print(f"警告：生成 CoTracker 整体可视化失败：{exc}")
            else:
                visualizer = Visualizer(
                    save_dir=str(output_dir),
                    fps=fps_value,
                    mode=self.config.cotracker_visualization_mode,
                )

                try:
                    visualizer.visualize(
                        video=video_tensor.cpu(),
                        tracks=tracks.cpu(),
                        visibility=visibility.cpu(),
                        filename="cotracker_tracks_full",
                        save_video=True,
                    )
                    print("[CoTracker可视化] 已保存整体视频的追踪可视化")
                except Exception as exc:
                    print(f"警告：导出 CoTracker 整体可视化视频失败：{exc}")
        else:
            print("[CoTracker可视化] 已禁用整体视频的追踪可视化（仅生成聚焦于目标mask的可视化）")

    def _save_structure_visualization(
        self,
        frame_image,
        frame_masks: MaskDictionary,
        output_dir: Path,
        frame_idx: int,
    ) -> None:
        if not frame_masks.labels:
            return

        frame_rgb = np.array(frame_image.convert("RGB"), dtype=np.uint8)
        overlay = frame_rgb.copy()

        palette = self._PALETTE
        fill_color = np.array([0, 0, 0], dtype=np.uint8)
        for color_idx, (instance_id, obj) in enumerate(frame_masks.labels.items()):
            mask_tensor = obj.mask
            if mask_tensor is None:
                continue
            mask_np = mask_tensor.cpu().numpy().astype(bool)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            overlay[mask_np] = fill_color
            border_color = palette[color_idx % len(palette)]

            if obj.x1 is not None and obj.y1 is not None and obj.x2 is not None and obj.y2 is not None:
                cv2.rectangle(
                    overlay,
                    (int(obj.x1), int(obj.y1)),
                    (int(obj.x2), int(obj.y2)),
                    border_color.tolist(),
                    2,
                )
                label_text = f"{instance_id}:{obj.class_name}" if obj.class_name else str(instance_id)
                cv2.putText(
                    overlay,
                    label_text,
                    (int(obj.x1), max(0, int(obj.y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    border_color.tolist(),
                    1,
                    cv2.LINE_AA,
                )

        blended = cv2.addWeighted(frame_rgb, 0.5, overlay, 0.5, 0.0)
        save_path = output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    def process_meta_info(self, meta_infos: List[Dict]) -> List[Dict]:
        results = []
        for meta_info in tqdm(meta_infos, desc="Temporal coherence"):
            try:
                subject = meta_info.get("subject_noun")
                prompts = [subject] if subject else None
                result = self.evaluate_video(meta_info["filepath"], prompts)
                meta_info = dict(meta_info)
                meta_info["temporal_coherence_score"] = result.coherence_score
                meta_info["temporal_coherence_vanish"] = result.vanish_score
                meta_info["temporal_coherence_emerge"] = result.emerge_score
                meta_info["temporal_coherence_objects_count"] = result.metadata.get("objects_count", 0)
            except Exception as exc:
                meta_info = dict(meta_info)
                meta_info["temporal_coherence_error"] = str(exc)
            results.append(meta_info)
        return results

    def process_meta_info_file(self, meta_info_path: Optional[str] = None) -> List[Dict]:
        path = meta_info_path or self.config.meta_info_path
        with open(path, "r", encoding="utf-8") as f:
            meta_infos = json.load(f)
        processed = self.process_meta_info(meta_infos)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=4, ensure_ascii=False)
        return processed

    def _build_structure_anomalies(
        self,
        disappear_objects: List[Dict],
        appear_objects: List[Dict],
        vanish_score: float,
        emerge_score: float,
        fps: int,
    ) -> List[Dict[str, Any]]:
        anomalies: List[Dict[str, Any]] = []

        def _to_tensor_mask(mask: Any) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if isinstance(mask, torch.Tensor):
                return mask
            mask_array = np.array(mask)
            if mask_array.ndim == 2:
                mask_array = mask_array.astype(np.float32)
            return torch.from_numpy(mask_array)

        vanish_confidence = max(0.0, min(1.0, 1.0 - vanish_score))
        emerge_confidence = max(0.0, min(1.0, 1.0 - emerge_score))
        fps_safe = max(fps, 1)

        for obj in disappear_objects:
            mask_tensor = _to_tensor_mask(obj.get("mask"))
            frame_id = int(obj.get("last_frame", obj.get("first_appearance", 0)))
            anomalies.append(
                {
                    "type": "structural_disappearance",
                    "modality": "structure",
                    "frame_id": frame_id,
                    "timestamp": frame_id / fps_safe,
                    "confidence": vanish_confidence,
                    "description": "Object disappeared abruptly",
                    "location": {"mask": mask_tensor},
                    "metadata": {
                        "object_id": obj.get("object_id"),
                        "first_appearance": obj.get("first_appearance"),
                        "last_frame": obj.get("last_frame"),
                    },
                }
            )

        for obj in appear_objects:
            mask_tensor = _to_tensor_mask(obj.get("mask"))
            frame_id = int(obj.get("first_appearance", 0))
            anomalies.append(
                {
                    "type": "structural_appearance",
                    "modality": "structure",
                    "frame_id": frame_id,
                    "timestamp": frame_id / fps_safe,
                    "confidence": emerge_confidence,
                    "description": "Object appeared unexpectedly",
                    "location": {"mask": mask_tensor},
                    "metadata": {
                        "object_id": obj.get("object_id"),
                        "first_appearance": obj.get("first_appearance"),
                    },
                }
            )

        return anomalies

    def _detect_shape_anomalies(
        self,
        tracking_result: List[Dict],
        fps: int,
    ) -> List[Dict[str, Any]]:
        anomalies: List[Dict[str, Any]] = []
        if not tracking_result:
            return anomalies

        prev_stats: Dict[int, Dict[str, float]] = {}
        fps_safe = max(fps, 1)
        area_ratio_threshold = max(1.0, self.config.size_change_area_ratio_threshold)
        height_ratio_threshold = max(1.0, self.config.size_change_height_ratio_threshold)
        min_area = max(0, self.config.size_change_min_area)

        for frame_idx, frame_objects in enumerate(tracking_result):
            if not frame_objects:
                continue
            for obj_id_str, obj_meta in frame_objects.items():
                try:
                    obj_id = int(obj_id_str)
                except (TypeError, ValueError):
                    obj_id = obj_meta.get("instance_id", obj_id_str)

                mask_data = obj_meta.get("mask")
                if mask_data is None:
                    continue
                mask_np = np.array(mask_data)
                if mask_np.size == 0:
                    continue
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                mask_bool = mask_np.astype(bool)
                area = float(mask_bool.sum())
                if area < min_area:
                    prev_stats[obj_id] = {
                        "area": area,
                        "height": float(obj_meta.get("y2", 0) - obj_meta.get("y1", 0) + 1),
                    }
                    continue

                bbox_height = 0.0
                y1 = obj_meta.get("y1")
                y2 = obj_meta.get("y2")
                if y1 is not None and y2 is not None:
                    bbox_height = float(max(1, int(y2) - int(y1) + 1))

                prev = prev_stats.get(obj_id)
                if prev:
                    prev_area = max(prev["area"], 1.0)
                    prev_height = max(prev["height"], 1.0)

                    area_ratio = max(area / prev_area, prev_area / area)
                    height_ratio = max(bbox_height / prev_height, prev_height / bbox_height) if prev_height > 0 else 1.0

                    if area_ratio >= area_ratio_threshold or height_ratio >= height_ratio_threshold:
                        anomalies.append(
                            {
                                "type": "structural_size_jump",
                                "modality": "structure",
                                "frame_id": frame_idx,
                                "timestamp": frame_idx / fps_safe,
                                "confidence": min(1.0, max(area_ratio / area_ratio_threshold, height_ratio / height_ratio_threshold)),
                                "description": (
                                    f"Object {obj_id} size changed abruptly "
                                    f"(area_ratio={area_ratio:.2f}, height_ratio={height_ratio:.2f})"
                                ),
                                "metadata": {
                                    "object_id": obj_id,
                                    "area_ratio": area_ratio,
                                    "height_ratio": height_ratio,
                                    "current_area": area,
                                    "previous_area": prev_area,
                                    "current_height": bbox_height,
                                    "previous_height": prev_height,
                                },
                            }
                        )

                prev_stats[obj_id] = {"area": area, "height": bbox_height}

        return anomalies

    def _build_detection_failure_anomalies(
        self,
        detection_failures: List[Dict],
        fps: int,
    ) -> List[Dict[str, Any]]:
        """
        构建检测失败异常（采样帧未检测到但之前有传播对象）
        
        这种情况说明：
        1. 对象突然消失（检测不到）
        2. 对象突然畸变（形状变化太大，检测不到）
        3. 时序一致性很差
        
        Args:
            detection_failures: 检测失败列表
            fps: 帧率
        
        Returns:
            异常列表
        """
        anomalies: List[Dict[str, Any]] = []
        fps_safe = max(fps, 1)
        
        for failure in detection_failures:
            frame_id = failure.get('frame', 0)
            propagated_objects = failure.get('propagated_objects', [])
            description = failure.get('description', '')
            
            # 为每个传播对象创建异常
            for obj_id in propagated_objects:
                anomalies.append({
                    "type": "structural_detection_failure",
                    "modality": "structure",
                    "frame_id": frame_id,
                    "timestamp": frame_id / fps_safe,
                    "confidence": 0.9,  # 高置信度：采样帧应该检测到但没检测到
                    "description": f"采样帧{frame_id}未检测到对象{obj_id}，但之前有传播对象，说明对象突然消失或畸变",
                    "metadata": {
                        "object_id": obj_id,
                        "failure_frame": frame_id,
                        "propagated_objects_count": len(propagated_objects),
                        "reason": "采样帧检测失败但之前有传播对象，可能是对象突然消失或形状畸变",
                    },
                })
        
        return anomalies

    def _analyze_region_temporal_changes(
        self,
        frames: List[np.ndarray],
        video_object_data: List[Dict],
        fps: int,
    ) -> List[Dict[str, Any]]:
        """
        使用区域时序变化检测逻辑分析SAM2检测到的对象
        
        对每个检测到的对象，使用RegionTemporalChangeDetector进行时序变化检测
        复用区域分析的有效逻辑
        
        Args:
            frames: 视频帧序列
            video_object_data: SAM2完整追踪数据（每帧的对象信息）
            fps: 帧率
        
        Returns:
            区域时序变化异常列表
        """
        if not REGION_ANALYSIS_AVAILABLE:
            print("[警告] 区域时序变化检测模块不可用，跳过该分析")
            return []
        
        if not video_object_data or len(video_object_data) == 0:
            return []
        
        # 初始化光流分析器（如果还没有）
        from src.temporal_reasoning.motion_flow.flow_analyzer import MotionFlowAnalyzer
        from src.temporal_reasoning.core.config import RAFTConfig
        
        if not hasattr(self, '_flow_analyzer') or self._flow_analyzer is None:
            # 使用默认RAFT配置
            raft_config = RAFTConfig()
            self._flow_analyzer = MotionFlowAnalyzer(raft_config)
            self._flow_analyzer.initialize()
        
        # 使用配置的区域时序变化检测配置，或使用默认配置
        region_config = self.config.region_temporal_config
        if region_config is None:
            region_config = RegionTemporalChangeConfig()
        
        detector = RegionTemporalChangeDetector(self._flow_analyzer, region_config)
        
        # 提取每帧的对象mask
        # video_object_data 结构: List[Dict[int, Dict]]，每个Dict的key是object_id
        # 需要转换为每帧的mask列表，对每个对象分别分析
        
        all_anomalies: List[Dict[str, Any]] = []
        
        # 收集所有对象的ID
        all_object_ids = set()
        for frame_data in video_object_data:
            if isinstance(frame_data, dict):
                all_object_ids.update(frame_data.keys())
        
        # 对每个对象进行时序变化检测
        for obj_id in all_object_ids:
            # 提取该对象在所有帧中的mask
            object_masks: List[Optional[np.ndarray]] = []
            
            for frame_data in video_object_data:
                if isinstance(frame_data, dict) and obj_id in frame_data:
                    obj_info = frame_data[obj_id]
                    # 从序列化的数据中恢复mask
                    if isinstance(obj_info, dict):
                        mask_data = obj_info.get('mask')
                        if mask_data is not None:
                            # mask可能是list（序列化后的numpy数组）
                            if isinstance(mask_data, list):
                                mask_np = np.array(mask_data, dtype=bool)
                            elif isinstance(mask_data, torch.Tensor):
                                mask_np = mask_data.cpu().numpy().astype(bool)
                            else:
                                mask_np = np.array(mask_data, dtype=bool)
                            
                            # 确保mask是2D
                            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                                mask_np = mask_np[0]
                            elif mask_np.ndim == 3:
                                mask_np = mask_np[:, :, 0] if mask_np.shape[2] == 1 else mask_np
                            
                            object_masks.append(mask_np.astype(bool))
                        else:
                            object_masks.append(None)
                    else:
                        object_masks.append(None)
                else:
                    object_masks.append(None)
            
            # 如果对象在至少一帧中出现，进行时序变化检测
            if any(m is not None for m in object_masks):
                # 确保frames和masks长度一致
                min_len = min(len(frames), len(object_masks))
                frames_subset = frames[:min_len]
                masks_subset = object_masks[:min_len]
                
                # 将PIL Image对象转换为numpy数组（RegionTemporalChangeDetector需要numpy数组）
                frames_np = []
                for frame in frames_subset:
                    if isinstance(frame, Image.Image):
                        # PIL Image转换为numpy数组
                        frames_np.append(np.array(frame.convert("RGB")))
                    elif isinstance(frame, np.ndarray):
                        frames_np.append(frame)
                    else:
                        # 尝试转换为numpy数组
                        frames_np.append(np.array(frame))
                
                # 获取对象类别名称
                class_name = ""
                for frame_data in video_object_data:
                    if isinstance(frame_data, dict) and obj_id in frame_data:
                        obj_info = frame_data[obj_id]
                        if isinstance(obj_info, dict):
                            class_name = obj_info.get('class_name', '')
                            break
                
                label = f"object_{obj_id}_{class_name}" if class_name else f"object_{obj_id}"
                
                try:
                    # 使用区域时序变化检测器分析
                    region_result = detector.analyze(
                        frames_np,
                        masks_subset,
                        fps=fps,
                        label=label,
                    )
                    
                    # 将区域异常转换为结构异常格式，并添加对象信息
                    for anomaly in region_result.get('anomalies', []):
                        anomaly['metadata'] = anomaly.get('metadata', {})
                        anomaly['metadata']['object_id'] = obj_id
                        anomaly['metadata']['class_name'] = class_name
                        anomaly['type'] = 'structural_region_temporal_change'
                        anomaly['modality'] = 'structure'
                        all_anomalies.append(anomaly)
                except Exception as e:
                    print(f"[警告] 对象 {obj_id} 的区域时序变化检测失败: {e}")
                    continue
        
        if all_anomalies:
            print(f"[区域时序分析] 检测到 {len(all_anomalies)} 个区域时序变化异常")
        
        return all_anomalies


def read_video_fps(video_path: str) -> Optional[int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    try:
        return int(fps)
    except (TypeError, ValueError):
        return None

