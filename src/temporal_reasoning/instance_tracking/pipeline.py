from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
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
    cotracker_visualization_mode: str = "rainbow"


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
        [255, 99, 71],
        [135, 206, 250],
        [144, 238, 144],
        [255, 215, 0],
        [216, 191, 216],
        [255, 182, 193],
        [64, 224, 208],
        [255, 140, 0],
        [173, 216, 230],
        [189, 183, 107],
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

    def evaluate_video(
        self,
        video_path: str,
        text_prompts: Optional[Sequence[str]] = None,
    ) -> TemporalCoherenceResult:
        if self.event_evaluator is None:
            raise RuntimeError("TemporalEventEvaluator is not initialized.")

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

        for start_frame_idx in range(0, len(frames), step):
            image = frames[start_frame_idx]
            mask_dict = self.detection_engine.detect(image, text_prompt)
            detected = bool(mask_dict.labels)
            print(
                f"[Structure] 帧 {start_frame_idx:04d} 检测{'命中' if detected else '未命中'} "
                f"(数量: {len(mask_dict.labels)})"
            )

            if not mask_dict.labels:
                video_object_data.extend([{} for _ in range(step + 1)])
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
        tracking_result = self._prepare_tracking_result(video_object_data, step)
        disappear_objects = get_disappear_objects(tracking_result)
        appear_objects = get_appear_objects(tracking_result)
        vanish_score, emerge_score = self.event_evaluator.score(
            video_tensor,
            tracking_result,
            objects_count,
        )
        coherence_score = (vanish_score + emerge_score) / 2

        anomalies = self._build_structure_anomalies(
            disappear_objects,
            appear_objects,
            vanish_score,
            emerge_score,
            fps=fps,
        )

        if self.config.cotracker_visualization_enable:
            self._save_cotracker_visualization(
                video_tensor=video_tensor,
                video_path=video_path,
                fps=fps,
            )

        metadata = {
            "objects_count": objects_count,
            "tracking_result_length": len(tracking_result),
            "step": step,
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
        if self.cotracker_model is None:
            print("警告: CoTracker 模型未初始化，无法生成可视化。")
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
        try:
            tracks, visibility = self.cotracker_model(
                video_on_device,
                grid_size=self.config.grid_size,
                grid_query_frame=0,
                backward_tracking=True,
            )
        except Exception as exc:
            print(f"警告: 生成 CoTracker 可视化失败: {exc}")
            return

        fps_value = self.config.cotracker_visualization_fps or fps
        fps_value = max(1, int(fps_value))
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
                filename="cotracker_tracks",
                save_video=True,
            )
        except Exception as exc:
            print(f"警告: 保存 CoTracker 可视化视频失败: {exc}")

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
        for color_idx, (instance_id, obj) in enumerate(frame_masks.labels.items()):
            mask_tensor = obj.mask
            if mask_tensor is None:
                continue
            mask_np = mask_tensor.cpu().numpy().astype(bool)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            color = palette[color_idx % len(palette)]
            overlay[mask_np] = (
                0.6 * overlay[mask_np] + 0.4 * color
            ).astype(np.uint8)

            if obj.x1 is not None and obj.y1 is not None and obj.x2 is not None and obj.y2 is not None:
                cv2.rectangle(
                    overlay,
                    (int(obj.x1), int(obj.y1)),
                    (int(obj.x2), int(obj.y2)),
                    color.tolist(),
                    2,
                )
                cv2.putText(
                    overlay,
                    str(instance_id),
                    (int(obj.x1), max(0, int(obj.y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color.tolist(),
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

