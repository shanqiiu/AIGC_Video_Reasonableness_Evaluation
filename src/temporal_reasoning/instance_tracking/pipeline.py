from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
COTRACKER_ROOT = _ROOT / "third_party" / "co-tracker"
if COTRACKER_ROOT.exists() and str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore
from cotracker.utils.visualizer import read_video_from_path  # type: ignore

from .detection import DetectionConfig, Sam2DetectionEngine
from .evaluation import TemporalEventEvaluator
from .mask_manager import MaskDictionary
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


class TemporalCoherencePipeline:
    """
    End-to-end pipeline that mirrors the logic in `temporal_coherence_score.py`
    while offering a reusable and testable interface.
    """

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

    def _build_text_prompt(self, meta_info: Dict) -> str:
        subject = meta_info.get("subject_noun", "").strip()
        if not subject:
            return self.config.text_prompt
        if subject.endswith("."):
            return subject
        return f"{subject}."

    def _prepare_tracking_result(self, video_object_data: List[Dict], step: int) -> List[Dict]:
        if not video_object_data:
            return []
        filtered = [item for idx, item in enumerate(video_object_data, start=1) if idx % (step + 1) != 0]
        return filtered[::max(step, 1)]

    def _object_info_from_mask(self, mask, class_name: str, instance_id: int) -> ObjectInfo:
        obj = ObjectInfo(instance_id=instance_id, mask=mask, class_name=class_name)
        obj.update_box()
        return obj

    def _process_single_video(self, meta_info: Dict) -> Tuple[float, float, float]:
        video_path = meta_info["filepath"]
        frames, _ = extract_frames_from_video(video_path)
        fps_value = read_video_fps(video_path)
        fps = max(1, int(fps_value) if fps_value else 24)
        step = max(1, fps - 1)

        inference_state = self.detection_engine.init_video_state(video_path)
        sam2_masks = MaskDictionary()
        objects_count = 0
        video_object_data: List[Dict] = []

        for start_frame_idx in range(0, len(frames), step):
            image = frames[start_frame_idx]
            mask_dict = self.detection_engine.detect(image, self._build_text_prompt(meta_info))

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

            for out_frame_idx, out_obj_ids, out_mask_logits in self.detection_engine.propagate(
                inference_state, step, start_frame_idx
            ):
                frame_masks = MaskDictionary()
                frame_data: Dict[int, Dict] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    class_name = mask_dict.get_class_name(out_obj_id) if out_obj_id in mask_dict.labels else ""
                    mask_2d = out_mask[0] if out_mask.ndim == 3 else out_mask
                    obj = self._object_info_from_mask(mask_2d, class_name, out_obj_id)
                    frame_masks.labels[out_obj_id] = obj
                    frame_data[out_obj_id] = obj.to_serializable()
                sam2_masks = frame_masks.clone()
                video_object_data.append(frame_data)

        video_array = read_video_from_path(video_path)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)[None].float()
        tracking_result = self._prepare_tracking_result(video_object_data, step)
        if self.event_evaluator is None:
            raise RuntimeError("TemporalEventEvaluator is not initialized.")
        vanish_score, emerge_score = self.event_evaluator.score(video_tensor, tracking_result, objects_count)
        coherence_score = (vanish_score + emerge_score) / 2
        return coherence_score, vanish_score, emerge_score

    def process_meta_info(self, meta_infos: List[Dict]) -> List[Dict]:
        results = []
        for meta_info in tqdm(meta_infos, desc="Temporal coherence"):
            try:
                coherence_score, vanish_score, emerge_score = self._process_single_video(meta_info)
                meta_info = dict(meta_info)
                meta_info["temporal_coherence_score"] = coherence_score
                meta_info["temporal_coherence_vanish"] = vanish_score
                meta_info["temporal_coherence_emerge"] = emerge_score
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

