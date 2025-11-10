from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
COTRACKER_ROOT = _ROOT / "third_party" / "co-tracker"
if COTRACKER_ROOT.exists() and str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore

from .tcs_utils import (  # type: ignore
    get_appear_objects,
    get_disappear_objects,
    is_edge_emerge,
    is_edge_vanish,
    is_emerge_detect_error,
    is_small_emerge,
    is_small_vanish,
    is_vanish_detect_error,
)


@dataclass
class CoTrackerConfig:
    checkpoint: str
    grid_size: int
    device: str = "cuda"
    window_len: int = 60


class TemporalEventEvaluator:
    """
    Responsible for measuring vanish/emerge scores using CoTracker.
    """

    def __init__(self, cotracker_model: CoTrackerPredictor, grid_size: int):
        self.cotracker_model = cotracker_model
        self.grid_size = grid_size

    @staticmethod
    def _prepare_masks(objects: List[Dict]) -> Tuple[List[torch.Tensor], List[int]]:
        masks: List[torch.Tensor] = []
        frames: List[int] = []
        for obj in objects:
            mask = obj.get("mask")
            frame = obj.get("first_appearance")
            if mask is None or getattr(mask, "ndim", 0) != 2:
                continue
            image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            tensor_mask = torch.from_numpy(np.array(image))[None, None]
            masks.append(tensor_mask)
            frames.append(frame)
        return masks, frames

    def score(self, video_tensor: torch.Tensor, tracking_result: List[Dict], objects_count: int) -> Tuple[float, float]:
        """
        Args:
            video_tensor: Tensor with shape (1, T, C, H, W)
            tracking_result: Simplified tracking result sampled by step
            objects_count: Total object count from SAM tracking
        """
        device = next(self.cotracker_model.parameters()).device
        video_tensor = video_tensor.to(device)
        _, _, _, video_height, video_width = video_tensor.shape

        tracking_result = tracking_result or []

        disappear_objects = get_disappear_objects(tracking_result)
        vanish_score = self._compute_disappear_score(
            disappear_objects, video_tensor, video_width, video_height, objects_count
        )

        appear_objects = get_appear_objects(tracking_result)
        emerge_score = self._compute_appear_score(
            appear_objects, video_tensor, video_width, video_height, objects_count
        )

        return vanish_score, emerge_score

    def _compute_disappear_score(
        self,
        disappear_objects: List[Dict],
        video_tensor: torch.Tensor,
        video_width: int,
        video_height: int,
        objects_count: int,
    ) -> float:
        if not disappear_objects:
            return 1.0

        masks, frames = self._prepare_masks(disappear_objects)
        disappear_objects_count = 0
        for mask, frame in zip(masks, frames):
            pred_tracks, pred_visibility = self.cotracker_model(
                video_tensor,
                grid_size=self.grid_size,
                grid_query_frame=frame,
                backward_tracking=True,
                segm_mask=mask.to(video_tensor.device),
            )
            edge_vanish = is_edge_vanish(pred_tracks, pred_visibility, frame, video_width, video_height)
            small_vanish = is_small_vanish(pred_tracks, pred_visibility, frame, video_width, video_height)
            disappear_detect_error = is_vanish_detect_error(pred_tracks, pred_visibility, frame)
            if not edge_vanish and not small_vanish and not disappear_detect_error:
                disappear_objects_count += 1
        if objects_count == 0:
            return 1.0
        return (objects_count - disappear_objects_count) / objects_count

    def _compute_appear_score(
        self,
        appear_objects: List[Dict],
        video_tensor: torch.Tensor,
        video_width: int,
        video_height: int,
        objects_count: int,
    ) -> float:
        if not appear_objects:
            return 1.0

        masks, frames = self._prepare_masks(appear_objects)
        appear_objects_count = 0
        for mask, frame in zip(masks, frames):
            pred_tracks, pred_visibility = self.cotracker_model(
                video_tensor,
                grid_size=self.grid_size,
                grid_query_frame=frame,
                backward_tracking=True,
                segm_mask=mask.to(video_tensor.device),
            )
            edge_emerge = is_edge_emerge(pred_tracks, pred_visibility, frame, video_width, video_height)
            small_emerge = is_small_emerge(pred_tracks, pred_visibility, frame, video_width, video_height)
            appear_detect_error = is_emerge_detect_error(pred_tracks, pred_visibility, frame)
            if not edge_emerge and not small_emerge and not appear_detect_error:
                appear_objects_count += 1
        if objects_count == 0:
            return 1.0
        return (objects_count - appear_objects_count) / objects_count

