from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from ..instance_tracking.detection import DetectionConfig, Sam2DetectionEngine
from ..instance_tracking.mask_manager import MaskDictionary
from ..instance_tracking.types import ObjectInfo
from ..motion_flow.flow_analyzer import MotionFlowAnalyzer
from ..core.config import RAFTConfig
from .tongue_flow_change_detector import TongueFlowChangeConfig, TongueFlowChangeDetector


@dataclass
class TongueAnalysisPipelineConfig:
    detection: DetectionConfig
    raft: RAFTConfig
    flow_change: TongueFlowChangeConfig = TongueFlowChangeConfig()
    prompts: Sequence[str] = ("mouth", "tongue")
    min_area: int = 64
    mask_dilation: int = 2


class TongueAnalysisPipeline:
    """
    使用 GroundingDINO + SAM2 获取嘴部掩膜，并结合 RAFT 光流检测舌头突变。
    """

    def __init__(self, config: TongueAnalysisPipelineConfig):
        self.config = config
        self.detection_engine = Sam2DetectionEngine(config.detection)
        self.flow_analyzer = MotionFlowAnalyzer(config.raft)
        self.change_detector = TongueFlowChangeDetector(self.flow_analyzer, config.flow_change)
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        self.detection_engine.initialize()
        self.flow_analyzer.initialize()
        self._initialized = True

    def analyze(
        self,
        video_frames: Sequence[np.ndarray],
        fps: float = 30.0,
    ) -> Dict[str, object]:
        if not self._initialized:
            self.initialize()
        mouth_masks = self._generate_mouth_masks(video_frames, self.config.prompts)
        result = self.change_detector.analyze(video_frames, mouth_masks, fps=fps)
        coverage = []
        for frame, mask in zip(video_frames, mouth_masks):
            if mask is None:
                coverage.append(0.0)
                continue
            coverage.append(float(mask.sum()) / float(mask.size))
        result["metadata"]["mouth_mask_coverage"] = [
            float(np.clip(val, 0.0, 1.0)) for val in coverage
        ]
        return result

    def _generate_mouth_masks(
        self,
        video_frames: Sequence[np.ndarray],
        prompts: Sequence[str],
    ) -> List[Optional[np.ndarray]]:
        masks: List[Optional[np.ndarray]] = []
        text_prompt = self._build_prompt(prompts)

        for frame in video_frames:
            image = Image.fromarray(self._ensure_uint8(frame))
            mask_dict = self.detection_engine.detect(image, text_prompt)
            mask = self._select_best_mask(mask_dict)
            if mask is not None and self.config.mask_dilation > 0:
                mask = self._dilate_mask(mask, self.config.mask_dilation)
            masks.append(mask)
        return masks

    @staticmethod
    def _build_prompt(prompts: Sequence[str]) -> str:
        valid_prompts = [p.strip() for p in prompts if p and p.strip()]
        if not valid_prompts:
            return "mouth."
        joined = ". ".join(valid_prompts)
        if not joined.endswith("."):
            joined += "."
        return joined

    def _select_best_mask(self, mask_dict: MaskDictionary) -> Optional[np.ndarray]:
        best_mask: Optional[np.ndarray] = None
        best_score = 0.0
        for obj in mask_dict.labels.values():
            if not self._is_mouth_like(obj):
                continue
            mask = obj.mask
            if mask is None:
                continue
            mask_np = mask.cpu().numpy().astype(bool)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            area = float(mask_np.sum())
            if area < self.config.min_area:
                continue
            if area > best_score:
                best_mask = mask_np
                best_score = area
        return best_mask

    @staticmethod
    def _is_mouth_like(obj: ObjectInfo) -> bool:
        label = (obj.class_name or "").lower()
        keywords = ("mouth", "tongue", "lip")
        return any(keyword in label for keyword in keywords)

    @staticmethod
    def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_uint8 = mask.astype(np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        return dilated.astype(bool)

    @staticmethod
    def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame, 0, 255).astype(np.uint8)

