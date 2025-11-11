from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..motion_flow.flow_analyzer import MotionFlowAnalyzer
from ..core.config import RAFTConfig, KeypointConfig
from ..keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from .tongue_flow_change_detector import TongueFlowChangeConfig, TongueFlowChangeDetector


@dataclass
class TongueAnalysisPipelineConfig:
    raft: RAFTConfig
    keypoint: KeypointConfig = KeypointConfig()
    flow_change: TongueFlowChangeConfig = TongueFlowChangeConfig()
    min_mouth_area: int = 64
    mouth_margin_ratio: float = 0.05
    enable_visualization: bool = False
    visualization_output_dir: Optional[str] = None
    visualization_max_frames: Optional[int] = 150


class TongueAnalysisPipeline:
    """
    使用 GroundingDINO + SAM2 获取嘴部掩膜，并结合 RAFT 光流检测舌头突变。
    """

    def __init__(self, config: TongueAnalysisPipelineConfig):
        self.config = config
        self.flow_analyzer = MotionFlowAnalyzer(config.raft)
        self.change_detector = TongueFlowChangeDetector(self.flow_analyzer, config.flow_change)
        self._initialized = False
        self._vis_dir: Optional[Path] = None
        self._vis_counter: int = 0
        self.keypoint_analyzer = KeypointAnalyzer(config.keypoint)
        self._mouth_outer_indices = np.array(
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308], dtype=np.int32
        )

    def initialize(self):
        if self._initialized:
            return
        self.flow_analyzer.initialize()
        self.keypoint_analyzer.initialize()
        self._initialized = True

    def analyze(
        self,
        video_frames: Sequence[np.ndarray],
        fps: float = 30.0,
        video_path: Optional[str] = None,
    ) -> Dict[str, object]:
        if not self._initialized:
            self.initialize()
        self._prepare_visualization(video_path)
        if hasattr(self.keypoint_analyzer.extractor, "reset_timestamp"):
            self.keypoint_analyzer.extractor.reset_timestamp()
        mouth_masks = self._generate_mouth_masks(video_frames, fps)
        result = self.change_detector.analyze(video_frames, mouth_masks, fps=fps)
        coverage = []
        for frame, mask in zip(video_frames, mouth_masks):
            if mask is None:
                coverage.append(0.0)
                continue
            coverage.append(float(mask.sum()) / float(mask.size))
        metadata = result.setdefault("metadata", {})
        metadata["mouth_mask_coverage"] = [float(np.clip(val, 0.0, 1.0)) for val in coverage]
        metadata["visualization_dir"] = str(self._vis_dir) if self._vis_dir else None
        return result

    def _generate_mouth_masks(
        self,
        video_frames: Sequence[np.ndarray],
        fps: float,
    ) -> List[Optional[np.ndarray]]:
        masks: List[Optional[np.ndarray]] = []
        max_frames = self.config.visualization_max_frames or 0

        for idx, frame in enumerate(video_frames):
            frame_uint8 = self._ensure_uint8(frame)
            keypoints = self.keypoint_analyzer.extractor.extract_keypoints(frame_uint8, fps=fps)
            mask = self._build_mouth_mask(keypoints, frame_uint8.shape[0], frame_uint8.shape[1])
            masks.append(mask)
            if self._vis_dir and (max_frames <= 0 or self._vis_counter < max_frames):
                self._save_mask_visualization(frame_uint8, keypoints, mask, idx)
        return masks

    def _build_mouth_mask(
        self,
        keypoints: Dict[str, Optional[np.ndarray]],
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        face = keypoints.get("face")
        if face is None or face.shape[0] == 0:
            return None

        try:
            points = face[self._mouth_outer_indices, :2].copy()
        except Exception:
            return None

        if points.shape[0] != len(self._mouth_outer_indices):
            return None

        points = np.clip(points, 0.0, 1.0)
        pts_px = np.stack(
            [points[:, 0] * width, points[:, 1] * height],
            axis=1,
        )

        if np.any(np.isnan(pts_px)):
            return None

        mask_uint8 = np.zeros((height, width), dtype=np.uint8)
        polygon = pts_px.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask_uint8, [polygon], 1)

        dilation_ratio = max(0.0, float(self.config.mouth_margin_ratio))
        if dilation_ratio > 0.0:
            iterations = max(1, int(round(max(height, width) * dilation_ratio)))
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)

        area = int(mask_uint8.sum())
        if area < self.config.min_mouth_area:
            return None

        return mask_uint8.astype(bool)

    @staticmethod
    def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _prepare_visualization(self, video_path: Optional[str]) -> None:
        if not self.config.enable_visualization:
            self._vis_dir = None
            self._vis_counter = 0
            return

        if self.config.visualization_output_dir:
            base_dir = Path(self.config.visualization_output_dir).expanduser()
        else:
            base_dir = Path(__file__).resolve().parents[3] / "outputs" / "tongue_analysis"
        base_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem if video_path else "video"
        target_dir = (base_dir / video_name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        self._vis_dir = target_dir
        self._vis_counter = 0

    def _save_mask_visualization(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, Optional[np.ndarray]],
        selected_mask: Optional[np.ndarray],
        frame_idx: int,
    ) -> None:
        if self._vis_dir is None:
            return

        frame_uint8 = self._ensure_uint8(frame)
        frame_rgb = frame_uint8.copy()
        overlay = frame_rgb.copy()
        overlay[:] = frame_rgb

        if selected_mask is not None:
            mask_uint8 = selected_mask.astype(np.uint8)
            overlay[mask_uint8.astype(bool)] = (40, 40, 40)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        face = keypoints.get("face")
        if face is not None and face.shape[0] > 0:
            h, w = frame_rgb.shape[:2]
            pts = np.clip(face[self._mouth_outer_indices, :2], 0.0, 1.0)
            pts_px = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1).astype(np.int32)
            for x, y in pts_px:
                cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 255), -1)

        blended = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)
        cv2.putText(
            blended,
            f"frame {frame_idx:04d}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        save_path = self._vis_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        self._vis_counter += 1

