from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..motion_flow.flow_analyzer import MotionFlowAnalyzer


@dataclass
class TongueFlowChangeConfig:
    """
    控制舌头区域光流 / 外观突变检测的参数。
    """

    motion_threshold: float = 6.0
    similarity_threshold: float = 0.25
    consecutive_frames: int = 1
    baseline_window: int = 5
    min_roi_size: int = 12
    use_color_similarity: bool = True
    use_flow_change: bool = True
    hist_diff_threshold: float = 0.012


class TongueFlowChangeDetector:
    """
    利用 RAFT 光流与局部特征变化检测舌头突变。

    使用流程：
        1. 初始化时复用现有 MotionFlowAnalyzer。
        2. 调用 analyze() 时传入视频帧序列以及每帧嘴部 ROI（mask 或 bbox）。
        3. 在 ROI 处计算光流幅值分布对比、颜色直方图相似度，一旦发生突变持续若干帧，
           输出 `tongue_flow_change` 异常。
    """

    def __init__(self, flow_analyzer: MotionFlowAnalyzer, config: Optional[TongueFlowChangeConfig] = None):
        self.flow_analyzer = flow_analyzer
        self.config = config or TongueFlowChangeConfig()

    def analyze(
        self,
        video_frames: Sequence[np.ndarray],
        mouth_masks: Sequence[Optional[np.ndarray]],
        fps: float = 30.0,
        label: str = "tongue",
    ) -> Dict[str, object]:
        if not video_frames:
            raise ValueError("video_frames 为空")
        if len(video_frames) != len(mouth_masks):
            raise ValueError("mouth_masks 与 video_frames 长度不匹配")

        roi_stats = self._extract_roi_stats(video_frames, mouth_masks)
        flow_diffs = self._compute_flow_change(video_frames, mouth_masks)
        anomalies, frame_stats, baseline_motion, max_hist_diff = self._detect_anomalies(
            roi_stats, flow_diffs, fps, label
        )
        score = 0.0 if anomalies else 1.0

        return {
            "score": score,
            "anomalies": anomalies,
            "metadata": {
                "use_flow_change": self.config.use_flow_change,
                "use_color_similarity": self.config.use_color_similarity,
                "motion_threshold": self.config.motion_threshold,
                "similarity_threshold": self.config.similarity_threshold,
                "hist_diff_threshold": self.config.hist_diff_threshold,
                "baseline_motion": baseline_motion,
                "frame_stats": frame_stats,
                "max_hist_diff": max_hist_diff,
            },
        }

    def _extract_roi_stats(
        self,
        video_frames: Sequence[np.ndarray],
        mouth_masks: Sequence[Optional[np.ndarray]],
    ) -> List[Dict[str, float]]:
        stats: List[Dict[str, float]] = []

        for frame, mask in zip(video_frames, mouth_masks):
            if mask is None:
                stats.append({"hist_similarity": 1.0, "valid": False})
                continue

            if mask.dtype == bool:
                roi_pixels = frame[mask]
            else:
                roi_pixels = frame[mask > 0]
            if roi_pixels.size == 0 or roi_pixels.shape[0] < self.config.min_roi_size:
                stats.append({"hist_similarity": 1.0, "valid": False})
                continue

            hist = self._compute_histogram(roi_pixels)
            stats.append({"hist": hist, "hist_similarity": 1.0, "valid": True})

        baseline_hist = self._compute_baseline_histogram(stats)
        for stat in stats:
            if not stat.get("valid"):
                continue
            hist = stat["hist"]
            similarity = self._histogram_similarity(hist, baseline_hist)
            stat["hist_similarity"] = similarity
        return stats

    def _compute_flow_change(
        self,
        video_frames: Sequence[np.ndarray],
        mouth_masks: Sequence[Optional[np.ndarray]],
    ) -> List[float]:
        if not self.config.use_flow_change or len(video_frames) < 2:
            return [0.0 for _ in video_frames]

        flows = self._compute_raft_flows(video_frames)
        flow_diffs: List[float] = [0.0]

        for idx in range(1, len(video_frames)):
            prev_mask = mouth_masks[idx - 1]
            if prev_mask is None:
                flow_diffs.append(0.0)
                continue

            flow = flows[idx - 1]
            if flow is None:
                flow_diffs.append(0.0)
                continue

            u, v = flow
            magnitude = np.sqrt(u ** 2 + v ** 2)
            if prev_mask.dtype == bool:
                roi_values = magnitude[prev_mask]
            else:
                roi_values = magnitude[prev_mask > 0]
            if roi_values.size == 0:
                flow_diffs.append(0.0)
            else:
                mean_flow = float(np.mean(roi_values))
                flow_diffs.append(mean_flow)
        return flow_diffs

    def _compute_raft_flows(self, video_frames: Sequence[np.ndarray]) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        if self.flow_analyzer.raft_model is None:
            self.flow_analyzer.initialize()
        if self.flow_analyzer.raft_model is None:
            raise RuntimeError("RAFT 模型未能成功初始化，无法计算光流。")

        flows: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for idx in range(len(video_frames) - 1):
            frame_a = self._ensure_uint8(video_frames[idx])
            frame_b = self._ensure_uint8(video_frames[idx + 1])
            try:
                u, v = self.flow_analyzer.raft_model.compute_flow(frame_a, frame_b)
            except Exception:
                u = v = None
            flows.append((u, v) if u is not None and v is not None else None)
        return flows

    @staticmethod
    def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        frame_clip = np.clip(frame, 0, 255).astype(np.uint8)
        return frame_clip

    def _detect_anomalies(
        self,
        roi_stats: List[Dict[str, float]],
        flow_diffs: List[float],
        fps: float,
        label: str,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], float, float]:
        anomalies: List[Dict[str, object]] = []
        frame_stats: List[Dict[str, object]] = []

        consecutive = 0
        baseline_motion = self._compute_baseline_motion(flow_diffs)
        fps_safe = max(fps, 1.0)
        prev_hist_similarity: Optional[float] = None
        max_hist_diff = 0.0

        for idx, (stat, flow_val) in enumerate(zip(roi_stats, flow_diffs)):
            valid = stat.get("valid", False)
            hist_similarity = stat.get("hist_similarity", 1.0)
            hist_diff = (
                abs(hist_similarity - prev_hist_similarity)
                if prev_hist_similarity is not None
                else 0.0
            )
            prev_hist_similarity = hist_similarity
            if hist_diff > max_hist_diff:
                max_hist_diff = hist_diff

            motion_change = abs(flow_val - baseline_motion)
            similarity_drop = 1.0 - hist_similarity

            frame_entry: Dict[str, object] = {
                "frame_id": idx,
                "timestamp": idx / fps_safe,
                "hist_similarity": float(hist_similarity),
                "similarity_drop": float(similarity_drop),
                "hist_diff": float(hist_diff),
                "motion_value": float(flow_val),
                "motion_change": float(motion_change),
            }

            if not valid:
                frame_entry.update({"triggers": [], "valid": False})
                frame_stats.append(frame_entry)
                consecutive = 0
                continue

            flow_trigger = self.config.use_flow_change and motion_change >= self.config.motion_threshold
            color_trigger = self.config.use_color_similarity and similarity_drop >= self.config.similarity_threshold
            hist_diff_trigger = (
                self.config.use_color_similarity and hist_diff >= self.config.hist_diff_threshold
            )

            triggers = []
            if flow_trigger:
                triggers.append("flow")
            if color_trigger:
                triggers.append("color")
            if hist_diff_trigger:
                triggers.append("hist_diff")

            frame_entry.update({"triggers": triggers, "valid": True})
            frame_stats.append(frame_entry)

            if flow_trigger or color_trigger or hist_diff_trigger:
                consecutive += 1
                if consecutive >= self.config.consecutive_frames:
                    anomalies.append(
                        {
                            "type": "tongue_flow_change",
                            "modality": "appearance",
                            "frame_id": idx,
                            "timestamp": idx / fps_safe,
                            "confidence": min(
                                1.0,
                                max(
                                    motion_change / (self.config.motion_threshold * 2),
                                    similarity_drop / (self.config.similarity_threshold * 2),
                                    hist_diff / (self.config.hist_diff_threshold * 2)
                                ),
                            ),
                            "description": (
                                f"{label} region change detected "
                                f"(motion={motion_change:.2f}, similarity_drop={similarity_drop:.2f}, hist_diff={hist_diff:.3f}, triggers={triggers})"
                            ),
                            "metadata": {
                                "motion_change": motion_change,
                                "similarity_drop": similarity_drop,
                                "baseline_motion": baseline_motion,
                                "hist_similarity": hist_similarity,
                                "hist_diff": hist_diff,
                                "flow_value": flow_val,
                                "triggers": triggers,
                            },
                        }
                    )
                    consecutive = 0
            else:
                consecutive = 0

        return anomalies, frame_stats, baseline_motion, max_hist_diff

    def _compute_baseline_histogram(self, stats: List[Dict[str, float]]) -> np.ndarray:
        baseline_window = max(1, self.config.baseline_window)
        hist_list = []
        for stat in stats:
            if not stat.get("valid"):
                continue
            hist = stat.get("hist")
            if hist is None:
                continue
            hist_list.append(hist)
            if len(hist_list) >= baseline_window:
                break
        if not hist_list:
            return np.ones(16 * 3, dtype=np.float32) / (16 * 3)
        return np.mean(hist_list, axis=0)

    def _compute_baseline_motion(self, flow_diffs: List[float]) -> float:
        baseline_window = max(1, self.config.baseline_window)
        values = [val for val in flow_diffs[:baseline_window] if val > 0]
        if not values:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _compute_histogram(pixels: np.ndarray, bins: int = 16) -> np.ndarray:
        hist_channels = []
        for channel_idx in range(3):
            channel = pixels[:, channel_idx]
            hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
            hist_channels.append(hist)
        return np.concatenate(hist_channels).astype(np.float32)

    @staticmethod
    def _histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        if hist_a.shape != hist_b.shape:
            raise ValueError("hist_a 与 hist_b 维度不一致")
        eps = 1e-6
        numerator = np.sum(hist_a * hist_b)
        denominator = np.sqrt(np.sum(hist_a ** 2) * np.sum(hist_b ** 2)) + eps
        return float(numerator / denominator)

