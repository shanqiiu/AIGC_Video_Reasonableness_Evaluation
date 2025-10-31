# -*- coding: utf-8 -*-
"""
Refactored Blur detector facade that adapts existing pipeline to project API.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Optional

from .config import BlurDetectionConfig


class BlurDetector:
    """Primary blur detector that wraps the legacy pipeline with a stable API."""

    def __init__(self, config: Optional[BlurDetectionConfig] = None):
        self.config = config or BlurDetectionConfig()

        # Lazy import legacy pipeline to avoid circulars and heavy deps during module import
        from .blur_detection_pipeline import BlurDetectionPipeline  # type: ignore

        self._pipeline = BlurDetectionPipeline(
            device=self.config.get_device_config("device") or "cuda",
            model_paths=self.config.model_paths,
        )

    def detect(self, video_path: str, subject_noun: str = "person") -> Dict:
        """Detect blur for a single video and return unified result format."""
        start_time = time.time()

        raw = self._pipeline.detect_blur_in_video(video_path, subject_noun=subject_noun)

        unified = self._to_unified_result(video_path, raw, processing_time=time.time() - start_time)
        if self.config.get_output_param("save_json_results"):
            self._save_single_json(unified)
        return unified

    def batch_detect(self, video_dir: str) -> Dict:
        """Run batch detection on a directory of videos. Returns summary and per-video results."""
        start_time = time.time()
        batch_raw = self._pipeline.batch_detect_blur(video_dir, str(self.config.output_dir))

        results: List[Dict] = []
        for item in batch_raw.get("results", []):
            vp = item.get("video_path", "")
            results.append(self._to_unified_result(vp, item))

        summary = {
            "module": "perceptual_quality.blur",
            "video_dir": video_dir,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "device": self.config.get_device_config("device"),
                "processing_time": round(time.time() - start_time, 3),
            },
            "result": {
                "total_videos": batch_raw.get("total_videos", len(results)),
                "processed_videos": batch_raw.get("processed_videos", len(results)),
                "blur_detected_count": batch_raw.get("blur_detected_count", sum(1 for r in results if r.get("result", {}).get("blur_detected"))),
            },
            "results": results,
        }

        if self.config.get_output_param("save_json_results"):
            out_path = os.path.join(str(self.config.output_dir), "batch_blur_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary

    def _to_unified_result(self, video_path: str, raw: Dict, processing_time: Optional[float] = None) -> Dict:
        # Extract fields from legacy output
        blur_detected = bool(raw.get("blur_detected", False))
        confidence = float(raw.get("confidence", 0.0))
        score = float(raw.get("mss_score", 0.0))  # primary score
        severity = raw.get("blur_severity") or self._map_severity(raw.get("blur_severity", ""))
        blur_frames = raw.get("blur_frames", []) or []

        details = {
            "mss_score": float(raw.get("mss_score", 0.0)),
            "pas_score": float(raw.get("pas_score", 0.0)),
            "threshold": float(raw.get("threshold", self.config.get_detection_param("blur_thresholds").get("moderate_blur", 0.025) if isinstance(self.config.get_detection_param("blur_thresholds"), dict) else 0.025)),
        }

        unified = {
            "module": "perceptual_quality.blur",
            "video_path": video_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": {
                "blur_detected": blur_detected,
                "confidence": confidence,
                "score": score,
                "blur_severity": severity,
                "blur_frames": blur_frames,
                "details": details,
            },
            "metadata": {
                "device": self.config.get_device_config("device"),
                "processing_time": round(processing_time, 3) if processing_time is not None else None,
            },
        }

        return unified

    def _save_single_json(self, unified: Dict) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(unified.get("video_path", "video")))[0]
        out_path = os.path.join(str(self.config.output_dir), f"{base}_blur_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(unified, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _map_severity(raw: str) -> str:
        # Map legacy CN labels to EN labels used across modules
        mapping = {
            "严重模糊": "severe",
            "中等模糊": "moderate",
            "轻微模糊": "mild",
            "无模糊": "none",
        }
        return mapping.get(str(raw), str(raw))


