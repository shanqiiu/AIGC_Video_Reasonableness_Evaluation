from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.temporal_reasoning.core.config import TemporalReasoningConfig, load_config_from_yaml
from src.temporal_reasoning.instance_tracking.detection import DetectionConfig
from src.temporal_reasoning.utils.video_utils import get_video_info, load_video_frames
from src.temporal_reasoning.tongue_analysis.pipeline import TongueAnalysisPipeline, TongueAnalysisPipelineConfig
from src.temporal_reasoning.tongue_analysis.tongue_flow_change_detector import TongueFlowChangeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tongue flow/appearance change analysis on a video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--config",
        help="Temporal reasoning YAML config. If omitted, default config is used.",
    )
    parser.add_argument(
        "--output",
        default="tongue_analysis_report.json",
        help="Path to save the analysis report (JSON).",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["mouth", "tongue"],
        help="Text prompts used for GroundingDINO detection.",
    )
    parser.add_argument(
        "--mask-min-area",
        type=int,
        default=64,
        help="Minimum area (pixels) required to keep a detected mouth mask.",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=2,
        help="Number of dilation iterations applied to mouth masks.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=6.0,
        help="Threshold for motion change within the mouth ROI.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.25,
        help="Threshold for color histogram similarity drop inside the mouth ROI.",
    )
    parser.add_argument(
        "--consecutive-frames",
        type=int,
        default=2,
        help="Number of consecutive frames exceeding thresholds required to flag anomaly.",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=5,
        help="Number of initial frames used to build baseline statistics.",
    )
    parser.add_argument(
        "--disable-flow",
        action="store_true",
        help="Disable optical-flow change detection (only use appearance similarity).",
    )
    parser.add_argument(
        "--disable-color",
        action="store_true",
        help="Disable color histogram similarity check (only use motion).",
    )
    return parser.parse_args()


def build_pipeline_config(
    temporal_config: TemporalReasoningConfig,
    prompts: Sequence[str],
    mask_min_area: int,
    mask_dilation: int,
    motion_threshold: float,
    similarity_threshold: float,
    consecutive_frames: int,
    baseline_window: int,
    disable_flow: bool,
    disable_color: bool,
) -> TongueAnalysisPipelineConfig:
    sam_config_path = temporal_config.sam.config_path
    if not sam_config_path and temporal_config.sam.resolved_config_path:
        sam_config_path = temporal_config.sam.resolved_config_path

    detection_config = DetectionConfig(
        grounding_config_path=temporal_config.grounding_dino.config_path,
        grounding_checkpoint_path=temporal_config.grounding_dino.model_path,
        bert_path=temporal_config.grounding_dino.bert_path,
        sam2_config_path=sam_config_path,
        sam2_checkpoint_path=temporal_config.sam.model_path,
        box_threshold=temporal_config.grounding_dino.box_threshold,
        text_threshold=temporal_config.grounding_dino.text_threshold,
        device=temporal_config.device,
    )

    flow_change_config = TongueFlowChangeConfig(
        motion_threshold=motion_threshold,
        similarity_threshold=similarity_threshold,
        consecutive_frames=consecutive_frames,
        baseline_window=baseline_window,
        use_flow_change=not disable_flow,
        use_color_similarity=not disable_color,
    )

    return TongueAnalysisPipelineConfig(
        detection=detection_config,
        raft=temporal_config.raft,
        flow_change=flow_change_config,
        prompts=prompts,
        min_area=mask_min_area,
        mask_dilation=mask_dilation,
    )


def load_temporal_config(config_path: str | None) -> TemporalReasoningConfig:
    if config_path:
        return load_config_from_yaml(config_path)
    return TemporalReasoningConfig()


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    temporal_config = load_temporal_config(args.config)
    pipeline_config = build_pipeline_config(
        temporal_config=temporal_config,
        prompts=args.prompts,
        mask_min_area=args.mask_min_area,
        mask_dilation=args.mask_dilation,
        motion_threshold=args.motion_threshold,
        similarity_threshold=args.similarity_threshold,
        consecutive_frames=args.consecutive_frames,
        baseline_window=args.baseline_window,
        disable_flow=args.disable_flow,
        disable_color=args.disable_color,
    )

    pipeline = TongueAnalysisPipeline(pipeline_config)

    video_info = get_video_info(str(video_path))
    fps = float(video_info.get("fps") or 30.0)
    frames = load_video_frames(str(video_path))

    analysis_result = pipeline.analyze(frames, fps=fps)

    report: Dict[str, Any] = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": len(frames),
        "anomaly_count": len(analysis_result.get("anomalies", [])),
        "analysis": analysis_result,
        "video_info": video_info,
    }
    return report


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def main():
    args = parse_args()
    report = run_analysis(args)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, ensure_ascii=False, indent=2)

    print(f"[TongueAnalysis] 分析完成，报告已保存到: {output_path}")


if __name__ == "__main__":
    main()

