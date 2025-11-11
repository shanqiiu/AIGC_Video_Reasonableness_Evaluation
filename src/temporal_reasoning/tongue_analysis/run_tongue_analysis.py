from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List

import numpy as np
import copy

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.temporal_reasoning.core.config import TemporalReasoningConfig, load_config_from_yaml
from src.temporal_reasoning.utils.video_utils import get_video_info, load_video_frames
from src.temporal_reasoning.tongue_analysis.pipeline import (
    TongueAnalysisPipeline,
    TongueAnalysisPipelineConfig,
    RegionDefinition,
)
from src.temporal_reasoning.tongue_analysis.tongue_flow_change_detector import TongueFlowChangeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tongue flow/appearance change analysis on a video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["mouth"],
        help="Regions to analyze (default: mouth). Available: mouth, left_eye, right_eye, left_hand, right_hand",
    )
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
        "--min_area",
        type=int,
        help="Override minimum area (pixels) required for region masks.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        help="Override additional margin ratio around region polygons.",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        help="Override motion change threshold for regions.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        help="Override color histogram similarity drop threshold.",
    )
    parser.add_argument(
        "--hist_diff_threshold",
        type=float,
        help="Override histogram similarity frame-to-frame difference threshold.",
    )
    parser.add_argument(
        "--consecutive_frames",
        type=int,
        help="Override required consecutive frames for anomaly.",
    )
    parser.add_argument(
        "--baseline_window",
        type=int,
        help="Override baseline window used to compute reference motion.",
    )
    parser.add_argument(
        "--disable_flow",
        action="store_true",
        help="Disable optical-flow change detection (only use appearance similarity).",
    )
    parser.add_argument(
        "--disable_color",
        action="store_true",
        help="Disable color histogram similarity check (only use motion).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save mouth mask visualizations for each frame.",
    )
    parser.add_argument(
        "--vis_dir",
        help="Directory to store visualization frames.",
    )
    parser.add_argument(
        "--vis_max_frames",
        type=int,
        default=150,
        help="Maximum number of visualization frames to save (0 for unlimited).",
    )
    parser.add_argument(
        "--debug_stats",
        help="Optional path to save per-frame statistics (JSON).",
    )
    return parser.parse_args()


def build_pipeline_config(
    temporal_config: TemporalReasoningConfig,
    regions: Sequence[str],
    min_area: Optional[int],
    margin: Optional[float],
    motion_threshold: Optional[float],
    similarity_threshold: Optional[float],
    hist_diff_threshold: Optional[float],
    consecutive_frames: Optional[int],
    baseline_window: Optional[int],
    disable_flow: bool,
    disable_color: bool,
    enable_visualization: bool,
    visualization_output_dir: Optional[str],
    visualization_max_frames: int,
) -> TongueAnalysisPipelineConfig:
    pipeline_config = TongueAnalysisPipelineConfig(
        raft=temporal_config.raft,
        keypoint=temporal_config.keypoint,
        enable_visualization=enable_visualization,
        visualization_output_dir=visualization_output_dir,
        visualization_max_frames=visualization_max_frames,
    )

    available_regions = {region.name: copy.deepcopy(region) for region in pipeline_config.regions}
    if not regions:
        selected_names = list(available_regions.keys())
    else:
        selected_names = list(dict.fromkeys(regions))  # preserve order, remove duplicates

    selected_definitions: List[RegionDefinition] = []
    for name in selected_names:
        if name not in available_regions:
            raise ValueError(f"Unknown region '{name}'. Available options: {', '.join(available_regions.keys())}")

        region_def = available_regions[name]
        config = copy.deepcopy(region_def.temporal_config)

        if motion_threshold is not None:
            config.motion_threshold = motion_threshold
        if similarity_threshold is not None:
            config.similarity_threshold = similarity_threshold
        if hist_diff_threshold is not None:
            config.hist_diff_threshold = hist_diff_threshold
        if consecutive_frames is not None:
            config.consecutive_frames = consecutive_frames
        if baseline_window is not None:
            config.baseline_window = baseline_window
        if disable_flow:
            config.use_flow_change = False
        if disable_color:
            config.use_color_similarity = False

        region_def.temporal_config = config

        if min_area is not None:
            region_def.min_area = min_area
        if margin is not None:
            region_def.margin_ratio = margin

        selected_definitions.append(region_def)

    pipeline_config.regions = selected_definitions
    return pipeline_config


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
        regions=args.regions,
        min_area=args.min_area,
        margin=args.margin,
        motion_threshold=args.motion_threshold,
        similarity_threshold=args.similarity_threshold,
        hist_diff_threshold=args.hist_diff_threshold,
        consecutive_frames=args.consecutive_frames,
        baseline_window=args.baseline_window,
        disable_flow=args.disable_flow,
        disable_color=args.disable_color,
        enable_visualization=args.visualize,
        visualization_output_dir=args.vis_dir,
        visualization_max_frames=args.vis_max_frames,
    )

    pipeline = TongueAnalysisPipeline(pipeline_config)

    video_info = get_video_info(str(video_path))
    fps = float(video_info.get("fps") or 30.0)
    frames = load_video_frames(str(video_path))
    analysis_result = pipeline.analyze(frames, fps=fps, video_path=str(video_path))

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

    if args.debug_stats:
        debug_path = Path(args.debug_stats).expanduser().resolve()
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        region_stats = {
            name: region_data.get("metadata", {}).get("frame_stats", [])
            for name, region_data in report.get("analysis", {}).get("regions", {}).items()
        }
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(to_serializable(region_stats), f, ensure_ascii=False, indent=2)
        print(f"[TongueAnalysis] 帧级统计已保存到: {debug_path}")

    print(f"[TongueAnalysis] 分析完成，报告已保存到: {output_path}")


if __name__ == "__main__":
    main()

