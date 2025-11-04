# -*- coding: utf-8 -*-
"""Script entry for batch blur detection over a directory, with configurable visualization and params."""

import os
import sys
from pathlib import Path
import argparse
import json


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    # Ensure project root import
    sys.path.insert(0, _project_root())

    from src.perceptual_quality.blur import BlurDetector, BlurDetectionConfig
    from src.perceptual_quality.blur.blur_visualization import BlurVisualization  # type: ignore

    parser = argparse.ArgumentParser(description="Run blur detection for a directory of videos")

    # Inputs / outputs
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos to analyze")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to config.output_dir)")

    # Device config
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")

    # Detection params
    parser.add_argument("--window_size", type=int, default=None, help="Sliding window size for MSS scoring")
    parser.add_argument("--confidence_threshold", type=float, default=None, help="Combined confidence threshold")
    parser.add_argument("--min_frames", type=int, default=None, help="Minimum frames required")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames allowed")
    parser.add_argument("--mild_blur", type=float, default=None, help="Mild blur threshold")
    parser.add_argument("--moderate_blur", type=float, default=None, help="Moderate blur threshold")
    parser.add_argument("--severe_blur", type=float, default=None, help="Severe blur threshold")

    # Model paths (optional overrides)
    parser.add_argument("--q_align_model", type=str, default=None)
    parser.add_argument("--grounding_dino_config", type=str, default=None)
    parser.add_argument("--grounding_dino_checkpoint", type=str, default=None)
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--cotracker_checkpoint", type=str, default=None)

    # Output behaviors
    parser.add_argument("--save_json_results", action="store_true", help="Save JSON results to output_dir")
    parser.add_argument("--save_csv_summary", action="store_true", help="Save CSV summary to output_dir")
    parser.add_argument("--save_detailed_reports", action="store_true", help="Save per-video detailed reports (visual)")

    # Visualization controls
    parser.add_argument("--visualize", action="store_true", help="Generate batch visualization after detection")
    parser.add_argument("--viz_output_dir", type=str, default=None, help="Directory to save visualization outputs")

    args = parser.parse_args()

    # Build config
    config = BlurDetectionConfig()
    if args.output_dir:
        # Point config output dir to user-provided location (keep as Path)
        config.output_dir = Path(args.output_dir).resolve()
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    config.update_device_config("device", args.device)

    # Detection params
    if args.window_size is not None:
        config.update_detection_param("window_size", args.window_size)
    if args.confidence_threshold is not None:
        config.update_detection_param("confidence_threshold", args.confidence_threshold)
    if args.min_frames is not None:
        config.update_detection_param("min_frames", args.min_frames)
    if args.max_frames is not None:
        config.update_detection_param("max_frames", args.max_frames)
    # Blur thresholds (nested dict)
    if any(v is not None for v in [args.mild_blur, args.moderate_blur, args.severe_blur]):
        blur_thresholds = dict(config.get_detection_param("blur_thresholds") or {})
        if args.mild_blur is not None:
            blur_thresholds["mild_blur"] = args.mild_blur
        if args.moderate_blur is not None:
            blur_thresholds["moderate_blur"] = args.moderate_blur
        if args.severe_blur is not None:
            blur_thresholds["severe_blur"] = args.severe_blur
        config.update_detection_param("blur_thresholds", blur_thresholds)

    # Model path overrides
    for key in [
        "q_align_model",
        "grounding_dino_config",
        "grounding_dino_checkpoint",
        "bert_path",
        "sam_checkpoint",
        "cotracker_checkpoint",
    ]:
        value = getattr(args, key)
        if value:
            config.update_model_path(key, value)

    # Output toggles
    if args.save_json_results:
        config.update_output_param("save_json_results", True)
    if args.save_csv_summary:
        config.update_output_param("save_csv_summary", True)
    if args.save_detailed_reports:
        config.update_output_param("save_detailed_reports", True)

    # Validate (non-fatal warnings are printed inside)
    _ = config.validate_config()

    # Run detection
    print("=== 批量视频模糊检测 ===")
    print("正在初始化检测器...")
    
    detector = BlurDetector(config)
    
    print("检测器初始化完成！")
    print(f"开始批量检测视频目录: {args.video_dir}")
    
    import time as time_module
    start_time = time_module.time()
    summary = detector.batch_detect(args.video_dir)
    detection_time = time_module.time() - start_time

    print(f"批量检测完成，耗时: {detection_time:.2f}秒")
    
    # 输出统计信息
    result_info = summary.get("result", {})
    total_videos = result_info.get("total_videos", 0)
    processed_videos = result_info.get("processed_videos", 0)
    blur_detected_count = result_info.get("blur_detected_count", 0)
    
    print(f"总视频数: {total_videos}")
    print(f"处理成功: {processed_videos}")
    print(f"检测到模糊: {blur_detected_count}")

    # Optional visualization based on raw pipeline results
    if args.visualize:
        print("生成批量可视化结果...")
        # Raw results saved by pipeline
        raw_results_path = os.path.join(config.output_dir, "blur_detection_results.json")
        if not os.path.exists(raw_results_path):
            print(f"警告: 原始结果未找到 {raw_results_path}; 跳过可视化")
        else:
            try:
                with open(raw_results_path, "r", encoding="utf-8") as f:
                    raw_results = json.load(f)

                viz_dir = args.viz_output_dir or os.path.join(config.output_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                visualizer = BlurVisualization(output_dir=viz_dir)

                # Batch visualization
                try:
                    batch_plot = visualizer.visualize_batch_results(raw_results)
                    print(f"批量结果可视化已保存到: {batch_plot}")
                except Exception as e:
                    print(f"批量可视化生成失败: {e}")

                # Per-video detailed reports (optional)
                if args.save_detailed_reports and isinstance(raw_results, list):
                    for item in raw_results:
                        try:
                            report_path = visualizer.create_detection_report(item)
                            print(f"详细报告已保存到: {report_path}")
                        except Exception as e:
                            print(f"为 {item.get('video_path', '')} 创建报告失败: {e}")
            except Exception as e:
                print(f"读取结果文件失败: {e}")
    
    print("批量检测完成！")


if __name__ == "__main__":
    main()


