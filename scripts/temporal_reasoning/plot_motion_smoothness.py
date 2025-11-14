#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化基于光流的运动平滑度
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径（虽然这个脚本不直接导入src模块，但保持一致性）
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_smoothness_data(result_path: Path) -> Dict:
    """
    从分析结果中加载平滑度数据
    
    支持两种格式：
    1. run_analysis.py 的输出：从motion_metrics或metadata中提取
    2. 直接包含smoothness_scores的JSON文件
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))
    
    # 尝试从不同位置提取平滑度数据
    smoothness_scores = None
    timestamps = None
    fps = None
    
    # 方式1: 从motion_metrics中提取（run_analysis.py的输出格式）
    if "motion_metrics" in data:
        motion_metrics = data["motion_metrics"]
        if "smoothness_scores" in motion_metrics:
            smoothness_scores = motion_metrics["smoothness_scores"]
            if "smoothness_timestamps" in motion_metrics:
                timestamps = motion_metrics["smoothness_timestamps"]
            fps = data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
    
    # 方式2: 从顶层metadata中提取
    if smoothness_scores is None and "metadata" in data:
        metadata = data["metadata"]
        if "smoothness_scores" in metadata:
            smoothness_scores = metadata["smoothness_scores"]
            if "smoothness_timestamps" in metadata:
                timestamps = metadata["smoothness_timestamps"]
            fps = metadata.get("fps", 30.0)
    
    # 方式3: 直接从顶层提取
    if smoothness_scores is None and "smoothness_scores" in data:
        smoothness_scores = data["smoothness_scores"]
        if "smoothness_timestamps" in data:
            timestamps = data["smoothness_timestamps"]
        elif "timestamps" in data:
            timestamps = data["timestamps"]
        fps = data.get("fps", 30.0)
    
    # 如果没有找到平滑度数据，尝试从sub_scores中提取
    if smoothness_scores is None and "sub_scores" in data:
        sub_scores = data["sub_scores"]
        if "motion_smoothness" in sub_scores:
            smoothness_scores = sub_scores["motion_smoothness"]
            # 需要根据视频信息生成时间戳
            fps = data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
            if timestamps is None and fps > 0:
                timestamps = [i / fps for i in range(len(smoothness_scores))]
    
    if smoothness_scores is None:
        raise ValueError(
            f"{result_path} 中没有找到平滑度数据。\n"
            "请确认分析结果中包含了 motion_metrics.smoothness_scores。\n"
            "提示：运行 run_analysis.py 时会自动计算并保存平滑度数据。"
        )
    
    # 如果没有时间戳，根据fps生成
    if timestamps is None:
        fps = fps or data.get("fps") or data.get("video_info", {}).get("fps", 30.0)
        timestamps = [i / fps for i in range(len(smoothness_scores))]
    
    return {
        "smoothness_scores": smoothness_scores,
        "timestamps": timestamps,
        "fps": fps,
        "video_path": data.get("video_path") or data.get("video_info", {}).get("path", ""),
    }


def plot_smoothness(
    smoothness_scores: List[float],
    timestamps: List[float],
    save_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
) -> None:
    """
    绘制运动平滑度曲线
    
    Args:
        smoothness_scores: 平滑度分数列表
        timestamps: 时间戳列表
        save_path: 保存路径
        show: 是否显示
        title: 图表标题
        threshold: 可选的阈值线
    """
    if len(smoothness_scores) != len(timestamps):
        raise ValueError("smoothness_scores 和 timestamps 长度不匹配")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. 平滑度曲线
    axes[0].plot(timestamps, smoothness_scores, label="Motion Smoothness", color="tab:blue", linewidth=2)
    axes[0].axhline(y=np.mean(smoothness_scores), color="green", linestyle="--", 
                    label=f"Mean: {np.mean(smoothness_scores):.3f}", alpha=0.7)
    if threshold is not None:
        axes[0].axhline(y=threshold, color="red", linestyle="--", 
                       label=f"Threshold: {threshold:.3f}", alpha=0.7)
    axes[0].set_ylabel("Smoothness Score", fontsize=12)
    axes[0].set_title("Motion Smoothness Over Time", fontsize=14, fontweight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # 2. 平滑度分布直方图
    axes[1].hist(smoothness_scores, bins=30, alpha=0.7, color="tab:blue", edgecolor="black")
    axes[1].axvline(x=np.mean(smoothness_scores), color="green", linestyle="--", 
                   label=f"Mean: {np.mean(smoothness_scores):.3f}", linewidth=2)
    axes[1].axvline(x=np.median(smoothness_scores), color="orange", linestyle="--", 
                   label=f"Median: {np.median(smoothness_scores):.3f}", linewidth=2)
    if threshold is not None:
        axes[1].axvline(x=threshold, color="red", linestyle="--", 
                       label=f"Threshold: {threshold:.3f}", linewidth=2)
    axes[1].set_xlabel("Smoothness Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Smoothness Score Distribution", fontsize=14, fontweight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # 添加统计信息文本
    stats_text = (
        f"Mean: {np.mean(smoothness_scores):.3f}\n"
        f"Std: {np.std(smoothness_scores):.3f}\n"
        f"Min: {np.min(smoothness_scores):.3f}\n"
        f"Max: {np.max(smoothness_scores):.3f}\n"
        f"Median: {np.median(smoothness_scores):.3f}"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] 平滑度可视化已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化基于光流的运动平滑度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用（显示图表）
  python plot_motion_smoothness.py --result outputs/temporal_reasoning/video_result.json
  
  # 保存图像到文件
  python plot_motion_smoothness.py --result outputs/temporal_reasoning/video_result.json \\
      --output outputs/smoothness_plot.png --no-show
  
  # 添加阈值线
  python plot_motion_smoothness.py --result outputs/temporal_reasoning/video_result.json \\
      --threshold 0.5 --title "Video Motion Smoothness"
        """
    )
    parser.add_argument("--result", required=True, help="分析结果JSON文件路径（run_analysis.py的输出）")
    parser.add_argument("--output", help="可选：保存图像的路径（png/jpg等）")
    parser.add_argument("--no-show", action="store_true", help="仅保存图像，不在窗口展示")
    parser.add_argument("--title", help="图像标题，例如视频名")
    parser.add_argument("--threshold", type=float, help="可选的阈值线（用于标记低平滑度区域）")
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result).expanduser().resolve()
    
    if not result_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    
    payload = load_smoothness_data(result_path)
    
    default_title = (
        f"{Path(payload['video_path']).name} - Motion Smoothness"
        if payload["video_path"]
        else "Motion Smoothness Analysis"
    )
    title = args.title or default_title
    
    plot_smoothness(
        smoothness_scores=payload["smoothness_scores"],
        timestamps=payload["timestamps"],
        save_path=Path(args.output).expanduser().resolve() if args.output else None,
        show=not args.no_show,
        title=title,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

