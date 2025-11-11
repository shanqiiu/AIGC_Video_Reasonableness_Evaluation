#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从舌头检测报告中可视化光流与颜色指标。
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_report(report_path: Path) -> dict:
    """读取 JSON 报告"""
    data = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = data.get("analysis", {})
    metadata = analysis.get("metadata", {})
    frame_stats = metadata.get("frame_stats", [])
    if not frame_stats:
        raise ValueError(f"{report_path} 中没有 frame_stats 数据，请确认脚本是否开启 --debug-stats。")
    return {
        "frame_stats": frame_stats,
        "motion_threshold": metadata.get("motion_threshold", 0.0),
        "similarity_threshold": metadata.get("similarity_threshold", 0.0),
        "baseline_motion": metadata.get("baseline_motion", 0.0),
        "fps": data.get("fps", 0.0),
        "video_path": data.get("video_path", ""),
    }


def plot_metrics(
    frame_stats: list,
    motion_threshold: float,
    similarity_threshold: float,
    baseline_motion: float,
    save_path: Optional[Path] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    """绘制光流 / 颜色指标曲线"""
    df = pd.DataFrame(frame_stats)
    if "timestamp" not in df.columns:
        raise ValueError("frame_stats 缺少 timestamp 字段；请使用新版检测脚本重新生成报告。")
    df["time"] = df["timestamp"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. 光流幅值
    axes[0].plot(df["time"], df["motion_value"], label="motion_value")
    axes[0].axhline(baseline_motion, color="gray", linestyle="--", label="baseline_motion")
    axes[0].set_ylabel("Motion Value")
    axes[0].legend()
    axes[0].grid(True)

    # 2. 光流变化量
    axes[1].plot(df["time"], df["motion_change"], label="motion_change", color="tab:orange")
    axes[1].axhline(motion_threshold, color="red", linestyle="--", label="motion_threshold")
    axes[1].set_ylabel("Motion Change")
    axes[1].legend()
    axes[1].grid(True)

    # 3. 颜色相似度
    axes[2].plot(df["time"], df["hist_similarity"], label="hist_similarity", color="tab:green")
    axes[2].axhline(1 - similarity_threshold, color="purple", linestyle="--", label="1 - similarity_threshold")
    axes[2].set_ylabel("Hist Similarity")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        save_path = save_path.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[plot] 曲线已保存到: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化舌头分析报告中的光流 & 颜色指标")
    parser.add_argument("--report", required=True, help="tongue_analysis_report.json 路径")
    parser.add_argument("--output", help="可选：保存图像的路径（png/jpg 等）")
    parser.add_argument("--no-show", action="store_true", help="仅保存图像，不在窗口展示")
    parser.add_argument("--title", help="图像标题，例如视频名")
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report).expanduser().resolve()
    payload = load_report(report_path)

    title = args.title or Path(payload["video_path"]).name
    plot_metrics(
        frame_stats=payload["frame_stats"],
        motion_threshold=payload["motion_threshold"],
        similarity_threshold=payload["similarity_threshold"],
        baseline_motion=payload["baseline_motion"],
        save_path=Path(args.output).expanduser().resolve() if args.output else None,
        show=not args.no_show,
        title=title,
    )


if __name__ == "__main__":
    main()