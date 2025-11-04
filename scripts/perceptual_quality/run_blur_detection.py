# -*- coding: utf-8 -*-
"""Script entry for single-video blur detection."""

import os
import sys
import argparse
import json
import time
from pathlib import Path


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    # Ensure project root on path
    sys.path.insert(0, _project_root())

    from src.perceptual_quality.blur import BlurDetector, BlurDetectionConfig
    from src.perceptual_quality.blur.blur_visualization import BlurVisualization

    parser = argparse.ArgumentParser(description="视频模糊检测运行脚本")
    parser.add_argument("--video_path", type=str, required=True, help="单个视频文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--subject_noun", type=str, default="person", help="主体对象名称")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--no_visualization", action='store_true', help="不生成可视化结果")
    args = parser.parse_args()

    # 创建配置
    config = BlurDetectionConfig()
    config.update_device_config("device", args.device)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 启用JSON结果保存
    config.update_output_param("save_json_results", True)

    print("=== 单视频模糊检测 ===")
    print("正在初始化检测器...")
    
    detector = BlurDetector(config)
    
    print("检测器初始化完成！")
    print(f"开始检测视频: {args.video_path}")
    
    start_time = time.time()
    result = detector.detect(args.video_path, subject_noun=args.subject_noun)
    detection_time = time.time() - start_time
    
    print(f"检测完成，耗时: {detection_time:.2f}秒")
    
    # 获取检测结果信息
    result_data = result.get('result', {})
    blur_severity = result_data.get('blur_severity_cn') or result_data.get('blur_severity', '未知')
    confidence = result_data.get('confidence', 0.0)
    
    print(f"检测结果: {blur_severity} (置信度: {confidence:.3f})")
    
    # 生成可视化结果
    if not args.no_visualization:
        try:
            print("生成可视化结果...")
            viz_dir = os.path.join(str(config.output_dir), "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            visualizer = BlurVisualization(output_dir=viz_dir)
            
            # 从已检测结果获取原始数据（避免重复检测）
            raw_result = result.get('_raw_result', {})
            
            if 'quality_scores' in raw_result and 'blur_frames' in raw_result:
                # 生成质量分数可视化
                quality_viz_path = visualizer.visualize_quality_scores(
                    args.video_path,
                    raw_result['quality_scores'],
                    raw_result['blur_frames'],
                    raw_result.get('threshold', 0.025)
                )
                print(f"质量分数可视化已保存到: {quality_viz_path}")
            
            # 生成检测报告
            report_path = visualizer.create_detection_report(raw_result)
            print(f"检测报告已保存到: {report_path}")
            
        except Exception as e:
            print(f"可视化生成失败: {e}")
    
    # 保存结果
    output_path = os.path.join(str(config.output_dir), f"blur_detection_{os.path.basename(args.video_path)}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"检测结果已保存到: {output_path}")
    print("检测完成！")


if __name__ == "__main__":
    main()


