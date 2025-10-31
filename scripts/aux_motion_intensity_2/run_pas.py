"""
可感知幅度评分启动脚本
"""

import argparse
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

import json
from src.aux_motion_intensity_2 import PASAnalyzer
from src.aux_motion_intensity_2.batch import batch_analyze_videos


def main():
    parser = argparse.ArgumentParser(
        description="Perceptible Amplitude Score - Analyze video motion using Grounded-SAM and Co-Tracker"
    )
    
    # 输入输出
    parser.add_argument("--meta_info_path", type=str, required=True, 
                       help="Path to meta info JSON file")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output JSON file path (default: overwrite input)")
    
    # 模型参数
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--grid_size", type=int, default=30,
                       help="Co-Tracker grid size (default: 30)")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                       help="GroundingDINO box threshold (default: 0.3)")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                       help="GroundingDINO text threshold (default: 0.25)")
    
    # 场景分类参数
    parser.add_argument("--enable_scene_classification", action="store_true",
                       help="Enable scene classification")
    parser.add_argument("--static_threshold", type=float, default=0.1,
                       help="Threshold for static scenes (default: 0.1)")
    parser.add_argument("--low_dynamic_threshold", type=float, default=0.3,
                       help="Threshold for low dynamic scenes (default: 0.3)")
    parser.add_argument("--medium_dynamic_threshold", type=float, default=0.6,
                       help="Threshold for medium dynamic scenes (default: 0.6)")
    parser.add_argument("--high_dynamic_threshold", type=float, default=1.0,
                       help="Threshold for high dynamic scenes (default: 1.0)")
    parser.add_argument("--motion_ratio_threshold", type=float, default=1.5,
                       help="Motion ratio threshold for camera vs object motion (default: 1.5)")
    
    # 其他选项
    parser.add_argument("--no_subject_diag_norm", action="store_true",
                       help="Disable subject-diagonal normalization")
    
    args = parser.parse_args()
    
    # 场景分类器参数
    scene_classifier_params = {
        'static_threshold': args.static_threshold,
        'low_dynamic_threshold': args.low_dynamic_threshold,
        'medium_dynamic_threshold': args.medium_dynamic_threshold,
        'high_dynamic_threshold': args.high_dynamic_threshold,
        'motion_ratio_threshold': args.motion_ratio_threshold
    }
    
    # 初始化分析器
    print("Initializing PAS Analyzer...")
    analyzer = PASAnalyzer(
        device=args.device,
        grid_size=args.grid_size,
        enable_scene_classification=args.enable_scene_classification,
        scene_classifier_params=scene_classifier_params
    )
    
    # 加载元信息
    print(f"Loading meta info from {args.meta_info_path}...")
    with open(args.meta_info_path, 'r', encoding='utf-8') as f:
        meta_infos = json.load(f)
    
    print(f"Found {len(meta_infos)} videos to analyze")
    
    # 批量分析
    print("Starting batch analysis...")
    results = batch_analyze_videos(
        analyzer=analyzer,
        meta_info_list=meta_infos,
        output_path=args.output_path or args.meta_info_path
    )
    
    print(f"\nAnalysis completed! Processed {len(results)} videos")
    
    # 统计信息
    success_count = sum(1 for r in results if r['result']['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"\nSummary:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    main()

