# -*- coding: utf-8 -*-
"""
视频模糊检测运行脚本
提供完整的模糊检测流程
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict
import time
import numpy as np


def _get_project_root() -> str:
    """获取项目根目录路径"""
    current_file = os.path.abspath(__file__)
    # scripts/perceptual_quality/run_blur_detection.py -> 项目根目录
    return os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))


def _setup_module_paths():
    """设置模块导入路径"""
    project_root = _get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 添加 blur_new 模块路径
    blur_new_path = os.path.join(project_root, 'src', 'perceptual_quality', 'blur_new')
    if blur_new_path not in sys.path:
        sys.path.insert(0, blur_new_path)


# 设置模块路径
_setup_module_paths()

# 导入 src 模块
from simple_blur_detector import BlurDetector
from blur_visualization import BlurVisualization
from config import BlurDetectionConfig, get_preset_config


class BlurDetectionRunner:
    """模糊检测运行器"""
    
    def __init__(self, config: BlurDetectionConfig = None):
        self.config = config or BlurDetectionConfig()
        self.detector = None
        self.visualizer = None
        
    def initialize_detector(self, use_simple: bool = True):
        """初始化检测器"""
        print("正在初始化检测器...")
        
        if use_simple:
            # 使用模糊检测器
            self.detector = BlurDetector(
                device=self.config.get_device_config('device'),
                model_path=self.config.get_model_path('q_align_model')
            )
        
        # 初始化可视化工具
        self.visualizer = BlurVisualization(str(self.config.output_dir / "visualizations"))
        
        print("检测器初始化完成！")
    
    def _make_json_serializable(self, obj):
        """将NumPy/PyTorch类型转换为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, 'item'):  # PyTorch tensor
            return obj.item()
        else:
            return obj
    
    def detect_single_video(self, video_path: str, generate_visualization: bool = True) -> Dict:
        """
        检测单个视频
        
        Args:
            video_path: 视频路径
            generate_visualization: 是否生成可视化
            
        Returns:
            检测结果
        """
        if not self.detector:
            self.initialize_detector()
        
        print(f"开始检测视频: {video_path}")
        start_time = time.time()
        
        # 执行检测
        result = self.detector.detect_blur(video_path)
        
        detection_time = time.time() - start_time
        result['detection_time'] = detection_time
        
        print(f"检测完成，耗时: {detection_time:.2f}秒")
        print(f"检测结果: {result.get('blur_severity', '未知')} (置信度: {result.get('confidence', 0.0):.3f})")
        
        # 生成可视化
        if generate_visualization and self.visualizer:
            try:
                print("生成可视化结果...")
                if 'quality_scores' in result and 'blur_frames' in result:
                    # 生成质量分数可视化
                    quality_viz_path = self.visualizer.visualize_quality_scores(
                        video_path, 
                        result['quality_scores'], 
                        result['blur_frames'], 
                        result.get('threshold', 0.025)
                    )
                    print(f"质量分数可视化已保存到: {quality_viz_path}")
                
                # 生成检测报告
                report_path = self.visualizer.create_detection_report(result)
                print(f"检测报告已保存到: {report_path}")
                
            except Exception as e:
                print(f"可视化生成失败: {e}")
        
        return result
    
    def detect_batch_videos(self, video_dir: str, generate_visualization: bool = True) -> Dict:
        """
        批量检测视频
        
        Args:
            video_dir: 视频目录
            generate_visualization: 是否生成可视化
            
        Returns:
            批量检测结果
        """
        if not self.detector:
            self.initialize_detector()
        
        print(f"开始批量检测视频目录: {video_dir}")
        start_time = time.time()
        
        # 执行批量检测
        results = self.detector.batch_detect(video_dir, str(self.config.output_dir))
        
        detection_time = time.time() - start_time
        
        print(f"批量检测完成，耗时: {detection_time:.2f}秒")
        print(f"总视频数: {results.get('total_videos', 0)}")
        print(f"检测到模糊: {results.get('blur_detected_count', 0)}")
        
        # 生成批量可视化
        if generate_visualization and self.visualizer and 'results' in results:
            try:
                print("生成批量可视化结果...")
                batch_viz_path = self.visualizer.visualize_batch_results(results['results'])
                print(f"批量结果可视化已保存到: {batch_viz_path}")
                
            except Exception as e:
                print(f"批量可视化生成失败: {e}")
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """保存检测结果"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"blur_detection_results_{timestamp}.json"
        
        output_path = self.config.output_dir / filename
        
        # 转换数据为JSON可序列化格式
        serializable_results = self._make_json_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已保存到: {output_path}")
        return str(output_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="视频模糊检测运行脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="单个视频文件路径"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        help="视频目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./blur_detection_results",
        help="输出目录"
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        choices=['fast', 'accurate', 'balanced'],
        default='balanced',
        help="配置预设"
    )
    parser.add_argument(
        "--use_simple",
        action='store_true',
        default=True,
        help="使用简化版检测器"
    )
    parser.add_argument(
        "--no_visualization",
        action='store_true',
        help="不生成可视化结果"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = get_preset_config(args.config_preset)
    config.output_dir = Path(args.output_dir)
    config.update_device_config('device', args.device)
    
    # 验证配置
    if not config.validate_config():
        print("配置验证失败，请检查模型文件和路径设置")
        return
    
    # 创建运行器
    runner = BlurDetectionRunner(config)
    
    try:
        if args.video_path:
            # 单视频检测
            print("=== 单视频模糊检测 ===")
            result = runner.detect_single_video(
                args.video_path, 
                generate_visualization=not args.no_visualization
            )
            
            # 保存结果
            runner.save_results(result)
            
        elif args.video_dir:
            # 批量检测
            print("=== 批量视频模糊检测 ===")
            results = runner.detect_batch_videos(
                args.video_dir, 
                generate_visualization=not args.no_visualization
            )
            
            # 保存结果
            runner.save_results(results)
            
        else:
            print("请指定 --video_path 或 --video_dir 参数")
            parser.print_help()
            return
        
        print("检测完成！")
        
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
