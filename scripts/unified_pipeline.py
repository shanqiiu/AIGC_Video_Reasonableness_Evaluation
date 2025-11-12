# -*- coding: utf-8 -*-
"""
统一评测流水线
整合动态度检测、模糊程度检测、时序性评估、人体时序性四个模块
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import numpy as np

# 添加项目根目录到路径
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入各模块
import copy
from src.aux_motion_intensity.analyzer import MotionIntensityAnalyzer
from src.temporal_reasoning.core.temporal_analyzer import TemporalReasoningAnalyzer
from src.temporal_reasoning.core.config import TemporalReasoningConfig, get_default_config
from src.temporal_reasoning.region_analysis.pipeline import (
    RegionAnalysisPipeline, 
    RegionAnalysisPipelineConfig
)
from src.temporal_reasoning.utils.video_utils import load_video_frames, get_video_info

# 设置blur_new模块路径
blur_new_path = os.path.join(project_root, 'src', 'perceptual_quality', 'blur_new')
if blur_new_path not in sys.path:
    sys.path.insert(0, blur_new_path)
from simple_blur_detector import BlurDetector


class UnifiedEvaluationPipeline:
    """统一评测流水线"""
    
    def __init__(
        self,
        device: str = "cuda:0",
        raft_model_path: Optional[str] = None,
        q_align_model_path: str = ".cache/q-future/one-align",
        batch_size: int = 32,
        camera_fov: float = 60.0,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
        temporal_config: Optional[TemporalReasoningConfig] = None,
        region_regions: Optional[List[str]] = None,
    ):
        """
        初始化统一评测流水线
        
        Args:
            device: 计算设备
            raft_model_path: RAFT模型路径
            q_align_model_path: Q-Align模型路径
            batch_size: 批处理大小
            camera_fov: 相机视场角
            max_frames: 最大处理帧数
            frame_skip: 帧采样间隔
            temporal_config: 时序性评估配置
            region_regions: 区域分析的目标区域列表
        """
        self.device = device
        self.camera_fov = camera_fov
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        
        # 初始化各模块
        print("=" * 60)
        print("正在初始化统一评测流水线...")
        print("=" * 60)
        
        # 1. 动态度检测模块
        print("\n[1/4] 初始化动态度检测模块...")
        self.motion_analyzer = MotionIntensityAnalyzer(
            raft_model_path=raft_model_path,
            device=device,
            method='raft',
            enable_camera_compensation=True,
            use_normalized_flow=False,
        )
        print("✓ 动态度检测模块初始化完成")
        
        # 2. 模糊检测模块
        print("\n[2/4] 初始化模糊检测模块...")
        self.blur_detector = BlurDetector(
            device=device,
            model_path=q_align_model_path,
            batch_size=batch_size
        )
        print("✓ 模糊检测模块初始化完成")
        
        # 3. 时序性评估模块
        print("\n[3/4] 初始化时序性评估模块...")
        self.temporal_config = temporal_config or get_default_config()
        if device:
            self.temporal_config.device = device
        self.temporal_analyzer = TemporalReasoningAnalyzer(self.temporal_config)
        self.temporal_analyzer.initialize()
        print("✓ 时序性评估模块初始化完成")
        
        # 4. 人体时序性（区域分析）模块
        print("\n[4/4] 初始化人体时序性（区域分析）模块...")
        region_config = RegionAnalysisPipelineConfig(
            raft=self.temporal_config.raft,
            keypoint=self.temporal_config.keypoint,
            enable_visualization=False,
            visualization_output_dir=None,
            visualization_max_frames=0,
            per_region_visualization=False,
        )
        # 如果指定了区域，则只分析指定区域
        if region_regions:
            available_regions = {region.name: copy.deepcopy(region) for region in region_config.regions}
            selected_definitions = []
            for name in region_regions:
                if name not in available_regions:
                    print(f"警告: 未知区域 '{name}'，将跳过。可用区域: {', '.join(available_regions.keys())}")
                    continue
                selected_definitions.append(available_regions[name])
            if selected_definitions:
                region_config.regions = selected_definitions
            else:
                print("警告: 没有有效的区域，将使用默认区域")
        self.region_analyzer = RegionAnalysisPipeline(region_config)
        print("✓ 人体时序性（区域分析）模块初始化完成")
        
        print("\n" + "=" * 60)
        print("所有模块初始化完成！")
        print("=" * 60)
    
    def evaluate_single_video(
        self,
        video_path: str,
        text_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        评测单个视频
        
        Args:
            video_path: 视频文件路径
            text_prompts: 文本提示（用于时序性评估）
            
        Returns:
            评测结果字典
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print("\n" + "=" * 60)
        print(f"开始评测视频: {video_name}")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            'video_path': video_path,
            'video_name': video_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        try:
            # 一次性加载视频帧和视频信息
            print("\n>>> 步骤0: 加载视频帧（一次性完成）")
            video_info = get_video_info(video_path)
            print(f"视频信息: {video_info['width']}x{video_info['height']}, "
                  f"{video_info['frame_count']}帧, {video_info['fps']:.2f}fps")
            
            video_frames = load_video_frames(
                video_path,
                max_frames=self.max_frames,
                target_size=None  # 保持原尺寸
            )
            print(f"已加载 {len(video_frames)} 帧")
            results['video_info'] = video_info
            
            if len(video_frames) < 2:
                raise ValueError("视频帧数不足，至少需要2帧")
            
            # 1. 动态度检测
            print("\n>>> 步骤1: 动态度检测 (aux_motion_intensity)")
            try:
                camera_matrix = self.motion_analyzer.estimate_camera_matrix(
                    video_frames[0].shape, 
                    self.camera_fov
                )
                motion_result = self.motion_analyzer.analyze_frames(
                    video_frames, 
                    camera_matrix
                )
                results['motion_intensity'] = {
                    'motion_intensity': motion_result['motion_intensity'],
                    'scene_type': motion_result['scene_type'],
                    'temporal_stats': self._make_serializable(motion_result.get('temporal_stats', {})),
                    'component_scores': self._make_serializable(motion_result.get('component_scores', {})),
                }
                print(f"✓ 动态度: {motion_result['motion_intensity']:.4f}, 场景类型: {motion_result['scene_type']}")
            except Exception as e:
                print(f"✗ 动态度检测失败: {e}")
                results['motion_intensity'] = {'error': str(e)}
            
            # 2. 模糊程度检测
            print("\n>>> 步骤2: 模糊程度检测 (blur_new)")
            try:
                blur_result = self.blur_detector.detect_blur(video_path)
                results['blur_detection'] = {
                    'blur_detected': blur_result.get('blur_detected', False),
                    'blur_severity': blur_result.get('blur_severity', 'unknown'),
                    'confidence': blur_result.get('confidence', 0.0),
                    'blur_ratio': blur_result.get('blur_ratio', 0.0),
                    'mss_score': blur_result.get('mss_score', 0.0),
                }
                print(f"✓ 模糊检测: {blur_result.get('blur_severity', 'unknown')}, "
                      f"置信度: {blur_result.get('confidence', 0.0):.4f}")
            except Exception as e:
                print(f"✗ 模糊检测失败: {e}")
                results['blur_detection'] = {'error': str(e)}
            
            # 3. 时序性评估
            print("\n>>> 步骤3: 时序性评估 (temporal_reasoning)")
            try:
                temporal_result = self.temporal_analyzer.analyze(
                    video_frames=video_frames,
                    text_prompts=text_prompts,
                    fps=video_info['fps'],
                    video_path=video_path
                )
                results['temporal_reasoning'] = {
                    'motion_reasonableness_score': temporal_result.get('motion_reasonableness_score', 0.0),
                    'structure_stability_score': temporal_result.get('structure_stability_score', 0.0),
                    'anomaly_count': len(temporal_result.get('anomalies', [])),
                    'anomalies': self._make_serializable(temporal_result.get('anomalies', [])),
                }
                print(f"✓ 运动合理性: {temporal_result.get('motion_reasonableness_score', 0.0):.4f}, "
                      f"结构稳定性: {temporal_result.get('structure_stability_score', 0.0):.4f}, "
                      f"异常数: {len(temporal_result.get('anomalies', []))}")
            except Exception as e:
                print(f"✗ 时序性评估失败: {e}")
                import traceback
                traceback.print_exc()
                results['temporal_reasoning'] = {'error': str(e)}
            
            # 4. 人体时序性（区域分析）
            print("\n>>> 步骤4: 人体时序性（区域分析） (region_analysis)")
            try:
                region_result = self.region_analyzer.analyze(
                    video_frames=video_frames,
                    fps=video_info['fps'],
                    video_path=video_path
                )
                results['region_analysis'] = {
                    'score': region_result.get('score', 0.0),
                    'anomaly_count': len(region_result.get('anomalies', [])),
                    'anomalies': self._make_serializable(region_result.get('anomalies', [])),
                    'regions': self._make_serializable(region_result.get('regions', {})),
                }
                print(f"✓ 区域分析得分: {region_result.get('score', 0.0):.4f}, "
                      f"异常数: {len(region_result.get('anomalies', []))}")
            except Exception as e:
                print(f"✗ 区域分析失败: {e}")
                import traceback
                traceback.print_exc()
                results['region_analysis'] = {'error': str(e)}
            
            # 计算总耗时
            total_time = time.time() - start_time
            results['processing_time'] = total_time
            results['status'] = 'success'
            
            print("\n" + "=" * 60)
            print(f"评测完成！总耗时: {total_time:.2f}秒")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ 评测过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = 'failed'
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
        
        return results
    
    def evaluate_batch_videos(
        self,
        video_paths: List[str],
        text_prompts: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        批量评测视频
        
        Args:
            video_paths: 视频文件路径列表
            text_prompts: 文本提示（用于时序性评估）
            output_dir: 输出目录
            
        Returns:
            批量评测结果字典
        """
        print("\n" + "=" * 60)
        print(f"开始批量评测，共 {len(video_paths)} 个视频")
        print("=" * 60)
        
        all_results = []
        successful = 0
        failed = 0
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] 处理视频: {os.path.basename(video_path)}")
            result = self.evaluate_single_video(video_path, text_prompts)
            all_results.append(result)
            
            if result.get('status') == 'success':
                successful += 1
            else:
                failed += 1
        
        # 生成汇总报告
        summary = {
            'total_videos': len(video_paths),
            'successful': successful,
            'failed': failed,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': all_results
        }
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'unified_evaluation_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(summary), f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_path}")
        
        print("\n" + "=" * 60)
        print(f"批量评测完成！成功: {successful}, 失败: {failed}")
        print("=" * 60)
        
        return summary
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, 'item'):  # PyTorch tensor
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


def collect_video_paths(inputs: List[str]) -> List[str]:
    """收集视频文件路径"""
    video_paths = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    for input_path in inputs:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            print(f"警告: 路径不存在: {input_path}")
            continue
        
        if path.is_file():
            if path.suffix.lower() in video_extensions:
                video_paths.append(str(path))
            else:
                print(f"警告: 不是视频文件: {input_path}")
        elif path.is_dir():
            for ext in video_extensions:
                video_paths.extend([str(p) for p in path.glob(f'*{ext}')])
                video_paths.extend([str(p) for p in path.glob(f'*{ext.upper()}')])
    
    # 去重并排序
    video_paths = sorted(list(set(video_paths)))
    return video_paths


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='统一评测流水线 - 整合动态度检测、模糊程度检测、时序性评估、人体时序性四个模块',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评测单个视频
  python unified_pipeline.py --video path/to/video.mp4 --output results.json
  
  # 批量评测视频目录
  python unified_pipeline.py --video_dir path/to/videos --output_dir results/
  
  # 指定设备和其他参数
  python unified_pipeline.py --video video.mp4 --device cuda:0 --max_frames 100
        """
    )
    
    # 输入参数
    parser.add_argument(
        '--video',
        type=str,
        nargs='+',
        help='视频文件路径（可指定多个）'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        help='视频目录路径'
    )
    
    # 输出参数
    parser.add_argument(
        '--output',
        type=str,
        help='输出JSON文件路径（单视频模式）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/unified_evaluation',
        help='输出目录（批量模式，默认: outputs/unified_evaluation）'
    )
    
    # 设备参数
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备（默认: cuda:0）'
    )
    
    # 模型路径参数
    parser.add_argument(
        '--raft_model',
        type=str,
        default=None,
        help='RAFT模型路径'
    )
    parser.add_argument(
        '--q_align_model',
        type=str,
        default='.cache/q-future/one-align',
        help='Q-Align模型路径（默认: .cache/q-future/one-align）'
    )
    
    # 处理参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批处理大小（默认: 32）'
    )
    parser.add_argument(
        '--camera_fov',
        type=float,
        default=60.0,
        help='相机视场角（默认: 60.0）'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='最大处理帧数（默认: 全部）'
    )
    parser.add_argument(
        '--frame_skip',
        type=int,
        default=1,
        help='帧采样间隔（默认: 1）'
    )
    
    # 时序性评估参数
    parser.add_argument(
        '--temporal_config',
        type=str,
        default=None,
        help='时序性评估配置文件路径（YAML格式）'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        default=None,
        help='文本提示列表（用于时序性评估）'
    )
    
    # 区域分析参数
    parser.add_argument(
        '--regions',
        type=str,
        nargs='+',
        default=None,
        help='区域分析的目标区域（默认: 全部）'
    )
    
    args = parser.parse_args()
    
    # 收集视频路径
    video_paths = []
    if args.video:
        video_paths.extend(args.video)
    if args.video_dir:
        video_paths.extend(collect_video_paths([args.video_dir]))
    
    if not video_paths:
        print("错误: 未指定视频文件或目录")
        parser.print_help()
        return
    
    # 加载时序性评估配置
    temporal_config = None
    if args.temporal_config:
        from src.temporal_reasoning.core.config import load_config_from_yaml
        temporal_config = load_config_from_yaml(args.temporal_config)
    
    # 创建流水线
    pipeline = UnifiedEvaluationPipeline(
        device=args.device,
        raft_model_path=args.raft_model,
        q_align_model_path=args.q_align_model,
        batch_size=args.batch_size,
        camera_fov=args.camera_fov,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        temporal_config=temporal_config,
        region_regions=args.regions,
    )
    
    # 执行评测
    if len(video_paths) == 1:
        # 单视频模式
        result = pipeline.evaluate_single_video(video_paths[0], args.prompts)
        
        # 保存结果
        if args.output:
            output_path = args.output
        else:
            video_name = os.path.splitext(os.path.basename(video_paths[0]))[0]
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{video_name}_evaluation.json')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline._make_serializable(result), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
    else:
        # 批量模式
        summary = pipeline.evaluate_batch_videos(
            video_paths,
            text_prompts=args.prompts,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()

