# -*- coding: utf-8 -*-
"""
简化版视频模糊检测器
基于VMBench的MSS评分器，专门用于快速模糊检测
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Tuple
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 添加VMBench路径
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (blur目录的父目录)
project_root = os.path.dirname(current_dir)

# 保存原始工作目录
original_cwd = os.getcwd()

try:
    # 临时切换到项目根目录以便导入
    os.chdir(project_root)
    
    # 添加项目根目录到路径
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "Q-Align"))
    
    # 导入VMBench相关模块
    from motion_smoothness_score import QAlignVideoScorer, load_video_sliding_window, set_threshold, get_artifacts_frames
    
finally:
    # 恢复原始工作目录
    os.chdir(original_cwd)


class SimpleBlurDetector:
    """简化版模糊检测器，主要基于MSS评分"""
    
    def __init__(self, device="cuda:0", model_path=".cache/q-future/one-align"):
        """
        初始化简化版模糊检测器
        
        Args:
            device: 计算设备
            model_path: Q-Align模型路径
        """
        self.device = device
        self.model_path = model_path
        
        # 初始化Q-Align模型
        print("正在初始化Q-Align模型...")
        try:
            self.q_align_scorer = QAlignVideoScorer(
                pretrained=model_path, 
                device=device
            )
            print("Q-Align模型初始化完成！")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
        
        # 模糊检测参数
        self.blur_thresholds = {
            'mild_blur': 0.015,    # 轻微模糊阈值
            'moderate_blur': 0.025, # 中等模糊阈值
            'severe_blur': 0.04    # 严重模糊阈值
        }
    
    def detect_blur(self, video_path: str, window_size: int = 3) -> Dict:
        """
        检测视频模糊
        
        Args:
            video_path: 视频文件路径
            window_size: 滑动窗口大小
            
        Returns:
            模糊检测结果
        """
        print(f"开始检测视频模糊: {os.path.basename(video_path)}")
        
        try:
            # 1. 加载视频并计算质量分数
            video_frames = load_video_sliding_window(video_path, window_size=window_size)
            _, _, quality_scores = self.q_align_scorer(video_frames)
            quality_scores = quality_scores.tolist()
            
            # 2. 估算相机运动幅度
            camera_movement = self._estimate_camera_movement(video_path)
            
            # 3. 设置自适应阈值
            threshold = set_threshold(camera_movement)
            
            # 4. 检测模糊帧
            blur_frames = get_artifacts_frames(quality_scores, threshold)
            
            # 5. 计算模糊指标
            blur_metrics = self._calculate_blur_metrics(quality_scores, blur_frames, threshold)
            
            # 6. 生成检测结果
            result = self._generate_detection_result(video_path, blur_metrics)
            
            return result
            
        except Exception as e:
            print(f"模糊检测失败: {e}")
            return {
                'video_path': video_path,
                'blur_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'blur_severity': '检测失败'
            }
    
    def _estimate_camera_movement(self, video_path: str) -> float:
        """估算相机运动幅度"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # 读取前10帧估算运动
            frame_count = 0
            while frame_count < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray_frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) < 2:
                return 0.0
            
            # 计算帧间差异
            total_diff = 0.0
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i-1])
                total_diff += np.mean(diff)
            
            # 归一化运动幅度
            movement = total_diff / (len(frames) - 1) / 255.0
            return min(1.0, movement)
            
        except Exception as e:
            print(f"相机运动估算失败: {e}")
            return 0.0
    
    def _calculate_blur_metrics(self, quality_scores: List[float], blur_frames: List[int], threshold: float) -> Dict:
        """计算模糊指标"""
        total_frames = len(quality_scores)
        blur_frame_count = len(blur_frames)
        
        # 基础指标
        blur_ratio = blur_frame_count / total_frames if total_frames > 0 else 0
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        
        # 计算质量分数变化
        quality_diffs = np.abs(np.diff(quality_scores))
        max_quality_drop = np.max(quality_diffs) if len(quality_diffs) > 0 else 0
        
        # 计算模糊严重程度
        blur_severity = self._determine_blur_severity(blur_ratio, max_quality_drop, threshold)
        
        # 计算综合置信度
        confidence = self._calculate_confidence(blur_ratio, max_quality_drop, avg_quality)
        
        return {
            'total_frames': total_frames,
            'blur_frames': blur_frames,
            'blur_frame_count': blur_frame_count,
            'blur_ratio': blur_ratio,
            'avg_quality': avg_quality,
            'quality_std': quality_std,
            'max_quality_drop': max_quality_drop,
            'threshold': threshold,
            'blur_severity': blur_severity,
            'confidence': confidence,
            'blur_detected': blur_ratio > 0.05 or max_quality_drop > threshold
        }
    
    def _determine_blur_severity(self, blur_ratio: float, max_quality_drop: float, threshold: float) -> str:
        """确定模糊严重程度"""
        if blur_ratio > 0.3 or max_quality_drop > threshold * 2:
            return "严重模糊"
        elif blur_ratio > 0.1 or max_quality_drop > threshold * 1.5:
            return "中等模糊"
        elif blur_ratio > 0.05 or max_quality_drop > threshold:
            return "轻微模糊"
        else:
            return "无模糊"
    
    def _calculate_confidence(self, blur_ratio: float, max_quality_drop: float, avg_quality: float) -> float:
        """计算模糊检测置信度"""
        # 基于模糊比例和质量下降计算置信度
        blur_confidence = min(1.0, blur_ratio * 2)  # 模糊比例权重
        quality_confidence = min(1.0, max_quality_drop * 10)  # 质量下降权重
        avg_quality_confidence = max(0.0, 1.0 - avg_quality)  # 平均质量权重
        
        # 综合置信度
        confidence = (blur_confidence * 0.4 + quality_confidence * 0.4 + avg_quality_confidence * 0.2)
        return min(1.0, confidence)
    
    def _generate_detection_result(self, video_path: str, blur_metrics: Dict) -> Dict:
        """生成检测结果"""
        return {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'blur_detected': blur_metrics['blur_detected'],
            'confidence': blur_metrics['confidence'],
            'blur_severity': blur_metrics['blur_severity'],
            'blur_ratio': blur_metrics['blur_ratio'],
            'blur_frame_count': blur_metrics['blur_frame_count'],
            'total_frames': blur_metrics['total_frames'],
            'avg_quality': blur_metrics['avg_quality'],
            'max_quality_drop': blur_metrics['max_quality_drop'],
            'threshold': blur_metrics['threshold'],
            'blur_frames': blur_metrics['blur_frames'],
            'recommendations': self._generate_recommendations(blur_metrics)
        }
    
    def _generate_recommendations(self, blur_metrics: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if blur_metrics['blur_detected']:
            severity = blur_metrics['blur_severity']
            
            if severity == "严重模糊":
                recommendations.append("建议重新录制视频")
                recommendations.append("使用三脚架或稳定器")
                recommendations.append("检查相机对焦设置")
            elif severity == "中等模糊":
                recommendations.append("建议使用稳定器")
                recommendations.append("提高录制帧率")
                recommendations.append("确保充足光线")
            else:  # 轻微模糊
                recommendations.append("可考虑后期处理")
                recommendations.append("轻微模糊，影响较小")
        else:
            recommendations.append("视频质量良好")
            recommendations.append("无需特殊处理")
        
        return recommendations
    
    def batch_detect(self, video_dir: str, output_dir: str = "./blur_detection_results") -> Dict:
        """批量检测视频模糊"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        print(f"开始批量检测 {len(video_files)} 个视频...")
        
        for video_file in video_files:
            try:
                result = self.detect_blur(str(video_file))
                results.append(result)
                print(f"  {video_file.name}: {result['blur_severity']} (置信度: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"  处理 {video_file.name} 时出错: {e}")
                results.append({
                    'video_path': str(video_file),
                    'blur_detected': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # 保存结果
        self._save_results(results, output_dir)
        
        return {
            'total_videos': len(video_files),
            'processed_videos': len(results),
            'blur_detected_count': sum(1 for r in results if r.get('blur_detected', False)),
            'results': results
        }
    
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
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """保存检测结果"""
        # 保存JSON结果
        json_path = os.path.join(output_dir, 'blur_detection_results.json')
        # 转换数据为JSON可序列化格式
        serializable_results = self._make_json_serializable(results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV摘要
        csv_path = os.path.join(output_dir, 'blur_detection_summary.csv')
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Blur_Detected', 'Confidence', 'Severity', 'Blur_Ratio', 'Blur_Frames'])
            
            for result in results:
                writer.writerow([
                    os.path.basename(result.get('video_path', '')),
                    result.get('blur_detected', False),
                    f"{result.get('confidence', 0.0):.3f}",
                    result.get('blur_severity', ''),
                    f"{result.get('blur_ratio', 0.0):.3f}",
                    result.get('blur_frame_count', 0)
                ])
        
        # 生成统计报告
        self._generate_statistics_report(results, output_dir)
        
        print(f"检测结果已保存到: {output_dir}")
    
    def _generate_statistics_report(self, results: List[Dict], output_dir: str):
        """生成统计报告"""
        # 计算统计信息
        total_videos = len(results)
        blur_detected_count = sum(1 for r in results if r.get('blur_detected', False))
        confidence_scores = [r.get('confidence', 0.0) for r in results if 'error' not in r]
        
        # 统计模糊严重程度
        severity_counts = {}
        for result in results:
            if result.get('blur_detected', False):
                severity = result.get('blur_severity', '未知')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        report = f"""
# 视频模糊检测统计报告

## 基本统计
- 总视频数量: {total_videos}
- 检测到模糊的视频: {blur_detected_count}
- 模糊检测率: {blur_detected_count/total_videos*100:.1f}%

## 置信度统计
- 平均置信度: {np.mean(confidence_scores):.3f}
- 最低置信度: {np.min(confidence_scores):.3f}
- 最高置信度: {np.max(confidence_scores):.3f}
- 置信度标准差: {np.std(confidence_scores):.3f}

## 模糊严重程度分布
"""
        
        for severity, count in severity_counts.items():
            report += f"- {severity}: {count} 个视频\n"
        
        # 保存报告
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

