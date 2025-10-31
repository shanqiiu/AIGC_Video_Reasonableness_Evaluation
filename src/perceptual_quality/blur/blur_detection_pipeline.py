# -*- coding: utf-8 -*-
"""
基于VMBench的视频模糊检测系统
主要使用MSS (运动平滑度评分) 和 PAS (可感知幅度评分) 进行模糊检测
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 添加VMBench路径
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# AIGC_Video_Reasonableness_Evaluation 项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
third_party_dir = os.path.join(project_root, 'third_party')
workspace_root = os.path.abspath(os.path.join(project_root, '..'))
vmb_root = os.path.join(workspace_root, 'VMBench_diy')

# 保存原始工作目录
original_cwd = os.getcwd()

try:
    # 临时切换到项目根目录以便导入
    os.chdir(project_root)
    
    # 添加模块搜索路径（按新工程结构与外部VMBench目录）
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)
    # Grounded-Segment-Anything 子模块
    gsa_dir = os.path.join(third_party_dir, "Grounded-Segment-Anything")
    for p in [
        gsa_dir,
        os.path.join(gsa_dir, "GroundingDINO"),
        os.path.join(gsa_dir, "segment_anything"),
    ]:
        if p not in sys.path:
            sys.path.append(p)
    # Co-Tracker
    cot_dir = os.path.join(third_party_dir, "co-tracker")
    if cot_dir not in sys.path:
        sys.path.append(cot_dir)
    # VMBench_diy 根目录（提供 motion_smoothness_score / perceptible_amplitude_score）
    if os.path.isdir(vmb_root) and vmb_root not in sys.path:
        sys.path.insert(0, vmb_root)
    
    # 导入本地 MSS/PAS 封装（移除对外部 VMBench 实现的直接依赖）
    from .mss_scorer import MSSScorer
    from .pas_scorer import PASScorer
    
finally:
    # 恢复原始工作目录
    os.chdir(original_cwd)


class BlurDetectionPipeline:
    """基于VMBench的视频模糊检测管道"""
    
    def __init__(self, device="cuda:0", model_paths=None):
        """
        初始化模糊检测管道
        
        Args:
            device: 计算设备
            model_paths: 模型路径配置
        """
        self.device = device
        self.model_paths = model_paths or self._get_default_model_paths()
        
        # 初始化模型
        self._init_models()
        
        # 检测参数
        self.blur_thresholds = {
            'mss_threshold': 0.025,  # MSS检测阈值
            'pas_threshold': 0.1,   # PAS检测阈值
            'confidence_threshold': 0.7  # 综合置信度阈值
        }
        
    def _get_default_model_paths(self):
        """获取默认模型路径"""
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        return {
            'q_align_model': ".cache/q-future/one-align",
            'grounding_dino_config': os.path.join(project_root, "Grounded-Segment-Anything", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinB.py"),
            'grounding_dino_checkpoint': ".cache/groundingdino_swinb_cogcoor.pth",
            'bert_path': ".cache/google-bert/bert-base-uncased",
            'sam_checkpoint': ".cache/sam_vit_h_4b8939.pth",
            'cotracker_checkpoint': ".cache/scaled_offline.pth"
        }
    
    def _init_models(self):
        """初始化所有需要的模型"""
        print("正在初始化模糊检测模型...")
        
        try:
            # 初始化 MSS 评分器
            print("  初始化MSS评分器...")
            self.mss_scorer = MSSScorer(
                device=self.device,
                model_paths=self.model_paths,
            )
            
            # 初始化 PAS 评分器（本地薄封装）
            print("  初始化PAS评分器...")
            self.pas_scorer = PASScorer(
                device=self.device,
                model_paths=self.model_paths,
            )
            
            print("所有模型初始化完成！")
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
    
    def detect_blur_in_video(self, video_path: str, subject_noun: str = "person") -> Dict:
        """
        检测视频中的模糊异常
        
        Args:
            video_path: 视频文件路径
            subject_noun: 主体对象名称
            
        Returns:
            检测结果字典
        """
        print(f"开始检测视频模糊: {video_path}")
        
        try:
            # 1. 使用MSS评分器检测模糊
            mss_results = self._detect_blur_with_mss(video_path)
            
            # 2. 使用PAS评分器辅助验证
            pas_results = self._detect_blur_with_pas(video_path, subject_noun)
            
            # 3. 综合判断模糊检测结果
            blur_results = self._combine_blur_detection(mss_results, pas_results)
            
            # 4. 生成检测报告
            detection_report = self._generate_blur_report(video_path, blur_results)
            
            return detection_report
            
        except Exception as e:
            print(f"模糊检测过程中出错: {e}")
            return {
                'blur_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'mss_score': 0.0,
                'pas_score': 0.0,
                'blur_frames': []
            }
    
    def _detect_blur_with_mss(self, video_path: str) -> Dict:
        """使用MSS评分器检测模糊"""
        try:
            # 计算质量分数
            mss_output = self.mss_scorer.score(video_path)
            quality_scores = mss_output.get('quality_scores', [])
            
            # 计算相机运动幅度（用于调整阈值）
            camera_movement = self._estimate_camera_movement(video_path)
            
            # 设置自适应阈值
            threshold = self._set_threshold(camera_movement)
            
            # 检测模糊帧
            blur_frames = self._get_artifacts_frames(quality_scores, threshold)
            
            # 计算MSS分数
            mss_score = 1 - len(blur_frames) / len(quality_scores)
            
            # 转换blur_frames为列表（如果是NumPy数组）
            if hasattr(blur_frames, 'tolist'):
                blur_frames_list = blur_frames.tolist()
            else:
                blur_frames_list = list(blur_frames)
            
            return {
                'mss_score': float(mss_score),
                'blur_frames': blur_frames_list,
                'quality_scores': quality_scores,
                'threshold': float(threshold),
                'camera_movement': float(camera_movement)
            }
            
        except Exception as e:
            print(f"MSS检测失败: {e}")
            return {
                'mss_score': 0.0,
                'blur_frames': [],
                'quality_scores': [],
                'threshold': 0.025,
                'camera_movement': 0.0,
                'error': str(e)
            }
    
    def _detect_blur_with_pas(self, video_path: str, subject_noun: str) -> Dict:
        """使用PAS评分器辅助检测模糊"""
        try:
            # 委托给本地 PAS 评分器（可后续替换为完整实现）
            out = self.pas_scorer.score(video_path, subject_noun=subject_noun)
            return {
                'pas_score': float(out.get('pas_score', 0.0)),
                'subject_detected': bool(out.get('subject_detected', False)),
                'motion_degree': float(out.get('motion_degree', 0.0)) if isinstance(out.get('motion_degree', 0.0), (int, float)) else 0.0,
                'error': out.get('error')
            }
            
        except Exception as e:
            print(f"PAS检测失败: {e}")
            return {
                'pas_score': 0.0,
                'subject_detected': False,
                'motion_degree': 0.0,
                'error': str(e)
            }
    
    def _estimate_camera_movement(self, video_path: str) -> float:
        """估算相机运动幅度"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # 读取关键帧
            frame_count = 0
            while frame_count < 10:  # 只取前10帧估算
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
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
    
    def _combine_blur_detection(self, mss_results: Dict, pas_results: Dict) -> Dict:
        """综合MSS和PAS结果判断模糊"""
        mss_score = mss_results.get('mss_score', 0.0)
        pas_score = pas_results.get('pas_score', 0.0)
        blur_frames = mss_results.get('blur_frames', [])
        
        # 计算综合置信度
        # MSS权重0.8，PAS权重0.2
        confidence = mss_score * 0.8 + pas_score * 0.2
        
        # 判断是否检测到模糊
        blur_detected = (
            len(blur_frames) > 0 and 
            confidence < self.blur_thresholds['confidence_threshold']
        )
        
        return {
            'blur_detected': blur_detected,
            'confidence': confidence,
            'mss_score': mss_score,
            'pas_score': pas_score,
            'blur_frames': blur_frames,
            'blur_severity': self._calculate_blur_severity(blur_frames, confidence)
        }

    def _set_threshold(self, camera_movement: float) -> float:
        """根据相机运动幅度设定阈值（简单自适应）。"""
        base = self.blur_thresholds.get('mss_threshold', 0.025)
        # 相机运动越大，阈值略放宽，避免将正常运动误判为模糊
        # 限制在 [base*0.8, base*1.5]
        adjusted = base * (1.0 + 0.5 * min(max(camera_movement, 0.0), 1.0))
        return max(base * 0.8, min(adjusted, base * 1.5))

    def _get_artifacts_frames(self, quality_scores: List[float], threshold: float) -> List[int]:
        """根据质量分数与阈值提取模糊帧索引。"""
        return [i for i, s in enumerate(quality_scores) if s < threshold]
    
    def _calculate_blur_severity(self, blur_frames: List[int], confidence: float) -> str:
        """计算模糊严重程度"""
        blur_ratio = len(blur_frames) / 100  # 假设总帧数100
        
        if blur_ratio > 0.3 or confidence < 0.3:
            return "严重模糊"
        elif blur_ratio > 0.1 or confidence < 0.5:
            return "中等模糊"
        elif blur_ratio > 0.05 or confidence < 0.7:
            return "轻微模糊"
        else:
            return "无模糊"
    
    def _generate_blur_report(self, video_path: str, blur_results: Dict) -> Dict:
        """生成模糊检测报告"""
        report = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'detection_timestamp': str(np.datetime64('now')),
            'blur_detected': blur_results['blur_detected'],
            'confidence': blur_results['confidence'],
            'blur_severity': blur_results['blur_severity'],
            'mss_score': blur_results['mss_score'],
            'pas_score': blur_results['pas_score'],
            'blur_frames': blur_results['blur_frames'],
            'total_blur_frames': len(blur_results['blur_frames']),
            'blur_ratio': len(blur_results['blur_frames']) / 100.0,  # 假设总帧数100
            'recommendations': self._generate_recommendations(blur_results)
        }
        
        return report
    
    def _generate_recommendations(self, blur_results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if blur_results['blur_detected']:
            if blur_results['blur_severity'] == "严重模糊":
                recommendations.append("建议重新录制视频，确保相机稳定")
                recommendations.append("检查相机对焦设置")
            elif blur_results['blur_severity'] == "中等模糊":
                recommendations.append("建议使用三脚架或稳定器")
                recommendations.append("提高录制帧率")
            else:
                recommendations.append("轻微模糊，可考虑后期处理")
        else:
            recommendations.append("视频质量良好，无需处理")
        
        return recommendations
    
    def batch_detect_blur(self, video_dir: str, output_dir: str = "./blur_detection_results") -> Dict:
        """批量检测视频模糊"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
        
        results = []
        
        print(f"开始批量检测 {len(video_files)} 个视频...")
        
        for video_file in tqdm(video_files, desc="模糊检测进度"):
            try:
                result = self.detect_blur_in_video(str(video_file))
                results.append(result)
                
            except Exception as e:
                print(f"处理视频 {video_file.name} 时出错: {e}")
                results.append({
                    'video_path': str(video_file),
                    'blur_detected': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # 保存批量结果
        self._save_batch_results(results, output_dir)
        
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
    
    def _save_batch_results(self, results: List[Dict], output_dir: str):
        """保存批量检测结果"""
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
            writer.writerow(['Video', 'Blur_Detected', 'Confidence', 'Severity', 'MSS_Score', 'PAS_Score', 'Blur_Frames'])
            
            for result in results:
                writer.writerow([
                    os.path.basename(result.get('video_path', '')),
                    result.get('blur_detected', False),
                    f"{result.get('confidence', 0.0):.3f}",
                    result.get('blur_severity', ''),
                    f"{result.get('mss_score', 0.0):.3f}",
                    f"{result.get('pas_score', 0.0):.3f}",
                    len(result.get('blur_frames', []))
                ])
        
        # 生成统计报告
        self._generate_statistics_report(results, output_dir)
        
        print(f"批量检测结果已保存到: {output_dir}")
    
    def _generate_statistics_report(self, results: List[Dict], output_dir: str):
        """生成统计报告"""
        # 计算统计信息
        total_videos = len(results)
        blur_detected_count = sum(1 for r in results if r.get('blur_detected', False))
        confidence_scores = [r.get('confidence', 0.0) for r in results if 'error' not in r]
        
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
        
        # 统计模糊严重程度
        severity_counts = {}
        for result in results:
            if result.get('blur_detected', False):
                severity = result.get('blur_severity', '未知')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            report += f"- {severity}: {count} 个视频\n"
        
        # 保存报告
        report_path = os.path.join(output_dir, 'statistics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
