# -*- coding: utf-8 -*-
"""PAS scorer thin wrapper (Grounded-SAM + Co-Tracker). Delegates to aux_motion_intensity_2 implementation."""

import os
import sys
from typing import Dict, Optional

# 娣诲姞椤圭洰鏍圭洰褰曞埌璺緞浠ヤ究瀵煎叆 aux_motion_intensity_2
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.aux_motion_intensity_2 import PASAnalyzer  # type: ignore


class PASScorer:
    """Thin facade wrapper around aux_motion_intensity_2's PASAnalyzer."""

    def __init__(self, device: str, model_paths: Dict[str, str]):
        self.device = device
        self.model_paths = model_paths
        self._initialized = False
        self._analyzer: Optional[PASAnalyzer] = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        
        # �? model_paths 中提取模型路径，如果不存在则使用默认路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        grounded_checkpoint = self.model_paths.get(
            'grounding_dino_checkpoint',
            os.path.join(project_root, '.cache', 'groundingdino_swinb_cogcoor.pth')
        )
        sam_checkpoint = self.model_paths.get(
            'sam_checkpoint',
            os.path.join(project_root, '.cache', 'sam_vit_h_4b8939.pth')
        )
        cotracker_checkpoint = self.model_paths.get(
            'cotracker_checkpoint',
            os.path.join(project_root, '.cache', 'scaled_offline.pth')
        )
        
        # 初始�? PASAnalyzer
        # 不启用场景分类以保持接口简洁，如需要可以在后续版本添加
        self._analyzer = PASAnalyzer(
            device=self.device,
            grid_size=30,  # 默认网格大小
            enable_scene_classification=False,
            grounded_checkpoint=grounded_checkpoint,
            sam_checkpoint=sam_checkpoint,
            cotracker_checkpoint=cotracker_checkpoint
        )
        
        self._initialized = True

    def score(self, video_path: str, subject_noun: str = "person") -> Dict:
        """
        计算PAS分数
        
        Args:
            video_path: 视频文件路径
            subject_noun: 主体对象名称（如 "person", "dog" 等）
            
        Returns:
            包含PAS分数的字典，格式�?
            {
                'pas_score': float,  # PAS分数（使用pure_subject_motion�?
                'subject_detected': bool,  # 是否检测到主体
                'motion_degree': float,  # 运动幅度（subject_motion�?
                'background_motion': float,  # 背景运动幅度
                'pure_subject_motion': float,  # 纯主体运动幅�?
                'error': str (可�?)  # 错误信息
            }
        """
        self._ensure_init()
        
        if self._analyzer is None:
            return {
                "pas_score": 0.0,
                "subject_detected": False,
                "motion_degree": 0.0,
                "error": "Analyzer not initialized"
            }
        
        try:
            # 调用 PASAnalyzer 进行分析
            result = self._analyzer.analyze_video(
                video_path=video_path,
                subject_noun=subject_noun,
                box_threshold=0.3,
                text_threshold=0.25,
                normalize_by_subject_diag=True
            )
            
            # 转换返回格式以匹�? blur_detection_pipeline 的期�?
            if result.get('status') == 'success':
                # 使用 pure_subject_motion 作为 PAS 分数（这是主体运动减去背景运动后的纯主体运动�?
                pas_score = result.get('pure_subject_motion', 0.0)
                subject_motion = result.get('subject_motion', 0.0)
                background_motion = result.get('background_motion', 0.0)
                
                return {
                    'pas_score': float(pas_score),
                    'subject_detected': True,
                    'motion_degree': float(subject_motion),
                    'background_motion': float(background_motion),
                    'pure_subject_motion': float(pas_score),
                    'total_motion': float(result.get('total_motion', 0.0)),
                    'motion_ratio': float(result.get('motion_ratio', 0.0))
                }
            else:
                # 检测失败或出错
                error_reason = result.get('error_reason', 'unknown_error')
                background_motion = result.get('background_motion', 0.0)
                
                return {
                    'pas_score': 0.0,
                    'subject_detected': False,
                    'motion_degree': 0.0,
                    'background_motion': float(background_motion),
                    'pure_subject_motion': 0.0,
                    'error': error_reason
                }
                
        except Exception as e:
            # 捕获所有异常并返回错误信息
            return {
                "pas_score": 0.0,
                "subject_detected": False,
                "motion_degree": 0.0,
                "background_motion": 0.0,
                "pure_subject_motion": 0.0,
                "error": str(e)
            }

    def preload_models(self) -> None:
        """Ensure analyzer and its heavy models are loaded once upfront."""
        self._ensure_init()
        if self._analyzer is not None:
            try:
                self._analyzer._load_models()
            except Exception:
                # Defer error handling to first score() call to surface within pipeline flow
                pass


