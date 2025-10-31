# -*- coding: utf-8 -*-
"""PAS scorer thin wrapper (Grounded-SAM + Co-Tracker). Delegates to aux_motion_intensity_2 implementation."""

import os
import sys
from typing import Dict, Optional

# �����Ŀ��Ŀ¼��·���Ա㵼�� aux_motion_intensity_2
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
        
        # �� model_paths ����ȡģ��·���������������ʹ��Ĭ��·��
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
        
        # ��ʼ�� PASAnalyzer
        # �����ó��������Ա��ֽӿڼ�࣬����Ҫ�����ں����汾���
        self._analyzer = PASAnalyzer(
            device=self.device,
            grid_size=30,  # Ĭ�������С
            enable_scene_classification=False,
            grounded_checkpoint=grounded_checkpoint,
            sam_checkpoint=sam_checkpoint,
            cotracker_checkpoint=cotracker_checkpoint
        )
        
        self._initialized = True

    def score(self, video_path: str, subject_noun: str = "person") -> Dict:
        """
        ����PAS����
        
        Args:
            video_path: ��Ƶ�ļ�·��
            subject_noun: ����������ƣ��� "person", "dog" �ȣ�
            
        Returns:
            ����PAS�������ֵ䣬��ʽ��
            {
                'pas_score': float,  # PAS������ʹ��pure_subject_motion��
                'subject_detected': bool,  # �Ƿ��⵽����
                'motion_degree': float,  # �˶����ȣ�subject_motion��
                'background_motion': float,  # �����˶�����
                'pure_subject_motion': float,  # �������˶�����
                'error': str (��ѡ)  # ������Ϣ
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
            # ���� PASAnalyzer ���з���
            result = self._analyzer.analyze_video(
                video_path=video_path,
                subject_noun=subject_noun,
                box_threshold=0.3,
                text_threshold=0.25,
                normalize_by_subject_diag=True
            )
            
            # ת�����ظ�ʽ��ƥ�� blur_detection_pipeline ������
            if result.get('status') == 'success':
                # ʹ�� pure_subject_motion ��Ϊ PAS ���������������˶���ȥ�����˶���Ĵ������˶���
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
                # ���ʧ�ܻ����
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
            # ���������쳣�����ش�����Ϣ
            return {
                "pas_score": 0.0,
                "subject_detected": False,
                "motion_degree": 0.0,
                "background_motion": 0.0,
                "pure_subject_motion": 0.0,
                "error": str(e)
            }


