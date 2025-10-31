"""
��������ӿ� - ���ڴ�������Ƶ
"""

import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm

from .analyzer import PASAnalyzer


def batch_analyze_videos(analyzer: PASAnalyzer,
                        meta_info_list: List[Dict],
                        output_path: Optional[str] = None) -> List[Dict]:
    """
    ����������Ƶ
    
    Args:
        analyzer: PASAnalyzerʵ��
        meta_info_list: Ԫ��Ϣ�б�ÿ��Ԫ�ذ��� 'filepath' �� 'subject_noun'
        output_path: ���JSON�ļ�·������ѡ��
        
    Returns:
        ����б�
    """
    results = []
    
    for meta_info in tqdm(meta_info_list, desc="PAS Analysis"):
        result = analyzer.analyze_video(
            video_path=meta_info['filepath'],
            subject_noun=meta_info.get('subject_noun', 'person')
        )
        
        # �������ӵ�Ԫ��Ϣ��
        meta_info['perceptible_amplitude_score'] = result
        
        results.append({
            'index': meta_info.get('index', len(results)),
            'prompt': meta_info.get('prompt', ''),
            'result': result
        })
    
    # ������
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info_list, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    return results

