"""
批量分析接口 —— 面向多视频的可感知幅度评分流程。
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
    遍历元信息列表，对每个视频进行可感知幅度分析。
    
    Args:
        analyzer: PASAnalyzer 实例
        meta_info_list: 元数据列表，每项需包含 'filepath' 与可选的 'subject_noun'
        output_path: 可选，分析结果写回的 JSON 路径
        
    Returns:
        List[Dict]: 每个视频的分析结果
    """
    results = []
    
    for meta_info in tqdm(meta_info_list, desc="PAS 分析"):
        result = analyzer.analyze_video(
            video_path=meta_info['filepath'],
            subject_noun=meta_info.get('subject_noun', 'person')
        )
        
        # 将分析结果合并回元信息
        meta_info['perceptible_amplitude_score'] = result
        
        results.append({
            'index': meta_info.get('index', len(results)),
            'prompt': meta_info.get('prompt', ''),
            'result': result
        })
    
    # 可选地写回结果文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info_list, f, indent=4, ensure_ascii=False)
        print(f"分析结果已保存至 {output_path}")
    
    return results

