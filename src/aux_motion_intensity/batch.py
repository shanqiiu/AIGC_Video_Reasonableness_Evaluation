"""
Batch processing utilities for motion intensity, refactored from AIGC_detector/video_processor.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .analyzer import MotionIntensityAnalyzer


def load_video_frames(video_path: str, max_frames: Optional[int] = None, frame_skip: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
        frame_count += 1
    cap.release()
    return frames


def analyze_single_video(analyzer: MotionIntensityAnalyzer,
                         video_path: str,
                         output_dir: Optional[str] = None,
                         camera_fov: float = 60.0,
                         max_frames: Optional[int] = None,
                         frame_skip: int = 1) -> Dict:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = load_video_frames(video_path, max_frames=max_frames, frame_skip=frame_skip)
    if len(frames) < 2:
        return {
            'video_name': video_name,
            'video_path': video_path,
            'status': 'failed',
            'error': '��Ƶ֡����'
        }
    camera_matrix = analyzer.estimate_camera_matrix(frames[0].shape, camera_fov)
    result = analyzer.analyze_frames(frames, camera_matrix)
    record = {
        'video_name': video_name,
        'video_path': video_path,
        'status': 'success',
        'motion_intensity': result['motion_intensity'],
        'scene_type': result['scene_type'],
        'temporal_stats': result['temporal_stats'],
        'component_scores': result['component_scores'],
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{video_name}_motion_intensity.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def batch_analyze_videos(analyzer: MotionIntensityAnalyzer,
                         input_dir: str,
                         output_dir: Optional[str] = None,
                         camera_fov: float = 60.0,
                         max_frames: Optional[int] = None,
                         frame_skip: int = 1) -> List[Dict]:
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files: List[str] = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    video_files = sorted(list(set(video_files)))
    results: List[Dict] = []
    per_video_dir = None
    if output_dir:
        per_video_dir = os.path.join(output_dir, 'motion_intensity')
        os.makedirs(per_video_dir, exist_ok=True)
    for vp in video_files:
        rec = analyze_single_video(
            analyzer,
            vp,
            output_dir=per_video_dir,
            camera_fov=camera_fov,
            max_frames=max_frames,
            frame_skip=frame_skip,
        )
        results.append(rec)
    if output_dir:
        summary = _build_summary(results)
        with open(os.path.join(output_dir, 'motion_intensity_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return results


def _build_summary(results: List[Dict]) -> Dict:
    successes = [r for r in results if r.get('status') == 'success']
    summary: Dict[str, Union[int, float, Dict[str, int]]] = {
        'total_videos': len(results),
        'successful': len(successes),
        'failed': len(results) - len(successes),
    }
    if successes:
        scores = [r['motion_intensity'] for r in successes]
        scene_types = [r['scene_type'] for r in successes]
        summary.update({
            'mean_motion_intensity': float(np.mean(scores)),
            'std_motion_intensity': float(np.std(scores)),
            'min_motion_intensity': float(np.min(scores)),
            'max_motion_intensity': float(np.max(scores)),
            'scene_type_distribution': {
                'static': int(scene_types.count('static')),
                'dynamic': int(scene_types.count('dynamic')),
            }
        })
    return summary


