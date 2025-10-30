"""
Motion Intensity Analyzer
High-level API to compute motion_intensity [0,1] and scene_type from frames.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .flow_predictor import SimpleRAFTPredictor
from .camera_compensation import CameraCompensator
from .static_dynamics import StaticObjectDynamicsCalculator
from .unified_dynamics_scorer import UnifiedDynamicsScorer


class MotionIntensityAnalyzer:
    def __init__(self,
                 raft_model_path: Optional[str] = None,
                 device: str = 'cpu',
                 method: str = 'farneback',
                 enable_camera_compensation: bool = True,
                 use_normalized_flow: bool = False,
                 flow_threshold_ratio: float = 0.002,
                 camera_compensation_params: Optional[Dict] = None) -> None:
        self.predictor = SimpleRAFTPredictor(model_path=raft_model_path, device=device, method=method)
        self.enable_camera_compensation = enable_camera_compensation
        self.camera = CameraCompensator(**(camera_compensation_params or {})) if enable_camera_compensation else None
        self.dynamics = StaticObjectDynamicsCalculator(use_normalized_flow=use_normalized_flow, flow_threshold_ratio=flow_threshold_ratio)
        self.scorer = UnifiedDynamicsScorer(mode='auto', use_normalized_flow=use_normalized_flow)

    def estimate_camera_matrix(self, frame_shape: Tuple[int, int], fov: float = 60.0) -> np.ndarray:
        h, w = frame_shape[:2]
        focal_length = w / (2 * np.tan(np.radians(fov / 2)))
        camera_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float32)
        return camera_matrix

    def analyze_frames(self, frames: List[np.ndarray], camera_matrix: Optional[np.ndarray] = None) -> Dict:
        if len(frames) < 2:
            raise ValueError('至少需要2帧图像')
        if camera_matrix is None:
            camera_matrix = self.estimate_camera_matrix(frames[0].shape)
        flows: List[np.ndarray] = []
        for i in range(len(frames) - 1):
            flow = self.predictor.predict_flow(frames[i], frames[i + 1])
            if flow.shape[0] == 2:
                flow = flow.transpose(1, 2, 0)
            if self.enable_camera_compensation and self.camera is not None:
                comp = self.camera.compensate(flow, frames[i], frames[i + 1])
                flows.append(comp['residual_flow'])
            else:
                flows.append(flow)
        temporal = self.dynamics.calculate_temporal_dynamics(flows, frames, camera_matrix)
        unified = self.scorer.calculate_unified_score(temporal, camera_compensation_enabled=self.enable_camera_compensation)
        return {
            'motion_intensity': float(unified['unified_dynamics_score']),
            'scene_type': unified['scene_type'],
            'temporal_stats': temporal['temporal_stats'],
            'component_scores': unified['component_scores'],
        }


