"""
相机运动补偿模块，迁移自 AIGC_detector/dynamic_motion_compensation/camera_compensation.py。
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CameraCompensator:
    def __init__(self, ransac_thresh: float = 0.5, ransac_iters: int = 5000, feature: str = 'SIFT',
                 max_features: int = 5000, temporal_smooth: bool = True, smooth_window: int = 3) -> None:
        if feature == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif feature == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.03, edgeThreshold=15)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.ransac_thresh = ransac_thresh
        self.ransac_iters = ransac_iters
        self.temporal_smooth = temporal_smooth
        self.smooth_window = smooth_window
        self.feature_type = feature
        self.homography_buffer: Optional[deque] = deque(maxlen=smooth_window) if temporal_smooth else None

    def estimate_homography(self, img1: np.ndarray, img2: np.ndarray, min_matches: int = 20) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

        k1, d1 = self.detector.detectAndCompute(g1, None)
        k2, d2 = self.detector.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < min_matches or len(k2) < min_matches:
            return None, None, []

        raw_matches = self.matcher.knnMatch(d1, d2, k=2)
        matches = []
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                ratio = 0.7 if self.feature_type == 'SIFT' else 0.75
                if m.distance < ratio * n.distance:
                    matches.append(m)
        if len(matches) < min_matches:
            return None, None, []

        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_thresh, maxIters=self.ransac_iters, confidence=0.999)
        return H, mask, matches

    def smooth_homography(self, H: np.ndarray) -> np.ndarray:
        if not self.temporal_smooth or self.homography_buffer is None:
            return H
        self.homography_buffer.append(H.copy())
        if len(self.homography_buffer) == 1:
            return H
        weights = np.linspace(0.5, 1.0, len(self.homography_buffer))
        weights = weights / weights.sum()
        smoothed_H = np.zeros_like(H)
        for w, h in zip(weights, self.homography_buffer):
            smoothed_H += w * h
        smoothed_H = smoothed_H / smoothed_H[2, 2]
        return smoothed_H

    def reset_temporal_buffer(self) -> None:
        if self.homography_buffer is not None:
            self.homography_buffer.clear()

    def camera_flow_from_homography(self, H: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
        tc = (H @ coords.T).T
        tc = tc[:, :2] / tc[:, 2:3]
        cam_flow = tc - coords[:, :2]
        return cam_flow.reshape(h, w, 2)

    def compensate(self, flow: np.ndarray, img1: np.ndarray, img2: np.ndarray) -> Dict:
        H, mask, matches = self.estimate_homography(img1, img2)
        if H is None:
            return {
                'homography': None,
                'camera_flow': np.zeros_like(flow),
                'residual_flow': flow,
                'inliers': 0,
                'total_matches': 0,
                'match_quality': 0.0,
            }
        H_smoothed = self.smooth_homography(H)
        cam_flow = self.camera_flow_from_homography(H_smoothed, flow.shape[:2])
        residual = flow - cam_flow
        inliers = int(mask.sum()) if mask is not None else 0
        total_matches = len(matches)
        match_quality = inliers / total_matches if total_matches > 0 else 0.0
        return {
            'homography': H_smoothed,
            'camera_flow': cam_flow,
            'residual_flow': residual,
            'inliers': inliers,
            'total_matches': total_matches,
            'match_quality': match_quality,
        }


