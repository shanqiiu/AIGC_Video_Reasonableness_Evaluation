"""
Static object dynamics analysis migrated from AIGC_detector/static_object_analyzer.py
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


class StaticObjectDetector:
    def __init__(self,
                 flow_threshold: float = 2.0,
                 flow_threshold_ratio: float = 0.002,
                 use_normalized_flow: bool = False,
                 consistency_threshold: float = 0.8,
                 min_region_size: int = 100) -> None:
        self.flow_threshold = flow_threshold
        self.flow_threshold_ratio = flow_threshold_ratio
        self.use_normalized_flow = use_normalized_flow
        self.consistency_threshold = consistency_threshold
        self.min_region_size = min_region_size

    def detect_static_regions(self, flow: np.ndarray, image_shape: Optional[np.ndarray] = None) -> np.ndarray:
        flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        if self.use_normalized_flow and image_shape is not None:
            h, w = image_shape[:2]
            diagonal = np.sqrt(h ** 2 + w ** 2)
            flow_magnitude = flow_magnitude / diagonal
            threshold = self.flow_threshold_ratio
            static_mask = flow_magnitude < threshold
        else:
            threshold = self.flow_threshold
            static_mask = flow_magnitude < threshold
        kernel = np.ones((5, 5), np.uint8)
        static_mask = cv2.morphologyEx(static_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel)
        static_mask = self.remove_small_regions(static_mask, self.min_region_size)
        return static_mask.astype(bool)

    def remove_small_regions(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        labeled, num_labels = ndimage.label(mask)
        for i in range(1, num_labels + 1):
            if np.sum(labeled == i) < min_size:
                mask[labeled == i] = 0
        return mask

    def refine_static_regions(self, static_mask: np.ndarray, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_mask = gradient_magnitude > np.percentile(gradient_magnitude, 75)
        flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        if self.use_normalized_flow:
            h, w = image.shape[:2]
            diagonal = np.sqrt(h ** 2 + w ** 2)
            flow_magnitude = flow_magnitude / diagonal
            strict_static_mask = flow_magnitude < (self.flow_threshold_ratio * 0.5)
        else:
            strict_static_mask = flow_magnitude < (self.flow_threshold * 0.5)
        refined_mask = static_mask.copy()
        if edge_mask.shape != refined_mask.shape:
            target_height, target_width = flow.shape[:2]
            if edge_mask.shape != (target_height, target_width):
                edge_mask = cv2.resize(edge_mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if refined_mask.shape != (target_height, target_width):
                refined_mask = cv2.resize(refined_mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        refined_mask[edge_mask] = strict_static_mask[edge_mask]
        return refined_mask


class StaticObjectDynamicsCalculator:
    def __init__(self,
                 temporal_window: int = 5,
                 spatial_kernel_size: int = 5,
                 dynamics_threshold: float = 1.0,
                 use_normalized_flow: bool = False,
                 flow_threshold_ratio: float = 0.002) -> None:
        self.temporal_window = temporal_window
        self.spatial_kernel_size = spatial_kernel_size
        self.dynamics_threshold = dynamics_threshold
        self.use_normalized_flow = use_normalized_flow
        self.static_detector = StaticObjectDetector(use_normalized_flow=use_normalized_flow, flow_threshold_ratio=flow_threshold_ratio)

    def calculate_frame_dynamics(self, flow: np.ndarray, image1: np.ndarray, image2: np.ndarray, camera_matrix: Optional[np.ndarray] = None) -> Dict:
        static_mask = self.static_detector.detect_static_regions(flow, image1.shape)
        refined_static_mask = self.static_detector.refine_static_regions(static_mask, image1, flow)
        normalization_factor = 1.0
        if self.use_normalized_flow:
            h, w = image1.shape[:2]
            normalization_factor = np.sqrt(h ** 2 + w ** 2)
        static_dynamics = self.calculate_static_region_dynamics(flow, refined_static_mask, normalization_factor)
        global_dynamics = self.calculate_global_dynamics(flow, refined_static_mask, normalization_factor)
        return {
            'static_mask': refined_static_mask,
            'compensated_flow': flow,
            'static_dynamics': static_dynamics,
            'global_dynamics': global_dynamics,
            'camera_motion': None,
            'original_flow': flow,
        }

    def calculate_static_region_dynamics(self, flow: np.ndarray, static_mask: np.ndarray, normalization_factor: float = 1.0) -> Dict:
        if not np.any(static_mask):
            return {
                'mean_magnitude': 0.0,
                'std_magnitude': 0.0,
                'max_magnitude': 0.0,
                'dynamics_score': 0.0,
                'normalization_factor': float(normalization_factor),
                'is_normalized': self.use_normalized_flow,
            }
        static_flow_x = flow[:, :, 0][static_mask]
        static_flow_y = flow[:, :, 1][static_mask]
        flow_magnitude = np.sqrt(static_flow_x ** 2 + static_flow_y ** 2)
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude = flow_magnitude / normalization_factor
        mean_magnitude = np.mean(flow_magnitude)
        std_magnitude = np.std(flow_magnitude)
        max_magnitude = np.max(flow_magnitude)
        dynamics_score = mean_magnitude + 0.5 * std_magnitude
        return {
            'mean_magnitude': float(mean_magnitude),
            'std_magnitude': float(std_magnitude),
            'max_magnitude': float(max_magnitude),
            'dynamics_score': float(dynamics_score),
            'normalization_factor': float(normalization_factor),
            'is_normalized': self.use_normalized_flow,
        }

    def calculate_global_dynamics(self, flow: np.ndarray, static_mask: np.ndarray, normalization_factor: float = 1.0) -> Dict:
        h, w = flow.shape[:2]
        total_pixels = h * w
        static_pixels = np.sum(static_mask)
        dynamic_pixels = total_pixels - static_pixels
        static_ratio = static_pixels / total_pixels
        if dynamic_pixels > 0:
            dynamic_flow_x = flow[:, :, 0][~static_mask]
            dynamic_flow_y = flow[:, :, 1][~static_mask]
            dynamic_magnitude = np.sqrt(dynamic_flow_x ** 2 + dynamic_flow_y ** 2)
            if self.use_normalized_flow and normalization_factor > 0:
                dynamic_magnitude = dynamic_magnitude / normalization_factor
            mean_dynamic_magnitude = np.mean(dynamic_magnitude)
        else:
            mean_dynamic_magnitude = 0.0
        flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        if self.use_normalized_flow and normalization_factor > 0:
            flow_magnitude = flow_magnitude / normalization_factor
        consistency_score = 1.0 - (np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-6))
        return {
            'static_ratio': float(static_ratio),
            'dynamic_ratio': float(1.0 - static_ratio),
            'mean_dynamic_magnitude': float(mean_dynamic_magnitude),
            'consistency_score': float(max(0.0, consistency_score)),
        }

    def calculate_temporal_dynamics(self, flows: List[np.ndarray], images: List[np.ndarray], camera_matrix: Optional[np.ndarray] = None) -> Dict:
        if len(flows) != len(images) - 1:
            raise ValueError('光流数量应该比图像数量少1')
        frame_results = []
        for i, flow in enumerate(flows):
            result = self.calculate_frame_dynamics(flow, images[i], images[i + 1], camera_matrix)
            frame_results.append(result)
        temporal_stats = self.calculate_temporal_statistics(frame_results)
        return {'frame_results': frame_results, 'temporal_stats': temporal_stats}

    def calculate_temporal_statistics(self, frame_results: List[Dict]) -> Dict:
        dynamics_scores = [r['static_dynamics']['dynamics_score'] for r in frame_results]
        static_ratios = [r['global_dynamics']['static_ratio'] for r in frame_results]
        consistency_scores = [r['global_dynamics']['consistency_score'] for r in frame_results]
        return {
            'mean_dynamics_score': float(np.mean(dynamics_scores)),
            'std_dynamics_score': float(np.std(dynamics_scores)),
            'max_dynamics_score': float(np.max(dynamics_scores)),
            'min_dynamics_score': float(np.min(dynamics_scores)),
            'mean_static_ratio': float(np.mean(static_ratios)),
            'std_static_ratio': float(np.std(static_ratios)),
            'mean_consistency_score': float(np.mean(consistency_scores)),
            'temporal_stability': float(1.0 / (1.0 + np.std(dynamics_scores))),
        }

    def visualize_results(self, image: np.ndarray, flow: np.ndarray, result: Dict, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        im1 = axes[0, 1].imshow(flow_magnitude, cmap='jet')
        axes[0, 1].set_title('Residual Flow Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        step = 16
        y, x = np.mgrid[step // 2:flow.shape[0]:step, step // 2:flow.shape[1]:step]
        fx = flow[::step, ::step, 0]
        fy = flow[::step, ::step, 1]
        axes[0, 2].imshow(image)
        axes[0, 2].quiver(x, y, fx, fy, color='red', alpha=0.7)
        axes[0, 2].set_title('Flow Vectors')
        axes[0, 2].axis('off')
        axes[1, 0].imshow(result['static_mask'], cmap='gray')
        axes[1, 0].set_title('Static Regions Mask')
        axes[1, 0].axis('off')
        overlay = image.copy()
        if len(overlay.shape) == 3:
            overlay[result['static_mask']] = [0, 255, 0]
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Static Regions Overlay')
        axes[1, 1].axis('off')
        static_mask = result['static_mask']
        compensated_flow = result['compensated_flow']
        if np.any(static_mask):
            static_flow_x = compensated_flow[:, :, 0][static_mask]
            static_flow_y = compensated_flow[:, :, 1][static_mask]
            static_magnitude = np.sqrt(static_flow_x ** 2 + static_flow_y ** 2)
            axes[1, 2].hist(static_magnitude, bins=50, alpha=0.7, label='Static Regions')
        if np.any(~static_mask):
            dynamic_flow_x = compensated_flow[:, :, 0][~static_mask]
            dynamic_flow_y = compensated_flow[:, :, 1][~static_mask]
            dynamic_magnitude = np.sqrt(dynamic_flow_x ** 2 + dynamic_flow_y ** 2)
            axes[1, 2].hist(dynamic_magnitude, bins=50, alpha=0.7, label='Dynamic Regions')
        axes[1, 2].set_title('Flow Magnitude Distribution')
        axes[1, 2].set_xlabel('Flow Magnitude')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


