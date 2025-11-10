"""
统一动态评分器，迁移自 AIGC_detector/unified_dynamics_scorer.py。
"""

from typing import Dict, List, Optional

import numpy as np


class UnifiedDynamicsScorer:
    def __init__(self,
                 mode: str = 'auto',
                 weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 use_normalized_flow: bool = False,
                 baseline_diagonal: float = 1469.0) -> None:
        self.mode = mode
        self.use_normalized_flow = use_normalized_flow
        self.default_weights = {
            'flow_magnitude': 0.35,
            'spatial_coverage': 0.25,
            'temporal_variation': 0.20,
            'spatial_consistency': 0.10,
            'camera_factor': 0.10,
        }
        self.weights = weights if weights is not None else self.default_weights
        if use_normalized_flow:
            self.default_thresholds = {
                'flow_low': 1.0 / baseline_diagonal,
                'flow_mid': 5.0 / baseline_diagonal,
                'flow_high': 15.0 / baseline_diagonal,
                'static_ratio': 0.5,
                'temporal_std': 1.0 / baseline_diagonal,
            }
        else:
            self.default_thresholds = {
                'flow_low': 1.0,
                'flow_mid': 5.0,
                'flow_high': 15.0,
                'static_ratio': 0.5,
                'temporal_std': 1.0,
            }
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds

    def calculate_unified_score(self, temporal_result: Dict, camera_compensation_enabled: bool = False) -> Dict:
        temporal_stats = temporal_result['temporal_stats']
        frame_results = temporal_result['frame_results']
        scene_type = self._detect_scene_type(temporal_stats, camera_compensation_enabled)
        component_scores = self._calculate_component_scores(temporal_stats, frame_results, camera_compensation_enabled, scene_type)
        unified_score = self._weighted_fusion(component_scores, scene_type)
        confidence = self._calculate_confidence(component_scores, temporal_stats)
        interpretation = self._generate_interpretation(unified_score, scene_type, component_scores)
        return {
            'unified_dynamics_score': float(unified_score),
            'scene_type': scene_type,
            'confidence': float(confidence),
            'component_scores': component_scores,
            'interpretation': interpretation,
            'normalization_params': {
                'mode': self.mode,
                'detected_scene': scene_type,
                'weights_used': self.weights,
            },
        }

    def _detect_scene_type(self, temporal_stats: Dict, camera_comp_enabled: bool) -> str:
        if self.mode == 'static_scene':
            return 'static'
        if self.mode == 'dynamic_scene':
            return 'dynamic'
        mean_static_ratio = temporal_stats['mean_static_ratio']
        if camera_comp_enabled and mean_static_ratio > self.thresholds['static_ratio']:
            return 'static'
        return 'dynamic'

    def _calculate_component_scores(self, temporal_stats: Dict, frame_results: List[Dict], camera_comp_enabled: bool, scene_type: str) -> Dict:
        flow_score = self._calculate_flow_magnitude_score(temporal_stats, camera_comp_enabled, scene_type)
        spatial_score = self._calculate_spatial_coverage_score(temporal_stats)
        temporal_score = self._calculate_temporal_variation_score(temporal_stats)
        consistency_score = self._calculate_spatial_consistency_score(temporal_stats)
        camera_score = self._calculate_camera_factor(temporal_stats, camera_comp_enabled)
        return {
            'flow_magnitude': flow_score,
            'spatial_coverage': spatial_score,
            'temporal_variation': temporal_score,
            'spatial_consistency': consistency_score,
            'camera_factor': camera_score,
        }

    def _calculate_flow_magnitude_score(self, temporal_stats: Dict, camera_comp_enabled: bool, scene_type: str) -> float:
        if scene_type == 'static' and camera_comp_enabled:
            raw_value = temporal_stats['mean_dynamics_score']
            score = self._sigmoid_normalize(raw_value, threshold=self.thresholds['flow_mid'], steepness=0.5)
        else:
            raw_value = temporal_stats.get('mean_flow_magnitude', temporal_stats['mean_dynamics_score'] * 2)
            score = self._sigmoid_normalize(raw_value, threshold=self.thresholds['flow_high'], steepness=0.3)
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_spatial_coverage_score(self, temporal_stats: Dict) -> float:
        dynamic_ratio = 1.0 - temporal_stats['mean_static_ratio']
        return float(np.clip(dynamic_ratio, 0.0, 1.0))

    def _calculate_temporal_variation_score(self, temporal_stats: Dict) -> float:
        std_dynamics = temporal_stats['std_dynamics_score']
        temporal_threshold = self.thresholds.get('temporal_std', 1.0)
        score = self._sigmoid_normalize(std_dynamics, threshold=temporal_threshold, steepness=1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_spatial_consistency_score(self, temporal_stats: Dict) -> float:
        consistency = temporal_stats['mean_consistency_score']
        score = 1.0 - consistency
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_camera_factor(self, temporal_stats: Dict, camera_comp_enabled: bool) -> float:
        if not camera_comp_enabled:
            return 0.5
        if 'camera_compensation_stats' in temporal_stats:
            comp_stats = temporal_stats['camera_compensation_stats']
            success_rate = comp_stats.get('success_rate', 0.5)
            score = 1.0 - success_rate
            return float(np.clip(score, 0.0, 1.0))
        return 0.5

    def _sigmoid_normalize(self, value: float, threshold: float = 5.0, steepness: float = 0.5) -> float:
        return 1.0 / (1.0 + np.exp(-steepness * (value - threshold)))

    def _weighted_fusion(self, component_scores: Dict, scene_type: str) -> float:
        if scene_type == 'static':
            adjusted_weights = {
                'flow_magnitude': 0.45,
                'spatial_coverage': 0.30,
                'temporal_variation': 0.10,
                'spatial_consistency': 0.05,
                'camera_factor': 0.10,
            }
        else:
            adjusted_weights = {
                'flow_magnitude': 0.45,
                'spatial_coverage': 0.30,
                'temporal_variation': 0.15,
                'spatial_consistency': 0.05,
                'camera_factor': 0.05,
            }
        weighted_sum = 0.0
        total_weight = 0.0
        for key, score in component_scores.items():
            weight = adjusted_weights.get(key, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        unified_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        return float(np.clip(unified_score, 0.0, 1.0))

    def _calculate_confidence(self, component_scores: Dict, temporal_stats: Dict) -> float:
        stability = temporal_stats.get('temporal_stability', 0.5)
        scores = list(component_scores.values())
        score_std = np.std(scores)
        consistency = 1.0 / (1.0 + score_std)
        confidence = 0.6 * stability + 0.4 * consistency
        return float(np.clip(confidence, 0.0, 1.0))

    def _generate_interpretation(self, unified_score: float, scene_type: str, component_scores: Dict) -> str:
        if unified_score < 0.2:
            level = '极低动态（纯静态）'
        elif unified_score < 0.4:
            level = '低动态'
        elif unified_score < 0.6:
            level = '中等动态'
        elif unified_score < 0.8:
            level = '高动态'
        else:
            level = '极高动态'
        scene_desc = '静态场景（相机运动）' if scene_type == 'static' else '动态场景（物体运动）'
        max_component = max(component_scores, key=component_scores.get)
        component_names = {
            'flow_magnitude': '光流幅度',
            'spatial_coverage': '运动覆盖',
            'temporal_variation': '时序变化',
            'spatial_consistency': '空间一致性',
            'camera_factor': '相机因子',
        }
        main_factor = component_names.get(max_component, max_component)
        interpretation = (
            f"动态度: {unified_score:.3f} ({level})\n"
            f"场景类型: {scene_desc}\n"
            f"主要贡献: {main_factor} ({component_scores[max_component]:.3f})\n"
        )
        return interpretation.strip()


class DynamicsClassifier:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.default_thresholds = {
            'pure_static': 0.15,
            'low_dynamic': 0.35,
            'medium_dynamic': 0.65,
            'high_dynamic': 0.85,
        }
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds

    def classify(self, unified_score: float) -> Dict:
        if unified_score < self.thresholds['pure_static']:
            return {'category': 'pure_static', 'category_id': 0, 'description': '纯静态物体', 'typical_examples': ['建筑物', '雕塑', '静止的风景']}
        if unified_score < self.thresholds['low_dynamic']:
            return {'category': 'low_dynamic', 'category_id': 1, 'description': '低动态场景', 'typical_examples': ['飘动的旗帜', '缓慢移动的云', '微风中的树叶']}
        if unified_score < self.thresholds['medium_dynamic']:
            return {'category': 'medium_dynamic', 'category_id': 2, 'description': '中等动态场景', 'typical_examples': ['行走的人', '慢跑', '日常活动']}
        if unified_score < self.thresholds['high_dynamic']:
            return {'category': 'high_dynamic', 'category_id': 3, 'description': '高动态场景', 'typical_examples': ['跑步', '跳舞', '体育运动']}
        return {'category': 'extreme_dynamic', 'category_id': 4, 'description': '极高动态场景', 'typical_examples': ['快速舞蹈', '激烈运动', '打斗场面']}

    def get_binary_label(self, unified_score: float, threshold: float = 0.5) -> int:
        return 1 if unified_score >= threshold else 0


