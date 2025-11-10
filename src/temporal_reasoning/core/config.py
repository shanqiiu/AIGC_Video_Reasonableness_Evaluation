# -*- coding: utf-8 -*-
"""
时序合理性主流程的配置定义。
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class RAFTConfig:
    """RAFT 光流模型配置。"""
    model_path: str = ""
    model_type: str = "large"  # 支持 large 或 small
    use_gpu: bool = True
    batch_size: int = 1
    motion_discontinuity_threshold: float = 0.3  # 运动突变判定阈值


@dataclass
class GroundingDINOConfig:
    """Grounding DINO 检测模型配置。"""
    model_path: str = ""  # 模型权重路径
    config_path: str = ""  # 配置文件路径
    bert_path: str = ""  # 本地 BERT 模型目录
    text_threshold: float = 0.25
    box_threshold: float = 0.3
    use_gpu: bool = True


@dataclass
class SAMConfig:
    """SAM / SAM2 分割模型配置。"""
    model_path: str = ""  # 权重文件路径
    config_path: str = ""  # 配置文件路径（SAM2 必填）
    model_type: str = "sam2_h"  # 可选 sam2_h / sam2_l / sam2_b
    use_gpu: bool = True


@dataclass
class TrackerConfig:
    """跟踪 / 验证模块配置。"""
    type: str = "deaot"  # 支持 deaot 或 cotracker
    model_path: Optional[str] = None
    use_gpu: bool = True
    enable_cotracker_validation: bool = True  # 是否启用 Co-Tracker 校验
    cotracker_checkpoint: Optional[str] = None  # Co-Tracker 权重路径
    grid_size: int = 30  # Co-Tracker 网格密度


@dataclass
class KeypointConfig:
    """关键点分析配置。"""
    model_type: str = "mediapipe"  # 支持 mediapipe 或 mmpose
    model_path: Optional[str] = None
    use_gpu: bool = False  # MediaPipe 默认仅支持 CPU
    
    # 可选可视化配置
    enable_visualization: bool = False  # 是否开启关键点可视化
    visualization_output_dir: Optional[str] = None  # 可视化结果输出目录
    show_face: bool = False  # 是否绘制人脸关键点
    show_face_mesh: bool = False  # 是否绘制人脸网格
    point_radius: int = 3  # 关键点绘制半径
    line_thickness: int = 2  # 关键点连线粗细
    save_visualization: bool = True  # 是否保存可视化结果
    show_visualization: bool = False  # 是否弹窗展示可视化


@dataclass
class FusionConfig:
    """多模态融合与打分配置。"""
    multimodal_confidence_boost: float = 1.2
    min_anomaly_duration_frames: int = 3
    single_modality_confidence_threshold: float = 0.8


@dataclass
class ThresholdsConfig:
    """通用阈值配置。"""
    motion_discontinuity_threshold: float = 0.3
    structure_disappearance_threshold: float = 0.3  # 结构消失判断阈值
    keypoint_displacement_threshold: int = 10  # 关键点位移阈值（像素）


@dataclass
class TemporalReasoningConfig:
    """时序合理性主流程配置。"""
    device: str = "cuda:0"
    
    # 子模块配置
    raft: RAFTConfig = field(default_factory=RAFTConfig)
    grounding_dino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    structure_prompts: Optional[List[str]] = None
    
    # 输出配置
    output_dir: str = ""
    save_visualizations: bool = True
    save_detailed_reports: bool = True
    
    def __post_init__(self):
        """初始化默认路径与依赖。"""
        # 推导默认目录结构
        base_dir = Path(__file__).parent.parent.parent.parent
        third_party_dir = base_dir / "third_party"
        cache_dir = base_dir / ".cache"
        
        # RAFT 默认路径（权重：.cache，代码：third_party）
        if not self.raft.model_path:
            # 优先查找 .cache
            raft_cache_path = cache_dir / "raft-things.pth"
            if raft_cache_path.exists():
                self.raft.model_path = str(raft_cache_path)
            else:
                raise FileNotFoundError(
                    f"未找到 RAFT 模型权重: {raft_cache_path}\n"
                    f"请确认权重位于 .cache 目录。"
                )
        
        # Grounding DINO 默认路径
        if not self.grounding_dino.model_path:
            # 默认查找 .cache
            gdino_weight = cache_dir / "groundingdino_swinb_cogcoor.pth"
            if gdino_weight.exists():
                self.grounding_dino.model_path = str(gdino_weight)
            else:
                raise FileNotFoundError(
                    f"未找到 Grounding DINO 权重: {gdino_weight}\n"
                    f"请确认权重位于 .cache 目录。"
                )
        
        # Grounding DINO 配置文件
        if not self.grounding_dino.config_path:
            gdino_config = third_party_dir / "Grounded-SAM-2" / "grounding_dino" / "config" / "GroundingDINO_SwinB.py"
            if not gdino_config.exists():
                # 兼容旧项目目录
                gdino_config = third_party_dir / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB.py"
            if gdino_config.exists():
                self.grounding_dino.config_path = str(gdino_config)
            else:
                raise FileNotFoundError(
                    f"未找到 Grounding DINO 配置文件: {gdino_config}\n"
                    f"请确认配置文件存在于 third_party 目录。"
                )
        
        # BERT 模型路径
        if not self.grounding_dino.bert_path:
            bert_path = cache_dir / "google-bert" / "bert-base-uncased"
            if bert_path.exists():
                self.grounding_dino.bert_path = str(bert_path)
            else:
                raise FileNotFoundError(
                    f"未找到 BERT 模型文件: {bert_path}\n"
                    f"请确认模型位于 .cache/google-bert/bert-base-uncased。"
                )
        
        # SAM / SAM2 默认路径
        if not self.sam.model_path:
            # 优先使用 SAM2 权重
            sam2_weight = cache_dir / "sam2.1_hiera_large.pt"
            if sam2_weight.exists():
                self.sam.model_path = str(sam2_weight)
            else:
                # 兼容传统 SAM 权重
                sam_weight = cache_dir / "sam_vit_h_4b8939.pth"
                if sam_weight.exists():
                    self.sam.model_path = str(sam_weight)
                else:
                    raise FileNotFoundError(
                        f"未找到 SAM 权重: {sam2_weight} 或 {sam_weight}\n"
                        f"请确认权重位于 .cache 目录。"
                    )
        
        # SAM2 配置文件
        if not self.sam.config_path and self.sam.model_type.startswith("sam2"):
            sam2_config = third_party_dir / "Grounded-SAM-2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
            if not sam2_config.exists():
                # 兼容旧路径
                sam2_config = third_party_dir / "Grounded-SAM-2" / "sam2" / "configs" / "sam2_hiera_l.yaml"
            if sam2_config.exists():
                self.sam.config_path = str(sam2_config)
            else:
                raise FileNotFoundError(
                    f"未找到 SAM2 配置文件: {sam2_config}\n"
                    f"请确认配置文件存在于 third_party/Grounded-SAM-2/sam2/configs。"
                )
        
        # Co-Tracker 默认路径
        if not self.tracker.cotracker_checkpoint and self.tracker.enable_cotracker_validation:
            cotracker_weight = cache_dir / "scaled_offline.pth"
            if cotracker_weight.exists():
                self.tracker.cotracker_checkpoint = str(cotracker_weight)
            else:
                raise FileNotFoundError(
                    f"未找到 Co-Tracker 权重: {cotracker_weight}\n"
                    f"请将权重放置在 .cache 目录，或将 tracker.enable_cotracker_validation 设为 False。"
                )
        
        # 输出目录
        if not self.output_dir:
            self.output_dir = str(base_dir / "outputs" / "temporal_reasoning")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        获取指定子模块的模型路径。
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型路径字符串，若未配置则返回空字符串
        """
        path_map = {
            'raft': self.raft.model_path,
            'grounding_dino': self.grounding_dino.model_path,
            'sam': self.sam.model_path,
            'tracker': self.tracker.model_path,
            'keypoint': self.keypoint.model_path
        }
        return path_map.get(model_name, "")
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        用传入字典更新配置。
        
        Args:
            config_dict: 配置字典
        """
        if 'raft' in config_dict:
            for key, value in config_dict['raft'].items():
                setattr(self.raft, key, value)
        
        if 'grounding_dino' in config_dict:
            for key, value in config_dict['grounding_dino'].items():
                setattr(self.grounding_dino, key, value)
        
        if 'sam' in config_dict:
            for key, value in config_dict['sam'].items():
                setattr(self.sam, key, value)
        
        if 'tracker' in config_dict:
            for key, value in config_dict['tracker'].items():
                setattr(self.tracker, key, value)
        
        if 'keypoint' in config_dict:
            for key, value in config_dict['keypoint'].items():
                setattr(self.keypoint, key, value)
        
        if 'fusion' in config_dict:
            for key, value in config_dict['fusion'].items():
                setattr(self.fusion, key, value)
        
        if 'thresholds' in config_dict:
            for key, value in config_dict['thresholds'].items():
                setattr(self.thresholds, key, value)
        
        # 鏇存柊鍏朵粬閰嶇�?
        for key, value in config_dict.items():
            if key not in ['raft', 'grounding_dino', 'sam', 'tracker', 'keypoint', 'fusion', 'thresholds']:
                if hasattr(self, key):
                    setattr(self, key, value)


def load_config_from_yaml(config_path: str) -> TemporalReasoningConfig:
    """
    从 YAML 文件加载配置。
    
    Args:
        config_path: YAML 配置文件路径
    
    Returns:
        TemporalReasoningConfig 实例
    """
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = TemporalReasoningConfig()
        if 'temporal_reasoning' in config_dict:
            config.update_from_dict(config_dict['temporal_reasoning'])
        else:
            config.update_from_dict(config_dict)
        
        return config
    except ImportError:
        raise ImportError("请先安装 pyyaml：pip install pyyaml")
    except Exception as e:
        raise ValueError(f"加载配置文件失败: {e}")


def get_default_config() -> TemporalReasoningConfig:
    """
    获取默认配置。
    
    Returns:
        TemporalReasoningConfig 实例
    """
    return TemporalReasoningConfig()

