# -*- coding: utf-8 -*-
"""
时序合理性分析模块配置管理
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class RAFTConfig:
    """RAFT光流配置"""
    model_path: str = ""
    model_type: str = "large"  # large or small
    use_gpu: bool = True
    batch_size: int = 1
    motion_discontinuity_threshold: float = 0.3  # 运动突变阈值


@dataclass
class GroundingDINOConfig:
    """Grounding DINO配置"""
    model_path: str = ""  # 权重文件路径
    config_path: str = ""  # 配置文件路径
    bert_path: str = ""  # BERT模型路径
    text_threshold: float = 0.25
    box_threshold: float = 0.3
    use_gpu: bool = True


@dataclass
class SAMConfig:
    """SAM配置"""
    model_path: str = ""  # 权重文件路径
    config_path: str = ""  # 配置文件路径（SAM2需要）
    model_type: str = "sam2_h"  # sam2_h, sam2_l, sam2_b
    use_gpu: bool = True


@dataclass
class TrackerConfig:
    """追踪器配置"""
    type: str = "deaot"  # deaot or cotracker
    model_path: Optional[str] = None
    use_gpu: bool = True
    enable_cotracker_validation: bool = True  # 是否启用Co-Tracker验证
    cotracker_checkpoint: Optional[str] = None  # Co-Tracker模型路径
    grid_size: int = 30  # Co-Tracker网格大小


@dataclass
class KeypointConfig:
    """关键点配置"""
    model_type: str = "mediapipe"  # mediapipe or mmpose
    model_path: Optional[str] = None
    use_gpu: bool = False  # MediaPipe不支持GPU


@dataclass
class FusionConfig:
    """融合配置"""
    multimodal_confidence_boost: float = 1.2
    min_anomaly_duration_frames: int = 3
    single_modality_confidence_threshold: float = 0.8


@dataclass
class ThresholdsConfig:
    """阈值配置"""
    motion_discontinuity_threshold: float = 0.3
    structure_disappearance_threshold: float = 0.3  # 掩码面积变化率
    keypoint_displacement_threshold: int = 10  # 像素


@dataclass
class TemporalReasoningConfig:
    """时序合理性分析配置"""
    device: str = "cuda:0"
    
    # 子模块配置
    raft: RAFTConfig = field(default_factory=RAFTConfig)
    grounding_dino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    
    # 输出配置
    output_dir: str = ""
    save_visualizations: bool = True
    save_detailed_reports: bool = True
    
    def __post_init__(self):
        """初始化默认配置"""
        # 设置默认模型路径（基于项目结构）
        base_dir = Path(__file__).parent.parent.parent.parent
        third_party_dir = base_dir / "third_party"
        cache_dir = base_dir / ".cache"
        
        # RAFT默认路径（权重在.cache，代码在third_party）
        if not self.raft.model_path:
            # 从.cache查找RAFT权重
            raft_cache_path = cache_dir / "raft-things.pth"
            if raft_cache_path.exists():
                self.raft.model_path = str(raft_cache_path)
            else:
                raise FileNotFoundError(
                    f"RAFT模型文件未找到: {raft_cache_path}\n"
                    f"请确保权重文件存在于 .cache 目录中"
                )
        
        # Grounding DINO默认路径（权重在.cache，代码在third_party）
        if not self.grounding_dino.model_path:
            # Grounding DINO权重在.cache
            gdino_weight = cache_dir / "groundingdino_swinb_cogcoor.pth"
            if gdino_weight.exists():
                self.grounding_dino.model_path = str(gdino_weight)
            else:
                raise FileNotFoundError(
                    f"Grounding DINO模型文件未找到: {gdino_weight}\n"
                    f"请确保权重文件存在于 .cache 目录中"
                )
        
        # Grounding DINO配置文件路径
        if not self.grounding_dino.config_path:
            gdino_config = third_party_dir / "Grounded-SAM-2" / "grounding_dino" / "config" / "GroundingDINO_SwinB.py"
            if not gdino_config.exists():
                # 尝试另一个路径
                gdino_config = third_party_dir / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB.py"
            if gdino_config.exists():
                self.grounding_dino.config_path = str(gdino_config)
            else:
                raise FileNotFoundError(
                    f"Grounding DINO配置文件未找到: {gdino_config}\n"
                    f"请确保配置文件存在于 third_party 目录中"
                )
        
        # BERT模型路径
        if not self.grounding_dino.bert_path:
            bert_path = cache_dir / "google-bert" / "bert-base-uncased"
            if bert_path.exists():
                self.grounding_dino.bert_path = str(bert_path)
            else:
                raise FileNotFoundError(
                    f"BERT模型文件未找到: {bert_path}\n"
                    f"请确保BERT模型存在于 .cache/google-bert/bert-base-uncased 目录中"
                )
        
        # SAM默认路径（权重在.cache，代码在third_party）
        if not self.sam.model_path:
            # SAM2权重在.cache
            sam2_weight = cache_dir / "sam2.1_hiera_large.pt"
            if sam2_weight.exists():
                self.sam.model_path = str(sam2_weight)
            else:
                # 如果.cache中没有SAM2，尝试旧版SAM权重
                sam_weight = cache_dir / "sam_vit_h_4b8939.pth"
                if sam_weight.exists():
                    self.sam.model_path = str(sam_weight)
                else:
                    raise FileNotFoundError(
                        f"SAM模型文件未找到: {sam2_weight} 或 {sam_weight}\n"
                        f"请确保权重文件存在于 .cache 目录中"
                    )
        
        # SAM2配置文件路径
        if not self.sam.config_path and self.sam.model_type.startswith("sam2"):
            sam2_config = third_party_dir / "Grounded-SAM-2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
            if not sam2_config.exists():
                # 尝试其他路径
                sam2_config = third_party_dir / "Grounded-SAM-2" / "sam2" / "configs" / "sam2_hiera_l.yaml"
            if sam2_config.exists():
                self.sam.config_path = str(sam2_config)
            else:
                raise FileNotFoundError(
                    f"SAM2配置文件未找到: {sam2_config}\n"
                    f"请确保配置文件存在于 third_party/Grounded-SAM-2/sam2/configs 目录中"
                )
        
        # Co-Tracker默认路径（权重在.cache，代码在third_party）
        if not self.tracker.cotracker_checkpoint and self.tracker.enable_cotracker_validation:
            cotracker_weight = cache_dir / "scaled_offline.pth"
            if cotracker_weight.exists():
                self.tracker.cotracker_checkpoint = str(cotracker_weight)
            else:
                raise FileNotFoundError(
                    f"Co-Tracker模型文件未找到: {cotracker_weight}\n"
                    f"请确保权重文件存在于 .cache 目录中，或设置 tracker.enable_cotracker_validation=False 禁用验证"
                )
        
        # 输出目录
        if not self.output_dir:
            self.output_dir = str(base_dir / "outputs" / "temporal_reasoning")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        获取模型路径
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型路径
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
        从字典更新配置
        
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
        
        # 更新其他配置
        for key, value in config_dict.items():
            if key not in ['raft', 'grounding_dino', 'sam', 'tracker', 'keypoint', 'fusion', 'thresholds']:
                if hasattr(self, key):
                    setattr(self, key, value)


def load_config_from_yaml(config_path: str) -> TemporalReasoningConfig:
    """
    从YAML文件加载配置
    
    Args:
        config_path: YAML配置文件路径
    
    Returns:
        配置对象
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
        raise ImportError("请安装pyyaml: pip install pyyaml")
    except Exception as e:
        raise ValueError(f"加载配置文件失败: {e}")


def get_default_config() -> TemporalReasoningConfig:
    """
    获取默认配置
    
    Returns:
        默认配置对象
    """
    return TemporalReasoningConfig()

