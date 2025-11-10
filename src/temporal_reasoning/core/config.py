# -*- coding: utf-8 -*-
"""
鏃跺簭鍚堢悊鎬у垎鏋愭ā鍧楅厤缃锟�?
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class RAFTConfig:
    """RAFT鍏夋祦閰嶇疆"""
    model_path: str = ""
    model_type: str = "large"  # large or small
    use_gpu: bool = True
    batch_size: int = 1
    motion_discontinuity_threshold: float = 0.3  # 杩愬姩绐佸彉闃堬�??


@dataclass
class GroundingDINOConfig:
    """Grounding DINO閰嶇�?"""
    model_path: str = ""  # 鏉冮噸鏂囦欢璺�?
    config_path: str = ""  # 閰嶇疆鏂囦欢璺�?
    bert_path: str = ""  # BERT妯″瀷璺緞
    text_threshold: float = 0.25
    box_threshold: float = 0.3
    use_gpu: bool = True


@dataclass
class SAMConfig:
    """SAM閰嶇�?"""
    model_path: str = ""  # 鏉冮噸鏂囦欢璺�?
    config_path: str = ""  # 閰嶇疆鏂囦欢璺緞锛圫AM2闇€瑕侊�?
    model_type: str = "sam2_h"  # sam2_h, sam2_l, sam2_b
    use_gpu: bool = True


@dataclass
class TrackerConfig:
    """杩借釜鍣ㄩ厤锟�??"""
    type: str = "deaot"  # deaot or cotracker
    model_path: Optional[str] = None
    use_gpu: bool = True
    enable_cotracker_validation: bool = True  # 鏄惁鍚敤Co-Tracker楠岃�?
    cotracker_checkpoint: Optional[str] = None  # Co-Tracker妯″瀷璺緞
    grid_size: int = 30  # Co-Tracker缃戞牸澶у�?


@dataclass
class KeypointConfig:
    """鍏抽敭鐐归厤锟�?"""
    model_type: str = "mediapipe"  # mediapipe or mmpose
    model_path: Optional[str] = None
    use_gpu: bool = False  # MediaPipe锟斤拷支锟斤拷GPU
    
    # 锟斤拷锟接伙拷锟斤拷锟斤�?
    enable_visualization: bool = False  # 锟角凤拷锟斤拷锟矫匡拷锟接伙�?
    visualization_output_dir: Optional[str] = None  # 锟斤拷锟接伙拷锟斤拷锟侥柯�
    show_face: bool = False  # 锟角凤拷锟斤拷示锟芥部锟截硷拷锟斤拷
    show_face_mesh: bool = False  # 锟角凤拷锟斤拷示锟芥部锟斤拷锟斤�?
    point_radius: int = 3  # 锟截硷拷锟斤拷刖�?
    line_thickness: int = 2  # 锟斤拷锟斤拷锟竭达拷�?
    save_visualization: bool = True  # 锟角否保达拷锟斤拷踊锟斤拷锟斤拷
    show_visualization: bool = False  # 锟角凤拷锟斤拷示锟斤拷锟接伙拷锟斤拷锟斤拷锟紾UI锟斤�?


@dataclass
class FusionConfig:
    """铻嶅悎閰嶇疆"""
    multimodal_confidence_boost: float = 1.2
    min_anomaly_duration_frames: int = 3
    single_modality_confidence_threshold: float = 0.8


@dataclass
class ThresholdsConfig:
    """闃堝€奸厤锟�??"""
    motion_discontinuity_threshold: float = 0.3
    structure_disappearance_threshold: float = 0.3  # 鎺╃爜闈㈢Н鍙樺寲锟�?
    keypoint_displacement_threshold: int = 10  # 鍍忕�?


@dataclass
class TemporalReasoningConfig:
    """鏃跺簭鍚堢悊鎬у垎鏋愰厤锟�??"""
    device: str = "cuda:0"
    
    # 瀛愭ā鍧楅厤锟�??
    raft: RAFTConfig = field(default_factory=RAFTConfig)
    grounding_dino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    structure_prompts: Optional[List[str]] = None
    
    # 杈撳嚭閰嶇疆
    output_dir: str = ""
    save_visualizations: bool = True
    save_detailed_reports: bool = True
    
    def __post_init__(self):
        """鍒濆鍖栭粯璁ら厤锟�??"""
        # 璁剧疆榛樿妯″瀷璺緞锛堝熀浜庨」鐩粨鏋勶�?
        base_dir = Path(__file__).parent.parent.parent.parent
        third_party_dir = base_dir / "third_party"
        cache_dir = base_dir / ".cache"
        
        # RAFT榛樿璺緞锛堟潈閲嶅湪.cache锛屼唬鐮佸湪third_party锟�?
        if not self.raft.model_path:
            # 锟�?.cache鏌ユ壘RAFT鏉冮�?
            raft_cache_path = cache_dir / "raft-things.pth"
            if raft_cache_path.exists():
                self.raft.model_path = str(raft_cache_path)
            else:
                raise FileNotFoundError(
                    f"RAFT妯″瀷鏂囦欢鏈壘锟�??: {raft_cache_path}\n"
                    f"璇风‘淇濇潈閲嶆枃浠跺瓨鍦ㄤ�? .cache 鐩綍锟�??"
                )
        
        # Grounding DINO榛樿璺緞锛堟潈閲嶅湪.cache锛屼唬鐮佸湪third_party锟�?
        if not self.grounding_dino.model_path:
            # Grounding DINO鏉冮噸锟�??.cache
            gdino_weight = cache_dir / "groundingdino_swinb_cogcoor.pth"
            if gdino_weight.exists():
                self.grounding_dino.model_path = str(gdino_weight)
            else:
                raise FileNotFoundError(
                    f"Grounding DINO妯″瀷鏂囦欢鏈壘锟�??: {gdino_weight}\n"
                    f"璇风‘淇濇潈閲嶆枃浠跺瓨鍦ㄤ�? .cache 鐩綍锟�??"
                )
        
        # Grounding DINO閰嶇疆鏂囦欢璺�?
        if not self.grounding_dino.config_path:
            gdino_config = third_party_dir / "Grounded-SAM-2" / "grounding_dino" / "config" / "GroundingDINO_SwinB.py"
            if not gdino_config.exists():
                # 灏濊瘯鍙︿竴涓矾锟�??
                gdino_config = third_party_dir / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB.py"
            if gdino_config.exists():
                self.grounding_dino.config_path = str(gdino_config)
            else:
                raise FileNotFoundError(
                    f"Grounding DINO閰嶇疆鏂囦欢鏈壘锟�??: {gdino_config}\n"
                    f"璇风‘淇濋厤缃枃浠跺瓨鍦ㄤ�? third_party 鐩綍锟�??"
                )
        
        # BERT妯″瀷璺緞
        if not self.grounding_dino.bert_path:
            bert_path = cache_dir / "google-bert" / "bert-base-uncased"
            if bert_path.exists():
                self.grounding_dino.bert_path = str(bert_path)
            else:
                raise FileNotFoundError(
                    f"BERT妯″瀷鏂囦欢鏈壘锟�??: {bert_path}\n"
                    f"璇风‘淇滲ERT妯″瀷瀛樺湪锟�?? .cache/google-bert/bert-base-uncased 鐩綍锟�??"
                )
        
        # SAM榛樿璺緞锛堟潈閲嶅湪.cache锛屼唬鐮佸湪third_party锟�?
        if not self.sam.model_path:
            # SAM2鏉冮噸锟�??.cache
            sam2_weight = cache_dir / "sam2.1_hiera_large.pt"
            if sam2_weight.exists():
                self.sam.model_path = str(sam2_weight)
            else:
                # 濡傛�?.cache涓病鏈塖AM2锛屽皾璇曟棫鐗圫AM鏉冮�?
                sam_weight = cache_dir / "sam_vit_h_4b8939.pth"
                if sam_weight.exists():
                    self.sam.model_path = str(sam_weight)
                else:
                    raise FileNotFoundError(
                        f"SAM妯″瀷鏂囦欢鏈壘锟�??: {sam2_weight} 锟�? {sam_weight}\n"
                        f"璇风‘淇濇潈閲嶆枃浠跺瓨鍦ㄤ�? .cache 鐩綍锟�??"
                    )
        
        # SAM2閰嶇疆鏂囦欢璺�?
        if not self.sam.config_path and self.sam.model_type.startswith("sam2"):
            sam2_config = third_party_dir / "Grounded-SAM-2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
            if not sam2_config.exists():
                # 灏濊瘯鍏朵粬璺�?
                sam2_config = third_party_dir / "Grounded-SAM-2" / "sam2" / "configs" / "sam2_hiera_l.yaml"
            if sam2_config.exists():
                self.sam.config_path = str(sam2_config)
            else:
                raise FileNotFoundError(
                    f"SAM2閰嶇疆鏂囦欢鏈壘锟�??: {sam2_config}\n"
                    f"璇风‘淇濋厤缃枃浠跺瓨鍦ㄤ�? third_party/Grounded-SAM-2/sam2/configs 鐩綍锟�??"
                )
        
        # Co-Tracker榛樿璺緞锛堟潈閲嶅湪.cache锛屼唬鐮佸湪third_party锟�?
        if not self.tracker.cotracker_checkpoint and self.tracker.enable_cotracker_validation:
            cotracker_weight = cache_dir / "scaled_offline.pth"
            if cotracker_weight.exists():
                self.tracker.cotracker_checkpoint = str(cotracker_weight)
            else:
                raise FileNotFoundError(
                    f"Co-Tracker妯″瀷鏂囦欢鏈壘锟�??: {cotracker_weight}\n"
                    f"璇风‘淇濇潈閲嶆枃浠跺瓨鍦ㄤ�? .cache 鐩綍涓紝鎴栬锟�?? tracker.enable_cotracker_validation=False 绂佺敤楠岃瘉"
                )
        
        # 杈撳嚭鐩綍
        if not self.output_dir:
            self.output_dir = str(base_dir / "outputs" / "temporal_reasoning")
        
        # 纭繚杈撳嚭鐩綍瀛樺�?
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """
        鑾峰彇妯″瀷璺緞
        
        Args:
            model_name: 妯″瀷鍚嶇�?
        
        Returns:
            妯″瀷璺緞
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
        浠庡瓧鍏告洿鏂伴厤锟�??
        
        Args:
            config_dict: 閰嶇疆瀛楀吀
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
    浠嶻AML鏂囦欢鍔犺浇閰嶇�?
    
    Args:
        config_path: YAML閰嶇疆鏂囦欢璺�?
    
    Returns:
        閰嶇疆瀵硅�?
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
        raise ImportError("璇峰畨瑁卲yyaml: pip install pyyaml")
    except Exception as e:
        raise ValueError(f"鍔犺浇閰嶇疆鏂囦欢澶辫触: {e}")


def get_default_config() -> TemporalReasoningConfig:
    """
    鑾峰彇榛樿閰嶇�?
    
    Returns:
        榛樿閰嶇疆瀵硅�?
    """
    return TemporalReasoningConfig()

