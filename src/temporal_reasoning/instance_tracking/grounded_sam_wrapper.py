# -*- coding: utf-8 -*-
"""
Grounded-SAM 封装器
复用 aux_motion_intensity_2 中的实现
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# 导入 aux_motion_intensity_2 的实现
HAS_AUX_MOTION = False
PASAnalyzer = None
get_grounding_output = None

try:
    # 添加项目路径
    base_dir = Path(__file__).parent.parent.parent.parent
    aux_motion_path = base_dir / "src" / "aux_motion_intensity_2"
    
    if str(aux_motion_path) not in sys.path:
        sys.path.insert(0, str(aux_motion_path))
    
    # 动态导入
    import importlib.util
    analyzer_file = aux_motion_path / "analyzer.py"
    if analyzer_file.exists():
        spec = importlib.util.spec_from_file_location(
            "aux_motion_analyzer",
            str(analyzer_file)
        )
        if spec and spec.loader:
            analyzer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analyzer_module)
            PASAnalyzer = analyzer_module.PASAnalyzer
            get_grounding_output = analyzer_module.get_grounding_output
            HAS_AUX_MOTION = True
        else:
            HAS_AUX_MOTION = False
    else:
        HAS_AUX_MOTION = False
except (ImportError, FileNotFoundError, AttributeError) as e:
    HAS_AUX_MOTION = False
    PASAnalyzer = None
    get_grounding_output = None
    print(f"警告: 无法导入 aux_motion_intensity_2 模块: {e}")


class GroundedSAMWrapper:
    """Grounded-SAM 封装器，复用 aux_motion_intensity_2 的实现"""
    
    def __init__(
        self,
        gdino_config_path: str,
        gdino_checkpoint_path: str,
        sam_checkpoint_path: str,
        device: str = "cuda:0",
        text_threshold: float = 0.25,
        box_threshold: float = 0.3,
        grid_size: int = 30
    ):
        """
        初始化 Grounded-SAM 封装器
        
        Args:
            gdino_config_path: Grounding DINO 配置文件路径
            gdino_checkpoint_path: Grounding DINO 模型权重路径
            sam_checkpoint_path: SAM 模型权重路径
            device: 计算设备
            text_threshold: 文本阈值
            box_threshold: 框阈值
            grid_size: Co-Tracker 网格大小
        """
        if not HAS_AUX_MOTION or PASAnalyzer is None:
            raise ImportError(
                "无法导入 aux_motion_intensity_2 模块\n"
                "请确保 src/aux_motion_intensity_2 目录存在且可访问"
            )
        
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        self.grid_size = grid_size
        
        # 初始化 PASAnalyzer
        try:
            self.pas_analyzer = PASAnalyzer(
                device=self.device,
                grid_size=self.grid_size,
                enable_scene_classification=False,  # 不需要场景分类
                grounded_checkpoint=gdino_checkpoint_path,
                sam_checkpoint=sam_checkpoint_path,
                cotracker_checkpoint=None  # Co-Tracker 由外部管理
            )
            # 延迟加载模型
            self.pas_analyzer._load_models()
            print("Grounded-SAM 封装器初始化成功")
        except Exception as e:
            raise RuntimeError(f"Grounded-SAM 封装器初始化失败: {e}")
    
    def detect_and_segment(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        检测和分割实例
        
        Args:
            image: 输入图像 (H, W, 3) RGB，范围 [0, 255]
            text_prompts: 文本提示列表
        
        Returns:
            掩码列表，每个元素为 (mask, confidence) 元组
        """
        if self.pas_analyzer.grounding_model is None or self.pas_analyzer.sam_predictor is None:
            raise RuntimeError("模型未初始化")
        
        # 合并文本提示
        text = ". ".join(text_prompts) if len(text_prompts) > 1 else text_prompts[0]
        
        # 转换图像格式
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # 准备图像（Grounding DINO 格式）
        import groundingdino.datasets.transforms as T  # type: ignore
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        image_array = np.asarray(image_pil.convert("RGB"))
        
        # 使用 Grounding DINO 检测
        boxes_filt, pred_phrases = get_grounding_output(
            self.pas_analyzer.grounding_model,
            image_tensor,
            text,
            self.box_threshold,
            self.text_threshold,
            device=self.device
        )
        
        if boxes_filt.shape[0] == 0:
            return []
        
        # 设置 SAM 图像
        self.pas_analyzer.sam_predictor.set_image(image_array)
        
        # 转换框格式
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.pas_analyzer.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_array.shape[:2]
        ).to(self.device)
        
        # 使用 SAM 分割
        masks, scores, _ = self.pas_analyzer.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        
        # 处理掩码形状
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # (n, H, W)
        
        # 转换为 numpy 数组
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # 返回结果
        results = []
        for i in range(len(masks)):
            mask = masks[i].astype(bool)
            confidence = float(scores[i][0] if len(scores[i]) > 0 else 0.5)
            results.append((mask, confidence))
        
        return results
    
    def get_models(self):
        """
        获取底层模型（用于外部访问）
        
        Returns:
            (grounding_model, sam_predictor): 底层模型
        """
        return self.pas_analyzer.grounding_model, self.pas_analyzer.sam_predictor

