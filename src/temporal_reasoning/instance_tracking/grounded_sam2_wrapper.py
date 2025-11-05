# -*- coding: utf-8 -*-
"""
Grounded-SAM-2 封装器
用于封装 Grounding DINO 和 SAM2 的集成调用
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image
from torchvision.ops import box_convert
import warnings

warnings.filterwarnings("ignore")

# 导入 Grounding DINO
HAS_GROUNDING_DINO = False
load_model = None
load_image = None
predict = None

# 导入 SAM2
HAS_SAM2 = False
build_sam2 = None
build_sam2_video_predictor = None
SAM2ImagePredictor = None
SAM2VideoPredictor = None

try:
    # 添加项目路径
    base_dir = Path(__file__).parent.parent.parent.parent
    grounded_sam2_path = base_dir / "third_party" / "Grounded-SAM-2"
    
    if str(grounded_sam2_path) not in sys.path:
        sys.path.insert(0, str(grounded_sam2_path))
    
    # 导入 Grounding DINO
    try:
        from grounding_dino.groundingdino.util.inference import (
            load_model,
            load_image,
            predict
        )
        HAS_GROUNDING_DINO = True
    except ImportError as e:
        print(f"警告: 无法导入 Grounding DINO: {e}")
        HAS_GROUNDING_DINO = False
    
    # 导入 SAM2
    try:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        HAS_SAM2 = True
    except ImportError as e:
        print(f"警告: 无法导入 SAM2: {e}")
        HAS_SAM2 = False
        
except Exception as e:
    print(f"警告: Grounded-SAM-2 模块导入失败: {e}")
    HAS_GROUNDING_DINO = False
    HAS_SAM2 = False


def preprocess_caption(caption: str) -> str:
    """预处理文本提示：小写并添加句号"""
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


class GroundedSAM2Wrapper:
    """Grounded-SAM-2 封装器"""
    
    def __init__(
        self,
        gdino_config_path: str,
        gdino_checkpoint_path: str,
        sam2_config_path: str,
        sam2_checkpoint_path: str,
        device: str = "cuda:0",
        text_threshold: float = 0.25,
        box_threshold: float = 0.3
    ):
        """
        初始化 Grounded-SAM-2 封装器
        
        Args:
            gdino_config_path: Grounding DINO 配置文件路径
            gdino_checkpoint_path: Grounding DINO 模型权重路径
            sam2_config_path: SAM2 配置文件路径
            sam2_checkpoint_path: SAM2 模型权重路径
            device: 计算设备
            text_threshold: 文本阈值
            box_threshold: 框阈值
        """
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        
        self.grounding_dino_model = None
        self.sam2_image_predictor = None
        self.sam2_video_predictor = None
        
        # 初始化模型
        self._initialize_models(
            gdino_config_path,
            gdino_checkpoint_path,
            sam2_config_path,
            sam2_checkpoint_path
        )
    
    def _initialize_models(
        self,
        gdino_config_path: str,
        gdino_checkpoint_path: str,
        sam2_config_path: str,
        sam2_checkpoint_path: str
    ):
        """初始化模型"""
        if not HAS_GROUNDING_DINO:
            raise ImportError(
                "无法导入 Grounding DINO 模块\n"
                "请确保 third_party/Grounded-SAM-2 目录存在且可访问"
            )
        
        if not HAS_SAM2:
            raise ImportError(
                "无法导入 SAM2 模块\n"
                "请确保 third_party/Grounded-SAM-2 目录存在且可访问"
            )
        
        # 初始化 Grounding DINO
        try:
            print("正在初始化 Grounding DINO...")
            self.grounding_dino_model = load_model(
                model_config_path=gdino_config_path,
                model_checkpoint_path=gdino_checkpoint_path,
                device=self.device
            )
            print("Grounding DINO 初始化成功")
        except Exception as e:
            raise RuntimeError(f"Grounding DINO 初始化失败: {e}")
        
        # 初始化 SAM2
        try:
            print("正在初始化 SAM2...")
            sam2_model = build_sam2(
                model_cfg=sam2_config_path,
                sam2_checkpoint=sam2_checkpoint_path,
                device=self.device
            )
            self.sam2_image_predictor = SAM2ImagePredictor(sam2_model)
            
            # 如果需要视频追踪，也初始化视频预测器
            self.sam2_video_predictor = build_sam2_video_predictor(
                model_cfg=sam2_config_path,
                sam2_checkpoint=sam2_checkpoint_path
            )
            print("SAM2 初始化成功")
        except Exception as e:
            raise RuntimeError(f"SAM2 初始化失败: {e}")
    
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
        if self.grounding_dino_model is None or self.sam2_image_predictor is None:
            raise RuntimeError("模型未初始化")
        
        # 合并文本提示
        text = ". ".join(text_prompts) if len(text_prompts) > 1 else text_prompts[0]
        text = preprocess_caption(text)
        
        # 转换图像格式
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # 加载图像（Grounding DINO 格式）
        image_source, image_tensor = load_image_from_array(image_pil)
        
        # 设置 SAM2 图像
        self.sam2_image_predictor.set_image(image_source)
        
        # 使用 Grounding DINO 检测
        boxes, confidences, labels = predict(
            model=self.grounding_dino_model,
            image=image_tensor,
            caption=text,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            return []
        
        # 转换框格式
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # 使用 SAM2 分割
        masks, scores, logits = self.sam2_image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # 处理掩码形状
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # (n, H, W)
        
        # 转换为 numpy 数组
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        # 返回结果
        results = []
        for i in range(len(masks)):
            mask = masks[i].astype(bool)
            confidence = float(confidences[i].item() if hasattr(confidences[i], 'item') else confidences[i])
            results.append((mask, confidence))
        
        return results
    
    def track_instances(
        self,
        video_frames: List[np.ndarray],
        initial_mask: np.ndarray,
        query_frame: int = 0
    ) -> Dict[int, Dict]:
        """
        追踪实例（使用 SAM2 视频预测器）
        
        Args:
            video_frames: 视频帧序列，每帧为 (H, W, 3) RGB
            initial_mask: 初始帧的掩码 (H, W)
            query_frame: 查询帧索引
        
        Returns:
            追踪结果字典
        """
        if self.sam2_video_predictor is None:
            raise RuntimeError("SAM2 视频预测器未初始化")
        
        # 这里需要实现视频追踪逻辑
        # 由于 SAM2 视频追踪需要特定的帧格式，这里提供基本框架
        # 实际实现需要根据 SAM2 视频预测器的 API 进行调整
        
        # TODO: 实现视频追踪逻辑
        # 1. 准备视频帧（转换为 SAM2 需要的格式）
        # 2. 初始化视频预测器状态
        # 3. 使用初始掩码进行追踪
        # 4. 返回追踪结果
        
        return {}


def load_image_from_array(image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    """
    从 PIL Image 加载图像（Grounding DINO 格式）
    
    Args:
        image: 输入图像 PIL Image 格式
    
    Returns:
        (image_source, image_tensor):
        - image_source: 原始图像数组 (H, W, 3)
        - image_tensor: 预处理后的张量
    """
    import grounding_dino.groundingdino.datasets.transforms as T
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 确保图像是 RGB 格式
    image_pil = image.convert("RGB")
    image_source = np.asarray(image_pil)
    image_tensor, _ = transform(image_pil, None)
    
    return image_source, image_tensor

