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
    # __file__ 路径: src/temporal_reasoning/instance_tracking/grounded_sam_wrapper.py
    # 需要计算: 项目根目录/src/aux_motion_intensity_2
    base_dir = Path(__file__).parent.parent.parent.parent
    src_path = base_dir / "src"
    
    # 将 src 目录添加到路径，以便可以导入 aux_motion_intensity_2 包
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # 使用包导入方式，这样可以正确处理相对导入
    try:
        # 尝试作为包导入（推荐方式）
        import aux_motion_intensity_2.analyzer as analyzer_module
        PASAnalyzer = analyzer_module.PASAnalyzer
        get_grounding_output = analyzer_module.get_grounding_output
        HAS_AUX_MOTION = True
    except ImportError:
        # 如果包导入失败，尝试直接导入模块（需要处理相对导入）
        aux_motion_path = src_path / "aux_motion_intensity_2"
        if str(aux_motion_path) not in sys.path:
            sys.path.insert(0, str(aux_motion_path))
        
        # 动态导入
        import importlib.util
        analyzer_file = aux_motion_path / "analyzer.py"
        if analyzer_file.exists():
            # 使用包的完整名称，以便相对导入可以工作
            spec = importlib.util.spec_from_file_location(
                "aux_motion_intensity_2.analyzer",
                str(analyzer_file)
            )
            if spec and spec.loader:
                analyzer_module = importlib.util.module_from_spec(spec)
                # 设置 __package__ 以便相对导入可以工作
                analyzer_module.__package__ = "aux_motion_intensity_2"
                # 设置 __file__ 和 __name__
                analyzer_module.__file__ = str(analyzer_file)
                analyzer_module.__name__ = "aux_motion_intensity_2.analyzer"
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
        
        # 确保第三方库路径已设置（PASAnalyzer 需要这些路径）
        project_root = Path(__file__).parent.parent.parent.parent
        gsa_path = project_root / "third_party" / "Grounded-Segment-Anything"
        gdn_path = gsa_path / "GroundingDINO"
        sam_path = gsa_path / "segment_anything"
        cotracker_path = project_root / "third_party" / "co-tracker"
        
        for path in [gsa_path, gdn_path, sam_path, cotracker_path]:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        
        # 检查并修正 SAM 模型路径
        # PASAnalyzer 使用 SAM v1，但配置可能传入 SAM2 路径
        sam_checkpoint_path_resolved = Path(sam_checkpoint_path)
        
        # 如果传入的是 SAM2 文件，但 PASAnalyzer 需要 SAM v1，自动查找 SAM v1 文件
        if sam_checkpoint_path_resolved.suffix == '.pt' or 'sam2' in sam_checkpoint_path_resolved.name.lower():
            # 这是 SAM2 文件，但 PASAnalyzer 需要 SAM v1
            sam_v1_path = project_root / ".cache" / "sam_vit_h_4b8939.pth"
            
            if sam_v1_path.exists():
                print(f"警告: 检测到 SAM2 模型文件 ({sam_checkpoint_path})")
                print(f"PASAnalyzer 需要 SAM v1，自动切换到: {sam_v1_path}")
                sam_checkpoint_path = str(sam_v1_path)
            else:
                raise FileNotFoundError(
                    f"SAM v1 模型文件未找到: {sam_v1_path}\n"
                    f"PASAnalyzer 需要 SAM v1 (sam_vit_h_4b8939.pth)，而不是 SAM2\n"
                    f"当前传入的文件: {sam_checkpoint_path}\n"
                    f"请下载 SAM v1 模型文件到 .cache 目录"
                )
        
        # 初始化 PASAnalyzer
        try:
            # 如果不需要 Co-Tracker，可以传入一个临时路径（但不会被使用）
            # 或者修改 _load_models() 以跳过 Co-Tracker
            cotracker_checkpoint = str(project_root / ".cache" / "scaled_offline.pth")
            
            self.pas_analyzer = PASAnalyzer(
                device=self.device,
                grid_size=self.grid_size,
                enable_scene_classification=False,  # 不需要场景分类
                grounded_checkpoint=gdino_checkpoint_path,
                sam_checkpoint=sam_checkpoint_path,  # 使用修正后的路径
                cotracker_checkpoint=cotracker_checkpoint  # 传入路径（即使不使用）
            )
            
            # 延迟加载模型（只加载 Grounding DINO 和 SAM）
            # 注意：这会尝试加载 Co-Tracker，但我们可以忽略错误
            try:
                self.pas_analyzer._load_models()
            except Exception as cotracker_error:
                # 如果 Co-Tracker 加载失败，尝试只加载 Grounding DINO 和 SAM
                print(f"警告: Co-Tracker 加载失败（可忽略）: {cotracker_error}")
                # 手动加载 Grounding DINO 和 SAM
                self._load_grounding_dino_and_sam()
            
            print("Grounded-SAM 封装器初始化成功")
        except Exception as e:
            error_msg = (
                f"Grounded-SAM 封装器初始化失败: {e}\n"
                f"请检查：\n"
                f"1. 模型文件是否存在: {gdino_checkpoint_path}, {sam_checkpoint_path}\n"
                f"2. 第三方库路径是否正确: {gsa_path}\n"
                f"3. 设备是否可用: {self.device}"
            )
            raise RuntimeError(error_msg)
    
    def _load_grounding_dino_and_sam(self):
        """手动加载 Grounding DINO 和 SAM（跳过 Co-Tracker）"""
        from groundingdino.util.slconfig import SLConfig  # type: ignore
        from groundingdino.models import build_model  # type: ignore
        from groundingdino.util.utils import clean_state_dict  # type: ignore
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        
        # 加载 Grounding DINO
        args = SLConfig.fromfile(self.pas_analyzer.config_file)
        args.device = self.device
        args.bert_base_uncased_path = self.pas_analyzer.bert_base_uncased_path
        self.pas_analyzer.grounding_model = build_model(args)
        checkpoint = torch.load(self.pas_analyzer.grounded_checkpoint, map_location="cpu")
        self.pas_analyzer.grounding_model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        self.pas_analyzer.grounding_model.eval()
        
        # 加载 SAM
        sam_version = "vit_h"
        self.pas_analyzer.sam_predictor = SamPredictor(
            sam_model_registry[sam_version](checkpoint=self.pas_analyzer.sam_checkpoint).to(self.device)
        )
        
        # 标记模型已加载（但不包括 Co-Tracker）
        self.pas_analyzer._models_loaded = True
        print("Grounding DINO 和 SAM 加载成功（跳过 Co-Tracker）")
    
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

