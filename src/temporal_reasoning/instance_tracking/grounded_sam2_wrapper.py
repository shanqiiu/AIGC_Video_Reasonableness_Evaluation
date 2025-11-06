# -*- coding: utf-8 -*-
"""
Grounded-SAM-2 封装�?
用于封装 Grounding DINO �? SAM2 的集成调�?
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
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
        # 添加 grounding_dino 路径
        grounding_dino_path = grounded_sam2_path / "grounding_dino"
        if str(grounding_dino_path) not in sys.path:
            sys.path.insert(0, str(grounding_dino_path))
        
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
        # 确保导入 sam2 模块以初始化 Hydra

        import sam2  # 这会触发 sam2/__init__.py 中的 Hydra 初始�?
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
    """预处理文本提示：小写并添加句�?"""
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


class GroundedSAM2Wrapper:
    """Grounded-SAM-2 封装�?"""
    
    def __init__(
        self,
        gdino_config_path: str,
        gdino_checkpoint_path: str,
        sam2_config_path: str,
        sam2_checkpoint_path: str,
        device: str = "cuda:0",
        text_threshold: float = 0.25,
        box_threshold: float = 0.3,
        bert_path: Optional[str] = None
    ):
        """
        初始�? Grounded-SAM-2 封装�?
        
        Args:
            gdino_config_path: Grounding DINO 配置文件路径
            gdino_checkpoint_path: Grounding DINO 模型权重路径
            sam2_config_path: SAM2 配置文件路径
            sam2_checkpoint_path: SAM2 模型权重路径
            device: 计算设备
            text_threshold: 文本阈�?
            box_threshold: 框阈�?
            bert_path: BERT 模型本地路径（如果为 None，会尝试�? Hugging Face Hub 下载�?
        """
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        self.bert_path = bert_path
        
        self.grounding_dino_model = None
        self.sam2_image_predictor = None
        self.sam2_video_predictor = None
        
        # 初始化模�?
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
        """初始化模�?"""
        if not HAS_GROUNDING_DINO:
            raise ImportError(
                "无法导入 Grounding DINO 模块\n"
                "请确�? third_party/Grounded-SAM-2 目录存在且可访问"
            )
        
        if not HAS_SAM2:
            raise ImportError(
                "无法导入 SAM2 模块\n"
                "请确�? third_party/Grounded-SAM-2 目录存在且可访问"
            )
        
        # 初始�? Grounding DINO
        try:
            print("正在初始�? Grounding DINO...")
            
            # 加载配置文件
            from grounding_dino.groundingdino.util.slconfig import SLConfig
            from grounding_dino.groundingdino.models import build_model
            from grounding_dino.groundingdino.util.misc import clean_state_dict
            import os
            
            args = SLConfig.fromfile(gdino_config_path)
            args.device = self.device
            
            # 如果提供了本�? BERT 路径，设置它以避免从 Hugging Face Hub 下载
            if self.bert_path:
                # 检查路径是否存�?
                bert_path_obj = Path(self.bert_path)
                if bert_path_obj.exists() and bert_path_obj.is_dir():
                    # �? text_encoder_type 设置为本地路径，这样 from_pretrained 会使用本地模�?
                    args.text_encoder_type = str(bert_path_obj.absolute())
                    print(f"使用本地 BERT 模型: {self.bert_path}")
                    
                    # 设置环境变量，强制使用离线模式，避免尝试下载
                    # 注意：这会影响整�? transformers 库的行为
                    original_hf_offline = os.environ.get("HF_HUB_OFFLINE", None)
                    original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", None)
                    try:
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        os.environ["TRANSFORMERS_OFFLINE"] = "1"
                        
                        # 构建模型（此时会使用本地 BERT�?
                        self.grounding_dino_model = build_model(args)
                    finally:
                        # 恢复环境变量
                        if original_hf_offline is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                        
                        if original_transformers_offline is None:
                            os.environ.pop("TRANSFORMERS_OFFLINE", None)
                        else:
                            os.environ["TRANSFORMERS_OFFLINE"] = original_transformers_offline
                else:
                    print(f"警告: BERT 路径不存�?: {self.bert_path}，将尝试�? Hugging Face Hub 下载")
                    # 构建模型（可能会尝试下载�?
                    self.grounding_dino_model = build_model(args)
            else:
                print("警告: 未提�? BERT 路径，将尝试�? Hugging Face Hub 下载")
                # 构建模型（可能会尝试下载�?
                self.grounding_dino_model = build_model(args)
            
            # 加载权重
            checkpoint = torch.load(gdino_checkpoint_path, map_location="cpu")
            self.grounding_dino_model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False
            )
            self.grounding_dino_model.eval()
            
            print("Grounding DINO 初始化成�?")
        except Exception as e:
            raise RuntimeError(f"Grounding DINO 初始化失�?: {e}")
        
        # 初始�? SAM2
        try:
            print("正在初始�? SAM2...")
            
            # build_sam2 的参数签名：
            # build_sam2(config_file, ckpt_path=None, device="cuda", ...)
            # 注意：config_file 是第一个位置参数，需要使�? Hydra 配置名称
            # Hydra 需要相对于 sam2/configs 的相对路径，例如：sam2.1/sam2.1_hiera_l
            
            # 将配置文件路径转换为 Hydra 配置名称
            sam2_config_path_obj = Path(sam2_config_path)
            
            # 查找 sam2/configs 目录
            base_dir = Path(__file__).parent.parent.parent.parent
            sam2_configs_dir = base_dir / "third_party" / "Grounded-SAM-2" / "sam2" / "configs"
            
            config_name = None
            
            if sam2_config_path_obj.exists():
                # 如果是绝对路径，尝试转换为相对于 sam2/configs 的相对路�?
                try:
                    # 获取相对�? sam2/configs 的相对路�?
                    relative_path = sam2_config_path_obj.relative_to(sam2_configs_dir)
                    # 移除 .yaml 扩展名，转换�? Hydra 配置名称
                    # 例如：sam2.1/sam2.1_hiera_l.yaml -> sam2.1/sam2.1_hiera_l
                    # Windows ·��ʹ�÷�б�ܣ���Ҫת��Ϊ��б�ܣ�Hydra ����������Ҫ��
                    # 需要包含 configs/ 前缀，因为 Hydra 在 sam2 包内查找配置文件
                    config_name = "configs/" + str(relative_path.with_suffix('')).replace('\\', '/')
                    print(f"转换配置文件路径: {sam2_config_path} -> {config_name}")
                except ValueError as ve:
                    # 如果不在 sam2/configs 下，尝试从路径中提取配置名称
                    print(f"路径不在 sam2/configs 下，尝试提取配置名称: {ve}")
                    config_str = str(sam2_config_path_obj).replace('\\', '/')
                    if 'sam2.1' in config_str:
                        if 'sam2.1_hiera_l' in config_str or 'sam2.1_hiera_large' in config_str:
                            config_name = "configs/sam2.1/sam2.1_hiera_l"
                        elif 'sam2.1_hiera_s' in config_str or 'sam2.1_hiera_small' in config_str:
                            config_name = "configs/sam2.1/sam2.1_hiera_s"
                        elif 'sam2.1_hiera_t' in config_str or 'sam2.1_hiera_tiny' in config_str:
                            config_name = "configs/sam2.1/sam2.1_hiera_t"
                        elif 'sam2.1_hiera_b+' in config_str or 'sam2.1_hiera_base_plus' in config_str:
                            config_name = "configs/sam2.1/sam2.1_hiera_b+"
                    elif 'sam2_hiera_l' in config_str or 'sam2_hiera_large' in config_str:
                        config_name = "configs/sam2/sam2_hiera_l"
                    elif 'sam2_hiera_s' in config_str or 'sam2_hiera_small' in config_str:
                        config_name = "configs/sam2/sam2_hiera_s"
                    elif 'sam2_hiera_t' in config_str or 'sam2_hiera_tiny' in config_str:
                        config_name = "configs/sam2/sam2_hiera_t"
                    elif 'sam2_hiera_b+' in config_str or 'sam2_hiera_base_plus' in config_str:
                        config_name = "configs/sam2/sam2_hiera_b+"
                    
                    if config_name is None:
                        # 尝试从文件名提取
                        config_stem = sam2_config_path_obj.stem  # 例如：sam2.1_hiera_l
                        if config_stem.startswith('sam2.1'):
                            config_name = f"configs/sam2.1/{config_stem}"
                        elif config_stem.startswith('sam2'):
                            config_name = f"configs/sam2/{config_stem}"
            else:
                # 如果路径不存在，假设已经�? Hydra 配置名称或相对路�?
                # 例如：sam2.1/sam2.1_hiera_l �? configs/sam2.1/sam2.1_hiera_l.yaml
                config_str = sam2_config_path.replace('\\', '/')
                
                if 'configs/' in config_str:
                    # 提取 configs/ 之后的部分，移除扩展�?
                    parts = config_str.split('configs/')
                    if len(parts) > 1:
                        # 保留 configs/ 前缀
                        config_name = "configs/" + parts[1].replace('.yaml', '').replace('.yml', '')
                    else:
                        config_name = sam2_config_path
                elif config_str.startswith('sam2.1/') or config_str.startswith('sam2/'):
                    # 已经�? Hydra 配置名称格式
                    # 已经是 Hydra 配置名称格式，但缺少 configs/ 前缀，需要添加
                    config_name = "configs/" + config_str.replace('.yaml', '').replace('.yml', '')
                elif '/' in config_str or '\\' in config_str:
                    # 看起来像是路径，尝试提取配置名称
                    # 例如：third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
                    if 'sam2/configs/' in config_str:
                        parts = config_str.split('sam2/configs/')
                        if len(parts) > 1:
                            # 保留 configs/ 前缀
                            config_name = "configs/" + parts[1].replace('.yaml', '').replace('.yml', '')
                        else:
                            # 尝试提取最后的目录和文件名
                            path_parts = config_str.split('/')
                            if 'sam2.1' in path_parts:
                                idx = path_parts.index('sam2.1')
                                if idx + 1 < len(path_parts):
                                    filename = path_parts[idx + 1].replace('.yaml', '').replace('.yml', '')
                                    config_name = f"configs/sam2.1/{filename}"
                            elif 'sam2' in path_parts:
                                idx = path_parts.index('sam2')
                                if idx + 1 < len(path_parts):
                                    filename = path_parts[idx + 1].replace('.yaml', '').replace('.yml', '')
                                    config_name = f"configs/sam2/{filename}"
                            else:
                                config_name = sam2_config_path
                    else:
                        config_name = sam2_config_path
                else:
                    # 可能是配置名称，直接使用
                    config_name = sam2_config_path
            
            if config_name is None:
                raise ValueError(
                    f"无法识别 SAM2 配置名称。\n"
                    f"配置文件路径: {sam2_config_path}\n"
                    f"请确保配置文件路径正确，或使�? Hydra 配置名称格式（如：sam2.1/sam2.1_hiera_l�?"
                )
            
            # ȷ����������ʹ����б�ܣ�Hydra ��Ҫ��Windows ·������ʹ�÷�б�ܣ�
            config_name = config_name.replace('\\', '/')
            
            # 确保配置名称包含 configs/ 前缀（如果还没有）
            if not config_name.startswith('configs/'):
                # 如果配置名称以 sam2.1/ 或 sam2/ 开头，添加 configs/ 前缀
                if config_name.startswith('sam2.1/') or config_name.startswith('sam2/'):
                    config_name = "configs/" + config_name
                else:
                    # 尝试从配置名称推断
                    if 'sam2.1' in config_name:
                        config_name = config_name.replace('sam2.1/', 'configs/sam2.1/')
                    elif 'sam2' in config_name:
                        config_name = config_name.replace('sam2/', 'configs/sam2/')
            
            print(f"ʹ�� SAM2 ����: {config_name}")
            
            # 调用 build_sam2，使用位置参�?
            # build_sam2(config_file, ckpt_path, device, ...)
            print("正在调用 build_sam2...")
            try:
                sam2_model = build_sam2(
                    config_name,  # 位置参数：config_file
                    sam2_checkpoint_path,  # 位置参数：ckpt_path
                    device=self.device  # 关键字参数：device
                )
                print("build_sam2 调用成功")
            except Exception as build_error:
                print(f"build_sam2 调用失败: {type(build_error).__name__}: {build_error}")
                import traceback
                print("完整错误堆栈:")
                traceback.print_exc()
                raise
            
            print("正在创建 SAM2ImagePredictor...")
            try:
                self.sam2_image_predictor = SAM2ImagePredictor(sam2_model)
                print("SAM2ImagePredictor 创建成功")
            except Exception as predictor_error:
                print(f"SAM2ImagePredictor 创建失败: {type(predictor_error).__name__}: {predictor_error}")
                import traceback
                print("完整错误堆栈:")
                traceback.print_exc()
                raise
            
            # 如果需要视频追踪，也初始化视频预测�?
            self.sam2_video_predictor = build_sam2_video_predictor(
                config_name,  # 位置参数：config_file
                sam2_checkpoint_path,  # 位置参数：ckpt_path
                device=self.device  # 关键字参数：device
            )
            print("SAM2 初始化成�?")
        except Exception as e:
            raise RuntimeError(f"SAM2 初始化失�?: {e}")
    
    def detect_and_segment(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        检测和分割实例
        
        Args:
            image: 输入图像 (H, W, 3) RGB，范�? [0, 255]
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
        
        # 加载图像（Grounding DINO 格式�?
        image_source, image_tensor = load_image_from_array(image_pil)
        
        # 设置 SAM2 图像
        self.sam2_image_predictor.set_image(image_source)
        
        # 使用 Grounding DINO 检�?
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
        
        # 转换框格�?
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
        
        # 转换�? numpy 数组
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
        追踪实例（使�? SAM2 视频预测器）
        
        Args:
            video_frames: 视频帧序列，每帧�? (H, W, 3) RGB
            initial_mask: 初始帧的掩码 (H, W)
            query_frame: 查询帧索�?
        
        Returns:
            追踪结果字典
        """
        if self.sam2_video_predictor is None:
            raise RuntimeError("SAM2 视频预测器未初始�?")
        
        # 这里需要实现视频追踪逻辑
        # 由于 SAM2 视频追踪需要特定的帧格式，这里提供基本框架
        # 实际实现需要根�? SAM2 视频预测器的 API 进行调整
        
        # TODO: 实现视频追踪逻辑
        # 1. 准备视频帧（转换�? SAM2 需要的格式�?
        # 2. 初始化视频预测器状�?
        # 3. 使用初始掩码进行追踪
        # 4. 返回追踪结果
        
        return {}


def load_image_from_array(image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    """
    �? PIL Image 加载图像（Grounding DINO 格式�?
    
    Args:
        image: 输入图像 PIL Image 格式
    
    Returns:
        (image_source, image_tensor):
        - image_source: 原始图像数组 (H, W, 3)
        - image_tensor: 预处理后的张�?
    """
    # 确保路径已添�?
    base_dir = Path(__file__).parent.parent.parent.parent
    grounded_sam2_path = base_dir / "third_party" / "Grounded-SAM-2"
    grounding_dino_path = grounded_sam2_path / "grounding_dino"
    if str(grounding_dino_path) not in sys.path:
        sys.path.insert(0, str(grounding_dino_path))
    
    import grounding_dino.groundingdino.datasets.transforms as T
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 确保图像�? RGB 格式
    image_pil = image.convert("RGB")
    image_source = np.asarray(image_pil)
    image_tensor, _ = transform(image_pil, None)
    
    return image_source, image_tensor

