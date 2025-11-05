# -*- coding: utf-8 -*-
"""
RAFT光流模型封装
复用 aux_motion_intensity 模块中的 SimpleRAFT 实现
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 导入 aux_motion_intensity 模块中的光流计算器
HAS_AUX_MOTION = False
SimpleRAFT = None

try:
    # 添加项目路径
    base_dir = Path(__file__).parent.parent.parent.parent
    aux_motion_path = base_dir / "src" / "aux_motion_intensity"
    if str(aux_motion_path) not in sys.path:
        sys.path.insert(0, str(aux_motion_path))
    
    # 动态导入，避免静态分析警告
    import importlib.util
    flow_predictor_file = aux_motion_path / "flow_predictor.py"
    if flow_predictor_file.exists():
        spec = importlib.util.spec_from_file_location(
            "flow_predictor",
            str(flow_predictor_file)
        )
        if spec and spec.loader:
            flow_predictor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(flow_predictor_module)
            SimpleRAFT = flow_predictor_module.SimpleRAFT
            HAS_AUX_MOTION = True
        else:
            HAS_AUX_MOTION = False
            SimpleRAFT = None
    else:
        HAS_AUX_MOTION = False
        SimpleRAFT = None
except (ImportError, FileNotFoundError, AttributeError) as e:
    HAS_AUX_MOTION = False
    SimpleRAFT = None
    print(f"警告: 无法导入 aux_motion_intensity 模块，使用简化实现: {e}")


class RAFTWrapper:
    """
    RAFT光流模型封装
    复用 aux_motion_intensity 模块中的 SimpleRAFT 实现
    """
    
    def __init__(
        self,
        model_path: str = "",
        model_type: str = "large",
        device: str = "cuda:0"
    ):
        """
        初始化RAFT模型
        
        Args:
            model_path: 模型路径，如果为空则尝试从项目路径加载
            model_type: 模型类型 ("large" or "small")，影响是否使用RAFT
            device: 计算设备
        """
        self.device = device if "cuda" in device else "cpu"
        self.model_type = model_type
        self.model_path = model_path
        
        # 如果模型路径为空，从项目路径加载
        if not self.model_path:
            base_dir = Path(__file__).parent.parent.parent.parent
            # 从.cache目录查找
            cache_dir = base_dir / ".cache"
            raft_cache_path = cache_dir / "raft-things.pth"
            if raft_cache_path.exists():
                self.model_path = str(raft_cache_path)
            else:
                raise FileNotFoundError(
                    f"RAFT模型文件未找到: {raft_cache_path}\n"
                    f"请确保权重文件存在于 .cache 目录中"
                )
        
        # 选择方法：如果有模型路径且为large，使用raft；否则使用farneback
        if self.model_path and model_type == "large" and HAS_AUX_MOTION:
            method = "raft"
        else:
            method = "farneback"
        
        # 初始化 SimpleRAFT
        if not HAS_AUX_MOTION or SimpleRAFT is None:
            raise ImportError(
                "无法导入 aux_motion_intensity 模块中的 SimpleRAFT\n"
                "请确保 aux_motion_intensity 模块存在且可访问"
            )
        
        try:
            self.raft_model = SimpleRAFT(
                device=self.device,
                method=method,
                model_path=self.model_path if self.model_path else None
            )
            print(f"使用 SimpleRAFT (方法: {self.raft_model.method})")
        except Exception as e:
            raise RuntimeError(
                f"SimpleRAFT初始化失败: {e}\n"
                f"模型路径: {self.model_path}\n"
                f"设备: {self.device}\n"
                f"方法: {method}"
            )
    
    def compute_flow(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算光流
        
        Args:
            image1: 第一帧图像 (H, W, 3) RGB，范围[0, 255]
            image2: 第二帧图像 (H, W, 3) RGB，范围[0, 255]
        
        Returns:
            (u, v): 光流场，u和v分别为x和y方向的光流 (H, W)
        """
        if self.raft_model is not None:
            # 使用 SimpleRAFT 计算光流
            # SimpleRAFT 返回的格式是 (2, H, W)
            flow = self.raft_model.predict_flow(image1, image2)
            
            # 转换为 (H, W) 格式
            if flow.shape[0] == 2:
                u = flow[0]  # x方向
                v = flow[1]  # y方向
            else:
                # 如果已经是 (H, W, 2) 格式
                u = flow[:, :, 0]
                v = flow[:, :, 1]
            
            return u, v
        else:
            raise RuntimeError(
                "RAFT模型未初始化\n"
                "请检查模型路径是否正确，以及 aux_motion_intensity 模块是否可用"
            )
    

