# -*- coding: utf-8 -*-
"""
Co-Tracker验证模块
用于验证对象消失/出现是否合理，过滤假阳性
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# 导入Co-Tracker
HAS_COTRACKER = False
CoTrackerPredictor = None

try:
    base_dir = Path(__file__).parent.parent.parent.parent
    cotracker_path = base_dir / "third_party" / "co-tracker"
    if str(cotracker_path) not in sys.path:
        sys.path.insert(0, str(cotracker_path))
    
    # 尝试导入Co-Tracker
    try:
        from cotracker.predictor import CoTrackerPredictor
        HAS_COTRACKER = True
    except ImportError:
        # 如果导入失败，尝试其他方式
        cotracker_module_path = cotracker_path / "cotracker"
        if cotracker_module_path.exists() and str(cotracker_module_path) not in sys.path:
            sys.path.insert(0, str(cotracker_module_path))
        try:
            from predictor import CoTrackerPredictor
            HAS_COTRACKER = True
        except ImportError:
            HAS_COTRACKER = False
except (ImportError, FileNotFoundError) as e:
    HAS_COTRACKER = False
    CoTrackerPredictor = None
    print(f"警告: 无法导入Co-Tracker，验证功能将不可用: {e}")


class CoTrackerValidator:
    """Co-Tracker验证器，用于验证对象消失/出现是否合理"""
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
        grid_size: int = 30
    ):
        """
        初始化Co-Tracker验证器
        
        Args:
            checkpoint_path: Co-Tracker模型路径
            device: 计算设备
            grid_size: 网格大小
        """
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.grid_size = grid_size
        self.cotracker_model = None
        
        if not HAS_COTRACKER or CoTrackerPredictor is None:
            raise ImportError(
                "无法导入Co-Tracker模块\n"
                "请确保 third_party/co-tracker 目录存在且可访问"
            )
        
        # 查找模型路径（权重在.cache，代码在third_party）
        if checkpoint_path is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            # 从.cache目录查找
            cache_dir = base_dir / ".cache"
            checkpoint_path = str(cache_dir / "scaled_offline.pth")
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(
                    f"Co-Tracker模型文件未找到: {checkpoint_path}\n"
                    f"请确保权重文件存在于 .cache 目录中"
                )
        
        # 验证文件是否存在
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Co-Tracker模型文件不存在: {checkpoint_path}\n"
                f"请检查路径是否正确"
            )
        
        # 初始化Co-Tracker
        try:
            self.cotracker_model = CoTrackerPredictor(
                checkpoint=checkpoint_path,
                v2=False,
                offline=True,
                window_len=60,
            ).to(self.device)
            print(f"Co-Tracker验证器初始化成功: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"Co-Tracker初始化失败: {e}\n"
                f"模型路径: {checkpoint_path}\n"
                f"设备: {self.device}"
            )
    
    def validate_disappearance(
        self,
        video: torch.Tensor,
        mask: np.ndarray,
        query_frame: int,
        video_width: int,
        video_height: int
    ) -> Tuple[bool, dict]:
        """
        验证对象消失是否合理
        
        Args:
            video: 视频tensor (1, T, C, H, W)
            mask: 消失对象的掩码 (H, W)
            query_frame: 查询帧索引
            video_width: 视频宽度
            video_height: 视频高度
        
        Returns:
            (is_valid, validation_info):
            - is_valid: 是否为有效异常（True表示异常，False表示假阳性）
            - validation_info: 验证信息字典
        """
        if self.cotracker_model is None:
            raise RuntimeError(
                "Co-Tracker模型未初始化\n"
                "请检查模型路径是否正确，以及Co-Tracker模块是否可用"
            )
        
        try:
            # 转换掩码格式
            if isinstance(mask, np.ndarray):
                mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                disappear_mask = torch.from_numpy(np.array(mask_image))[None, None].to(self.device)
            else:
                disappear_mask = mask.to(self.device) if isinstance(mask, torch.Tensor) else mask
            
            # 使用Co-Tracker进行反向追踪
            pred_tracks, pred_visibility = self.cotracker_model(
                video,
                grid_size=self.grid_size,
                grid_query_frame=query_frame,
                backward_tracking=True,
                segm_mask=disappear_mask
            )
            
            # 检查各种情况
            edge_vanish = self._is_edge_vanish(
                pred_tracks, pred_visibility, query_frame, video_width, video_height
            )
            
            small_vanish = self._is_small_vanish(
                pred_tracks, pred_visibility, query_frame, video_width, video_height
            )
            
            detect_error = self._is_vanish_detect_error(
                pred_tracks, pred_visibility, query_frame
            )
            
            # 如果满足任何合理情况，认为是假阳性
            is_false_positive = edge_vanish or small_vanish or detect_error
            is_valid = not is_false_positive
            
            validation_info = {
                'edge_vanish': edge_vanish,
                'small_vanish': small_vanish,
                'detect_error': detect_error,
                'is_valid': is_valid,
                'reason': 'edge_vanish' if edge_vanish else 'small_vanish' if small_vanish else 'detect_error' if detect_error else 'valid_anomaly'
            }
            
            return is_valid, validation_info
            
        except Exception as e:
            raise RuntimeError(
                f"Co-Tracker验证失败: {e}\n"
                f"查询帧: {query_frame}\n"
                f"视频尺寸: {video_width}x{video_height}"
            )
    
    def validate_appearance(
        self,
        video: torch.Tensor,
        mask: np.ndarray,
        query_frame: int,
        video_width: int,
        video_height: int
    ) -> Tuple[bool, dict]:
        """
        验证对象出现是否合理
        
        Args:
            video: 视频tensor (1, T, C, H, W)
            mask: 出现对象的掩码 (H, W)
            query_frame: 查询帧索引
            video_width: 视频宽度
            video_height: 视频高度
        
        Returns:
            (is_valid, validation_info):
            - is_valid: 是否为有效异常（True表示异常，False表示假阳性）
            - validation_info: 验证信息字典
        """
        if self.cotracker_model is None:
            raise RuntimeError(
                "Co-Tracker模型未初始化\n"
                "请检查模型路径是否正确，以及Co-Tracker模块是否可用"
            )
        
        try:
            # 转换掩码格式
            if isinstance(mask, np.ndarray):
                mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                appear_mask = torch.from_numpy(np.array(mask_image))[None, None].to(self.device)
            else:
                appear_mask = mask.to(self.device) if isinstance(mask, torch.Tensor) else mask
            
            # 使用Co-Tracker进行反向追踪
            pred_tracks, pred_visibility = self.cotracker_model(
                video,
                grid_size=self.grid_size,
                grid_query_frame=query_frame,
                backward_tracking=True,
                segm_mask=appear_mask
            )
            
            # 检查各种情况
            edge_emerge = self._is_edge_emerge(
                pred_tracks, pred_visibility, query_frame, video_width, video_height
            )
            
            small_emerge = self._is_small_emerge(
                pred_tracks, pred_visibility, query_frame, video_width, video_height
            )
            
            detect_error = self._is_emerge_detect_error(
                pred_tracks, pred_visibility, query_frame
            )
            
            # 如果满足任何合理情况，认为是假阳性
            is_false_positive = edge_emerge or small_emerge or detect_error
            is_valid = not is_false_positive
            
            validation_info = {
                'edge_emerge': edge_emerge,
                'small_emerge': small_emerge,
                'detect_error': detect_error,
                'is_valid': is_valid,
                'reason': 'edge_emerge' if edge_emerge else 'small_emerge' if small_emerge else 'detect_error' if detect_error else 'valid_anomaly'
            }
            
            return is_valid, validation_info
            
        except Exception as e:
            raise RuntimeError(
                f"Co-Tracker验证失败: {e}\n"
                f"查询帧: {query_frame}\n"
                f"视频尺寸: {video_width}x{video_height}"
            )
    
    def _is_edge_vanish(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        start: int,
        width: int,
        height: int,
        visibility_ratio: float = 0.8,
        point_ratio: float = 0.5
    ) -> bool:
        """检查是否从边缘消失"""
        # 获取不可见的帧
        false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
        indices = torch.where(false_ratio >= visibility_ratio)[0]
        # 过滤开始帧之后的帧
        indices = indices[indices > start]
        
        if len(indices) == 0:
            return False
        
        selected_frames = pred_tracks[0, indices]  # shape: [len(indices), point_num, 2]
        
        # 检查各个边缘
        left_mask = selected_frames[:, :, 0] < 0
        right_mask = selected_frames[:, :, 0] > width
        top_mask = selected_frames[:, :, 1] < 0
        bottom_mask = selected_frames[:, :, 1] > height
        
        # 满足任一条件
        out_of_screen_mask = left_mask | right_mask | top_mask | bottom_mask
        
        # 计算超出屏幕的比例
        out_of_screen_ratio = out_of_screen_mask.float().mean(dim=1)
        
        # 检查超出屏幕比例是否 >= point_ratio
        valid_frames_mask = out_of_screen_ratio >= point_ratio
        vanish_indices = indices[valid_frames_mask]
        
        return len(vanish_indices) > 0
    
    def _is_small_vanish(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        start: int,
        width: int,
        height: int,
        visibility_ratio: float = -1,
        point_ratio: float = 0.8,
        size_threshold: float = 0.07
    ) -> bool:
        """检查是否因为太小而消失"""
        # 获取不可见的帧
        false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
        indices = torch.where(false_ratio >= visibility_ratio)[0]
        # 过滤开始帧之后的帧
        indices = indices[indices > start]
        
        if len(indices) == 0:
            return False
        
        pred_tracks = pred_tracks[0, indices]
        small_object_frames = []
        
        for i, frame in enumerate(pred_tracks):
            # 确定对象是否在屏幕上
            left_mask = frame[:, 0] > 0
            right_mask = frame[:, 0] < width
            top_mask = frame[:, 1] > 0
            bottom_mask = frame[:, 1] < height
            in_screen_mask = left_mask & right_mask & top_mask & bottom_mask
            
            # 计算由关键点形成的对象是否非常小
            valid_points = frame[in_screen_mask]
            if in_screen_mask.float().mean(dim=0) >= point_ratio and valid_points.shape[0] > 1:
                q_low = torch.quantile(valid_points, 0.1, dim=0)
                q_high = torch.quantile(valid_points, 0.9, dim=0)
                object_width = (q_high[0] - q_low[0]) / width
                object_height = (q_high[1] - q_low[1]) / height
                object_size = max(object_width, object_height)
                
                if object_size < size_threshold:
                    small_object_frames.append(i)
        
        return len(small_object_frames) > 0
    
    def _is_vanish_detect_error(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        start: int,
        visibility_ratio: float = 1.0
    ) -> bool:
        """检查是否因为检测错误而消失"""
        # 获取不可见的帧
        false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
        indices = torch.where(false_ratio >= visibility_ratio)[0]
        # 过滤开始帧之后的帧
        indices = indices[indices > start]
        
        return len(indices) == 0
    
    def _is_edge_emerge(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        start: int,
        width: int,
        height: int,
        visibility_ratio: float = 0.85,
        point_ratio: float = 0.5
    ) -> bool:
        """检查是否从边缘出现"""
        # 过滤对象不可见的帧
        false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
        indices = torch.where(false_ratio >= visibility_ratio)[0]
        # 过滤开始帧之前的帧
        indices = indices[indices < start]
        
        if len(indices) == 0:
            return False
        
        selected_frames = pred_tracks[0, indices]
        
        # 检查各个边缘
        left_mask = selected_frames[:, :, 0] < 0
        right_mask = selected_frames[:, :, 0] > width
        top_mask = selected_frames[:, :, 1] < 0
        bottom_mask = selected_frames[:, :, 1] > height
        
        # 满足任一条件
        out_of_screen_mask = left_mask | right_mask | top_mask | bottom_mask
        
        # 计算超出屏幕的比例
        out_of_screen_ratio = out_of_screen_mask.float().mean(dim=1)
        
        # 检查超出屏幕比例是否 >= point_ratio
        valid_frames_mask = out_of_screen_ratio >= point_ratio
        emerge_indices = indices[valid_frames_mask]
        
        return len(emerge_indices) > 0
    
    def _is_small_emerge(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        start: int,
        width: int,
        height: int,
        visibility_ratio: float = -1,
        point_ratio: float = 0.8,
        size_threshold: float = 0.03
    ) -> bool:
        """检查是否因为太小而出现"""
        false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
        indices = torch.where(false_ratio >= visibility_ratio)[0]
        indices = indices[indices < start]
        
        if len(indices) == 0:
            return False
        
        pred_tracks = pred_tracks[0, indices]
        small_object_frames = []
        
        for i, frame in enumerate(pred_tracks):
            # 确定对象是否在屏幕上
            left_mask = frame[:, 0] > 0
            right_mask = frame[:, 0] < width
            top_mask = frame[:, 1] > 0
            bottom_mask = frame[:, 1] < height
            in_screen_mask = left_mask & right_mask & top_mask & bottom_mask
            
            # 计算由关键点形成的对象是否非常小
            valid_points = frame[in_screen_mask]
            if in_screen_mask.float().mean(dim=0) >= point_ratio and valid_points.shape[0] > 1:
                q_low = torch.quantile(valid_points, 0.1, dim=0)
                q_high = torch.quantile(valid_points, 0.9, dim=0)
                object_width = (q_high[0] - q_low[0]) / width
                object_height = (q_high[1] - q_low[1]) / height
                object_size = max(object_width, object_height)
                
                if object_size < size_threshold:
                    small_object_frames.append(i)
        
        return len(small_object_frames) > 0
    
    def _is_emerge_detect_error(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        start: int,
        visibility_ratio: float = 0.8
    ) -> bool:
        """检查是否因为检测错误而出现"""
        # 获取不可见的帧
        false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
        indices = torch.where(false_ratio >= visibility_ratio)[0]
        # 过滤开始帧之前的帧
        indices = indices[indices < start]
        
        return len(indices) == 0

