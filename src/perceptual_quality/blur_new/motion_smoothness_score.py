# -*- coding: utf-8 -*-
"""
视频运动平滑度评分模块
基于 Q-Align 模型计算视频质量分数，用于检测模糊帧
"""

import os
import sys
import json
from typing import List, Tuple, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from decord import VideoReader
from tqdm import tqdm

# Add Q-Align to path if not installed as package
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..', '..', '..'))
_Q_ALIGN_PATH = os.path.join(_PROJECT_ROOT, 'third_party', 'Q-Align')
if _Q_ALIGN_PATH not in sys.path:
    sys.path.insert(0, _Q_ALIGN_PATH)

from q_align.model.builder import load_pretrained_model  # type: ignore
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
from q_align.mm_utils import tokenizer_image_token  # type: ignore


class QAlignVideoScorer(nn.Module):
    """基于 Q-Align 模型的视频质量评分器"""
    
    # 质量等级权重
    QUALITY_WEIGHTS = torch.Tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    # 质量等级关键词
    QUALITY_KEYWORDS = ["excellent", "good", "fair", "poor", "bad"]
    # 质量评估提示词
    QUALITY_PROMPT = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"
    
    def __init__(self, pretrained: str = "q-future/one-align", device: str = "cuda:0"):
        """
        初始化 Q-Align 视频评分器
        
        Args:
            pretrained: 预训练模型路径或名称
            device: 计算设备
        """
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, "mplug_owl2", device=device
        )
        
        # 提取质量等级对应的 token ID
        quality_token_ids = tokenizer(self.QUALITY_KEYWORDS)["input_ids"]
        self.preferential_ids = [token_id[1] for token_id in quality_token_ids]
        
        # 初始化权重张量
        self.weight_tensor = self.QUALITY_WEIGHTS.half().to(model.device)
        
        # 保存模型组件
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        
        # 准备输入 token
        input_ids = tokenizer_image_token(
            self.QUALITY_PROMPT, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        )
        self.input_ids = input_ids.unsqueeze(0).to(model.device)
    
    def _expand_to_square(self, image: Image.Image, background_color: Tuple[int, int, int]) -> Image.Image:
        """
        将图像扩展为正方形
        
        Args:
            image: PIL 图像对象
            background_color: 背景颜色 (R, G, B)
            
        Returns:
            扩展后的正方形图像
        """
        width, height = image.size
        if width == height:
            return image
        
        if width > height:
            # 宽度大于高度，在上下两侧填充
            result = Image.new(image.mode, (width, width), background_color)
            result.paste(image, (0, (width - height) // 2))
        else:
            # 高度大于宽度，在左右两侧填充
            result = Image.new(image.mode, (height, height), background_color)
            result.paste(image, ((height - width) // 2, 0))
        
        return result
    
    def forward(
        self,
        video_frames: List[List[Image.Image]],
        batch_size: Optional[int] = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对视频帧进行质量评分（支持批处理，防止 OOM）
        
        Args:
            video_frames: 视频帧列表，每个元素是一个帧组列表
            batch_size: 批处理大小，默认 32。None 表示一次性处理所有帧
            
        Returns:
            (logits, probabilities, weighted_scores)
            - logits: 原始 logits
            - probabilities: 质量等级概率分布
            - weighted_scores: 加权质量分数
        """
        # 如果没有指定批处理大小，或批处理大小大于等于总帧数，一次性处理
        if batch_size is None or batch_size >= len(video_frames):
            return self._forward_batch(video_frames)
        
        # 分批处理
        all_logits = []
        all_probabilities = []
        all_weighted_scores = []
        
        total_batches = (len(video_frames) + batch_size - 1) // batch_size
        
        for i in range(0, len(video_frames), batch_size):
            batch_frames = video_frames[i:i + batch_size]
            logits, probabilities, weighted_scores = self._forward_batch(batch_frames)
            
            all_logits.append(logits.cpu())  # 移到 CPU 释放 GPU 内存
            all_probabilities.append(probabilities.cpu())
            all_weighted_scores.append(weighted_scores.cpu())
            
            # 释放 GPU 内存
            del logits, probabilities, weighted_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有批次的结果（在 CPU 上合并）
        all_logits = torch.cat(all_logits, dim=0).to(self.model.device)
        all_probabilities = torch.cat(all_probabilities, dim=0).to(self.model.device)
        all_weighted_scores = torch.cat(all_weighted_scores, dim=0).to(self.model.device)
        
        return all_logits, all_probabilities, all_weighted_scores
    
    def _forward_batch(self, video_frames: List[List[Image.Image]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单个批次的视频帧进行质量评分
        
        Args:
            video_frames: 视频帧批次列表
            
        Returns:
            (logits, probabilities, weighted_scores)
        """
        # 将图像扩展为正方形
        background_color = tuple(
            int(x * 255) for x in self.image_processor.image_mean
        )
        processed_frames = [
            [
                self._expand_to_square(frame, background_color)
                for frame in frame_group
            ]
            for frame_group in video_frames
        ]
        
        with torch.inference_mode():
            # 预处理视频帧
            video_tensors = [
                self.image_processor.preprocess(
                    frame_group, return_tensors="pt"
                )["pixel_values"].half().to(self.model.device)
                for frame_group in processed_frames
            ]
            
            # 准备输入
            input_tensors = self.input_ids.repeat(len(video_tensors), 1)
            
            # 模型推理
            output = self.model(input_tensors, images=video_tensors)
            
            # 提取质量等级的 logits
            output_logits = output["logits"][:, -1, self.preferential_ids]
            
            # 计算概率分布
            probabilities = torch.softmax(output_logits, dim=-1)
            
            # 计算加权分数
            weighted_scores = probabilities @ self.weight_tensor
            
            return output_logits, probabilities, weighted_scores

def load_video_with_sliding_window(video_path: str, window_size: int = 5) -> List[List[Image.Image]]:
    """
    使用滑动窗口方式加载视频帧（一次性加载所有帧）
    
    Args:
        video_path: 视频文件路径
        window_size: 滑动窗口大小（帧数）
        
    Returns:
        视频帧组列表，每个元素是一个帧组（包含 window_size 帧）
    """
    video_reader = VideoReader(video_path)
    total_frames = len(video_reader)
    frame_groups = []
    
    # 计算窗口左右扩展帧数
    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend
    
    for current_frame_idx in range(total_frames):
        # 计算窗口的起始和结束帧索引
        start_frame_idx = max(0, current_frame_idx - left_extend)
        end_frame_idx = min(total_frames, current_frame_idx + right_extend + 1)
        
        frame_indices = list(range(start_frame_idx, end_frame_idx))
        
        # 如果帧数不足，进行填充
        while len(frame_indices) < window_size:
            if start_frame_idx == 0:
                # 如果窗口在开头，向后填充
                frame_indices.append(frame_indices[-1])
            else:
                # 否则向前填充
                frame_indices.insert(0, frame_indices[0])
        
        # 读取帧数据
        frames_array = video_reader.get_batch(frame_indices).asnumpy()
        
        # 处理开头帧，确保一致性
        if current_frame_idx < left_extend:
            frame_groups.append([Image.fromarray(frames_array[0])] * window_size)
        else:
            frame_groups.append([Image.fromarray(frame) for frame in frames_array])
    
    return frame_groups


def load_video_with_sliding_window_generator(
    video_path: str,
    window_size: int = 5
) -> Iterator[List[Image.Image]]:
    """
    使用滑动窗口方式加载视频帧（生成器版本，节省内存）
    
    Args:
        video_path: 视频文件路径
        window_size: 滑动窗口大小（帧数）
        
    Yields:
        视频帧组，每个帧组包含 window_size 帧
    """
    video_reader = VideoReader(video_path)
    total_frames = len(video_reader)
    
    # 计算窗口左右扩展帧数
    left_extend = (window_size - 1) // 2
    right_extend = window_size - 1 - left_extend
    
    for current_frame_idx in range(total_frames):
        # 计算窗口的起始和结束帧索引
        start_frame_idx = max(0, current_frame_idx - left_extend)
        end_frame_idx = min(total_frames, current_frame_idx + right_extend + 1)
        
        frame_indices = list(range(start_frame_idx, end_frame_idx))
        
        # 如果帧数不足，进行填充
        while len(frame_indices) < window_size:
            if start_frame_idx == 0:
                # 如果窗口在开头，向后填充
                frame_indices.append(frame_indices[-1])
            else:
                # 否则向前填充
                frame_indices.insert(0, frame_indices[0])
        
        # 读取帧数据
        frames_array = video_reader.get_batch(frame_indices).asnumpy()
        
        # 处理开头帧，确保一致性
        if current_frame_idx < left_extend:
            frame_group = [Image.fromarray(frames_array[0])] * window_size
        else:
            frame_group = [Image.fromarray(frame) for frame in frames_array]
        
        # 生成当前帧组
        yield frame_group
        
        # 自动释放内存
        del frames_array, frame_group


def calculate_adaptive_threshold(camera_movement: float = None) -> float:
    """
    根据相机运动幅度计算自适应阈值
    
    Args:
        camera_movement: 相机运动幅度（0-1之间）
        
    Returns:
        自适应阈值
    """
    if camera_movement is None:
        return 0.01
    
    if camera_movement < 0.1:
        return 0.01
    elif 0.1 <= camera_movement < 0.3:
        return 0.015
    elif 0.3 <= camera_movement < 0.5:
        return 0.025
    else:  # camera_movement >= 0.5
        return 0.03


def detect_artifact_frames(quality_scores: List[float], threshold: float = 0.025) -> np.ndarray:
    """
    基于质量分数差异检测异常帧
    
    Args:
        quality_scores: 质量分数列表
        threshold: 检测阈值
        
    Returns:
        异常帧索引数组
    """
    # 计算相邻帧之间的分数差异
    score_differences = np.abs(np.diff(quality_scores))
    
    # 找出分数差异超过阈值的帧
    artifact_indices = np.where(score_differences > threshold)[0]
    
    # 返回当前帧和下一帧（因为显著的分数差异可能由任一帧引起）
    artifact_frame_indices = np.unique(
        np.concatenate([artifact_indices, artifact_indices + 1])
    )
    
    return artifact_frame_indices


