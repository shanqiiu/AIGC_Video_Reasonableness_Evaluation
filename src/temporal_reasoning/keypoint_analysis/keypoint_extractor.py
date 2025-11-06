# -*- coding: utf-8 -*-
"""
关键点提取器 - 使用最新MediaPipe API
"""

import os
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import warnings
import urllib.request
warnings.filterwarnings("ignore")

# MediaPipe官方模型下载URL
DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


class MediaPipeKeypointExtractor:
    """基于MediaPipe的关键点提取器（使用最新API）"""
    
    def __init__(self, model_path: Optional[str] = None, cache_dir: str = ".cache"):
        """
        初始化MediaPipe关键点提取器
        
        Args:
            model_path: .task模型文件路径，如果为None则使用默认模型
            cache_dir: 模型缓存目录，默认为.cache
        """
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置MediaPipe缓存目录环境变量
        mediapipe_cache = self.cache_dir / "mediapipe"
        mediapipe_cache.mkdir(parents=True, exist_ok=True)
        os.environ['MEDIAPIPE_CACHE_DIR'] = str(mediapipe_cache)
        
        # 模型路径处理
        if model_path:
            # 检查是否是URL
            if str(model_path).startswith(('http://', 'https://')):
                # 如果是URL，直接下载
                self.model_path = self._download_model(str(model_path), mediapipe_cache)
            else:
                # 如果是本地路径
                self.model_path = Path(model_path)
                if not self.model_path.exists():
                    # 文件不存在，尝试从默认URL下载
                    print(f"模型文件不存在: {model_path}")
                    print("尝试从默认URL下载模型...")
                    self.model_path = self._download_model(DEFAULT_MODEL_URL, mediapipe_cache)
        else:
            # 使用默认模型，检查缓存目录是否有模型文件
            default_model_path = mediapipe_cache / "pose_landmarker_heavy.task"
            if not default_model_path.exists():
                print("默认模型文件不存在，正在从网络下载...")
                self.model_path = self._download_model(DEFAULT_MODEL_URL, mediapipe_cache)
            else:
                self.model_path = default_model_path
        
        self.landmarker = None
        self._initialize()
    
    def _initialize(self):
        """初始化MediaPipe模型（仅使用新API）"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from mediapipe import Image, ImageFormat
            
            # 配置BaseOptions
            if self.model_path:
                base_options = python.BaseOptions(
                    model_asset_path=str(self.model_path)
                )
            else:
                # 使用默认模型（MediaPipe会自动下载到缓存目录）
                base_options = python.BaseOptions()
            
            # 配置PoseLandmarker选项
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.VIDEO
            )
            
            # 创建PoseLandmarker
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.mp_image = Image
            self.image_format = ImageFormat.SRGB
            
            model_info = str(self.model_path) if self.model_path else "默认模型"
            print(f"MediaPipe模型初始化成功: {model_info}")
            print(f"模型缓存目录: {os.environ.get('MEDIAPIPE_CACHE_DIR', '系统默认')}")
            
        except ImportError as e:
            raise ImportError(
                f"MediaPipe新API不可用: {e}\n"
                "请确保已安装最新版本的MediaPipe: pip install mediapipe>=0.10.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"MediaPipe初始化失败: {e}\n"
                "请确保MediaPipe版本>=0.10.0，并检查模型文件路径是否正确"
            )
    
    def extract_keypoints(self, image: np.ndarray) -> Dict:
        """
        提取关键点
        
        Args:
            image: 输入图像 (H, W, 3) RGB，范围[0, 255]，uint8类型
        
        Returns:
            关键点字典，包含：
            - body: 身体关键点 (N, 3) 或 None
            - left_hand: 左手关键点 (N, 3) 或 None
            - right_hand: 右手关键点 (N, 3) 或 None
            - face: 面部关键点 (N, 3) 或 None
        """
        if self.landmarker is None:
            return self._empty_keypoints()
        
        try:
            # 确保图像格式正确
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # 确保图像是连续的
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            # 转换为MediaPipe Image格式
            mp_image = self.mp_image(
                image_format=self.image_format,
                data=image
            )
            
            # 检测关键点
            detection_result = self.landmarker.detect_for_video(
                mp_image,
                timestamp_ms=0
            )
            
            # 提取关键点
            keypoints = {
                'body': None,
                'left_hand': None,
                'right_hand': None,
                'face': None
            }
            
            # 提取身体关键点（PoseLandmarker主要提供身体姿态）
            if detection_result.pose_landmarks:
                keypoints['body'] = np.array([
                    [lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]
                ])
            
            # 注意：PoseLandmarker主要关注身体姿态
            # 如果需要手部和面部关键点，需要使用HandLandmarker和FaceLandmarker
            
            return keypoints
            
        except Exception as e:
            print(f"警告: 关键点提取失败: {e}")
            return self._empty_keypoints()
    
    def _download_model(self, url: str, cache_dir: Path) -> Path:
        """
        从URL下载模型文件
        
        Args:
            url: 模型文件URL
            cache_dir: 缓存目录
        
        Returns:
            下载后的模型文件路径
        """
        # 从URL提取文件名
        filename = url.split('/')[-1].split('?')[0]  # 处理URL参数
        if not filename.endswith('.task'):
            filename = 'pose_landmarker_heavy.task'  # 默认文件名
        
        model_path = cache_dir / filename
        
        # 如果文件已存在，直接返回
        if model_path.exists():
            print(f"模型文件已存在: {model_path}")
            return model_path
        
        print(f"正在从 {url} 下载模型文件...")
        print(f"保存到: {model_path}")
        
        try:
            # 使用urllib下载文件，带进度条
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
            
            urllib.request.urlretrieve(url, model_path, reporthook=show_progress)
            print(f"\n模型下载完成: {model_path}")
            
            # 验证文件是否存在且大小合理
            if model_path.exists() and model_path.stat().st_size > 0:
                return model_path
            else:
                raise RuntimeError("下载的文件无效或为空")
                
        except Exception as e:
            # 如果下载失败，删除可能的不完整文件
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(
                f"模型下载失败: {e}\n"
                f"请检查网络连接，或手动下载模型文件到: {model_path}\n"
                f"下载URL: {url}"
            )
    
    def _empty_keypoints(self) -> Dict:
        """返回空的关键点字典"""
        return {
            'body': None,
            'left_hand': None,
            'right_hand': None,
            'face': None
        }
