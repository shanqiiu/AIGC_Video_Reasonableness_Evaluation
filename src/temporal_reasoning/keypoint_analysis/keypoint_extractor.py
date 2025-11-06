# -*- coding: utf-8 -*-
"""
关键点提取器 - 使用MediaPipe旧API
仅支持Holistic模型（身体+手部+面部）
仅支持离线模式，从缓存目录加载模型
"""

import os
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class MediaPipeKeypointExtractor:
    """基于MediaPipe的关键点提取器（使用旧API，仅离线模式）"""
    
    def __init__(self, model_path: Optional[str] = None, cache_dir: str = ".cache"):
        """
        初始化MediaPipe关键点提取器（仅离线模式）
        
        Args:
            model_path: 模型文件路径（旧API不使用此参数，保留以兼容接口）
            cache_dir: 模型缓存目录，默认为.cache
                      模型文件应位于 cache_dir/mediapipe/models/ 目录中
        
        Raises:
            RuntimeError: 如果模型文件不存在或初始化失败
        """
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置MediaPipe缓存目录环境变量
        # 默认从 .cache/mediapipe 加载模型（模型文件直接在此目录下，不在models子目录中）
        mediapipe_cache = self.cache_dir / "mediapipe"
        mediapipe_cache.mkdir(parents=True, exist_ok=True)
        os.environ['MEDIAPIPE_CACHE_DIR'] = str(mediapipe_cache)
        
        # 禁用自动下载
        os.environ['MEDIAPIPE_DISABLE_AUTO_DOWNLOAD'] = '1'
        
        print(f"MediaPipe缓存目录: {mediapipe_cache}")
        print("离线模式：仅从缓存目录加载模型（不会自动下载）")
        print("注意：模型文件应直接放在 mediapipe/ 目录中")
        
        # 初始化Holistic模型
        self.holistic = None
        self.mp_holistic = None
        self._initialize_holistic()
    
    def _initialize_holistic(self):
        """初始化Holistic模型（仅离线模式）"""
        try:
            import mediapipe as mp
            
            # 检查模型是否已缓存
            # 优先从指定的缓存目录加载模型
            # 模型文件直接放在 mediapipe 目录下，不在 models 子目录中
            primary_cache_dir = self.cache_dir / "mediapipe"
            
            # 也检查其他可能的缓存位置（作为后备）
            user_home = Path.home()
            possible_cache_dirs = [
                primary_cache_dir,  # 优先使用指定的缓存目录
                user_home / ".mediapipe" / "models",  # 系统默认位置（如果有models子目录）
                Path(os.environ.get('MEDIAPIPE_CACHE_DIR', '')) if os.environ.get('MEDIAPIPE_CACHE_DIR') else None,
            ]
            
            # 过滤掉None
            possible_cache_dirs = [d for d in possible_cache_dirs if d is not None]
            
            # 检查是否有模型文件（MediaPipe旧API使用.tflite或.binarypb格式）
            model_found = False
            found_cache_dir = None
            model_files = []
            
            for cache_dir in possible_cache_dirs:
                if cache_dir.exists():
                    # 检查是否有模型文件
                    found_files = list(cache_dir.glob("*.tflite")) + list(cache_dir.glob("*.binarypb"))
                    if found_files:
                        model_found = True
                        found_cache_dir = cache_dir
                        model_files = found_files
                        print(f"在缓存目录发现模型文件: {cache_dir}")
                        print(f"  找到 {len(model_files)} 个模型文件:")
                        for f in model_files[:5]:  # 显示前5个文件
                            print(f"    - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
                        if len(model_files) > 5:
                            print(f"    ... 还有 {len(model_files) - 5} 个文件")
                        break
            
            if not model_found:
                error_msg = (
                    "错误：未找到MediaPipe模型文件\n"
                    f"请确保模型文件位于以下目录：\n"
                    f"  主要缓存目录: {primary_cache_dir}\n"
                )
                for cache_dir in possible_cache_dirs[1:]:  # 跳过主要目录
                    if cache_dir:
                        error_msg += f"  - {cache_dir}\n"
                error_msg += (
                    "\n模型文件格式：.tflite 或 .binarypb\n"
                    "模型文件应直接放在 mediapipe/ 目录中（不在 models/ 子目录中）\n"
                    "提示：请在联网环境下首次运行以下载模型，或手动将模型文件复制到缓存目录"
                )
                raise RuntimeError(error_msg)
            
            self.mp_holistic = mp.solutions.holistic
            
            # 尝试初始化Holistic模型
            try:
                self.holistic = self.mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    refine_face_landmarks=True
                )
                
                print("MediaPipe Holistic模型（旧API）初始化成功")
                print("支持检测：身体（33个）+ 手部（42个）+ 面部（468个）关键点")
                
            except Exception as init_error:
                error_msg = (
                    f"MediaPipe Holistic模型初始化失败: {init_error}\n"
                    "可能原因：\n"
                    "1. 模型文件损坏或不完整\n"
                    "2. MediaPipe版本不兼容\n"
                    "3. 模型文件格式不正确\n\n"
                    "解决方案：\n"
                    "1. 检查模型文件是否完整\n"
                    "2. 重新下载模型文件\n"
                    "3. 检查MediaPipe版本兼容性\n"
                    f"4. 检查模型缓存目录: {found_cache_dir}"
                )
                raise RuntimeError(error_msg)
            
        except ImportError as e:
            raise ImportError(
                f"MediaPipe未安装: {e}\n"
                "请安装MediaPipe: pip install mediapipe"
            )
        except RuntimeError:
            # 重新抛出RuntimeError（包含详细的错误信息）
            raise
        except Exception as e:
            raise RuntimeError(
                f"MediaPipe Holistic模型初始化失败: {e}\n"
                "请检查模型文件是否存在且完整"
            )
    
    def reset_timestamp(self):
        """
        重置timestamp计数器（用于处理新视频时）
        
        注意：旧API不需要timestamp，此方法为空实现
        保留此方法是为了兼容性，避免调用时出错
        """
        # 旧API不需要timestamp，无需操作
        pass
    
    def extract_keypoints(self, image: np.ndarray, fps: float = 30.0) -> Dict:
        """
        提取关键点
        
        Args:
            image: 输入图像 (H, W, 3) RGB，范围[0, 255]，uint8类型
            fps: 视频帧率（保留参数以兼容接口，旧API不使用）
        
        Returns:
            关键点字典，包含：
            - body: 身体关键点 (33, 3) 或 None
            - left_hand: 左手关键点 (21, 3) 或 None
            - right_hand: 右手关键点 (21, 3) 或 None
            - face: 面部关键点 (468, 3) 或 None
        """
        return self._extract_keypoints_holistic(image)
    
    def _extract_keypoints_holistic(self, image: np.ndarray) -> Dict:
        """使用Holistic模型提取关键点（支持身体+手部+面部）"""
        if self.holistic is None:
            raise RuntimeError("MediaPipe Holistic模型未初始化")
        
        try:
            # 确保图像格式正确
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # 使用Holistic模型处理
            results = self.holistic.process(image)
            
            keypoints = {
                'body': None,
                'left_hand': None,
                'right_hand': None,
                'face': None
            }
            
            # 提取身体关键点（33个）
            if results.pose_landmarks:
                keypoints['body'] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ])
            
            # 提取手部关键点（每只手21个）
            if results.left_hand_landmarks:
                keypoints['left_hand'] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                ])
            if results.right_hand_landmarks:
                keypoints['right_hand'] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                ])
            
            # 提取面部关键点（468个）
            if results.face_landmarks:
                keypoints['face'] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
                ])
            
            return keypoints
            
        except Exception as e:
            print(f"警告: 关键点提取失败（Holistic）: {e}")
            return self._empty_keypoints()
    
    def _empty_keypoints(self) -> Dict:
        """返回空的关键点字典"""
        return {
            'body': None,
            'left_hand': None,
            'right_hand': None,
            'face': None
        }
