# -*- coding: utf-8 -*-
"""MSS scorer thin wrapper (Q-Align based). Placeholder for future decoupling."""

from typing import Any, Callable, Dict, Optional


class MSSScorer:
    """Thin facade around Q-Align video scoring with decoupled frame loading.

    - Removes cross-project dependency by using an internal sliding-window loader.
    - Allows injecting a custom frame loader for maximum encapsulation and testability.
    """

    def __init__(
        self,
        device: str,
        model_paths: Dict[str, str],
        frame_loader: Optional[Callable[[str, int], Any]] = None,
        window_size: int = 3,
        use_8bit: bool = False,
        use_4bit: bool = False,
    ):
        self.device = device
        self.model_paths = model_paths
        self.window_size = window_size
        self._frame_loader = frame_loader
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit

        # Defer heavy imports to when we actually use this class.
        self._initialized = False
        self._scorer = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        
        # Set environment variables for memory management
        import os
        os.environ.setdefault('ACCELERATE_USE_CPU', 'false')
        # Enable expandable segments to avoid memory fragmentation
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        
        # Clear GPU cache before loading model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"已清理GPU缓存。可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"清理GPU缓存时出错: {e}")
        
        # Local imports to avoid global side-effects and hard deps at import time
        try:
            # Prefer the package export if available
            from q_align import QAlignVideoScorer  # type: ignore
        except Exception:
            # Fallback to the scorer defined in evaluate.scorer within the vendor package
            from q_align.evaluate.scorer import QAlignVideoScorer  # type: ignore

        if self._frame_loader is None:
            # Use the internal implementation to avoid cross-repo dependency
            from src.io.video import load_video_sliding_window  # type: ignore
            self._frame_loader = load_video_sliding_window
        
        # Get model path and handle both HuggingFace format and local path
        model_path = self.model_paths.get("q_align_model", "")
        if not model_path:
            # Fallback to HuggingFace format if not specified
            model_path = "q-future/one-align"
        elif model_path.startswith(".cache/") or model_path.startswith("./cache/"):
            # Handle relative path - convert to absolute or HuggingFace format
            # Try HuggingFace format first as it's more reliable
            if not os.path.exists(model_path) and not os.path.isabs(model_path):
                # If relative path doesn't exist, use HuggingFace format
                model_path = "q-future/one-align"
        
        # Normalize device string to avoid device_map issues
        # According to builder.py: if device != "cuda", it sets device_map={"": device}
        # If device == "cuda", it uses device_map="auto" which may cause offload
        # So we keep "cuda:0" format to avoid device_map="auto" and prevent offload
        device_for_model = self.device
        if device_for_model == "cuda":
            # If just "cuda", convert to "cuda:0" to avoid device_map="auto"
            device_for_model = "cuda:0"
        
        try:
            # Check if QAlignVideoScorer supports quantization parameters
            # If not, we may need to use a custom initialization
            if self.use_8bit or self.use_4bit:
                # Try to use quantization - this requires monkey-patching load_pretrained_model
                # or using a custom scorer class
                print(f"使用量化加载模型 (8bit={self.use_8bit}, 4bit={self.use_4bit})")
                # Note: Standard QAlignVideoScorer doesn't support quantization directly
                # We'll need to load the model manually if quantization is required
                # For now, just warn and use standard loading
                if self.use_8bit:
                    print("警告: 8bit量化需要自定义加载逻辑，暂时使用标准加载")
                if self.use_4bit:
                    print("警告: 4bit量化需要自定义加载逻辑，暂时使用标准加载")
            
            self._scorer = QAlignVideoScorer(pretrained=model_path, device=device_for_model)
            
        except RuntimeError as e:
            error_str = str(e)
            if "out of memory" in error_str.lower() or ("cuda" in error_str.lower() and "memory" in error_str.lower()):
                print(f"⚠️ GPU内存不足: {e}")
                print("建议解决方案：")
                print("1. 清理其他占用GPU的程序")
                print("2. 使用更小的batch size或window_size")
                print("3. 设置环境变量: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                print("4. 使用CPU模式（性能较慢）")
                print("5. 考虑使用模型量化（需要修改代码支持）")
                
                # Try to clear cache and retry
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("已清理GPU缓存，可以重试...")
                except:
                    pass
                raise
            elif "device_map" in error_str.lower() or "offload" in error_str.lower():
                print(f"警告: 模型加载遇到device_map问题: {e}")
                print("尝试安装 safetensors 或检查模型路径...")
                print("建议: pip install safetensors")
                raise
            else:
                raise
        except Exception as e:
            raise
        
        self._initialized = True

    def score(self, video_path: str) -> Dict:
        self._ensure_init()
        frames = self._frame_loader(video_path, window_size=self.window_size)
        output = self._scorer(frames)

        # Handle different return formats from QAlignVideoScorer
        # VMBench's custom version returns (logits, softmax, weighted_score) - use the last (weighted_score)
        # Q-Align package version returns only weighted_score directly
        if isinstance(output, tuple):
            # If tuple, use the last element which should be the weighted quality score
            quality_scores = output[-1]
        else:
            # If single tensor, use it directly (should be the weighted quality score)
            quality_scores = output
        
        # Ensure it's a tensor and convert to list
        if hasattr(quality_scores, 'tolist'):
            quality_scores_list = quality_scores.tolist()
        elif isinstance(quality_scores, list):
            quality_scores_list = quality_scores
        else:
            # If it's a numpy array or other type
            import numpy as np
            if isinstance(quality_scores, np.ndarray):
                quality_scores_list = quality_scores.tolist()
            else:
                # Convert to list if needed
                quality_scores_list = list(quality_scores)

        return {"quality_scores": quality_scores_list}


