# MediaPipe 模型配置说明

## 概述

MediaPipe 模型可以通过配置指定模型缓存目录。默认情况下，MediaPipe 会将模型文件下载到系统默认缓存目录。

## 配置方法

### 方法1: 通过配置文件设置

在配置文件中设置 `keypoint.model_path` 为缓存目录：

```python
from src.temporal_reasoning import TemporalReasoningConfig

config = TemporalReasoningConfig()
# 设置MediaPipe模型缓存目录为 .cache/mediapipe
config.keypoint.model_path = ".cache/mediapipe"
```

### 方法2: 直接使用 MediaPipeKeypointExtractor

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 指定模型缓存目录
extractor = MediaPipeKeypointExtractor(model_cache_dir=".cache/mediapipe")
```

### 方法3: 通过 KeypointAnalyzer 配置

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from src.temporal_reasoning.core.config import KeypointConfig

# 创建配置
keypoint_config = KeypointConfig()
keypoint_config.model_path = ".cache/mediapipe"  # 指定缓存目录

# 创建分析器
analyzer = KeypointAnalyzer(keypoint_config)
analyzer.initialize()
```

## 模型文件说明

MediaPipe Holistic 模型包含以下模型文件：
- 姿态检测模型 (pose_landmark)
- 手部检测模型 (hand_landmark)
- 面部检测模型 (face_landmark)

这些模型文件会在首次使用时自动下载到系统默认缓存目录。

## 注意事项

1. **模型下载位置**: MediaPipe 会将模型文件下载到系统默认缓存目录，**不会下载到指定的 `.cache` 目录**：
   - Windows: `C:\Users\<username>\AppData\Local\Temp\mediapipe`
   - Linux/Mac: `~/.cache/mediapipe`

2. **重要说明**: MediaPipe **不支持直接指定自定义缓存目录**。即使指定了 `model_cache_dir` 参数，模型仍会下载到系统默认缓存目录。

3. **解决方案**: 首次使用后，使用 `copy_models_from_system_cache()` 方法将模型复制到指定目录：
   ```python
   extractor = MediaPipeKeypointExtractor(model_cache_dir=".cache/mediapipe")
   # 首次使用，触发模型下载
   extractor.extract_keypoints(test_image)
   # 复制模型文件到指定目录
   extractor.copy_models_from_system_cache()
   ```

4. **手动复制模型**: 如果需要手动复制模型文件：
   - 让 MediaPipe 自动下载一次（下载到系统默认目录）
   - 使用代码的 `copy_models_from_system_cache()` 方法自动复制
   - 或手动将模型文件从系统缓存目录复制到 `.cache/mediapipe` 目录

详细说明请参考: [MEDIAPIPE_CACHE_ISSUE.md](./MEDIAPIPE_CACHE_ISSUE.md)

## 示例

### 完整示例

```python
from src.temporal_reasoning import TemporalReasoningConfig, TemporalReasoningAnalyzer

# 创建配置
config = TemporalReasoningConfig()

# 设置MediaPipe模型缓存目录
config.keypoint.model_path = ".cache/mediapipe"

# 创建分析器并初始化
analyzer = TemporalReasoningAnalyzer(config)
analyzer.initialize()

# 使用分析器
video_frames = [...]  # 你的视频帧
score, anomalies = analyzer.keypoint_analyzer.analyze(video_frames, fps=30.0)
```

## 模型文件查找

代码会自动检查指定目录下的以下格式的模型文件：
- `*.tflite`
- `*.binarypb`
- `*.pb`

如果找到模型文件，会显示找到的文件数量；如果未找到，MediaPipe 会在首次使用时自动下载。

