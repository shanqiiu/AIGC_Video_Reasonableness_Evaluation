# 关键点分析器重写总结

## 一、重写内容

### 1.1 主要改进

1. **采用最新MediaPipe API**
   - 使用 `mediapipe.tasks.python` 模块
   - 使用 `BaseOptions` 和 `PoseLandmarker` API
   - 支持 `.task` 模型文件格式

2. **简化代码结构**
   - 删除冗余的模型文件检查逻辑
   - 删除系统缓存目录复制功能
   - 删除环境变量恢复逻辑
   - 精简代码从 253 行减少到约 150 行

3. **指定缓存目录**
   - 模型缓存目录设置为 `.cache/mediapipe`
   - 通过环境变量 `MEDIAPIPE_CACHE_DIR` 指定

### 1.2 代码对比

#### 旧实现（已删除）：
- 使用 `mp.solutions.holistic.Holistic()` API
- 复杂的缓存目录管理
- 模型文件检查和复制逻辑
- 环境变量恢复机制
- 约 253 行代码

#### 新实现：
- 使用 `mediapipe.tasks.python.vision.PoseLandmarker` API
- 简化的缓存目录设置
- 直接使用 `.task` 文件或默认模型
- 约 150 行代码

## 二、核心实现

### 2.1 初始化流程

```python
# 1. 设置缓存目录
cache_dir = Path(".cache")
mediapipe_cache = cache_dir / "mediapipe"
os.environ['MEDIAPIPE_CACHE_DIR'] = str(mediapipe_cache)

# 2. 配置BaseOptions
base_options = python.BaseOptions(
    model_asset_path=model_path  # 可选，如果为None则使用默认模型
)

# 3. 创建PoseLandmarker
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)
```

### 2.2 关键点提取

```python
# 1. 转换图像格式
mp_image = Image(image_format=ImageFormat.SRGB, data=image)

# 2. 检测关键点
detection_result = landmarker.detect_for_video(mp_image, timestamp_ms=0)

# 3. 提取关键点
if detection_result.pose_landmarks:
    keypoints['body'] = np.array([
        [lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]
    ])
```

## 三、文件变更

### 3.1 修改的文件

1. **`keypoint_extractor.py`**
   - 完全重写，使用新API
   - 删除冗余代码
   - 简化缓存管理

2. **`keypoint_analyzer.py`**
   - 更新初始化逻辑
   - 使用新的 `cache_dir` 参数

### 3.2 删除的文件

1. **`MEDIAPIPE_UPGRADE_EXAMPLE.py`**
   - 升级示例代码已集成到主代码中

## 四、使用方式

### 4.1 基本使用

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 使用默认模型（自动下载到.cache/mediapipe）
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")

# 或指定自定义模型文件
extractor = MediaPipeKeypointExtractor(
    model_path="path/to/pose_landmarker.task",
    cache_dir=".cache"
)

# 提取关键点
keypoints = extractor.extract_keypoints(image)
```

### 4.2 配置使用

```python
from src.temporal_reasoning.core.config import KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer

config = KeypointConfig(
    model_type="mediapipe",
    model_path=None  # 使用默认模型
)

analyzer = KeypointAnalyzer(config)
analyzer.initialize()
```

## 五、注意事项

1. **MediaPipe版本要求**
   - 需要 MediaPipe >= 0.10.0
   - 安装命令：`pip install mediapipe>=0.10.0`

2. **模型文件**
   - 如果未指定 `model_path`，MediaPipe会自动下载默认模型
   - 模型会下载到 `.cache/mediapipe` 目录

3. **缓存目录**
   - 默认缓存目录为 `.cache/mediapipe`
   - 可通过 `cache_dir` 参数自定义

4. **关键点类型**
   - 当前实现主要提取身体关键点
   - 手部和面部关键点需要额外的 `HandLandmarker` 和 `FaceLandmarker`

## 六、优势

1. **代码简洁**：代码量减少约 40%
2. **API最新**：使用最新的 MediaPipe Task API
3. **易于维护**：删除冗余代码，逻辑更清晰
4. **缓存可控**：明确指定缓存目录为 `.cache`

---

**重写完成时间：** 2024年
**重写人员：** AI Assistant
**文档版本：** 1.0

