# MediaPipe旧API vs 新API说明

## 旧API和新API的定义

### 旧API（当前使用）

**API路径**：`mediapipe.solutions.holistic.Holistic`

**代码示例**：
```python
import mediapipe as mp

self.mp_holistic = mp.solutions.holistic
self.holistic = self.mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True
)

# 使用方式
results = self.holistic.process(image)
```

**特点**：
- ? **更稳定**：经过充分测试，不容易崩溃
- ? **不需要.task文件**：模型自动下载到系统缓存
- ? **功能完整**：支持身体+手部+面部检测（543个关键点）
- ? **跨平台一致**：Linux和Windows行为一致
- ? **无Packet错误**：不会出现Packet错误

**模型文件**：
- 不需要手动下载.task文件
- MediaPipe会自动下载模型到系统缓存目录
- 模型文件格式：`.tflite`或`.binarypb`（内部格式）

**使用场景**：
- 推荐用于生产环境
- 需要稳定性的场景
- 不需要.task文件管理的场景

### 新API（可选，可能不稳定）

**API路径**：`mediapipe.tasks.python.vision.HolisticLandmarker`

**代码示例**：
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

base_options = python.BaseOptions(
    model_asset_path="path/to/holistic_landmarker.task"
)

options = vision.HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

self.landmarker = vision.HolisticLandmarker.create_from_options(options)

# 使用方式
mp_image = Image(image_format=ImageFormat.SRGB, data=image)
detection_result = self.landmarker.detect(mp_image)
```

**特点**：
- ? **可能不稳定**：某些版本中可能出现Packet错误
- ? **需要.task文件**：需要手动下载或指定.task文件
- ? **功能完整**：支持身体+手部+面部检测（543个关键点）
- ? **可能有bug**：某些版本中可能不稳定

**模型文件**：
- 需要`.task`文件（如`holistic_landmarker.task`）
- 可以从URL下载或使用本地文件
- 模型文件格式：`.task`（MediaPipe Task格式）

**使用场景**：
- 实验性使用
- 需要.task文件管理的场景
- MediaPipe版本较新且稳定的场景

## 为什么代码找到了.task文件但使用了旧API？

### 原因分析

根据日志：
```
发现holistic模型文件: D:\my_git_projects\AIGC_Video_Reasonableness_Evaluation\.cache\mediapipe\holistic_landmarker.task
MediaPipe Holistic模型（旧API）初始化成功
```

**原因**：
1. **默认配置**：代码默认使用旧API（`use_new_api=False`）
2. **即使找到.task文件**：代码也会优先使用旧API，因为旧API更稳定
3. **.task文件被忽略**：旧API不需要.task文件，所以即使找到了也不会使用

### 代码逻辑

```python
# 1. 如果启用新API，尝试使用新API
if self.use_new_api:
    # 尝试使用HolisticLandmarker（新API）
    ...
    return

# 2. 否则，使用旧API（默认）
# 使用旧API的Holistic模型
self.holistic = mp.solutions.holistic.Holistic(...)
```

**当前行为**：
- `use_new_api=False`（默认）
- 即使找到了`.task`文件，也不会使用新API
- 直接使用旧API（`mp.solutions.holistic`）

## 如何切换到新API？

### 方法1：修改初始化参数

```python
# 在keypoint_analyzer.py中
extractor = MediaPipeKeypointExtractor(
    model_path=None,
    cache_dir=cache_dir,
    use_new_api=True  # 启用新API
)
```

### 方法2：修改默认值

在`keypoint_extractor.py`中：
```python
def __init__(self, model_path: Optional[str] = None, cache_dir: str = ".cache", use_new_api: bool = True):
    # 改为True，默认使用新API
```

**注意**：如果启用新API，可能仍然会出现Packet错误，因为这是MediaPipe库本身的问题。

## 对比总结

| 特性 | 旧API | 新API |
|------|-------|-------|
| **API路径** | `mp.solutions.holistic` | `mediapipe.tasks.python.vision.HolisticLandmarker` |
| **模型文件** | 不需要（自动下载） | 需要`.task`文件 |
| **稳定性** | ? 更稳定 | ? 可能不稳定 |
| **Packet错误** | ? 不会出现 | ? 可能出现 |
| **功能** | ? 完整（543个关键点） | ? 完整（543个关键点） |
| **推荐使用** | ? 推荐 | ? 不推荐（除非版本稳定） |

## 建议

### 当前推荐
- ? **使用旧API**（默认）：更稳定，不容易崩溃
- ? **不需要.task文件**：MediaPipe会自动下载模型
- ? **功能完整**：支持身体+手部+面部检测

### 如果必须使用新API
1. **检查MediaPipe版本**：确保使用稳定版本
2. **启用新API**：设置`use_new_api=True`
3. **准备处理错误**：如果出现Packet错误，会自动fallback到旧API

## 总结

**旧API**：`mp.solutions.holistic.Holistic()` - 稳定、推荐使用  
**新API**：`mediapipe.tasks.python.vision.HolisticLandmarker` - 可能不稳定、需要.task文件

**当前代码行为**：即使找到了`.task`文件，也使用旧API，因为旧API更稳定。

