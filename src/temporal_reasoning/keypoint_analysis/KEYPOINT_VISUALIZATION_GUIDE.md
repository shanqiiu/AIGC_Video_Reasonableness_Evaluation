# 关键点可视化使用指南

## 概述

`keypoint_visualizer.py` 提供了MediaPipe关键点检测的可视化功能，支持可视化身体、手部和面部关键点。

## 功能特点

- ? **身体关键点可视化**：33个身体关键点，包括骨架连接
- ? **手部关键点可视化**：左右手各21个关键点，包括手指连接
- ? **面部关键点可视化**：468个面部关键点（可选）
- ? **支持图像和视频**：可以处理单张图像或视频文件
- ? **多种输出方式**：可以显示、保存或同时进行

## 使用方法

### 1. 命令行使用

#### 处理图像

```bash
# 基本使用（显示图像）
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer image.jpg

# 保存结果
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer image.jpg --output output.jpg

# 显示面部关键点
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer image.jpg --show-face

# 仅保存，不显示
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer image.jpg --output output.jpg --no-show
```

#### 处理视频

```bash
# 基本使用（处理所有帧）
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer video.mp4 --output output.mp4

# 每隔5帧处理一次
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer video.mp4 --output output.mp4 --frame-interval 5

# 最多处理100帧
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer video.mp4 --output output.mp4 --max-frames 100

# 显示面部关键点
python -m src.temporal_reasoning.keypoint_analysis.keypoint_visualizer video.mp4 --output output.mp4 --show-face
```

### 2. Python代码使用

#### 可视化单张图像

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_visualizer import (
    KeypointVisualizer,
    visualize_keypoints_from_image
)

# 方法1：使用便捷函数
visualize_keypoints_from_image(
    image_path="image.jpg",
    output_path="output.jpg",
    cache_dir=".cache",
    show_face=False,
    show=True
)

# 方法2：手动控制
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor
import cv2

# 加载图像
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 提取关键点
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
keypoints = extractor.extract_keypoints(image_rgb)

# 可视化
visualizer = KeypointVisualizer(show_face=False)
vis_image = visualizer.visualize(image, keypoints, output_path="output.jpg", show=True)
```

#### 可视化视频

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_visualizer import visualize_keypoints_from_video

visualize_keypoints_from_video(
    video_path="video.mp4",
    output_path="output.mp4",
    cache_dir=".cache",
    show_face=False,
    frame_interval=1,  # 每隔1帧处理一次
    max_frames=None     # 处理所有帧
)
```

## 参数说明

### KeypointVisualizer类参数

- `show_face` (bool): 是否显示面部关键点（468个点），默认False
- `show_face_mesh` (bool): 是否显示面部网格（仅轮廓），默认False
- `point_radius` (int): 关键点半径，默认3
- `line_thickness` (int): 连接线粗细，默认2

### 可视化函数参数

#### visualize_keypoints_from_image

- `image_path` (str): 输入图像路径
- `output_path` (str, optional): 输出图像路径
- `cache_dir` (str): 模型缓存目录，默认".cache"
- `show_face` (bool): 是否显示面部关键点，默认False
- `show` (bool): 是否显示图像，默认True

#### visualize_keypoints_from_video

- `video_path` (str): 输入视频路径
- `output_path` (str, optional): 输出视频路径
- `cache_dir` (str): 模型缓存目录，默认".cache"
- `show_face` (bool): 是否显示面部关键点，默认False
- `frame_interval` (int): 帧间隔（每隔N帧处理一次），默认1
- `max_frames` (int, optional): 最大处理帧数，默认None（处理所有帧）

## 关键点颜色

- **身体关键点**：绿色 (0, 255, 0)
- **左手关键点**：蓝色 (255, 0, 0)
- **右手关键点**：红色 (0, 0, 255)
- **面部关键点**：青色 (255, 255, 0)

## 关键点数量

- **身体关键点**：33个（MediaPipe Pose）
- **手部关键点**：每只手21个（MediaPipe Hand）
- **面部关键点**：468个（MediaPipe Face Mesh）

## 示例

### 示例1：可视化单张图像

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_visualizer import visualize_keypoints_from_image

# 可视化图像，显示身体和手部关键点
visualize_keypoints_from_image(
    image_path="test.jpg",
    output_path="output.jpg",
    show_face=False,
    show=True
)
```

### 示例2：可视化视频（每隔5帧）

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_visualizer import visualize_keypoints_from_video

# 处理视频，每隔5帧处理一次
visualize_keypoints_from_video(
    video_path="test.mp4",
    output_path="output.mp4",
    frame_interval=5,
    max_frames=100
)
```

### 示例3：自定义可视化

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor
from src.temporal_reasoning.keypoint_analysis.keypoint_visualizer import KeypointVisualizer
import cv2

# 加载图像
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 提取关键点
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
keypoints = extractor.extract_keypoints(image_rgb)

# 创建可视化器（自定义参数）
visualizer = KeypointVisualizer(
    show_face=True,          # 显示面部关键点
    show_face_mesh=False,    # 不显示面部网格
    point_radius=5,          # 关键点半径5
    line_thickness=3         # 连接线粗细3
)

# 可视化
vis_image = visualizer.visualize(
    image=image,
    keypoints=keypoints,
    output_path="output.jpg",
    show=True
)
```

## 注意事项

1. **模型文件**：确保MediaPipe模型文件已下载到缓存目录（`.cache/mediapipe/`）
2. **图像格式**：支持常见图像格式（jpg, png, bmp等）
3. **视频格式**：支持常见视频格式（mp4, avi, mov等）
4. **性能**：处理视频时，建议使用`frame_interval`参数减少处理帧数
5. **面部关键点**：468个面部关键点较多，默认不显示，需要时使用`--show-face`参数

## 故障排除

### 问题1：模型文件未找到

**错误信息**：
```
错误：未找到MediaPipe模型文件
```

**解决方案**：
1. 确保模型文件位于`.cache/mediapipe/`目录
2. 检查模型文件格式（.tflite或.binarypb）
3. 参考`MODEL_DOWNLOAD_GUIDE.md`下载模型

### 问题2：无法打开图像/视频

**错误信息**：
```
无法加载图像: xxx.jpg
```

**解决方案**：
1. 检查文件路径是否正确
2. 检查文件格式是否支持
3. 检查文件是否损坏

### 问题3：可视化结果不准确

**可能原因**：
1. 图像中人物太小或模糊
2. 人物被遮挡
3. 光照条件不佳

**解决方案**：
1. 使用高质量图像
2. 确保人物清晰可见
3. 调整图像亮度和对比度

## 输出示例

可视化后的图像将包含：
- 身体骨架（绿色线条）
- 手部关键点和连接（蓝色/红色）
- 面部关键点（如果启用，青色）

所有关键点都会用圆圈标记，连接线用相应颜色的线条表示。

