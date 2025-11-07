# 关键点可视化配置指南

## 概述

关键点可视化功能已集成到整个模块中，可以通过配置参数启用和自定义。

## 配置参数

### KeypointConfig 可视化参数

在 `KeypointConfig` 中添加了以下可视化相关参数：

```python
@dataclass
class KeypointConfig:
    # ... 其他参数 ...
    
    # 可视化配置
    enable_visualization: bool = False  # 是否启用可视化
    visualization_output_dir: Optional[str] = None  # 可视化输出目录
    show_face: bool = False  # 是否显示面部关键点
    show_face_mesh: bool = False  # 是否显示面部网格
    point_radius: int = 3  # 关键点半径
    line_thickness: int = 2  # 连接线粗细
    save_visualization: bool = True  # 是否保存可视化结果
    show_visualization: bool = False  # 是否显示可视化结果（GUI）
```

## 使用方法

### 方法1：通过配置启用可视化

```python
from src.temporal_reasoning.core.config import KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer

# 创建配置，启用可视化
config = KeypointConfig(
    model_type="mediapipe",
    enable_visualization=True,  # 启用可视化
    visualization_output_dir="outputs/keypoint_visualization",  # 输出目录
    show_face=False,  # 不显示面部关键点
    save_visualization=True,  # 保存可视化结果
    show_visualization=False  # 不显示GUI窗口
)

# 初始化分析器
analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频（会自动生成可视化结果）
video_frames = [...]  # 视频帧列表
fps = 30.0
video_path = "test.mp4"  # 视频路径（用于生成输出文件名）

physiological_score, anomalies = analyzer.analyze(
    video_frames=video_frames,
    fps=fps,
    video_path=video_path  # 传入视频路径
)
```

### 方法2：通过配置文件启用

```python
# config.yaml
keypoint:
  model_type: "mediapipe"
  enable_visualization: true
  visualization_output_dir: "outputs/keypoint_visualization"
  show_face: false
  point_radius: 3
  line_thickness: 2
  save_visualization: true
  show_visualization: false
```

### 方法3：动态启用可视化

```python
from src.temporal_reasoning.core.config import KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer

# 创建配置
config = KeypointConfig(model_type="mediapipe")

# 动态启用可视化
config.enable_visualization = True
config.visualization_output_dir = "outputs/keypoint_visualization"
config.show_face = True  # 显示面部关键点

# 初始化分析器
analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频
physiological_score, anomalies = analyzer.analyze(
    video_frames=video_frames,
    fps=fps,
    video_path=video_path
)
```

## 参数说明

### enable_visualization (bool)

- **默认值**: `False`
- **说明**: 是否启用可视化功能
- **作用**: 控制是否在分析过程中生成可视化结果

### visualization_output_dir (Optional[str])

- **默认值**: `None`
- **说明**: 可视化输出目录
- **作用**: 指定可视化结果的保存目录
- **默认行为**: 如果为 `None`，使用 `outputs/keypoint_visualization/`

### show_face (bool)

- **默认值**: `False`
- **说明**: 是否显示面部关键点（468个点）
- **作用**: 控制是否在可视化中显示面部关键点
- **注意**: 面部关键点较多，可能影响可视化效果

### show_face_mesh (bool)

- **默认值**: `False`
- **说明**: 是否显示面部网格（仅轮廓）
- **作用**: 控制是否显示面部网格轮廓
- **注意**: 比 `show_face` 更轻量，仅显示轮廓

### point_radius (int)

- **默认值**: `3`
- **说明**: 关键点半径（像素）
- **作用**: 控制关键点圆圈的大小

### line_thickness (int)

- **默认值**: `2`
- **说明**: 连接线粗细（像素）
- **作用**: 控制骨架连接线的粗细

### save_visualization (bool)

- **默认值**: `True`
- **说明**: 是否保存可视化结果
- **作用**: 控制是否将可视化结果保存为视频文件

### show_visualization (bool)

- **默认值**: `False`
- **说明**: 是否显示可视化结果（GUI窗口）
- **作用**: 控制是否在GUI窗口中显示第一帧的可视化结果
- **注意**: 仅显示第一帧，用于预览

## 输出文件

### 输出位置

- **自定义目录**: 如果设置了 `visualization_output_dir`，保存到指定目录
- **默认目录**: `outputs/keypoint_visualization/`

### 输出文件名

- **有视频路径**: `{视频名}_keypoints.mp4`
- **无视频路径**: `keypoints_{时间戳}.mp4`

### 输出格式

- **视频格式**: MP4
- **编码**: MP4V
- **帧率**: 与输入视频相同

## 示例

### 示例1：基本使用（启用可视化）

```python
from src.temporal_reasoning.core.config import KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer

# 创建配置
config = KeypointConfig(
    model_type="mediapipe",
    enable_visualization=True,  # 启用可视化
    visualization_output_dir="outputs/keypoint_visualization"
)

# 初始化分析器
analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频
video_frames = load_video_frames("test.mp4")
fps = 30.0

physiological_score, anomalies = analyzer.analyze(
    video_frames=video_frames,
    fps=fps,
    video_path="test.mp4"
)

# 可视化结果会自动保存到 outputs/keypoint_visualization/test_keypoints.mp4
```

### 示例2：显示面部关键点

```python
config = KeypointConfig(
    model_type="mediapipe",
    enable_visualization=True,
    show_face=True,  # 显示面部关键点
    point_radius=5,  # 增大关键点半径
    line_thickness=3  # 增大连接线粗细
)

analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频...
```

### 示例3：仅保存，不显示GUI

```python
config = KeypointConfig(
    model_type="mediapipe",
    enable_visualization=True,
    save_visualization=True,  # 保存可视化结果
    show_visualization=False  # 不显示GUI窗口
)

analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频...
```

### 示例4：显示GUI预览

```python
config = KeypointConfig(
    model_type="mediapipe",
    enable_visualization=True,
    save_visualization=True,
    show_visualization=True  # 显示GUI窗口（第一帧）
)

analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频...
# 会显示第一帧的可视化结果
```

## 工作流程

1. **初始化阶段**：
   - 如果 `enable_visualization=True`，创建 `KeypointVisualizer` 实例
   - 根据配置参数设置可视化器参数

2. **分析阶段**：
   - 提取关键点时，如果启用可视化，对每一帧进行可视化
   - 将可视化后的帧保存到 `visualized_frames` 列表

3. **保存阶段**：
   - 分析完成后，如果启用可视化且有可视化帧，保存为视频文件
   - 如果 `show_visualization=True`，显示第一帧的GUI窗口

## 性能考虑

### 启用可视化的影响

- **内存占用**: 需要存储可视化后的帧，内存占用增加
- **处理时间**: 每帧需要额外的可视化处理时间
- **磁盘空间**: 需要保存可视化视频文件

### 优化建议

1. **仅保存，不显示GUI**: 设置 `show_visualization=False`
2. **不显示面部关键点**: 设置 `show_face=False`（默认）
3. **降低关键点半径**: 设置较小的 `point_radius`
4. **降低连接线粗细**: 设置较小的 `line_thickness`

## 注意事项

1. **视频路径**: 建议传入 `video_path` 参数，用于生成有意义的输出文件名
2. **输出目录**: 确保输出目录有写入权限
3. **内存管理**: 处理长视频时，注意内存占用
4. **性能**: 启用可视化会增加处理时间，建议仅在需要时启用

## 总结

通过配置参数，可以灵活控制关键点可视化功能：
- ? **启用/禁用**: `enable_visualization`
- ? **输出位置**: `visualization_output_dir`
- ? **显示选项**: `show_face`, `show_face_mesh`
- ? **样式设置**: `point_radius`, `line_thickness`
- ? **保存/显示**: `save_visualization`, `show_visualization`

可视化功能已完全集成到模块中，可以通过配置轻松启用和自定义。

