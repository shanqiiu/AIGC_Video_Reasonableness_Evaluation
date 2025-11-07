# 通过run_analysis.py配置关键点可视化

## 概述

`run_analysis.py` 已集成关键点可视化功能，可以通过命令行参数配置是否启用可视化。

## 命令行参数

### 关键点可视化参数

```bash
# 启用关键点可视化
--enable-keypoint-visualization

# 指定可视化输出目录
--keypoint-visualization-dir <目录路径>

# 显示面部关键点（468个点）
--show-face-keypoints

# 显示GUI窗口（第一帧预览）
--show-keypoint-visualization
```

## 使用示例

### 示例1：基本使用（启用可视化）

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video path/to/video.mp4 \
    --enable-keypoint-visualization
```

**效果**：
- 启用关键点可视化
- 可视化结果保存到 `outputs/keypoint_visualization/{视频名}_keypoints.mp4`

### 示例2：指定可视化输出目录

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video path/to/video.mp4 \
    --enable-keypoint-visualization \
    --keypoint-visualization-dir outputs/my_visualization
```

**效果**：
- 启用关键点可视化
- 可视化结果保存到 `outputs/my_visualization/{视频名}_keypoints.mp4`

### 示例3：显示面部关键点

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video path/to/video.mp4 \
    --enable-keypoint-visualization \
    --show-face-keypoints
```

**效果**：
- 启用关键点可视化
- 显示面部关键点（468个点）
- 可视化结果保存到默认目录

### 示例4：显示GUI窗口预览

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video path/to/video.mp4 \
    --enable-keypoint-visualization \
    --show-keypoint-visualization
```

**效果**：
- 启用关键点可视化
- 保存可视化结果
- 显示第一帧的GUI窗口预览

### 示例5：完整配置

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video path/to/video.mp4 \
    --enable-keypoint-visualization \
    --keypoint-visualization-dir outputs/keypoint_viz \
    --show-face-keypoints \
    --show-keypoint-visualization \
    --output results.json
```

**效果**：
- 启用关键点可视化
- 显示面部关键点
- 可视化结果保存到 `outputs/keypoint_viz/`
- 显示GUI窗口预览
- 分析结果保存到 `results.json`

## 参数说明

### --enable-keypoint-visualization

- **类型**: 标志（action='store_true'）
- **默认值**: False
- **说明**: 启用关键点可视化功能
- **作用**: 设置 `config.keypoint.enable_visualization = True`

### --keypoint-visualization-dir

- **类型**: 字符串
- **默认值**: None
- **说明**: 关键点可视化输出目录
- **作用**: 设置 `config.keypoint.visualization_output_dir`
- **默认行为**: 如果未指定，使用 `outputs/keypoint_visualization/`

### --show-face-keypoints

- **类型**: 标志（action='store_true'）
- **默认值**: False
- **说明**: 显示面部关键点（468个点）
- **作用**: 设置 `config.keypoint.show_face = True`
- **注意**: 面部关键点较多，可能影响可视化效果

### --show-keypoint-visualization

- **类型**: 标志（action='store_true'）
- **默认值**: False
- **说明**: 显示关键点可视化GUI窗口（第一帧）
- **作用**: 设置 `config.keypoint.show_visualization = True`
- **注意**: 仅显示第一帧，用于预览

## 配置文件方式

也可以通过配置文件启用可视化：

```yaml
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

然后使用配置文件：

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video path/to/video.mp4 \
    --config config.yaml
```

## 输出文件

### 可视化输出

- **位置**: `keypoint-visualization-dir` 或默认 `outputs/keypoint_visualization/`
- **文件名**: `{视频名}_keypoints.mp4`
- **格式**: MP4视频文件

### 分析结果输出

- **位置**: `--output` 参数指定或默认 `outputs/temporal_reasoning/`
- **文件名**: `{视频名}_result.json`
- **格式**: JSON文件

## 工作流程

1. **解析参数**: `run_analysis.py` 解析命令行参数
2. **更新配置**: 根据参数更新 `config.keypoint` 配置
3. **初始化分析器**: `TemporalReasoningAnalyzer` 初始化 `KeypointAnalyzer`
4. **执行分析**: 调用 `keypoint_analyzer.analyze()`，传入 `video_path`
5. **生成可视化**: 如果启用可视化，自动生成可视化视频
6. **保存结果**: 保存分析结果和可视化结果

## 注意事项

1. **性能影响**: 启用可视化会增加处理时间和内存占用
2. **输出目录**: 确保输出目录有写入权限
3. **视频路径**: 建议传入视频路径，用于生成有意义的输出文件名
4. **面部关键点**: 468个面部关键点较多，建议仅在需要时启用

## 完整示例

```bash
# 完整分析，启用关键点可视化
python scripts/temporal_reasoning/run_analysis.py \
    --video data/videos/test.mp4 \
    --enable-keypoint-visualization \
    --keypoint-visualization-dir outputs/keypoint_viz \
    --show-face-keypoints \
    --show-keypoint-visualization \
    --output outputs/results/test_result.json \
    --device cuda:0
```

**输出**：
- 分析结果: `outputs/results/test_result.json`
- 可视化视频: `outputs/keypoint_viz/test_keypoints.mp4`
- GUI窗口: 显示第一帧预览

## 总结

通过 `run_analysis.py` 可以轻松配置关键点可视化：
- ? **启用/禁用**: `--enable-keypoint-visualization`
- ? **输出目录**: `--keypoint-visualization-dir`
- ? **显示选项**: `--show-face-keypoints`, `--show-keypoint-visualization`
- ? **完全集成**: 与整个分析流程无缝集成

可视化功能已完全集成到 `run_analysis.py` 中，可以通过命令行参数轻松配置。

