# 默认文本提示使用指南

## 概述

实例追踪模块已添加默认文本提示支持，当用户未提供文本提示时，系统会自动使用默认提示进行检测。

## 默认文本提示

```python
DEFAULT_TEXT_PROMPTS = [
    "person",      # 人（最常见的检测目标）
    "face",        # 脸部
    "hand",        # 手部
    "body",        # 身体
    "object"       # 通用物体
]
```

### 为什么选择这些提示？

1. **person**: 视频中最常见的主体，适用于人物视频
2. **face**: 检测人脸，用于面部特征追踪
3. **hand**: 检测手部，用于手势分析
4. **body**: 检测身体部位
5. **object**: 通用物体检测，兜底方案

## 使用方法

### 方法1：不提供文本提示（使用默认）

```bash
# 命令行
python scripts/temporal_reasoning/run_analysis.py \
    --video test.mp4
```

```python
# Python API
from src.temporal_reasoning.core.temporal_analyzer import TemporalReasoningAnalyzer

analyzer = TemporalReasoningAnalyzer(config)
analyzer.initialize()

# 不提供text_prompts，自动使用默认提示
result = analyzer.analyze(
    video_frames=video_frames,
    fps=30.0
)
```

**输出示例**：
```
正在分析结构稳定性...
?? 使用默认文本提示: person, face, hand, body, object
正在检测实例...
...
```

### 方法2：提供自定义文本提示

```bash
# 命令行
python scripts/temporal_reasoning/run_analysis.py \
    --video test.mp4 \
    --prompts "person" "car"
```

```python
# Python API
result = analyzer.analyze(
    video_frames=video_frames,
    text_prompts=["person", "car"],  # 自定义提示
    fps=30.0
)
```

**输出示例**：
```
正在分析结构稳定性...
?? 使用自定义文本提示: person, car
正在检测实例...
...
```

### 方法3：禁用默认提示

```python
# Python API（高级用法）
from src.temporal_reasoning.instance_tracking.instance_analyzer import InstanceTrackingAnalyzer

instance_analyzer = InstanceTrackingAnalyzer(...)
instance_analyzer.initialize()

# 显式禁用默认提示
structure_score, anomalies = instance_analyzer.analyze(
    video_frames=video_frames,
    text_prompts=None,
    fps=30.0,
    use_default_prompts=False  # 禁用默认提示
)
```

**输出示例**：
```
正在分析结构稳定性...
?? 警告: 未提供文本提示，跳过实例检测
结构稳定性得分: 1.000
检测到 0 个结构异常
```

## 行为对比

### 修改前

| 场景 | text_prompts | 行为 |
|------|-------------|------|
| 未提供 | None | ?? 跳过检测，返回1.0 |
| 空列表 | [] | ?? 跳过检测，返回1.0 |
| 有提示 | ["person"] | ? 正常检测 |

### 修改后

| 场景 | text_prompts | use_default_prompts | 行为 |
|------|-------------|---------------------|------|
| 未提供 | None | True（默认） | ? 使用默认提示检测 |
| 未提供 | None | False | ?? 跳过检测，返回1.0 |
| 空列表 | [] | True（默认） | ? 使用默认提示检测 |
| 空列表 | [] | False | ?? 跳过检测，返回1.0 |
| 有提示 | ["person"] | 任意 | ? 使用自定义提示检测 |

## 使用场景

### 场景1：人物视频（默认提示适用）

```bash
# 默认提示包含person, face, hand，非常适合人物视频
python run_analysis.py --video person_video.mp4
```

### 场景2：车辆视频（需要自定义提示）

```bash
# 自定义提示，检测车辆和道路
python run_analysis.py --video traffic_video.mp4 \
    --prompts "car" "vehicle" "road" "traffic light"
```

### 场景3：动物视频

```bash
# 自定义提示，检测动物
python run_analysis.py --video animal_video.mp4 \
    --prompts "dog" "cat" "animal"
```

### 场景4：混合场景

```bash
# 自定义提示，检测多种物体
python run_analysis.py --video mixed_video.mp4 \
    --prompts "person" "car" "building" "tree"
```

## 自定义默认提示

如果需要修改默认提示，编辑 `instance_analyzer.py`：

```python
# 在文件开头修改
DEFAULT_TEXT_PROMPTS = [
    "person",      # 必须：人物检测
    "vehicle",     # 可选：车辆检测
    "building",    # 可选：建筑检测
    # ... 添加你需要的其他提示
]
```

**注意事项**：
1. 提示越多，检测时间越长
2. 建议保留 "person" 和 "object"（最通用）
3. 根据实际应用场景调整

## 配置参数（未来扩展）

未来可以将默认提示添加到配置文件中：

```yaml
# config.yaml
instance_tracking:
  default_text_prompts:
    - person
    - face
    - hand
    - object
  use_default_prompts: true
```

## 性能影响

### 默认提示的性能开销

- **检测目标数量**：5个（person, face, hand, body, object）
- **额外时间**：约5-10%（取决于GPU性能）
- **内存占用**：基本无影响

### 优化建议

如果性能是关键考虑：
1. 减少默认提示数量（如只保留 "person"）
2. 提供精确的自定义提示
3. 禁用默认提示（设置 `use_default_prompts=False`）

## 常见问题

### Q1: 默认提示会检测所有这些物体吗？

**A**: 是的，Grounding DINO会尝试检测所有提示中的物体。如果视频中没有某个物体（如没有"car"），则该提示不会产生检测结果。

### Q2: 默认提示会影响性能吗？

**A**: 会有轻微影响（约5-10%），但可以接受。如果性能关键，建议提供精确的自定义提示。

### Q3: 我可以完全禁用实例追踪吗？

**A**: 可以通过设置 `use_default_prompts=False` 并不提供 `text_prompts` 来跳过实例检测，但更好的方法是在配置中禁用整个模块。

### Q4: 默认提示适用于所有场景吗？

**A**: 默认提示主要针对人物视频。对于特殊场景（如车辆、动物、建筑等），建议提供自定义提示。

## 示例

### 示例1：人物视频（使用默认）

```bash
python run_analysis.py --video person_walking.mp4
# 输出: 使用默认文本提示: person, face, hand, body, object
# 检测: ? person, ? face, ? hand, ? body
```

### 示例2：车辆视频（自定义）

```bash
python run_analysis.py --video traffic.mp4 --prompts "car" "vehicle"
# 输出: 使用自定义文本提示: car, vehicle
# 检测: ? car, ? vehicle
```

### 示例3：混合场景（自定义）

```bash
python run_analysis.py --video street_scene.mp4 \
    --prompts "person" "car" "building" "tree"
# 输出: 使用自定义文本提示: person, car, building, tree
# 检测: ? person, ? car, ? building, ? tree
```

## 总结

通过添加默认文本提示：

1. ? **用户友好**：不需要手动指定提示即可运行
2. ? **向后兼容**：仍支持自定义提示
3. ? **灵活可控**：可以禁用默认提示
4. ? **覆盖常见场景**：默认提示适用于大多数人物视频

**推荐使用方式**：
- 人物视频：不提供提示（使用默认）
- 特殊场景：提供自定义提示
- 性能优先：提供精确的少量提示

