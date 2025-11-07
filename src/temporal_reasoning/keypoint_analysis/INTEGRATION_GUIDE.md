# 关键点时序一致性评估集成指南

## 概述

本指南说明如何将关键点检测接入时序一致性评估系统，实现眨眼、嘴型、手势等生理动作的自动分析。

## 文件结构

```
keypoint_analysis/
├── keypoint_extractor.py              # 关键点提取器（已实现）
├── keypoint_analyzer.py               # 关键点分析器（需要更新）
├── keypoint_visualizer.py             # 关键点可视化（已实现）
├── physiological_metrics.py           # 生理指标计算模块（新增）
├── keypoint_analyzer_impl_example.py  # 完整实现示例（新增）
├── PHYSIOLOGICAL_ANALYSIS_IMPLEMENTATION.md  # 实现方案文档（新增）
└── INTEGRATION_GUIDE.md               # 本文档（新增）
```

## 快速开始

### 方案1：直接替换 keypoint_analyzer.py

最简单的方法是将 `keypoint_analyzer_impl_example.py` 的内容复制到 `keypoint_analyzer.py`，并将类名改回 `KeypointAnalyzer`。

```bash
# 备份原文件
cp keypoint_analyzer.py keypoint_analyzer.py.backup

# 复制新实现
cp keypoint_analyzer_impl_example.py keypoint_analyzer.py

# 修改类名
sed -i 's/KeypointAnalyzerWithMetrics/KeypointAnalyzer/g' keypoint_analyzer.py
```

### 方案2：逐步集成

如果希望保留原有代码结构，可以逐步集成：

#### 步骤1：导入新模块

在 `keypoint_analyzer.py` 开头添加：

```python
from .physiological_metrics import PhysiologicalMetrics, AnomalyBuilder
```

#### 步骤2：初始化指标计算器

在 `__init__` 方法中添加：

```python
self.metrics = PhysiologicalMetrics()
```

#### 步骤3：替换占位符方法

将以下方法从 `keypoint_analyzer_impl_example.py` 复制到 `keypoint_analyzer.py`：

- `_analyze_blink_pattern()`
- `_analyze_mouth_pattern()`
- `_analyze_hand_gesture()`

## 关键点数据结构

MediaPipe Holistic 提供的关键点数据：

```python
keypoints = {
    'body': np.ndarray,      # 身体关键点 (33, 3) - [x, y, z]
    'left_hand': np.ndarray,  # 左手关键点 (21, 3) - [x, y, z]
    'right_hand': np.ndarray, # 右手关键点 (21, 3) - [x, y, z]
    'face': np.ndarray        # 面部关键点 (468, 3) - [x, y, z]
}
```

## 实现的功能

### 1. 眨眼模式分析

**指标**：

- 眼睛纵横比（EAR - Eye Aspect Ratio）
- 眨眼频率（次/分钟）
- 眨眼持续时间（毫秒）

**异常检测**：

- 眨眼频率过低（< 5次/分钟）
- 眨眼频率过高（> 30次/分钟）
- 眨眼持续时间过长（> 500ms）

**示例输出**：

```python
anomaly = {
    'type': 'high_blink_rate',
    'frame_id': 0,
    'timestamp': '0.00s',
    'blink_rate': 35.2,  # 次/分钟
    'severity': 'medium',
    'confidence': 0.8,
    'description': '眨眼频率过高: 35.2次/分钟'
}
```

### 2. 嘴型模式分析

**指标**：

- 嘴部纵横比（MAR - Mouth Aspect Ratio）
- 嘴型变化连续性
- 张嘴持续时间

**异常检测**：

- 嘴型变化不连续（跳跃）
- 嘴部持续张开（> 3秒）

**示例输出**：

```python
anomaly = {
    'type': 'mouth_discontinuity',
    'frame_id': 125,
    'timestamp': '4.17s',
    'mar_jump': 0.35,
    'severity': 'medium',
    'confidence': 0.8,
    'description': '嘴型变化不连续: MAR跳跃 0.35'
}
```

### 3. 手势分析

**指标**：

- 手部运动速度
- 手部位置稳定性（抖动）
- 手部可见性变化

**异常检测**：

- 手部运动速度突变
- 手部抖动过大
- 手部突然消失/出现

**示例输出**：

```python
anomaly = {
    'type': 'hand_velocity_jump',
    'hand': 'left_hand',
    'frame_id': 89,
    'timestamp': '2.97s',
    'velocity_jump': 0.42,
    'severity': 'medium',
    'confidence': 0.75,
    'description': 'left_hand运动速度突变: 0.42'
}
```

## 使用示例

### Python API

```python
from src.temporal_reasoning.core.config import KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer

# 创建配置
config = KeypointConfig(
    model_type="mediapipe",
    enable_visualization=True
)

# 创建分析器
analyzer = KeypointAnalyzer(config)
analyzer.initialize()

# 分析视频
video_frames = load_video_frames("test.mp4")
fps = 30.0

score, anomalies = analyzer.analyze(
    video_frames=video_frames,
    fps=fps,
    video_path="test.mp4"
)

print(f"生理动作自然性得分: {score:.3f}")
print(f"检测到 {len(anomalies)} 个异常")

for anomaly in anomalies:
    print(f"  - [{anomaly['severity']}] {anomaly['type']}: {anomaly['description']}")
```

### 命令行

```bash
python scripts/temporal_reasoning/run_analysis.py \
    --video test.mp4 \
    --enable-keypoint-visualization \
    --output results.json
```

## 参数调整

所有分析参数都在 `PhysiologicalMetrics.PARAMS` 中定义，可以根据需要调整：

```python
# 在 physiological_metrics.py 中
PARAMS = {
    # 眨眼参数
    'ear_threshold': 0.25,         # EAR阈值
    'min_blink_frames': 2,         # 最少闭眼帧数
    'max_blink_frames_ratio': 0.5, # 最多闭眼帧数比例（秒）
    'normal_blink_rate_min': 5/60, # 最小眨眼频率（次/秒）
    'normal_blink_rate_max': 30/60,# 最大眨眼频率（次/秒）
  
    # 嘴型参数
    'mar_threshold': 0.5,          # 嘴部开合阈值
    'mar_jump_threshold': 0.3,     # MAR跳跃阈值
    'max_open_duration_s': 3.0,    # 最大持续张嘴时间（秒）
  
    # 手势参数
    'velocity_threshold': 0.3,     # 速度突变阈值
    'jitter_threshold': 0.05,      # 抖动阈值
    'window_size': 5,              # 滑动窗口大小
}
```

## 测试和验证

### 单元测试

创建测试文件 `test_physiological_metrics.py`：

```python
import numpy as np
from src.temporal_reasoning.keypoint_analysis.physiological_metrics import PhysiologicalMetrics

def test_eye_aspect_ratio():
    """测试EAR计算"""
    metrics = PhysiologicalMetrics()
  
    # 模拟面部关键点
    face_landmarks = np.random.rand(468, 3)
  
    # 计算左眼EAR
    ear = metrics.compute_eye_aspect_ratio(
        face_landmarks,
        metrics.FACE_LANDMARKS['left_eye']
    )
  
    assert ear is not None
    assert 0 <= ear <= 1.0
    print(f"左眼EAR: {ear:.3f}")

def test_blink_detection():
    """测试眨眼检测"""
    metrics = PhysiologicalMetrics()
  
    # 模拟EAR序列（包含一次眨眼）
    ear_sequence = [0.3] * 10 + [0.2, 0.15, 0.2] + [0.3] * 10
    fps = 30.0
  
    blinks = metrics.detect_blinks(ear_sequence, fps)
  
    assert len(blinks) > 0
    print(f"检测到 {len(blinks)} 次眨眼")

if __name__ == '__main__':
    test_eye_aspect_ratio()
    test_blink_detection()
```

### 端到端测试

```bash
# 测试单个视频
python scripts/temporal_reasoning/run_analysis.py \
    --video data/test_videos/blink_normal.mp4 \
    --enable-keypoint-visualization \
    --output results/blink_normal_result.json

# 查看结果
cat results/blink_normal_result.json | jq '.physiological_score'
```

## 调试技巧

### 1. 可视化EAR/MAR曲线

添加到 `_analyze_blink_pattern` 方法：

```python
# 保存EAR序列用于可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(ear_sequence)
plt.axhline(y=params['ear_threshold'], color='r', linestyle='--', label='阈值')
plt.xlabel('帧')
plt.ylabel('EAR')
plt.title('眼睛纵横比（EAR）时序变化')
plt.legend()
plt.savefig('ear_sequence.png')
```

### 2. 打印详细日志

在关键位置添加日志：

```python
print(f"帧{frame_idx}: EAR={ear:.3f}, MAR={mar:.3f}")
```

### 3. 导出关键点数据

```python
# 保存关键点序列
import json

keypoints_data = []
for frame_idx, kp in enumerate(keypoint_sequences):
    keypoints_data.append({
        'frame': frame_idx,
        'has_face': kp['face'] is not None,
        'has_left_hand': kp['left_hand'] is not None,
        'has_right_hand': kp['right_hand'] is not None,
    })

with open('keypoints_stats.json', 'w') as f:
    json.dump(keypoints_data, f, indent=2)
```

## 常见问题

### Q1: 为什么所有得分都是1.0？

**A**: 可能是因为：

1. 没有替换占位符方法，仍在使用返回 `1.0` 的简化实现
2. 视频太短，无法检测到有效的模式
3. 阈值设置不合理，无法检测到异常

### Q2: 如何调整阈值？

**A**: 在 `physiological_metrics.py` 中修改 `PARAMS` 字典，或者在运行时传入自定义参数：

```python
custom_params = PhysiologicalMetrics.PARAMS.copy()
custom_params['ear_threshold'] = 0.20  # 降低阈值，更容易检测闭眼

blinks = metrics.detect_blinks(ear_sequence, fps, custom_params)
```

### Q3: 如何处理检测不到面部/手部的情况？

**A**: 代码已经处理了这种情况，会跳过 `None` 值：

```python
if keypoints['face'] is None:
    ear_sequence.append(None)
    continue
```

### Q4: 如何提高检测准确性？

**A**:

1. 确保视频质量良好，光照充足
2. 确保人物正面面向摄像头
3. 调整MediaPipe检测参数
4. 使用更长的视频进行分析
5. 根据实际数据调整阈值

## 下一步

1. **实现高级特征**：

   - 双眼不同步检测
   - 嘴型与音频同步分析
   - 手势识别（特定手势检测）
2. **优化性能**：

   - 并行处理多个分析任务
   - 批量计算关键点
   - 缓存中间结果
3. **增强可视化**：

   - 实时显示EAR/MAR曲线
   - 在视频上标注异常帧
   - 生成分析报告

## 总结

通过本指南，您可以：

1. ? 理解关键点时序一致性评估的原理
2. ? 集成眨眼、嘴型、手势分析功能
3. ? 使用完整的实现替换占位符代码
4. ? 调整参数以适应不同场景
5. ? 测试和验证实现效果

所有代码都已准备就绪，只需按照指南进行集成即可！
