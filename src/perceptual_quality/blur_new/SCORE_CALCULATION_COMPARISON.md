# 模糊度计算方式对比

## 两种计算方式的区别

### 方式1: 原始 `motion_smoothness_score.py` 的 main 函数

**计算逻辑**：
```python
# 1. 计算质量分数
_, _, scores = scorer(load_video_sliding_window(video_file, window_size))
scores = scores.tolist()

# 2. 设置阈值
threshold = set_threshold(meta_info['perceptible_amplitude_score'])

# 3. 检测异常帧
artifacts_frames = get_artifacts_frames(scores, threshold)

# 4. 计算最终分数：正常帧占总帧数的比例
final_score = (1 - len(artifacts_frames) / len(scores))
```

**输出结果**：
- **单一分数**：`motion_smoothness_score`（运动平滑度分数）
- **范围**：0.0 - 1.0
- **含义**：
  - 1.0 = 所有帧都正常（无模糊）
  - 0.0 = 所有帧都有问题（全部模糊）
  - 0.95 = 95% 的帧正常，5% 的帧模糊

**特点**：
- ? 简单直接，输出单一分数
- ? 适合批量处理和元信息更新
- ? 缺乏详细指标
- ? 不能区分模糊严重程度

---

### 方式2: 当前 `simple_blur_detector.py` 的计算方式

**计算逻辑**：
```python
# 1. 计算质量分数
_, _, quality_scores = self.scorer(video_frames, batch_size=batch_size)
quality_scores = quality_scores.tolist()

# 2. 估算相机运动幅度（自己计算）
camera_movement = self._estimate_camera_movement(video_path)

# 3. 计算自适应阈值
threshold = calculate_adaptive_threshold(camera_movement)

# 4. 检测模糊帧
blur_frame_indices = detect_artifact_frames(quality_scores, threshold)

# 5. 计算多种模糊指标
blur_metrics = self._calculate_blur_metrics(quality_scores, blur_frame_indices, threshold)
# 包括：
# - blur_ratio: 模糊帧比例
# - avg_quality: 平均质量分数
# - quality_std: 质量分数标准差
# - max_quality_drop: 最大质量下降
# - blur_severity: 模糊严重程度
# - confidence: 检测置信度
# - blur_detected: 是否检测到模糊
```

**输出结果**：
- **多种指标**：包含模糊检测的详细信息
- **主要指标**：
  - `blur_ratio`: 模糊帧比例（0.0 - 1.0）
  - `blur_severity`: 模糊严重程度（"无模糊"/"轻微模糊"/"中等模糊"/"严重模糊"）
  - `confidence`: 检测置信度（0.0 - 1.0）
  - `blur_detected`: 布尔值，是否检测到模糊
  - `avg_quality`: 平均质量分数
  - `max_quality_drop`: 最大质量下降

**特点**：
- ? 提供详细的模糊检测信息
- ? 可以区分模糊严重程度
- ? 提供置信度评估
- ? 输出更复杂，不是单一分数

---

## 关键区别对比

| 维度 | 原始 main 函数 | 当前 simple_blur_detector |
|------|---------------|--------------------------|
| **输出类型** | 单一分数 | 多指标字典 |
| **主要输出** | `motion_smoothness_score` | `blur_ratio`, `blur_severity`, `confidence` |
| **分数含义** | 正常帧比例 | 模糊帧比例 |
| **相机运动估算** | 从元信息读取 | 自己计算（前10帧） |
| **阈值来源** | `meta_info['perceptible_amplitude_score']` | `_estimate_camera_movement()` |
| **严重程度** | 无 | 有（4个等级） |
| **置信度** | 无 | 有 |
| **用途** | 批量评分 | 详细检测分析 |

---

## 分数关系

### 原始方式
```
motion_smoothness_score = 1 - (模糊帧数 / 总帧数)
```

### 当前方式
```
blur_ratio = 模糊帧数 / 总帧数
```

**关系**：
```
motion_smoothness_score = 1 - blur_ratio
```

**示例**：
- 如果 `blur_ratio = 0.05`（5% 模糊帧）
- 则 `motion_smoothness_score = 1 - 0.05 = 0.95`（95% 正常帧）

---

## 相机运动估算的区别

### 原始方式
```python
# 从元信息中读取已有的相机运动分数
threshold = set_threshold(meta_info['perceptible_amplitude_score'])
```
- **依赖外部数据**：需要预先计算 `perceptible_amplitude_score`
- **优点**：使用更准确的运动估算方法
- **缺点**：需要额外的元信息文件

### 当前方式
```python
# 自己估算相机运动（读取前10帧）
camera_movement = self._estimate_camera_movement(video_path)
threshold = calculate_adaptive_threshold(camera_movement)
```
- **自包含**：不依赖外部数据
- **优点**：独立运行，不需要元信息
- **缺点**：估算可能不如专业的运动估算方法准确

---

## 计算流程对比

### 原始方式流程
```
1. 加载视频帧（滑动窗口）
2. 计算质量分数
3. 从元信息读取相机运动 → 计算阈值
4. 检测异常帧
5. 计算运动平滑度分数 = 1 - 模糊帧比例
6. 保存到元信息文件
```

### 当前方式流程
```
1. 加载视频帧（滑动窗口）
2. 计算质量分数（批处理）
3. 自己估算相机运动 → 计算阈值
4. 检测模糊帧
5. 计算多种模糊指标：
   - blur_ratio（模糊帧比例）
   - blur_severity（严重程度）
   - confidence（置信度）
   - avg_quality（平均质量）
   - max_quality_drop（最大质量下降）
6. 生成详细检测结果
```

---

## 使用场景建议

### 使用原始方式（单一分数）
- ? 批量处理大量视频
- ? 只需要运动平滑度分数
- ? 已有 `perceptible_amplitude_score` 元信息
- ? 需要快速评估

### 使用当前方式（详细检测）
- ? 需要详细的模糊分析
- ? 需要区分模糊严重程度
- ? 需要置信度评估
- ? 独立运行，不依赖元信息
- ? 需要可视化结果

---

## 总结

**核心区别**：

1. **输出格式**：
   - 原始：单一分数（`motion_smoothness_score`）
   - 当前：多指标字典（`blur_ratio`, `blur_severity`, `confidence` 等）

2. **分数含义**：
   - 原始：正常帧比例（越高越好）
   - 当前：模糊帧比例（越低越好）

3. **相机运动估算**：
   - 原始：从元信息读取
   - 当前：自己计算（前10帧）

4. **详细信息**：
   - 原始：无严重程度和置信度
   - 当前：有严重程度和置信度

5. **依赖关系**：
   - 原始：依赖元信息文件
   - 当前：独立运行

**数值关系**：
```
motion_smoothness_score = 1 - blur_ratio
```

如果需要将当前方式的输出转换为原始方式的分数：
```python
motion_smoothness_score = 1 - result['blur_ratio']
```

