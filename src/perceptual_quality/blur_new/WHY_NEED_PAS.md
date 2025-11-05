# 为什么原始方式的阈值需要 PAS？

## 核心问题

**为什么原始方式需要使用 `perceptible_amplitude_score` (PAS) 来设置阈值？**

---

## 关键理解

### 1. PAS 是什么？

`perceptible_amplitude_score` (PAS) 是**可感知运动幅度分数**，用于衡量视频中的**相机运动幅度**或**场景运动幅度**。

**PAS 的计算方式**（来自 `perceptible_amplitude_score.py`）：
- 使用 Grounding DINO 检测主体
- 使用 SAM 分割主体和背景
- 使用 Co-Tracker 跟踪运动
- 计算背景运动幅度和主体运动幅度
- 输出综合运动幅度（0-1之间）

**PAS 的值**：
- `0.0`：完全静止的场景
- `0.1`：轻微运动
- `0.3`：中等运动
- `0.5`：剧烈运动
- `1.0`：极剧烈运动（如快速旋转）

---

### 2. 为什么相机运动会影响阈值？

#### 问题场景

**场景 A：静止场景**
```
相机固定，物体静止
相邻帧之间几乎没有差异
质量分数应该非常稳定：
[0.95, 0.96, 0.95, 0.96, 0.95]
相邻差异: [0.01, 0.01, 0.01, 0.01]
→ 如果出现 0.02 的差异，可能是模糊！
```

**场景 B：快速运动场景**
```
相机快速旋转，场景快速变化
相邻帧之间有很大差异（这是正常的）
质量分数自然会有波动：
[0.95, 0.92, 0.98, 0.90, 0.97]
相邻差异: [0.03, 0.06, 0.08, 0.07]
→ 0.03 的差异是正常的，不是模糊！
```

#### 核心矛盾

**如果使用固定阈值**：
- 静止场景：阈值 0.025 太高，可能漏检轻微模糊
- 运动场景：阈值 0.025 太低，会误判正常运动为模糊

**解决方案：自适应阈值**
- 根据相机运动幅度动态调整阈值
- 运动越大，阈值越高（允许更大的质量波动）

---

### 3. 阈值设置逻辑

```python
def set_threshold(camera_movement):
    """
    根据相机运动幅度设置阈值
    
    逻辑：
    - 相机运动小 → 阈值低（更敏感，因为正常波动小）
    - 相机运动大 → 阈值高（更宽松，因为正常波动大）
    """
    if camera_movement is None:
        return 0.01  # 默认值（保守）
    
    if camera_movement < 0.1:
        return 0.01      # 静止场景：阈值低，任何波动都可能是模糊
    elif camera_movement < 0.3:
        return 0.015     # 轻微运动：阈值稍高
    elif camera_movement < 0.5:
        return 0.025     # 中等运动：阈值中等
    else:  # camera_movement >= 0.5
        return 0.03      # 剧烈运动：阈值高，允许较大的质量波动
```

**阈值映射表**：

| PAS 值范围 | 场景类型 | 阈值 | 原因 |
|-----------|---------|------|------|
| < 0.1 | 静止场景 | 0.01 | 正常波动很小，任何变化都可能是模糊 |
| 0.1 - 0.3 | 轻微运动 | 0.015 | 有轻微运动，允许小幅度波动 |
| 0.3 - 0.5 | 中等运动 | 0.025 | 中等运动，允许中等幅度波动 |
| >= 0.5 | 剧烈运动 | 0.03 | 剧烈运动，允许大幅度波动 |

---

### 4. 为什么需要 PAS 而不是自己计算？

#### 原始方式（使用 PAS）

```python
# 从元信息读取预先计算的 PAS
threshold = set_threshold(meta_info['perceptible_amplitude_score'])
```

**优点**：
- ? **更准确**：PAS 使用专业的运动估算方法
  - Grounding DINO：检测主体
  - SAM：精确分割
  - Co-Tracker：专业运动跟踪
- ? **区分主体和背景**：可以区分相机运动（背景运动）和物体运动（主体运动）
- ? **归一化**：考虑了视频分辨率，归一化到标准范围

**缺点**：
- ? 需要预先计算 PAS
- ? 依赖元信息文件
- ? 计算 PAS 需要额外的时间和资源

#### 当前方式（自己计算）

```python
# 自己估算相机运动（读取前10帧）
camera_movement = self._estimate_camera_movement(video_path)
threshold = calculate_adaptive_threshold(camera_movement)
```

**优点**：
- ? **独立运行**：不依赖外部数据
- ? **快速**：只需读取前10帧
- ? **简单**：计算逻辑简单

**缺点**：
- ? **精度较低**：只是简单的帧间差异，不能区分相机运动和物体运动
- ? **可能不准确**：前10帧可能不能代表整个视频的运动情况

---

### 5. PAS 的详细计算过程

#### 步骤1：检测主体
```python
# 使用 Grounding DINO 检测视频中的主体
detections = get_grounding_output(
    model, image, "person . object", 
    box_threshold, text_threshold
)
```

#### 步骤2：分割主体和背景
```python
# 使用 SAM 分割主体和背景
subject_mask = sam_predictor.predict(
    point_coords=detections.bboxes,
    point_labels=detections.labels
)
background_mask = ~subject_mask
```

#### 步骤3：跟踪运动
```python
# 使用 Co-Tracker 跟踪运动轨迹
tracks, visibility = cotracker.predict(
    video=video_frames,
    queries=query_points
)
```

#### 步骤4：计算运动幅度
```python
# 分别计算背景运动（相机运动）和主体运动
background_motion = calculate_motion_degree(
    tracks, visibility, background_mask
)
subject_motion = calculate_motion_degree(
    tracks, visibility, subject_mask
)

# 综合运动幅度
total_motion = background_motion + subject_motion
```

#### 步骤5：归一化
```python
# 归一化到 [0, 1] 范围
# 考虑视频分辨率（归一化到 1080p）
normalized_motion = motion / video_diagonal
```

---

### 6. 为什么 PAS 更准确？

#### 专业方法 vs 简单方法

**PAS（专业方法）**：
- ? 区分相机运动和物体运动
- ? 使用专业运动跟踪算法（Co-Tracker）
- ? 精确分割主体和背景
- ? 考虑视频分辨率归一化

**简单方法（当前方式）**：
- ? 只计算帧间差异，无法区分运动来源
- ? 可能将物体运动误判为相机运动
- ? 没有归一化，可能受视频分辨率影响

#### 示例对比

**场景：相机固定，物体快速移动**

**PAS 方法**：
```
背景运动（相机运动）：0.05（很小）
主体运动（物体运动）：0.8（很大）
综合运动：0.05（相机基本不动）
→ 阈值 = 0.01（静止场景，严格检测）
```

**简单方法**：
```
帧间差异：0.8（很大）
→ 阈值 = 0.03（剧烈运动，宽松检测）
→ 误判！应该严格检测，因为相机没动
```

---

### 7. 实际应用场景

#### 场景1：静止场景（PAS < 0.1）
```
场景：相机固定，人物静止
PAS ≈ 0.05
阈值 = 0.01（很严格）
→ 任何质量波动都可能是模糊
→ 能检测到轻微模糊
```

#### 场景2：相机旋转（PAS 0.3-0.5）
```
场景：相机环绕物体旋转
PAS ≈ 0.4（背景运动大）
阈值 = 0.025（中等）
→ 允许中等的质量波动
→ 不会将正常运动误判为模糊
```

#### 场景3：快速运动（PAS >= 0.5）
```
场景：相机快速移动，场景快速变化
PAS ≈ 0.7（剧烈运动）
阈值 = 0.03（很宽松）
→ 允许较大的质量波动
→ 只检测明显的模糊
```

---

### 8. 总结

**为什么原始方式需要 PAS？**

1. **自适应阈值**：根据相机运动幅度动态调整阈值，避免误判
2. **更准确**：PAS 使用专业方法，能区分相机运动和物体运动
3. **减少误判**：
   - 静止场景：阈值低，严格检测
   - 运动场景：阈值高，宽松检测

**核心思想**：
> **相机运动越大，相邻帧之间的质量分数自然波动就越大（这是正常的）。因此需要根据相机运动幅度来调整阈值，避免将正常运动误判为模糊。**

**数值关系**：
```
PAS 小 → 阈值低 → 严格检测 → 能检测轻微模糊
PAS 大 → 阈值高 → 宽松检测 → 只检测明显模糊
```

---

### 9. 当前方式的改进建议

如果当前方式需要更准确的阈值，可以考虑：

1. **改进运动估算**：
   - 使用光流法估算相机运动
   - 区分全局运动（相机）和局部运动（物体）

2. **使用 PAS**：
   - 如果已有 PAS 数据，直接使用
   - 如果没有，可以预先计算 PAS

3. **混合策略**：
   - 先用简单方法快速估算
   - 如果发现运动明显，再用 PAS 方法精确计算

---

## 参考代码

### 原始方式（使用 PAS）
```python
# motion_smoothness_score.py
threshold = set_threshold(meta_info['perceptible_amplitude_score'])
artifacts_frames = get_artifacts_frames(scores, threshold)
```

### 当前方式（自己计算）
```python
# simple_blur_detector.py
camera_movement = self._estimate_camera_movement(video_path)
threshold = calculate_adaptive_threshold(camera_movement)
blur_frame_indices = detect_artifact_frames(quality_scores, threshold)
```

两种方式都实现了自适应阈值，但 PAS 方法更准确。

