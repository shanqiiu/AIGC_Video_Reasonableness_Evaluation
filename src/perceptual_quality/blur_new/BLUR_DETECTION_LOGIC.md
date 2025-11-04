# 视频模糊检测逻辑说明

## 核心评估思路

本模糊检测算法基于**运动平滑度（Motion Smoothness Score, MSS）**的概念，核心思想是：

**即使视频的平均质量分数很高，如果相邻帧之间的质量分数波动很大，说明视频存在模糊或不稳定问题。**

这与传统的静态图像质量评估不同，我们关注的是**帧间质量的一致性**，而不是绝对质量值。

---

## 为什么质量分数很高还会判定为模糊？

### 关键理解

1. **算法不是看平均质量分数，而是看质量分数的变化**
   - 平均质量分数高：说明整体画面质量不错
   - 但质量分数波动大：说明某些帧质量突然下降，可能是模糊导致的

2. **示例场景**
   ```
   帧1: 质量分数 0.95 (清晰)
   帧2: 质量分数 0.98 (清晰)  
   帧3: 质量分数 0.75 (模糊) ← 突然下降
   帧4: 质量分数 0.97 (清晰)
   帧5: 质量分数 0.96 (清晰)
   
   平均质量: 0.92 (很高)
   但帧2→帧3的差异: |0.98 - 0.75| = 0.23 (超过阈值)
   → 判定为模糊
   ```

3. **检测逻辑**
   ```python
   # 计算相邻帧之间的分数差异
   score_differences = np.abs(np.diff(quality_scores))
   
   # 找出分数差异超过阈值的帧
   artifact_indices = np.where(score_differences > threshold)[0]
   ```

---

## 滑动窗口实现逻辑

### 1. 窗口大小
- 默认 `window_size = 3`（可配置）
- 每个窗口包含 3 帧，用于计算一个质量分数

### 2. 滑动方式
```
假设 window_size = 3，视频有 5 帧：

帧索引:  0    1    2    3    4
窗口1:   [0]  [1]  [2]        → 计算质量分数 S0
窗口2:        [1]  [2]  [3]   → 计算质量分数 S1
窗口3:             [2]  [3]  [4] → 计算质量分数 S2

得到质量分数序列: [S0, S1, S2]
```

### 3. 边界处理

**开头帧处理**（窗口在开头时）：
```python
if current_frame_idx < left_extend:
    # 如果窗口在开头，用第一帧填充整个窗口
    frame_groups.append([Image.fromarray(frames_array[0])] * window_size)
```

**帧数不足时**：
```python
# 如果帧数不足 window_size，进行填充
while len(frame_indices) < window_size:
    if start_frame_idx == 0:
        frame_indices.append(frame_indices[-1])  # 向后填充
    else:
        frame_indices.insert(0, frame_indices[0])  # 向前填充
```

### 4. 代码实现
```python
def load_video_with_sliding_window(video_path: str, window_size: int = 5):
    # 计算窗口左右扩展帧数
    left_extend = (window_size - 1) // 2  # 左侧扩展帧数
    right_extend = window_size - 1 - left_extend  # 右侧扩展帧数
    
    for current_frame_idx in range(total_frames):
        # 计算窗口范围
        start_frame_idx = max(0, current_frame_idx - left_extend)
        end_frame_idx = min(total_frames, current_frame_idx + right_extend + 1)
        
        # 读取窗口内的帧
        frame_indices = list(range(start_frame_idx, end_frame_idx))
        # ... 处理填充和边界情况
```

---

## 整体评估流程

### 步骤1: 加载视频帧（滑动窗口）
```python
video_frames = load_video_with_sliding_window(video_path, window_size=3)
# 返回: List[List[Image]]，每个元素是一个帧组（3帧）
```

### 步骤2: 计算质量分数
```python
_, _, quality_scores = self.scorer(video_frames)
# 使用 Q-Align 模型对每个帧组评分
# 返回: 每个帧对应的质量分数列表
```

### 步骤3: 估算相机运动幅度
```python
camera_movement = self._estimate_camera_movement(video_path)
# 读取前10帧，计算帧间差异
# 归一化到 [0, 1]
```

**运动幅度计算**：
```python
# 计算相邻帧之间的差异
for i in range(1, len(frames)):
    diff = cv2.absdiff(frames[i], frames[i-1])
    total_diff += np.mean(diff)

# 归一化
movement = total_diff / (len(frames) - 1) / 255.0
```

### 步骤4: 计算自适应阈值
```python
threshold = calculate_adaptive_threshold(camera_movement)
```

**阈值规则**：
```python
if camera_movement < 0.1:
    return 0.01      # 静止场景，阈值低
elif camera_movement < 0.3:
    return 0.015     # 轻微运动
elif camera_movement < 0.5:
    return 0.025     # 中等运动
else:
    return 0.03      # 剧烈运动，阈值高
```

**为什么需要自适应阈值？**
- 相机运动大时，帧间质量自然会有波动（这是正常的）
- 需要提高阈值，避免误判
- 静止场景时，任何质量波动都可能是模糊

### 步骤5: 检测模糊帧
```python
blur_frame_indices = detect_artifact_frames(quality_scores, threshold)
```

**检测逻辑**：
```python
# 计算相邻帧之间的分数差异
score_differences = np.abs(np.diff(quality_scores))
# 例如: [0.95, 0.98, 0.75, 0.97] → [0.03, 0.23, 0.22]

# 找出差异超过阈值的帧
artifact_indices = np.where(score_differences > threshold)[0]
# 例如: 如果 threshold=0.02，则找到索引 [1, 2]

# 返回当前帧和下一帧（因为差异可能由任一帧引起）
artifact_frame_indices = np.unique(
    np.concatenate([artifact_indices, artifact_indices + 1])
)
# 例如: [1, 2, 3]
```

### 步骤6: 计算模糊指标

```python
blur_metrics = self._calculate_blur_metrics(
    quality_scores, blur_frame_indices, threshold
)
```

**计算的指标**：
1. **blur_ratio**: 模糊帧比例 = 模糊帧数 / 总帧数
2. **avg_quality**: 平均质量分数
3. **quality_std**: 质量分数标准差（衡量波动）
4. **max_quality_drop**: 最大质量下降 = max(相邻帧差异)

### 步骤7: 判断模糊严重程度

```python
if blur_ratio > 0.3 or max_quality_drop > threshold * 2:
    return "严重模糊"
elif blur_ratio > 0.1 or max_quality_drop > threshold * 1.5:
    return "中等模糊"
elif blur_ratio > 0.05 or max_quality_drop > threshold:
    return "轻微模糊"
else:
    return "无模糊"
```

### 步骤8: 计算置信度

```python
confidence = (
    blur_confidence * 0.4 +      # 模糊比例权重 40%
    quality_confidence * 0.4 +    # 质量下降权重 40%
    avg_quality_confidence * 0.2  # 平均质量权重 20%
)
```

---

## 关键参数说明

### 1. window_size (滑动窗口大小)
- **默认值**: 3
- **作用**: 决定每次评估多少帧
- **影响**: 
  - 值越大：评估更稳定，但计算量更大
  - 值越小：对单帧变化更敏感

### 2. threshold (检测阈值)
- **默认值**: 根据相机运动自适应（0.01 ~ 0.03）
- **作用**: 判断相邻帧质量差异是否异常
- **影响**:
  - 值越小：更容易检测到模糊（敏感）
  - 值越大：只检测明显的模糊（保守）

### 3. BLUR_RATIO_THRESHOLD
- **默认值**: 0.05 (5%)
- **作用**: 模糊帧比例阈值
- **含义**: 如果超过 5% 的帧被判定为模糊，则认为视频有模糊问题

---

## 算法优势

1. **不受绝对质量影响**: 即使整体质量高，也能检测到局部模糊
2. **自适应阈值**: 根据相机运动调整，减少误判
3. **时间一致性**: 关注帧间变化，更符合人眼感知
4. **鲁棒性**: 使用滑动窗口平滑，减少单帧噪声影响

---

## 示例分析

### 场景1: 高质量稳定视频
```
质量分数序列: [0.95, 0.96, 0.95, 0.97, 0.96]
相邻差异:     [0.01, 0.01, 0.02, 0.01]
最大差异: 0.02 < 阈值(0.025)
→ 判定: 无模糊 ?
```

### 场景2: 高质量但有模糊帧
```
质量分数序列: [0.95, 0.98, 0.75, 0.97, 0.96]
相邻差异:     [0.03, 0.23, 0.22, 0.01]
最大差异: 0.23 > 阈值(0.025)
→ 判定: 轻微模糊 ??
平均质量: 0.92 (很高，但仍判定为模糊)
```

### 场景3: 低质量但稳定
```
质量分数序列: [0.60, 0.61, 0.60, 0.62, 0.61]
相邻差异:     [0.01, 0.01, 0.02, 0.01]
最大差异: 0.02 < 阈值(0.025)
→ 判定: 无模糊 ?
平均质量: 0.61 (较低，但稳定，不判定为模糊)
```

---

## 总结

**核心思想**: 模糊检测关注的是**质量分数的波动**，而不是绝对质量值。

**关键指标**:
1. **max_quality_drop**: 相邻帧质量分数的最大差异
2. **blur_ratio**: 模糊帧占总帧数的比例
3. **threshold**: 根据相机运动自适应的检测阈值

即使平均质量分数很高，如果帧间质量波动大（说明某些帧突然变模糊），仍会被判定为模糊。

