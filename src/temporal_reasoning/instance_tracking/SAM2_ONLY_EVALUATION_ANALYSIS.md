# SAM2 独立评估时序一致性分析

## 问题：是否仅需要SAM2即可针对mask的形态变化、是否消失、突变情况进行评估？

### 答案：**是的，仅需要SAM2就可以完成这些评估**

## 当前实现逻辑分析

### 1. SAM2提供的功能

**SAM2在时序一致性评估中的作用**：

1. **目标检测**：使用 Grounding DINO + SAM2 在采样帧上检测目标
2. **Mask传播**：使用 SAM2 的 `propagate` 方法将检测结果传播到后续帧
3. **生成tracking_result**：记录每帧的对象列表和对应的mask

**tracking_result结构**：
```python
tracking_result = [
    {obj_id1: {'mask': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}, ...},  # 帧0
    {obj_id1: {'mask': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}, ...},  # 帧1
    {obj_id1: {'mask': ..., ...}},  # 帧2（obj_id2消失）
    ...
]
```

### 2. 基于SAM2的评估功能

#### 功能1：消失/出现检测（`get_disappear_objects` / `get_appear_objects`）

**实现位置**：`tcs_utils.py` 第89-237行

**逻辑**：
- 遍历 `tracking_result`，比较相邻帧的对象列表
- 找出消失的对象：`set(dict1.keys()) - set(dict2.keys())`
- 找出出现的对象：`set(dict2.keys()) - set(dict1.keys())`

**完全基于SAM2的tracking_result**：
```python
def get_disappear_objects(tracking_result):
    for i in range(len(tracking_result) - 1):
        dict1 = tracking_result[i]
        dict2 = tracking_result[i + 1]
        disappeared_keys = set(dict1.keys()) - set(dict2.keys())  # 基于SAM2的结果
        ...
```

#### 功能2：形状突变检测（`_detect_shape_anomalies`）

**实现位置**：`pipeline.py` 第502-580行

**逻辑**：
- 遍历 `tracking_result`，计算每帧每个对象的mask面积和高度
- 比较相邻帧的面积和高度变化
- 如果变化超过阈值，判定为形状突变

**完全基于SAM2的tracking_result**：
```python
def _detect_shape_anomalies(self, tracking_result: List[Dict], fps: int):
    for frame_idx, frame_objects in enumerate(tracking_result):
        for obj_id_str, obj_meta in frame_objects.items():
            mask_data = obj_meta.get("mask")  # 来自SAM2的mask
            mask_np = np.array(mask_data)
            area = float(mask_bool.sum())  # 计算mask面积
            
            # 比较相邻帧的面积和高度变化
            area_ratio = max(area / prev_area, prev_area / area)
            height_ratio = max(bbox_height / prev_height, prev_height / bbox_height)
            
            if area_ratio >= area_ratio_threshold or height_ratio >= height_ratio_threshold:
                # 判定为形状突变
```

#### 功能3：形态变化检测（基于mask的时序分析）

**当前实现**：
- 通过比较相邻帧的mask面积和高度来检测形态变化
- 可以扩展为更详细的形态分析（如mask的轮廓、形状特征等）

**基于SAM2的tracking_result**：
- 每帧都有完整的mask信息
- 可以计算mask的各种特征（面积、周长、形状因子等）
- 可以比较相邻帧的特征变化

### 3. CoTracker的作用（可选）

**CoTracker的作用**：
- **验证消失/出现是否合理**：
  - 边缘消失（合理）
  - 太小消失（合理）
  - 检测错误（不合理）
- **不是必须的**：如果只关注mask的形态变化、消失、突变，不需要CoTracker

**代码位置**：`evaluation.py` 第88-146行

```python
# CoTracker验证消失/出现是否合理
vanish_score, emerge_score = self.event_evaluator.score(
    video_tensor,
    tracking_result,
    objects_count,
)
```

## 总结

### SAM2可以独立完成的功能

| 功能 | 实现方式 | 是否需要CoTracker |
|------|---------|------------------|
| **消失检测** | 基于tracking_result比较相邻帧 | ❌ 不需要 |
| **出现检测** | 基于tracking_result比较相邻帧 | ❌ 不需要 |
| **形状突变检测** | 基于tracking_result计算mask面积/高度变化 | ❌ 不需要 |
| **形态变化检测** | 基于tracking_result分析mask特征变化 | ❌ 不需要 |

### CoTracker的作用（可选）

| 功能 | 实现方式 | 是否必须 |
|------|---------|---------|
| **验证消失合理性** | 使用CoTracker追踪验证是否从边缘消失 | ❌ 可选 |
| **验证出现合理性** | 使用CoTracker追踪验证是否从边缘出现 | ❌ 可选 |
| **过滤误检** | 识别检测错误，避免将误检当作异常 | ❌ 可选 |

### 结论

**是的，仅需要SAM2就可以完成以下评估**：

1. ✅ **Mask形态变化**：
   - 通过比较相邻帧的mask面积、高度、形状特征
   - 可以检测到面积变化、高度变化、形状变化

2. ✅ **消失检测**：
   - 通过比较相邻帧的对象列表
   - 可以检测到对象何时消失

3. ✅ **突变检测**：
   - 通过比较相邻帧的mask面积和高度变化
   - 可以检测到突然的大小变化（面积突变、高度突变）

**CoTracker的作用**：
- 验证消失/出现是否合理（避免误检）
- 提供更准确的异常评分
- **但不是必须的**，如果只关注mask的形态变化、消失、突变，可以不用CoTracker

### 建议

如果只需要评估mask的形态变化、消失、突变，可以：
1. **仅使用SAM2**：完全基于tracking_result进行分析
2. **可选使用CoTracker**：如果需要验证消失/出现的合理性，可以使用CoTracker

### 代码示例

**仅使用SAM2的评估流程**：

```python
# 1. SAM2检测和传播
tracking_result = self._prepare_tracking_result(video_object_data, step)

# 2. 检测消失/出现（仅基于SAM2）
disappear_objects = get_disappear_objects(tracking_result)
appear_objects = get_appear_objects(tracking_result)

# 3. 检测形状突变（仅基于SAM2）
shape_anomalies = self._detect_shape_anomalies(tracking_result, fps)

# 4. 构建异常列表（不需要CoTracker）
anomalies = self._build_structure_anomalies(
    disappear_objects,
    appear_objects,
    vanish_score=0.5,  # 可以设置默认值
    emerge_score=0.5,  # 可以设置默认值
    fps=fps,
)
anomalies.extend(shape_anomalies)
```

**使用CoTracker的评估流程**（当前实现）：

```python
# 1. SAM2检测和传播
tracking_result = self._prepare_tracking_result(video_object_data, step)

# 2. 检测消失/出现（基于SAM2）
disappear_objects = get_disappear_objects(tracking_result)
appear_objects = get_appear_objects(tracking_result)

# 3. CoTracker验证（可选）
vanish_score, emerge_score = self.event_evaluator.score(
    video_tensor,
    tracking_result,
    objects_count,
)

# 4. 检测形状突变（基于SAM2）
shape_anomalies = self._detect_shape_anomalies(tracking_result, fps)

# 5. 构建异常列表
anomalies = self._build_structure_anomalies(
    disappear_objects,
    appear_objects,
    vanish_score,
    emerge_score,
    fps=fps,
)
anomalies.extend(shape_anomalies)
```

## 总结

**仅需要SAM2就可以完成mask的形态变化、消失、突变评估**。CoTracker是可选的，用于验证消失/出现的合理性，但不是必须的。

