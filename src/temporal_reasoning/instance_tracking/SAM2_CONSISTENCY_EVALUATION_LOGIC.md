# SAM2 评估时序一致性的逻辑详解

## 整体流程

```
视频输入
  ↓
采样检测（Grounding DINO + SAM2）
  ↓
Mask传播（SAM2 propagate）
  ↓
生成tracking_result（每帧的对象列表和mask）
  ↓
识别消失/出现对象
  ↓
检测形状突变
  ↓
计算一致性分数
  ↓
生成异常报告
```

## 详细步骤

### 阶段1：采样检测与传播（`pipeline.py` 第178-240行）

#### 1.1 采样帧检测

**采样间隔计算**：
```python
fps = max(1, int(fps_value) if fps_value else 24)
step = max(1, fps - 1)  # 例如：30fps → step=29
```

**检测循环**：
```python
for start_frame_idx in range(0, len(frames), step):
    # 在采样帧上检测（帧0, 30, 60, 90...）
    image = frames[start_frame_idx]
    mask_dict = self.detection_engine.detect(image, text_prompt)
```

**检测过程**：
1. 使用 **Grounding DINO** 检测目标（基于文本提示）
2. 使用 **SAM2** 生成精确的mask
3. 返回 `MaskDictionary`，包含每个对象的mask、边界框、类别名称

#### 1.2 对象追踪与匹配

**更新追踪器**（第191-196行）：
```python
objects_count, updated_dict = mask_dict.update_with_tracker(
    sam2_masks,  # 之前帧的对象
    iou_threshold=self.config.iou_threshold,  # 默认0.75
    objects_count=objects_count,
)
```

**逻辑**：
- 使用 **IoU（交并比）** 匹配当前检测的对象与之前帧的对象
- 如果IoU >= 0.75，认为是同一个对象（保持ID）
- 如果IoU < 0.75，认为是新对象（分配新ID）

#### 1.3 Mask传播

**添加到视频状态**（第198-200行）：
```python
if hasattr(self.detection_engine.video_predictor, "reset_state"):
    self.detection_engine.video_predictor.reset_state(inference_state)
self.detection_engine.add_masks_to_video_state(inference_state, start_frame_idx, mask_dict)
```

**传播到后续帧**（第215-240行）：
```python
for out_frame_idx, out_obj_ids, out_mask_logits in self.detection_engine.propagate(
    inference_state,
    step,  # 传播step帧
    start_frame_idx,
):
    # 为每个传播帧生成mask和检测框
    for i, out_obj_id in enumerate(out_obj_ids):
        out_mask = out_mask_logits[i] > 0.0
        obj = self._object_info_from_mask(mask_2d, class_name, out_obj_id)
        frame_masks.labels[out_obj_id] = obj
        frame_data[out_obj_id] = obj.to_serializable()
    video_object_data.append(frame_data)
```

**传播结果**：
- 采样帧检测 → 传播到中间帧
- 例如：帧0检测 → 传播到帧1-29

### 阶段2：生成tracking_result（第244行）

**准备追踪结果**（第136-140行）：
```python
def _prepare_tracking_result(self, video_object_data: List[Dict], step: int) -> List[Dict]:
    # 过滤掉采样帧，只保留传播帧
    filtered = [item for idx, item in enumerate(video_object_data, start=1) if idx % (step + 1) != 0]
    return filtered[:: max(step, 1)]
```

**tracking_result结构**：
```python
tracking_result = [
    {obj_id1: {'mask': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}, ...},  # 帧0
    {obj_id1: {'mask': ..., ...}, obj_id2: {...}},  # 帧1
    {obj_id1: {...}},  # 帧2（obj_id2消失）
    ...
]
```

### 阶段3：识别消失/出现对象（第245-246行）

#### 3.1 消失对象检测（`tcs_utils.py` 第89-121行）

**逻辑**：
```python
def get_disappear_objects(tracking_result):
    # 1. 记录每个对象的首次出现帧和mask
    for i, current_dict in enumerate(tracking_result):
        for key, value in current_dict.items():
            if key not in first_appearances:
                first_appearances[key] = {
                    'frame': i,
                    'mask': np.array(value['mask'])
                }
    
    # 2. 比较相邻帧，找出消失的对象
    for i in range(len(tracking_result) - 1):
        dict1 = tracking_result[i]
        dict2 = tracking_result[i + 1]
        disappeared_keys = set(dict1.keys()) - set(dict2.keys())  # 集合差集
        
        # 3. 记录消失对象的信息
        for key in disappeared_keys:
            disappeared_object_info = {
                'object_id': key,
                'mask': first_appearances[key]['mask'],
                'first_appearance': first_appearances[key]['frame'],
                'last_frame': i
            }
```

**关键点**：
- 使用**集合差集**找出消失的对象：`set(dict1.keys()) - set(dict2.keys())`
- 记录对象首次出现和最后出现的帧索引
- 保存对象首次出现时的mask

#### 3.2 出现对象检测（`tcs_utils.py` 第206-237行）

**逻辑**：
```python
def get_appear_objects(dict_list):
    # 类似逻辑，但找出新出现的对象
    for i in range(1, len(dict_list)):
        dict1 = dict_list[i - 1]
        dict2 = dict_list[i]
        appeared_keys = set(dict2.keys()) - set(dict1.keys())  # 集合差集
```

**关键点**：
- 使用**集合差集**找出出现的对象：`set(dict2.keys()) - set(dict1.keys())`
- 记录对象首次出现的帧索引和mask

### 阶段4：计算一致性分数（第248-263行）

#### 4.1 CoTracker验证（可选）

**如果启用CoTracker**（第249-255行）：
```python
if self.config.enable_cotracker and self.event_evaluator is not None:
    vanish_score, emerge_score = self.event_evaluator.score(
        video_tensor,
        tracking_result,
        objects_count,
    )
```

**CoTracker验证逻辑**：
- 对每个消失/出现的对象，使用CoTracker追踪其轨迹
- 验证消失/出现是否合理：
  - 边缘消失/出现（合理）
  - 太小消失/出现（合理）
  - 检测错误（不合理）
- 计算分数：`(objects_count - 不合理消失/出现数) / objects_count`

#### 4.2 仅使用SAM2评估（默认）

**如果未启用CoTracker**（第256-261行）：
```python
else:
    # 仅使用SAM2评估，使用默认分数
    vanish_score = 1.0 if not disappear_objects else 0.5
    emerge_score = 1.0 if not appear_objects else 0.5
```

**逻辑**：
- 如果没有消失对象：`vanish_score = 1.0`（完美）
- 如果有消失对象：`vanish_score = 0.5`（降低）
- 如果没有出现对象：`emerge_score = 1.0`（完美）
- 如果有出现对象：`emerge_score = 0.5`（降低）

#### 4.3 一致性分数

**计算**（第263行）：
```python
coherence_score = (vanish_score + emerge_score) / 2
```

**含义**：
- 范围：[0, 1]
- 1.0：完全一致（无消失/出现）
- 0.5：部分一致（有消失/出现，但可能是合理的）
- 0.0：完全不一致（大量不合理消失/出现）

### 阶段5：检测形状突变（第272-277行）

**实现**（`_detect_shape_anomalies`，第525-604行）：

#### 5.1 遍历tracking_result

```python
for frame_idx, frame_objects in enumerate(tracking_result):
    for obj_id_str, obj_meta in frame_objects.items():
        mask_data = obj_meta.get("mask")
        # 计算mask面积
        mask_bool = mask_np.astype(bool)
        area = float(mask_bool.sum())
        
        # 计算边界框高度
        bbox_height = float(max(1, int(y2) - int(y1) + 1))
```

#### 5.2 比较相邻帧

```python
prev = prev_stats.get(obj_id)
if prev:
    prev_area = max(prev["area"], 1.0)
    prev_height = max(prev["height"], 1.0)
    
    # 计算变化比率
    area_ratio = max(area / prev_area, prev_area / area)
    height_ratio = max(bbox_height / prev_height, prev_height / bbox_height)
    
    # 判断是否突变
    if area_ratio >= area_ratio_threshold or height_ratio >= height_ratio_threshold:
        # 判定为形状突变
```

**阈值**（默认值）：
- `area_ratio_threshold = 3.0`：面积变化超过3倍
- `height_ratio_threshold = 2.5`：高度变化超过2.5倍
- `min_area = 200`：最小面积（小于此值忽略）

#### 5.3 生成异常报告

```python
anomalies.append({
    "type": "structural_size_jump",
    "modality": "structure",
    "frame_id": frame_idx,
    "timestamp": frame_idx / fps_safe,
    "confidence": min(1.0, max(area_ratio / area_ratio_threshold, height_ratio / height_ratio_threshold)),
    "description": f"Object {obj_id} size changed abruptly (area_ratio={area_ratio:.2f}, height_ratio={height_ratio:.2f})",
    "metadata": {
        "object_id": obj_id,
        "area_ratio": area_ratio,
        "height_ratio": height_ratio,
        "current_area": area,
        "previous_area": prev_area,
        "current_height": bbox_height,
        "previous_height": prev_height,
    },
})
```

### 阶段6：生成异常报告（第265-277行）

#### 6.1 消失/出现异常（`_build_structure_anomalies`，第460-523行）

**消失异常**：
```python
for obj in disappear_objects:
    anomalies.append({
        "type": "structural_disappearance",
        "modality": "structure",
        "frame_id": frame_id,
        "timestamp": frame_id / fps_safe,
        "confidence": vanish_confidence,  # 1.0 - vanish_score
        "description": "Object disappeared abruptly",
        "location": {"mask": mask_tensor},
        "metadata": {
            "object_id": obj.get("object_id"),
            "first_appearance": obj.get("first_appearance"),
            "last_frame": obj.get("last_frame"),
        },
    })
```

**出现异常**：
```python
for obj in appear_objects:
    anomalies.append({
        "type": "structural_appearance",
        "modality": "structure",
        "frame_id": frame_id,
        "timestamp": frame_id / fps_safe,
        "confidence": emerge_confidence,  # 1.0 - emerge_score
        "description": "Object appeared unexpectedly",
        "location": {"mask": mask_tensor},
        "metadata": {
            "object_id": obj.get("object_id"),
            "first_appearance": obj.get("first_appearance"),
        },
    })
```

#### 6.2 形状突变异常

已在阶段5中生成，添加到异常列表。

## 评估指标

### 1. 一致性分数（coherence_score）

**计算**：
```python
coherence_score = (vanish_score + emerge_score) / 2
```

**含义**：
- 衡量整体时序一致性
- 范围：[0, 1]
- 越高表示越一致

### 2. 消失分数（vanish_score）

**仅使用SAM2时**：
- 无消失对象：1.0
- 有消失对象：0.5

**使用CoTracker时**：
- 基于不合理消失对象的数量计算
- `(objects_count - 不合理消失数) / objects_count`

### 3. 出现分数（emerge_score）

**仅使用SAM2时**：
- 无出现对象：1.0
- 有出现对象：0.5

**使用CoTracker时**：
- 基于不合理出现对象的数量计算
- `(objects_count - 不合理出现数) / objects_count`

## 异常类型

### 1. structural_disappearance（结构消失）

**触发条件**：
- 对象在某一帧消失（从tracking_result中消失）

**信息**：
- 对象ID
- 首次出现帧
- 最后出现帧
- 首次出现时的mask

### 2. structural_appearance（结构出现）

**触发条件**：
- 对象在某一帧出现（在tracking_result中新增）

**信息**：
- 对象ID
- 首次出现帧
- 首次出现时的mask

### 3. structural_size_jump（形状突变）

**触发条件**：
- 对象面积变化超过3倍，或高度变化超过2.5倍

**信息**：
- 对象ID
- 变化比率（面积比率、高度比率）
- 当前和之前的大小

## 关键参数

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `step` | `fps - 1` | 采样间隔 |
| `iou_threshold` | 0.75 | 对象匹配的IoU阈值 |
| `box_threshold` | 0.35 | Grounding DINO检测阈值 |
| `text_threshold` | 0.35 | Grounding DINO文本阈值 |

### 形状突变参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `size_change_area_ratio_threshold` | 3.0 | 面积变化阈值 |
| `size_change_height_ratio_threshold` | 2.5 | 高度变化阈值 |
| `size_change_min_area` | 200 | 最小面积（小于此值忽略） |

## 总结

### SAM2评估时序一致性的核心逻辑

1. **采样检测**：在采样帧上使用Grounding DINO + SAM2检测目标
2. **Mask传播**：使用SAM2的propagate将检测结果传播到中间帧
3. **对象追踪**：使用IoU匹配相邻帧的对象，维护对象ID
4. **识别异常**：
   - 消失/出现：比较相邻帧的对象列表
   - 形状突变：比较相邻帧的mask面积和高度
5. **计算分数**：基于异常数量计算一致性分数
6. **生成报告**：生成详细的异常报告

### 关键特点

- **完全基于SAM2**：不依赖CoTracker（CoTracker可选）
- **采样检测**：不是每帧都检测，而是采样检测+传播
- **IoU匹配**：使用IoU匹配对象，维护对象ID
- **时序分析**：比较相邻帧的变化，检测异常

### 评估能力

✅ **可以评估**：
- 对象消失
- 对象出现
- 形状突变（面积、高度）
- 时序一致性

❌ **不能评估**（需要CoTracker）：
- 消失/出现的合理性（边缘消失、太小消失等）
- 误检过滤

