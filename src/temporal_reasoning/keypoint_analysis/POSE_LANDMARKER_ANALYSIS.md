# PoseLandmarker模型功能分析

## 一、pose_landmarker_heavy.task 功能说明

### 1.1 模型作用

`pose_landmarker_heavy.task` 是MediaPipe提供的**身体姿态检测模型**，专门用于检测人体姿态关键点。

### 1.2 支持的关键点

**仅支持身体姿态关键点：**
- ? **33个身体关键点**（Body Pose Landmarks）
- ? **不支持人脸关键点**（Face Landmarks）
- ? **不支持手部关键点**（Hand Landmarks）

**33个身体关键点包括：**
- 头部：鼻子、眼睛、耳朵等（5个）
- 上身：肩膀、肘部、手腕等（10个）
- 躯干：胸部、腰部等（2个）
- 下身：臀部、膝盖、脚踝等（16个）

### 1.3 模型特点

- **精度**：Heavy版本，精度较高
- **速度**：相对较慢（相比Lite版本）
- **大小**：约27MB
- **用途**：专注于身体姿态估计

## 二、当前实现的功能

### 2.1 代码实现

查看 `keypoint_extractor.py` 第152-159行：

```python
# 提取身体关键点（PoseLandmarker主要提供身体姿态）
if detection_result.pose_landmarks:
    keypoints['body'] = np.array([
        [lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]
    ])

# 注意：PoseLandmarker主要关注身体姿态
# 如果需要手部和面部关键点，需要使用HandLandmarker和FaceLandmarker
```

### 2.2 当前返回结果

```python
keypoints = {
    'body': np.array([[x, y, z], ...]),  # 33个身体关键点
    'left_hand': None,   # 不支持
    'right_hand': None,  # 不支持
    'face': None         # 不支持
}
```

## 三、功能限制分析

### 3.1 不支持的功能

1. **人脸关键点检测**
   - 无法检测面部特征点（如眼睛、鼻子、嘴巴等）
   - 无法进行眨眼、嘴型等分析

2. **手部关键点检测**
   - 无法检测手部关键点（如手指关节等）
   - 无法进行手势分析

### 3.2 对当前需求的影响

查看 `keypoint_analyzer.py` 中的分析功能：

```python
# 分析眨眼
blink_score, blink_anomalies = self._analyze_blink_pattern(keypoint_sequences, fps)

# 分析嘴型
mouth_score, mouth_anomalies = self._analyze_mouth_pattern(keypoint_sequences, fps)

# 分析手势
gesture_score, gesture_anomalies = self._analyze_hand_gesture(keypoint_sequences, fps)
```

**问题：**
- `_analyze_blink_pattern` 需要面部关键点，但当前模型不支持
- `_analyze_mouth_pattern` 需要面部关键点，但当前模型不支持
- `_analyze_hand_gesture` 需要手部关键点，但当前模型不支持

**当前状态：**
- 这些方法都返回默认值（1.0, []），无法真正分析

## 四、解决方案

### 4.1 方案1：使用多个模型（推荐）

**使用三个独立的模型：**
- `PoseLandmarker` - 身体姿态（33个关键点）
- `HandLandmarker` - 手部检测（每只手21个关键点）
- `FaceLandmarker` - 面部检测（468个关键点）

**优点：**
- ? 使用最新Task API
- ? 可以独立控制每个模型
- ? 可以选择性使用需要的模型
- ? 性能优化（只加载需要的模型）

**缺点：**
- ? 需要加载多个模型
- ? 需要多次推理
- ? 代码复杂度增加

### 4.2 方案2：使用Holistic模型（旧API）

**使用MediaPipe Holistic模型：**
- 同时检测身体、手部、面部关键点
- 使用旧版API（`mp.solutions.holistic`）

**优点：**
- ? 一个模型解决所有需求
- ? 代码简单
- ? 性能较好（单次推理）

**缺点：**
- ? 使用旧版API（可能被弃用）
- ? 不支持新API的.task文件格式
- ? 无法单独控制各个部分

### 4.3 方案3：混合方案

**结合使用：**
- 使用 `PoseLandmarker` 检测身体姿态（新API）
- 使用 `HandLandmarker` 和 `FaceLandmarker` 检测手部和面部（新API）

**优点：**
- ? 使用最新API
- ? 功能完整
- ? 可以灵活配置

**缺点：**
- ? 需要加载多个模型
- ? 代码复杂度较高

## 五、推荐方案

### 5.1 根据需求选择

**如果只需要身体姿态：**
- ? 当前 `pose_landmarker_heavy.task` 已满足需求
- ? 无需修改

**如果需要完整功能（身体+手部+面部）：**
- ? 推荐使用**方案1**（多个模型）
- ? 使用最新API，功能完整
- ? 可以灵活配置

### 5.2 实现建议

**修改 `keypoint_extractor.py`：**

1. **添加HandLandmarker和FaceLandmarker支持**
2. **可选加载模型**（通过配置控制）
3. **统一接口**（保持extract_keypoints接口不变）

## 六、MediaPipe模型对比

### 6.1 模型功能对比

| 模型 | 身体 | 手部 | 面部 | API版本 |
|------|------|------|------|---------|
| `PoseLandmarker` | ? 33个 | ? | ? | 新API |
| `HandLandmarker` | ? | ? 21个/手 | ? | 新API |
| `FaceLandmarker` | ? | ? | ? 468个 | 新API |
| `Holistic` | ? 33个 | ? 21个/手 | ? 468个 | 旧API |

### 6.2 模型下载URL

**PoseLandmarker（当前使用）：**
```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

**HandLandmarker：**
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**FaceLandmarker：**
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 七、总结

### 7.1 当前状态

- ? **支持**：身体姿态检测（33个关键点）
- ? **不支持**：人脸关键点检测
- ? **不支持**：手部关键点检测

### 7.2 需求匹配

**如果需求包括：**
- ? 身体姿态分析 → **满足**
- ? 眨眼分析 → **不满足**（需要面部关键点）
- ? 嘴型分析 → **不满足**（需要面部关键点）
- ? 手势分析 → **不满足**（需要手部关键点）

### 7.3 建议

**如果需要完整功能：**
1. 添加 `HandLandmarker` 和 `FaceLandmarker` 支持
2. 修改 `extract_keypoints` 方法，同时提取所有关键点
3. 更新 `keypoint_analyzer.py` 中的分析方法，使用实际关键点数据

**如果只需要身体姿态：**
- 当前实现已满足需求，无需修改

---

**文档版本：** 1.0  
**创建时间：** 2024年  
**作者：** AI Assistant

