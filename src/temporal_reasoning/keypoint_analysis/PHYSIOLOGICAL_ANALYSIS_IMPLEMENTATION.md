# 关键点时序一致性评估实现方案

## 概述

本文档提供将关键点检测接入时序一致性评估的具体实现方案，包括眨眼、嘴型、手势分析的详细算法。

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

### MediaPipe 关键点索引

#### 面部关键点（468个）
- 左眼上眼睑: 159, 158, 157, 156, 155, 154, 153
- 左眼下眼睑: 145, 144, 143, 142, 141, 140, 139
- 右眼上眼睑: 386, 385, 384, 383, 382, 381, 380
- 右眼下眼睑: 374, 373, 372, 371, 370, 369, 368
- 嘴部外轮廓: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
- 嘴部内轮廓: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308

#### 身体关键点（33个）
- 鼻子: 0
- 左眼内: 1, 左眼: 2, 左眼外: 3
- 右眼内: 4, 右眼: 5, 右眼外: 6
- 左耳: 7, 右耳: 8
- 左肩: 11, 右肩: 12
- 左肘: 13, 右肘: 14
- 左腕: 15, 右腕: 16
- 左手掌: 17, 右手掌: 18
- 左臀: 23, 右臀: 24
- 左膝: 25, 右膝: 26
- 左踝: 27, 右踝: 28

#### 手部关键点（21个）
- 手腕: 0
- 拇指: 1-4
- 食指: 5-8
- 中指: 9-12
- 无名指: 13-16
- 小指: 17-20

## 1. 眨眼模式分析

### 算法原理

1. **眼睛纵横比（EAR - Eye Aspect Ratio）**
   - 计算上下眼睑之间的距离
   - EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
   - EAR 值越小，眼睛越闭合

2. **眨眼检测**
   - 当 EAR < 阈值（通常0.2-0.25）时，认为眼睛闭合
   - 连续多帧闭合后再睁开，判定为一次眨眼

3. **异常检测**
   - 眨眼频率过高（> 30次/分钟）
   - 眨眼频率过低（< 5次/分钟）
   - 单次眨眼持续时间过长（> 500ms）
   - 左右眼不同步

### 实现代码

```python
def _analyze_blink_pattern(
    self,
    keypoint_sequences: List[Dict],
    fps: float
) -> Tuple[float, List[Dict]]:
    """分析眨眼模式"""
    if not keypoint_sequences:
        return 1.0, []
    
    # 眨眼检测参数
    EAR_THRESHOLD = 0.25  # EAR阈值
    MIN_BLINK_FRAMES = 2  # 最少闭眼帧数
    MAX_BLINK_FRAMES = int(fps * 0.5)  # 最多闭眼帧数（0.5秒）
    NORMAL_BLINK_RATE_MIN = 5 / 60.0  # 最小眨眼频率（次/秒）
    NORMAL_BLINK_RATE_MAX = 30 / 60.0  # 最大眨眼频率（次/秒）
    
    # 面部关键点索引
    LEFT_EYE_INDICES = {
        'vertical1': (159, 145),  # 上下眼睑中心
        'vertical2': (158, 144),  # 左侧
        'horizontal': (33, 133)    # 左右眼角
    }
    RIGHT_EYE_INDICES = {
        'vertical1': (386, 374),
        'vertical2': (385, 373),
        'horizontal': (362, 263)
    }
    
    anomalies = []
    ear_sequence = []
    blinks = []
    
    # 计算每帧的EAR
    for frame_idx, keypoints in enumerate(keypoint_sequences):
        if keypoints['face'] is None:
            ear_sequence.append(None)
            continue
        
        face = keypoints['face']
        
        # 计算左眼EAR
        left_ear = self._compute_eye_aspect_ratio(face, LEFT_EYE_INDICES)
        # 计算右眼EAR
        right_ear = self._compute_eye_aspect_ratio(face, RIGHT_EYE_INDICES)
        
        # 平均EAR
        avg_ear = (left_ear + right_ear) / 2.0 if left_ear and right_ear else None
        ear_sequence.append(avg_ear)
    
    # 检测眨眼事件
    is_closed = False
    blink_start = -1
    
    for frame_idx, ear in enumerate(ear_sequence):
        if ear is None:
            continue
        
        if not is_closed and ear < EAR_THRESHOLD:
            # 开始闭眼
            is_closed = True
            blink_start = frame_idx
        elif is_closed and ear >= EAR_THRESHOLD:
            # 睁眼
            is_closed = False
            blink_duration = frame_idx - blink_start
            
            # 有效眨眼
            if MIN_BLINK_FRAMES <= blink_duration <= MAX_BLINK_FRAMES:
                blinks.append({
                    'start': blink_start,
                    'end': frame_idx,
                    'duration_frames': blink_duration,
                    'duration_ms': (blink_duration / fps) * 1000
                })
            # 眨眼持续时间过长
            elif blink_duration > MAX_BLINK_FRAMES:
                anomalies.append({
                    'type': 'abnormal_blink_duration',
                    'frame_id': blink_start,
                    'timestamp': f"{blink_start / fps:.2f}s",
                    'duration_ms': (blink_duration / fps) * 1000,
                    'severity': 'medium',
                    'confidence': 0.8,
                    'description': f"眨眼持续时间过长: {(blink_duration / fps) * 1000:.0f}ms"
                })
    
    # 计算眨眼频率
    video_duration = len(keypoint_sequences) / fps
    blink_rate = len(blinks) / video_duration if video_duration > 0 else 0
    
    # 检测异常眨眼频率
    if blink_rate < NORMAL_BLINK_RATE_MIN:
        anomalies.append({
            'type': 'low_blink_rate',
            'frame_id': 0,
            'timestamp': '0.00s',
            'blink_rate': blink_rate * 60,  # 转换为次/分钟
            'severity': 'low',
            'confidence': 0.7,
            'description': f"眨眼频率过低: {blink_rate * 60:.1f}次/分钟"
        })
    elif blink_rate > NORMAL_BLINK_RATE_MAX:
        anomalies.append({
            'type': 'high_blink_rate',
            'frame_id': 0,
            'timestamp': '0.00s',
            'blink_rate': blink_rate * 60,
            'severity': 'medium',
            'confidence': 0.8,
            'description': f"眨眼频率过高: {blink_rate * 60:.1f}次/分钟"
        })
    
    # 计算得分
    if not anomalies:
        score = 1.0
    else:
        # 根据异常严重程度计算得分
        penalty = sum(0.1 if a['severity'] == 'low' else 0.2 for a in anomalies)
        score = max(0.0, 1.0 - penalty)
    
    return score, anomalies

def _compute_eye_aspect_ratio(self, face_landmarks: np.ndarray, eye_indices: dict) -> float:
    """计算眼睛纵横比（EAR）"""
    try:
        # 获取眼睛关键点
        v1_top, v1_bottom = eye_indices['vertical1']
        v2_top, v2_bottom = eye_indices['vertical2']
        h_left, h_right = eye_indices['horizontal']
        
        # 计算垂直距离
        vertical1 = np.linalg.norm(face_landmarks[v1_top] - face_landmarks[v1_bottom])
        vertical2 = np.linalg.norm(face_landmarks[v2_top] - face_landmarks[v2_bottom])
        
        # 计算水平距离
        horizontal = np.linalg.norm(face_landmarks[h_left] - face_landmarks[h_right])
        
        # 计算EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0
        return ear
    except Exception as e:
        return 0.0
```

## 2. 嘴型模式分析

### 算法原理

1. **嘴部开合度（MAR - Mouth Aspect Ratio）**
   - MAR = 嘴部垂直距离 / 嘴部水平距离
   - MAR 值越大，嘴巴张开越大

2. **嘴型变化检测**
   - 分析MAR的时序变化
   - 检测说话、张嘴等动作

3. **异常检测**
   - 嘴部突然大幅度张开（可能是模型错误）
   - 嘴部持续张开（不自然）
   - 嘴型变化不连续（跳跃）

### 实现代码

```python
def _analyze_mouth_pattern(
    self,
    keypoint_sequences: List[Dict],
    fps: float
) -> Tuple[float, List[Dict]]:
    """分析嘴型模式"""
    if not keypoint_sequences:
        return 1.0, []
    
    # 嘴型分析参数
    MAR_THRESHOLD = 0.5  # 嘴部开合阈值
    MAR_JUMP_THRESHOLD = 0.3  # MAR跳跃阈值
    MAX_OPEN_DURATION = fps * 3.0  # 最大持续张嘴时间（3秒）
    
    # 嘴部关键点索引（外轮廓）
    MOUTH_INDICES = {
        'top': 13,      # 上唇中心
        'bottom': 14,   # 下唇中心
        'left': 78,     # 左嘴角
        'right': 308    # 右嘴角
    }
    
    anomalies = []
    mar_sequence = []
    
    # 计算每帧的MAR
    for frame_idx, keypoints in enumerate(keypoint_sequences):
        if keypoints['face'] is None:
            mar_sequence.append(None)
            continue
        
        face = keypoints['face']
        mar = self._compute_mouth_aspect_ratio(face, MOUTH_INDICES)
        mar_sequence.append(mar)
    
    # 检测MAR跳跃（不连续）
    for i in range(1, len(mar_sequence)):
        if mar_sequence[i] is None or mar_sequence[i-1] is None:
            continue
        
        mar_diff = abs(mar_sequence[i] - mar_sequence[i-1])
        if mar_diff > MAR_JUMP_THRESHOLD:
            anomalies.append({
                'type': 'mouth_discontinuity',
                'frame_id': i,
                'timestamp': f"{i / fps:.2f}s",
                'mar_jump': mar_diff,
                'severity': 'medium',
                'confidence': 0.8,
                'description': f"嘴型变化不连续: MAR跳跃 {mar_diff:.2f}"
            })
    
    # 检测持续张嘴
    open_start = -1
    for frame_idx, mar in enumerate(mar_sequence):
        if mar is None:
            continue
        
        if mar > MAR_THRESHOLD:
            if open_start == -1:
                open_start = frame_idx
        else:
            if open_start != -1:
                open_duration = frame_idx - open_start
                if open_duration > MAX_OPEN_DURATION:
                    anomalies.append({
                        'type': 'prolonged_mouth_opening',
                        'frame_id': open_start,
                        'timestamp': f"{open_start / fps:.2f}s",
                        'duration_s': open_duration / fps,
                        'severity': 'low',
                        'confidence': 0.7,
                        'description': f"嘴部持续张开: {open_duration / fps:.1f}秒"
                    })
                open_start = -1
    
    # 计算得分
    if not anomalies:
        score = 1.0
    else:
        penalty = sum(0.1 if a['severity'] == 'low' else 0.15 for a in anomalies)
        score = max(0.0, 1.0 - penalty)
    
    return score, anomalies

def _compute_mouth_aspect_ratio(self, face_landmarks: np.ndarray, mouth_indices: dict) -> float:
    """计算嘴部纵横比（MAR）"""
    try:
        # 获取嘴部关键点
        top = mouth_indices['top']
        bottom = mouth_indices['bottom']
        left = mouth_indices['left']
        right = mouth_indices['right']
        
        # 计算垂直和水平距离
        vertical = np.linalg.norm(face_landmarks[top] - face_landmarks[bottom])
        horizontal = np.linalg.norm(face_landmarks[left] - face_landmarks[right])
        
        # 计算MAR
        mar = vertical / horizontal if horizontal > 0 else 0
        return mar
    except Exception as e:
        return 0.0
```

## 3. 手势分析

### 算法原理

1. **手部运动速度**
   - 计算手腕位置的帧间位移
   - 速度 = 位移 / 时间

2. **手势稳定性**
   - 分析手部关键点的抖动
   - 计算位置标准差

3. **异常检测**
   - 手部运动速度突变（不自然的加速/减速）
   - 手部抖动过大
   - 手部突然消失/出现

### 实现代码

```python
def _analyze_hand_gesture(
    self,
    keypoint_sequences: List[Dict],
    fps: float
) -> Tuple[float, List[Dict]]:
    """分析手势模式"""
    if not keypoint_sequences:
        return 1.0, []
    
    # 手势分析参数
    VELOCITY_THRESHOLD = 0.3  # 速度突变阈值（归一化坐标）
    JITTER_THRESHOLD = 0.05   # 抖动阈值
    WINDOW_SIZE = 5           # 滑动窗口大小
    
    anomalies = []
    
    # 分析左手和右手
    for hand_type in ['left_hand', 'right_hand']:
        hand_positions = []  # 手腕位置序列
        
        # 提取手腕位置（索引0）
        for frame_idx, keypoints in enumerate(keypoint_sequences):
            if keypoints[hand_type] is not None:
                wrist_pos = keypoints[hand_type][0][:2]  # 只取x, y
                hand_positions.append((frame_idx, wrist_pos))
            else:
                hand_positions.append((frame_idx, None))
        
        # 检测速度突变
        velocities = []
        for i in range(1, len(hand_positions)):
            frame_idx1, pos1 = hand_positions[i-1]
            frame_idx2, pos2 = hand_positions[i]
            
            if pos1 is not None and pos2 is not None:
                displacement = np.linalg.norm(pos2 - pos1)
                time_delta = (frame_idx2 - frame_idx1) / fps
                velocity = displacement / time_delta if time_delta > 0 else 0
                velocities.append((frame_idx2, velocity))
            else:
                velocities.append((frame_idx2, None))
        
        # 检测速度跳跃
        for i in range(1, len(velocities)):
            frame_idx1, vel1 = velocities[i-1]
            frame_idx2, vel2 = velocities[i]
            
            if vel1 is not None and vel2 is not None:
                vel_diff = abs(vel2 - vel1)
                if vel_diff > VELOCITY_THRESHOLD:
                    anomalies.append({
                        'type': 'hand_velocity_jump',
                        'hand': hand_type,
                        'frame_id': frame_idx2,
                        'timestamp': f"{frame_idx2 / fps:.2f}s",
                        'velocity_jump': vel_diff,
                        'severity': 'medium',
                        'confidence': 0.75,
                        'description': f"{hand_type}运动速度突变: {vel_diff:.2f}"
                    })
        
        # 检测手部抖动（滑动窗口）
        for i in range(WINDOW_SIZE, len(hand_positions)):
            window = hand_positions[i-WINDOW_SIZE:i]
            valid_positions = [pos for _, pos in window if pos is not None]
            
            if len(valid_positions) >= WINDOW_SIZE * 0.8:  # 至少80%的帧有效
                positions_array = np.array(valid_positions)
                std = np.std(positions_array, axis=0).mean()
                
                if std > JITTER_THRESHOLD:
                    frame_idx = hand_positions[i][0]
                    anomalies.append({
                        'type': 'hand_jitter',
                        'hand': hand_type,
                        'frame_id': frame_idx,
                        'timestamp': f"{frame_idx / fps:.2f}s",
                        'jitter_std': std,
                        'severity': 'low',
                        'confidence': 0.7,
                        'description': f"{hand_type}抖动过大: std={std:.3f}"
                    })
        
        # 检测手部突然消失/出现
        for i in range(1, len(hand_positions)):
            frame_idx1, pos1 = hand_positions[i-1]
            frame_idx2, pos2 = hand_positions[i]
            
            if pos1 is not None and pos2 is None:
                anomalies.append({
                    'type': 'hand_disappear',
                    'hand': hand_type,
                    'frame_id': frame_idx2,
                    'timestamp': f"{frame_idx2 / fps:.2f}s",
                    'severity': 'low',
                    'confidence': 0.6,
                    'description': f"{hand_type}突然消失"
                })
            elif pos1 is None and pos2 is not None:
                anomalies.append({
                    'type': 'hand_appear',
                    'hand': hand_type,
                    'frame_id': frame_idx2,
                    'timestamp': f"{frame_idx2 / fps:.2f}s",
                    'severity': 'low',
                    'confidence': 0.6,
                    'description': f"{hand_type}突然出现"
                })
    
    # 计算得分
    if not anomalies:
        score = 1.0
    else:
        penalty = sum(0.05 if a['severity'] == 'low' else 0.1 for a in anomalies)
        score = max(0.0, 1.0 - penalty)
    
    return score, anomalies
```

## 4. 集成实现

### 修改 keypoint_analyzer.py

将上述方法添加到 `KeypointAnalyzer` 类中：

```python
# 在 KeypointAnalyzer 类中添加以下方法

def _compute_eye_aspect_ratio(self, face_landmarks: np.ndarray, eye_indices: dict) -> float:
    """计算眼睛纵横比（EAR）"""
    # ... 实现代码 ...

def _compute_mouth_aspect_ratio(self, face_landmarks: np.ndarray, mouth_indices: dict) -> float:
    """计算嘴部纵横比（MAR）"""
    # ... 实现代码 ...

def _analyze_blink_pattern(self, keypoint_sequences: List[Dict], fps: float) -> Tuple[float, List[Dict]]:
    """分析眨眼模式"""
    # ... 实现代码 ...

def _analyze_mouth_pattern(self, keypoint_sequences: List[Dict], fps: float) -> Tuple[float, List[Dict]]:
    """分析嘴型模式"""
    # ... 实现代码 ...

def _analyze_hand_gesture(self, keypoint_sequences: List[Dict], fps: float) -> Tuple[float, List[Dict]]:
    """分析手势模式"""
    # ... 实现代码 ...
```

## 5. 测试和验证

### 测试用例

1. **正常视频**：眨眼频率正常、嘴型变化连续、手势平滑
2. **眨眼异常**：长时间不眨眼、眨眼频率过高
3. **嘴型异常**：嘴型跳跃、持续张嘴
4. **手势异常**：手部运动速度突变、手部抖动

### 调试建议

1. 添加可视化：绘制EAR、MAR曲线
2. 添加日志：记录每帧的关键指标
3. 调整阈值：根据实际数据调整阈值参数

## 6. 优化建议

### 性能优化

1. **批量处理**：一次处理所有帧，而不是逐帧处理
2. **并行计算**：使用多线程/多进程处理不同的分析任务
3. **缓存结果**：缓存中间计算结果

### 准确性优化

1. **自适应阈值**：根据视频内容动态调整阈值
2. **多模态融合**：结合多个指标进行异常判断
3. **时序建模**：使用LSTM等时序模型建模关键点变化

## 7. 参考资料

- MediaPipe Holistic文档: https://google.github.io/mediapipe/solutions/holistic
- 眨眼检测算法: Eye Aspect Ratio (EAR) - Soukupová and ?ech, 2016
- 嘴型分析: Mouth Aspect Ratio (MAR) - Similar to EAR
- 手势识别: Hand Landmark Detection - MediaPipe Hands

## 总结

本方案提供了完整的关键点时序一致性评估实现：

1. ? 眨眼模式分析：基于EAR检测眨眼频率和持续时间
2. ? 嘴型模式分析：基于MAR检测嘴型变化和不连续
3. ? 手势分析：检测手部运动速度、抖动和突变

所有方法都可以直接替换 `KeypointAnalyzer` 中的占位符方法，实现完整的生理动作自然性评估。

