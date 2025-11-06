# Linux环境下Timestamp错误修复说明

## 问题描述

在Linux环境下，代码仍然报错："Input timestamp must be monotonically increasing"

## 问题原因分析

### 1. **Extractor对象被重复使用**
- 在Linux环境下，`KeypointAnalyzer`的`extractor`对象可能被重复使用
- 如果同一个extractor对象处理多个视频，timestamp会继续累积
- 例如：第一个视频处理完后，timestamp_ms = 10000，第二个视频开始时仍然是10000，导致错误

### 2. **Windows vs Linux的差异**
- Windows环境下可能每次都是新创建的extractor对象
- Linux环境下extractor对象可能被缓存或重复使用
- 这导致timestamp没有被重置

### 3. **MediaPipe版本差异**
- Linux和Windows上安装的MediaPipe版本可能不同
- 某些版本对timestamp的要求更严格

## 解决方案

### 修复代码

在`keypoint_analyzer.py`的`analyze`方法中，在处理视频之前重置timestamp：

```python
def analyze(
    self,
    video_frames: List[np.ndarray],
    fps: float = 30.0
) -> Tuple[float, List[Dict]]:
    if self.extractor is None:
        self.initialize()
    
    # 重置timestamp计数器（每次处理新视频时都必须重置）
    # 这是关键：Linux环境下，如果extractor对象被重复使用，timestamp会累积
    # 必须在每次处理新视频时重置，否则会报"timestamp must be monotonically increasing"错误
    if hasattr(self.extractor, 'reset_timestamp'):
        self.extractor.reset_timestamp()
    
    print("正在分析生理动作自然性...")
    
    # 1. 提取关键点序列
    print("正在提取关键点...")
    keypoint_sequences = []
    for frame in tqdm(video_frames, desc="提取关键点"):
        keypoints = self.extractor.extract_keypoints(frame, fps=fps)
        keypoint_sequences.append(keypoints)
    
    # ... 其余代码
```

### 具体修改位置

在`keypoint_analyzer.py`的第88行之后（`self.initialize()`之后），添加：

```python
# 重置timestamp计数器（每次处理新视频时都必须重置）
if hasattr(self.extractor, 'reset_timestamp'):
    self.extractor.reset_timestamp()
```

## 验证方法

1. 在Linux环境下运行代码
2. 检查是否还有"timestamp must be monotonically increasing"错误
3. 如果处理多个视频，确保每个视频都能正常处理

## 其他注意事项

1. **确保fps参数正确传递**：timestamp的递增依赖于fps，确保fps参数正确传递
2. **检查MediaPipe版本**：确保Linux和Windows上安装的MediaPipe版本一致
3. **检查extractor对象生命周期**：如果extractor对象被多个线程共享，需要考虑线程安全问题

## 相关文件

- `keypoint_extractor.py`: 包含`reset_timestamp()`方法
- `keypoint_analyzer.py`: 需要调用`reset_timestamp()`的地方

