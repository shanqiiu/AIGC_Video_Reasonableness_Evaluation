# 内存使用分析与优化建议

## 当前实现的内存风险点

### ? **高风险：一次性加载所有视频帧**

**位置**: `motion_smoothness_score.py` - `load_video_with_sliding_window()`

```python
def load_video_with_sliding_window(video_path: str, window_size: int = 5):
    video_reader = VideoReader(video_path)
    total_frames = len(video_reader)
    frame_groups = []  # ?? 所有帧组都存储在内存中
    
    for current_frame_idx in range(total_frames):
        # ... 处理每一帧
        frame_groups.append([Image.fromarray(frame) for frame in frames_array])
    
    return frame_groups  # ?? 返回所有帧组
```

**内存占用估算**：
- 假设视频：1080p (1920x1080), 30fps, 60秒
- 总帧数：1800 帧
- 窗口大小：3 帧
- 每帧内存：1920 × 1080 × 3 (RGB) × 1 byte = ~6.2 MB
- **总内存占用**：1800 × 3 × 6.2 MB ≈ **33.5 GB** ??

**问题**：
- 对于长视频，会一次性将整个视频的所有帧加载到内存
- PIL Image 对象在内存中占用较大空间
- 没有内存释放机制

---

### ? **高风险：一次性处理所有帧组**

**位置**: `motion_smoothness_score.py` - `QAlignVideoScorer.forward()`

```python
def forward(self, video_frames: List[List[Image.Image]]):
    # ?? 一次性预处理所有帧组
    video_tensors = [
        self.image_processor.preprocess(
            frame_group, return_tensors="pt"
        )["pixel_values"].half().to(self.model.device)
        for frame_group in processed_frames  # 所有帧组
    ]
    
    # ?? 一次性推理所有帧组
    output = self.model(input_tensors, images=video_tensors)
```

**内存占用估算**：
- 预处理后的张量：每个帧组约 3 × 224 × 224 × 3 (假设输入尺寸) × 2 bytes (half) = ~900 KB
- 1800 个帧组：1800 × 900 KB ≈ **1.6 GB** (仅输入张量)
- 模型权重：Q-Align 模型本身约 **2-4 GB**
- **峰值内存**：模型 + 输入 + 中间激活 ≈ **6-8 GB** ??

**问题**：
- 没有批处理大小限制
- 所有帧组同时加载到 GPU
- 对于超长视频，GPU 内存会溢出

---

### ? **中等风险：视频帧组列表累积**

**位置**: `simple_blur_detector.py` - `detect_blur()`

```python
video_frames = load_video_with_sliding_window(video_path, window_size)
# ?? 所有帧组存储在 video_frames 中
_, _, quality_scores = self.scorer(video_frames)
```

**问题**：
- 帧组列表在内存中保持到处理完成
- 没有及时释放

---

## 内存使用场景分析

### 场景1: 短视频 (< 30秒)
- **风险**: ? 低
- **内存占用**: < 2 GB
- **说明**: 通常不会出现问题

### 场景2: 中等视频 (30-120秒)
- **风险**: ? 中等
- **内存占用**: 2-8 GB
- **说明**: 可能在 GPU 内存较小的设备上出现问题

### 场景3: 长视频 (> 120秒)
- **风险**: ? 高
- **内存占用**: > 8 GB
- **说明**: **很可能出现 OOM 错误**

---

## 优化方案

### 方案1: 批处理推理（推荐）

**修改 `QAlignVideoScorer.forward()` 方法**：

```python
def forward(self, video_frames: List[List[Image.Image]], batch_size: int = 32):
    """
    对视频帧进行质量评分（分批处理）
    
    Args:
        video_frames: 视频帧列表
        batch_size: 批处理大小（默认32）
    """
    all_weighted_scores = []
    
    # 分批处理
    for i in range(0, len(video_frames), batch_size):
        batch_frames = video_frames[i:i + batch_size]
        
        # 预处理当前批次
        processed_batch = [...]
        video_tensors = [
            self.image_processor.preprocess(...)
            for frame_group in processed_batch
        ]
        
        # 推理当前批次
        with torch.inference_mode():
            output = self.model(input_tensors, images=video_tensors)
            # ... 处理输出
        
        # 释放当前批次内存
        del video_tensors, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_weighted_scores
```

**优点**：
- 限制峰值内存使用
- 可以通过调整 `batch_size` 控制内存
- 适合处理长视频

---

### 方案2: 流式处理视频帧（最佳）

**修改 `load_video_with_sliding_window()` 为生成器**：

```python
def load_video_with_sliding_window_generator(video_path: str, window_size: int = 5):
    """
    使用滑动窗口方式加载视频帧（生成器版本，节省内存）
    """
    video_reader = VideoReader(video_path)
    total_frames = len(video_reader)
    
    for current_frame_idx in range(total_frames):
        # 计算窗口范围
        start_frame_idx = max(0, current_frame_idx - left_extend)
        end_frame_idx = min(total_frames, current_frame_idx + right_extend + 1)
        
        # 读取当前窗口的帧
        frame_indices = list(range(start_frame_idx, end_frame_idx))
        frames_array = video_reader.get_batch(frame_indices).asnumpy()
        
        # 转换为 PIL Image
        frame_group = [Image.fromarray(frame) for frame in frames_array]
        
        # 生成当前帧组
        yield frame_group
        
        # 自动释放内存
        del frames_array, frame_group
```

**修改 `BlurDetector.detect_blur()`**：

```python
def detect_blur(self, video_path: str, window_size: int = 3, batch_size: int = 32):
    # 使用生成器而不是一次性加载
    video_frame_generator = load_video_with_sliding_window_generator(
        video_path, window_size
    )
    
    # 分批处理
    all_quality_scores = []
    batch_frames = []
    
    for frame_group in video_frame_generator:
        batch_frames.append(frame_group)
        
        if len(batch_frames) >= batch_size:
            # 处理当前批次
            _, _, scores = self.scorer(batch_frames)
            all_quality_scores.extend(scores.tolist())
            batch_frames = []  # 清空批次
    
    # 处理剩余帧
    if batch_frames:
        _, _, scores = self.scorer(batch_frames)
        all_quality_scores.extend(scores.tolist())
    
    return all_quality_scores
```

**优点**：
- 内存占用固定（只存储当前批次）
- 适合处理任意长度的视频
- 内存使用与视频长度无关

---

### 方案3: 添加内存监控和清理

**添加内存检查函数**：

```python
def check_memory_usage():
    """检查当前内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU 内存: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
        
        # 如果内存使用超过阈值，清理缓存
        if allocated > 6.0:  # 6GB 阈值
            torch.cuda.empty_cache()
            print("已清理 GPU 缓存")
```

---

## 推荐实施步骤

### 优先级1: 立即实施（防止 OOM）

1. **添加批处理支持**（方案1）
   - 修改 `QAlignVideoScorer.forward()` 添加 `batch_size` 参数
   - 默认 `batch_size=32`，可根据 GPU 内存调整

2. **添加内存监控**
   - 在处理前检查可用内存
   - 根据内存情况自动调整批处理大小

### 优先级2: 长期优化（提升效率）

1. **实现流式处理**（方案2）
   - 将 `load_video_with_sliding_window()` 改为生成器
   - 修改 `detect_blur()` 支持流式处理

2. **添加进度显示**
   - 显示处理进度和内存使用情况

---

## 内存使用估算公式

```
总内存占用 = 模型内存 + 输入张量内存 + 中间激活内存

模型内存 ≈ 2-4 GB (Q-Align 模型)

输入张量内存 = 总帧数 × 窗口大小 × 输入尺寸 × 数据类型大小
            = N × W × H × W_img × C × 2 bytes (half precision)

中间激活 ≈ 输入张量内存 × 2-3 倍

峰值内存 ≈ 模型内存 + (输入张量内存 × 3)
```

**示例**：
- 1080p 视频，60秒，30fps：1800 帧
- 窗口大小：3
- 输入尺寸：224×224
- **输入张量**：1800 × 3 × 224 × 224 × 3 × 2 bytes ≈ 1.6 GB
- **峰值内存**：4 GB + (1.6 GB × 3) ≈ **8.8 GB**

**建议**：
- 如果 GPU 内存 < 8GB，必须使用批处理
- 如果 GPU 内存 < 6GB，必须使用流式处理

---

## 当前代码的 OOM 风险评估

| 视频长度 | 帧数 (30fps) | 风险等级 | 可能 OOM |
|---------|-------------|---------|---------|
| < 30秒  | < 900       | ? 低   | 否      |
| 30-60秒 | 900-1800    | ? 中   | 可能    |
| 60-120秒| 1800-3600   | ? 高   | 很可能  |
| > 120秒 | > 3600      | ? 极高 | 几乎肯定 |

**结论**：当前实现**对于长视频（>60秒）很可能导致 OOM**，建议立即实施批处理优化。

