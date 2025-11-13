# CoTracker 可视化改进方案

## 当前实现分析

### 1. 当前可视化方式

**代码位置**：`pipeline.py` 第346-397行

```python
def _save_cotracker_visualization(
    self,
    video_tensor: torch.Tensor,
    video_path: str,
    fps: int,
) -> None:
    # 对整个视频进行密集追踪（dense tracks）
    tracks, visibility = self.cotracker_model(
        video_on_device,
        grid_size=self.config.grid_size,  # 在整个视频上生成网格点
        grid_query_frame=0,
        backward_tracking=True,
        # 注意：没有使用 segm_mask 参数
    )
    
    # 可视化整个视频的追踪结果
    visualizer.visualize(
        video=video_tensor.cpu(),
        tracks=tracks.cpu(),
        visibility=visibility.cpu(),
        filename="cotracker_tracks",
        save_video=True,
        # 注意：没有传入 segm_mask 参数
    )
```

### 2. 问题分析

**当前实现的问题**：
1. ✅ **是整个视频的追踪可视化**：使用 `grid_size` 在整个视频上生成密集追踪点
2. ❌ **没有聚焦于目标mask**：没有使用 `segm_mask` 参数限制追踪区域
3. ❌ **没有针对特定对象**：所有对象的追踪结果混在一起

### 3. Visualizer 支持的功能

**Visualizer 的 `visualize` 方法支持**（`visualizer.py` 第87-134行）：
- `segm_mask` 参数：可以传入分割mask来过滤或高亮特定区域
- `query_frame` 参数：指定查询帧
- 支持多种可视化模式：`rainbow`, `cool`, `optical_flow`

## 改进方案

### 方案1：为每个消失/出现对象生成独立可视化（推荐）

**优点**：
- 聚焦于特定对象的追踪结果
- 可以清楚地看到每个对象的追踪轨迹
- 便于分析和调试

**实现思路**：
1. 在验证消失/出现对象时，为每个对象生成独立的追踪可视化
2. 使用对象的初始mask作为 `segm_mask`
3. 为每个对象保存单独的可视化视频

### 方案2：在整体可视化中高亮目标mask区域

**优点**：
- 保留整体视频的追踪结果
- 同时高亮目标mask区域

**实现思路**：
1. 使用 `segm_mask` 参数在可视化中高亮特定区域
2. 可以叠加多个mask来高亮多个对象

### 方案3：只追踪目标mask区域

**优点**：
- 只追踪目标对象，减少计算量
- 可视化结果更清晰

**实现思路**：
1. 在调用 CoTracker 时传入 `segm_mask` 参数
2. CoTracker 只在mask区域内生成追踪点

## 推荐实现：方案1 + 方案3 结合

结合两种方案的优点：
1. 为每个消失/出现对象生成独立的追踪可视化
2. 使用对象的初始mask限制追踪区域
3. 生成清晰、聚焦的可视化结果

