# MediaPipe模型自动下载功能说明

## 一、功能概述

当前代码已支持**联网自动下载模型**功能，不再仅限于离线模型导入。

## 二、支持的场景

### 2.1 场景1：使用默认模型（自动下载）

**代码：**
```python
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
```

**行为：**
- 检查缓存目录 `.cache/mediapipe/pose_landmarker_heavy.task` 是否存在
- 如果不存在，自动从官方URL下载模型
- 下载完成后保存到缓存目录，后续直接使用

### 2.2 场景2：指定本地路径（文件不存在时自动下载）

**代码：**
```python
extractor = MediaPipeKeypointExtractor(
    model_path=".cache/mediapipe/custom_model.task",
    cache_dir=".cache"
)
```

**行为：**
- 检查指定路径的文件是否存在
- 如果不存在，自动从默认URL下载模型
- 下载完成后保存到指定路径

### 2.3 场景3：指定URL（直接下载）

**代码：**
```python
extractor = MediaPipeKeypointExtractor(
    model_path="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    cache_dir=".cache"
)
```

**行为：**
- 识别为URL，直接下载模型文件
- 下载完成后保存到缓存目录
- 后续使用缓存的模型文件

### 2.4 场景4：指定本地路径（文件存在）

**代码：**
```python
extractor = MediaPipeKeypointExtractor(
    model_path=".cache/mediapipe/existing_model.task",
    cache_dir=".cache"
)
```

**行为：**
- 检查文件是否存在
- 如果存在，直接使用本地模型文件
- 不会尝试下载

## 三、实现细节

### 3.1 默认模型URL

```python
DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
```

这是MediaPipe官方的姿态检测模型（heavy版本）。

### 3.2 下载逻辑

**`_download_model` 方法：**

1. **检查文件是否存在**
   - 如果已存在，直接返回路径
   - 避免重复下载

2. **下载文件**
   - 使用 `urllib.request.urlretrieve` 下载
   - 显示下载进度（百分比和字节数）
   - 保存到缓存目录

3. **验证文件**
   - 检查文件是否存在
   - 检查文件大小是否大于0
   - 如果验证失败，删除不完整文件并抛出异常

4. **错误处理**
   - 捕获下载异常
   - 清理不完整的文件
   - 提供清晰的错误信息和建议

### 3.3 缓存机制

- **缓存目录：** `.cache/mediapipe/`
- **缓存文件名：** 从URL提取，默认为 `pose_landmarker_heavy.task`
- **缓存检查：** 每次初始化时检查缓存文件是否存在

## 四、使用示例

### 4.1 基本使用（自动下载）

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 首次使用：自动下载模型
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
# 输出：默认模型文件不存在，正在从网络下载...
# 输出：正在从 https://... 下载模型文件...
# 输出：下载进度: 45.2% (12345678/27234567 bytes)
# 输出：模型下载完成: .cache/mediapipe/pose_landmarker_heavy.task

# 后续使用：直接使用缓存
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
# 输出：模型文件已存在: .cache/mediapipe/pose_landmarker_heavy.task
```

### 4.2 指定URL下载

```python
# 从自定义URL下载模型
custom_url = "https://example.com/models/custom_pose.task"
extractor = MediaPipeKeypointExtractor(
    model_path=custom_url,
    cache_dir=".cache"
)
```

### 4.3 指定本地路径（自动下载备用）

```python
# 如果本地文件不存在，自动从默认URL下载
extractor = MediaPipeKeypointExtractor(
    model_path=".cache/mediapipe/my_model.task",
    cache_dir=".cache"
)
# 如果 my_model.task 不存在，会自动下载默认模型
```

## 五、优势

### 5.1 用户体验

- ? **零配置**：首次使用自动下载，无需手动下载模型
- ? **智能缓存**：自动缓存下载的模型，避免重复下载
- ? **进度显示**：显示下载进度，用户了解下载状态
- ? **错误提示**：下载失败时提供清晰的错误信息

### 5.2 灵活性

- ? **支持URL**：可以直接指定模型URL
- ? **支持本地路径**：可以使用本地模型文件
- ? **自动回退**：本地文件不存在时自动下载
- ? **缓存管理**：自动管理模型缓存

## 六、注意事项

### 6.1 网络要求

- **首次使用需要网络连接**：下载模型文件需要网络
- **下载速度**：取决于网络速度，模型文件约27MB
- **离线使用**：下载完成后可以离线使用

### 6.2 存储空间

- **模型大小**：约27MB（heavy版本）
- **缓存位置**：`.cache/mediapipe/` 目录
- **磁盘空间**：确保有足够的磁盘空间

### 6.3 错误处理

如果下载失败，代码会：
1. 删除不完整的文件
2. 抛出 `RuntimeError` 异常
3. 提供错误信息和建议

**处理方式：**
```python
try:
    extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
except RuntimeError as e:
    print(f"模型下载失败: {e}")
    # 可以手动下载模型文件到指定目录
```

## 七、对比

### 7.1 修改前（仅支持离线）

```python
# 必须手动下载模型文件
# 如果文件不存在，直接抛出异常
extractor = MediaPipeKeypointExtractor(
    model_path=".cache/mediapipe/model.task"  # 必须存在
)
# 如果文件不存在：FileNotFoundError
```

### 7.2 修改后（支持自动下载）

```python
# 自动下载模型文件
# 如果文件不存在，自动从网络下载
extractor = MediaPipeKeypointExtractor(cache_dir=".cache")
# 自动检查并下载模型
```

## 八、总结

### 8.1 功能状态

- ? **支持离线模型导入**：可以使用本地模型文件
- ? **支持联网自动下载**：自动从URL下载模型
- ? **支持URL指定**：可以直接指定模型URL
- ? **智能缓存管理**：自动管理模型缓存

### 8.2 使用建议

1. **首次使用**：直接使用默认配置，自动下载模型
2. **离线环境**：提前下载模型文件到缓存目录
3. **自定义模型**：指定模型URL或本地路径
4. **生产环境**：建议提前下载模型，避免运行时下载

---

**文档版本：** 1.0  
**创建时间：** 2024年  
**作者：** AI Assistant

