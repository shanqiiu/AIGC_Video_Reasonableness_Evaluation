# MediaPipe 模型缓存问题说明

## 问题描述

运行脚本后，`.cache` 文件夹中并没有 MediaPipe 模型文件。

## 原因分析

**MediaPipe 不支持直接指定自定义缓存目录**。MediaPipe 会将模型文件自动下载到系统默认缓存目录，而不是用户指定的目录。

### MediaPipe 默认缓存目录位置

- **Windows**: `C:\Users\<username>\AppData\Local\Temp\mediapipe`
- **Linux**: `~/.cache/mediapipe`
- **macOS**: `~/.cache/mediapipe`

### 为什么代码中的 `model_cache_dir` 参数不起作用？

1. MediaPipe 的模型下载机制是硬编码的，不检查 `MEDIAPIPE_CACHE_DIR` 环境变量
2. MediaPipe 使用系统临时目录或用户缓存目录，无法通过环境变量直接修改
3. 模型文件在首次使用时自动下载，下载位置由 MediaPipe 内部决定

## 解决方案

### 方案1: 首次使用后手动复制模型文件（推荐）

1. **首次运行脚本**，让 MediaPipe 自动下载模型到系统默认缓存目录
2. **查找系统缓存目录**：
   ```python
   from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor
   
   extractor = MediaPipeKeypointExtractor(model_cache_dir=".cache/mediapipe")
   system_cache = extractor.get_system_cache_path()
   print(f"系统缓存目录: {system_cache}")
   ```
3. **使用代码自动复制**：
   ```python
   extractor = MediaPipeKeypointExtractor(model_cache_dir=".cache/mediapipe")
   # 首次使用后，复制模型文件
   extractor.copy_models_from_system_cache()
   ```

### 方案2: 手动复制模型文件

1. **找到系统缓存目录**：
   - Windows: `C:\Users\<你的用户名>\AppData\Local\Temp\mediapipe`
   - Linux/Mac: `~/.cache/mediapipe`

2. **复制模型文件到 `.cache/mediapipe`**：
   ```bash
   # Windows (PowerShell)
   Copy-Item -Path "C:\Users\<用户名>\AppData\Local\Temp\mediapipe\*" -Destination ".cache\mediapipe\" -Recurse
   
   # Linux/Mac
   cp -r ~/.cache/mediapipe/* .cache/mediapipe/
   ```

### 方案3: 使用符号链接（仅限 Linux/Mac）

```bash
# 创建符号链接，将系统缓存目录链接到 .cache/mediapipe
ln -s ~/.cache/mediapipe .cache/mediapipe
```

## 使用示例

### 完整示例代码

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor
import numpy as np

# 创建提取器，指定缓存目录
extractor = MediaPipeKeypointExtractor(model_cache_dir=".cache/mediapipe")

# 首次使用后，尝试从系统缓存复制模型文件
# 注意：需要先运行一次 extract_keypoints 让 MediaPipe 下载模型
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
extractor.extract_keypoints(test_image)  # 这会触发模型下载

# 复制模型文件到指定目录
success = extractor.copy_models_from_system_cache()
if success:
    print("模型文件已复制到 .cache/mediapipe")
else:
    print("请手动复制模型文件")
```

## 验证模型文件位置

### 检查系统缓存目录

```python
from pathlib import Path
import platform

system = platform.system()
if system == "Windows":
    username = os.environ.get('USERNAME', 'user')
    cache_path = Path(f"C:/Users/{username}/AppData/Local/Temp/mediapipe")
else:
    home = os.environ.get('HOME', '~')
    cache_path = Path(home) / ".cache" / "mediapipe"

print(f"MediaPipe 系统缓存目录: {cache_path}")
if cache_path.exists():
    model_files = list(cache_path.rglob("*.tflite"))
    print(f"找到 {len(model_files)} 个模型文件")
    for f in model_files:
        print(f"  - {f}")
else:
    print("缓存目录不存在，模型尚未下载")
```

### 检查 .cache 目录

```python
from pathlib import Path

cache_dir = Path(".cache/mediapipe")
if cache_dir.exists():
    model_files = list(cache_dir.rglob("*.tflite"))
    print(f".cache 目录中找到 {len(model_files)} 个模型文件")
    for f in model_files:
        print(f"  - {f}")
else:
    print(".cache/mediapipe 目录不存在")
```

## 注意事项

1. **首次运行**: MediaPipe 模型在首次使用时才会下载，需要网络连接
2. **模型大小**: MediaPipe Holistic 模型文件较大（约几十MB），下载可能需要一些时间
3. **权限问题**: 确保有权限访问系统缓存目录和 `.cache` 目录
4. **路径问题**: Windows 路径中的反斜杠需要正确处理

## 总结

MediaPipe 模型不会自动下载到 `.cache` 文件夹，因为 MediaPipe 不支持自定义缓存目录。解决方案是：

1. 让 MediaPipe 正常下载模型到系统默认位置
2. 使用 `copy_models_from_system_cache()` 方法将模型复制到 `.cache/mediapipe`
3. 或者手动复制模型文件

这样既能让 MediaPipe 正常工作，又能将模型文件集中管理在 `.cache` 目录中。

