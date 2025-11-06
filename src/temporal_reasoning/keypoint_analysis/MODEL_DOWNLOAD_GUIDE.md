# MediaPipe旧API模型下载指南

## 概述

MediaPipe旧API（`mp.solutions.holistic`）的模型是**自动下载**的，不需要手动下载。首次使用时，MediaPipe会自动从服务器下载模型文件并缓存到本地。

## 自动下载机制

### 1. 首次运行自动下载

当您首次运行以下代码时，MediaPipe会自动下载模型：

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 首次运行，MediaPipe会自动下载模型
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=False  # 联网模式，允许自动下载
)

# 初始化时会自动下载模型
# 下载过程可能需要几分钟，取决于网络速度
```

### 2. 下载过程

**下载时机**：
- 首次调用 `Holistic()` 初始化时
- 如果模型文件不存在或损坏时

**下载过程**：
1. MediaPipe检查缓存目录中是否有模型文件
2. 如果不存在，自动从Google服务器下载
3. 下载完成后，保存到缓存目录
4. 后续使用直接从缓存加载，无需重新下载

**下载时间**：
- 首次下载可能需要几分钟（取决于网络速度）
- 模型文件大小约几十MB到几百MB

## 模型缓存位置

### 默认缓存位置

MediaPipe旧API的模型文件通常缓存在以下位置：

**Windows**:
```
%USERPROFILE%\.mediapipe\models\
例如: C:\Users\用户名\.mediapipe\models\
```

**Linux/Mac**:
```
~/.mediapipe/models/
例如: /home/用户名/.mediapipe/models/
```

### 自定义缓存位置

可以通过环境变量 `MEDIAPIPE_CACHE_DIR` 指定自定义缓存目录：

```python
import os
from pathlib import Path

# 设置自定义缓存目录
custom_cache = Path(".cache") / "mediapipe"
custom_cache.mkdir(parents=True, exist_ok=True)
os.environ['MEDIAPIPE_CACHE_DIR'] = str(custom_cache.absolute())

# 然后初始化
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=False
)
```

**注意**：代码中已经设置了 `MEDIAPIPE_CACHE_DIR`，模型会下载到 `.cache/mediapipe/models/` 目录。

## 模型文件格式

MediaPipe旧API使用的模型文件格式：
- **`.tflite`** - TensorFlow Lite格式
- **`.binarypb`** - MediaPipe二进制协议格式

**模型文件示例**：
```
.mediapipe/models/
├── holistic_landmarker.tflite
├── hand_landmarker.tflite
├── face_landmarker.tflite
└── ...
```

## 手动触发下载

### 方法1：运行初始化代码

最简单的方法是运行初始化代码：

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 确保 offline=False（默认值）
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=False  # 允许自动下载
)

# 初始化时会自动下载模型
print("模型下载完成！")
```

### 方法2：直接初始化MediaPipe

也可以直接初始化MediaPipe的Holistic模型：

```python
import mediapipe as mp

# 初始化Holistic模型，会自动下载模型
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True
)

# 测试使用（可选）
import numpy as np
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
results = holistic.process(test_image)
print("模型下载并初始化成功！")
```

## 验证模型是否已下载

### 方法1：检查缓存目录

```python
from pathlib import Path
import os

# 检查默认缓存位置
user_home = Path.home()
default_cache = user_home / ".mediapipe" / "models"

# 检查自定义缓存位置
custom_cache = Path(".cache") / "mediapipe" / "models"

# 检查环境变量指定的位置
env_cache = Path(os.environ.get('MEDIAPIPE_CACHE_DIR', '')) / "models" if os.environ.get('MEDIAPIPE_CACHE_DIR') else None

cache_dirs = [default_cache, custom_cache]
if env_cache:
    cache_dirs.append(env_cache)

for cache_dir in cache_dirs:
    if cache_dir and cache_dir.exists():
        model_files = list(cache_dir.glob("*.tflite")) + list(cache_dir.glob("*.binarypb"))
        if model_files:
            print(f"找到模型文件: {cache_dir}")
            for f in model_files:
                print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"目录存在但无模型文件: {cache_dir}")
    else:
        print(f"目录不存在: {cache_dir}")
```

### 方法2：运行代码检查

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

try:
    extractor = MediaPipeKeypointExtractor(
        cache_dir=".cache",
        offline=True  # 离线模式，如果模型不存在会报错
    )
    print("模型已下载并可用！")
except RuntimeError as e:
    print(f"模型未下载: {e}")
    print("请运行以下代码下载模型：")
    print("extractor = MediaPipeKeypointExtractor(cache_dir='.cache', offline=False)")
```

## 离线环境使用

### 步骤1：在联网环境下载模型

在联网环境下，运行以下代码下载模型：

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 联网模式，自动下载模型
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=False  # 允许下载
)

# 测试提取关键点（确保模型已下载）
import numpy as np
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
keypoints = extractor.extract_keypoints(test_image)
print("模型下载成功！")
```

### 步骤2：查找模型文件位置

运行以下代码查找模型文件：

```python
from pathlib import Path
import os

user_home = Path.home()
possible_cache_dirs = [
    user_home / ".mediapipe" / "models",
    Path(".cache") / "mediapipe" / "models",
    Path(os.environ.get('MEDIAPIPE_CACHE_DIR', '')) / "models" if os.environ.get('MEDIAPIPE_CACHE_DIR') else None,
]

for cache_dir in possible_cache_dirs:
    if cache_dir and cache_dir.exists():
        model_files = list(cache_dir.glob("*.tflite")) + list(cache_dir.glob("*.binarypb"))
        if model_files:
            print(f"模型文件位置: {cache_dir}")
            for f in model_files:
                print(f"  - {f}")
```

### 步骤3：复制模型文件到离线环境

将找到的模型文件复制到离线环境的相应目录：

**Windows**:
```bash
# 从联网环境复制到离线环境
xcopy "%USERPROFILE%\.mediapipe\models\*" "离线环境\.cache\mediapipe\models\" /E /I
```

**Linux/Mac**:
```bash
# 从联网环境复制到离线环境
cp -r ~/.mediapipe/models/* 离线环境/.cache/mediapipe/models/
```

### 步骤4：在离线环境使用

在离线环境下，使用离线模式：

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 离线模式，仅从缓存加载
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=True  # 离线模式
)
```

## 常见问题

### Q1: 如何强制重新下载模型？

**A**: 删除缓存目录中的模型文件，然后重新运行：

```python
from pathlib import Path
import shutil

# 删除缓存目录
cache_dir = Path(".cache") / "mediapipe" / "models"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("已删除缓存，下次运行时会重新下载")

# 或者删除默认缓存
default_cache = Path.home() / ".mediapipe" / "models"
if default_cache.exists():
    shutil.rmtree(default_cache)
    print("已删除默认缓存，下次运行时会重新下载")
```

### Q2: 下载失败怎么办？

**A**: 检查以下几点：
1. **网络连接**：确保可以访问Google服务器
2. **防火墙**：检查防火墙是否阻止了下载
3. **代理设置**：如果使用代理，确保代理配置正确
4. **磁盘空间**：确保有足够的磁盘空间

### Q3: 模型文件有多大？

**A**: MediaPipe Holistic模型的典型大小：
- 完整模型：约50-200MB（取决于模型复杂度）
- 模型复杂度0：较小
- 模型复杂度2：较大（推荐）

### Q4: 可以手动下载模型文件吗？

**A**: MediaPipe旧API的模型文件是自动下载的，不提供直接下载链接。建议：
1. 在联网环境下首次运行，让MediaPipe自动下载
2. 将下载的模型文件复制到离线环境

### Q5: 如何查看下载进度？

**A**: MediaPipe的自动下载过程不会显示进度条。首次运行时：
- 如果网络较慢，可能需要等待几分钟
- 可以通过检查缓存目录来确认下载是否完成

## 最佳实践

### 1. 开发环境（联网）

```python
# 首次运行，自动下载模型
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=False  # 允许下载
)
```

### 2. 生产环境（离线）

```python
# 使用已缓存的模型
extractor = MediaPipeKeypointExtractor(
    cache_dir=".cache",
    offline=True  # 离线模式
)
```

### 3. 模型文件管理

- **版本控制**：如果项目允许，可以将模型文件纳入版本控制
- **部署脚本**：提供模型文件下载和部署脚本
- **文档说明**：在部署文档中说明模型文件位置

## 总结

1. **自动下载**：MediaPipe旧API的模型是自动下载的，无需手动操作
2. **首次运行**：首次运行时会自动下载模型到缓存目录
3. **缓存位置**：默认在 `~/.mediapipe/models/` 或通过环境变量指定
4. **离线使用**：将模型文件复制到离线环境，使用离线模式
5. **验证方法**：检查缓存目录或运行离线模式测试

通过以上步骤，可以轻松下载和使用MediaPipe旧API的模型。

