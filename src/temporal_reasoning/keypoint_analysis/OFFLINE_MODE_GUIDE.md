# 离线模式使用指南

## 概述

在离线环境下使用MediaPipe旧API（`mp.solutions.holistic`）时，需要确保模型文件已缓存到本地。本指南说明如何在离线环境下正确配置和使用。

## 离线模式配置

### 方法1：通过参数启用离线模式

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 启用离线模式
extractor = MediaPipeKeypointExtractor(
    model_path=None,
    cache_dir=".cache",
    use_new_api=False,  # 使用旧API
    offline=True  # 启用离线模式
)
```

### 方法2：通过配置类启用离线模式

```python
from src.temporal_reasoning.core.config import KeypointConfig
from src.temporal_reasoning.keypoint_analysis.keypoint_analyzer import KeypointAnalyzer

# 创建配置
config = KeypointConfig(
    model_type="mediapipe",
    model_path=None,
    offline=True  # 启用离线模式
)

# 初始化分析器
analyzer = KeypointAnalyzer(config)
analyzer.initialize()
```

## 离线模式工作原理

### 1. 模型缓存位置

MediaPipe旧API的模型文件通常缓存在以下位置：

**Windows**:
- `%USERPROFILE%\.mediapipe\models\`
- 或通过`MEDIAPIPE_CACHE_DIR`环境变量指定的目录

**Linux/Mac**:
- `~/.mediapipe/models/`
- 或通过`MEDIAPIPE_CACHE_DIR`环境变量指定的目录

**自定义缓存目录**:
- 通过`cache_dir`参数指定的目录下的`mediapipe/models/`

### 2. 模型文件格式

MediaPipe旧API使用的模型文件格式：
- `.tflite` - TensorFlow Lite格式
- `.binarypb` - MediaPipe二进制协议格式

### 3. 离线模式检查

启用离线模式后，代码会：
1. 检查常见的缓存目录
2. 查找`.tflite`或`.binarypb`格式的模型文件
3. 如果找到模型文件，显示提示信息
4. 如果未找到模型文件，显示警告和解决方案

## 准备工作（联网环境）

在离线环境使用之前，需要在联网环境下完成以下步骤：

### 步骤1：首次运行（下载模型）

在联网环境下，首次运行以下代码：

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 不使用离线模式，让MediaPipe自动下载模型
extractor = MediaPipeKeypointExtractor(
    model_path=None,
    cache_dir=".cache",
    use_new_api=False,
    offline=False  # 联网模式，允许下载
)

# 初始化（会自动下载模型）
extractor._initialize_holistic()

# 测试提取关键点
import numpy as np
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
keypoints = extractor.extract_keypoints(test_image)
print("模型下载成功！")
```

### 步骤2：查找模型文件位置

运行后，模型文件会被下载到缓存目录。查找模型文件：

```python
from pathlib import Path
import os

# 检查常见的缓存位置
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
            print(f"找到模型文件: {cache_dir}")
            for f in model_files:
                print(f"  - {f}")
```

### 步骤3：复制模型文件到离线环境

将找到的模型文件复制到离线环境的相应目录：

```bash
# Windows示例
# 从联网环境复制到离线环境
copy "%USERPROFILE%\.mediapipe\models\*" "离线环境\.cache\mediapipe\models\"

# Linux/Mac示例
# 从联网环境复制到离线环境
cp -r ~/.mediapipe/models/* 离线环境/.cache/mediapipe/models/
```

## 离线环境使用

### 1. 设置缓存目录

确保模型文件在正确的缓存目录中：

```python
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

# 指定缓存目录（包含模型文件）
extractor = MediaPipeKeypointExtractor(
    model_path=None,
    cache_dir=".cache",  # 模型文件应在此目录下的mediapipe/models/
    use_new_api=False,
    offline=True  # 启用离线模式
)
```

### 2. 验证模型文件

代码会自动检查模型文件是否存在：

```
离线模式：在缓存目录发现模型文件: .cache/mediapipe/models
MediaPipe Holistic模型（旧API）初始化成功
支持检测：身体（33个）+ 手部（42个）+ 面部（468个）关键点
```

### 3. 如果模型文件不存在

如果模型文件不存在，会显示警告：

```
警告：离线模式下未找到缓存的模型文件
提示：请在联网环境下首次运行以下载模型，或手动将模型文件放置到以下目录之一：
  - C:\Users\用户名\.mediapipe\models
  - D:\项目路径\.cache\mediapipe\models
注意：MediaPipe旧API的模型文件格式为.tflite或.binarypb
```

## 常见问题

### Q1: 离线模式下初始化失败怎么办？

**A**: 检查以下几点：
1. 模型文件是否已复制到缓存目录
2. 缓存目录路径是否正确
3. 模型文件是否完整（未损坏）
4. MediaPipe版本是否兼容

### Q2: 如何确认模型文件位置？

**A**: 运行以下代码检查：

```python
from pathlib import Path
import os

user_home = Path.home()
cache_dirs = [
    user_home / ".mediapipe" / "models",
    Path(".cache") / "mediapipe" / "models",
]

for cache_dir in cache_dirs:
    if cache_dir.exists():
        files = list(cache_dir.glob("*"))
        print(f"{cache_dir}: {len(files)} 个文件")
        for f in files[:5]:  # 显示前5个文件
            print(f"  - {f.name}")
```

### Q3: 可以手动下载模型文件吗？

**A**: MediaPipe旧API的模型文件是自动下载的，不提供直接下载链接。建议：
1. 在联网环境下首次运行，让MediaPipe自动下载
2. 将下载的模型文件复制到离线环境

### Q4: 离线模式和新API兼容吗？

**A**: 离线模式主要针对旧API设计。新API（`.task`文件）的离线使用：
1. 手动下载`.task`文件
2. 在代码中指定`.task`文件路径
3. 不需要设置`offline=True`（新API本身就是离线使用的）

## 最佳实践

### 1. 开发环境（联网）

```python
# 首次运行，下载模型
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
    offline=True  # 禁用下载
)
```

### 3. 模型文件管理

- 将模型文件纳入版本控制（如果项目允许）
- 或提供模型文件下载脚本
- 在部署文档中说明模型文件位置

## 总结

1. **联网环境**：首次运行，让MediaPipe自动下载模型
2. **复制模型**：将模型文件复制到离线环境
3. **启用离线模式**：设置`offline=True`
4. **验证**：检查模型文件是否存在

通过以上步骤，可以在离线环境下正常使用MediaPipe旧API。

