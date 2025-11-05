# 错误处理策略说明

> **更新日期**：2025年10月30日  
> **策略**：移除降级策略，直接抛出异常

---

## 一、错误处理原则

### 1.1 基本原则

- **严格验证**：所有必需的模型文件和代码必须存在
- **直接抛出异常**：找不到文件或初始化失败时，直接抛出异常
- **清晰的错误信息**：异常信息包含详细的错误原因和解决建议

### 1.2 不再支持降级策略

- ? 不再使用OpenCV Farneback作为RAFT的降级方案
- ? 不再在Co-Tracker不可用时返回默认值
- ? 不再在模型初始化失败时使用简化实现

---

## 二、异常类型

### 2.1 文件未找到异常 (`FileNotFoundError`)

**触发条件**：
- 模型权重文件不存在
- 配置文件不存在
- 模型代码目录不存在

**示例**：
```python
FileNotFoundError(
    "RAFT模型文件未找到: .cache/raft-things.pth\n"
    "请确保权重文件存在于 .cache 目录中"
)
```

### 2.2 导入错误 (`ImportError`)

**触发条件**：
- 无法导入模型模块
- 模型代码路径不正确

**示例**：
```python
ImportError(
    "无法导入 aux_motion_intensity 模块中的 SimpleRAFT\n"
    "请确保 aux_motion_intensity 模块存在且可访问"
)
```

### 2.3 运行时错误 (`RuntimeError`)

**触发条件**：
- 模型初始化失败
- 模型推理失败
- 验证过程失败

**示例**：
```python
RuntimeError(
    "SimpleRAFT初始化失败: {error}\n"
    "模型路径: {path}\n"
    "设备: {device}\n"
    "方法: {method}"
)
```

### 2.4 值错误 (`ValueError`)

**触发条件**：
- 参数值不正确
- 缺少必需的参数

**示例**：
```python
ValueError(
    "无法获取视频tensor进行Co-Tracker验证\n"
    "请提供 video_frames 或 video_tensor 参数"
)
```

---

## 三、各模块的错误处理

### 3.1 配置模块 (`core/config.py`)

**检查项**：
- ? RAFT模型权重文件
- ? Grounding DINO权重文件
- ? Grounding DINO配置文件
- ? BERT模型路径
- ? SAM权重文件
- ? SAM2配置文件（如果使用SAM2）
- ? Co-Tracker权重文件（如果启用验证）

**异常示例**：
```python
# RAFT模型文件未找到
if not raft_cache_path.exists():
    raise FileNotFoundError(
        f"RAFT模型文件未找到: {raft_cache_path}\n"
        f"请确保权重文件存在于 .cache 目录中"
    )
```

### 3.2 RAFT包装器 (`motion_flow/raft_wrapper.py`)

**检查项**：
- ? SimpleRAFT模块导入
- ? RAFT模型权重文件
- ? SimpleRAFT初始化

**异常示例**：
```python
# 模块导入失败
if not HAS_AUX_MOTION or SimpleRAFT is None:
    raise ImportError(
        "无法导入 aux_motion_intensity 模块中的 SimpleRAFT\n"
        "请确保 aux_motion_intensity 模块存在且可访问"
    )

# 模型未初始化
if self.raft_model is None:
    raise RuntimeError(
        "RAFT模型未初始化\n"
        "请检查模型路径是否正确，以及 aux_motion_intensity 模块是否可用"
    )
```

### 3.3 Co-Tracker验证器 (`instance_tracking/cotracker_validator.py`)

**检查项**：
- ? Co-Tracker模块导入
- ? Co-Tracker权重文件
- ? Co-Tracker初始化
- ? 验证过程

**异常示例**：
```python
# 模块导入失败
if not HAS_COTRACKER or CoTrackerPredictor is None:
    raise ImportError(
        "无法导入Co-Tracker模块\n"
        "请确保 third_party/co-tracker 目录存在且可访问"
    )

# 模型文件未找到
if not Path(checkpoint_path).exists():
    raise FileNotFoundError(
        f"Co-Tracker模型文件未找到: {checkpoint_path}\n"
        f"请确保权重文件存在于 .cache 目录中"
    )
```

### 3.4 异常过滤器 (`fusion/anomaly_filter.py`)

**检查项**：
- ? Co-Tracker验证器初始化
- ? 视频tensor可用性
- ? 异常掩码信息

**异常示例**：
```python
# 验证器未初始化
if self.cotracker_validator is None:
    raise RuntimeError(
        "Co-Tracker验证器未初始化\n"
        "请确保Co-Tracker验证器已正确初始化"
    )

# 无法获取视频tensor
if video_tensor is None:
    raise ValueError(
        "无法获取视频tensor进行Co-Tracker验证\n"
        "请提供 video_frames 或 video_tensor 参数"
    )
```

---

## 四、使用建议

### 4.1 确保所有文件存在

在使用前，请确保以下文件存在：

```python
# 检查必需的文件
required_files = {
    'raft': '.cache/raft-things.pth',
    'grounding_dino': '.cache/groundingdino_swinb_cogcoor.pth',
    'sam2': '.cache/sam2.1_hiera_large.pt',
    'cotracker': '.cache/scaled_offline.pth',
    'bert': '.cache/google-bert/bert-base-uncased'
}

for name, path in required_files.items():
    if not Path(path).exists():
        print(f"警告: {name} 文件不存在: {path}")
```

### 4.2 异常处理

在使用时，建议使用try-except捕获异常：

```python
try:
    config = get_default_config()
    analyzer = TemporalReasoningAnalyzer(config)
    analyzer.initialize()
    result = analyzer.analyze(video_frames)
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    print("请确保所有模型文件存在于正确的路径")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有模型代码存在于 third_party 目录")
except RuntimeError as e:
    print(f"运行时错误: {e}")
    print("请检查模型初始化和配置")
```

### 4.3 禁用可选功能

如果需要禁用某些功能（如Co-Tracker验证），可以设置配置：

```python
config = get_default_config()
config.tracker.enable_cotracker_validation = False  # 禁用Co-Tracker验证
```

---

## 五、错误信息格式

### 5.1 标准格式

所有异常信息遵循以下格式：

```
错误类型: 错误描述
详细信息1
详细信息2
解决建议
```

### 5.2 示例

```python
FileNotFoundError(
    "RAFT模型文件未找到: .cache/raft-things.pth\n"
    "请确保权重文件存在于 .cache 目录中"
)
```

---

## 六、总结

### 6.1 改进点

- ? **严格验证**：所有必需文件必须存在
- ? **清晰错误**：异常信息详细明确
- ? **快速失败**：问题立即发现，不隐藏错误

### 6.2 优势

- ? **更容易调试**：问题立即暴露
- ? **更可靠**：不会使用不完整的配置运行
- ? **更清晰**：错误信息明确，易于解决

### 6.3 注意事项

- ?? **必须确保所有文件存在**：使用前检查文件
- ?? **必须处理异常**：使用try-except捕获异常
- ?? **必须提供完整配置**：所有必需路径必须正确

---

**文档结束**

