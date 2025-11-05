# 警告解释：无法导入 aux_motion_intensity_2 模块

## 警告信息

```
警告: Grounded-SAM 初始化失败: 无法导入 aux_motion_intensity_2 模块
请确保 src/aux_motion_intensity_2 目录存在且可访问
```

## 问题原因

### 1. 相对导入问题

**核心问题：** `aux_motion_intensity_2/analyzer.py` 使用了相对导入：

```python
from .motion_calculator import calculate_motion_degree, is_mask_suitable_for_tracking
from .scene_classifier import SceneClassifier
```

**问题分析：**
- 原始的 `grounded_sam_wrapper.py` 使用 `importlib.util.spec_from_file_location` 直接导入模块
- 这种方式导入的模块没有正确的包上下文（`__package__` 未设置）
- Python 无法解析相对导入（`.` 开头的导入），导致 `ImportError: attempted relative import with no known parent package`

### 2. 路径问题

**原始代码：**
```python
base_dir = Path(__file__).parent.parent.parent.parent
aux_motion_path = base_dir / "src" / "aux_motion_intensity_2"
sys.path.insert(0, str(aux_motion_path))
```

**问题：**
- 将 `aux_motion_intensity_2` 目录添加到 `sys.path`，但这不是包导入的正确方式
- 应该将 `src` 目录添加到 `sys.path`，然后使用 `import aux_motion_intensity_2.analyzer`

## 修复方案

### 方案 1: 使用包导入（推荐）

**修复后的代码：**
```python
# 将 src 目录添加到路径
base_dir = Path(__file__).parent.parent.parent.parent
src_path = base_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 使用包导入方式
try:
    import aux_motion_intensity_2.analyzer as analyzer_module
    PASAnalyzer = analyzer_module.PASAnalyzer
    get_grounding_output = analyzer_module.get_grounding_output
    HAS_AUX_MOTION = True
except ImportError:
    # 降级处理...
```

**优点：**
- ? 正确处理相对导入
- ? 符合 Python 包导入规范
- ? 代码更简洁

### 方案 2: 动态导入 + 设置包上下文（备用）

如果包导入失败，使用动态导入，但需要设置正确的包上下文：

```python
# 设置包上下文
analyzer_module.__package__ = "aux_motion_intensity_2"
analyzer_module.__file__ = str(analyzer_file)
analyzer_module.__name__ = "aux_motion_intensity_2.analyzer"
spec.loader.exec_module(analyzer_module)
```

## 修复后的完整代码

```python
try:
    # 添加项目路径
    base_dir = Path(__file__).parent.parent.parent.parent
    src_path = base_dir / "src"
    
    # 将 src 目录添加到路径，以便可以导入 aux_motion_intensity_2 包
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # 使用包导入方式，这样可以正确处理相对导入
    try:
        # 尝试作为包导入（推荐方式）
        import aux_motion_intensity_2.analyzer as analyzer_module
        PASAnalyzer = analyzer_module.PASAnalyzer
        get_grounding_output = analyzer_module.get_grounding_output
        HAS_AUX_MOTION = True
    except ImportError:
        # 如果包导入失败，尝试直接导入模块（需要处理相对导入）
        # ... 降级处理 ...
except (ImportError, FileNotFoundError, AttributeError) as e:
    HAS_AUX_MOTION = False
    print(f"警告: 无法导入 aux_motion_intensity_2 模块: {e}")
```

## 验证修复

修复后，应该能够成功导入：

```python
# 测试导入
from src.temporal_reasoning.instance_tracking.grounded_sam_wrapper import GroundedSAMWrapper
print("导入成功！")
```

## 其他可能的问题

### 1. 依赖缺失

如果仍然出现导入错误，可能是以下依赖缺失：
- `groundingdino` 模块（在 `third_party/Grounded-Segment-Anything`）
- `segment_anything` 模块（在 `third_party/Grounded-Segment-Anything`）
- `cotracker` 模块（在 `third_party/co-tracker`）

**解决方案：** 确保这些第三方库的路径已正确添加到 `sys.path`

### 2. 模型文件缺失

如果导入成功但初始化失败，可能是模型文件缺失：
- `.cache/groundingdino_swinb_cogcoor.pth`
- `.cache/sam_vit_h_4b8939.pth`
- `.cache/scaled_offline.pth`

**解决方案：** 确保模型文件已下载到 `.cache` 目录

## 总结

- **问题根源：** 相对导入需要正确的包上下文
- **解决方案：** 使用包导入方式（`import aux_motion_intensity_2.analyzer`）
- **修复状态：** ? 已修复

修复后，`Grounded-SAM` 应该能够成功初始化。

