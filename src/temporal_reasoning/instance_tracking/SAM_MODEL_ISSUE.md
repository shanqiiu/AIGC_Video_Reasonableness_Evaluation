# SAM 模型加载失败原因分析

## 问题概述

SAM 模型加载失败，而 Co-Tracker 正常初始化成功。

## 根本原因

### 模型版本不匹配

**问题：**
1. **配置期望：** `SAMConfig` 默认 `model_type = "sam2_h"`，期望使用 SAM2
2. **配置行为：** `config.py` 会优先查找 SAM2 权重文件 `sam2.1_hiera_large.pt`
3. **实际使用：** `PASAnalyzer`（aux_motion_intensity_2）使用的是 **SAM v1**，代码：
   ```python
   sam_version = "vit_h"
   self.sam_predictor = SamPredictor(
       sam_model_registry[sam_version](checkpoint=self.sam_checkpoint).to(self.device)
   )
   ```
4. **冲突：** 当传入 `sam2.1_hiera_large.pt`（SAM2 权重）给 `PASAnalyzer` 时，它尝试用 SAM v1 的代码加载 SAM2 的权重，导致失败

### 模型文件对比

**当前 .cache 目录中的文件：**
- `sam2.1_hiera_large.pt` - SAM2 权重文件（用于 SAM2）
- `sam_vit_h_4b8939.pth` - SAM v1 权重文件（用于 SAM v1）

**PASAnalyzer 需要：**
- `sam_vit_h_4b8939.pth` - SAM v1 权重文件

**但传入的可能是：**
- `sam2.1_hiera_large.pt` - SAM2 权重文件（不兼容）

## 解决方案

### 方案 1: 自动检测并使用正确的模型文件（推荐）

修改 `GroundedSAMWrapper.__init__`，自动检测模型文件类型：

```python
def __init__(self, ...):
    # ... 现有代码 ...
    
    # 检查 SAM 模型文件类型
    sam_checkpoint_path_resolved = Path(sam_checkpoint_path)
    
    # 如果传入的是 SAM2 文件，但 PASAnalyzer 需要 SAM v1，自动查找 SAM v1 文件
    if sam_checkpoint_path_resolved.suffix == '.pt' or 'sam2' in sam_checkpoint_path_resolved.name.lower():
        # 这是 SAM2 文件，但 PASAnalyzer 需要 SAM v1
        # 查找 SAM v1 文件
        project_root = Path(__file__).parent.parent.parent.parent
        sam_v1_path = project_root / ".cache" / "sam_vit_h_4b8939.pth"
        
        if sam_v1_path.exists():
            print(f"警告: 检测到 SAM2 模型文件，但 PASAnalyzer 需要 SAM v1")
            print(f"自动切换到 SAM v1 文件: {sam_v1_path}")
            sam_checkpoint_path = str(sam_v1_path)
        else:
            raise FileNotFoundError(
                f"SAM v1 模型文件未找到: {sam_v1_path}\n"
                f"PASAnalyzer 需要 SAM v1 (sam_vit_h_4b8939.pth)，而不是 SAM2\n"
                f"请下载 SAM v1 模型文件到 .cache 目录"
            )
    
    self.pas_analyzer = PASAnalyzer(
        ...
        sam_checkpoint=sam_checkpoint_path,  # 使用修正后的路径
        ...
    )
```

### 方案 2: 修改配置，强制使用 SAM v1

在初始化时，确保使用 SAM v1 的路径：

```python
# 在 instance_analyzer.py 中
sam_config.model_path = str(project_root / ".cache" / "sam_vit_h_4b8939.pth")
sam_config.model_type = "sam_v1_h"  # 明确指定 SAM v1
```

### 方案 3: 修改 PASAnalyzer 以支持 SAM2（不推荐）

这需要大量修改 `aux_motion_intensity_2` 的代码，不推荐。

## 推荐修复代码

修改 `grounded_sam_wrapper.py`：

```python
def __init__(
    self,
    gdino_config_path: str,
    gdino_checkpoint_path: str,
    sam_checkpoint_path: str,
    device: str = "cuda:0",
    text_threshold: float = 0.25,
    box_threshold: float = 0.3,
    grid_size: int = 30
):
    # ... 现有代码 ...
    
    # 检查并修正 SAM 模型路径
    sam_checkpoint_path_resolved = Path(sam_checkpoint_path)
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 如果传入的是 SAM2 文件，但 PASAnalyzer 需要 SAM v1，自动查找 SAM v1 文件
    if sam_checkpoint_path_resolved.suffix == '.pt' or 'sam2' in sam_checkpoint_path_resolved.name.lower():
        # 这是 SAM2 文件，但 PASAnalyzer 需要 SAM v1
        sam_v1_path = project_root / ".cache" / "sam_vit_h_4b8939.pth"
        
        if sam_v1_path.exists():
            print(f"警告: 检测到 SAM2 模型文件 ({sam_checkpoint_path})")
            print(f"PASAnalyzer 需要 SAM v1，自动切换到: {sam_v1_path}")
            sam_checkpoint_path = str(sam_v1_path)
        else:
            raise FileNotFoundError(
                f"SAM v1 模型文件未找到: {sam_v1_path}\n"
                f"PASAnalyzer 需要 SAM v1 (sam_vit_h_4b8939.pth)，而不是 SAM2\n"
                f"当前传入的文件: {sam_checkpoint_path}\n"
                f"请下载 SAM v1 模型文件到 .cache 目录"
            )
    
    # ... 继续初始化 ...
```

## 错误信息示例

如果问题确实是因为模型版本不匹配，可能会看到类似这样的错误：

```
RuntimeError: SAM 模型加载失败: ...
FileNotFoundError: 无法加载模型权重
ValueError: 模型权重格式不匹配
```

## 验证步骤

1. **检查传入的模型路径：**
   ```python
   print(f"SAM 模型路径: {sam_checkpoint_path}")
   print(f"文件存在: {Path(sam_checkpoint_path).exists()}")
   print(f"文件扩展名: {Path(sam_checkpoint_path).suffix}")
   ```

2. **检查模型文件类型：**
   - SAM v1: `.pth` 文件，文件名包含 `sam_vit_h`
   - SAM2: `.pt` 文件，文件名包含 `sam2`

3. **确认 PASAnalyzer 使用的版本：**
   - 查看 `analyzer.py` 中的 `sam_model_registry["vit_h"]` - 这是 SAM v1

## 总结

- **问题根源：** SAM2 模型文件被传入给需要 SAM v1 的 `PASAnalyzer`
- **解决方案：** 自动检测并切换到 SAM v1 模型文件
- **修复状态：** 需要添加模型文件类型检测和自动切换逻辑

