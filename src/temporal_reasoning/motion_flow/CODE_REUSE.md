# 光流计算代码复用说明

## 复用情况

`motion_flow/raft_wrapper.py` 已更新为复用 `aux_motion_intensity` 模块中的 `SimpleRAFT` 实现。

## 优势

### 1. 经过验证的代码
- `SimpleRAFT` 已在项目中使用，经过测试和验证
- 支持多种光流方法：RAFT、Farneback、TV-L1
- 完善的错误处理和降级策略

### 2. 功能完整
- 自动查找模型路径
- 支持GPU/CPU自动切换
- CUDA兼容性测试和CPU降级
- 支持模型路径自动发现

### 3. 代码复用
- 避免重复实现
- 统一的光流计算接口
- 维护成本更低

## 使用方式

### 自动复用

`RAFTWrapper` 会自动尝试导入并使用 `SimpleRAFT`：

```python
from src.temporal_reasoning.motion_flow.raft_wrapper import RAFTWrapper

# 初始化（会自动使用 SimpleRAFT）
wrapper = RAFTWrapper(
    model_path="path/to/model.pth",
    model_type="large",
    device="cuda:0"
)

# 计算光流
u, v = wrapper.compute_flow(image1, image2)
```

### 降级策略

如果 `aux_motion_intensity` 模块不可用，会自动降级到 OpenCV 的 Farneback 方法：

```python
# 如果 SimpleRAFT 不可用，会自动使用简化实现
# 使用 OpenCV 的 Farneback 方法
```

## 实现细节

### 1. 导入逻辑

```python
try:
    # 添加项目路径
    base_dir = Path(__file__).parent.parent.parent.parent
    aux_motion_path = base_dir / "src" / "aux_motion_intensity"
    if str(aux_motion_path) not in sys.path:
        sys.path.insert(0, str(aux_motion_path))
    
    from flow_predictor import SimpleRAFT
    HAS_AUX_MOTION = True
except ImportError:
    HAS_AUX_MOTION = False
    print("警告: 无法导入 aux_motion_intensity 模块，使用简化实现")
```

### 2. 初始化逻辑

```python
# 选择方法：如果有模型路径且为large，使用raft；否则使用farneback
if self.model_path and model_type == "large" and HAS_AUX_MOTION:
    method = "raft"
else:
    method = "farneback"

# 初始化 SimpleRAFT
if HAS_AUX_MOTION:
    self.raft_model = SimpleRAFT(
        device=self.device,
        method=method,
        model_path=self.model_path if self.model_path else None
    )
```

### 3. 格式转换

`SimpleRAFT` 返回的光流格式是 `(2, H, W)`，需要转换为 `(H, W)` 格式：

```python
flow = self.raft_model.predict_flow(image1, image2)

# 转换为 (H, W) 格式
if flow.shape[0] == 2:
    u = flow[0]  # x方向
    v = flow[1]  # y方向
```

## 配置说明

### 模型路径

`SimpleRAFT` 会自动查找模型路径：

1. 如果提供了 `model_path`，使用提供的路径
2. 如果没有提供，尝试从 `third_party/pretrained_models/raft-things.pth` 加载
3. 如果模型不可用，降级到 Farneback 方法

### 方法选择

- **RAFT**: 需要模型文件，精度高，速度慢
- **Farneback**: 不需要模型，精度中等，速度快
- **TV-L1**: 不需要模型，精度高，速度慢

## 注意事项

1. **依赖关系**: 确保 `aux_motion_intensity` 模块在 Python 路径中
2. **模型文件**: 如果要使用 RAFT，需要下载对应的模型文件
3. **设备兼容性**: `SimpleRAFT` 会自动处理 GPU/CPU 兼容性问题
4. **错误处理**: 如果 `SimpleRAFT` 初始化失败，会自动降级到简化实现

## 优势总结

? **代码复用**: 避免重复实现，统一维护  
? **功能完整**: 支持多种光流方法  
? **错误处理**: 完善的降级策略  
? **自动适配**: 自动查找模型和路径  
? **兼容性好**: 支持 GPU/CPU 自动切换  

