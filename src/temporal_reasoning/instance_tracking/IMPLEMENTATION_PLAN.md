# Grounded-SAM-2 实现方案

## 为什么是占位实现？

当前 `InstanceTrackingAnalyzer` 中的 Grounded-SAM-2 实现是占位实现，原因如下：

### 1. **集成复杂度高**
- 需要同时初始化 **Grounding DINO** 和 **SAM2** 两个模型
- 两个模型的 API 不同，需要协调处理
- 需要处理模型路径、配置文件路径等多个配置项

### 2. **依赖路径复杂**
- Grounding DINO 模型在 `.cache` 目录
- SAM2 模型也在 `.cache` 目录
- 配置文件在 `third_party/Grounded-SAM-2` 目录
- 代码模块也在 `third_party/Grounded-SAM-2` 目录

### 3. **视频追踪逻辑复杂**
- SAM2 视频追踪需要特定的初始化流程
- 需要处理视频帧序列
- 需要维护追踪状态

### 4. **错误处理复杂**
- 模型加载可能失败
- CUDA 内存可能不足
- 需要优雅降级

## 实现方案

### 方案概述

创建一个 `GroundedSAM2Wrapper` 封装类，封装 Grounding DINO 和 SAM2 的集成调用。

### 实现步骤

#### 步骤 1: 创建封装器类

已创建 `grounded_sam2_wrapper.py`，包含：
- `GroundedSAM2Wrapper` 类：封装模型初始化和调用
- `load_image_from_array` 函数：处理图像格式转换

#### 步骤 2: 集成到 InstanceTrackingAnalyzer

修改 `instance_analyzer.py`：

```python
from .grounded_sam2_wrapper import GroundedSAM2Wrapper

class InstanceTrackingAnalyzer:
    def __init__(self, ...):
        # ... 现有代码 ...
        self.grounded_sam2 = None
    
    def initialize(self):
        """初始化模型"""
        # 初始化 Grounded-SAM-2
        try:
            self.grounded_sam2 = GroundedSAM2Wrapper(
                gdino_config_path=self.gdino_config.config_path,
                gdino_checkpoint_path=self.gdino_config.model_path,
                sam2_config_path=self.sam_config.config_path,
                sam2_checkpoint_path=self.sam_config.model_path,
                device=self.device,
                text_threshold=self.gdino_config.text_threshold,
                box_threshold=self.gdino_config.box_threshold
            )
            print("Grounded-SAM-2 初始化成功")
        except Exception as e:
            print(f"警告: Grounded-SAM-2 初始化失败: {e}")
            print("将使用简化实现")
            self.grounded_sam2 = None
    
    def detect_instances(self, image, text_prompts):
        """检测和分割实例"""
        if self.grounded_sam2 is None:
            return []
        
        return self.grounded_sam2.detect_and_segment(image, text_prompts)
```

#### 步骤 3: 实现视频追踪

在 `GroundedSAM2Wrapper` 中实现 `track_instances` 方法：

```python
def track_instances(self, video_frames, initial_mask, query_frame=0):
    """
    追踪实例
    """
    # 1. 准备视频帧（保存为临时目录）
    temp_dir = Path("./temp_video_frames")
    temp_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(video_frames):
        frame_path = temp_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # 2. 初始化视频预测器状态
    inference_state = self.sam2_video_predictor.init_state(
        video_path=str(temp_dir)
    )
    
    # 3. 使用初始掩码进行追踪
    # 这里需要根据 SAM2 视频预测器的实际 API 实现
    
    # 4. 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    return tracked_results
```

### 配置要求

确保以下配置正确：

```python
# config.py 中的配置
grounding_dino:
    model_path: ".cache/groundingdino_swinb_cogcoor.pth"
    config_path: "third_party/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB.py"
    text_threshold: 0.25
    box_threshold: 0.3

sam:
    model_path: ".cache/sam2.1_hiera_large.pt"
    config_path: "third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    model_type: "sam2_h"
```

### 使用示例

```python
# 初始化
analyzer = InstanceTrackingAnalyzer(
    gdino_config=gdino_config,
    sam_config=sam_config,
    tracker_config=tracker_config
)
analyzer.initialize()

# 检测实例
image = np.array(...)  # (H, W, 3) RGB
text_prompts = ["person", "car"]
masks = analyzer.detect_instances(image, text_prompts)

# 追踪实例
video_frames = [...]  # 视频帧列表
tracked = analyzer.track_instances(video_frames, masks[0][0])
```

### 错误处理

实现应包含以下错误处理：

1. **模型加载失败**：降级到简化实现
2. **CUDA 内存不足**：自动切换到 CPU
3. **模块导入失败**：提供清晰的错误信息
4. **配置文件缺失**：提供默认配置路径

### 性能优化

1. **模型缓存**：避免重复加载模型
2. **批处理**：批量处理多个图像
3. **内存管理**：及时释放不需要的模型

### 测试建议

1. **单元测试**：测试各个方法
2. **集成测试**：测试完整流程
3. **性能测试**：测试处理速度
4. **错误测试**：测试各种错误情况

## 总结

Grounded-SAM-2 实现需要：
1. 创建封装器类（已完成）
2. 集成到 InstanceTrackingAnalyzer（待实现）
3. 实现视频追踪逻辑（待实现）
4. 添加错误处理和性能优化（待实现）

当前占位实现允许系统在其他模块正常工作的同时，逐步完善 Grounded-SAM-2 的集成。

