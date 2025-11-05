# 复用 aux_motion_intensity_2 实现指南

## 已完成的工作

1. ? 创建了 `grounded_sam_wrapper.py` - 封装器类
2. ? 在 `instance_analyzer.py` 中导入了 `GroundedSAMWrapper`
3. ?? 需要手动修改 `instance_analyzer.py` 的两个方法

## 需要手动修改的部分

由于编码问题，需要手动修改 `instance_analyzer.py` 中的以下部分：

### 1. 修改 `initialize()` 方法

**位置：** 第 46-84 行

**替换内容：**

将第 50-53 行：
```python
            # 这里可以添加实际的模型初始化代码
            # 由于Grounded-SAM-2的集成较复杂，这里提供一个占位实?
            print("警告: Grounded-SAM-2模型初始化需要根据实际实现调?")
            print("实例追踪分析器使用简化实?")
```

替换为：
```python
            # 初始化 Grounded-SAM（复用 aux_motion_intensity_2 的实现）
            try:
                # 根据use_gpu配置正确设置设备字符串
                if self.gdino_config.use_gpu and torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    device = "cpu"
                
                self.grounded_sam_wrapper = GroundedSAMWrapper(
                    gdino_config_path=self.gdino_config.config_path,
                    gdino_checkpoint_path=self.gdino_config.model_path,
                    sam_checkpoint_path=self.sam_config.model_path,
                    device=device,
                    text_threshold=self.gdino_config.text_threshold,
                    box_threshold=self.gdino_config.box_threshold,
                    grid_size=self.tracker_config.grid_size
                )
                print("Grounded-SAM 初始化成功")
            except Exception as e:
                print(f"警告: Grounded-SAM 初始化失败: {e}")
                print("将使用简化实现")
                self.grounded_sam_wrapper = None
```

### 2. 修改 `detect_instances()` 方法

**位置：** 第 86-103 行

**替换内容：**

将第 101-103 行：
```python
        # 简化实现：返回空列?
        # 实际实现需要调用Grounded DINO + SAM
        return []
```

替换为：
```python
        if self.grounded_sam_wrapper is None:
            # 如果 Grounded-SAM 未初始化，返回空列表
            return []
        
        try:
            return self.grounded_sam_wrapper.detect_and_segment(image, text_prompts)
        except Exception as e:
            print(f"警告: 实例检测失败: {e}")
            return []
```

## 验证修改

修改完成后，运行以下代码验证：

```python
from src.temporal_reasoning.instance_tracking.instance_analyzer import InstanceTrackingAnalyzer
from src.temporal_reasoning.core.config import GroundingDINOConfig, SAMConfig, TrackerConfig

# 创建配置
gdino_config = GroundingDINOConfig()
sam_config = SAMConfig()
tracker_config = TrackerConfig()

# 创建分析器
analyzer = InstanceTrackingAnalyzer(gdino_config, sam_config, tracker_config)

# 初始化
analyzer.initialize()

# 测试检测
import numpy as np
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
masks = analyzer.detect_instances(test_image, ["person", "car"])
print(f"检测到 {len(masks)} 个实例")
```

## 文件结构

```
src/temporal_reasoning/instance_tracking/
├── instance_analyzer.py          # 主分析器（需要修改）
├── grounded_sam_wrapper.py        # 封装器（已创建）?
├── cotracker_validator.py         # Co-Tracker 验证器
└── INTEGRATION_GUIDE.md          # 本文件
```

## 工作原理

1. **GroundedSAMWrapper** 封装了 `aux_motion_intensity_2` 中的 `PASAnalyzer`
2. **InstanceTrackingAnalyzer** 使用 `GroundedSAMWrapper` 进行实例检测
3. 保持了接口一致性，便于后续迁移到 SAM2

## 注意事项

1. 确保 `aux_motion_intensity_2` 模块可用
2. 确保模型文件已下载到 `.cache` 目录
3. 如果初始化失败，会自动降级到简化实现

## 下一步

完成上述修改后，`instance_analyzer.py` 就可以使用 `aux_motion_intensity_2` 的实现进行实例检测了。

