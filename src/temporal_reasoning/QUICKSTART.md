# 时序合理性分析模块快速开始指南

> **目标**：快速上手时序合理性分析模块的实现

---

## 一、准备工作

### 1.1 环境要求

- Python >= 3.8
- PyTorch >= 1.12
- CUDA（如果使用GPU）
- 其他依赖包见 `requirements.txt`

### 1.2 模型准备

确保以下模型已下载并放置在正确路径：

1. **RAFT模型**：`third_party/RAFT/checkpoints/`
2. **Grounding DINO模型**：`third_party/Grounded-SAM-2/gdino_checkpoints/`
3. **SAM2模型**：`third_party/Grounded-SAM-2/checkpoints/`
4. **DeAOT/Co-Tracker模型**（可选）

### 1.3 项目结构

确保项目结构如下：

```
src/temporal_reasoning/
├── core/
│   ├── __init__.py
│   ├── temporal_analyzer.py
│   └── config.py
├── motion_flow/
│   ├── __init__.py
│   ├── raft_wrapper.py
│   ├── flow_analyzer.py
│   └── motion_smoothness.py
├── instance_tracking/
│   ├── __init__.py
│   ├── grounded_sam_wrapper.py
│   ├── tracker_wrapper.py
│   ├── instance_analyzer.py
│   └── structure_stability.py
├── keypoint_analysis/
│   ├── __init__.py
│   ├── keypoint_extractor.py
│   ├── keypoint_analyzer.py
│   └── physiological_analysis.py
├── fusion/
│   ├── __init__.py
│   ├── feature_alignment.py
│   ├── anomaly_fusion.py
│   └── decision_engine.py
└── utils/
    ├── __init__.py
    ├── video_utils.py
    ├── visualization.py
    └── metrics.py
```

---

## 二、实现步骤

### 步骤1：创建基础框架（1-2天）

**任务**：
1. 创建目录结构
2. 实现配置管理模块（`core/config.py`）
3. 实现主分析器框架（`core/temporal_analyzer.py`）
4. 实现视频工具模块（`utils/video_utils.py`）

**验证**：
```python
# 测试配置加载
from src.temporal_reasoning.core.config import TemporalReasoningConfig
config = TemporalReasoningConfig()
print(config.device)

# 测试视频加载
from src.temporal_reasoning.utils.video_utils import load_video_frames
frames = load_video_frames("test_video.mp4", max_frames=10)
print(f"加载了 {len(frames)} 帧")
```

### 步骤2：实现光流分析子模块（2-3天）

**任务**：
1. 实现RAFT封装（`motion_flow/raft_wrapper.py`）
2. 实现光流分析器（`motion_flow/flow_analyzer.py`）
3. 实现运动平滑度计算（`motion_flow/motion_smoothness.py`）

**验证**：
```python
# 测试光流计算
from src.temporal_reasoning.motion_flow.flow_analyzer import MotionFlowAnalyzer
from src.temporal_reasoning.core.config import RAFTConfig

config = RAFTConfig()
analyzer = MotionFlowAnalyzer(config)
analyzer.initialize()

# 测试视频帧
import numpy as np
frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

u, v = analyzer.raft_model.compute_flow(frame1, frame2)
print(f"光流形状: {u.shape}, {v.shape}")
```

### 步骤3：实现实例追踪子模块（3-4天）

**任务**：
1. 实现Grounded-SAM封装（`instance_tracking/grounded_sam_wrapper.py`）
2. 实现追踪器封装（`instance_tracking/tracker_wrapper.py`）
3. 实现实例分析器（`instance_tracking/instance_analyzer.py`）
4. 实现结构稳定性计算（`instance_tracking/structure_stability.py`）

**验证**：
```python
# 测试实例检测和分割
from src.temporal_reasoning.instance_tracking.instance_analyzer import InstanceTrackingAnalyzer

analyzer = InstanceTrackingAnalyzer(config)
analyzer.initialize()

# 测试检测
masks = analyzer.detect_instances(frame, text_prompts=["tongue"])
print(f"检测到 {len(masks)} 个实例")
```

### 步骤4：实现关键点分析子模块（2-3天）

**任务**：
1. 实现关键点提取器（`keypoint_analysis/keypoint_extractor.py`）
2. 实现关键点分析器（`keypoint_analysis/keypoint_analyzer.py`）
3. 实现生理动作分析（`keypoint_analysis/physiological_analysis.py`）

**验证**：
```python
# 测试关键点提取
from src.temporal_reasoning.keypoint_analysis.keypoint_extractor import MediaPipeKeypointExtractor

extractor = MediaPipeKeypointExtractor()
keypoints = extractor.extract_keypoints(frame)
print(f"提取到身体关键点: {keypoints['body'] is not None}")
```

### 步骤5：实现融合决策引擎（2-3天）

**任务**：
1. 实现特征对齐（`fusion/feature_alignment.py`）
2. 实现异常融合（`fusion/anomaly_fusion.py`）
3. 实现决策引擎（`fusion/decision_engine.py`）

**验证**：
```python
# 测试融合决策
from src.temporal_reasoning.fusion.decision_engine import FusionDecisionEngine

engine = FusionDecisionEngine(config)
fused_anomalies = engine.fuse(motion_anomalies, structure_anomalies, physiological_anomalies)
print(f"融合后异常数量: {len(fused_anomalies)}")
```

### 步骤6：集成测试（2-3天）

**任务**：
1. 实现端到端测试
2. 优化性能
3. 添加错误处理
4. 完善文档

**验证**：
```python
# 测试完整流程
from src.temporal_reasoning.core.temporal_analyzer import TemporalReasoningAnalyzer
from src.temporal_reasoning.core.config import TemporalReasoningConfig

config = TemporalReasoningConfig()
analyzer = TemporalReasoningAnalyzer(config)
analyzer.initialize()

# 加载测试视频
from src.temporal_reasoning.utils.video_utils import load_video_frames
video_frames = load_video_frames("test_video.mp4")

# 执行分析
result = analyzer.analyze(video_frames, text_prompts=["tongue"])

# 验证结果
print(f"运动合理性得分: {result['motion_reasonableness_score']:.2f}")
print(f"结构稳定性得分: {result['structure_stability_score']:.2f}")
print(f"异常数量: {len(result['anomalies'])}")
```

---

## 三、实现建议

### 3.1 分阶段实现

建议按照以下顺序实现：

1. **最小可行版本（MVP）**：
   - 只实现光流分析子模块
   - 验证整体流程
   - 确保基础框架正确

2. **功能扩展**：
   - 逐步添加其他子模块
   - 每次添加一个子模块并测试
   - 确保每个子模块独立工作

3. **融合优化**：
   - 实现融合机制
   - 优化性能和精度
   - 完善错误处理

### 3.2 测试策略

- **单元测试**：为每个子模块编写单元测试
- **集成测试**：测试子模块之间的交互
- **端到端测试**：测试完整流程
- **性能测试**：测试处理速度和内存使用

### 3.3 调试技巧

1. **使用小视频**：先用小视频测试，避免内存问题
2. **逐步验证**：每个步骤都验证输出
3. **日志记录**：添加详细的日志记录
4. **可视化**：使用可视化工具检查中间结果

---

## 四、常见问题

### Q1: 如何加载RAFT模型？

**A**: 需要根据RAFT的实际实现调整。参考实现：

```python
# 方法1：使用官方实现
from raft import RAFT
model = RAFT(model_path, model_type="large")

# 方法2：使用自定义加载
import torch
model = torch.load(model_path)
```

### Q2: 如何处理模型路径？

**A**: 使用相对路径或配置文件：

```python
# 使用项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "third_party" / "RAFT" / "checkpoints" / "raft_large.pth"
```

### Q3: 内存不足怎么办？

**A**: 使用以下策略：

1. **分块处理**：将视频分成多个块处理
2. **降低分辨率**：先降低分辨率分析，需要时再提高
3. **跳帧处理**：对长视频可以跳帧处理
4. **批处理优化**：控制批处理大小

### Q4: 如何调试异常检测？

**A**: 使用可视化工具：

```python
# 可视化光流
import matplotlib.pyplot as plt
plt.imshow(flow_magnitude)
plt.show()

# 可视化分割掩码
import cv2
mask_overlay = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
cv2.imshow("mask", mask_overlay)
```

---

## 五、参考资源

### 5.1 文档

- **技术方案文档**：`TECHNICAL_DESIGN.md`
- **实现指南**：`IMPLEMENTATION_GUIDE.md`
- **模块说明**：`README.md`

### 5.2 代码参考

- **perceptual_quality模块**：参考`src/perceptual_quality/blur_new/`的实现
- **其他模块**：参考`src/`下其他模块的实现风格

### 5.3 第三方库文档

- **RAFT**：https://github.com/princeton-vl/RAFT
- **Grounded-SAM-2**：https://github.com/IDEA-Research/Grounded-SAM-2
- **MediaPipe**：https://google.github.io/mediapipe/
- **DeAOT**：https://github.com/yoxu515/aot-benchmark

---

## 六、下一步

完成基础实现后，可以：

1. **性能优化**：优化计算速度和内存使用
2. **精度优化**：提高异常检测的准确率
3. **功能扩展**：添加更多异常类型和功能
4. **集成测试**：与整个系统集成测试

---

**祝实现顺利！**

