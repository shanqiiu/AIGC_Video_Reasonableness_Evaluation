# 时序合理性分析模块实现指南

> **版本**：1.0  
> **日期**：2025年10月30日  
> **目标**：指导时序合理性分析模块的具体实现

---

## 一、实现概览

本文档提供时序合理性分析模块的详细实现指南，包括代码结构、实现步骤、关键函数和测试策略。

### 1.1 实现目标

基于技术方案文档，实现以下核心功能：

1. **光流分析子模块**：基于RAFT的光流计算和运动平滑度评估
2. **实例追踪子模块**：基于Grounded-SAM的实例分割和跨帧追踪
3. **关键点分析子模块**：基于MediaPipe/mmpose的关键点提取和生理动作分析
4. **融合决策引擎**：多模态特征融合和异常决策

### 1.2 实现顺序建议

**阶段1：基础框架搭建**（优先级：高）
- 创建模块结构和接口
- 实现视频加载和预处理
- 实现配置管理

**阶段2：单模态实现**（优先级：高）
- 实现光流分析子模块
- 实现实例追踪子模块（简化版）
- 实现关键点分析子模块（简化版）

**阶段3：融合机制**（优先级：中）
- 实现多模态特征对齐
- 实现异常一致性验证
- 实现融合决策引擎

**阶段4：优化与完善**（优先级：中）
- 性能优化
- 精度优化
- 错误处理和日志

**阶段5：测试与集成**（优先级：高）
- 单元测试
- 集成测试
- 端到端测试

---

## 二、代码结构设计

### 2.1 目录结构

```
src/temporal_reasoning/
├── __init__.py                    # 模块导出
├── README.md                      # 模块说明
├── TECHNICAL_DESIGN.md            # 技术方案文档
├── IMPLEMENTATION_GUIDE.md        # 实现指南（本文档）
│
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── temporal_analyzer.py      # 主分析器类
│   └── config.py                  # 配置管理
│
├── motion_flow/                   # 光流分析子模块
│   ├── __init__.py
│   ├── raft_wrapper.py           # RAFT封装
│   ├── flow_analyzer.py           # 光流分析器
│   └── motion_smoothness.py       # 运动平滑度计算
│
├── instance_tracking/             # 实例追踪子模块
│   ├── __init__.py
│   ├── grounded_sam_wrapper.py   # Grounded-SAM封装
│   ├── tracker_wrapper.py         # DeAOT/Co-Tracker封装
│   ├── instance_analyzer.py       # 实例分析器
│   └── structure_stability.py     # 结构稳定性计算
│
├── keypoint_analysis/             # 关键点分析子模块
│   ├── __init__.py
│   ├── keypoint_extractor.py      # 关键点提取器
│   ├── keypoint_analyzer.py       # 关键点分析器
│   └── physiological_analysis.py  # 生理动作分析
│
├── fusion/                        # 融合决策引擎
│   ├── __init__.py
│   ├── feature_alignment.py       # 特征对齐
│   ├── anomaly_fusion.py          # 异常融合
│   └── decision_engine.py         # 决策引擎
│
├── utils/                         # 工具函数
│   ├── __init__.py
│   ├── video_utils.py             # 视频处理工具
│   ├── visualization.py           # 可视化工具
│   └── metrics.py                 # 评估指标
│
└── tests/                         # 测试代码
    ├── __init__.py
    ├── test_motion_flow.py
    ├── test_instance_tracking.py
    ├── test_keypoint_analysis.py
    └── test_fusion.py
```

### 2.2 核心类设计

#### 2.2.1 TemporalReasoningAnalyzer（主分析器）

```python
# core/temporal_analyzer.py
class TemporalReasoningAnalyzer:
    """
    时序合理性分析器主类
    """
    def __init__(self, config: TemporalReasoningConfig):
        """初始化分析器"""
        self.config = config
        self.motion_analyzer = None
        self.instance_analyzer = None
        self.keypoint_analyzer = None
        self.fusion_engine = None
    
    def initialize(self):
        """初始化所有子模块"""
        pass
    
    def analyze(self, video_frames, text_prompts=None):
        """分析视频时序合理性"""
        pass
```

#### 2.2.2 MotionFlowAnalyzer（光流分析器）

```python
# motion_flow/flow_analyzer.py
class MotionFlowAnalyzer:
    """光流分析器"""
    def __init__(self, config):
        self.raft_model = None
    
    def compute_optical_flow(self, frame1, frame2):
        """计算光流"""
        pass
    
    def analyze_motion_smoothness(self, video_frames):
        """分析运动平滑度"""
        pass
    
    def detect_motion_discontinuities(self, optical_flows):
        """检测运动突变"""
        pass
```

#### 2.2.3 InstanceTrackingAnalyzer（实例追踪分析器）

```python
# instance_tracking/instance_analyzer.py
class InstanceTrackingAnalyzer:
    """实例追踪分析器"""
    def __init__(self, config):
        self.grounding_dino = None
        self.sam_model = None
        self.tracker = None
    
    def detect_instances(self, frame, text_prompts):
        """检测实例"""
        pass
    
    def track_instances(self, video_frames, detections):
        """追踪实例"""
        pass
    
    def analyze_structure_stability(self, tracked_instances):
        """分析结构稳定性"""
        pass
```

---

## 三、实现步骤详解

### 3.1 阶段1：基础框架搭建

#### 步骤1.1：创建配置管理模块

**文件**：`core/config.py`

```python
# -*- coding: utf-8 -*-
"""
时序合理性分析模块配置管理
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class RAFTConfig:
    """RAFT光流配置"""
    model_path: str = "path/to/raft_model.pth"
    model_type: str = "large"  # large or small
    use_gpu: bool = True
    batch_size: int = 1


@dataclass
class GroundingDINOConfig:
    """Grounding DINO配置"""
    model_path: str = "path/to/grounding_dino.pth"
    text_threshold: float = 0.25
    box_threshold: float = 0.3
    use_gpu: bool = True


@dataclass
class SAMConfig:
    """SAM配置"""
    model_path: str = "path/to/sam2_model.pth"
    model_type: str = "sam2_h"  # sam2_h, sam2_l, sam2_b
    use_gpu: bool = True


@dataclass
class TrackerConfig:
    """追踪器配置"""
    type: str = "deaot"  # deaot or cotracker
    model_path: Optional[str] = None
    use_gpu: bool = True


@dataclass
class KeypointConfig:
    """关键点配置"""
    model_type: str = "mediapipe"  # mediapipe or mmpose
    model_path: Optional[str] = None
    use_gpu: bool = False  # MediaPipe不支持GPU


@dataclass
class FusionConfig:
    """融合配置"""
    multimodal_confidence_boost: float = 1.2
    min_anomaly_duration_frames: int = 3
    single_modality_confidence_threshold: float = 0.8


@dataclass
class ThresholdsConfig:
    """阈值配置"""
    motion_discontinuity_threshold: float = 0.3
    structure_disappearance_threshold: float = 0.3  # 掩码面积变化率
    keypoint_displacement_threshold: int = 10  # 像素


@dataclass
class TemporalReasoningConfig:
    """时序合理性分析配置"""
    device: str = "cuda:0"
    
    # 子模块配置
    raft: RAFTConfig = None
    grounding_dino: GroundingDINOConfig = None
    sam: SAMConfig = None
    tracker: TrackerConfig = None
    keypoint: KeypointConfig = None
    fusion: FusionConfig = None
    thresholds: ThresholdsConfig = None
    
    def __post_init__(self):
        """初始化默认配置"""
        if self.raft is None:
            self.raft = RAFTConfig()
        if self.grounding_dino is None:
            self.grounding_dino = GroundingDINOConfig()
        if self.sam is None:
            self.sam = SAMConfig()
        if self.tracker is None:
            self.tracker = TrackerConfig()
        if self.keypoint is None:
            self.keypoint = KeypointConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.thresholds is None:
            self.thresholds = ThresholdsConfig()


def load_config_from_yaml(config_path: str) -> TemporalReasoningConfig:
    """从YAML文件加载配置"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 构建配置对象
    config = TemporalReasoningConfig(**config_dict.get('temporal_reasoning', {}))
    return config
```

#### 步骤1.2：创建主分析器框架

**文件**：`core/temporal_analyzer.py`

```python
# -*- coding: utf-8 -*-
"""
时序合理性分析器主类
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from .config import TemporalReasoningConfig
from ..motion_flow.flow_analyzer import MotionFlowAnalyzer
from ..instance_tracking.instance_analyzer import InstanceTrackingAnalyzer
from ..keypoint_analysis.keypoint_analyzer import KeypointAnalyzer
from ..fusion.decision_engine import FusionDecisionEngine


class TemporalReasoningAnalyzer:
    """
    时序合理性分析器
    """
    
    def __init__(self, config: TemporalReasoningConfig):
        """
        初始化分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.motion_analyzer = None
        self.instance_analyzer = None
        self.keypoint_analyzer = None
        self.fusion_engine = None
    
    def initialize(self):
        """初始化所有子模块"""
        print("正在初始化时序合理性分析器...")
        
        # 初始化光流分析器
        print("初始化光流分析器...")
        self.motion_analyzer = MotionFlowAnalyzer(self.config.raft)
        self.motion_analyzer.initialize()
        
        # 初始化实例追踪分析器
        print("初始化实例追踪分析器...")
        self.instance_analyzer = InstanceTrackingAnalyzer(
            self.config.grounding_dino,
            self.config.sam,
            self.config.tracker
        )
        self.instance_analyzer.initialize()
        
        # 初始化关键点分析器
        print("初始化关键点分析器...")
        self.keypoint_analyzer = KeypointAnalyzer(self.config.keypoint)
        self.keypoint_analyzer.initialize()
        
        # 初始化融合决策引擎
        print("初始化融合决策引擎...")
        self.fusion_engine = FusionDecisionEngine(self.config.fusion)
        
        print("时序合理性分析器初始化完成！")
    
    def analyze(
        self,
        video_frames: List[np.ndarray],
        text_prompts: Optional[List[str]] = None
    ) -> Dict:
        """
        分析视频时序合理性
        
        Args:
            video_frames: 视频帧序列，每帧为RGB图像 (H, W, 3)
            text_prompts: 可选文本提示列表（如["tongue", "finger"]）
        
        Returns:
            dict: {
                'motion_reasonableness_score': float,  # 0-1
                'structure_stability_score': float,    # 0-1
                'anomalies': List[dict],               # 异常实例列表
            }
        """
        if not self.motion_analyzer or not self.instance_analyzer:
            raise RuntimeError("分析器未初始化，请先调用 initialize()")
        
        # 1. 光流分析
        motion_score, motion_anomalies = self.motion_analyzer.analyze(video_frames)
        
        # 2. 实例追踪分析
        structure_score, structure_anomalies = self.instance_analyzer.analyze(
            video_frames, text_prompts
        )
        
        # 3. 关键点分析
        physiological_score, physiological_anomalies = self.keypoint_analyzer.analyze(
            video_frames
        )
        
        # 4. 多模态融合
        fused_anomalies = self.fusion_engine.fuse(
            motion_anomalies,
            structure_anomalies,
            physiological_anomalies
        )
        
        # 5. 计算最终得分
        final_motion_score, final_structure_score = self.fusion_engine.compute_final_scores(
            motion_score,
            structure_score,
            physiological_score,
            fused_anomalies
        )
        
        return {
            'motion_reasonableness_score': final_motion_score,
            'structure_stability_score': final_structure_score,
            'anomalies': fused_anomalies
        }
```

#### 步骤1.3：创建视频工具模块

**文件**：`utils/video_utils.py`

```python
# -*- coding: utf-8 -*-
"""
视频处理工具函数
"""

import cv2
import numpy as np
from typing import List, Optional
from pathlib import Path


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    加载视频帧
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数，None表示加载所有帧
    
    Returns:
        视频帧列表，每帧为RGB图像 (H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR转RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    return frames


def resize_frames(frames: List[np.ndarray], target_size: tuple) -> List[np.ndarray]:
    """
    调整帧大小
    
    Args:
        frames: 视频帧列表
        target_size: 目标尺寸 (width, height)
    
    Returns:
        调整后的帧列表
    """
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        resized_frames.append(resized)
    return resized_frames


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        视频信息字典
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info
```

---

### 3.2 阶段2：单模态实现

#### 步骤2.1：实现光流分析子模块

**文件**：`motion_flow/raft_wrapper.py`

```python
# -*- coding: utf-8 -*-
"""
RAFT光流模型封装
"""

import torch
import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# 添加RAFT路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_RAFT_PATH = _PROJECT_ROOT / 'third_party' / 'RAFT'
if str(_RAFT_PATH) not in sys.path:
    sys.path.insert(0, str(_RAFT_PATH))


class RAFTWrapper:
    """RAFT光流模型封装"""
    
    def __init__(self, model_path: str, model_type: str = "large", device: str = "cuda:0"):
        """
        初始化RAFT模型
        
        Args:
            model_path: 模型路径
            model_type: 模型类型 ("large" or "small")
            device: 计算设备
        """
        self.device = device
        self.model_type = model_type
        
        # 加载RAFT模型
        # 注意：需要根据实际RAFT实现调整
        try:
            from raft import RAFT
            self.model = RAFT(model_path, model_type=model_type)
            self.model.to(device)
            self.model.eval()
        except ImportError:
            raise ImportError("RAFT模型未找到，请检查third_party/RAFT路径")
    
    def compute_flow(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算光流
        
        Args:
            image1: 第一帧图像 (H, W, 3) RGB
            image2: 第二帧图像 (H, W, 3) RGB
        
        Returns:
            (u, v): 光流场，u和v分别为x和y方向的光流 (H, W)
        """
        # 转换为torch tensor
        img1_tensor = self._preprocess(image1)
        img2_tensor = self._preprocess(image2)
        
        # 计算光流
        with torch.no_grad():
            flow = self.model(img1_tensor, img2_tensor)
        
        # 转换为numpy
        flow_np = flow[0].cpu().numpy()
        u = flow_np[0]  # x方向
        v = flow_np[1]  # y方向
        
        return u, v
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 转换为torch tensor并归一化
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
```

**文件**：`motion_flow/flow_analyzer.py`

```python
# -*- coding: utf-8 -*-
"""
光流分析器
"""

import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

from .raft_wrapper import RAFTWrapper
from .motion_smoothness import compute_motion_smoothness, detect_motion_discontinuities


class MotionFlowAnalyzer:
    """光流分析器"""
    
    def __init__(self, config):
        """
        初始化光流分析器
        
        Args:
            config: RAFTConfig配置对象
        """
        self.config = config
        self.raft_model = None
    
    def initialize(self):
        """初始化RAFT模型"""
        self.raft_model = RAFTWrapper(
            model_path=self.config.model_path,
            model_type=self.config.model_type,
            device=self.config.use_gpu if self.config.use_gpu else "cpu"
        )
    
    def analyze(self, video_frames: List[np.ndarray]) -> Tuple[float, List[Dict]]:
        """
        分析视频运动平滑度
        
        Args:
            video_frames: 视频帧序列
        
        Returns:
            (motion_score, anomalies): 运动合理性得分和异常列表
        """
        # 1. 计算光流序列
        optical_flows = []
        for i in tqdm(range(len(video_frames) - 1), desc="计算光流"):
            u, v = self.raft_model.compute_flow(video_frames[i], video_frames[i+1])
            optical_flows.append((u, v))
        
        # 2. 计算运动平滑度
        motion_smoothness = compute_motion_smoothness(optical_flows)
        
        # 3. 检测运动突变
        motion_anomalies = detect_motion_discontinuities(
            optical_flows,
            threshold=self.config.motion_discontinuity_threshold
        )
        
        # 4. 计算得分
        base_score = np.mean(motion_smoothness)
        anomaly_penalty = len(motion_anomalies) * 0.1  # 每个异常扣10%
        final_score = max(0.0, base_score * (1.0 - anomaly_penalty))
        
        return final_score, motion_anomalies
```

**文件**：`motion_flow/motion_smoothness.py`

```python
# -*- coding: utf-8 -*-
"""
运动平滑度计算
"""

import numpy as np
from typing import List, Tuple, Dict


def compute_flow_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """计算光流幅值"""
    return np.sqrt(u**2 + v**2)


def compute_flow_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """计算光流方向"""
    return np.arctan2(v, u)


def compute_motion_smoothness(optical_flows: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
    """
    计算运动平滑度
    
    Args:
        optical_flows: 光流序列，每个元素为(u, v)元组
    
    Returns:
        平滑度分数列表，每个元素对应相邻帧对的平滑度
    """
    smoothness_scores = []
    
    for i in range(len(optical_flows) - 1):
        u1, v1 = optical_flows[i]
        u2, v2 = optical_flows[i+1]
        
        # 计算光流差异
        du = u2 - u1
        dv = v2 - v1
        flow_diff = np.sqrt(du**2 + dv**2)
        
        # 归一化为平滑度分数 (0-1)
        max_diff = np.percentile(flow_diff, 95)  # 使用95分位数作为归一化基准
        smoothness = 1.0 - np.clip(flow_diff / (max_diff + 1e-6), 0, 1)
        smoothness_scores.append(np.mean(smoothness))
    
    return smoothness_scores


def detect_motion_discontinuities(
    optical_flows: List[Tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.3
) -> List[Dict]:
    """
    检测运动突变
    
    Args:
        optical_flows: 光流序列
        threshold: 突变阈值
    
    Returns:
        异常列表
    """
    anomalies = []
    
    for i in range(len(optical_flows) - 1):
        u1, v1 = optical_flows[i]
        u2, v2 = optical_flows[i+1]
        
        # 计算光流变化率
        flow1_mag = compute_flow_magnitude(u1, v1)
        flow2_mag = compute_flow_magnitude(u2, v2)
        
        # 计算相对变化率
        change_rate = np.abs(flow2_mag - flow1_mag) / (flow1_mag + 1e-6)
        max_change = np.max(change_rate)
        
        if max_change > threshold:
            anomalies.append({
                'type': 'motion_discontinuity',
                'frame_id': i,
                'timestamp': f"{i/30:.2f}s",  # 假设30fps
                'confidence': min(1.0, max_change / threshold),
                'description': f"第{i}帧检测到运动突变"
            })
    
    return anomalies
```

#### 步骤2.2：实现实例追踪子模块（简化版）

**文件**：`instance_tracking/grounded_sam_wrapper.py`

```python
# -*- coding: utf-8 -*-
"""
Grounded-SAM封装
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# 添加Grounded-SAM路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_GROUNDED_SAM_PATH = _PROJECT_ROOT / 'third_party' / 'Grounded-SAM-2'
if str(_GROUNDED_SAM_PATH) not in sys.path:
    sys.path.insert(0, str(_GROUNDED_SAM_PATH))


class GroundedSAMWrapper:
    """Grounded-SAM封装"""
    
    def __init__(self, gdino_config, sam_config, device: str = "cuda:0"):
        """
        初始化Grounded-SAM
        
        Args:
            gdino_config: GroundingDINO配置
            sam_config: SAM配置
            device: 计算设备
        """
        self.device = device
        self.gdino_config = gdino_config
        self.sam_config = sam_config
        
        # 初始化模型（需要根据实际实现调整）
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        # 这里需要根据实际的Grounded-SAM-2实现进行调用
        # 示例代码框架
        try:
            # 导入Grounded-SAM-2的相关模块
            # from groundingdino.models import build_model
            # from sam2.build_sam import build_sam2
            pass
        except ImportError:
            raise ImportError("Grounded-SAM-2未找到，请检查third_party/Grounded-SAM-2路径")
    
    def detect_and_segment(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        检测和分割实例
        
        Args:
            image: 输入图像 (H, W, 3) RGB
            text_prompts: 文本提示列表
        
        Returns:
            掩码列表，每个元素为(mask, confidence)元组
        """
        # 实现检测和分割逻辑
        # 1. 使用Grounding DINO检测边界框
        # 2. 使用SAM分割掩码
        masks = []
        # ... 实现细节
        return masks
```

#### 步骤2.3：实现关键点分析子模块（简化版）

**文件**：`keypoint_analysis/keypoint_extractor.py`

```python
# -*- coding: utf-8 -*-
"""
关键点提取器
"""

import numpy as np
from typing import List, Dict, Optional
import cv2


class MediaPipeKeypointExtractor:
    """基于MediaPipe的关键点提取器"""
    
    def __init__(self):
        """初始化MediaPipe"""
        try:
            import mediapipe as mp
            self.mp_holistic = mp.solutions.holistic
            self.mp_hands = mp.solutions.hands
            self.mp_face_mesh = mp.solutions.face_mesh
            
            # 初始化模型
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                refine_face_landmarks=True
            )
        except ImportError:
            raise ImportError("MediaPipe未安装，请运行: pip install mediapipe")
    
    def extract_keypoints(self, image: np.ndarray) -> Dict:
        """
        提取关键点
        
        Args:
            image: 输入图像 (H, W, 3) RGB
        
        Returns:
            关键点字典，包含身体、手部、面部关键点
        """
        results = self.holistic.process(image)
        
        keypoints = {
            'body': None,
            'left_hand': None,
            'right_hand': None,
            'face': None
        }
        
        # 提取身体关键点
        if results.pose_landmarks:
            keypoints['body'] = self._landmarks_to_array(results.pose_landmarks.landmark)
        
        # 提取手部关键点
        if results.left_hand_landmarks:
            keypoints['left_hand'] = self._landmarks_to_array(results.left_hand_landmarks.landmark)
        if results.right_hand_landmarks:
            keypoints['right_hand'] = self._landmarks_to_array(results.right_hand_landmarks.landmark)
        
        # 提取面部关键点
        if results.face_landmarks:
            keypoints['face'] = self._landmarks_to_array(results.face_landmarks.landmark)
        
        return keypoints
    
    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """将landmarks转换为numpy数组"""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
```

---

### 3.3 阶段3：融合机制实现

**文件**：`fusion/decision_engine.py`

```python
# -*- coding: utf-8 -*-
"""
融合决策引擎
"""

from typing import List, Dict, Tuple
import numpy as np

from .feature_alignment import align_anomalies_spatially_and_temporally
from .anomaly_fusion import fuse_multimodal_anomalies


class FusionDecisionEngine:
    """融合决策引擎"""
    
    def __init__(self, config):
        """
        初始化融合决策引擎
        
        Args:
            config: FusionConfig配置对象
        """
        self.config = config
    
    def fuse(
        self,
        motion_anomalies: List[Dict],
        structure_anomalies: List[Dict],
        physiological_anomalies: List[Dict]
    ) -> List[Dict]:
        """
        融合多模态异常
        
        Args:
            motion_anomalies: 光流异常列表
            structure_anomalies: 结构异常列表
            physiological_anomalies: 生理异常列表
        
        Returns:
            融合后的异常列表
        """
        # 1. 异常对齐
        aligned_anomalies = align_anomalies_spatially_and_temporally(
            motion_anomalies,
            structure_anomalies,
            physiological_anomalies
        )
        
        # 2. 多模态融合
        fused_anomalies = fuse_multimodal_anomalies(
            aligned_anomalies,
            multimodal_confidence_boost=self.config.multimodal_confidence_boost,
            single_modality_threshold=self.config.single_modality_confidence_threshold
        )
        
        # 3. 时序验证
        validated_anomalies = self._validate_temporal_consistency(fused_anomalies)
        
        return validated_anomalies
    
    def _validate_temporal_consistency(self, anomalies: List[Dict]) -> List[Dict]:
        """验证时序一致性"""
        # 过滤持续时间过短的异常
        validated = []
        for anomaly in anomalies:
            # 检查是否在连续多帧中出现
            # 这里需要根据实际实现进行判断
            validated.append(anomaly)
        return validated
    
    def compute_final_scores(
        self,
        motion_score: float,
        structure_score: float,
        physiological_score: float,
        fused_anomalies: List[Dict]
    ) -> Tuple[float, float]:
        """
        计算最终得分
        
        Args:
            motion_score: 运动得分
            structure_score: 结构得分
            physiological_score: 生理得分
            fused_anomalies: 融合后的异常列表
        
        Returns:
            (motion_reasonableness_score, structure_stability_score)
        """
        # 计算异常惩罚
        motion_anomaly_count = len([a for a in fused_anomalies if 'motion' in a.get('modalities', [])])
        structure_anomaly_count = len([a for a in fused_anomalies if 'structure' in a.get('modalities', [])])
        
        # 应用惩罚
        motion_reasonableness = motion_score * (1.0 - motion_anomaly_count * 0.1)
        structure_stability = structure_score * (1.0 - structure_anomaly_count * 0.1)
        
        return max(0.0, motion_reasonableness), max(0.0, structure_stability)
```

---

## 四、测试策略

### 4.1 单元测试

为每个子模块编写单元测试：

```python
# tests/test_motion_flow.py
import unittest
import numpy as np
from src.temporal_reasoning.motion_flow.flow_analyzer import MotionFlowAnalyzer

class TestMotionFlowAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MotionFlowAnalyzer(config)
        self.analyzer.initialize()
    
    def test_compute_optical_flow(self):
        # 测试光流计算
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        u, v = self.analyzer.raft_model.compute_flow(frame1, frame2)
        self.assertEqual(u.shape, (480, 640))
        self.assertEqual(v.shape, (480, 640))
```

### 4.2 集成测试

测试整个分析流程：

```python
# tests/test_temporal_analyzer.py
def test_full_analysis():
    """测试完整分析流程"""
    analyzer = TemporalReasoningAnalyzer(config)
    analyzer.initialize()
    
    # 加载测试视频
    video_frames = load_video_frames("test_video.mp4")
    
    # 执行分析
    result = analyzer.analyze(video_frames, text_prompts=["tongue"])
    
    # 验证输出
    assert 'motion_reasonableness_score' in result
    assert 'structure_stability_score' in result
    assert 'anomalies' in result
    assert 0 <= result['motion_reasonableness_score'] <= 1
    assert 0 <= result['structure_stability_score'] <= 1
```

---

## 五、实现注意事项

### 5.1 模型路径配置

确保所有第三方模型的路径配置正确：

```python
# 检查模型路径是否存在
def check_model_paths(config):
    """检查模型路径"""
    models = [
        (config.raft.model_path, "RAFT"),
        (config.grounding_dino.model_path, "Grounding DINO"),
        (config.sam.model_path, "SAM")
    ]
    
    for path, name in models:
        if not Path(path).exists():
            print(f"警告: {name}模型路径不存在: {path}")
```

### 5.2 内存管理

对于长视频，需要注意内存管理：

```python
# 使用生成器处理长视频
def process_video_in_chunks(video_frames, chunk_size=100):
    """分块处理视频"""
    for i in range(0, len(video_frames), chunk_size):
        chunk = video_frames[i:i+chunk_size]
        yield chunk
```

### 5.3 错误处理

添加完善的错误处理：

```python
try:
    result = analyzer.analyze(video_frames)
except Exception as e:
    print(f"分析过程中出现错误: {e}")
    # 记录错误日志
    # 返回默认值或部分结果
```

---

## 六、后续优化建议

### 6.1 性能优化

1. **批处理**：对光流计算进行批处理优化
2. **模型量化**：使用INT8量化减少计算量
3. **多GPU支持**：支持多GPU并行处理

### 6.2 精度优化

1. **多尺度分析**：使用多尺度特征提高检测精度
2. **时序建模**：使用Transformer等模型建模长时依赖
3. **后处理优化**：改进异常过滤和合并逻辑

### 6.3 功能扩展

1. **实时处理**：支持实时视频流处理
2. **交互式标注**：支持用户交互式标注和反馈
3. **更多异常类型**：扩展更多类型的时序异常检测

---

## 七、总结

本实现指南提供了时序合理性分析模块的详细实现步骤和代码框架。建议按照以下顺序实现：

1. **阶段1**：搭建基础框架（配置、主分析器、工具函数）
2. **阶段2**：实现单模态功能（光流、分割、关键点）
3. **阶段3**：实现融合机制
4. **阶段4**：优化和完善
5. **阶段5**：测试和集成

在实现过程中，建议：
- 先实现简化版本，验证整体流程
- 逐步添加完整功能
- 持续进行测试和优化
- 参考现有模块的实现风格

如有问题，请参考技术方案文档（`TECHNICAL_DESIGN.md`）获取更多技术细节。

---

**文档结束**

