# 集成说明

## 概述

本模块从 `VMBench_diy/perceptible_amplitude_score.py` 重构而来，已集成到 `AIGC_Video_Reasonableness_Evaluation` 项目中。

## 目录结构

```
src/aux_motion_intensity_2/
├── __init__.py                    # 模块入口
├── analyzer.py                    # 主分析器（PASAnalyzer）
├── scene_classifier.py            # 场景分类器
├── motion_calculator.py          # 运动计算工具
├── batch.py                       # 批量处理接口
├── README.md                      # 使用文档
└── INTEGRATION.md                # 本文档

scripts/aux_motion_intensity_2/
├── run_pas.py                     # 启动脚本
└── README.md                      # 脚本使用说明
```

## 主要改动

### 1. 模块化重构

原脚本 `perceptible_amplitude_score.py` (819行) 被拆分为多个子模块：

- **analyzer.py**: 核心分析逻辑，包含 `PASAnalyzer` 类
- **scene_classifier.py**: 场景分类逻辑（SceneClassifier）
- **motion_calculator.py**: 运动幅度计算函数
- **batch.py**: 批量处理接口

### 2. 路径调整

第三方库路径已调整为指向项目内的 `third_party` 目录：

```python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
gsa_path = os.path.join(project_root, "third_party", "Grounded-Segment-Anything")
```

**说明**：从 `src/aux_motion_intensity_2/analyzer.py` 向上两级（`../..`）到达项目根目录。

### 3. 导入方式调整

原脚本中的导入方式：
```python
import GroundingDINO.groundingdino.datasets.transforms as T
```

调整为：
```python
import groundingdino.datasets.transforms as T
```

### 4. 配置路径调整

模型配置文件路径从相对路径调整为绝对路径：
```python
self.config_file = os.path.join(
    gsa_path,
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinB.py"
)
```

### 5. API简化

原脚本包含大量可视化函数（`visualize_detection`, `visualize_masks`, `visualize_tracks` 等），
在新版本中被精简，保留核心功能。如需可视化，可参考原脚本。

### 6. 延迟加载

采用延迟加载策略，模型在首次调用 `analyze_video` 时加载，减少初始化开销。

## 使用对比

### 原版本使用方式

```bash
python VMBench_diy/perceptible_amplitude_score.py \
    --meta_info_path meta_info.json \
    --device cuda \
    --enable_scene_classification
```

### 新版本使用方式

#### 方式1：使用脚本

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path meta_info.json \
    --device cuda \
    --enable_scene_classification
```

#### 方式2：直接导入

```python
from src.aux_motion_intensity_2 import PASAnalyzer

analyzer = PASAnalyzer(device="cuda", enable_scene_classification=True)
result = analyzer.analyze_video("video.mp4", subject_noun="person")
```

## 依赖关系

```
PASAnalyzer
├── GroundingDINO (third_party/Grounded-Segment-Anything/GroundingDINO)
├── SAM (third_party/Grounded-Segment-Anything/segment_anything)
├── Co-Tracker (third_party/co-tracker)
├── SceneClassifier (本地)
└── motion_calculator (本地)
```

## 模型依赖

所有模型文件应放在 `.cache/` 目录下：

- `groundingdino_swinb_cogcoor.pth`
- `sam_vit_h_4b8939.pth`
- `scaled_offline.pth`
- `google-bert/bert-base-uncased/`

## 与原项目的关系

本模块作为 `aux_motion_intensity_2` 与现有的 `aux_motion_intensity` 模块并列存在，
可提供基于 Grounded-SAM + Co-Tracker 的另一种动态程度分析方案。

- **aux_motion_intensity**: 使用 RAFT 光流分析
- **aux_motion_intensity_2**: 使用 Grounded-SAM + Co-Tracker 分析

两者可以并行运行，提供互补的运动分析结果。

## 测试建议

1. 创建测试元信息文件
2. 运行脚本验证导入和模型加载
3. 检查输出结果是否符合预期
4. 对比与原版本的输出一致性

## 已知限制

1. 可视化功能未完全迁移（可按需添加）
2. 模型文件需要手动下载
3. 首次运行需要较长时间加载模型

## 后续优化

- [ ] 添加模型自动下载脚本
- [ ] 恢复可视化功能（可选）
- [ ] 添加单元测试
- [ ] 性能优化（批处理、多GPU支持等）

