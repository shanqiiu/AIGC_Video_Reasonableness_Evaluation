# 模型路径配置说明

> **更新日期**：2025年10月30日  
> **配置结构**：权重文件在 `.cache` 文件夹，模型代码在 `third_party` 文件夹

---

## 一、路径结构

### 1.1 项目结构

```
AIGC_Video_Reasonableness_Evaluation/
├── .cache/                    # 权重文件目录
│   ├── groundingdino_swinb_cogcoor.pth
│   ├── sam2.1_hiera_large.pt
│   ├── sam_vit_h_4b8939.pth
│   ├── scaled_offline.pth     # Co-Tracker权重
│   ├── raft-things.pth        # RAFT权重（如果存在）
│   └── google-bert/
│       └── bert-base-uncased/
│
└── third_party/               # 模型代码目录
    ├── RAFT/                  # RAFT代码
    ├── Grounded-SAM-2/        # Grounded-SAM-2代码
    ├── co-tracker/            # Co-Tracker代码
    └── ...
```

### 1.2 配置原则

- **权重文件**：优先从 `.cache` 目录查找
- **模型代码**：从 `third_party` 目录查找
- **配置文件**：从 `third_party` 目录查找

---

## 二、模型路径配置

### 2.1 RAFT配置

**权重路径**：
- 优先：`.cache/raft-things.pth`
- 备选：`third_party/pretrained_models/raft-things.pth`

**代码路径**：
- `third_party/RAFT/`

**配置示例**：
```python
config.raft.model_path = ".cache/raft-things.pth"  # 如果存在
# 或者使用默认路径（自动查找）
```

### 2.2 Grounding DINO配置

**权重路径**：
- `.cache/groundingdino_swinb_cogcoor.pth`

**配置文件路径**：
- `third_party/Grounded-SAM-2/grounding_dino/config/GroundingDINO_SwinB.py`
- 或 `third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py`

**BERT模型路径**：
- `.cache/google-bert/bert-base-uncased/`

**配置示例**：
```python
config.grounding_dino.model_path = ".cache/groundingdino_swinb_cogcoor.pth"
config.grounding_dino.config_path = "third_party/Grounded-SAM-2/grounding_dino/config/GroundingDINO_SwinB.py"
config.grounding_dino.bert_path = ".cache/google-bert/bert-base-uncased"
```

### 2.3 SAM配置

**权重路径**：
- 优先：`.cache/sam2.1_hiera_large.pt`（SAM2）
- 备选：`.cache/sam_vit_h_4b8939.pth`（SAM1）

**配置文件路径**（SAM2）：
- `third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`
- 或 `third_party/Grounded-SAM-2/sam2/configs/sam2_hiera_l.yaml`

**配置示例**：
```python
config.sam.model_path = ".cache/sam2.1_hiera_large.pt"
config.sam.config_path = "third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
config.sam.model_type = "sam2_h"
```

### 2.4 Co-Tracker配置

**权重路径**：
- `.cache/scaled_offline.pth`

**代码路径**：
- `third_party/co-tracker/`

**配置示例**：
```python
config.tracker.cotracker_checkpoint = ".cache/scaled_offline.pth"
config.tracker.enable_cotracker_validation = True
config.tracker.grid_size = 30
```

---

## 三、自动路径查找

### 3.1 配置初始化

配置类会自动查找模型路径：

```python
from src.temporal_reasoning import get_default_config

config = get_default_config()
# 自动查找并设置所有模型路径
```

### 3.2 查找顺序

1. **权重文件**：
   - 首先检查 `.cache` 目录
   - 如果不存在，检查 `third_party` 目录

2. **配置文件**：
   - 首先检查 `third_party/Grounded-SAM-2`
   - 如果不存在，检查 `third_party/Grounded-Segment-Anything`

3. **代码路径**：
   - 自动添加到 `sys.path`

---

## 四、使用示例

### 4.1 使用默认配置

```python
from src.temporal_reasoning import TemporalReasoningAnalyzer, get_default_config

# 使用默认配置（自动查找路径）
config = get_default_config()
analyzer = TemporalReasoningAnalyzer(config)
analyzer.initialize()
```

### 4.2 自定义路径

```python
from src.temporal_reasoning import TemporalReasoningAnalyzer, get_default_config

config = get_default_config()

# 自定义路径（如果需要）
config.raft.model_path = "custom/path/to/raft.pth"
config.grounding_dino.model_path = ".cache/groundingdino_swinb_cogcoor.pth"
config.sam.model_path = ".cache/sam2.1_hiera_large.pt"
config.tracker.cotracker_checkpoint = ".cache/scaled_offline.pth"

analyzer = TemporalReasoningAnalyzer(config)
analyzer.initialize()
```

### 4.3 从配置文件加载

```yaml
# config.yaml
temporal_reasoning:
  device: "cuda:0"
  
  raft:
    model_path: ".cache/raft-things.pth"
    model_type: "large"
    use_gpu: true
  
  grounding_dino:
    model_path: ".cache/groundingdino_swinb_cogcoor.pth"
    config_path: "third_party/Grounded-SAM-2/grounding_dino/config/GroundingDINO_SwinB.py"
    bert_path: ".cache/google-bert/bert-base-uncased"
    text_threshold: 0.25
    box_threshold: 0.3
  
  sam:
    model_path: ".cache/sam2.1_hiera_large.pt"
    config_path: "third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    model_type: "sam2_h"
    use_gpu: true
  
  tracker:
    enable_cotracker_validation: true
    cotracker_checkpoint: ".cache/scaled_offline.pth"
    grid_size: 30
```

```python
from src.temporal_reasoning import TemporalReasoningAnalyzer, load_config_from_yaml

config = load_config_from_yaml("config.yaml")
analyzer = TemporalReasoningAnalyzer(config)
analyzer.initialize()
```

---

## 五、路径验证

### 5.1 检查路径是否存在

```python
from pathlib import Path

# 检查权重文件
cache_dir = Path(".cache")
weights = {
    'raft': cache_dir / "raft-things.pth",
    'grounding_dino': cache_dir / "groundingdino_swinb_cogcoor.pth",
    'sam2': cache_dir / "sam2.1_hiera_large.pt",
    'cotracker': cache_dir / "scaled_offline.pth"
}

for name, path in weights.items():
    if path.exists():
        print(f"? {name}: {path}")
    else:
        print(f"? {name}: {path} (不存在)")
```

### 5.2 路径调试

如果模型加载失败，可以检查路径：

```python
config = get_default_config()

print("模型路径配置：")
print(f"RAFT: {config.raft.model_path}")
print(f"Grounding DINO: {config.grounding_dino.model_path}")
print(f"Grounding DINO Config: {config.grounding_dino.config_path}")
print(f"Grounding DINO BERT: {config.grounding_dino.bert_path}")
print(f"SAM: {config.sam.model_path}")
print(f"SAM Config: {config.sam.config_path}")
print(f"Co-Tracker: {config.tracker.cotracker_checkpoint}")
```

---

## 六、常见问题

### Q1: 模型文件找不到怎么办？

**A**: 检查以下几点：
1. 确认权重文件在 `.cache` 目录
2. 确认模型代码在 `third_party` 目录
3. 检查路径是否正确配置

### Q2: 如何下载缺失的模型文件？

**A**: 根据模型类型：
- **RAFT**: 从官方仓库下载权重文件
- **Grounding DINO**: 从HuggingFace下载或使用官方脚本
- **SAM2**: 从Meta官方下载
- **Co-Tracker**: 从官方仓库下载权重文件

### Q3: 可以使用相对路径吗？

**A**: 可以，配置支持相对路径和绝对路径。相对路径相对于项目根目录。

### Q4: 配置优先级是什么？

**A**: 
1. 用户显式设置的路径（最高优先级）
2. 配置文件中的路径
3. 自动查找的默认路径（最低优先级）

---

## 七、总结

### 7.1 路径配置原则

- ? **权重文件**：`.cache` 目录
- ? **模型代码**：`third_party` 目录
- ? **配置文件**：`third_party` 目录
- ? **自动查找**：优先 `.cache`，备选 `third_party`

### 7.2 配置方式

- ? **默认配置**：自动查找路径
- ? **自定义配置**：显式设置路径
- ? **配置文件**：从YAML加载配置

### 7.3 优势

- ? **路径统一**：权重和代码分离，便于管理
- ? **自动查找**：减少配置工作量
- ? **灵活配置**：支持自定义路径
- ? **向后兼容**：如果路径不存在，自动降级

---

**文档结束**

