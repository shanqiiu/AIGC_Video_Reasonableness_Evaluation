# BERT 模型下载问题修复说明

## 问题描述

即使所有模型都已经在本地下载好了，系统仍然会尝试从 Hugging Face Hub 下载 BERT 模型，导致出现联网下载的警告。

## 问题原因

### 根本原因

在 `grounded_sam2_wrapper.py` 中，调用 `load_model()` 函数加载 Grounding DINO 时：

1. **配置文件中的 `text_encoder_type`**：`GroundingDINO_SwinB_cfg.py` 中硬编码为 `text_encoder_type = "bert-base-uncased"`（字符串）
2. **BERT 加载逻辑**：`get_tokenlizer.py` 中的 `get_tokenlizer()` 和 `get_pretrained_language_model()` 函数使用 `AutoTokenizer.from_pretrained()` 和 `BertModel.from_pretrained()`
3. **transformers 库行为**：当传入字符串 `"bert-base-uncased"` 时，`from_pretrained()` 会尝试从 Hugging Face Hub 下载模型，即使本地可能有缓存

### 代码执行流程

```
grounded_sam2_wrapper.py:144
  ↓ load_model()
inference.py:29-36
  ↓ SLConfig.fromfile(gdino_config_path)
  ↓ args.text_encoder_type = "bert-base-uncased"  ← 硬编码的字符串
  ↓ build_model(args)
groundingdino.py:406
  ↓ text_encoder_type=args.text_encoder_type
get_tokenlizer.py:19
  ↓ AutoTokenizer.from_pretrained("bert-base-uncased")  ← 尝试从 Hugging Face Hub 下载
get_tokenlizer.py:25
  ↓ BertModel.from_pretrained("bert-base-uncased")  ← 尝试从 Hugging Face Hub 下载
```

## 解决方案

### 修复内容

在 `grounded_sam2_wrapper.py` 中：

1. **添加 `bert_path` 参数**：允许传入本地 BERT 模型路径
2. **修改配置加载逻辑**：在加载配置后，将 `text_encoder_type` 设置为本地路径
3. **设置离线模式**：临时设置环境变量 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，强制使用本地模型

### 修复后的代码执行流程

```
grounded_sam2_wrapper.py:144
  ↓ SLConfig.fromfile(gdino_config_path)
  ↓ args.text_encoder_type = bert_path (本地路径)  ← 设置为本地路径
  ↓ 设置 HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
  ↓ build_model(args)
groundingdino.py:406
  ↓ text_encoder_type=本地路径
get_tokenlizer.py:19
  ↓ AutoTokenizer.from_pretrained(本地路径)  ← 使用本地模型
get_tokenlizer.py:25
  ↓ BertModel.from_pretrained(本地路径)  ← 使用本地模型
  ↓ 恢复环境变量
```

## 修复详情

### 1. 修改 `GroundedSAM2Wrapper.__init__`

**位置：** `grounded_sam2_wrapper.py` 第 83-92 行

添加了 `bert_path` 参数：
```python
def __init__(
    self,
    gdino_config_path: str,
    gdino_checkpoint_path: str,
    sam2_config_path: str,
    sam2_checkpoint_path: str,
    device: str = "cuda:0",
    text_threshold: float = 0.25,
    box_threshold: float = 0.3,
    bert_path: Optional[str] = None  # ← 新增参数
):
```

### 2. 修改 `_initialize_models` 方法

**位置：** `grounded_sam2_wrapper.py` 第 144-205 行

添加了本地 BERT 路径设置逻辑：
```python
# 加载配置文件
args = SLConfig.fromfile(gdino_config_path)
args.device = self.device

# 如果提供了本地 BERT 路径，设置它以避免从 Hugging Face Hub 下载
if self.bert_path:
    bert_path_obj = Path(self.bert_path)
    if bert_path_obj.exists() and bert_path_obj.is_dir():
        # 将 text_encoder_type 设置为本地路径
        args.text_encoder_type = str(bert_path_obj.absolute())
        print(f"使用本地 BERT 模型: {self.bert_path}")
        
        # 设置离线模式环境变量
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        try:
            # 构建模型（会使用本地 BERT）
            self.grounding_dino_model = build_model(args)
        finally:
            # 恢复环境变量
            ...
```

### 3. 修改 `instance_analyzer.py`

**位置：** `instance_analyzer.py` 第 80-88 行

在创建 `GroundedSAM2Wrapper` 时传递 `bert_path`：
```python
self.grounded_sam2_wrapper = GroundedSAM2Wrapper(
    ...
    bert_path=self.gdino_config.bert_path  # ← 传递本地 BERT 路径
)
```

## 配置传递

### BERT 路径的配置

**位置：** `src/temporal_reasoning/core/config.py` 第 141-150 行

系统会自动检测并设置 BERT 路径：
```python
# BERT模型路径
if not self.grounding_dino.bert_path:
    bert_path = cache_dir / "google-bert" / "bert-base-uncased"
    if bert_path.exists():
        self.grounding_dino.bert_path = str(bert_path)
```

### 路径传递流程

```
config.py: __post_init__()
  ↓ self.grounding_dino.bert_path = ".cache/google-bert/bert-base-uncased"
  ↓
instance_analyzer.py: initialize()
  ↓ self.gdino_config.bert_path
  ↓
GroundedSAM2Wrapper.__init__(bert_path=...)
  ↓ self.bert_path
  ↓
_initialize_models()
  ↓ args.text_encoder_type = bert_path (本地路径)
  ↓ build_model(args)
```

## 验证修复

### 检查 BERT 路径

确保 BERT 模型在以下位置：
```
.cache/google-bert/bert-base-uncased/
  ├── config.json
  ├── pytorch_model.bin (或 model.safetensors)
  ├── tokenizer_config.json
  ├── vocab.txt
  └── ...
```

### 测试修复

运行代码后，应该看到：
```
正在初始化 Grounding DINO...
使用本地 BERT 模型: .cache/google-bert/bert-base-uncased
Grounding DINO 初始化成功
```

**而不是：**
```
正在初始化 Grounding DINO...
警告: 尝试从 Hugging Face Hub 下载 BERT 模型...
```

## 注意事项

1. **BERT 模型目录结构**：确保本地 BERT 模型目录包含所有必需文件（config.json, tokenizer_config.json, vocab.txt 等）
2. **环境变量影响**：设置 `HF_HUB_OFFLINE=1` 会影响当前进程中的所有 transformers 库调用
3. **路径格式**：使用绝对路径 `bert_path_obj.absolute()` 确保路径解析正确

## 相关文件

- **修复文件：** `src/temporal_reasoning/instance_tracking/grounded_sam2_wrapper.py`
- **配置传递：** `src/temporal_reasoning/instance_tracking/instance_analyzer.py`
- **配置定义：** `src/temporal_reasoning/core/config.py`
- **BERT 加载逻辑：** `third_party/Grounded-SAM-2/grounding_dino/groundingdino/util/get_tokenlizer.py`

