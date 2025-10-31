# 快速开始

## 1. 环境准备

确保已安装必要的依赖：

```bash
pip install torch torchvision
pip install opencv-python pillow numpy
pip install tqdm
pip install transformers  # for BERT
```

## 2. 下载模型

在项目根目录创建 `.cache` 目录，并下载以下模型：

```bash
mkdir -p .cache

# 下载 GroundingDINO
wget -O .cache/groundingdino_swinb_cogcoor.pth \
  https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth

# 下载 SAM
wget -O .cache/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 下载 Co-Tracker
wget -O .cache/scaled_offline.pth \
  https://huggingface.co/facebook/cotracker/resolve/main/cotracker_scaled_offline.pth
```

## 3. 准备输入数据

创建元信息JSON文件 `meta_info.json`：

```json
[
  {
    "index": 0,
    "filepath": "data/videos/video1.mp4",
    "subject_noun": "person",
    "prompt": "A person walking in the park"
  },
  {
    "index": 1,
    "filepath": "data/videos/video2.mp4",
    "subject_noun": "dog",
    "prompt": "A dog running"
  }
]
```

## 4. 运行分析

### 方式A：使用脚本

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path meta_info.json \
    --output_path results/pas_results.json \
    --enable_scene_classification \
    --device cuda
```

### 方式B：Python代码

```python
import json
from src.aux_motion_intensity_2 import PASAnalyzer

# 初始化分析器
analyzer = PASAnalyzer(
    device="cuda",
    enable_scene_classification=True
)

# 加载元信息
with open("meta_info.json", "r") as f:
    meta_infos = json.load(f)

# 分析视频
for meta_info in meta_infos:
    result = analyzer.analyze_video(
        video_path=meta_info["filepath"],
        subject_noun=meta_info["subject_noun"]
    )
    
    # 将结果添加到元信息
    meta_info["perceptible_amplitude_score"] = result
    print(f"Video {meta_info['index']}: {result['status']}")
    
    if result["status"] == "success":
        print(f"  Background motion: {result['background_motion']:.4f}")
        print(f"  Subject motion: {result['subject_motion']:.4f}")
        print(f"  Scene type: {result.get('scene_classification', {}).get('scene_type', 'N/A')}")

# 保存结果
with open("results/pas_results.json", "w") as f:
    json.dump(meta_infos, f, indent=2)
```

## 5. 查看结果

结果将写入JSON文件，包含以下信息：

```json
{
  "status": "success",
  "background_motion": 0.0234,
  "subject_motion": 0.0567,
  "pure_subject_motion": 0.0333,
  "total_motion": 0.0801,
  "motion_ratio": 1.423,
  "video_resolution": {
    "width": 1920,
    "height": 1080,
    "diagonal": 2202.9
  },
  "scene_classification": {
    "scene_type": "low_dynamic_object",
    "scene_description": "低动态物体运动场景",
    "motion_dominant": "object_motion",
    "intensity_level": "low_dynamic",
    "confidence": 0.75
  }
}
```

## 常见问题

### Q1: 模型下载失败

**A**: 可以手动下载模型文件，或使用代理。确保文件下载完整（检查文件大小）。

### Q2: CUDA out of memory

**A**: 尝试：
- 降低 `grid_size` 参数（默认30）
- 使用更小的批次处理
- 或使用 CPU 模式（`--device cpu`，速度较慢）

### Q3: 无法检测到主体

**A**: 检查：
- `subject_noun` 是否与视频内容匹配
- 尝试调整 `--box_threshold` 和 `--text_threshold`
- 确保视频中有清晰可见的主体

### Q4: 导入错误

**A**: 确保：
- 正确安装所有依赖
- 项目路径已添加到 PYTHONPATH
- 第三方库已正确放置在 `third_party/` 目录

## 下一步

- 阅读 [README.md](README.md) 了解详细功能
- 查看 [INTEGRATION.md](INTEGRATION.md) 了解集成细节
- 参考示例调整参数以优化结果

