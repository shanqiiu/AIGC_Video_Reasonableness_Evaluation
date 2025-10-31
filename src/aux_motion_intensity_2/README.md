# 可感知幅度评分 (Perceptible Amplitude Score)

## 概述

本模块使用 **Grounded-SAM** 和 **Co-Tracker** 计算视频中主体与背景的运动幅度，并输出可感知运动幅度分数。

## 功能特点

- **主体/背景解耦**：使用 Grounding DINO + SAM 进行语义分割
- **运动跟踪**：使用 Co-Tracker 进行点跟踪，计算运动幅度
- **场景分类**：自动判断场景类型（静态/动态相机运动、物体运动等）
- **可感知幅度**：输出背景运动、主体运动、纯主体运动等多种指标

## 依赖模型

请确保以下模型文件已下载到 `.cache` 目录：

```
.cache/
├── groundingdino_swinb_cogcoor.pth  # GroundingDINO模型
├── sam_vit_h_4b8939.pth             # SAM模型
├── scaled_offline.pth               # Co-Tracker模型
└── google-bert/                     # BERT模型
    └── bert-base-uncased/
```

### 模型下载说明

1. **GroundingDINO模型**: `groundingdino_swinb_cogcoor.pth`
   - 下载地址: https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
   - 保存到: `.cache/groundingdino_swinb_cogcoor.pth`

2. **SAM模型**: `sam_vit_h_4b8939.pth`
   - 下载地址: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   - 保存到: `.cache/sam_vit_h_4b8939.pth`

3. **Co-Tracker模型**: `scaled_offline.pth`
   - 下载地址: https://huggingface.co/facebook/cotracker/resolve/main/cotracker_scaled_offline.pth
   - 保存到: `.cache/scaled_offline.pth`

4. **BERT模型**: `google-bert/bert-base-uncased/`
   - 可以使用 Hugging Face 的 transformers 库自动下载
   - 或手动下载到: `.cache/google-bert/bert-base-uncased/`

## 使用方法

### 基础使用

```python
from src.aux_motion_intensity_2 import PASAnalyzer

# 初始化分析器
analyzer = PASAnalyzer(
    device="cuda",
    enable_scene_classification=True
)

# 分析单个视频
result = analyzer.analyze_video(
    video_path="path/to/video.mp4",
    subject_noun="person"  # 主体名词
)

print(result)
```

### 批量处理

```python
from src.aux_motion_intensity_2.batch import batch_analyze_videos

# 准备元信息列表
meta_infos = [
    {
        'filepath': 'video1.mp4',
        'subject_noun': 'person',
        'prompt': 'A person walking',
        'index': 0
    },
    # ... 更多视频
]

# 批量分析
results = batch_analyze_videos(
    analyzer=analyzer,
    meta_info_list=meta_infos,
    output_path='pas_results.json'
)
```

### 使用脚本

参见 `scripts/aux_motion_intensity_2/run_pas.py`

## 输出格式

### 成功案例

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

### 错误案例

```json
{
  "status": "error",
  "error_reason": "no_subject_detected",
  "background_motion": 0.0234
}
```

## 参数说明

### PASAnalyzer参数

- `device`: 设备类型 ("cuda" 或 "cpu")
- `grid_size`: Co-Tracker网格大小（默认30）
- `enable_scene_classification`: 是否启用场景分类
- `scene_classifier_params`: 场景分类器参数字典

### analyze_video参数

- `video_path`: 视频文件路径
- `subject_noun`: 主体名词（用于GroundingDINO检测）
- `box_threshold`: 检测框阈值（默认0.3）
- `text_threshold`: 文本阈值（默认0.25）
- `normalize_by_subject_diag`: 是否按主体对角线归一化

## 场景分类

### 场景类型

- `static_camera`: 静态相机运动场景
- `low_dynamic_camera`: 低动态相机运动场景
- `dynamic_camera`: 动态相机运动场景
- `static_object`: 静态物体场景
- `low_dynamic_object`: 低动态物体运动场景
- `medium_dynamic_object`: 中等动态物体运动场景
- `high_dynamic_object`: 高动态物体运动场景
- `extreme_dynamic_object`: 极高动态物体运动场景
- `mixed_scene`: 混合运动场景

### 运动主导类型

- `camera_motion`: 相机运动主导
- `object_motion`: 物体运动主导
- `mixed_motion`: 混合运动

## 注意事项

1. 首次运行需要下载模型文件
2. 建议使用GPU加速（device="cuda"）
3. 主体名词应与视频中实际物体匹配
4. 运动幅度已归一化，可跨分辨率比较

## 第三方依赖

- Grounded-Segment-Anything (third_party/)
- Co-Tracker (third_party/)
- GroundingDINO (third_party/Grounded-Segment-Anything/GroundingDINO/)
- Segment-Anything (third_party/Grounded-Segment-Anything/segment_anything/)

