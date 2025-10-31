# 可感知幅度评分脚本

## 使用方法

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json \
    --output_path results/pas_results.json \
    --enable_scene_classification \
    --device cuda
```

## 参数说明

### 必需参数

- `--meta_info_path`: 元信息JSON文件路径（必需）

### 可选参数

#### 输入输出
- `--output_path`: 输出JSON文件路径（默认：覆盖输入文件）

#### 模型参数
- `--device`: 设备类型 (cuda/cpu，默认: cuda)
- `--grid_size`: Co-Tracker网格大小（默认: 30）
- `--box_threshold`: GroundingDINO检测框阈值（默认: 0.3）
- `--text_threshold`: GroundingDINO文本阈值（默认: 0.25）

#### 场景分类
- `--enable_scene_classification`: 启用场景分类
- `--static_threshold`: 静态场景阈值（默认: 0.1）
- `--low_dynamic_threshold`: 低动态场景阈值（默认: 0.3）
- `--medium_dynamic_threshold`: 中等动态场景阈值（默认: 0.6）
- `--high_dynamic_threshold`: 高动态场景阈值（默认: 1.0）
- `--motion_ratio_threshold`: 运动比率阈值（默认: 1.5）

#### 其他选项
- `--no_subject_diag_norm`: 禁用主体对角线归一化

## 输入格式

元信息JSON文件应包含视频信息列表：

```json
[
  {
    "index": 0,
    "filepath": "data/video1.mp4",
    "subject_noun": "person",
    "prompt": "A person walking"
  },
  {
    "index": 1,
    "filepath": "data/video2.mp4",
    "subject_noun": "dog",
    "prompt": "A dog running"
  }
]
```

## 输出格式

脚本会在每个视频的元信息中添加 `perceptible_amplitude_score` 字段。

### 成功案例

```json
{
  "index": 0,
  "filepath": "data/video1.mp4",
  "subject_noun": "person",
  "prompt": "A person walking",
  "perceptible_amplitude_score": {
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
}
```

### 错误案例

```json
{
  "index": 1,
  "filepath": "data/video2.mp4",
  "subject_noun": "dog",
  "prompt": "A dog running",
  "perceptible_amplitude_score": {
    "status": "error",
    "error_reason": "no_subject_detected",
    "background_motion": 0.0234
  }
}
```

## 示例

### 基础使用

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json
```

### 启用场景分类

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json \
    --enable_scene_classification \
    --static_threshold 0.1 \
    --low_dynamic_threshold 0.3
```

### 使用CPU

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
    --meta_info_path data/meta_info.json \
    --device cpu
```

## 注意事项

1. 首次运行需要下载模型文件到 `.cache` 目录
2. 建议使用GPU加速（`--device cuda`）
3. 元信息文件中的 `subject_noun` 应与视频中实际物体匹配
4. 结果会自动写入元信息文件（默认覆盖原文件）

