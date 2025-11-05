# 时序合理性分析执行脚本

## 使用说明

### 基本用法

```bash
# 基本使用
python run_analysis.py --video path/to/video.mp4

# 指定文本提示（用于检测特定部位）
python run_analysis.py --video path/to/video.mp4 --prompts "tongue" "finger"

# 使用配置文件
python run_analysis.py --video path/to/video.mp4 --config config.yaml

# 指定输出路径
python run_analysis.py --video path/to/video.mp4 --output results.json

# 指定计算设备
python run_analysis.py --video path/to/video.mp4 --device cuda:0
```

### 参数说明

- `--video`: **必需**，视频文件路径
- `--config`: 可选，配置文件路径（YAML格式）
- `--prompts`: 可选，文本提示列表，用于引导检测特定部位（如："tongue", "finger"）
- `--output`: 可选，输出结果文件路径（JSON格式），默认保存在 `outputs/temporal_reasoning/` 目录
- `--device`: 可选，计算设备（如："cuda:0", "cpu"），默认使用配置中的设备
- `--raft-model`: 可选，RAFT模型路径
- `--raft-type`: 可选，RAFT模型类型（"large" 或 "small"）
- `--output-dir`: 可选，输出目录
- `--save-visualizations`: 可选，保存可视化结果

### 配置文件格式

创建一个YAML配置文件（例如 `config.yaml`）：

```yaml
temporal_reasoning:
  device: "cuda:0"
  
  raft:
    model_path: "path/to/raft_model.pth"
    model_type: "large"
    use_gpu: true
    motion_discontinuity_threshold: 0.3
  
  grounding_dino:
    model_path: "path/to/grounding_dino"
    text_threshold: 0.25
    box_threshold: 0.3
    use_gpu: true
  
  sam:
    model_path: "path/to/sam2_model.pth"
    model_type: "sam2_h"
    use_gpu: true
  
  tracker:
    type: "deaot"
    use_gpu: true
  
  keypoint:
    model_type: "mediapipe"
    use_gpu: false
  
  fusion:
    multimodal_confidence_boost: 1.2
    min_anomaly_duration_frames: 3
    single_modality_confidence_threshold: 0.8
  
  thresholds:
    motion_discontinuity_threshold: 0.3
    structure_disappearance_threshold: 0.3
    keypoint_displacement_threshold: 10
  
  output_dir: "outputs/temporal_reasoning"
  save_visualizations: true
```

### 输出结果格式

输出结果为JSON格式，包含以下字段：

```json
{
  "motion_reasonableness_score": 0.85,
  "structure_stability_score": 0.92,
  "anomalies": [
    {
      "type": "motion_discontinuity",
      "timestamp": "3.2s",
      "frame_id": 96,
      "confidence": 0.85,
      "description": "第96帧检测到运动突变",
      "modalities": ["motion"],
      "severity": "Moderate",
      "location": {}
    }
  ],
  "sub_scores": {
    "motion_score": 0.88,
    "structure_score": 0.95,
    "physiological_score": 0.92
  },
  "anomaly_counts": {
    "motion": 2,
    "structure": 0,
    "physiological": 0,
    "fused": 1
  },
  "video_info": {
    "path": "path/to/video.mp4",
    "width": 1920,
    "height": 1080,
    "frame_count": 300,
    "fps": 30.0,
    "duration": 10.0
  }
}
```

### 示例

```bash
# 检测视频中的舌头异常
python run_analysis.py \
  --video data/videos/test.mp4 \
  --prompts "tongue" \
  --output results/tongue_analysis.json \
  --device cuda:0

# 使用配置文件
python run_analysis.py \
  --video data/videos/test.mp4 \
  --config configs/temporal_reasoning.yaml \
  --prompts "finger" "hand"
```

### 注意事项

1. **模型路径**：确保RAFT、Grounded-SAM-2等模型已正确下载并配置路径
2. **设备配置**：如果使用GPU，确保CUDA环境正确配置
3. **内存使用**：长视频可能需要较大内存，建议先测试小视频
4. **文本提示**：文本提示用于引导实例检测，不提供时会跳过实例追踪分析

### 故障排除

1. **模型加载失败**：检查模型路径是否正确，模型文件是否存在
2. **内存不足**：尝试减小视频分辨率或使用CPU模式
3. **导入错误**：确保所有依赖已安装，路径配置正确
