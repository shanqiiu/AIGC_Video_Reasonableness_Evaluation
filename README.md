# AIGC Video Reasonableness Evaluation

一套针对 **AIGC（生成式视频）合理性校验** 的多模态分析框架，涵盖结构时序一致性、运动强度、场景真实度、感知质量等多个维度，并提供统一的脚本、配置与可视化能力，便于快速构建自动化评测流水线。

---

## 功能概览

- 🧠 **Temporal Reasoning**：基于光流、实例跟踪、关键点等信息评估视频在时间尺度上的结构合理性，支持可视化导出。
- 📈 **Aux Motion Intensity**：衡量主体/背景运动强度与场景类别，支持 RAFT / CoTracker / SAM2 等多种组合。
- 👁️ **Perceptual Quality**：借助 Q-Align 视频质量模型，对模糊等感知缺陷进行检测与报告生成。
- 🎬 **Scene Realism & VLM Reasoning**：面向场景真实性、跨模态一致性等维度的扩展能力（脚本/模块化接口预留）。
- 🛠️ **统一工具链**：脚本化的数据准备、批量执行、可视化与结果汇总；模块化配置，易于集成到生产/研究流程。

---

## 目录结构

```
AIGC_Video_Reasonableness_Evaluation
├─ data/                   # 示例数据与测试视频（需按需放置）
├─ outputs/                # 任务结果输出目录（JSON、CSV、可视化图表等）
├─ scripts/                # 命令行脚本入口，按功能划分子目录
│  ├─ temporal_reasoning/  # 时序合理性分析脚本
│  ├─ aux_motion_intensity/    # 旧版运动强度分析流程
│  ├─ aux_motion_intensity_2/  # 基于 Grounded-SAM + CoTracker 的 PAS 流程
│  ├─ perceptual_quality/  # 感知质量（模糊检测）脚本
│  └─ ...                  # 其他场景/融合脚本
├─ src/                    # 核心源码（按模块拆分）
│  ├─ temporal_reasoning/
│  ├─ aux_motion_intensity/
│  ├─ aux_motion_intensity_2/
│  ├─ perceptual_quality/
│  ├─ scene_realism/
│  ├─ fusion_engine/
│  ├─ video_io/
│  └─ vlm_reasoning/
├─ third_party/            # 外部依赖仓库（Grounded-SAM-2、CoTracker、Q-Align、RAFT 等）
└─ README.md               # 项目说明（本文档）
```

各子模块在 `src/<module>/README.md` 中提供了更细化的说明与设计思路。

---

## 环境准备

建议使用 **Python 3.10+**，并预先准备 GPU（CUDA）环境以获得最佳性能。

1. 创建虚拟环境并安装依赖：

   ```bash
   conda create -n aigc_eval python=3.10
   conda activate aigc_eval
   pip install -r third_party/requirements.txt  # 包含 Grounded-SAM-2 依赖
   ```
2. 编译/安装第三方项目（若使用）：

   - **Co-Tracker**：参考 `third_party/co-tracker/README.md` 安装依赖与权重。
   - **Grounded-SAM-2 / Segment Anything / Grounding DINO**：按其官方说明准备模型文件。
   - **Q-Align**：需要下载 `q-future/one-align` 权重至 `.cache/q-future/one-align`。
   - **RAFT**：将 `raft-things.pth` 等权重放置在 `.cache/`，并根据需要编译 CUDA 扩展。
3. 将模型权重放置到项目根目录 `.cache/` 下，常见示例：

   ```
   .cache/
   ├─ grounddingdino_swinb_cogcoor.pth
   ├─ sam_vit_h_4b8939.pth
   ├─ scaled_offline.pth                # CoTracker
   ├─ raft-things.pth                   # RAFT
   ├─ q-future/one-align/               # Q-Align
   └─ google-bert/bert-base-uncased/    # Grounding DINO 所需
   ```

---

## 快速上手

### 1. 时序合理性分析

```bash
python scripts/temporal_reasoning/run_analysis.py \
  --video path/to/video.mp4 \
  --enable_structure_visualization \
  --prompts "person" "car"
```

- 结果将保存至 `outputs/temporal_reasoning/<video>_result.json`。
- 若指定 `--enable_cotracker_visualization`，会额外生成 CoTracker 轨迹视频。
- CLI 支持传入 YAML 配置，或通过命令行覆盖关键参数（设备、模型路径、可视化参数等）。

### 2. 可感知幅度（运动强度）分析

```bash
python scripts/aux_motion_intensity_2/run_pas.py \
  --meta_info_path data/meta_info.json \
  --output_path outputs/pas_results.json \
  --enable_scene_classification
```

- `meta_info.json` 应包含 `filepath`、`subject_noun` 等字段。
- 输出会在原元数据中追加 `perceptible_amplitude_score` 字段，并生成汇总日志。

### 3. 模糊检测（感知质量）

```bash
python scripts/perceptual_quality/run_blur_detection.py \
  --video path/to/video.mp4 \
  --output-dir outputs/perceptual_quality/blur \
  --enable-visual-report
```

- 会调用 Q-Align 质量模型计算滑动窗口分数，输出 JSON/CSV/可视化图表。
- `outputs/perceptual_quality/blur` 下包含详细报告与统计。

---

## 核心模块简介

| 模块                       | 描述                                                                           | 主要依赖                                               |
| -------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------ |
| `temporal_reasoning`     | 光流、实例追踪、关键点分析三路融合，输出运动合理性、结构稳定性、异常列表等指标 | RAFT、Grounded-DINO、SAM2、CoTracker、MediaPipe/MMPose |
| `aux_motion_intensity`   | 旧版运动强度分析流水线，仍可用于轻量评估                                       | OpenCV (Farneback/TV-L1)、RAFT                         |
| `aux_motion_intensity_2` | 基于 Grounded-SAM + CoTracker 的 PAS 分析，支持场景分类/可视化                 | Grounded-SAM-2、CoTracker、Torch                       |
| `perceptual_quality`     | 使用 Q-Align 质量模型检测模糊、生成报告                                        | Q-Align、Decord、Matplotlib                            |
| `scene_realism`          | 场景真实度相关接口预留模块                                                     | （按需扩展）                                           |
| `vlm_reasoning`          | 大语言/视觉模型联合推理接口                                                    | （按需扩展）                                           |
| `fusion_engine`          | 多模态结果融合与决策逻辑                                                       | 自研                                                   |
| `video_io`               | 通用视频读取、帧抽样、缓存工具                                                 | OpenCV、Decord                                         |

每个子目录均提供 README / QUICKSTART / INTEGRATION 文档说明用法与设计。

---

## 数据与输出

- **输入数据**：项目未直接提供数据，可自行将测试视频置于 `data/` 或任意路径后通过 CLI 指定。
- **输出内容**：默认写入 `outputs/` 下，包含 JSON 结果、图表、统计报表等；路径可通过 CLI/配置覆盖。
- **可视化**：结构可视化（SAM2 分割覆盖）、CoTracker 轨迹视频、模糊检测曲线/报告等均可启用。

---

## 自定义与扩展

- **配置管理**：各模块在 `src/<module>/config.py` 或相关 README 中给出可覆盖字段；可通过 YAML / CLI 参数覆写。
- **模型替换**：如需替换成其他检测或分割模型，可在对应模块里扩展 `*_Analyzer`、`DetectionEngine` 实现。
- **融合策略**：`src/fusion_engine` 中的 `FusionDecisionEngine` 支持自定义多模态融合逻辑与阈值。
- **脚本扩展**：`scripts/` 目录按功能划分，推荐在对应子目录新增脚本，保持 CLI 参数风格一致。

---

## 常见问题

1. **模型权重无法找到 / 加载失败**

   - 确认 `.cache/` 目录下存在相应文件，路径区分大小写；可通过 CLI 直接指定模型路径进行覆盖。
2. **CUDA 内存不足**

   - 降低批大小（`--batch-size`）、减少滑动窗口长度，或改用 CPU 模式（性能会下降）。
3. **第三方依赖安装困难**

   - 建议参考各第三方仓库 README；如需快速体验，可先关闭相关功能（例如禁用 CoTracker 验证）。
4. **生成的可视化缺失**

   - 确认命令行参数已开启对应选项，并检查输出目录是否可写（Windows 下注意权限与路径长度）。

---

## 致谢与版权

项目中包含多个开源子模块，版权归原作者所有。使用前请仔细阅读 `third_party` 目录下各项目的 License 与使用规范：

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Co-Tracker](https://github.com/facebookresearch/co-tracker)
- [Q-Align](https://github.com/VQAssessment/Q-Align)
- [RAFT](https://github.com/princeton-vl/RAFT)
- 以及其他随附库与模型。
