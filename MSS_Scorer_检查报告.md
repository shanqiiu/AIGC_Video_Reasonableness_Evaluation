# MSS Scorer 检查报告

## 问题描述

从输出文件 `outputs/perceptual_quality/blur/blur_detection_results.json` 中发现：
- 所有视频的 `mss_score` 都是 `0.0`
- `blur_frames` 都是空数组 `[]`
- `quality_scores` 可能为空

这表明 MSS 评分器可能没有正常工作。

## 可能的原因

### 1. **quality_scores 为空**
如果 `mss_scorer.score()` 返回的 `quality_scores` 为空列表，会导致：
- `_get_artifacts_frames()` 无法正常工作
- `mss_score = 1 - len(blur_frames) / len(quality_scores)` 会除以零错误

### 2. **MSS Scorer 初始化失败**
- Q-Align 模型加载失败
- 设备配置问题
- 模型路径错误

### 3. **QAlignVideoScorer 返回值处理问题**
- 返回值格式不匹配（元组 vs 单个张量）
- 返回值处理逻辑错误

## 检查步骤

### 步骤 1: 运行诊断脚本

运行我创建的诊断脚本：
```bash
python scripts/perceptual_quality/test_mss_scorer.py
```

这个脚本会检查：
1. Q-Align 导入是否成功
2. MSS Scorer 初始化是否成功
3. `score()` 方法调用是否成功
4. `quality_scores` 是否为空

### 步骤 2: 检查输出文件

查看输出文件中的错误信息：
```bash
cat outputs/perceptual_quality/blur/blur_detection_results.json | grep -A 5 "error"
```

### 步骤 3: 添加调试输出

在 `blur_detection_pipeline.py` 的 `_detect_blur_with_mss` 方法中添加调试输出：

```python
def _detect_blur_with_mss(self, video_path: str) -> Dict:
    try:
        mss_output = self.mss_scorer.score(video_path)
        quality_scores = mss_output.get('quality_scores', [])
        
        # 添加调试输出
        print(f"DEBUG: mss_output = {mss_output}")
        print(f"DEBUG: quality_scores 长度 = {len(quality_scores)}")
        if len(quality_scores) > 0:
            print(f"DEBUG: quality_scores 前5个 = {quality_scores[:5]}")
        
        # 检查 quality_scores 是否为空
        if not quality_scores or len(quality_scores) == 0:
            print(f"警告: MSS 评分器返回空的 quality_scores")
            print(f"  视频路径: {video_path}")
            return {
                'mss_score': 0.0,
                'blur_frames': [],
                'quality_scores': [],
                'threshold': 0.025,
                'camera_movement': 0.0,
                'error': 'quality_scores is empty'
            }
        
        # ... 其余代码
```

## 需要修复的问题

### 1. 添加空值检查

在 `blur_detection_pipeline.py` 的 `_detect_blur_with_mss` 方法中，需要添加对空 `quality_scores` 的检查：

```python
# 在计算 mss_score 之前检查
if not quality_scores or len(quality_scores) == 0:
    print(f"警告: MSS 评分器返回空的 quality_scores")
    return {
        'mss_score': 0.0,
        'blur_frames': [],
        'quality_scores': [],
        'threshold': 0.025,
        'camera_movement': 0.0,
        'error': 'quality_scores is empty'
    }
```

### 2. 改进错误处理

在异常处理中添加更详细的错误信息：

```python
except Exception as e:
    print(f"MSS 检测失败: {e}")
    import traceback
    traceback.print_exc()  # 打印完整堆栈跟踪
    return {
        'mss_score': 0.0,
        'blur_frames': [],
        'quality_scores': [],
        'threshold': 0.025,
        'camera_movement': 0.0,
        'error': str(e)
    }
```

### 3. 检查 MSS Scorer 初始化

确保 MSS Scorer 正确初始化：
- 检查模型路径是否正确
- 检查设备配置是否正确
- 检查 Q-Align 模型是否正确加载

## 验证方法

### 方法 1: 运行单个视频测试

```bash
python scripts/perceptual_quality/run_blur_detection.py \
    --video_path data/videos/test.mp4 \
    --device cuda
```

观察控制台输出，查看是否有错误信息。

### 方法 2: 检查日志

查看是否有以下错误：
- "MSS 检测失败"
- "quality_scores is empty"
- GPU 内存不足错误
- 模型加载错误

### 方法 3: 检查输出文件

检查输出文件中的 `error` 字段：
```bash
cat outputs/perceptual_quality/blur/blur_detection_results.json | jq '.[] | select(.error)'
```

## 建议的修复方案

1. **立即修复**：添加空值检查和详细的错误日志
2. **长期优化**：改进 MSS Scorer 的错误处理机制
3. **监控**：添加运行时监控，检测 MSS Scorer 是否正常工作

## 相关文件

- `src/perceptual_quality/blur/mss_scorer.py` - MSS Scorer 实现
- `src/perceptual_quality/blur/blur_detection_pipeline.py` - 模糊检测流程
- `scripts/perceptual_quality/test_mss_scorer.py` - 诊断脚本

