# MediaPipe警告信息分析

## 警告信息

```
W0000 00:00:1762434645.419877   12816 landmark_projection_calculator.cc:186] 
Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. 
Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
```

## 警告含义

### 1. 警告级别
- **级别**：W（Warning，警告）
- **来源**：`landmark_projection_calculator.cc:186`
- **影响**：不影响功能，但可能影响精度

### 2. 警告内容
- **问题**：使用 `NORM_RECT`（归一化矩形）但没有提供 `IMAGE_DIMENSIONS`（图像尺寸）
- **限制**：只支持方形ROI（Region of Interest，感兴趣区域）
- **建议**：提供 `IMAGE_DIMENSIONS` 或使用 `PROJECTION_MATRIX`

### 3. 原因分析

MediaPipe在计算关键点投影时，需要知道图像的尺寸信息。如果：
- 使用归一化坐标（NORM_RECT）但没有提供图像尺寸
- 图像不是方形（宽高比不是1:1）
- MediaPipe只能假设ROI是方形的，这可能导致投影计算不准确

## 当前代码问题

### 问题位置
在创建MediaPipe Image对象时，没有提供图像尺寸信息：

```python
mp_image = self.mp_image(
    image_format=self.image_format,
    data=image
)
```

### 问题分析
1. **缺少图像尺寸**：MediaPipe Image对象创建时没有指定图像的高度和宽度
2. **归一化坐标计算**：MediaPipe内部使用归一化坐标，需要图像尺寸来正确计算
3. **非方形图像**：如果输入图像不是方形（宽高比不是1:1），可能导致投影计算不准确

## 解决方案

### 方案1：在创建Image时提供尺寸信息（推荐）

修改Image创建代码，添加图像尺寸：

```python
# 获取图像尺寸
height, width = image.shape[:2]

# 创建MediaPipe Image对象，提供图像尺寸
mp_image = self.mp_image(
    image_format=self.image_format,
    data=image
)
# 注意：MediaPipe的Image类可能不支持直接传入尺寸参数
# 需要检查MediaPipe API文档
```

### 方案2：使用PROJECTION_MATRIX

如果MediaPipe支持，可以使用投影矩阵来指定图像变换：

```python
# 创建投影矩阵（需要根据MediaPipe API调整）
projection_matrix = create_projection_matrix(width, height)
```

### 方案3：预处理图像为方形（不推荐）

将输入图像预处理为方形，但这会改变图像比例，可能影响检测精度。

## 影响评估

### 1. 功能影响
- **不影响**：关键点检测功能正常
- **可能影响**：关键点坐标的精度，特别是对于非方形图像

### 2. 性能影响
- **无影响**：警告不会影响性能

### 3. 精度影响
- **轻微影响**：对于非方形图像，关键点坐标可能有轻微偏差
- **方形图像**：无影响（因为警告说明只支持方形ROI）

## 建议

### 短期方案（当前）
1. **保持现状**：警告不影响功能，可以暂时忽略
2. **监控精度**：如果发现关键点坐标不准确，再考虑修复

### 长期方案（优化）
1. **检查MediaPipe API**：查看最新版本的MediaPipe是否支持在Image创建时指定尺寸
2. **提供图像尺寸**：如果API支持，在创建Image时提供图像尺寸
3. **使用投影矩阵**：如果API支持，使用投影矩阵来精确控制坐标变换

## 相关代码位置

- `keypoint_extractor.py` 第275-277行：创建MediaPipe Image对象（Holistic）
- `keypoint_extractor.py` 第385-387行：创建MediaPipe Image对象（Pose）

## 参考

- MediaPipe官方文档：https://developers.google.com/mediapipe
- MediaPipe Python API：https://google.github.io/mediapipe/solutions/pose.html

