# SAM2 切换功能 - 手动修改指南

由于文件编码问题，需要手动修改以下两个方法。

## 修改 1: `initialize` 方法 (第 54-74 行)

**找到这段代码：**

```python
            try:
                # 鏍规嵁use_gpu閰嶇疆姝ｇ'璁剧疆璁惧瀛楃涓
                if self.gdino_config.use_gpu and torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    device = "cpu"
              
                self.grounded_sam_wrapper = GroundedSAMWrapper(
                    gdino_config_path=self.gdino_config.config_path,
                    gdino_checkpoint_path=self.gdino_config.model_path,
                    sam_checkpoint_path=self.sam_config.model_path,
                    device=device,
                    text_threshold=self.gdino_config.text_threshold,
                    box_threshold=self.gdino_config.box_threshold,
                    grid_size=self.tracker_config.grid_size
                )
                print("Grounded-SAM 鍒濆鍖栨垚鍔")
            except Exception as e:
                print(f"璀﹀憡: Grounded-SAM 鍒濆鍖栧け璐: {e}")
                print("灏嗕娇鐢ㄧ畝鍖栧疄鐜")
                self.grounded_sam_wrapper = None
```

**替换为：**

```python
            try:
                # 判断使用 SAM v1 还是 SAM2
                use_sam2 = False
                if self.sam_config.model_type and self.sam_config.model_type.startswith("sam2"):
                    use_sam2 = True
                elif self.sam_config.model_path:
                    from pathlib import Path
                    sam_path = Path(self.sam_config.model_path)
                    if sam_path.suffix == '.pt' or 'sam2' in sam_path.name.lower():
                        use_sam2 = True
              
                self.use_sam2 = use_sam2
              
                # 根据配置选择使用 SAM v1 或 SAM2
                if use_sam2:
                    # 使用 SAM2
                    if self.gdino_config.use_gpu and torch.cuda.is_available():
                        device = "cuda:0"
                    else:
                        device = "cpu"
                  
                    if not self.sam_config.config_path:
                        raise ValueError(
                            "SAM2 需要配置文件路径 (config_path)\n"
                            "请在 SAMConfig 中设置 config_path"
                        )
                  
                    self.grounded_sam2_wrapper = GroundedSAM2Wrapper(
                        gdino_config_path=self.gdino_config.config_path,
                        gdino_checkpoint_path=self.gdino_config.model_path,
                        sam2_config_path=self.sam_config.config_path,
                        sam2_checkpoint_path=self.sam_config.model_path,
                        device=device,
                        text_threshold=self.gdino_config.text_threshold,
                        box_threshold=self.gdino_config.box_threshold
                    )
                    print("Grounded-SAM2 初始化成功")
                else:
                    # 使用 SAM v1
                    if self.gdino_config.use_gpu and torch.cuda.is_available():
                        device = "cuda:0"
                    else:
                        device = "cpu"
                  
                    self.grounded_sam_wrapper = GroundedSAMWrapper(
                        gdino_config_path=self.gdino_config.config_path,
                        gdino_checkpoint_path=self.gdino_config.model_path,
                        sam_checkpoint_path=self.sam_config.model_path,
                        device=device,
                        text_threshold=self.gdino_config.text_threshold,
                        box_threshold=self.gdino_config.box_threshold,
                        grid_size=self.tracker_config.grid_size
                    )
                    print("Grounded-SAM (v1) 初始化成功")
            except Exception as e:
                print(f"警告: Grounded-SAM 初始化失败: {e}")
                print("将使用简化实现")
                self.grounded_sam_wrapper = None
```

## 修改 2: `detect_instances` 方法 (第 122-130 行)

**找到这段代码：**

```python
        if self.grounded_sam_wrapper is None:
        # 濡傛灉 Grounded-SAM 鏈鍒濆鍖栵紝杩斿洖绌哄垪琛
            return []
      
        try:
            return self.grounded_sam_wrapper.detect_and_segment(image, text_prompts)
        except Exception as e:
            print(f"璀﹀憡: 瀹炰緥妫�娴嬪け璐: {e}")
            return []
```

**替换为：**

```python
        # 根据配置选择使用 SAM v1 或 SAM2
        if self.use_sam2:
            if self.grounded_sam2_wrapper is None:
                # 如果 Grounded-SAM2 未初始化，返回空列表
                return []
          
            try:
                return self.grounded_sam2_wrapper.detect_and_segment(image, text_prompts)
            except Exception as e:
                print(f"警告: 实例检测失败 (SAM2): {e}")
                return []
        else:
            if self.grounded_sam_wrapper is None:
                # 如果 Grounded-SAM 未初始化，返回空列表
                return []
          
            try:
                return self.grounded_sam_wrapper.detect_and_segment(image, text_prompts)
            except Exception as e:
                print(f"警告: 实例检测失败 (SAM v1): {e}")
                return []
```

## 完成后的检查

修改完成后，请确认：

1. ? 已导入 `GroundedSAM2Wrapper` (第 17 行)
2. ? 已添加实例变量 `self.grounded_sam2_wrapper` 和 `self.use_sam2` (第 45-47 行)
3. ? `initialize` 方法已修改为支持 SAM2/SAM v1 切换
4. ? `detect_instances` 方法已修改为根据配置调用对应封装器

## 使用说明

### 使用 SAM2：

```python
sam_config.model_type = "sam2_h"  # 或 "sam2_l", "sam2_b"
sam_config.config_path = "third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam_config.model_path = ".cache/sam2.1_hiera_large.pt"
```

### 使用 SAM v1：

```python
sam_config.model_type = "sam_v1_h"  # 或任何不以 "sam2" 开头的值
sam_config.model_path = ".cache/sam_vit_h_4b8939.pth"
```

系统会根据 `model_type` 或文件路径自动判断使用哪个版本。
