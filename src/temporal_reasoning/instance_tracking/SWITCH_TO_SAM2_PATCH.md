# SAM2 切换功能实现补丁

由于文件编码问题，需要手动应用以下修改：

## 1. 修改 `instance_analyzer.py` 的 `initialize` 方法

将第 49-74 行的代码替换为：

```python
    def initialize(self):
        """初始化模型"""
        print("正在初始化实例追踪分析器...")
        try:
            # 判断使用 SAM v1 还是 SAM2
            # 根据 model_type 或文件路径判断
            use_sam2 = False
            if self.sam_config.model_type and self.sam_config.model_type.startswith("sam2"):
                use_sam2 = True
            elif self.sam_config.model_path:
                # 检查文件路径或文件名
                from pathlib import Path
                sam_path = Path(self.sam_config.model_path)
                if sam_path.suffix == '.pt' or 'sam2' in sam_path.name.lower():
                    use_sam2 = True
            
            self.use_sam2 = use_sam2
            
            # 根据配置选择使用 SAM v1 或 SAM2
            if use_sam2:
                # 使用 SAM2
                try:
                    # 根据use_gpu配置正确设置设备字符串
                    if self.gdino_config.use_gpu and torch.cuda.is_available():
                        device = "cuda:0"
                    else:
                        device = "cpu"
                    
                    # 检查 SAM2 配置路径
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
                except Exception as e:
                    print(f"警告: Grounded-SAM2 初始化失败: {e}")
                    print("尝试回退到 SAM v1...")
                    use_sam2 = False
                    self.use_sam2 = False
            
            if not use_sam2:
                # 使用 SAM v1
                try:
                    # 根据use_gpu配置正确设置设备字符串
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
                    print(f"警告: Grounded-SAM (v1) 初始化失败: {e}")
                    print("将使用简化实现")
                    self.grounded_sam_wrapper = None
            
            # 初始化Co-Tracker验证器（如果启用）
            if self.tracker_config.enable_cotracker_validation:
                # ... 后续代码保持不变
```

## 2. 修改 `instance_analyzer.py` 的 `detect_instances` 方法

将第 107-130 行的代码替换为：

```python
    def detect_instances(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        检测和分割实例
        
        Args:
            image: 输入图像 (H, W, 3) RGB
            text_prompts: 文本提示列表
        
        Returns:
            掩码列表，每个元素为(mask, confidence)元组
        """
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

## 使用说明

1. 要使用 SAM2，需要在配置中设置：
   ```python
   sam_config.model_type = "sam2_h"  # 或 "sam2_l", "sam2_b"
   sam_config.config_path = "third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
   sam_config.model_path = ".cache/sam2.1_hiera_large.pt"
   ```

2. 要使用 SAM v1，设置：
   ```python
   sam_config.model_type = "sam_v1_h"  # 或任何不以 "sam2" 开头的值
   sam_config.model_path = ".cache/sam_vit_h_4b8939.pth"
   ```

3. 系统会自动根据 `model_type` 或文件路径判断使用哪个版本。

