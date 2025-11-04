# -*- coding: utf-8 -*-
"""
视频模糊检测配置文件
"""

import os
from pathlib import Path

class BlurDetectionConfig:
    """模糊检测配置类"""
    
    def __init__(self):
        """初始化配置"""
        # 基础路径配置
        self.base_dir = Path(__file__).parent.parent
        self.cache_dir = self.base_dir / ".cache"
        self.output_dir = self.base_dir / "视频模糊检测" / "results"
        
        # 模型路径配置
        self.model_paths = {
            'q_align_model': str(self.cache_dir / "q-future" / "one-align"),
            'grounding_dino_config': str(self.base_dir / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB.py"),
            'grounding_dino_checkpoint': str(self.cache_dir / "groundingdino_swinb_cogcoor.pth"),
            'bert_path': str(self.cache_dir / "google-bert" / "bert-base-uncased"),
            'sam_checkpoint': str(self.cache_dir / "sam_vit_h_4b8939.pth"),
            'cotracker_checkpoint': str(self.cache_dir / "scaled_offline.pth")
        }
        
        # 检测参数配置
        self.detection_params = {
            'window_size': 3,  # 滑动窗口大小
            'blur_thresholds': {
                'mild_blur': 0.015,    # 轻微模糊阈值
                'moderate_blur': 0.025, # 中等模糊阈值
                'severe_blur': 0.04    # 严重模糊阈值
            },
            'confidence_threshold': 0.7,  # 综合置信度阈值
            'min_frames': 10,  # 最小帧数要求
            'max_frames': 1000  # 最大帧数限制
        }
        
        # 可视化配置
        self.visualization_params = {
            'figure_size': (15, 10),
            'dpi': 300,
            'font_size': 12,
            'color_palette': 'husl',
            'style': 'whitegrid'
        }
        
        # 输出配置
        self.output_params = {
            'save_visualizations': True,
            'save_detailed_reports': True,
            'save_csv_summary': True,
            'save_json_results': True
        }
        
        # 设备配置
        self.device_config = {
            'device': 'cuda:0',
            'batch_size': 1,
            'num_workers': 4
        }
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型路径"""
        return self.model_paths.get(model_name, "")
    
    def get_detection_param(self, param_name: str):
        """获取检测参数"""
        return self.detection_params.get(param_name)
    
    def get_visualization_param(self, param_name: str):
        """获取可视化参数"""
        return self.visualization_params.get(param_name)
    
    def get_output_param(self, param_name: str):
        """获取输出参数"""
        return self.output_params.get(param_name)
    
    def get_device_config(self, config_name: str):
        """获取设备配置"""
        return self.device_config.get(config_name)
    
    def update_model_path(self, model_name: str, new_path: str):
        """更新模型路径"""
        self.model_paths[model_name] = new_path
    
    def update_detection_param(self, param_name: str, new_value):
        """更新检测参数"""
        self.detection_params[param_name] = new_value
    
    def update_visualization_param(self, param_name: str, new_value):
        """更新可视化参数"""
        self.visualization_params[param_name] = new_value
    
    def update_output_param(self, param_name: str, new_value):
        """更新输出参数"""
        self.output_params[param_name] = new_value
    
    def update_device_config(self, config_name: str, new_value):
        """更新设备配置"""
        self.device_config[config_name] = new_value
    
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        # 检查必要的模型文件是否存在
        required_models = ['q_align_model', 'grounding_dino_checkpoint', 'sam_checkpoint', 'cotracker_checkpoint']
        
        for model in required_models:
            model_path = self.get_model_path(model)
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在 - {model}: {model_path}")
                return False
        
        # 检查输出目录是否可写
        if not os.access(self.output_dir, os.W_OK):
            print(f"错误: 输出目录不可写 - {self.output_dir}")
            return False
        
        return True
    
    def print_config(self):
        """打印当前配置"""
        print("=== 视频模糊检测配置 ===")
        print(f"基础目录: {self.base_dir}")
        print(f"缓存目录: {self.cache_dir}")
        print(f"输出目录: {self.output_dir}")
        print("\n模型路径:")
        for name, path in self.model_paths.items():
            exists = "?" if os.path.exists(path) else "?"
            print(f"  {name}: {path} {exists}")
        print("\n检测参数:")
        for name, value in self.detection_params.items():
            print(f"  {name}: {value}")
        print("\n设备配置:")
        for name, value in self.device_config.items():
            print(f"  {name}: {value}")


# 默认配置实例
default_config = BlurDetectionConfig()

# 预定义配置
PRESET_CONFIGS = {
    'fast': {
        'window_size': 2,
        'confidence_threshold': 0.6,
        'min_frames': 5
    },
    'accurate': {
        'window_size': 5,
        'confidence_threshold': 0.8,
        'min_frames': 20
    },
    'balanced': {
        'window_size': 3,
        'confidence_threshold': 0.7,
        'min_frames': 10
    }
}

def get_preset_config(preset_name: str) -> BlurDetectionConfig:
    """获取预定义配置"""
    config = BlurDetectionConfig()
    
    if preset_name in PRESET_CONFIGS:
        preset_params = PRESET_CONFIGS[preset_name]
        for param, value in preset_params.items():
            config.update_detection_param(param, value)
    
    return config

def create_custom_config(**kwargs) -> BlurDetectionConfig:
    """创建自定义配置"""
    config = BlurDetectionConfig()
    
    # 更新检测参数
    if 'detection_params' in kwargs:
        for param, value in kwargs['detection_params'].items():
            config.update_detection_param(param, value)
    
    # 更新模型路径
    if 'model_paths' in kwargs:
        for model, path in kwargs['model_paths'].items():
            config.update_model_path(model, path)
    
    # 更新设备配置
    if 'device_config' in kwargs:
        for device_param, value in kwargs['device_config'].items():
            config.update_device_config(device_param, value)
    
    return config
