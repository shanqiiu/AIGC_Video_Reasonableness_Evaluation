# -*- coding: utf-8 -*-
"""诊断脚本：检查 MSS Scorer 是否正常工作"""

import os
import sys
import traceback

# 确保项目根目录在路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.perceptual_quality.blur.mss_scorer import MSSScorer
from src.perceptual_quality.blur.config import BlurDetectionConfig


def test_mss_scorer():
    """测试 MSS Scorer 是否正常工作"""
    print("=" * 60)
    print("MSS Scorer 诊断测试")
    print("=" * 60)
    
    # 创建配置
    config = BlurDetectionConfig()
    print(f"\n1. 配置信息:")
    print(f"   - 设备: {config.get_device_config('device')}")
    print(f"   - Q-Align 模型路径: {config.model_paths.get('q_align_model', '未设置')}")
    
    # 创建 MSS Scorer
    print(f"\n2. 初始化 MSS Scorer...")
    try:
        mss_scorer = MSSScorer(
            device=config.get_device_config("device") or "cuda",
            model_paths=config.model_paths,
            window_size=3
        )
        print("   ? MSS Scorer 初始化成功")
    except Exception as e:
        print(f"   ? MSS Scorer 初始化失败: {e}")
        traceback.print_exc()
        return False
    
    # 测试视频路径
    test_video = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "data", "videos", 
        "The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    )
    test_video = os.path.abspath(test_video)
    
    if not os.path.exists(test_video):
        print(f"\n3. 测试视频不存在: {test_video}")
        print("   请提供有效的视频路径进行测试")
        return False
    
    print(f"\n3. 测试视频: {test_video}")
    print(f"   文件存在: {os.path.exists(test_video)}")
    
    # 调用 score 方法
    print(f"\n4. 调用 MSS Scorer.score()...")
    try:
        result = mss_scorer.score(test_video)
        print(f"   ? Score 调用成功")
        
        # 检查结果
        quality_scores = result.get('quality_scores', [])
        print(f"\n5. 结果检查:")
        print(f"   - quality_scores 类型: {type(quality_scores)}")
        print(f"   - quality_scores 长度: {len(quality_scores)}")
        
        if len(quality_scores) == 0:
            print(f"   ? 警告: quality_scores 为空！")
            print(f"   这会导致 MSS 评分计算失败")
            return False
        
        print(f"   - 前5个分数: {quality_scores[:5] if len(quality_scores) >= 5 else quality_scores}")
        print(f"   - 分数范围: [{min(quality_scores):.4f}, {max(quality_scores):.4f}]")
        print(f"   - 平均分数: {sum(quality_scores) / len(quality_scores):.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ? Score 调用失败: {e}")
        traceback.print_exc()
        return False


def test_q_align_import():
    """测试 Q-Align 导入"""
    print(f"\n测试 Q-Align 导入...")
    try:
        from q_align import QAlignVideoScorer
        print("   ? 从 q_align 导入成功")
        return True
    except ImportError:
        try:
            from q_align.evaluate.scorer import QAlignVideoScorer
            print("   ? 从 q_align.evaluate.scorer 导入成功")
            return True
        except ImportError as e:
            print(f"   ? Q-Align 导入失败: {e}")
            return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MSS Scorer 诊断工具")
    print("=" * 60)
    
    # 测试 Q-Align 导入
    if not test_q_align_import():
        print("\n? Q-Align 导入失败，请检查安装")
        sys.exit(1)
    
    # 测试 MSS Scorer
    if test_mss_scorer():
        print("\n" + "=" * 60)
        print("? MSS Scorer 工作正常")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("? MSS Scorer 存在问题，请检查上述错误信息")
        print("=" * 60)
        sys.exit(1)

