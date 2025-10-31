"""
辅助模块2：可感知幅度评分（Perceptible Amplitude Score）
使用Grounded-SAM和Co-Tracker计算主体/背景运动幅度，输出可感知运动幅度分数。
"""

from .analyzer import PASAnalyzer

__all__ = ['PASAnalyzer']

