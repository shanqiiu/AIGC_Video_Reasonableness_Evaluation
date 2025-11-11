from __future__ import annotations

import warnings

from src.temporal_reasoning.region_analysis import (
    RegionAnalysisPipeline,
    RegionAnalysisPipelineConfig,
    RegionTemporalChangeConfig,
    RegionTemporalChangeDetector,
)

warnings.warn(
    "'src.temporal_reasoning.tongue_analysis' 已重命名为 'src.temporal_reasoning.region_analysis'，"
    "请更新导入路径以消除该警告。",
    DeprecationWarning,
    stacklevel=2,
)

TongueAnalysisPipeline = RegionAnalysisPipeline
TongueAnalysisPipelineConfig = RegionAnalysisPipelineConfig
TongueFlowChangeConfig = RegionTemporalChangeConfig
TongueFlowChangeDetector = RegionTemporalChangeDetector

__all__ = [
    "TongueAnalysisPipeline",
    "TongueAnalysisPipelineConfig",
    "TongueFlowChangeConfig",
    "TongueFlowChangeDetector",
    "RegionAnalysisPipeline",
    "RegionAnalysisPipelineConfig",
    "RegionTemporalChangeConfig",
    "RegionTemporalChangeDetector",
]
