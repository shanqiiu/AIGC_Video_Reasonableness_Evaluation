from __future__ import annotations

import warnings

from ..region_analysis.pipeline import (
    RegionAnalysisPipeline,
    RegionAnalysisPipelineConfig,
    RegionDefinition,
    RegionMaskMode,
    default_regions,
)

warnings.warn(
    "'src.temporal_reasoning.tongue_analysis.pipeline' 已重命名为 'src.temporal_reasoning.region_analysis.pipeline'，"
    "请更新导入路径以消除该警告。",
    DeprecationWarning,
    stacklevel=2,
)

TongueAnalysisPipeline = RegionAnalysisPipeline
TongueAnalysisPipelineConfig = RegionAnalysisPipelineConfig

__all__ = [
    "RegionAnalysisPipeline",
    "RegionAnalysisPipelineConfig",
    "RegionDefinition",
    "RegionMaskMode",
    "default_regions",
    "TongueAnalysisPipeline",
    "TongueAnalysisPipelineConfig",
]


