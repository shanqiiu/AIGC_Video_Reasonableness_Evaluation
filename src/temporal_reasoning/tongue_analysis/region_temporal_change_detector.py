from __future__ import annotations

import warnings

from ..region_analysis.region_temporal_change_detector import (
    RegionTemporalChangeConfig,
    RegionTemporalChangeDetector,
)

warnings.warn(
    "'src.temporal_reasoning.tongue_analysis.region_temporal_change_detector' 已重命名为 'src.temporal_reasoning.region_analysis.region_temporal_change_detector'，"
    "请更新导入路径以消除该警告。",
    DeprecationWarning,
    stacklevel=2,
)

TongueFlowChangeConfig = RegionTemporalChangeConfig
TongueFlowChangeDetector = RegionTemporalChangeDetector

__all__ = [
    "RegionTemporalChangeConfig",
    "RegionTemporalChangeDetector",
    "TongueFlowChangeConfig",
    "TongueFlowChangeDetector",
]
