from .region_temporal_change_detector import RegionTemporalChangeConfig, RegionTemporalChangeDetector
from .pipeline import RegionAnalysisPipelineConfig, RegionAnalysisPipeline

# Backwards compatibility aliases
TongueFlowChangeConfig = RegionTemporalChangeConfig
TongueFlowChangeDetector = RegionTemporalChangeDetector
TongueAnalysisPipelineConfig = RegionAnalysisPipelineConfig
TongueAnalysisPipeline = RegionAnalysisPipeline

__all__ = [
    "RegionTemporalChangeConfig",
    "RegionTemporalChangeDetector",
    "RegionAnalysisPipelineConfig",
    "RegionAnalysisPipeline",
    "TongueFlowChangeConfig",
    "TongueFlowChangeDetector",
    "TongueAnalysisPipelineConfig",
    "TongueAnalysisPipeline",
]

