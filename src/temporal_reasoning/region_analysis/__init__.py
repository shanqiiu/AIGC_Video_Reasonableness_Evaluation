from src.temporal_reasoning.region_analysis.region_temporal_change_detector import RegionTemporalChangeConfig, RegionTemporalChangeDetector
from src.temporal_reasoning.region_analysis.pipeline import RegionAnalysisPipelineConfig, RegionAnalysisPipeline

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

