from __future__ import annotations

import warnings

from src.temporal_reasoning.region_analysis import run_region_temporal_analysis as _region_cli

warnings.warn(
    "'src.temporal_reasoning.tongue_analysis.run_tongue_analysis' 已重命名为 'src.temporal_reasoning.region_analysis.run_region_temporal_analysis'，"
    "请更新脚本路径以消除该警告。",
    DeprecationWarning,
    stacklevel=2,
)

parse_args = _region_cli.parse_args
build_pipeline_config = _region_cli.build_pipeline_config
load_temporal_config = _region_cli.load_temporal_config
run_analysis = _region_cli.run_analysis
main = _region_cli.main

def to_serializable(obj):
    return _region_cli.to_serializable(obj)


if __name__ == "__main__":
    main()
