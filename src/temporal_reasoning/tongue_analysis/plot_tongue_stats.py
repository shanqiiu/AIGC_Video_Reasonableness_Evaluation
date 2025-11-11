from __future__ import annotations

import warnings

from src.temporal_reasoning.region_analysis import plot_region_temporal_stats as _region_plot

warnings.warn(
    "'src.temporal_reasoning.tongue_analysis.plot_tongue_stats' 已重命名为 'src.temporal_reasoning.region_analysis.plot_region_temporal_stats'，"
    "请更新脚本路径以消除该警告。",
    DeprecationWarning,
    stacklevel=2,
)

load_report = _region_plot.load_report
plot_metrics = _region_plot.plot_metrics
parse_args = _region_plot.parse_args
main = _region_plot.main

if __name__ == "__main__":
    main()
