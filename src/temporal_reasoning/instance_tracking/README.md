# Instance Tracking Module

This directory hosts the refactored temporal coherence pipeline derived from `temporal_coherence_score.py`.

## Key Entry Points

- `TemporalCoherencePipeline`: High-level API that orchestrates detection, tracking, and temporal scoring.
- `TemporalCoherenceConfig`: Configuration dataclass covering model paths, thresholds, and runtime options.

## Quick Start

```python
from src.temporal_reasoning.instance_tracking import (
    TemporalCoherencePipeline,
    TemporalCoherenceConfig,
)

config = TemporalCoherenceConfig(
    meta_info_path="./bad_case/meta_info.json",
    text_prompt="car. person.",
)

pipeline = TemporalCoherencePipeline(config)
pipeline.initialize()
results = pipeline.process_meta_info_file()
```

The call updates the configured `meta_info_path` in place and returns the processed metadata list. See `pipeline.py` for additional extension points.

