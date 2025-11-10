"""
Instance tracking module.

Provides a high-level temporal coherence scorer that reuses the
implementation ideas from `temporal_coherence_score.py` while exposing a
low-coupling, testable API for the rest of the package.
"""

from .pipeline import TemporalCoherencePipeline, TemporalCoherenceConfig

__all__ = ["TemporalCoherencePipeline", "TemporalCoherenceConfig"]

