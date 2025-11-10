# Temporal Reasoning Execution Flow

```
TemporalReasoningAnalyzer.analyze()
©À©¤ MotionFlowAnalyzer.analyze()          ¡ú motion_score, motion_anomalies
©À©¤ TemporalCoherencePipeline.evaluate_video()
©¦  ©À©¤ SAM2 + GroundingDINO detection per sampling step
©¦  ©À©¤ SAM2 video propagation + mask bookkeeping
©¦  ©À©¤ CoTracker-based vanish/emerge evaluation
©¦  ©¸©¤ Outputs coherence_score, vanish/emerge sub-scores, structure_anomalies
©À©¤ KeypointAnalyzer.analyze()            ¡ú physiological_score, physiological_anomalies
©¸©¤ FusionDecisionEngine
   ©À©¤ align_anomalies_spatially_and_temporally()
   ©À©¤ fuse_multimodal_anomalies()
   ©À©¤ (optional) AnomalyFilter.filter_anomalies()
   ©¸©¤ compute_final_scores()             ¡ú motion_reasonableness, structure_stability
```

Core data flow:

1. **Motion branch** computes optical-flow smoothness and discontinuities.
2. **Structure branch** reuses the refactored instance-tracking pipeline to track objects, detect disappear/appear events, and exposes both anomalies and vanish/emerge metrics.
3. **Keypoint branch** captures physiological consistency.
4. **Fusion layer** aligns anomalies temporally/spatially, fuses confidences, and derives final scores with penalties informed by the structure metrics.

The analyzer returns:

- Final motion/structure scores
- Aggregated anomaly list (with modalities & metadata)
- Per-modality sub-scores and anomaly counts
- Structure branch metrics (`coherence_score`, `vanish_score`, `emerge_score`, plus pipeline metadata)

