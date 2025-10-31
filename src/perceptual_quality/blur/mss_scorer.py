# -*- coding: utf-8 -*-
"""MSS scorer thin wrapper (Q-Align based). Placeholder for future decoupling."""

from typing import Dict


class MSSScorer:
    """Thin facade; currently unused by BlurDetector, kept for planned modularization."""

    def __init__(self, device: str, model_paths: Dict[str, str]):
        self.device = device
        self.model_paths = model_paths

        # Defer heavy imports to when we actually use this class.
        self._initialized = False
        self._scorer = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        # Local import to avoid global side-effects
        from q_align import QAlignVideoScorer  # type: ignore
        from motion_smoothness_score import load_video_sliding_window  # type: ignore
        self._load_video_sliding_window = load_video_sliding_window
        self._scorer = QAlignVideoScorer(pretrained=self.model_paths.get("q_align_model", ""), device=self.device)
        self._initialized = True

    def score(self, video_path: str) -> Dict:
        self._ensure_init()
        frames = self._load_video_sliding_window(video_path, window_size=3)
        _, _, quality_scores = self._scorer(frames)
        return {"quality_scores": quality_scores.tolist()}


