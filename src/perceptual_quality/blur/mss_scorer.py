# -*- coding: utf-8 -*-
"""MSS scorer thin wrapper (Q-Align based). Placeholder for future decoupling."""

from typing import Any, Callable, Dict, Optional


class MSSScorer:
    """Thin facade around Q-Align video scoring with decoupled frame loading.

    - Removes cross-project dependency by using an internal sliding-window loader.
    - Allows injecting a custom frame loader for maximum encapsulation and testability.
    """

    def __init__(
        self,
        device: str,
        model_paths: Dict[str, str],
        frame_loader: Optional[Callable[[str, int], Any]] = None,
        window_size: int = 3,
    ):
        self.device = device
        self.model_paths = model_paths
        self.window_size = window_size
        self._frame_loader = frame_loader

        # Defer heavy imports to when we actually use this class.
        self._initialized = False
        self._scorer = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        # Local imports to avoid global side-effects and hard deps at import time
        try:
            # Prefer the package export if available
            from q_align import QAlignVideoScorer  # type: ignore
        except Exception:
            # Fallback to the scorer defined in evaluate.scorer within the vendor package
            from q_align.evaluate.scorer import QAlignVideoScorer  # type: ignore

        if self._frame_loader is None:
            # Use the internal implementation to avoid cross-repo dependency
            from src.io.video import load_video_sliding_window  # type: ignore
            self._frame_loader = load_video_sliding_window
        self._scorer = QAlignVideoScorer(pretrained=self.model_paths.get("q_align_model", ""), device=self.device)
        self._initialized = True

    def score(self, video_path: str) -> Dict:
        self._ensure_init()
        frames = self._frame_loader(video_path, window_size=self.window_size)
        output = self._scorer(frames)

        # Support both wrappers: either returns a single tensor or a tuple
        if isinstance(output, tuple):
            last = output[-1]
            quality_scores = last
        else:
            quality_scores = output

        return {"quality_scores": quality_scores.tolist()}


