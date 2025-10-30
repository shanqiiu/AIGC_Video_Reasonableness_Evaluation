"""
Optical flow prediction utilities migrated from AIGC_detector/simple_raft.py
- Supports Farneback (fast), TV-L1 (accurate), optional RAFT if model present
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SimpleRAFT:
    """Unified optical-flow predictor supporting Farneback, TV-L1, RAFT."""

    def __init__(self, device: str = 'cpu', method: str = 'farneback', model_path: Optional[str] = None) -> None:
        self.device = device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu'
        self.method = method
        # If RAFT is requested but no model path provided, try default under project third_party
        if method == 'raft' and not model_path:
            model_path = self._default_raft_model_path()
        self.model_path = model_path
        self.raft_model = None

        if method == 'tvl1':
            self._init_tvl1()
        elif method == 'raft':
            self._init_raft()
        elif method == 'farneback':
            pass
        else:
            self.method = 'farneback'

    def _init_tvl1(self) -> None:
        try:
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            self.tvl1.setTau(0.25)
            self.tvl1.setLambda(0.15)
            self.tvl1.setTheta(0.3)
            self.tvl1.setScalesNumber(5)
            self.tvl1.setWarpingsNumber(5)
            self.tvl1.setEpsilon(0.01)
        except AttributeError:
            self.method = 'farneback'

    def _init_raft(self) -> None:
        if not self.model_path or not Path(self.model_path).exists():
            self.method = 'farneback'
            return
        try:
            # Try to locate third_party/RAFT/core relative to project
            raft_core_path = None
            for candidate in [
                Path(__file__).parent / 'third_party' / 'RAFT' / 'core',
                Path(__file__).parents[2] / 'third_party' / 'RAFT' / 'core',
            ]:
                if candidate.exists():
                    raft_core_path = candidate
                    break
            if raft_core_path is None:
                raise FileNotFoundError('third_party/RAFT/core not found')
            sys.path.insert(0, str(raft_core_path))
            from raft import RAFT  # type: ignore
            import argparse

            args = argparse.Namespace()
            args.small = False
            args.mixed_precision = False
            args.alternate_corr = False
            args.dropout = 0
            args.corr_levels = 4
            args.corr_radius = 4

            self.raft_model = RAFT(args)
            state_dict = torch.load(self.model_path, map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.raft_model.load_state_dict(new_state_dict, strict=False)
            self.raft_model.to(self.device)
            self.raft_model.eval()
        except Exception:
            self.method = 'farneback'

    def _default_raft_model_path(self) -> Optional[str]:
        # Project root assumed at parents[2]
        candidate = Path(__file__).parents[2] / 'third_party' / 'pretrained_models' / 'raft-things.pth'
        return str(candidate) if candidate.exists() else None

    def predict_flow(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        if self.method == 'raft' and self.raft_model is not None:
            return self._predict_flow_raft(image1, image2)
        return self._predict_flow_opencv(image1, image2)

    def _predict_flow_raft(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            img1 = self._preprocess_image_raft(image1)
            img2 = self._preprocess_image_raft(image2)
            _, flow_up = self.raft_model(img1, img2, iters=20, test_mode=True)
            flow = flow_up[0].cpu().numpy()
            return flow  # (2, H, W)

    def _preprocess_image_raft(self, img: np.ndarray) -> torch.Tensor:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        _, _, h, w = img_tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        return img_tensor.to(self.device)

    def _predict_flow_opencv(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if image1.ndim == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if image2.ndim == 3 else image2
        if self.method == 'tvl1':
            flow = self.tvl1.calc(gray1, gray2, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5,
                levels=5,
                winsize=15,
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0,
            )
        if isinstance(flow, np.ndarray):
            return np.asarray(flow.transpose(2, 0, 1))  # (2, H, W)
        h, w = image1.shape[:2]
        return np.zeros((2, h, w), dtype=np.float32)


class SimpleRAFTPredictor:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu', method: str = 'farneback') -> None:
        self.model = SimpleRAFT(device=device, method=method, model_path=model_path)

    def predict_flow(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        return self.model.predict_flow(image1, image2)

    def predict_flow_sequence(self, images: List[np.ndarray]) -> List[np.ndarray]:
        flows: List[np.ndarray] = []
        for i in range(len(images) - 1):
            flows.append(self.predict_flow(images[i], images[i + 1]))
        return flows


