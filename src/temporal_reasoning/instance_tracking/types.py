from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ObjectInfo:
    """Light-weight container for a tracked object's metadata."""

    instance_id: int
    mask: torch.Tensor
    class_name: str
    logit: float = 0.0
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None

    def update_box(self) -> None:
        """
        Populate the bounding box fields from the binary mask.
        Assumes mask is a boolean tensor of shape (H, W) or (1, H, W).
        """
        mask = self.mask
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        nonzero = torch.nonzero(mask, as_tuple=False)
        if nonzero.numel() == 0:
            self.x1 = self.y1 = self.x2 = self.y2 = None
            return

        y_min, x_min = torch.min(nonzero, dim=0)[0]
        y_max, x_max = torch.max(nonzero, dim=0)[0]
        self.x1 = int(x_min.item())
        self.y1 = int(y_min.item())
        self.x2 = int(x_max.item())
        self.y2 = int(y_max.item())

    def to_serializable(self) -> Dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "mask": self.mask.cpu().numpy().tolist(),
            "class_name": self.class_name,
            "logit": float(self.logit),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass
class FrameObjectSet:
    """Store objects detected/tracked in a single frame."""

    objects: Dict[int, ObjectInfo] = field(default_factory=dict)

    def to_serializable(self) -> Dict[int, Dict]:
        return {instance_id: obj.to_serializable() for instance_id, obj in self.objects.items()}

