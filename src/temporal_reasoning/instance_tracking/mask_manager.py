from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import torch

from .types import ObjectInfo


def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask.bool()


def _compute_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    mask_a = _prepare_mask(mask_a)
    mask_b = _prepare_mask(mask_b)
    intersection = torch.logical_and(mask_a, mask_b).sum().float()
    union = torch.logical_or(mask_a, mask_b).sum().float()
    if union.item() == 0:
        return 0.0
    return float((intersection / union).item())


@dataclass
class MaskDictionary:
    """
    Manage the set of object masks for a frame.

    This is a simplified, dependency-light alternative to the original
    `MaskDictionaryModel` used in `temporal_coherence_score.py`.
    """

    mask_name: str = ""
    mask_height: int = 0
    mask_width: int = 0
    promote_type: str = "mask"
    labels: Dict[int, ObjectInfo] = field(default_factory=dict)

    def add_annotations(
        self,
        mask_list: torch.Tensor,
        box_list: torch.Tensor,
        label_list: Sequence[str],
        background_value: int = 0,
    ) -> None:
        assert self.promote_type == "mask", "Only mask prompts are supported."
        mask_img_shape = mask_list.shape[-2:]
        self.mask_height, self.mask_width = mask_img_shape
        annotations: Dict[int, ObjectInfo] = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_idx = background_value + idx + 1
            prepared_mask = _prepare_mask(mask)
            obj = ObjectInfo(
                instance_id=final_idx,
                mask=prepared_mask,
                class_name=str(label),
            )
            if box is not None and len(box) == 4:
                obj.x1, obj.y1, obj.x2, obj.y2 = [int(x) for x in box.tolist()] if hasattr(box, "tolist") else [int(x) for x in box]
            obj.update_box()
            annotations[final_idx] = obj
        self.labels = annotations

    def update_with_tracker(
        self,
        tracking_dict: "MaskDictionary",
        iou_threshold: float,
        objects_count: int,
    ) -> Tuple[int, "MaskDictionary"]:
        updated: Dict[int, ObjectInfo] = {}
        for obj in self.labels.values():
            matched_id = None
            for tracked in tracking_dict.labels.values():
                iou = _compute_iou(obj.mask, tracked.mask)
                if iou >= iou_threshold:
                    matched_id = tracked.instance_id
                    break
            if matched_id is None:
                objects_count += 1
                matched_id = objects_count
            updated_obj = ObjectInfo(
                instance_id=matched_id,
                mask=obj.mask,
                class_name=obj.class_name,
            )
            updated_obj.update_box()
            updated[matched_id] = updated_obj
        self.labels = updated
        return objects_count, self

    def clone(self) -> "MaskDictionary":
        new_dict = MaskDictionary(
            mask_name=self.mask_name,
            mask_height=self.mask_height,
            mask_width=self.mask_width,
            promote_type=self.promote_type,
        )
        cloned_labels: Dict[int, ObjectInfo] = {}
        for instance_id, obj in self.labels.items():
            mask = obj.mask.clone()
            cloned_obj = ObjectInfo(
                instance_id=instance_id,
                mask=mask,
                class_name=obj.class_name,
                logit=obj.logit,
                x1=obj.x1,
                y1=obj.y1,
                x2=obj.x2,
                y2=obj.y2,
            )
            cloned_obj.update_box()
            cloned_labels[instance_id] = cloned_obj
        new_dict.labels = cloned_labels
        return new_dict

    def get_class_name(self, instance_id: int) -> str:
        return self.labels[instance_id].class_name

