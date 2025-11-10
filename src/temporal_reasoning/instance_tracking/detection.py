from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from PIL import Image

import sys
from pathlib import Path

# Ensure SAM2 package is importable when using the bundled third_party version.
SAM2_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Grounded-SAM-2"
if SAM2_ROOT.exists():
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2, build_sam2_video_predictor  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

from bench_utils.tcs_utils import load_model, get_grounding_output, transform_pil  # type: ignore

from .mask_manager import MaskDictionary


@dataclass
class DetectionConfig:
    grounding_config_path: str
    grounding_checkpoint_path: str
    bert_path: str
    sam2_config_path: str
    sam2_checkpoint_path: str
    box_threshold: float
    text_threshold: float
    device: str = "cuda"


class Sam2DetectionEngine:
    """
    Wrap Grounding DINO + SAM2 image/video predictor logic.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.video_predictor = None
        self.image_predictor = None
        self.grounding_model = None
        self.device = "cpu"

    def initialize(self) -> None:
        device = self.config.device if torch.cuda.is_available() else "cpu"
        self.device = device

        self.video_predictor = build_sam2_video_predictor(
            self.config.sam2_config_path,
            self.config.sam2_checkpoint_path,
        )
        sam2_image_model = build_sam2(
            self.config.sam2_config_path,
            self.config.sam2_checkpoint_path,
            device=device,
        )
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.grounding_model = load_model(
            self.config.grounding_config_path,
            self.config.grounding_checkpoint_path,
            self.config.bert_path,
            device=device,
        )

    def init_video_state(self, video_path: str):
        return self.video_predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )

    def detect(self, image: Image.Image, text_prompt: str, promote_type: str = "mask") -> MaskDictionary:
        assert self.image_predictor is not None and self.grounding_model is not None

        image_tensor = transform_pil(image)
        with torch.cuda.amp.autocast(enabled=False):
            boxes, phrases = get_grounding_output(
                self.grounding_model,
                image_tensor,
                text_prompt,
                self.config.box_threshold,
                self.config.text_threshold,
                with_logits=False,
                device=self.device,
            )

        if boxes.shape[0] == 0:
            empty_dict = MaskDictionary(promote_type=promote_type)
            return empty_dict

        size = image.size
        width, height = size[0], size[1]
        boxes_xyxy = boxes.clone()
        for i in range(boxes_xyxy.size(0)):
            boxes_xyxy[i] = boxes_xyxy[i] * torch.Tensor([width, height, width, height])
            boxes_xyxy[i][:2] -= boxes_xyxy[i][2:] / 2
            boxes_xyxy[i][2:] += boxes_xyxy[i][:2]

        self.image_predictor.set_image(np.array(image.convert("RGB")))
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )

        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        masks_tensor = torch.as_tensor(masks)
        mask_dict = MaskDictionary(promote_type=promote_type)
        mask_dict.add_annotations(masks_tensor, boxes_xyxy, phrases)
        return mask_dict

    def add_masks_to_video_state(self, inference_state, frame_idx: int, mask_dict: MaskDictionary) -> List[int]:
        object_ids = []
        for object_id, object_info in mask_dict.labels.items():
            _, out_obj_ids, _ = self.video_predictor.add_new_mask(
                inference_state,
                frame_idx,
                object_id,
                object_info.mask,
            )
            object_ids.extend(out_obj_ids)
        return object_ids

    def propagate(self, inference_state, step: int, start_frame_idx: int) -> Tuple[int, List[int], torch.Tensor]:
        yield from self.video_predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=step,
            start_frame_idx=start_frame_idx,
        )
