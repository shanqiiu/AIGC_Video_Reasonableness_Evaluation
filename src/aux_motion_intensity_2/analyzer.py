"""
�������� - ����Grounded-SAM��Co-Tracker���пɸ�֪��������
"""

import os
import sys
import math
import json
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import cv2

# ��ӵ�������·��
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
gsa_path = os.path.join(project_root, "third_party", "Grounded-Segment-Anything")
gdn_path = os.path.join(gsa_path, "GroundingDINO")
sys.path.insert(0, gsa_path)  # ��ҪGrounded-Segment-Anything��·����
sys.path.insert(0, gdn_path)  # GroundingDINO��Ҫ����
sys.path.insert(0, os.path.join(gsa_path, "segment_anything"))
sys.path.insert(0, os.path.join(project_root, "third_party", "co-tracker"))

# Grounding DINO - ʹ������ģ��·��
# type: ignore ע�����ں���linter���棨��Щģ��������ʱ���ã�
import groundingdino.datasets.transforms as T  # type: ignore
from groundingdino.models import build_model  # type: ignore
from groundingdino.util.slconfig import SLConfig  # type: ignore
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap  # type: ignore

# Segment Anything
from segment_anything import (  # type: ignore
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

# Co-Tracker
from cotracker.utils.visualizer import Visualizer  # type: ignore
from cotracker.predictor import CoTrackerPredictor  # type: ignore

from .motion_calculator import calculate_motion_degree, is_mask_suitable_for_tracking
from .scene_classifier import SceneClassifier


def load_video(video_path):
    """������Ƶ�����ص�һ֡������֡"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None, None, None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        print("Error reading frames from the video")
        return None, None, None, None

    # take the first frame as the query image
    frame_rgb = frames[0]
    image_pil = Image.fromarray(frame_rgb)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    
    return image_pil, image, frame_rgb, np.stack(frames)


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """ʹ��Grounding DINO����Ŀ����"""
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


class PASAnalyzer:
    """�ɸ�֪���ȷ�����"""
    
    def __init__(self,
                 device: str = "cuda",
                 grid_size: int = 30,
                 enable_scene_classification: bool = False,
                 scene_classifier_params: Optional[Dict] = None,
                 # ģ��·��
                 grounded_checkpoint: Optional[str] = None,
                 sam_checkpoint: Optional[str] = None,
                 cotracker_checkpoint: Optional[str] = None):
        """
        ��ʼ��������
        
        Args:
            device: �豸 ('cuda' or 'cpu')
            grid_size: Co-Tracker�����С
            enable_scene_classification: �Ƿ����ó�������
            scene_classifier_params: ���������������ֵ�
            grounded_checkpoint: GroundingDINOģ��·��
            sam_checkpoint: SAMģ��·��
            cotracker_checkpoint: Co-Trackerģ��·��
        """
        self.device = device
        self.grid_size = grid_size
        self.enable_scene_classification = enable_scene_classification
        
        # ����ģ��·��
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        gsa_path = os.path.join(project_root, "third_party", "Grounded-Segment-Anything")
        
        self.config_file = os.path.join(
            gsa_path,
            "GroundingDINO",
            "groundingdino",
            "config",
            "GroundingDINO_SwinB.py"
        )
        
        self.grounded_checkpoint = grounded_checkpoint or os.path.join(project_root, ".cache", "groundingdino_swinb_cogcoor.pth")
        self.bert_base_uncased_path = os.path.join(project_root, ".cache", "google-bert", "bert-base-uncased")
        self.sam_checkpoint = sam_checkpoint or os.path.join(project_root, ".cache", "sam_vit_h_4b8939.pth")
        self.cotracker_checkpoint = cotracker_checkpoint or os.path.join(project_root, ".cache", "scaled_offline.pth")
        
        # ��ʼ������������
        if self.enable_scene_classification:
            self.scene_classifier = SceneClassifier(**{**{
                'static_threshold': 0.1,
                'low_dynamic_threshold': 0.3,
                'medium_dynamic_threshold': 0.6,
                'high_dynamic_threshold': 1.0,
                'motion_ratio_threshold': 1.5
            }, **(scene_classifier_params or {})})
        else:
            self.scene_classifier = None
        
        # ģ�ͼ��ر�־
        self._models_loaded = False
        self.grounding_model = None
        self.sam_predictor = None
        self.cotracker_model = None
    
    def _load_models(self):
        """�ӳټ���ģ��"""
        if self._models_loaded:
            return
        
        print("Loading models...")
        # Load Grounding DINO
        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        args.bert_base_uncased_path = self.bert_base_uncased_path
        self.grounding_model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_model.eval()
        
        # Initialize SAM
        sam_version = "vit_h"
        self.sam_predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=self.sam_checkpoint).to(self.device))
        
        # Initialize Co-Tracker
        self.cotracker_model = CoTrackerPredictor(
            checkpoint=self.cotracker_checkpoint,
            v2=False,
            offline=True,
            window_len=60,
        ).to(self.device)
        
        self._models_loaded = True
        print("Models loaded successfully")
    
    def analyze_video(self, 
                     video_path: str,
                     subject_noun: str,
                     box_threshold: float = 0.3,
                     text_threshold: float = 0.25,
                     normalize_by_subject_diag: bool = True) -> Dict:
        """
        ������Ƶ������ɸ�֪���ȷ���
        
        Args:
            video_path: ��Ƶ·��
            subject_noun: �������ʣ��� "person", "dog"��
            box_threshold: ������ֵ
            text_threshold: �ı���ֵ
            normalize_by_subject_diag: �Ƿ�����Խ��߹�һ��
            
        Returns:
            �����˶������ͳ�������Ľ���ֵ�
        """
        # �ӳټ���ģ��
        if not self._models_loaded:
            self._load_models()
        
        # ������Ƶ
        image_pil, image, image_array, video = load_video(video_path)
        if video is None:
            return {
                'status': 'error',
                'error_reason': 'failed_to_load_video'
            }
        
        # ׼���ı���ʾ
        text_prompt = subject_noun + '.'
        
        # ���� Grounding DINO
        boxes_filt, pred_phrases = get_grounding_output(
            self.grounding_model, image, text_prompt, box_threshold, text_threshold, device=self.device
        )
        
        # ���ʧ��
        if boxes_filt.shape[0] == 0:
            print(f"Cannot detect {text_prompt} in video")
            background_motion = self._calculate_background_motion(video)
            return {
                'status': 'error',
                'error_reason': 'no_subject_detected',
                'background_motion': background_motion
            }
        
        # ���� SAM �ָ�
        self.sam_predictor.set_image(image_array)
        
        # ת�����ʽ
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        
        # ���� SAM ģ��
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        
        # ׼����Ƶ����
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_width, video_height = video_tensor.shape[-1], video_tensor.shape[-2]
        video_tensor = video_tensor.to(self.device)
        
        # ���㱳���˶�
        if boxes_filt.shape[0] != 0 and masks is not None:
            background_mask = torch.any(~masks, dim=0).to(torch.uint8) * 255
        else:
            background_mask = torch.ones((1, video_height, video_width), dtype=torch.uint8, device=self.device) * 255
        
        background_mask = background_mask.unsqueeze(0)
        
        pred_tracks, pred_visibility = self.cotracker_model(
            video_tensor,
            grid_size=self.grid_size,
            grid_query_frame=0,
            backward_tracking=True,
            segm_mask=background_mask
        )
        
        if pred_tracks.shape[2] == 0:
            background_motion = 0.0
        else:
            background_motion = calculate_motion_degree(pred_tracks, video_width, video_height).item()
        
        # ���������˶�
        if boxes_filt.shape[0] != 0 and masks is not None:
            subject_mask = torch.any(masks, dim=0).to(torch.uint8) * 255
            subject_mask = subject_mask.unsqueeze(0)
            
            subject_mask_valid = torch.sum(subject_mask > 0).item() > 0
            mask_suitable = is_mask_suitable_for_tracking(subject_mask, video_width, video_height, self.grid_size)
            
            if not subject_mask_valid:
                subject_motion = 0.0
                result = {
                    'status': 'error',
                    'error_reason': 'subject_mask_empty',
                    'background_motion': background_motion,
                    'subject_motion': 0.0
                }
            elif not mask_suitable:
                print(f"Warning: Mask unsuitable for tracking")
                subject_motion = 0.0
                result = {
                    'status': 'error',
                    'error_reason': 'mask_too_small',
                    'background_motion': background_motion,
                    'subject_motion': 0.0
                }
            else:
                pred_tracks, pred_visibility = self.cotracker_model(
                    video_tensor,
                    grid_size=self.grid_size,
                    grid_query_frame=0,
                    backward_tracking=True,
                    segm_mask=subject_mask
                )
                
                if pred_tracks.shape[2] == 0:
                    print(f"Warning: Empty tracks despite suitable mask")
                    subject_motion = 0.0
                    result = {
                        'status': 'error',
                        'error_reason': 'empty_tracks',
                        'background_motion': background_motion,
                        'subject_motion': 0.0
                    }
                else:
                    subject_motion = calculate_motion_degree(pred_tracks, video_width, video_height).item()
                    
                    # ������Խ��߹�һ��
                    if normalize_by_subject_diag:
                        try:
                            mask2d = subject_mask.squeeze()
                            coords = (mask2d > 0).nonzero()
                            if coords.numel() > 0:
                                ys = coords[:, -2]
                                xs = coords[:, -1]
                                subj_h = (ys.max() - ys.min() + 1).item()
                                subj_w = (xs.max() - xs.min() + 1).item()
                                subj_diag = math.sqrt(float(subj_w) ** 2 + float(subj_h) ** 2)
                                if subj_diag > 0:
                                    video_diag = math.sqrt(float(video_width) ** 2 + float(video_height) ** 2)
                                    scale = video_diag / subj_diag
                                    subject_motion *= scale
                        except Exception as e:
                            print(f"Warning: Failed to normalize by subject diagonal: {e}")
                    
                    # ������ϸ�˶�����
                    pure_subject = max(0, subject_motion - background_motion)
                    total_motion = background_motion + subject_motion
                    motion_ratio = pure_subject / (background_motion + 1e-8)
                    
                    result = {
                        'status': 'success',
                        'background_motion': float(background_motion),
                        'subject_motion': float(subject_motion),
                        'pure_subject_motion': float(pure_subject),
                        'total_motion': float(total_motion),
                        'motion_ratio': float(motion_ratio),
                        'video_resolution': {
                            'width': int(video_width),
                            'height': int(video_height),
                            'diagonal': float(torch.sqrt(torch.tensor(video_width**2 + video_height**2)).item()),
                        }
                    }
                    
                    # ��������
                    if self.scene_classifier:
                        scene_info = self.scene_classifier.classify_scene(
                            background_motion, subject_motion, pure_subject, motion_ratio
                        )
                        result['scene_classification'] = scene_info
                    
            return result
        else:
            return {
                'status': 'error',
                'error_reason': 'no_subject_detected',
                'background_motion': background_motion
            }
    
    def _calculate_background_motion(self, video: np.ndarray) -> float:
        """���㱳���˶�����������ʱʹ�ã�"""
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_width, video_height = video_tensor.shape[-1], video_tensor.shape[-2]
        video_tensor = video_tensor.to(self.device)
        
        background_mask = torch.ones((1, video_height, video_width), dtype=torch.uint8, device=self.device) * 255
        background_mask = background_mask.unsqueeze(0)
        
        pred_tracks, _ = self.cotracker_model(
            video_tensor,
            grid_size=self.grid_size,
            grid_query_frame=0,
            backward_tracking=True,
            segm_mask=background_mask
        )
        
        if pred_tracks.shape[2] == 0:
            return 0.0
        else:
            return calculate_motion_degree(pred_tracks, video_width, video_height).item()

