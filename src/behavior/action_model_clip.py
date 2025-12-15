"""CLIP-based zero-shot video action recognition.

This module provides a lightweight alternative to Kinetics-pretrained models,
allowing custom behavior definitions without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class ActionTopK:
    """Compatible with action_model.ActionTopK"""

    categories: List[Tuple[str, float]]


class CLIPVideoActionModel:
    """Zero-shot video action classification using OpenAI CLIP.

        This implementation:
        - Uses pretrained CLIP without any classroom-specific training
        - Supports custom behavior descriptions via text prompts
        - Performs temporal aggregation by averaging frame features
        - Optimized for GPU batch processing
    GPU-optimized: direct tensor processing without PIL conversion
        - Minimizes CPU-GPU transfers for maximum throughput

        Performance optimizations:
        - All preprocessing done in batched GPU operations
        - Single memory transfer from CPU to GPU per clip
        - No intermediate PIL Image conversions
        - Frame subsampling reduces computation by 2-8x
        Example:
            >>> model = CLIPVideoActionModel(device='cuda')
            >>> clip = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
            >>> scores, top = model.predict_proba(clip, topk=5)
    """

    # Default classroom behavior descriptions (English)
    DEFAULT_BEHAVIORS = [
        "a student holding a pen and writing on a notebook or tablet",
        "a student holding and reading a textbook or flipping through pages of a book",
        "a student sitting and looking straight ahead at the front of classroom",
        "a student holding a smartphone in hands and looking down at the phone screen with head lowered",
        "a student looking at a laptop computer screen on desk with hands on keyboard",
        "a student picking up a water bottle or cup, drinking water and putting it down",
        "a student looking around in different directions, not looking straight ahead",
        "a student with arms on desk and head lying down on the desk, or with eyes completely closed",
        "a student turning head to talk with a classmate sitting nearby",
    ]

    # Short labels for output (matched to behaviors)
    DEFAULT_LABELS = [
        "taking_notes",
        "reading",
        "listening",
        "using_phone",
        "using_computer",
        "drinking",
        "distracted",
        "sleeping",
        "talking",
    ]

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "auto",
        custom_behaviors: Optional[List[str]] = None,
        custom_labels: Optional[List[str]] = None,
        frame_subsample: int = 4,  # Process every Nth frame for speed
    ) -> None:
        """Initialize CLIP model.

        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
                       ViT-B/32 is fastest, ViT-L/14 is most accurate
            device: Device ('auto', 'cuda', 'cpu')
            custom_behaviors: Custom behavior text descriptions
            custom_labels: Custom short labels (must match behaviors length)
            frame_subsample: Process every Nth frame (1=all frames, 4=every 4th)
        """
        self.model_name = str(model_name)
        self.frame_subsample = max(1, int(frame_subsample))

        # Device setup
        dev = str(device).strip().lower()
        if dev in {"auto", "gpu"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = dev

        logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")

        # Load CLIP (will auto-download if needed)
        try:
            import clip
        except ImportError as e:
            raise ImportError(
                "CLIP not installed. Run: pip install ftfy regex tqdm && "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from e

        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

        # Setup behavior categories
        self.behaviors = custom_behaviors or self.DEFAULT_BEHAVIORS
        if custom_labels:
            if len(custom_labels) != len(self.behaviors):
                raise ValueError(
                    f"custom_labels length ({len(custom_labels)}) must match behaviors ({len(self.behaviors)})"
                )
            self.labels = custom_labels
        else:
            self.labels = self.DEFAULT_LABELS[: len(self.behaviors)]

        # Pre-encode all text descriptions (only once)
        logger.info(f"Encoding {len(self.behaviors)} behavior descriptions")
        with torch.no_grad():
            text_inputs = clip.tokenize(self.behaviors).to(self.device)
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        logger.info(f"CLIP model ready with {len(self.behaviors)} behaviors (frame_subsample={self.frame_subsample})")

    def _preprocess_frames_batch(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess frames batch directly without PIL conversion (GPU-optimized).

        Args:
            frames: (T, H, W, 3) uint8 numpy array

        Returns:
            Preprocessed tensor (T, 3, 224, 224) ready for CLIP
        """
        import torchvision.transforms.functional as F

        # Convert to torch tensor and move to GPU immediately: (T, H, W, 3) -> (T, 3, H, W)
        batch = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device).float()

        # Normalize to [0, 1]
        batch = batch / 255.0

        # Resize to 224x224 (CLIP input size) - batched operation on GPU
        batch = torch.nn.functional.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False)

        # CLIP normalization (ImageNet stats)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        return batch

    def _encode_frames_batch(self, frames: np.ndarray) -> torch.Tensor:
        """Encode multiple frames efficiently in batch.

        Args:
            frames: (T, H, W, 3) uint8 array

        Returns:
            Normalized frame features (T, D)
        """
        # Subsample frames for speed
        if self.frame_subsample > 1:
            indices = list(range(0, frames.shape[0], self.frame_subsample))
            frames = frames[indices]

        if frames.shape[0] == 0:
            raise ValueError("No frames after subsampling")

        # Preprocess all frames in batch on GPU (no PIL conversion)
        batch = self._preprocess_frames_batch(frames)  # (T, 3, 224, 224)

        # Encode all frames in one forward pass
        with torch.no_grad():
            features = self.model.encode_image(batch)  # (T, D)
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    def _predict_single(self, clip_rgb_uint8: np.ndarray, topk: int) -> Tuple[Dict[str, float], ActionTopK]:
        """Predict probabilities for a single clip.

        Args:
            clip_rgb_uint8: (T, H, W, 3) uint8 array
            topk: Number of top predictions

        Returns:
            (scores_dict, ActionTopK)
        """
        if not isinstance(clip_rgb_uint8, np.ndarray) or clip_rgb_uint8.ndim != 4:
            raise ValueError("clip_rgb_uint8 must be (T,H,W,3)")

        # 1. Encode all frames in batch
        frame_features = self._encode_frames_batch(clip_rgb_uint8)  # (T', D)

        # 2. Temporal aggregation: mean pooling
        video_features = frame_features.mean(dim=0, keepdim=True)  # (1, D)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # 3. Compute similarity with all behavior texts
        similarity = (100.0 * video_features @ self.text_features.T).softmax(dim=-1)
        probs = similarity.squeeze(0).cpu().numpy()  # (N_behaviors,)

        # 4. Build output
        scores = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        if topk <= 0:
            top_obj = ActionTopK(categories=[])
        else:
            k = min(int(topk), len(self.labels))
            idxs = np.argsort(-probs)[:k]
            top = [(self.labels[i], float(probs[i])) for i in idxs]
            top_obj = ActionTopK(categories=top)

        return scores, top_obj

    def predict_proba(
        self,
        clip_rgb_uint8: Union[np.ndarray, Sequence[np.ndarray]],
        *,
        topk: int = 10,
    ) -> Tuple[Union[Dict[str, float], List[Dict[str, float]]], Union[ActionTopK, List[ActionTopK]]]:
        """Predict behavior probabilities (compatible with TorchvisionVideoActionModel).

        Supports:
        - Single clip: ndarray (T, H, W, 3)
        - Batch clips: sequence of ndarrays

        Args:
            clip_rgb_uint8: Single or multiple video clips
            topk: Number of top predictions

        Returns:
            For single clip: (scores_dict, ActionTopK)
            For batch: (list[scores_dict], list[ActionTopK])
        """
        is_batch = isinstance(clip_rgb_uint8, (list, tuple))
        clips = list(clip_rgb_uint8) if is_batch else [clip_rgb_uint8]  # type: ignore[list-item]

        if not clips:
            return [], []

        scores_list = []
        tops_list = []

        for clip in clips:
            scores, top = self._predict_single(clip, topk)
            scores_list.append(scores)
            tops_list.append(top)

        if is_batch:
            return scores_list, tops_list
        return scores_list[0], tops_list[0]
