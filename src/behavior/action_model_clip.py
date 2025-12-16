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
        - Batch inference processes multiple clips in one forward pass

        Example:
            >>> model = CLIPVideoActionModel(device='cuda')
            >>> clip = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
            >>> scores, top = model.predict_proba(clip, topk=3)
    """

    # Default classroom behavior descriptions (English)
    # Simplified to 5 robust categories for better accuracy with small faces in surveillance video
    DEFAULT_BEHAVIORS = [
        "a student sitting and looking straight ahead at the front of classroom, attentive and listening",
        "a student reading a textbook or writing notes with pen, focused on paper or book",
        "a student looking at and interacting with computer keyboard or smartphone screen, hands visible with device",
        "a student looking around in different directions with head turning, not focused, distracted",
        "other activity or unclear action not matching specific behaviors",
    ]

    # Short labels for output (matched to behaviors)
    DEFAULT_LABELS = [
        "listening",
        "reading_or_writing",
        "using_device",
        "distracted",
        "other",
    ]

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "auto",
        custom_behaviors: Optional[List[str]] = None,
        custom_labels: Optional[List[str]] = None,
    ) -> None:
        """Initialize CLIP model.

        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
                       ViT-B/32 is fastest, ViT-L/14 is most accurate
            device: Device ('auto', 'cuda', 'cpu')
            custom_behaviors: Custom behavior text descriptions
            custom_labels: Custom short labels (must match behaviors length)
        """
        self.model_name = str(model_name)

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
            raise ImportError("CLIP not installed. Run: pip install ftfy regex tqdm && pip install openai-clip") from e

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

        logger.info(f"CLIP model ready with {len(self.behaviors)} behaviors")

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
        if frames.shape[0] == 0:
            raise ValueError("Empty frames array")

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

    def _predict_batch_optimized(
        self, clips: List[np.ndarray], topk: int
    ) -> Tuple[List[Dict[str, float]], List[ActionTopK]]:
        """真正的批量推理：所有clips的所有帧一起编码，最大化GPU利用率.

        Args:
            clips: List of (T, H, W, 3) uint8 arrays
            topk: Number of top predictions

        Returns:
            (list[scores_dict], list[ActionTopK])
        """
        if not clips:
            return [], []

        # 1. 预处理所有clips并收集frames（先转换为tensor以避免numpy shape不匹配）
        all_frame_tensors = []
        clip_frame_counts = []

        for clip in clips:
            if not isinstance(clip, np.ndarray) or clip.ndim != 4:
                raise ValueError("Each clip must be (T,H,W,3)")

            if clip.shape[0] == 0:
                raise ValueError("Empty clip")

            # 预处理此clip的所有帧（resize到224x224并归一化）
            frame_tensor = self._preprocess_frames_batch(clip)  # (T, 3, 224, 224)
            all_frame_tensors.append(frame_tensor)
            clip_frame_counts.append(frame_tensor.shape[0])

        # 2. 合并所有预处理后的帧tensor: (total_frames, 3, 224, 224)
        all_frames_concat = torch.cat(all_frame_tensors, dim=0)

        # 3. 一次性编码所有帧（已经预处理过了）
        with torch.no_grad():
            all_features = self.model.encode_image(all_frames_concat)  # (total_frames, D)
            all_features = all_features / all_features.norm(dim=-1, keepdim=True)

        # 4. 按clip分割特征并进行temporal aggregation
        start_idx = 0
        video_features_list = []

        for frame_count in clip_frame_counts:
            end_idx = start_idx + frame_count
            clip_features = all_features[start_idx:end_idx]  # (T', D)

            # Temporal aggregation: mean pooling
            video_feature = clip_features.mean(dim=0, keepdim=True)  # (1, D)
            video_feature = video_feature / video_feature.norm(dim=-1, keepdim=True)
            video_features_list.append(video_feature)

            start_idx = end_idx

        # 5. 批量计算所有video features与text features的相似度
        video_features_batch = torch.cat(video_features_list, dim=0)  # (B, D)
        similarity = (100.0 * video_features_batch @ self.text_features.T).softmax(dim=-1)  # (B, N_behaviors)
        probs_batch = similarity.cpu().numpy()  # (B, N_behaviors)

        # 6. 构建输出
        scores_list = []
        tops_list = []

        for probs in probs_batch:
            scores = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
            scores_list.append(scores)

            if topk <= 0:
                tops_list.append(ActionTopK(categories=[]))
            else:
                k = min(int(topk), len(self.labels))
                idxs = np.argsort(-probs)[:k]
                top = [(self.labels[i], float(probs[i])) for i in idxs]
                tops_list.append(ActionTopK(categories=top))

        return scores_list, tops_list

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

        # 使用优化的批处理方法
        scores_list, tops_list = self._predict_batch_optimized(clips, topk)

        if is_batch:
            return scores_list, tops_list
        return scores_list[0], tops_list[0]
