from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class ActionTopK:
    categories: List[Tuple[str, float]]


class TorchvisionVideoActionModel:
    """Torchvision video classification model wrapper.

    Notes:
    - This uses torchvision's pretrained video models (e.g., r3d_18, swin3d_t)
      which are typically trained on Kinetics. Classroom behaviors may require
      fine-tuning, but we can still extract strong action priors.
    - Input clip is (T, H, W, 3) RGB uint8.
    """

    def __init__(
        self,
        model_name: str = "swin3d_t",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)

        self.model, self.categories, self._preprocess = self._load_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_model(model_name: str):
        import torchvision
        from torchvision.models.video import (
            MViT_V1_B_Weights,
            R3D_18_Weights,
            S3D_Weights,
            Swin3D_T_Weights,
            mvit_v1_b,
            r3d_18,
            s3d,
            swin3d_t,
        )

        name = model_name.lower().strip()

        if name in {"swin3d_t", "swin3d-t", "swin"}:
            weights = Swin3D_T_Weights.DEFAULT
            model = swin3d_t(weights=weights)
        elif name in {"mvit_v1_b", "mvit", "mvitv1"}:
            weights = MViT_V1_B_Weights.DEFAULT
            model = mvit_v1_b(weights=weights)
        elif name in {"s3d"}:
            weights = S3D_Weights.DEFAULT
            model = s3d(weights=weights)
        elif name in {"r3d_18", "r3d", "r3d-18"}:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name={model_name}. Use one of: swin3d_t, mvit_v1_b, s3d, r3d_18")

        categories = list(weights.meta.get("categories") or [])
        preprocess = weights.transforms()

        return model, categories, preprocess

    def predict_proba(self, clip_rgb_uint8: np.ndarray, *, topk: int = 10) -> Tuple[Dict[str, float], ActionTopK]:
        if clip_rgb_uint8.ndim != 4 or clip_rgb_uint8.shape[-1] != 3:
            raise ValueError("clip_rgb_uint8 must have shape (T,H,W,3)")

        # Torchvision video weight presets expect video tensor shaped as (T, C, H, W).
        clip = torch.from_numpy(clip_rgb_uint8).to(torch.uint8)  # (T,H,W,3)
        clip = clip.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)

        with torch.no_grad():
            v = self._preprocess(clip)  # (3,T,H,W)
            inp = v.unsqueeze(0).to(self.device)  # (1,3,T,H,W)
            logits = self.model(inp)
            probs = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()

        cats = self.categories
        if not cats or len(cats) != int(probs.shape[0]):
            # Fallback to numeric labels.
            cats = [f"class_{i}" for i in range(int(probs.shape[0]))]

        scores: Dict[str, float] = {cats[i]: float(probs[i]) for i in range(len(cats))}

        if topk <= 0:
            return scores, ActionTopK(categories=[])

        k = int(min(topk, len(cats)))
        idxs = np.argpartition(-probs, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-probs[idxs])]
        top = [(cats[i], float(probs[i])) for i in idxs.tolist()]
        return scores, ActionTopK(categories=top)


def ensure_min_frames(clip: Sequence[np.ndarray], *, min_frames: int) -> List[np.ndarray]:
    frames = list(clip)
    if len(frames) >= int(min_frames):
        return frames
    if not frames:
        raise ValueError("empty clip")
    # Pad by repeating last frame.
    while len(frames) < int(min_frames):
        frames.append(frames[-1])
    return frames
