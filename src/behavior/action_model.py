from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

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
        device: str = "auto",
    ) -> None:
        self.model_name = str(model_name)
        dev = str(device).strip().lower()
        if dev in {"auto"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev in {"gpu", "cuda"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev in {"cpu"}:
            self.device = "cpu"
        else:
            # Allow advanced torch device strings like "cuda:0".
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

    def predict_proba(
        self,
        clip_rgb_uint8: Union[np.ndarray, Sequence[np.ndarray]],
        *,
        topk: int = 10,
    ) -> Tuple[Union[Dict[str, float], List[Dict[str, float]]], Union[ActionTopK, List[ActionTopK]]]:
        """Predict category probabilities.

        Supports:
        - single clip: ndarray shaped (T,H,W,3)
        - batch clips: sequence of ndarrays, each shaped (T,H,W,3)

        For backward compatibility, the return type is:
        - single clip: (scores_dict, ActionTopK)
        - batch: (list[scores_dict], list[ActionTopK])
        """

        is_batch = isinstance(clip_rgb_uint8, (list, tuple))
        clips = list(clip_rgb_uint8) if is_batch else [clip_rgb_uint8]  # type: ignore[list-item]

        if not clips:
            return [], []

        vids: List[torch.Tensor] = []
        for clip in clips:
            if not isinstance(clip, np.ndarray) or clip.ndim != 4 or clip.shape[-1] != 3:
                raise ValueError("clip_rgb_uint8 must have shape (T,H,W,3)")
            t = torch.from_numpy(clip).to(torch.uint8)  # (T,H,W,3)
            t = t.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
            v = self._preprocess(t)  # (3,T,H,W)
            vids.append(v)

        batch = torch.stack(vids, dim=0).to(self.device)  # (N,3,T,H,W)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy()  # (N,C)

        cats = self.categories
        if not cats or len(cats) != int(probs.shape[1]):
            cats = [f"class_{i}" for i in range(int(probs.shape[1]))]

        scores_list: List[Dict[str, float]] = []
        tops_list: List[ActionTopK] = []

        for n in range(int(probs.shape[0])):
            p = probs[n]
            scores: Dict[str, float] = {cats[i]: float(p[i]) for i in range(len(cats))}

            if topk <= 0:
                tops_list.append(ActionTopK(categories=[]))
            else:
                k = int(min(int(topk), len(cats)))
                idxs = np.argpartition(-p, kth=k - 1)[:k]
                idxs = idxs[np.argsort(-p[idxs])]
                top = [(cats[i], float(p[i])) for i in idxs.tolist()]
                tops_list.append(ActionTopK(categories=top))

            scores_list.append(scores)

        if is_batch:
            return scores_list, tops_list
        return scores_list[0], tops_list[0]


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
