from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ClipSpec:
    center_frame: int
    window_seconds: float = 2.0
    num_frames: int = 16


def _clip_bbox_xyxy(bbox: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(x) for x in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def expand_bbox_xyxy(
    bbox: List[int],
    width: int,
    height: int,
    *,
    scale_x: float = 2.0,
    scale_up: float = 1.0,
    scale_down: float = 3.0,
) -> List[int]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)

    new_w = w * float(scale_x)
    new_up = h * float(scale_up)
    new_down = h * float(scale_down)

    nx1 = int(round(cx - new_w * 0.5))
    nx2 = int(round(cx + new_w * 0.5))
    ny1 = int(round(cy - new_up))
    ny2 = int(round(cy + new_down))

    return _clip_bbox_xyxy([nx1, ny1, nx2, ny2], width, height)


def sample_frame_indices(center_frame: int, fps: float, window_seconds: float, num_frames: int) -> List[int]:
    if fps <= 0:
        fps = 25.0
    half = float(window_seconds) * 0.5
    start_t = (float(center_frame) / fps) - half
    end_t = (float(center_frame) / fps) + half
    if num_frames <= 1:
        return [int(center_frame)]

    out: List[int] = []
    for i in range(int(num_frames)):
        a = i / (num_frames - 1)
        t = start_t * (1.0 - a) + end_t * a
        fi = int(round(t * fps))
        if fi < 0:
            fi = 0
        out.append(fi)
    # Ensure monotonic non-decreasing indices
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def read_frames_by_index(cap: cv2.VideoCapture, frame_indices: List[int]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    last_pos = -1
    for fi in frame_indices:
        if fi != last_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            # Pad with last available frame if possible
            if frames:
                frames.append(frames[-1].copy())
                last_pos = fi
                continue
            raise RuntimeError(f"failed to read frame {fi}")
        frames.append(frame)
        last_pos = fi + 1
    return frames


def crop_clip(frames_bgr: List[np.ndarray], bbox_xyxy: List[int]) -> np.ndarray:
    if not frames_bgr:
        raise ValueError("empty frames")

    h, w = frames_bgr[0].shape[:2]
    x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, w, h)

    out_rgb: List[np.ndarray] = []
    for fr in frames_bgr:
        crop = fr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out_rgb.append(rgb)

    # (T,H,W,3)
    return np.stack(out_rgb, axis=0)
