from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PersonDet:
    bbox: List[int]  # xyxy
    conf: float


class UltralyticsPersonDetector:
    """Person detector using ultralytics YOLO.

    This is used to get a more stable upper-body crop for action inference.
    """

    def __init__(self, weights_path: str = "yolo11n.pt", device: str = "auto") -> None:
        from ultralytics import YOLO
        import torch

        self.model = YOLO(str(weights_path))
        dev = str(device).lower().strip()
        if dev == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

    def detect_persons(self, frame_bgr: np.ndarray, conf: float = 0.25) -> List[PersonDet]:
        # COCO person class id is 0 for YOLO models.
        results = self.model.predict(frame_bgr, conf=float(conf), device=self.device, verbose=False)
        if not results:
            return []
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        out: List[PersonDet] = []
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
        except Exception:
            return []

        for b, c, s in zip(xyxy, cls, confs):
            if int(c) != 0:
                continue
            out.append(PersonDet(bbox=[int(x) for x in b.tolist()], conf=float(s)))

        return out


def pick_person_bbox_for_face(
    persons: List[PersonDet],
    face_bbox: List[int],
) -> Optional[List[int]]:
    x1, y1, x2, y2 = [float(x) for x in face_bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    def _point_rect_dist2(px1: float, py1: float, px2: float, py2: float) -> float:
        dx = 0.0
        if cx < px1:
            dx = px1 - cx
        elif cx > px2:
            dx = cx - px2
        dy = 0.0
        if cy < py1:
            dy = py1 - cy
        elif cy > py2:
            dy = cy - py2
        return dx * dx + dy * dy

    best_bbox: Optional[List[int]] = None
    best_d2: Optional[float] = None
    best_area: Optional[float] = None

    for p in persons:
        px1, py1, px2, py2 = [float(x) for x in p.bbox]
        # Prefer reasonable horizontal overlap (avoid matching to far-away person).
        if cx < px1 - 0.25 * (px2 - px1) or cx > px2 + 0.25 * (px2 - px1):
            continue
        d2 = _point_rect_dist2(px1, py1, px2, py2)
        area = max(1.0, (px2 - px1) * (py2 - py1))
        if best_d2 is None or d2 < best_d2 or (d2 == best_d2 and area < float(best_area or area)):
            best_bbox = p.bbox
            best_d2 = float(d2)
            best_area = float(area)

    if best_bbox is None or best_d2 is None:
        return None

    # Reject if too far from the chosen bbox (normalized by bbox size).
    px1, py1, px2, py2 = [float(x) for x in best_bbox]
    diag2 = max(1.0, (px2 - px1) ** 2 + (py2 - py1) ** 2)
    if float(best_d2) / diag2 > 0.18:
        return None

    return best_bbox
