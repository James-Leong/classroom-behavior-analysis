from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DebugTraceEntry:
    frame_idx: int
    student_name: str
    raw_scores: Dict[str, float]
    ema_scores: Dict[str, float]
    chosen_label: str
    gating_info: Dict[str, Any]
    crop_bbox: List[int]  # [x1, y1, x2, y2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame_idx,
            "name": self.student_name,
            "raw_scores": self.raw_scores,
            "ema_scores": self.ema_scores,
            "label": self.chosen_label,
            "gating": self.gating_info,
            "bbox": self.crop_bbox,
        }


class DebugTraceCollector:
    def __init__(self):
        self.entries: List[DebugTraceEntry] = []

    def add_entry(
        self,
        frame_idx: int,
        student_name: str,
        raw_scores: Dict[str, float],
        ema_scores: Dict[str, float],
        chosen_label: str,
        gating_info: Dict[str, Any],
        crop_bbox: List[int],
    ):
        self.entries.append(
            DebugTraceEntry(
                frame_idx=frame_idx,
                student_name=student_name,
                raw_scores=raw_scores,
                ema_scores=ema_scores,
                chosen_label=chosen_label,
                gating_info=gating_info,
                crop_bbox=crop_bbox,
            )
        )

    def save_json(self, path: str):
        data = [e.to_dict() for e in self.entries]
        # Sort by frame index then name
        data.sort(key=lambda x: (x["frame"], x["name"]))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
