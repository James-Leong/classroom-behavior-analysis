from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Set, Tuple

from .stats import BehaviorSeriesConfig, Segment, build_segments_from_scores, compute_on_screen_seconds


def build_behavior_stats(
    *,
    frames: List[Dict],
    per_student_per_behavior_scores: Mapping[str, Mapping[str, Mapping[int, float]]],
    per_student_observed_frames: Mapping[str, Set[int]],
    cfg: Optional[BehaviorSeriesConfig] = None,
    fps: float,
    used_frame_interval: int,
    video_seconds: Optional[float] = None,
) -> Dict:
    """Build `behavior_stats` JSON block.

    Args:
        per_student_per_behavior_scores: student -> behavior -> {frame -> score}
        per_student_observed_frames: student -> set(frame) observed/attributable
        cfg: segmentation config
        fps: video fps
        used_frame_interval: sampling interval used in frames[]
        video_seconds: optional full video length
    """

    if cfg is None:
        cfg = BehaviorSeriesConfig()

    denom_type = "on_screen_seconds"

    per_student_on_screen: Dict[str, float] = {}
    for student, obs in per_student_observed_frames.items():
        per_student_on_screen[student] = compute_on_screen_seconds(frames, set(obs))

    by_student: Dict[str, Dict] = {}

    for student, behavior_scores in per_student_per_behavior_scores.items():
        obs = set(per_student_observed_frames.get(student, set()))
        denom_seconds = float(per_student_on_screen.get(student, 0.0) or 0.0)

        behaviors_out: Dict[str, Dict] = {}
        for behavior, scores_by_frame in behavior_scores.items():
            segs = build_segments_from_scores(frames, scores_by_frame, observed_frame_set=obs, cfg=cfg)
            total = sum(s.duration for s in segs)
            ratio = (total / denom_seconds) if denom_seconds > 1e-9 else 0.0
            behaviors_out[behavior] = {
                "segments": [
                    {
                        "start_frame": int(s.start_frame),
                        "end_frame": int(s.end_frame),
                        "start_time": float(s.start_time),
                        "end_time": float(s.end_time),
                    }
                    for s in segs
                ],
                "total_seconds": float(total),
                "ratio": float(ratio),
            }

        by_student[student] = {
            "total_observed_seconds": float(denom_seconds),
            "behaviors": behaviors_out,
            "overlap_policy": "multi_label",
        }

    timebase = {
        "fps": float(fps),
        "used_frame_interval": int(used_frame_interval),
        "sample_dt_seconds": (float(used_frame_interval) / float(fps)) if fps > 0 else None,
        "timestamp_rule": "t = frame / fps",
    }

    denom = {
        "type": denom_type,
        "video_seconds": float(video_seconds) if video_seconds is not None else None,
        "per_student_on_screen_seconds": {k: float(v) for k, v in per_student_on_screen.items()},
    }

    return {
        "behavior_schema_version": "v1",
        "identity_key": "track_display_identity",
        "timebase": timebase,
        "denominator": denom,
        "by_student": by_student,
    }
