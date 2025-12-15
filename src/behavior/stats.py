from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))


@dataclass
class BehaviorSeriesConfig:
    th_on: float = 0.60
    th_off: float = 0.45
    min_duration_seconds: float = 0.50
    merge_gap_seconds: float = 0.30


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _merge_segments(segments: List[Segment], merge_gap_seconds: float) -> List[Segment]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (s.start_time, s.end_time))
    out: List[Segment] = [segs[0]]
    gap = float(merge_gap_seconds)
    for s in segs[1:]:
        last = out[-1]
        if float(s.start_time) <= float(last.end_time) + gap:
            out[-1] = Segment(
                start_frame=last.start_frame,
                end_frame=max(last.end_frame, s.end_frame),
                start_time=min(last.start_time, s.start_time),
                end_time=max(last.end_time, s.end_time),
            )
        else:
            out.append(s)
    return out


def build_segments_from_scores(
    frames: List[Dict],
    scores_by_frame: Mapping[int, float],
    observed_frame_set: Optional[set[int]] = None,
    cfg: BehaviorSeriesConfig = BehaviorSeriesConfig(),
) -> List[Segment]:
    """Convert per-frame scores into merged time segments using hysteresis.

    Args:
        frames: result["frames"] list. Each has frame, timestamp.
        scores_by_frame: frame_index -> score for a single behavior label.
        observed_frame_set: if provided, only frames in this set are considered observed.

    Returns:
        List[Segment] in seconds.
    """

    th_on = float(cfg.th_on)
    th_off = float(cfg.th_off)
    if th_off > th_on:
        th_off = th_on

    active = False
    seg_start: Optional[Tuple[int, float]] = None
    raw: List[Segment] = []

    # Iterate frames in timestamp order.
    frames_sorted = sorted(frames, key=lambda fr: int(fr.get("frame", 0)))

    for i, fr in enumerate(frames_sorted):
        fidx = int(fr.get("frame", 0))
        ts = float(fr.get("timestamp", 0.0) or 0.0)

        if observed_frame_set is not None and fidx not in observed_frame_set:
            # Treat as unobserved: close any active segment.
            if active and seg_start is not None:
                raw.append(
                    Segment(
                        start_frame=seg_start[0],
                        end_frame=fidx,
                        start_time=seg_start[1],
                        end_time=ts,
                    )
                )
                active = False
                seg_start = None
            continue

        score = float(scores_by_frame.get(fidx, 0.0) or 0.0)
        score = _clamp01(score)

        if not active:
            if score >= th_on:
                active = True
                seg_start = (fidx, ts)
        else:
            if score < th_off:
                # Close segment at current ts.
                if seg_start is not None:
                    raw.append(
                        Segment(
                            start_frame=seg_start[0],
                            end_frame=fidx,
                            start_time=seg_start[1],
                            end_time=ts,
                        )
                    )
                active = False
                seg_start = None

    # Close at end.
    if active and seg_start is not None:
        # End time: use last timestamp + dt if possible.
        last = frames_sorted[-1]
        last_ts = float(last.get("timestamp", 0.0) or 0.0)
        dt = 0.0
        if len(frames_sorted) >= 2:
            prev_ts = float(frames_sorted[-2].get("timestamp", 0.0) or 0.0)
            dt = max(0.0, last_ts - prev_ts)
        raw.append(
            Segment(
                start_frame=seg_start[0],
                end_frame=int(last.get("frame", 0)),
                start_time=seg_start[1],
                end_time=last_ts + dt,
            )
        )

    # Filter by min duration.
    min_d = float(cfg.min_duration_seconds)
    filtered = [s for s in raw if s.duration + 1e-9 >= min_d]

    return _merge_segments(filtered, merge_gap_seconds=float(cfg.merge_gap_seconds))


def compute_on_screen_seconds(frames: List[Dict], observed_frame_set: set[int]) -> float:
    """Compute observed seconds by summing timestamp deltas for observed frames."""

    if not frames or not observed_frame_set:
        return 0.0

    frames_sorted = sorted(frames, key=lambda fr: int(fr.get("frame", 0)))
    total = 0.0

    for i in range(len(frames_sorted)):
        f0 = int(frames_sorted[i].get("frame", 0))
        if f0 not in observed_frame_set:
            continue

        t0 = float(frames_sorted[i].get("timestamp", 0.0) or 0.0)

        if i < len(frames_sorted) - 1:
            t1 = float(frames_sorted[i + 1].get("timestamp", 0.0) or 0.0)
            dt = max(0.0, t1 - t0)
        else:
            # Estimate tail duration using the previous interval.
            if len(frames_sorted) >= 2:
                t_prev = float(frames_sorted[i - 1].get("timestamp", 0.0) or 0.0)
                dt = max(0.0, t0 - t_prev)
            else:
                dt = 0.0

        total += float(dt)

    return float(total)
