from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2

from .action_model import TorchvisionVideoActionModel
from .behavior_mapping import build_regex_mapping, default_behavior_labels
from .person_detector import UltralyticsPersonDetector, pick_person_bbox_for_face
from .report import build_behavior_stats
from .stats import BehaviorSeriesConfig
from .video_clip import crop_clip, read_frames_by_index, sample_frame_indices
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class BehaviorPipelineConfig:
    enabled: bool = False
    target_names: Optional[List[str]] = None  # if None -> all names

    # Action model
    action_model_name: str = "swin3d_t"
    action_topk: int = 10

    # Clip
    clip_seconds: float = 2.0
    clip_num_frames: int = 16

    # Person detector (required): used to get body bbox
    person_detector_weights: str = "yolo11n.pt"
    person_conf: float = 0.25

    # Segmentation
    series_cfg: BehaviorSeriesConfig = field(default_factory=BehaviorSeriesConfig)


def _normalize_target_names(names: Optional[List[str]]) -> Optional[Set[str]]:
    if not names:
        return None
    out = set()
    for n in names:
        s = str(n).strip()
        if s:
            out.add(s)
    return out or None


def run_behavior_pipeline_on_result(
    *,
    input_video: str,
    result: Dict,
    cfg: BehaviorPipelineConfig,
) -> Optional[Dict]:
    """Run action model inference and build per-student behavior statistics.

    This function reads frames from the input video as needed. It uses the already
    computed face detections + stable identities in `result['frames']`.

    Returns:
        behavior_stats dict, or None if disabled.
    """

    if not cfg.enabled:
        return None

    frames = list(result.get("frames") or [])
    if not frames:
        return None

    fps = float(result.get("fps", 25.0) or 25.0)
    used_frame_interval = int(result.get("used_frame_interval", 1) or 1)

    target = _normalize_target_names(cfg.target_names)

    # Select per-frame best face detection per student (locked only). We will only count
    # a frame as observable if we can associate that face to a person bbox.
    selected: Dict[int, Dict[str, Dict]] = {}  # frame -> name -> det
    observed_frames: Dict[str, Set[int]] = {}

    for fr in frames:
        fidx = int(fr.get("frame", 0))
        best_for_name: Dict[str, Dict] = {}
        for det in fr.get("detections", []) or []:
            name = det.get("track_display_identity") or det.get("identity") or "未知"
            if not name or name == "未知":
                continue
            if target is not None and name not in target:
                continue
            if not bool(det.get("track_is_locked", False)):
                continue

            score = float(det.get("track_display_similarity", 0.0) or 0.0)
            q = float(det.get("quality", 0.0) or 0.0)
            key = score + 0.2 * q
            cur = best_for_name.get(name)
            if cur is None or key > float(cur.get("_rank_key", -1e9)):
                d2 = dict(det)
                d2["_rank_key"] = key
                best_for_name[name] = d2

        if best_for_name:
            selected[fidx] = best_for_name

    if not selected:
        logger.info("behavior: no eligible (locked) targets observed")
        return build_behavior_stats(
            frames=frames,
            per_student_per_behavior_scores={},
            per_student_observed_frames={},
            fps=fps,
            used_frame_interval=used_frame_interval,
            video_seconds=None,
            cfg=cfg.series_cfg,
        )

    # Init models.
    action_model = TorchvisionVideoActionModel(model_name=cfg.action_model_name)

    categories = list(action_model.categories)
    idx_mapping = build_regex_mapping(categories)

    # Convert idx mapping -> name mapping for aggregation.
    behavior_to_catnames: Dict[str, List[str]] = {}
    for beh, idxs in idx_mapping.items():
        behavior_to_catnames[beh] = [categories[i] for i in idxs if 0 <= i < len(categories)]

    labels = sorted(set(behavior_to_catnames.keys()))
    if not labels:
        labels = default_behavior_labels()

    try:
        person_detector = UltralyticsPersonDetector(weights_path=cfg.person_detector_weights)
    except Exception as e:
        raise RuntimeError("behavior: failed to init required person detector") from e

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for behavior analysis: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    per_student_scores: Dict[str, Dict[str, Dict[int, float]]] = {}

    # Iterate sampled frames in order.
    for fidx in sorted(selected.keys()):
        names_to_det = selected[fidx]

        # Read keyframe first (for person detection / crop selection)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fidx))
        ok, key_bgr = cap.read()
        if not ok or key_bgr is None:
            logger.debug(f"behavior: failed to read keyframe {fidx}")
            continue

        persons = []
        try:
            persons = person_detector.detect_persons(key_bgr, conf=cfg.person_conf)
        except Exception:
            persons = []

        for name, det in names_to_det.items():
            face_bbox = det.get("bbox")
            if not face_bbox:
                continue

            # Choose body bbox if available.
            if not persons:
                # Cannot attribute behavior without a person bbox.
                continue

            crop_bbox = pick_person_bbox_for_face(persons, face_bbox)
            if crop_bbox is None:
                continue

            observed_frames.setdefault(name, set()).add(int(fidx))

            # Build clip indices around keyframe.
            frame_indices = sample_frame_indices(
                center_frame=fidx,
                fps=fps,
                window_seconds=cfg.clip_seconds,
                num_frames=cfg.clip_num_frames,
            )

            try:
                frames_bgr = read_frames_by_index(cap, frame_indices)
                clip_rgb = crop_clip(frames_bgr, crop_bbox)
                cat_scores, top = action_model.predict_proba(clip_rgb, topk=cfg.action_topk)
            except Exception:
                logger.debug("behavior: action inference failed", exc_info=True)
                continue

            # Aggregate into behavior scores by max over mapped categories.
            beh_scores: Dict[str, float] = {b: 0.0 for b in labels}
            for beh, cat_names in behavior_to_catnames.items():
                best = 0.0
                for cn in cat_names:
                    s = float(cat_scores.get(cn, 0.0) or 0.0)
                    if s > best:
                        best = s
                beh_scores[beh] = best

            for beh, s in beh_scores.items():
                per_student_scores.setdefault(name, {}).setdefault(beh, {})[int(fidx)] = float(s)

    cap.release()

    # Compute video duration (optional)
    video_seconds = None
    try:
        if fps > 0:
            last_frame = max(int(fr.get("frame", 0)) for fr in frames)
            video_seconds = float(last_frame) / float(fps)
    except Exception:
        video_seconds = None

    return build_behavior_stats(
        frames=frames,
        per_student_per_behavior_scores=per_student_scores,
        per_student_observed_frames=observed_frames,
        cfg=cfg.series_cfg,
        fps=fps,
        used_frame_interval=used_frame_interval,
        video_seconds=video_seconds,
    )
