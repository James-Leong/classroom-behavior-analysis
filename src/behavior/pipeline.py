from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2

from src.behavior.action_model import TorchvisionVideoActionModel
from src.behavior.action_model_clip import CLIPVideoActionModel
from src.behavior.person_detector import UltralyticsPersonDetector, pick_person_bbox_for_face
from src.behavior.report import build_behavior_stats
from src.behavior.stats import BehaviorSeriesConfig
from src.behavior.video_clip import crop_clip, read_frames_by_index, sample_frame_indices
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class BehaviorPipelineConfig:
    enabled: bool = False
    target_names: Optional[List[str]] = None  # if None -> all names

    # Model selection
    model_type: str = "kinetics"  # "kinetics" or "clip"

    # Kinetics model config (when model_type="kinetics")
    action_model_name: str = "swin3d_t"

    # CLIP model config (when model_type="clip")
    clip_model_name: str = "ViT-B/32"  # ViT-B/32, ViT-B/16, ViT-L/14
    clip_frame_subsample: int = 4  # Process every Nth frame for speed
    clip_custom_behaviors: Optional[List[str]] = None
    clip_custom_labels: Optional[List[str]] = None

    # Device: reuse top-level --device (auto/cpu/gpu). Internally maps gpu->cuda.
    device: str = "auto"

    # Batch cap: can reuse top-level --batch-frames as an upper bound.
    batch_size_cap: int = 1

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


def _map_device_name(dev: str) -> str:
    s = str(dev).strip().lower()
    if s in {"gpu", "cuda"}:
        return "cuda"
    if s in {"cpu"}:
        return "cpu"
    if s in {"auto", ""}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


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
    dev = _map_device_name(cfg.device)

    # Select action model based on config
    if cfg.model_type.lower() == "clip":
        logger.info(f"Using CLIP zero-shot model: {cfg.clip_model_name}")
        action_model = CLIPVideoActionModel(
            model_name=cfg.clip_model_name,
            device=dev,
            custom_behaviors=cfg.clip_custom_behaviors,
            custom_labels=cfg.clip_custom_labels,
            frame_subsample=cfg.clip_frame_subsample,
        )
    else:
        logger.info(f"Using Kinetics pretrained model: {cfg.action_model_name}")
        action_model = TorchvisionVideoActionModel(model_name=cfg.action_model_name, device=dev)

    try:
        person_detector = UltralyticsPersonDetector(weights_path=cfg.person_detector_weights, device=dev)
    except Exception as e:
        raise RuntimeError("behavior: failed to init required person detector") from e

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for behavior analysis: {input_video}")

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    per_student_scores: Dict[str, Dict[str, Dict[int, float]]] = {}

    # Progress
    total_tasks = 0
    for _, names_to_det in selected.items():
        total_tasks += int(len(names_to_det))
    processed_tasks = 0
    progress_every = max(1, min(25, total_tasks // 20 if total_tasks > 0 else 10))
    t0 = time.time()

    # Batch settings (reuse top-level batch cap but clamp to a safe range)
    batch_cap = max(1, int(cfg.batch_size_cap) if int(cfg.batch_size_cap) > 0 else 1)
    behavior_batch = max(1, min(batch_cap, 8 if dev == "cuda" else 4))

    try:
        logger.info(
            f"behavior: device={dev}, batch_cap={batch_cap}, behavior_batch={behavior_batch}, clip={cfg.clip_num_frames}f/{cfg.clip_seconds:.2f}s"
        )
    except Exception:
        pass

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

        # Build clip indices around keyframe once per keyframe.
        frame_indices = sample_frame_indices(
            center_frame=fidx,
            fps=fps,
            window_seconds=cfg.clip_seconds,
            num_frames=cfg.clip_num_frames,
        )

        # Read clip frames once per keyframe (major perf win vs per-student reads).
        try:
            frames_bgr = read_frames_by_index(cap, frame_indices)
        except Exception:
            logger.debug(f"behavior: failed to read clip around keyframe {fidx}", exc_info=True)
            continue

        student_items: List[Tuple[str, List[int]]] = []
        student_clips: List = []

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

            # Ensure the student exists in output even if inference fails later.
            per_student_scores.setdefault(name, {})

            try:
                clip_rgb = crop_clip(frames_bgr, crop_bbox)
            except Exception:
                logger.debug("behavior: clip crop failed", exc_info=True)
                continue

            student_items.append((name, crop_bbox))
            student_clips.append(clip_rgb)

        # Run action inference for this keyframe.
        # No custom classroom-behavior mapping is applied: we directly use the
        # pretrained category name as the behavior label.
        if student_clips:
            try:
                start = 0
                while start < len(student_clips):
                    end = min(len(student_clips), start + int(behavior_batch))
                    batch = student_clips[start:end]
                    _scores, tops = action_model.predict_proba(batch, topk=1)
                    # batch call returns list[ActionTopK]
                    for top, (name, _bbox) in zip(tops, student_items[start:end]):
                        if getattr(top, "categories", None):
                            label, _prob = top.categories[0]
                            # Use a hard assignment score=1.0 for the selected label, so
                            # segmentation reflects the model's predicted label sequence.
                            per_student_scores.setdefault(name, {}).setdefault(str(label), {})[int(fidx)] = 1.0
                            processed_tasks += 1
                    start = end
            except Exception:
                logger.debug("behavior: action inference failed", exc_info=True)

        # Progress log (behavior stage).
        try:
            if total_tasks > 0 and (processed_tasks % progress_every == 0 or processed_tasks == total_tasks):
                elapsed = max(1e-6, time.time() - t0)
                rate = float(processed_tasks) / elapsed
                eta = float(total_tasks - processed_tasks) / rate if rate > 1e-9 else None
                if eta is not None:
                    logger.info(
                        f"behavior进度: {processed_tasks}/{total_tasks} tasks ({processed_tasks * 100.0 / total_tasks:.1f}%), {rate:.2f} task/s, ETA: {eta:.1f}s"
                    )
                else:
                    logger.info(f"behavior进度: {processed_tasks}/{total_tasks} tasks, {rate:.2f} task/s")
        except Exception:
            pass

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
