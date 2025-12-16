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
from src.video.ffmpeg_reader import FFmpegFrameReader

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
    clip_custom_behaviors: Optional[List[str]] = None
    clip_custom_labels: Optional[List[str]] = None

    # Device: reuse top-level --device (auto/cpu/gpu). Internally maps gpu->cuda.
    device: str = "auto"

    # Clip
    clip_seconds: float = 2.0
    clip_num_frames: int = 16

    # Person detector (optional): if None, will use body_bbox from JSON (v2 schema)
    person_detector_weights: Optional[str] = "yolo11n.pt"
    person_conf: float = 0.25

    # Segmentation
    series_cfg: BehaviorSeriesConfig = field(default_factory=BehaviorSeriesConfig)

    # Lock status filtering (v2 feature)
    ignore_lock_status: bool = False  # If True, analyze all detections regardless of lock status


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

            # Check lock status (can be bypassed with ignore_lock_status flag)
            if not cfg.ignore_lock_status:
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
        )
    else:
        logger.info(f"Using Kinetics pretrained model: {cfg.action_model_name}")
        action_model = TorchvisionVideoActionModel(model_name=cfg.action_model_name, device=dev)

    # Person detector is optional in v2: if not provided, use body_bbox from JSON
    person_detector = None
    if cfg.person_detector_weights is not None:
        try:
            person_detector = UltralyticsPersonDetector(weights_path=cfg.person_detector_weights, device=dev)
            logger.info(f"Person检测器已启用: {cfg.person_detector_weights}")
        except Exception as e:
            logger.warning(f"Person检测器初始化失败，将尝试使用JSON中的body_bbox: {e}")
            person_detector = None
    else:
        logger.info("Person检测器未启用，将使用JSON中的body_bbox（v2 schema）")

    # Use ffmpeg for faster video reading with hardware acceleration
    try:
        ffmpeg_reader = FFmpegFrameReader(str(input_video), hwaccel="auto")
        use_ffmpeg = True
        max_frame_index = ffmpeg_reader.total_frames
        logger.info(
            f"Using FFmpeg for video decoding (hwaccel={ffmpeg_reader.hwaccel or 'none'}, total_frames={max_frame_index})"
        )
    except Exception as e:
        logger.warning(f"FFmpeg reader initialization failed, falling back to OpenCV: {e}")
        ffmpeg_reader = None
        use_ffmpeg = False
        max_frame_index = None

    # Fallback to OpenCV if ffmpeg fails
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for behavior analysis: {input_video}")

    # Get total frames if not already set
    if max_frame_index is None:
        max_frame_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frame_index > 0:
            logger.info(f"Video has {max_frame_index} frames (from OpenCV)")

    per_student_scores: Dict[str, Dict[str, Dict[int, float]]] = {}

    # Progress
    total_tasks = 0
    for _, names_to_det in selected.items():
        total_tasks += int(len(names_to_det))
    processed_tasks = 0
    progress_every = max(1, min(25, total_tasks // 20 if total_tasks > 0 else 10))
    t0 = time.time()

    try:
        logger.info(f"behavior: device={dev}, clip={cfg.clip_num_frames}f/{cfg.clip_seconds:.2f}s")
    except Exception:
        pass

    # Iterate sampled frames in order.
    sorted_keys = sorted(selected.keys())

    for idx, fidx in enumerate(sorted_keys):
        names_to_det = selected[fidx]

        # Build clip indices around keyframe once per keyframe.
        frame_indices = sample_frame_indices(
            center_frame=fidx,
            fps=fps,
            window_seconds=cfg.clip_seconds,
            num_frames=cfg.clip_num_frames,
            max_frame=max_frame_index,
        )

        # Read frames using ffmpeg (faster) or fallback to OpenCV
        try:
            if use_ffmpeg and ffmpeg_reader:
                # FFmpeg can read all frames efficiently in one shot
                frames_bgr = ffmpeg_reader.read_frames(frame_indices)
                if not frames_bgr or len(frames_bgr) == 0:
                    logger.debug(f"behavior: ffmpeg returned no frames for keyframe {fidx}")
                    continue
                key_bgr = frames_bgr[len(frames_bgr) // 2]  # Use middle frame as keyframe
            else:
                # OpenCV fallback: seek + read
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fidx))
                ok, key_bgr = cap.read()
                if not ok or key_bgr is None:
                    logger.debug(f"behavior: failed to read keyframe {fidx}")
                    continue
                frames_bgr = read_frames_by_index(cap, frame_indices)
        except Exception as e:
            logger.debug(f"behavior: failed to read clip around keyframe {fidx}: {e}", exc_info=True)
            continue

        # Collect clips for all students in this keyframe
        keyframe_items = []  # [(name, crop_bbox), ...]
        keyframe_clips = []  # [clip_rgb, ...]

        for name, det in names_to_det.items():
            face_bbox = det.get("bbox")
            if not face_bbox:
                continue

            # Priority 1: Use body_bbox from JSON (v2 schema)
            crop_bbox = det.get("body_bbox")

            # Priority 2: If no body_bbox in JSON, try person detector (fallback)
            if crop_bbox is None and person_detector is not None:
                persons = []
                try:
                    persons = person_detector.detect_persons(key_bgr, conf=cfg.person_conf)
                except Exception:
                    persons = []

                if persons:
                    crop_bbox = pick_person_bbox_for_face(persons, face_bbox)

            # Priority 3: If still no body bbox, use face bbox as fallback
            if crop_bbox is None:
                # Use face bbox expanded by 2x for body estimation
                x1, y1, x2, y2 = face_bbox
                h, w = key_bgr.shape[:2]
                fw = x2 - x1
                fh = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                # Expand to roughly 2x width and 3x height to cover upper body
                bw = fw * 2.0
                bh = fh * 3.0
                bx1 = max(0, int(cx - bw / 2))
                by1 = max(0, int(y1 - fh * 0.2))  # Start slightly above face
                bx2 = min(w, int(cx + bw / 2))
                by2 = min(h, int(by1 + bh))
                crop_bbox = [bx1, by1, bx2, by2]

            observed_frames.setdefault(name, set()).add(int(fidx))

            # Ensure the student exists in output even if inference fails later.
            per_student_scores.setdefault(name, {})

            try:
                clip_rgb = crop_clip(frames_bgr, crop_bbox)
                keyframe_items.append((name, crop_bbox))
                keyframe_clips.append(clip_rgb)
            except Exception as e:
                logger.debug(f"clip crop failed for {name} at frame {fidx}: {e}")
                continue

        # Run action inference for this keyframe's clips
        if keyframe_clips:
            try:
                # Batch inference on all clips from this keyframe
                _scores, tops = action_model.predict_proba(keyframe_clips, topk=1)

                # tops should be a list of ActionTopK
                if not isinstance(tops, list):
                    tops = [tops]

                # Match results with student names
                for top, (name, _bbox) in zip(tops, keyframe_items):
                    if getattr(top, "categories", None) and len(top.categories) > 0:
                        label, _prob = top.categories[0]
                        # Use a hard assignment score=1.0 for the selected label
                        per_student_scores.setdefault(name, {}).setdefault(str(label), {})[int(fidx)] = 1.0
                        processed_tasks += 1

            except Exception as e:
                logger.warning(f"behavior: action inference failed for keyframe {fidx}: {e}")

        # Progress log
        try:
            if processed_tasks % progress_every == 0 or processed_tasks == total_tasks:
                elapsed = max(1e-6, time.time() - t0)
                rate = float(processed_tasks) / elapsed
                eta = float(total_tasks - processed_tasks) / rate if rate > 1e-9 else None

                logger.info(
                    f"behavior进度: {processed_tasks}/{total_tasks} ({processed_tasks * 100.0 / total_tasks:.1f}%), "
                    f"{rate:.2f} task/s" + (f", ETA: {eta:.1f}s" if eta else "")
                )
        except Exception:
            pass

    # Cleanup
    if ffmpeg_reader:
        ffmpeg_reader.close()
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
