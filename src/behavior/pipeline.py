from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    clip_temperature: float = 40.0

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

    # Temporal smoothing / uncertainty gating (primarily for CLIP)
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.80  # EMA: higher -> smoother
    uncertain_min_prob: float = 0.0  # 0 disables prob gating
    uncertain_min_margin: float = 0.0  # 0 disables margin gating
    uncertain_fallback_label: str = "other"  # used when gating triggers and label exists

    # Make it harder to enter distracted (reduce false positives)
    distracted_enter_min_prob: float = 0.0  # 0 disables
    distracted_enter_min_margin: float = 0.25  # 0 disables

    # Make it harder to stay distracted (avoid sticky false positives)
    distracted_stay_min_prob: float = 0.0  # 0 disables
    distracted_stay_min_margin: float = 0.20  # 0 disables

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
            temperature=float(cfg.clip_temperature),
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

    # Per-student EMA state for smoothing label decisions.
    # name -> label -> ema_score
    ema_state: Dict[str, Dict[str, float]] = {}

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

    # Iterate sampled frames in batches to optimize I/O and GPU usage.
    sorted_keys = sorted(selected.keys())

    # Configurable batch size (in seconds of video time)
    # 10s window balances memory usage vs I/O efficiency
    BATCH_WINDOW_SECONDS = 10.0

    batches = []
    if sorted_keys:
        current_batch = []
        batch_start_t = sorted_keys[0] / fps
        for fidx in sorted_keys:
            t = fidx / fps
            if t - batch_start_t > BATCH_WINDOW_SECONDS:
                batches.append(current_batch)
                current_batch = []
                batch_start_t = t
            current_batch.append(fidx)
        if current_batch:
            batches.append(current_batch)

    logger.info(
        f"behavior: processing {len(sorted_keys)} keyframes in {len(batches)} batches (window={BATCH_WINDOW_SECONDS}s)"
    )

    for batch_idx, batch_keys in enumerate(batches):
        # 1. Determine frame range needed for this batch
        batch_needed_indices = set()
        keyframe_indices_map = {}  # fidx -> indices

        for fidx in batch_keys:
            indices = sample_frame_indices(
                center_frame=fidx,
                fps=fps,
                window_seconds=cfg.clip_seconds,
                num_frames=cfg.clip_num_frames,
                max_frame=max_frame_index,
            )
            keyframe_indices_map[fidx] = indices
            batch_needed_indices.update(indices)

        if not batch_needed_indices:
            continue

        min_frame = min(batch_needed_indices)
        max_frame = max(batch_needed_indices)
        num_frames_to_read = max_frame - min_frame + 1

        # 2. Read frames into memory buffer
        buffer_frames = {}  # absolute_idx -> frame

        try:
            if use_ffmpeg and ffmpeg_reader:
                # Read contiguous chunk using ffmpeg seeking
                chunk = ffmpeg_reader.read_frame_range(min_frame, num_frames_to_read)
                for i, frame in enumerate(chunk):
                    buffer_frames[min_frame + i] = frame
            else:
                # Fallback OpenCV: seek and read range
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(min_frame))
                for i in range(num_frames_to_read):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    buffer_frames[min_frame + i] = frame
        except Exception as e:
            logger.warning(f"Batch read failed at batch {batch_idx}: {e}")
            continue

        # 3. Prepare clips for batch inference
        batch_clips = []
        batch_meta = []  # (name, fidx)

        for fidx in batch_keys:
            indices = keyframe_indices_map[fidx]
            names_to_det = selected[fidx]

            # Ensure we have the keyframe for detection/cropping
            if fidx not in buffer_frames:
                continue
            key_bgr = buffer_frames[fidx]

            # Gather clip frames
            clip_bgr = []
            for idx in indices:
                if idx in buffer_frames:
                    clip_bgr.append(buffer_frames[idx])
                else:
                    # Pad with keyframe if missing (e.g. EOF)
                    clip_bgr.append(key_bgr)

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
                    # Expand to roughly 2x width and 3x height to cover upper body
                    bw = fw * 2.0
                    bh = fh * 3.0
                    bx1 = max(0, int(cx - bw / 2))
                    by1 = max(0, int(y1 - fh * 0.2))  # Start slightly above face
                    bx2 = min(w, int(cx + bw / 2))
                    by2 = min(h, int(by1 + bh))
                    crop_bbox = [bx1, by1, bx2, by2]

                observed_frames.setdefault(name, set()).add(int(fidx))
                per_student_scores.setdefault(name, {})

                try:
                    clip_rgb = crop_clip(clip_bgr, crop_bbox)
                    batch_clips.append(clip_rgb)
                    batch_meta.append((name, fidx))
                except Exception as e:
                    logger.debug(f"clip crop failed for {name} at frame {fidx}: {e}")
                    continue

        # 4. Run batch inference
        if batch_clips:
            try:
                # Split into mini-batches to avoid OOM on GPU
                # Each clip has ~16 frames. 64 clips ~= 1024 frames.
                # 1024 frames * 224*224 * 3 * 4 bytes ~= 600MB input tensor.
                # Plus activations, this is safe for most GPUs (4GB+).
                MINI_BATCH_SIZE = 64

                all_scores = []

                for i in range(0, len(batch_clips), MINI_BATCH_SIZE):
                    chunk = batch_clips[i : i + MINI_BATCH_SIZE]
                    try:
                        # Batch inference on chunk
                        scores_list, tops = action_model.predict_proba(chunk, topk=3)

                        # Normalize batch return shape
                        if isinstance(scores_list, dict):
                            scores_list = [scores_list]

                        all_scores.extend(scores_list)

                    except Exception as e:
                        logger.warning(f"behavior: mini-batch inference failed for batch {batch_idx} chunk {i}: {e}")
                        # Fill with empty results to maintain alignment with batch_meta
                        all_scores.extend([{} for _ in range(len(chunk))])

                # Match results with student names
                alpha = float(cfg.smoothing_alpha)
                if alpha < 0.0:
                    alpha = 0.0
                if alpha > 0.99:
                    alpha = 0.99

                do_smooth = bool(cfg.enable_smoothing) and isinstance(action_model, CLIPVideoActionModel)
                min_prob = float(cfg.uncertain_min_prob)
                min_margin = float(cfg.uncertain_min_margin)
                fallback_label = str(cfg.uncertain_fallback_label or "").strip()
                distracted_enter_min_prob = float(cfg.distracted_enter_min_prob)
                distracted_enter_min_margin = float(cfg.distracted_enter_min_margin)
                distracted_stay_min_prob = float(cfg.distracted_stay_min_prob)
                distracted_stay_min_margin = float(cfg.distracted_stay_min_margin)

                def _pick_conservative_label(score_map: Dict[str, float]) -> str:
                    avoid = {"distracted", "other"}
                    items2 = sorted(score_map.items(), key=lambda kv: float(kv[1]), reverse=True)
                    for lbl2, _v2 in items2:
                        if str(lbl2) not in avoid:
                            return str(lbl2)
                    return str(items2[0][0]) if items2 else ""

                for scores, (name, fidx) in zip(all_scores, batch_meta):
                    if not isinstance(scores, dict) or not scores:
                        continue

                    # Convert to dense float map
                    cur_scores: Dict[str, float] = {str(k): float(v) for k, v in scores.items()}

                    # Uncertainty gating based on current scores
                    try:
                        items = sorted(cur_scores.items(), key=lambda kv: float(kv[1]), reverse=True)
                        top1_label, top1_prob = items[0]
                        top2_prob = float(items[1][1]) if len(items) >= 2 else 0.0
                        margin = float(top1_prob) - float(top2_prob)
                    except Exception:
                        top1_label, top1_prob, margin = "", 0.0, 0.0

                    gated = False
                    if min_prob > 0.0 and float(top1_prob) < min_prob:
                        gated = True
                    if min_margin > 0.0 and float(margin) < min_margin:
                        gated = True

                    # EMA smoothing (CLIP only)
                    if do_smooth:
                        st = ema_state.setdefault(name, {})

                        had_history = bool(st)
                        prev_lbl = max(st.items(), key=lambda kv: float(kv[1]))[0] if had_history else ""
                        if (
                            str(top1_label) == "distracted"
                            and str(prev_lbl) != "distracted"
                            and (distracted_enter_min_prob > 0.0 or distracted_enter_min_margin > 0.0)
                        ):
                            if distracted_enter_min_prob > 0.0 and float(top1_prob) < distracted_enter_min_prob:
                                gated = True
                            if distracted_enter_min_margin > 0.0 and float(margin) < distracted_enter_min_margin:
                                gated = True

                        # Always update EMA state
                        for lbl, v in cur_scores.items():
                            prev = float(st.get(lbl, v))
                            st[lbl] = prev * alpha + float(v) * (1.0 - alpha)
                        for lbl in list(st.keys()):
                            if lbl not in cur_scores:
                                st.pop(lbl, None)

                        if gated:
                            if had_history and prev_lbl:
                                chosen_label = str(prev_lbl)
                            else:
                                chosen_label = _pick_conservative_label(cur_scores)
                        else:
                            chosen_label = max(st.items(), key=lambda kv: float(kv[1]))[0] if st else top1_label

                        if str(prev_lbl) == "distracted" and (
                            distracted_stay_min_prob > 0.0 or distracted_stay_min_margin > 0.0
                        ):
                            enough = True
                            if str(top1_label) != "distracted":
                                enough = False
                            if distracted_stay_min_prob > 0.0 and float(top1_prob) < distracted_stay_min_prob:
                                enough = False
                            if distracted_stay_min_margin > 0.0 and float(margin) < distracted_stay_min_margin:
                                enough = False
                            if not enough:
                                chosen_label = _pick_conservative_label(cur_scores)
                    else:
                        if gated and fallback_label and fallback_label in cur_scores:
                            chosen_label = fallback_label
                        else:
                            chosen_label = top1_label

                    if chosen_label:
                        per_student_scores.setdefault(name, {}).setdefault(str(chosen_label), {})[int(fidx)] = 1.0
                        processed_tasks += 1

            except Exception as e:
                logger.warning(f"behavior: action inference failed for batch {batch_idx}: {e}")

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
