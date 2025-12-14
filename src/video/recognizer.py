from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.face.recognizer import FaceRecognizer
from src.utils.draw import draw_texts_cn, measure_text_cn
from src.utils.log import get_logger
from src.utils.serializer import serialize_detection, serialize_tracklet
from src.video.ffmpeg_pipe import FFmpegPipeWriter
from src.video.tracker import SimpleTracker

logger = get_logger(__name__)


SCHEMA_VERSION = "v1"


def _format_ts(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class VideoFaceRecognizer:
    """视频人脸识别：支持简单模式与轨迹模式，统一输出结构。"""

    def __init__(
        self,
        gallery_path: str = "data/id_photo",
        debug_identify: bool = False,
        *,
        det_size: int = 640,
        device: str = "auto",
        rebuild_gallery: bool = False,
        iface_use_tiling: bool = True,
    ):
        self.recognizer = FaceRecognizer(
            gallery_path=gallery_path,
            det_size=int(det_size),
            device=str(device),
            rebuild_gallery=bool(rebuild_gallery),
            iface_use_tiling=bool(iface_use_tiling),
        )
        self.debug_identify = bool(debug_identify)

    def _build_header_meta(
        self,
        input_video: str,
        output_video: Optional[str],
        fps: float,
        frame_interval: int,
        frame_interval_sec: float,
        frame_interval_frames: Optional[int],
        mode: str,
    ) -> Dict:
        return {
            "video": str(input_video),
            "output_video": str(output_video) if output_video is not None else None,
            "fps": float(fps),
            "used_frame_interval": int(frame_interval),
            "requested_interval_seconds": float(frame_interval_sec),
            "requested_interval_frames": int(frame_interval_frames) if frame_interval_frames is not None else None,
            "mode": mode,
            "schema_version": SCHEMA_VERSION,
        }

    def _build_frame_record(self, frame_idx: int, fps: float, detections: List[Dict]) -> Dict:
        timestamp = frame_idx / fps if fps > 0 else 0.0
        return {
            "frame": int(frame_idx),
            "timestamp": float(timestamp),
            "ts_str": _format_ts(timestamp),
            "detections": detections,
        }

    def _refresh_track_identities(self, tracker: SimpleTracker) -> None:
        """Re-identify tracks using aggregated embeddings for multi-frame reinforcement."""
        for tid, info in tracker.tracks.items():
            t = info.get("tracklet")
            if t is None:
                continue
            agg = getattr(t, "agg_embedding", None)
            if agg is None:
                continue
            try:
                if getattr(t, "qualities", None):
                    q = float(np.mean(np.asarray(t.qualities, dtype=np.float32)))
                else:
                    q = 0.5
            except Exception:
                q = 0.5
            try:
                identity, sim = self.recognizer.recognize_identity(agg, q, debug=False)
            except Exception:
                identity, sim = "未知", 0.0
            info["resolved_identity"] = identity
            info["resolved_similarity"] = float(sim)

    def _apply_identity_hysteresis(
        self,
        tracker: SimpleTracker,
        lock_threshold: float,
        lock_min_frames: int,
        switch_threshold: float,
        switch_min_frames: int,
        unlock_threshold: float,
        unlock_grace_frames: int,
        hold_unknown_frames: int,
    ) -> None:
        """Apply hysteresis/locking to stabilize per-track displayed identity."""
        for tid, info in tracker.tracks.items():
            cand = str(info.get("resolved_identity") or "未知")
            sim = float(info.get("resolved_similarity", 0.0) or 0.0)

            try:
                motion = float(info.get("motion_norm_ema", 0.0) or 0.0)
            except Exception:
                motion = 0.0
            stable_factor = 1.0 - max(0.0, min(1.0, motion / 0.25))
            try:
                if int(info.get("lost", 0) or 0) > 0:
                    stable_factor *= 0.35
            except Exception:
                pass

            eff_hold_unknown_frames = int(hold_unknown_frames) + int(round(20.0 * stable_factor))
            eff_unlock_grace_frames = int(unlock_grace_frames) + int(round(6.0 * stable_factor))
            eff_unlock_threshold = float(unlock_threshold) - (0.08 * stable_factor)

            locked = info.get("locked_identity")
            lock_ev = int(info.get("lock_evidence", 0) or 0)
            sw_ev = int(info.get("switch_evidence", 0) or 0)
            unk = int(info.get("unknown_streak", 0) or 0)

            if locked is None:
                if cand != "未知" and sim >= float(lock_threshold):
                    lock_ev += 1
                else:
                    lock_ev = 0

                if lock_ev >= int(lock_min_frames):
                    info["locked_identity"] = cand
                    info["locked_similarity"] = float(sim)
                    info["is_locked"] = True
                    locked = cand
                    sw_ev = 0
                    unk = 0

                info["display_identity"] = locked if locked is not None else cand
                info["display_similarity"] = float(info.get("locked_similarity", sim) if locked is not None else sim)
            else:
                if cand == locked:
                    sw_ev = 0
                    unk = 0
                    info["locked_similarity"] = float(max(float(info.get("locked_similarity", 0.0) or 0.0), sim))
                elif cand == "未知":
                    unk += 1
                    if unk >= int(eff_hold_unknown_frames) and sim < float(eff_unlock_threshold):
                        sw_ev += 1
                    else:
                        sw_ev = 0
                else:
                    unk = 0
                    if sim >= float(switch_threshold):
                        sw_ev += 1
                    else:
                        sw_ev = 0

                    if sw_ev >= int(switch_min_frames):
                        info["locked_identity"] = cand
                        info["locked_similarity"] = float(sim)
                        info["is_locked"] = True
                        locked = cand
                        sw_ev = 0

                if cand == "未知" and sim < float(eff_unlock_threshold) and sw_ev >= int(eff_unlock_grace_frames):
                    info["locked_identity"] = None
                    info["locked_similarity"] = 0.0
                    info["is_locked"] = False
                    locked = None
                    lock_ev = 0
                    sw_ev = 0
                    unk = 0

                info["display_identity"] = locked if locked is not None else cand
                info["display_similarity"] = float(info.get("locked_similarity", 0.0) if locked is not None else sim)

            info["lock_evidence"] = int(lock_ev)
            info["switch_evidence"] = int(sw_ev)
            info["unknown_streak"] = int(unk)

    def _draw_track_overlays_batch(self, frame: np.ndarray, overlays: List[Dict]) -> None:
        if frame is None or not overlays:
            return

        text_items: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = []

        def _pick_text_color_for_bg(bg_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
            b, g, r = [float(x) for x in bg_bgr]
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return (0, 0, 0) if y >= 140.0 else (255, 255, 255)

        for ov in overlays:
            x1, y1, x2, y2 = ov["bbox"]
            identity = str(ov.get("identity") or "未知")
            similarity = float(ov.get("similarity", 0.0) or 0.0)
            quality = float(ov.get("quality", 0.0) or 0.0)
            tid = ov.get("track_id")

            if identity == "未知":
                color = (0, 0, 255)
            elif quality >= 0.8:
                color = (0, 255, 0)
            elif quality >= 0.6:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            text_color = _pick_text_color_for_bg(color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{identity} ({similarity:.3f})"
            if tid is not None:
                label = f"#{int(tid)} {label}"
            qlabel = f"Q:{quality:.2f}"

            face_h = max(12, int(y2 - y1))
            label_font_size = max(12, int(face_h * 0.12))
            q_font_size = max(10, int(face_h * 0.08))

            text_w, text_h = measure_text_cn(label, label_font_size)
            padding_x = max(6, int(label_font_size * 0.3))
            padding_y = max(4, int(label_font_size * 0.2))

            bg_x1 = int(x1)
            bg_y1 = max(0, int(y1 - text_h - padding_y * 2))
            bg_x2 = int(x1 + text_w + padding_x * 2)
            bg_y2 = int(y1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

            text_org = (bg_x1 + padding_x, bg_y1 + padding_y)
            q_w, q_h = measure_text_cn(qlabel, q_font_size)
            q_org = (int(x1), int(min(frame.shape[0] - q_h - 2, y1 + 15)))

            text_items.append((label, text_org, int(label_font_size), text_color))
            text_items.append((qlabel, q_org, int(q_font_size), text_color))

        draw_texts_cn(frame, text_items)

    def _map_identity_to_person(
        self, identity: str, person_map: Dict[str, int], next_person_pid: int
    ) -> Tuple[Optional[int], int]:
        if identity and identity != "未知":
            if identity not in person_map:
                person_map[identity] = next_person_pid
                next_person_pid += 1
            return person_map[identity], next_person_pid
        return None, next_person_pid

    def _draw_tracks(
        self,
        frame,
        tracker: SimpleTracker,
        frame_idx: int,
        frame_interval: int,
        person_map: Dict[str, int],
    ) -> None:
        pid_best: Dict[int, Tuple[int, Dict]] = {}
        unknown_tracks: Dict[int, Dict] = {}
        for tid, info in tracker.tracks.items():
            last_bbox = info.get("last_bbox")
            if not last_bbox:
                continue
            try:
                sb = info.get("smoothed_bbox")
                curr = np.array(last_bbox, dtype=float)
                if sb is None:
                    info["smoothed_bbox"] = [int(x) for x in curr]
                else:
                    alpha = 0.65
                    sb_arr = np.array(sb, dtype=float)
                    merged_sb = alpha * curr + (1.0 - alpha) * sb_arr
                    info["smoothed_bbox"] = [int(x) for x in merged_sb]
            except Exception:
                pass
            t = info.get("tracklet")
            identity = "未知"
            if t and getattr(t, "identities", None):
                try:
                    identity = t.identities[-1]
                except Exception:
                    identity = str(t.identities[0]) if t.identities else "未知"

            pid = info.get("person_id") if info.get("person_id", None) is not None else None
            if pid is None and identity and identity != "未知" and identity in person_map:
                pid = person_map[identity]

            if pid is not None:
                cur = pid_best.get(pid)
                last_seen = int(info.get("last_seen", -1))
                if cur is None or last_seen > int(cur[1].get("last_seen", -1)):
                    pid_best[pid] = (tid, info)
            else:
                unknown_tracks[tid] = info

        overlays: List[Dict] = []

        for pid, (tid, info) in pid_best.items():
            last_bbox = info.get("last_bbox")
            if not last_bbox:
                continue
            sb = info.get("smoothed_bbox")
            coords = [int(x) for x in (sb if sb is not None else last_bbox)]
            t = info.get("tracklet")
            identity = info.get("display_identity") or info.get("resolved_identity") or "未知"
            similarity = float(info.get("display_similarity", 0.0) or info.get("resolved_similarity", 0.0) or 0.0)
            try:
                quality = float(np.mean(np.asarray(t.qualities, dtype=np.float32))) if t and t.qualities else 0.0
            except Exception:
                quality = 0.0
            overlays.append(
                {"bbox": coords, "identity": identity, "similarity": similarity, "quality": quality, "track_id": tid}
            )

        unknown_threshold = max(3, int(frame_interval // 2))
        for tid, info in unknown_tracks.items():
            last_seen = int(info.get("last_seen", -9999))
            if last_seen < frame_idx - unknown_threshold:
                continue
            last_bbox = info.get("last_bbox")
            if not last_bbox:
                continue
            sb = info.get("smoothed_bbox")
            coords = [int(x) for x in (sb if sb is not None else last_bbox)]
            t = info.get("tracklet")
            identity = info.get("display_identity") or info.get("resolved_identity") or "未知"
            similarity = float(info.get("display_similarity", 0.0) or info.get("resolved_similarity", 0.0) or 0.0)
            try:
                quality = float(np.mean(np.asarray(t.qualities, dtype=np.float32))) if t and t.qualities else 0.0
            except Exception:
                quality = 0.0
            overlays.append(
                {"bbox": coords, "identity": identity, "similarity": similarity, "quality": quality, "track_id": tid}
            )

        try:
            self._draw_track_overlays_batch(frame, overlays)
        except Exception:
            logger.debug("批量绘制轨迹失败", exc_info=True)

    def _compute_frame_interval(
        self, fps: float, frame_interval_sec: float, frame_interval_frames: Optional[int]
    ) -> int:
        if frame_interval_frames is not None and int(frame_interval_frames) > 0:
            frame_interval = max(1, int(frame_interval_frames))
            logger.info(f"视频 FPS={fps:.2f}, 使用帧间隔: 每 {frame_interval} 帧抽一帧")
        else:
            frame_interval = max(1, int(round(fps * frame_interval_sec)))
            logger.info(f"视频 FPS={fps:.2f}, 使用秒间隔: 每 {frame_interval_sec}s 抽一帧 -> 每 {frame_interval} 帧")
        return frame_interval

    def process(
        self,
        input_video: str,
        output_video: str,
        output_json: str,
        frame_interval_sec: float = 2.0,
        frame_interval_frames: int = None,
        batch_frames: int = 1,
        ffmpeg_codec: Optional[str] = None,
    ) -> Dict:
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_video:
            out = FFmpegPipeWriter(str(output_video), width=width, height=height, fps=fps, codec=ffmpeg_codec)

        frame_interval = self._compute_frame_interval(fps, frame_interval_sec, frame_interval_frames)

        frame_idx = 0
        occurrences: Dict[str, List[int]] = {}
        frames_info: List[Dict] = []

        batch_frames = max(1, int(batch_frames))
        frame_buffer: Dict[int, np.ndarray] = {}
        pending_sample_indices: List[int] = []
        write_pos = 0

        def flush_frames_up_to():
            nonlocal write_pos
            while True:
                if write_pos not in frame_buffer:
                    break
                if write_pos in pending_sample_indices:
                    break
                if out is not None:
                    out.write(frame_buffer[write_pos])
                del frame_buffer[write_pos]
                write_pos += 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer[frame_idx] = frame
            if frame_idx % frame_interval == 0:
                pending_sample_indices.append(frame_idx)

            while len(pending_sample_indices) >= batch_frames:
                current_batch_indices = pending_sample_indices[:batch_frames]
                batch_images = [frame_buffer[idx] for idx in current_batch_indices if idx in frame_buffer]
                if not batch_images:
                    pending_sample_indices = pending_sample_indices[batch_frames:]
                    break

                faces_batch = self.recognizer.detect_faces_batch(batch_images)
                for local_i, global_idx in enumerate(current_batch_indices):
                    if global_idx not in frame_buffer:
                        continue
                    faces = faces_batch[local_i] if local_i < len(faces_batch) else []
                    frame_ref = frame_buffer[global_idx]
                    detections: List[Dict] = []
                    for i, face in enumerate(faces):
                        quality = self.recognizer.assess_face_quality(face, frame_ref.shape)
                        identity, similarity = self.recognizer.recognize_identity(
                            face.embedding, quality, debug=self.debug_identify
                        )
                        bbox = face.bbox.astype(int).tolist()
                        det_raw = {
                            "identity": identity,
                            "similarity": similarity,
                            "quality": quality,
                            "bbox": bbox,
                            "det_size": getattr(face, "det_size", None),
                            "enhancement": getattr(face, "enhancement", "original"),
                        }
                        det = serialize_detection(det_raw, frame_ref.shape)
                        detections.append(det)

                        draw_result = {
                            "bbox": det["bbox"],
                            "identity": det["identity"],
                            "similarity": det["similarity"],
                            "quality": det["quality"],
                            "landmarks": getattr(face, "kps", None),
                        }
                        if output_video:
                            try:
                                self.recognizer._draw(frame_ref, draw_result, i)
                            except Exception:
                                logger.debug("绘制识别结果失败")

                        occurrences.setdefault(identity, []).append(global_idx)

                    frames_info.append(self._build_frame_record(global_idx, fps, detections))

                pending_sample_indices = pending_sample_indices[batch_frames:]

            flush_frames_up_to()
            frame_idx += 1

        if pending_sample_indices:
            batch_images = [frame_buffer[idx] for idx in pending_sample_indices if idx in frame_buffer]
            if batch_images:
                faces_batch = self.recognizer.detect_faces_batch(batch_images)
                for local_i, global_idx in enumerate(pending_sample_indices):
                    if global_idx not in frame_buffer:
                        continue
                    faces = faces_batch[local_i] if local_i < len(faces_batch) else []
                    frame_ref = frame_buffer[global_idx]
                    detections: List[Dict] = []
                    for i, face in enumerate(faces):
                        quality = self.recognizer.assess_face_quality(face, frame_ref.shape)
                        identity, similarity = self.recognizer.recognize_identity(
                            face.embedding, quality, debug=self.debug_identify
                        )
                        bbox = face.bbox.astype(int).tolist()
                        det_raw = {
                            "identity": identity,
                            "similarity": similarity,
                            "quality": quality,
                            "bbox": bbox,
                            "det_size": getattr(face, "det_size", None),
                            "enhancement": getattr(face, "enhancement", "original"),
                        }
                        det = serialize_detection(det_raw, frame_ref.shape)
                        detections.append(det)

                        draw_result = {
                            "bbox": det["bbox"],
                            "identity": det["identity"],
                            "similarity": det["similarity"],
                            "quality": det["quality"],
                            "landmarks": getattr(face, "kps", None),
                        }
                        if output_video:
                            try:
                                self.recognizer._draw(frame_ref, draw_result, i)
                            except Exception:
                                logger.debug("绘制识别结果失败")

                        occurrences.setdefault(identity, []).append(global_idx)

                    frames_info.append(self._build_frame_record(global_idx, fps, detections))

        while write_pos in frame_buffer:
            if out is not None:
                out.write(frame_buffer[write_pos])
            del frame_buffer[write_pos]
            write_pos += 1

        cap.release()
        if out is not None:
            out.close()

        result = self._build_header_meta(
            input_video,
            output_video if output_video else None,
            fps,
            frame_interval,
            frame_interval_sec,
            frame_interval_frames,
            mode="simple",
        )
        result["occurrences"] = {k: [int(v) for v in vs] for k, vs in occurrences.items()}
        result["frames"] = frames_info

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            f"处理完成（简单模式），输出视频: {output_video if output_video else '（未输出视频）'}, 输出数据: {output_json}"
        )
        return result

    def process_with_tracklets(
        self,
        input_video: str,
        output_video: str,
        output_json: str,
        frame_interval_sec: float = 2.0,
        frame_interval_frames: int = None,
        batch_frames: int = 1,
        iou_threshold: float = 0.3,
        max_lost: int = 5,
        merge_similarity_threshold: float = 0.86,
        tracklet_min_votes: int = 2,
        lock_threshold: float = 0.46,
        lock_min_frames: int = 2,
        switch_threshold: float = 0.54,
        switch_min_frames: int = 2,
        unlock_threshold: float = 0.35,
        unlock_grace_frames: int = 3,
        hold_unknown_frames: int = 8,
        ffmpeg_codec: Optional[str] = None,
        max_frames: int = None,
        max_seconds: float = None,
    ) -> Dict:
        tracker = SimpleTracker(iou_threshold=iou_threshold, max_lost=max_lost)
        person_map: Dict[str, int] = {}
        next_person_pid = 1

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_video:
            out = FFmpegPipeWriter(str(output_video), width=width, height=height, fps=fps, codec=ffmpeg_codec)

        frame_interval = self._compute_frame_interval(fps, frame_interval_sec, frame_interval_frames)

        frame_idx = 0
        frames_info: List[Dict] = []

        max_frames_i = int(max_frames) if max_frames is not None else None
        max_seconds_f = float(max_seconds) if max_seconds is not None else None

        batch_frames = max(1, int(batch_frames))

        frame_buffer: Dict[int, np.ndarray] = {}
        pending_sample_indices: List[int] = []
        write_pos = 0

        def flush_frames_up_to():
            nonlocal write_pos
            while True:
                if write_pos not in frame_buffer:
                    break
                if write_pos in pending_sample_indices:
                    break
                if output_video:
                    try:
                        self._draw_tracks(frame_buffer[write_pos], tracker, write_pos, frame_interval, person_map)
                    except Exception:
                        logger.debug("绘制轨迹失败", exc_info=True)
                    if out is not None:
                        out.write(frame_buffer[write_pos])
                del frame_buffer[write_pos]
                write_pos += 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames_i is not None and frame_idx >= max_frames_i:
                break
            if max_seconds_f is not None and fps > 0 and (frame_idx / fps) >= max_seconds_f:
                break

            frame_buffer[frame_idx] = frame
            if frame_idx % frame_interval == 0:
                pending_sample_indices.append(frame_idx)

            while len(pending_sample_indices) >= batch_frames:
                current_batch_indices = pending_sample_indices[:batch_frames]
                batch_images = [frame_buffer[idx] for idx in current_batch_indices if idx in frame_buffer]
                if not batch_images:
                    pending_sample_indices = pending_sample_indices[batch_frames:]
                    break

                faces_batch = self.recognizer.detect_faces_batch(batch_images)
                for local_i, global_idx in enumerate(current_batch_indices):
                    if global_idx not in frame_buffer:
                        continue
                    faces = faces_batch[local_i] if local_i < len(faces_batch) else []
                    frame_ref = frame_buffer[global_idx]
                    detections_raw: List[Dict] = []
                    for face in faces:
                        quality = self.recognizer.assess_face_quality(face, frame_ref.shape)
                        identity, similarity = self.recognizer.recognize_identity(
                            face.embedding, quality, debug=self.debug_identify
                        )
                        bbox = face.bbox.astype(int).tolist()
                        person_id, next_person_pid_local = self._map_identity_to_person(
                            identity, person_map, next_person_pid
                        )
                        next_person_pid = next_person_pid_local
                        det = {
                            "bbox": bbox,
                            "embedding": face.embedding,
                            "quality": quality,
                            "identity": identity,
                            "similarity": similarity,
                            "person_id": person_id,
                            "landmarks": getattr(face, "kps", None),
                            "det_size": getattr(face, "det_size", None),
                            "enhancement": getattr(face, "enhancement", "original"),
                        }
                        detections_raw.append(det)

                    tracker.update(detections_raw, global_idx)

                    try:
                        self._refresh_track_identities(tracker)
                        self._apply_identity_hysteresis(
                            tracker,
                            lock_threshold=float(lock_threshold),
                            lock_min_frames=int(lock_min_frames),
                            switch_threshold=float(switch_threshold),
                            switch_min_frames=int(switch_min_frames),
                            unlock_threshold=float(unlock_threshold),
                            unlock_grace_frames=int(unlock_grace_frames),
                            hold_unknown_frames=int(hold_unknown_frames),
                        )
                    except Exception:
                        pass

                    export_dets: List[Dict] = []
                    for det in detections_raw:
                        tid = det.get("track_id")
                        if tid is not None and tid in tracker.tracks:
                            info = tracker.tracks.get(tid, {})
                            det["track_identity"] = info.get("resolved_identity")
                            det["track_similarity"] = info.get("resolved_similarity")
                            det["track_display_identity"] = info.get("display_identity")
                            det["track_display_similarity"] = info.get("display_similarity")
                            det["track_is_locked"] = bool(info.get("is_locked", False))
                        ser = serialize_detection(det, frame_ref.shape)
                        export_dets.append(ser)
                    frames_info.append(self._build_frame_record(global_idx, fps, export_dets))

                try:
                    tracker.merge_similar_tracks(threshold=merge_similarity_threshold)
                except Exception:
                    logger.debug("合并相似轨迹失败", exc_info=True)

                pending_sample_indices = pending_sample_indices[batch_frames:]

            flush_frames_up_to()
            frame_idx += 1

        if pending_sample_indices:
            batch_images = [frame_buffer[idx] for idx in pending_sample_indices if idx in frame_buffer]
            if batch_images:
                faces_batch = self.recognizer.detect_faces_batch(batch_images)
                for local_i, global_idx in enumerate(pending_sample_indices):
                    if global_idx not in frame_buffer:
                        continue
                    faces = faces_batch[local_i] if local_i < len(faces_batch) else []
                    frame_ref = frame_buffer[global_idx]
                    detections_raw: List[Dict] = []
                    for face in faces:
                        quality = self.recognizer.assess_face_quality(face, frame_ref.shape)
                        identity, similarity = self.recognizer.recognize_identity(face.embedding, quality)
                        bbox = face.bbox.astype(int).tolist()
                        person_id, next_person_pid_local = self._map_identity_to_person(
                            identity, person_map, next_person_pid
                        )
                        next_person_pid = next_person_pid_local
                        det = {
                            "bbox": bbox,
                            "embedding": face.embedding,
                            "quality": quality,
                            "identity": identity,
                            "similarity": similarity,
                            "person_id": person_id,
                            "landmarks": getattr(face, "kps", None),
                            "det_size": getattr(face, "det_size", None),
                            "enhancement": getattr(face, "enhancement", "original"),
                        }
                        detections_raw.append(det)

                    tracker.update(detections_raw, global_idx)

                    try:
                        self._refresh_track_identities(tracker)
                        self._apply_identity_hysteresis(
                            tracker,
                            lock_threshold=float(lock_threshold),
                            lock_min_frames=int(lock_min_frames),
                            switch_threshold=float(switch_threshold),
                            switch_min_frames=int(switch_min_frames),
                            unlock_threshold=float(unlock_threshold),
                            unlock_grace_frames=int(unlock_grace_frames),
                            hold_unknown_frames=int(hold_unknown_frames),
                        )
                    except Exception:
                        pass

                    export_dets: List[Dict] = []
                    for det in detections_raw:
                        tid = det.get("track_id")
                        if tid is not None and tid in tracker.tracks:
                            info = tracker.tracks.get(tid, {})
                            det["track_identity"] = info.get("resolved_identity")
                            det["track_similarity"] = info.get("resolved_similarity")
                            det["track_display_identity"] = info.get("display_identity")
                            det["track_display_similarity"] = info.get("display_similarity")
                            det["track_is_locked"] = bool(info.get("is_locked", False))
                        ser = serialize_detection(det, frame_ref.shape)
                        try:
                            h, w = frame_ref.shape[:2]
                            if ser.get("bbox"):
                                x1, y1, x2, y2 = ser["bbox"]
                                ser["bbox_norm"] = [
                                    round(x1 / w, 4),
                                    round(y1 / h, 4),
                                    round(x2 / w, 4),
                                    round(y2 / h, 4),
                                ]
                                if ser.get("center"):
                                    cx, cy = ser["center"]
                                    ser["center_norm"] = [round(cx / w, 4), round(cy / h, 4)]
                                else:
                                    ser["center_norm"] = None
                        except Exception:
                            pass
                        export_dets.append(ser)
                    frames_info.append(self._build_frame_record(global_idx, fps, export_dets))

            try:
                tracker.merge_similar_tracks(threshold=merge_similarity_threshold)
            except Exception:
                logger.debug("合并相似轨迹失败", exc_info=True)

        while write_pos in frame_buffer:
            if output_video:
                try:
                    self._draw_tracks(frame_buffer[write_pos], tracker, write_pos, frame_interval, person_map)
                except Exception:
                    logger.debug("绘制轨迹失败", exc_info=True)
            if out is not None:
                out.write(frame_buffer[write_pos])
            del frame_buffer[write_pos]
            write_pos += 1

        cap.release()
        if out is not None:
            out.close()

        remaining = [info["tracklet"] for info in tracker.tracks.values()]

        tracklets_out: List[Dict] = []
        for t in remaining:
            tracklets_out.append(
                serialize_tracklet(t, video_size=(width, height), tracklet_min_votes=int(tracklet_min_votes))
            )

        for to in tracklets_out:
            tid = to.get("id")
            info = tracker.tracks.get(int(tid), None) if tid is not None else None
            if not info:
                continue
            tr = info.get("tracklet")
            agg = getattr(tr, "agg_embedding", None) if tr else None
            if agg is None:
                continue
            try:
                q = float(np.mean(np.asarray(tr.qualities, dtype=np.float32))) if tr and tr.qualities else 0.5
            except Exception:
                q = 0.5
            try:
                ident, sim = self.recognizer.recognize_identity(agg, q, debug=False)
                to["resolved_identity"] = ident
                to["resolved_similarity"] = float(sim)
            except Exception:
                pass

        for to in tracklets_out:
            tid = to.get("id")
            info = tracker.tracks.get(int(tid), None) if tid is not None else None
            if not info:
                continue
            to["display_identity"] = info.get("display_identity")
            to["display_similarity"] = float(info.get("display_similarity", 0.0) or 0.0)
            to["is_locked"] = bool(info.get("is_locked", False))

        try:
            for to in tracklets_out:
                rep = to.get("representative_bbox")
                if rep is None:
                    to["representative_bbox_norm"] = None
                    continue
                x1, y1, x2, y2 = rep
                try:
                    to["representative_bbox_norm"] = [
                        round(x1 / width, 4),
                        round(y1 / height, 4),
                        round(x2 / width, 4),
                        round(y2 / height, 4),
                    ]
                except Exception:
                    to["representative_bbox_norm"] = None
        except Exception:
            pass

        occurrences: Dict[str, List[int]] = {}
        for fr in frames_info:
            frame_no = int(fr.get("frame", 0))
            for det in fr.get("detections", []):
                identity = det.get("identity", "未知")
                occurrences.setdefault(identity, []).append(frame_no)

        result = self._build_header_meta(
            input_video,
            output_video,
            fps,
            frame_interval,
            frame_interval_sec,
            frame_interval_frames,
            mode="tracklet",
        )
        result["occurrences"] = {k: [int(v) for v in vs] for k, vs in occurrences.items()}
        result["frames"] = frames_info
        result["tracklets"] = tracklets_out

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"处理完成（轨迹模式），输出视频: {output_video}, 输出数据: {output_json}")
        return result
