import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_recognizer import FaceRecognizer
from utils.serializer import serialize_detection, serialize_tracklet
from utils.draw import draw_text_cn, measure_text_cn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCHEMA_VERSION = "v1"


def _format_ts(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = (boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


@dataclass
class Tracklet:
    id: int
    frame_indices: List[int] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    identities: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)

    def add(self, frame_idx: int, bbox: List[int], embedding: np.ndarray, quality: float, identity: str, similarity: float):
        self.frame_indices.append(int(frame_idx))
        self.bboxes.append([int(x) for x in bbox])
        self.embeddings.append(np.array(embedding))
        self.qualities.append(float(quality))
        self.identities.append(str(identity))
        self.similarities.append(float(similarity))

    def aggregate(self):
        if len(self.embeddings) == 0:
            return None
        embs = np.stack(self.embeddings, axis=0)
        quals = np.array(self.qualities)
        if quals.sum() == 0:
            weights = np.ones_like(quals) / len(quals)
        else:
            weights = quals / quals.sum()
        agg = np.average(embs, axis=0, weights=weights)
        return agg


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.person_to_track: Dict[int, int] = {}

    def update(self, detections: List[Dict], frame_idx: int) -> List[Tracklet]:
        assigned = set()
        det_bboxes = [d['bbox'] for d in detections]

        for tid, info in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            for j, db in enumerate(det_bboxes):
                if j in assigned:
                    continue
                score = _iou(info['last_bbox'], db)
                if score > best_iou:
                    best_iou = score
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_threshold:
                d = detections[best_j]
                d['track_id'] = tid
                info['tracklet'].add(frame_idx, d['bbox'], d['embedding'], d['quality'], d['identity'], d['similarity'])
                info['last_bbox'] = d['bbox']
                info['last_seen'] = int(frame_idx)
                if d.get('person_id') is not None:
                    info['person_id'] = d.get('person_id')
                    self.person_to_track[d['person_id']] = tid
                info['lost'] = 0
                assigned.add(best_j)
            else:
                info['lost'] += 1

        for j, d in enumerate(detections):
            if j in assigned:
                continue
            existing_tid = None
            pid = d.get('person_id')
            if pid is not None:
                mapped = self.person_to_track.get(pid)
                if mapped is not None and mapped in self.tracks:
                    existing_tid = mapped

            if existing_tid is not None:
                tid = existing_tid
                info = self.tracks[tid]
                d['track_id'] = tid
                info['tracklet'].add(frame_idx, d['bbox'], d['embedding'], d['quality'], d['identity'], d['similarity'])
                info['last_bbox'] = d['bbox']
                info['last_seen'] = int(frame_idx)
                info['person_id'] = pid
                info['lost'] = 0
                assigned.add(j)
                continue
            tid = self.next_id
            self.next_id += 1
            t = Tracklet(id=tid)
            d['track_id'] = tid
            t.add(frame_idx, d['bbox'], d['embedding'], d['quality'], d['identity'], d['similarity'])
            self.tracks[tid] = {
                'tracklet': t,
                'last_bbox': d['bbox'],
                'lost': 0,
                'last_seen': int(frame_idx),
                'person_id': d.get('person_id'),
            }
            if pid is not None:
                self.person_to_track[pid] = tid

        finished = []
        remove_keys = []
        for tid, info in list(self.tracks.items()):
            if info['lost'] > self.max_lost:
                finished.append(info['tracklet'])
                remove_keys.append(tid)

        for k in remove_keys:
            pid = self.tracks[k].get('person_id')
            if pid is not None and self.person_to_track.get(pid) == k:
                del self.person_to_track[pid]
            del self.tracks[k]

        active = [info['tracklet'] for info in self.tracks.values()]
        return active + finished

    def merge_similar_tracks(self, threshold: float = 0.86):
        tids = list(self.tracks.keys())
        if len(tids) < 2:
            return

        emb_map = {}
        for tid in tids:
            info = self.tracks.get(tid)
            try:
                agg = info['tracklet'].aggregate()
            except Exception:
                agg = None
            if agg is not None:
                emb_map[tid] = np.array(agg, dtype=float)

        def cos_sim(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        merged = set()
        for i in range(len(tids)):
            a = tids[i]
            if a not in self.tracks or a in merged:
                continue
            for j in range(i + 1, len(tids)):
                b = tids[j]
                if b not in self.tracks or b in merged:
                    continue
                pa = self.tracks[a].get('person_id')
                pb = self.tracks[b].get('person_id')
                if pa is not None and pb is not None and pa != pb:
                    continue
                ea = emb_map.get(a)
                eb = emb_map.get(b)
                if ea is None or eb is None:
                    continue
                sim = cos_sim(ea, eb)
                if sim >= threshold:
                    last_seen_a = int(self.tracks[a].get('last_seen', -1))
                    last_seen_b = int(self.tracks[b].get('last_seen', -1))
                    if last_seen_a >= last_seen_b:
                        keep, rem = a, b
                    else:
                        keep, rem = b, a
                    if keep not in self.tracks or rem not in self.tracks:
                        continue
                    kinfo = self.tracks[keep]
                    rinfo = self.tracks[rem]
                    kt = kinfo['tracklet']
                    rt = rinfo['tracklet']
                    kt.frame_indices.extend(rt.frame_indices)
                    kt.bboxes.extend(rt.bboxes)
                    kt.embeddings.extend(rt.embeddings)
                    kt.qualities.extend(rt.qualities)
                    kt.identities.extend(rt.identities)
                    kt.similarities.extend(rt.similarities)
                    kinfo['last_bbox'] = rinfo.get('last_bbox') or kinfo.get('last_bbox')
                    kinfo['last_seen'] = max(int(kinfo.get('last_seen', -1)), int(rinfo.get('last_seen', -1)))
                    if rinfo.get('person_id') is not None:
                        kinfo['person_id'] = rinfo.get('person_id')
                        self.person_to_track[rinfo['person_id']] = keep
                    if 'smoothed_bbox' in rinfo:
                        kinfo['smoothed_bbox'] = rinfo['smoothed_bbox']
                    try:
                        del self.tracks[rem]
                        merged.add(rem)
                    except KeyError:
                        pass


class VideoFaceRecognizer:
    """视频人脸识别：支持简单模式与轨迹模式，统一输出结构。"""

    def __init__(self, gallery_path: str = 'data/id_photos', debug_identify: bool = False):
        self.recognizer = FaceRecognizer(gallery_path=gallery_path)
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
            'video': str(input_video),
            'output_video': str(output_video) if output_video is not None else None,
            'fps': float(fps),
            'used_frame_interval': int(frame_interval),
            'requested_interval_seconds': float(frame_interval_sec),
            'requested_interval_frames': int(frame_interval_frames) if frame_interval_frames is not None else None,
            'mode': mode,
            'schema_version': SCHEMA_VERSION,
        }

    def _build_frame_record(self, frame_idx: int, fps: float, detections: List[Dict]) -> Dict:
        timestamp = frame_idx / fps if fps > 0 else 0.0
        return {
            'frame': int(frame_idx),
            'timestamp': float(timestamp),
            'ts_str': _format_ts(timestamp),
            'detections': detections,
        }

    pass

    def _map_identity_to_person(self, identity: str, person_map: Dict[str, int], next_person_pid: int) -> Tuple[Optional[int], int]:
        if identity and identity != '未知':
            if identity not in person_map:
                person_map[identity] = next_person_pid
                next_person_pid += 1
            return person_map[identity], next_person_pid
        return None, next_person_pid

    def _collect_detections(
        self,
        frame,
        person_map: Dict[str, int],
        next_person_pid: int,
    ) -> Tuple[List[Dict], int]:
        faces = self.recognizer.detect_faces(frame)
        detections: List[Dict] = []
        for face in faces:
            quality = self.recognizer.assess_face_quality(face, frame.shape)
            identity, similarity = self.recognizer.recognize_identity(face.embedding, quality, debug=self.debug_identify)
            bbox = face.bbox.astype(int).tolist()
            person_id, next_person_pid = self._map_identity_to_person(identity, person_map, next_person_pid)
            det = {
                'bbox': bbox,
                'embedding': face.embedding,
                'quality': quality,
                'identity': identity,
                'similarity': similarity,
                'person_id': person_id,
                'landmarks': getattr(face, 'kps', None),
                'det_size': getattr(face, 'det_size', None),
                'enhancement': getattr(face, 'enhancement', 'original'),
            }
            detections.append(det)
        return detections, next_person_pid

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
            last_bbox = info.get('last_bbox')
            if not last_bbox:
                continue
            try:
                sb = info.get('smoothed_bbox')
                curr = np.array(last_bbox, dtype=float)
                if sb is None:
                    info['smoothed_bbox'] = [int(x) for x in curr]
                else:
                    alpha = 0.65
                    sb_arr = np.array(sb, dtype=float)
                    merged_sb = alpha * curr + (1.0 - alpha) * sb_arr
                    info['smoothed_bbox'] = [int(x) for x in merged_sb]
            except Exception:
                pass
            t = info.get('tracklet')
            identity = '未知'
            if t and getattr(t, 'identities', None):
                try:
                    identity = t.identities[-1]
                except Exception:
                    identity = str(t.identities[0]) if t.identities else '未知'

            pid = info.get('person_id') if info.get('person_id', None) is not None else None
            if pid is None and identity and identity != '未知' and identity in person_map:
                pid = person_map[identity]

            if pid is not None:
                cur = pid_best.get(pid)
                last_seen = int(info.get('last_seen', -1))
                if cur is None or last_seen > int(cur[1].get('last_seen', -1)):
                    pid_best[pid] = (tid, info)
            else:
                unknown_tracks[tid] = info

        for pid, (tid, info) in pid_best.items():
            last_bbox = info.get('last_bbox')
            if not last_bbox:
                continue
            sb = info.get('smoothed_bbox')
            coords = [int(x) for x in (sb if sb is not None else last_bbox)]
            t = info.get('tracklet')
            identity = '未知'
            similarity = 0.0
            quality = 0.0
            if t:
                if t.identities:
                    identity = t.identities[-1]
                if t.similarities:
                    similarity = float(t.similarities[-1])
                if t.qualities:
                    quality = float(t.qualities[-1])
            draw_result = {
                'bbox': coords,
                'identity': identity,
                'similarity': similarity,
                'quality': quality,
                'landmarks': None,
            }
            try:
                self.recognizer._draw(frame, draw_result, tid)
            except Exception:
                logger.debug("绘制轨迹结果失败")

        unknown_threshold = max(3, int(frame_interval // 2))
        for tid, info in unknown_tracks.items():
            last_seen = int(info.get('last_seen', -9999))
            if last_seen < frame_idx - unknown_threshold:
                continue
            last_bbox = info.get('last_bbox')
            if not last_bbox:
                continue
            sb = info.get('smoothed_bbox')
            coords = [int(x) for x in (sb if sb is not None else last_bbox)]
            t = info.get('tracklet')
            identity = '未知'
            similarity = 0.0
            quality = 0.0
            if t:
                if t.identities:
                    identity = t.identities[-1]
                if t.similarities:
                    similarity = float(t.similarities[-1])
                if t.qualities:
                    quality = float(t.qualities[-1])
            draw_result = {
                'bbox': coords,
                'identity': identity,
                'similarity': similarity,
                'quality': quality,
                'landmarks': None,
            }
            try:
                self.recognizer._draw(frame, draw_result, tid)
            except Exception:
                logger.debug("绘制未知轨迹结果失败")

    def _compute_frame_interval(self, fps: float, frame_interval_sec: float, frame_interval_frames: Optional[int]) -> int:
        if frame_interval_frames is not None and int(frame_interval_frames) > 0:
            frame_interval = max(1, int(frame_interval_frames))
            logger.info(f"视频 FPS={fps:.2f}, 使用帧间隔: 每 {frame_interval} 帧抽一帧")
        else:
            frame_interval = max(1, int(round(fps * frame_interval_sec)))
            logger.info(
                f"视频 FPS={fps:.2f}, 使用秒间隔: 每 {frame_interval_sec}s 抽一帧 -> 每 {frame_interval} 帧",
            )
        return frame_interval

    def process(
        self,
        input_video: str,
        output_video: str,
        output_json: str,
        frame_interval_sec: float = 2.0,
        frame_interval_frames: int = None,
        batch_frames: int = 1,
    ) -> Dict:
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        frame_interval = self._compute_frame_interval(fps, frame_interval_sec, frame_interval_frames)

        frame_idx = 0
        occurrences: Dict[str, List[int]] = {}
        frames_info: List[Dict] = []

        batch_frames = max(1, int(batch_frames))
        # 统一批处理路径：batch_frames 可以为 1，此时等价于逐帧调用批接口
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
                # 只有在需要输出视频时才写帧，避免无谓开销
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

            # 凑够一个 batch 就做一次批量检测与识别
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
                        identity, similarity = self.recognizer.recognize_identity(face.embedding, quality, debug=self.debug_identify)
                        bbox = face.bbox.astype(int).tolist()
                        det_raw = {
                            'identity': identity,
                            'similarity': similarity,
                            'quality': quality,
                            'bbox': bbox,
                            'det_size': getattr(face, 'det_size', None),
                            'enhancement': getattr(face, 'enhancement', 'original'),
                        }
                        det = serialize_detection(det_raw, frame_ref.shape)
                        detections.append(det)

                        draw_result = {
                            'bbox': det['bbox'],
                            'identity': det['identity'],
                            'similarity': det['similarity'],
                            'quality': det['quality'],
                            'landmarks': getattr(face, 'kps', None),
                        }
                        # 绘制仅在需要输出视频时进行
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

        # 处理剩余未满一个 batch 的采样帧
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
                        identity, similarity = self.recognizer.recognize_identity(face.embedding, quality, debug=self.debug_identify)
                        bbox = face.bbox.astype(int).tolist()
                        det_raw = {
                            'identity': identity,
                            'similarity': similarity,
                            'quality': quality,
                            'bbox': bbox,
                            'det_size': getattr(face, 'det_size', None),
                            'enhancement': getattr(face, 'enhancement', 'original'),
                        }
                        det = serialize_detection(det_raw, frame_ref.shape)
                        detections.append(det)

                        draw_result = {
                            'bbox': det['bbox'],
                            'identity': det['identity'],
                            'similarity': det['similarity'],
                            'quality': det['quality'],
                            'landmarks': getattr(face, 'kps', None),
                        }
                        try:
                            self.recognizer._draw(frame_ref, draw_result, i)
                        except Exception:
                            logger.debug("绘制识别结果失败")

                        occurrences.setdefault(identity, []).append(global_idx)

                    frames_info.append(self._build_frame_record(global_idx, fps, detections))

        # 写出剩余所有帧（仅在输出视频时写），否则清理缓存
        while write_pos in frame_buffer:
            if out is not None:
                out.write(frame_buffer[write_pos])
            del frame_buffer[write_pos]
            write_pos += 1

        cap.release()
        if out is not None:
            out.release()

        # 构建并写入 JSON（与 tracklet 模式保持一致）
        result = self._build_header_meta(
            input_video,
            output_video if output_video else None,
            fps,
            frame_interval,
            frame_interval_sec,
            frame_interval_frames,
            mode='simple',
        )
        result['occurrences'] = {k: [int(v) for v in vs] for k, vs in occurrences.items()}
        result['frames'] = frames_info

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"处理完成（简单模式），输出视频: {output_video if output_video else '（未输出视频）'}, 输出数据: {output_json}")
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        frame_interval = self._compute_frame_interval(fps, frame_interval_sec, frame_interval_frames)

        frame_idx = 0
        frames_info: List[Dict] = []

        batch_frames = max(1, int(batch_frames))

        # 统一批处理路径：batch_frames 可以为 1，此时等价于逐帧调用批接口
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
                # 在写出前，为该帧绘制轨迹
                # 仅当需要输出视频时才绘制并写出帧
                if output_video:
                    try:
                        self._draw_tracks(frame_buffer[write_pos], tracker, write_pos, frame_interval, person_map)
                    except Exception:
                        logger.debug("绘制轨迹失败", exc_info=True)
                    if out is not None:
                        out.write(frame_buffer[write_pos])
                # 无论是否写出，清理缓存
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
                    detections_raw: List[Dict] = []
                    for face in faces:
                        quality = self.recognizer.assess_face_quality(face, frame_ref.shape)
                        identity, similarity = self.recognizer.recognize_identity(face.embedding, quality, debug=self.debug_identify)
                        bbox = face.bbox.astype(int).tolist()
                        person_id, next_person_pid_local = self._map_identity_to_person(identity, person_map, next_person_pid)
                        next_person_pid = next_person_pid_local
                        det = {
                            'bbox': bbox,
                            'embedding': face.embedding,
                            'quality': quality,
                            'identity': identity,
                            'similarity': similarity,
                            'person_id': person_id,
                            'landmarks': getattr(face, 'kps', None),
                            'det_size': getattr(face, 'det_size', None),
                            'enhancement': getattr(face, 'enhancement', 'original'),
                        }
                        detections_raw.append(det)

                    tracker.update(detections_raw, global_idx)

                    export_dets: List[Dict] = []
                    for det in detections_raw:
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

        # 处理剩余未满一个 batch 的采样帧
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
                        person_id, next_person_pid_local = self._map_identity_to_person(identity, person_map, next_person_pid)
                        next_person_pid = next_person_pid_local
                        det = {
                            'bbox': bbox,
                            'embedding': face.embedding,
                            'quality': quality,
                            'identity': identity,
                            'similarity': similarity,
                            'person_id': person_id,
                            'landmarks': getattr(face, 'kps', None),
                            'det_size': getattr(face, 'det_size', None),
                            'enhancement': getattr(face, 'enhancement', 'original'),
                        }
                        detections_raw.append(det)

                    tracker.update(detections_raw, global_idx)

                    export_dets: List[Dict] = []
                    for det in detections_raw:
                        ser = serialize_detection(det, frame_ref.shape)
                        try:
                            h, w = frame_ref.shape[:2]
                            if ser.get('bbox'):
                                x1, y1, x2, y2 = ser['bbox']
                                ser['bbox_norm'] = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
                                if ser.get('center'):
                                    cx, cy = ser['center']
                                    ser['center_norm'] = [round(cx / w, 4), round(cy / h, 4)]
                                else:
                                    ser['center_norm'] = None
                        except Exception:
                            pass
                        export_dets.append(ser)
                    frames_info.append(self._build_frame_record(global_idx, fps, export_dets))

            try:
                tracker.merge_similar_tracks(threshold=merge_similarity_threshold)
            except Exception:
                logger.debug("合并相似轨迹失败", exc_info=True)

        # 写出剩余所有帧（写出前绘制轨迹）
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
            out.release()

        remaining = [info['tracklet'] for info in tracker.tracks.values()]

        tracklets_out: List[Dict] = []
        for t in remaining:
            agg = t.aggregate()
            norm = float(np.linalg.norm(agg)) if agg is not None else None
            avg_q = float(np.mean(t.qualities)) if t.qualities else None
            ids: Dict[str, int] = {}
            for idv in t.identities:
                ids[idv] = ids.get(idv, 0) + 1
            cnt = len(t.frame_indices)
            reps: List[int] = []
            if cnt > 0:
                step = max(1, cnt // 3)
                reps = [t.frame_indices[i] for i in range(0, cnt, step)][:3]
            # 简化轨迹级别决策：使用轨迹内最高频身份（投票）
            resolved_identity = '未知'
            resolved_similarity = None
            try:
                # 找到投票最多的身份（排除"未知"）
                vote_candidates = {k: v for k, v in ids.items() if k and k != '未知'}
                if vote_candidates:
                    best_name, best_count = max(vote_candidates.items(), key=lambda x: x[1])
                    # 计算该身份在轨迹内的平均相似度
                    sims_for_name = [s for idn, s in zip(t.identities, t.similarities) if idn == best_name]
                    avg_sim = float(sum(sims_for_name) / len(sims_for_name)) if sims_for_name else 0.0
                    # 简单接受规则：票数 >= 阈值即接受
                    if best_count >= int(tracklet_min_votes):
                        resolved_identity = best_name
                        resolved_similarity = avg_sim
            except Exception:
                pass

            tracklets_out.append(
                {
                    'id': t.id,
                    'start_frame': t.frame_indices[0] if t.frame_indices else None,
                    'end_frame': t.frame_indices[-1] if t.frame_indices else None,
                    'frames_count': len(t.frame_indices),
                    'aggregated_embedding_norm': norm,
                    'avg_quality': avg_q,
                    'identities_freq': ids,
                    'representative_frames': reps,
                    'resolved_identity': resolved_identity,
                    'resolved_similarity': float(resolved_similarity) if resolved_similarity is not None else None,
                    # bbox history and representative bbox (absolute and normalized)
                    'bbox_history': [[int(x) for x in bb] for bb in t.bboxes],
                    'representative_bbox': (lambda bboxes: [int(x) for x in bboxes[len(bboxes)//2]] if bboxes else None)(t.bboxes),
                    'representative_bbox_norm': None,
                },
            )

        # compute normalized representative bbox for each tracklet (if video dims known)
        try:
            for to in tracklets_out:
                rep = to.get('representative_bbox')
                if rep is None:
                    to['representative_bbox_norm'] = None
                    continue
                x1, y1, x2, y2 = rep
                try:
                    to['representative_bbox_norm'] = [round(x1 / width, 4), round(y1 / height, 4), round(x2 / width, 4), round(y2 / height, 4)]
                except Exception:
                    to['representative_bbox_norm'] = None
        except Exception:
            pass

        occurrences: Dict[str, List[int]] = {}
        for fr in frames_info:
            frame_no = int(fr.get('frame', 0))
            for det in fr.get('detections', []):
                identity = det.get('identity', '未知')
                occurrences.setdefault(identity, []).append(frame_no)

        result = self._build_header_meta(
            input_video,
            output_video,
            fps,
            frame_interval,
            frame_interval_sec,
            frame_interval_frames,
            mode='tracklet',
        )
        result['occurrences'] = {k: [int(v) for v in vs] for k, vs in occurrences.items()}
        result['frames'] = frames_info
        result['tracklets'] = tracklets_out

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"处理完成（轨迹模式），输出视频: {output_video}, 输出数据: {output_json}")
        return result


def main():
    parser = argparse.ArgumentParser(description='视频人脸识别：按间隔抽帧并识别')
    parser.add_argument('input', help='输入视频文件路径')
    parser.add_argument('--output-video', '-o', help='输出带标注视频路径', default=None)
    parser.add_argument('--output-json', '-j', help='输出识别信息 JSON 路径', default='video_recognition_results.json')
    parser.add_argument('--gallery', '-g', help='图库路径', default='data/id_photos')
    parser.add_argument('--interval', '-i', type=float, help='抽帧间隔（秒），默认 2.0', default=2.0)
    parser.add_argument('--interval-frames', '-f', type=int, help='抽帧间隔（帧数），如果指定则优先使用', default=None)
    parser.add_argument('--mode', '-m', choices=['simple', 'tracklet'], default='simple', help='识别模式：simple 或 tracklet')
    parser.add_argument('--batch-frames', type=int, default=1, help='批量处理的帧数，>1 时启用多帧批识别')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='轨迹模式 IoU 阈值')
    parser.add_argument('--max-lost', type=int, default=5, help='轨迹模式允许的最大丢失帧数')
    parser.add_argument('--merge-sim-threshold', type=float, default=0.86, help='轨迹合并相似度阈值')
    parser.add_argument('--debug-identify', action='store_true', help='在识别时输出 top-k 相似度用于调试')
    parser.add_argument('--recognition-threshold', type=float, default=None, help='覆盖识别器的单帧相似度阈值（推荐 0.2）')
    parser.add_argument('--tracklet-min-votes', type=int, default=2, help='轨迹内最小票数阈值')

    args = parser.parse_args()

    vfr = VideoFaceRecognizer(gallery_path=args.gallery, debug_identify=args.debug_identify)
    # 可选：覆盖单帧识别阈值以提高召回
    if args.recognition_threshold is not None:
        try:
            vfr.recognizer.threshold = float(args.recognition_threshold)
            logger.info(f"已设置单帧识别阈值: {vfr.recognizer.threshold}")
        except Exception:
            logger.warning("设置 recognition_threshold 失败，忽略")
    if args.mode == 'tracklet':
        vfr.process_with_tracklets(
            args.input,
            args.output_video,
            args.output_json,
            frame_interval_sec=args.interval,
            frame_interval_frames=args.interval_frames,
            iou_threshold=args.iou_threshold,
            max_lost=args.max_lost,
            merge_similarity_threshold=args.merge_sim_threshold,
            tracklet_min_votes=args.tracklet_min_votes,
            batch_frames=args.batch_frames,
        )
    else:
        vfr.process(
            args.input,
            args.output_video,
            args.output_json,
            frame_interval_sec=args.interval,
            frame_interval_frames=args.interval_frames,
            batch_frames=args.batch_frames,
        )


if __name__ == '__main__':
    import time
    st = time.time()
    main()
    ed = time.time()
    logger.info(f"总耗时: {ed - st:.2f} 秒")
