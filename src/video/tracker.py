from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0


def _bbox_center_xyxy(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def _bbox_diag_xyxy(b: List[int]) -> float:
    x1, y1, x2, y2 = b
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return float((w * w + h * h) ** 0.5)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


@dataclass
class Tracklet:
    id: int
    frame_indices: List[int] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    identities: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)

    # Online aggregated embedding (EMA/weighted mean) for multi-frame reinforcement.
    agg_embedding: Optional[np.ndarray] = None
    agg_weight: float = 0.0

    def add(
        self, frame_idx: int, bbox: List[int], embedding: np.ndarray, quality: float, identity: str, similarity: float
    ):
        self.frame_indices.append(int(frame_idx))
        self.bboxes.append([int(x) for x in bbox])
        self.embeddings.append(np.array(embedding))
        self.qualities.append(float(quality))
        self.identities.append(str(identity))
        self.similarities.append(float(similarity))

        # Update online aggregated embedding.
        try:
            emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(emb))
            if n > 1e-12:
                emb = emb / n
            w = float(max(0.05, min(1.0, quality)))
            if self.agg_embedding is None or self.agg_weight <= 0.0:
                self.agg_embedding = emb
                self.agg_weight = w
            else:
                agg = (self.agg_embedding * float(self.agg_weight) + emb * w) / (float(self.agg_weight) + w)
                n2 = float(np.linalg.norm(agg))
                if n2 > 1e-12:
                    agg = agg / n2
                self.agg_embedding = agg
                self.agg_weight = float(self.agg_weight) + w
        except Exception:
            # Keep aggregation best-effort; do not break tracking.
            pass

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
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 5,
        appearance_threshold: float = 0.72,
        max_center_dist_ratio: float = 1.10,
        motion_alpha: float = 0.75,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_lost = int(max_lost)
        self.appearance_threshold = float(appearance_threshold)
        self.max_center_dist_ratio = float(max_center_dist_ratio)
        self.motion_alpha = float(motion_alpha)
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.person_to_track: Dict[int, int] = {}

    def _predict_bbox(self, info: Dict) -> Optional[List[int]]:
        lb = info.get("last_bbox")
        if not lb:
            return None
        vx, vy = info.get("vel_xy", (0.0, 0.0))
        try:
            x1, y1, x2, y2 = [float(x) for x in lb]
            return [int(round(x1 + vx)), int(round(y1 + vy)), int(round(x2 + vx)), int(round(y2 + vy))]
        except Exception:
            return None

    def _update_motion(self, info: Dict, new_bbox: List[int]) -> None:
        try:
            old_center = info.get("last_center")
            new_center = _bbox_center_xyxy(new_bbox)
            if old_center is None:
                info["last_center"] = new_center
                info["vel_xy"] = (0.0, 0.0)
                info["motion_norm_ema"] = 0.0
                return
            dx = float(new_center[0] - float(old_center[0]))
            dy = float(new_center[1] - float(old_center[1]))
            ovx, ovy = info.get("vel_xy", (0.0, 0.0))
            a = float(self.motion_alpha)
            info["vel_xy"] = (a * dx + (1.0 - a) * float(ovx), a * dy + (1.0 - a) * float(ovy))
            info["last_center"] = new_center

            # Motion stability (normalized by bbox diagonal). Smaller -> more stable.
            try:
                diag = _bbox_diag_xyxy(info.get("last_bbox") or new_bbox)
                step = float((dx * dx + dy * dy) ** 0.5) / max(1.0, float(diag))
                prev = float(info.get("motion_norm_ema", 0.0) or 0.0)
                # EMA with a fairly slow update, since videos are noisy.
                info["motion_norm_ema"] = 0.85 * prev + 0.15 * float(step)
            except Exception:
                pass
        except Exception:
            pass

    def update(self, detections: List[Dict], frame_idx: int) -> List[Tracklet]:
        assigned = set()
        det_bboxes = [d["bbox"] for d in detections]

        det_embs: List[Optional[np.ndarray]] = []
        for d in detections:
            e = d.get("embedding")
            det_embs.append(np.asarray(e, dtype=np.float32).reshape(-1) if e is not None else None)

        for tid, info in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            best_app = 0.0
            for j, db in enumerate(det_bboxes):
                if j in assigned:
                    continue

                try:
                    score_last = _iou(info["last_bbox"], db)
                except Exception:
                    score_last = 0.0
                pred = self._predict_bbox(info)
                try:
                    score_pred = _iou(pred, db) if pred else 0.0
                except Exception:
                    score_pred = 0.0
                score = max(score_last, score_pred)

                app = 0.0
                try:
                    t = info.get("tracklet")
                    agg = getattr(t, "agg_embedding", None) if t is not None else None
                    de = det_embs[j]
                    if agg is not None and de is not None:
                        app = _cos_sim(agg, de)
                except Exception:
                    app = 0.0

                ok_center = True
                try:
                    lb = info.get("last_bbox")
                    diag = _bbox_diag_xyxy(lb) if lb else 1.0
                    pred_center = _bbox_center_xyxy(pred if pred else lb)
                    det_center = _bbox_center_xyxy(db)
                    dist = float(np.linalg.norm(np.asarray(pred_center) - np.asarray(det_center)))
                    ok_center = (dist / max(1.0, diag)) <= float(self.max_center_dist_ratio)
                except Exception:
                    ok_center = True

                if not ok_center:
                    continue

                if score > best_iou or (score == best_iou and app > best_app):
                    best_iou = score
                    best_app = app
                    best_j = j

            accept = False
            if best_j >= 0:
                if best_iou >= float(self.iou_threshold):
                    accept = True
                elif best_app >= float(self.appearance_threshold):
                    accept = True

            if best_j >= 0 and accept:
                d = detections[best_j]
                d["track_id"] = tid
                info["tracklet"].add(
                    frame_idx, d["bbox"], d["embedding"], d["quality"], d["identity"], d["similarity"]
                )
                info["last_bbox"] = d["bbox"]
                self._update_motion(info, d["bbox"])
                info["last_seen"] = int(frame_idx)
                if d.get("person_id") is not None:
                    info["person_id"] = d.get("person_id")
                    self.person_to_track[d["person_id"]] = tid
                info["lost"] = 0
                assigned.add(best_j)
            else:
                info["lost"] += 1

        for j, d in enumerate(detections):
            if j in assigned:
                continue
            existing_tid = None
            pid = d.get("person_id")
            if pid is not None:
                mapped = self.person_to_track.get(pid)
                if mapped is not None and mapped in self.tracks:
                    existing_tid = mapped

            if existing_tid is not None:
                tid = existing_tid
                info = self.tracks[tid]
                d["track_id"] = tid
                info["tracklet"].add(
                    frame_idx, d["bbox"], d["embedding"], d["quality"], d["identity"], d["similarity"]
                )
                info["last_bbox"] = d["bbox"]
                self._update_motion(info, d["bbox"])
                info["last_seen"] = int(frame_idx)
                info["person_id"] = pid
                info["lost"] = 0
                assigned.add(j)
                continue

            tid = self.next_id
            self.next_id += 1
            t = Tracklet(id=tid)
            d["track_id"] = tid
            t.add(frame_idx, d["bbox"], d["embedding"], d["quality"], d["identity"], d["similarity"])
            self.tracks[tid] = {
                "tracklet": t,
                "last_bbox": d["bbox"],
                "lost": 0,
                "last_seen": int(frame_idx),
                "person_id": d.get("person_id"),
                "last_center": _bbox_center_xyxy(d["bbox"]),
                "vel_xy": (0.0, 0.0),
                # identity hysteresis state
                "lock_evidence": 0,
                "switch_evidence": 0,
                "unknown_streak": 0,
                "locked_identity": None,
                "locked_similarity": 0.0,
                "display_identity": None,
                "display_similarity": 0.0,
                "is_locked": False,
            }
            if pid is not None:
                self.person_to_track[pid] = tid

        finished = []
        remove_keys = []
        for tid, info in list(self.tracks.items()):
            if info["lost"] > self.max_lost:
                finished.append(info["tracklet"])
                remove_keys.append(tid)

        for k in remove_keys:
            pid = self.tracks[k].get("person_id")
            if pid is not None and self.person_to_track.get(pid) == k:
                del self.person_to_track[pid]
            del self.tracks[k]

        active = [info["tracklet"] for info in self.tracks.values()]
        return active + finished

    def merge_similar_tracks(self, threshold: float = 0.86):
        tids = list(self.tracks.keys())
        if len(tids) < 2:
            return

        emb_map = {}
        for tid in tids:
            info = self.tracks.get(tid)
            if not info:
                continue
            t = info.get("tracklet")
            if t is None:
                continue
            # 关键优化：使用在线聚合 embedding，避免每次 merge 都对全历史 embeddings 做 np.stack。
            agg = getattr(t, "agg_embedding", None)
            if agg is None:
                try:
                    if getattr(t, "embeddings", None):
                        agg = np.asarray(t.embeddings[-1], dtype=np.float32).reshape(-1)
                except Exception:
                    agg = None
            if agg is None:
                continue
            emb_map[tid] = np.asarray(agg, dtype=np.float32).reshape(-1)

        merged = set()
        for i in range(len(tids)):
            a = tids[i]
            if a not in self.tracks or a in merged:
                continue
            for j in range(i + 1, len(tids)):
                b = tids[j]
                if b not in self.tracks or b in merged:
                    continue
                pa = self.tracks[a].get("person_id")
                pb = self.tracks[b].get("person_id")
                if pa is not None and pb is not None and pa != pb:
                    continue
                # 若两条轨迹都已显示为“已知”且不同姓名，直接跳过（减少无意义比较）
                try:
                    da = self.tracks[a].get("display_identity")
                    db = self.tracks[b].get("display_identity")
                    if da and db and da != "未知" and db != "未知" and da != db:
                        continue
                except Exception:
                    pass
                ea = emb_map.get(a)
                eb = emb_map.get(b)
                if ea is None or eb is None:
                    continue
                sim = _cos_sim(ea, eb)
                if sim >= threshold:
                    last_seen_a = int(self.tracks[a].get("last_seen", -1))
                    last_seen_b = int(self.tracks[b].get("last_seen", -1))
                    if last_seen_a >= last_seen_b:
                        keep, rem = a, b
                    else:
                        keep, rem = b, a
                    if keep not in self.tracks or rem not in self.tracks:
                        continue
                    kinfo = self.tracks[keep]
                    rinfo = self.tracks[rem]
                    kt = kinfo["tracklet"]
                    rt = rinfo["tracklet"]
                    kt.frame_indices.extend(rt.frame_indices)
                    kt.bboxes.extend(rt.bboxes)
                    kt.embeddings.extend(rt.embeddings)
                    kt.qualities.extend(rt.qualities)
                    kt.identities.extend(rt.identities)
                    kt.similarities.extend(rt.similarities)

                    # 合并在线聚合 embedding 状态（用于后续 update/merge 更快）
                    try:
                        ka = getattr(kt, "agg_embedding", None)
                        ra = getattr(rt, "agg_embedding", None)
                        kw = float(getattr(kt, "agg_weight", 0.0) or 0.0)
                        rw = float(getattr(rt, "agg_weight", 0.0) or 0.0)
                        if ka is None and ra is not None:
                            kt.agg_embedding = np.asarray(ra, dtype=np.float32).reshape(-1)
                            kt.agg_weight = max(0.0, rw)
                        elif ka is not None and ra is not None:
                            ka2 = np.asarray(ka, dtype=np.float32).reshape(-1)
                            ra2 = np.asarray(ra, dtype=np.float32).reshape(-1)
                            denom = max(1e-6, kw + rw)
                            merged_emb = (ka2 * kw + ra2 * rw) / denom
                            n2 = float(np.linalg.norm(merged_emb))
                            if n2 > 1e-12:
                                merged_emb = merged_emb / n2
                            kt.agg_embedding = merged_emb
                            kt.agg_weight = float(kw + rw)
                    except Exception:
                        pass
                    kinfo["last_bbox"] = rinfo.get("last_bbox") or kinfo.get("last_bbox")
                    kinfo["last_seen"] = max(int(kinfo.get("last_seen", -1)), int(rinfo.get("last_seen", -1)))
                    if rinfo.get("person_id") is not None:
                        kinfo["person_id"] = rinfo.get("person_id")
                        self.person_to_track[rinfo["person_id"]] = keep
                    if "smoothed_bbox" in rinfo:
                        kinfo["smoothed_bbox"] = rinfo["smoothed_bbox"]
                    if kinfo.get("locked_identity") is None and rinfo.get("locked_identity") is not None:
                        kinfo["locked_identity"] = rinfo.get("locked_identity")
                        kinfo["locked_similarity"] = float(rinfo.get("locked_similarity", 0.0) or 0.0)
                        kinfo["is_locked"] = bool(rinfo.get("is_locked", False))
                        kinfo["display_identity"] = rinfo.get("display_identity")
                        kinfo["display_similarity"] = float(rinfo.get("display_similarity", 0.0) or 0.0)
                    try:
                        del self.tracks[rem]
                        merged.add(rem)
                    except KeyError:
                        pass
