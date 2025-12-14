from typing import Dict, List, Optional, Tuple

import numpy as np


def serialize_detection(det: Dict, frame_shape: Optional[Tuple[int, int]] = None) -> Dict:
    """Serialize a detection dict into JSON-safe form and optionally add normalized coords.

    det: dict with keys like 'bbox', 'embedding', 'similarity', 'quality', 'identity', 'person_id', 'track_id'
    frame_shape: (h, w)
    """
    ed = dict(det)
    if "bbox" in ed and ed["bbox"] is not None:
        try:
            ed["bbox"] = [int(x) for x in ed["bbox"]]
        except Exception:
            ed["bbox"] = None
        try:
            x1, y1, x2, y2 = ed["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            ed["center"] = [cx, cy]
        except Exception:
            ed["center"] = None

    if "similarity" in ed:
        try:
            ed["similarity"] = float(ed["similarity"])
        except Exception:
            pass
    if "quality" in ed:
        try:
            ed["quality"] = float(ed["quality"])
        except Exception:
            pass

    if "landmarks" in ed:
        ed.pop("landmarks", None)

    if "person_id" in ed and ed["person_id"] is not None:
        try:
            ed["person_id"] = int(ed["person_id"])
        except Exception:
            ed["person_id"] = None
    if "track_id" in ed and ed["track_id"] is not None:
        try:
            ed["track_id"] = int(ed["track_id"])
        except Exception:
            ed["track_id"] = None

    emb = ed.get("embedding", None)
    if emb is not None:
        try:
            ed["embedding_norm"] = float(np.linalg.norm(emb))
        except Exception:
            ed["embedding_norm"] = None
        ed.pop("embedding", None)

    if frame_shape is not None and ed.get("bbox"):
        try:
            h, w = frame_shape[0], frame_shape[1]
            x1, y1, x2, y2 = ed["bbox"]
            ed["bbox_norm"] = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
            if ed.get("center"):
                cx, cy = ed["center"]
                ed["center_norm"] = [round(cx / w, 4), round(cy / h, 4)]
            else:
                ed["center_norm"] = None
        except Exception:
            ed["bbox_norm"] = None
            ed["center_norm"] = None

    return ed


def serialize_tracklet(t, video_size: Optional[Tuple[int, int]] = None, tracklet_min_votes: int = 2) -> Dict:
    """Serialize a Tracklet (dataclass) into dict for JSON output.

    t: Tracklet-like object with attributes id, frame_indices, bboxes, embeddings, qualities, identities, similarities
    video_size: (width, height)
    """
    agg = None
    try:
        agg = np.array(t.aggregate(), dtype=float) if getattr(t, "aggregate", None) else None
    except Exception:
        agg = None
    norm = float(np.linalg.norm(agg)) if agg is not None else None
    avg_q = float(np.mean(t.qualities)) if getattr(t, "qualities", None) else None
    ids: Dict[str, int] = {}
    for idv in getattr(t, "identities", []) or []:
        ids[idv] = ids.get(idv, 0) + 1
    cnt = len(getattr(t, "frame_indices", []) or [])
    reps: List[int] = []
    if cnt > 0:
        step = max(1, cnt // 3)
        reps = [t.frame_indices[i] for i in range(0, cnt, step)][:3]

    resolved_identity = "未知"
    resolved_similarity = None
    try:
        vote_candidates = {k: v for k, v in ids.items() if k and k != "未知"}
        if vote_candidates:
            best_name, best_count = max(vote_candidates.items(), key=lambda x: x[1])
            sims_for_name = [s for idn, s in zip(t.identities, t.similarities) if idn == best_name]
            avg_sim = float(sum(sims_for_name) / len(sims_for_name)) if sims_for_name else 0.0
            if best_count >= int(tracklet_min_votes):
                resolved_identity = best_name
                resolved_similarity = avg_sim
    except Exception:
        pass

    rep_bbox = (lambda bboxes: [int(x) for x in bboxes[len(bboxes) // 2]] if bboxes else None)(
        getattr(t, "bboxes", None)
    )
    rep_bbox_norm = None
    try:
        if video_size and rep_bbox:
            w = int(video_size[0])
            h = int(video_size[1])
            x1, y1, x2, y2 = rep_bbox
            rep_bbox_norm = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
    except Exception:
        rep_bbox_norm = None

    return {
        "id": int(getattr(t, "id", -1)),
        "start_frame": t.frame_indices[0] if t.frame_indices else None,
        "end_frame": t.frame_indices[-1] if t.frame_indices else None,
        "frames_count": len(t.frame_indices),
        "aggregated_embedding_norm": norm,
        "avg_quality": avg_q,
        "identities_freq": ids,
        "representative_frames": reps,
        "resolved_identity": resolved_identity,
        "resolved_similarity": float(resolved_similarity) if resolved_similarity is not None else None,
        "bbox_history": [[int(x) for x in bb] for bb in (getattr(t, "bboxes", []) or [])],
        "representative_bbox": rep_bbox,
        "representative_bbox_norm": rep_bbox_norm,
    }
