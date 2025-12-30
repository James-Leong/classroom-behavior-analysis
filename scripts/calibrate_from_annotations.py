#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    video_path: str
    frame: int
    gt_fine: str
    pred_fine: str
    gt_coarse: Optional[str]
    pred_coarse: Optional[str]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _confusion_add(
    conf: Dict[str, Dict[str, int]],
    y_true: str,
    y_pred: str,
) -> None:
    conf.setdefault(y_true, {})
    conf[y_true][y_pred] = int(conf[y_true].get(y_pred, 0)) + 1


def _confusion_to_table(conf: Mapping[str, Mapping[str, int]]) -> Tuple[List[str], List[List[int]]]:
    labels: List[str] = sorted({*conf.keys(), *{p for row in conf.values() for p in row.keys()}})
    mat: List[List[int]] = []
    for t in labels:
        row = []
        for p in labels:
            row.append(int(conf.get(t, {}).get(p, 0)))
        mat.append(row)
    return labels, mat


def _metrics_from_confusion(conf: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    labels, mat = _confusion_to_table(conf)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    total = sum(sum(r) for r in mat)
    correct = sum(mat[i][i] for i in range(len(labels))) if labels else 0

    per_label: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        i = idx[lbl]
        tp = float(mat[i][i])
        fp = float(sum(mat[r][i] for r in range(len(labels))) - mat[i][i])
        fn = float(sum(mat[i][c] for c in range(len(labels))) - mat[i][i])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": float(tp + fn)}

    macro_f1 = sum(v["f1"] for v in per_label.values()) / float(len(per_label) or 1)
    accuracy = float(correct) / float(total or 1)
    return {
        "labels": labels,
        "total": int(total),
        "correct": int(correct),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion": {t: dict(p) for t, p in conf.items()},
    }


def _default_fine_to_coarse() -> Dict[str, str]:
    return {
        "listening_upright": "listening",
        "on_task_head_down": "reading_or_writing",
        "using_device": "using_device",
        "off_task": "distracted",
        "listening": "listening",
        "reading_or_writing": "reading_or_writing",
        "distracted": "distracted",
        "other": "other",
    }


def _map_fine_to_coarse(label: str, mapping: Mapping[str, str], allowed: Optional[Sequence[str]]) -> Optional[str]:
    s = str(label or "").strip()
    if not s:
        return None
    coarse = mapping.get(s, s)
    if allowed is None:
        return coarse
    return coarse if coarse in set(allowed) else None


def _predict_clip_coarse(
    model: Any,
    cap: Any,
    *,
    frame: int,
    crop_bbox_xyxy: List[int],
    fps: float,
    max_frame: Optional[int],
    clip_seconds: float,
    num_frames: int,
) -> Tuple[str, Dict[str, float]]:
    from src.behavior.video_clip import crop_clip, read_frames_by_index, sample_frame_indices

    indices = sample_frame_indices(
        center_frame=int(frame),
        fps=float(fps),
        window_seconds=float(clip_seconds),
        num_frames=int(num_frames),
        max_frame=max_frame,
    )
    frames_bgr = read_frames_by_index(cap, indices)
    clip_rgb = crop_clip(frames_bgr, crop_bbox_xyxy)
    scores, _top = model.predict_proba(clip_rgb, topk=0)
    if not scores:
        return "other", {}
    pred = max(scores.items(), key=lambda kv: float(kv[1]))[0]
    return str(pred), {str(k): float(v) for k, v in scores.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/annotations/manifest.jsonl")
    p.add_argument("--out", default="outputs/calibration_report.json")
    p.add_argument("--mode", default="heuristic", choices=["heuristic", "clip", "both"])
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--only-locked", action="store_true", default=False)

    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--device", default="auto")
    p.add_argument("--clip-seconds", type=float, default=2.0)
    p.add_argument("--clip-num-frames", type=int, default=16)
    p.add_argument("--fine-to-coarse", default="")
    p.add_argument("--save-samples", action="store_true", default=False)
    args = p.parse_args()

    manifest_path = (
        (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    )
    out_path = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fine_to_coarse = _default_fine_to_coarse()
    if str(args.fine_to_coarse or "").strip():
        fine_to_coarse.update(json.loads(args.fine_to_coarse))

    rows = list(_iter_jsonl(manifest_path))
    if args.max_samples and int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    filtered: List[Dict[str, Any]] = []
    for r in rows:
        if args.only_locked and not bool(r.get("track_is_locked", False)):
            continue
        gt = str(r.get("annotator_label", "") or "").strip()
        pred = str(r.get("target_label", "") or "").strip()
        if not gt or not pred:
            continue
        filtered.append(r)

    report: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "num_rows": int(len(rows)),
        "num_used": int(len(filtered)),
        "mode": str(args.mode),
    }

    heuristic_conf: Dict[str, Dict[str, int]] = {}
    for r in filtered:
        gt = str(r.get("annotator_label", "") or "").strip()
        pred = str(r.get("target_label", "") or "").strip()
        _confusion_add(heuristic_conf, gt, pred)
    report["heuristic_fine"] = _metrics_from_confusion(heuristic_conf)

    if args.mode in {"clip", "both"}:
        import cv2

        from src.behavior.action_model_clip import CLIPVideoActionModel

        model = CLIPVideoActionModel(model_name=str(args.clip_model), device=str(args.device))
        clip_conf: Dict[str, Dict[str, int]] = {}
        errors = 0
        skipped = 0
        score_aggr: Dict[str, Counter] = defaultdict(Counter)

        by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in filtered:
            vp = str(r.get("video_path", "") or "").strip()
            if not vp:
                skipped += 1
                continue
            by_video[vp].append(r)

        samples_out: List[Dict[str, Any]] = []
        for video_path, items in by_video.items():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                errors += len(items)
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
            max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

            for r in items:
                try:
                    bbox = r.get("crop_bbox_xyxy")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        skipped += 1
                        continue

                    pred_coarse, scores = _predict_clip_coarse(
                        model,
                        cap,
                        frame=int(r.get("frame", 0) or 0),
                        crop_bbox_xyxy=[int(x) for x in bbox],
                        fps=fps,
                        max_frame=max_frame,
                        clip_seconds=float(args.clip_seconds),
                        num_frames=int(args.clip_num_frames),
                    )
                    gt_fine = str(r.get("annotator_label", "") or "").strip()
                    gt_coarse = _map_fine_to_coarse(gt_fine, fine_to_coarse, model.labels)
                    if gt_coarse is None:
                        skipped += 1
                        continue

                    _confusion_add(clip_conf, gt_coarse, pred_coarse)
                    for k, v in scores.items():
                        score_aggr[gt_coarse][k] += float(v)

                    if args.save_samples:
                        samples_out.append(
                            SampleResult(
                                sample_id=str(r.get("sample_id", "") or ""),
                                video_path=str(video_path),
                                frame=int(r.get("frame", 0) or 0),
                                gt_fine=gt_fine,
                                pred_fine=str(r.get("target_label", "") or "").strip(),
                                gt_coarse=str(gt_coarse),
                                pred_coarse=str(pred_coarse),
                            ).__dict__
                        )
                except Exception:
                    errors += 1
                    continue

            cap.release()

        per_gt_mean_scores: Dict[str, Dict[str, float]] = {}
        for gt, c in score_aggr.items():
            denom = float(sum(c.values()) or 1.0)
            per_gt_mean_scores[gt] = {k: float(v) / denom for k, v in c.items()}

        report["clip_coarse"] = _metrics_from_confusion(clip_conf)
        report["clip_coarse"]["errors"] = int(errors)
        report["clip_coarse"]["skipped"] = int(skipped)
        report["clip_coarse"]["fine_to_coarse"] = dict(fine_to_coarse)
        report["clip_coarse"]["per_gt_mean_scores"] = per_gt_mean_scores
        if args.save_samples:
            report["samples"] = samples_out

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: report[k] for k in report.keys() if k in {"mode", "num_rows", "num_used"}}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
