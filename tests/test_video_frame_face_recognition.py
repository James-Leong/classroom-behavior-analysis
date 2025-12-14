from __future__ import annotations

from pathlib import Path

import sys
import cv2
import pytest

# Ensure repo root is on sys.path so tests can import top-level modules like `face_recognizer`.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.face.recognizer import FaceRecognizer


def _pick_video(video_dir: Path) -> Path:
    # Prefer a deterministic file name if present.
    preferred = video_dir / "20251115_clip.mp4"
    if preferred.exists():
        return preferred

    # Otherwise pick the first common video extension.
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(video_dir.glob(ext))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No video found under {video_dir}")


def _read_frame(video_path: Path, frame_index: int | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_index is None:
            # Prefer a mid-frame to avoid title/black frames.
            frame_index = total // 2 if total > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fallback: try first frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a frame from {video_path}")

        return frame
    finally:
        cap.release()


@pytest.mark.parametrize("frame_index", [None])
def test_face_recognition_on_video_frame(frame_index: int | None, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "data" / "video"
    gallery_dir = repo_root / "data" / "id_photo"

    if not video_dir.exists():
        pytest.skip("data/video not found")
    if not gallery_dir.exists():
        pytest.skip("data/id_photo (gallery) not found")

    try:
        video_path = _pick_video(video_dir)
    except FileNotFoundError as e:
        pytest.skip(str(e))

    frame = _read_frame(video_path, frame_index=frame_index)
    assert frame is not None
    assert frame.ndim == 3  # BGR

    # Use CPU-friendly defaults; avoid forcing rebuild for test speed.
    recognizer = FaceRecognizer(
        gallery_path=str(gallery_dir),
        device="auto",
        rebuild_gallery=False,
        det_size=640,
        threshold=0.40,
        quality_threshold=0.60,
    )

    faces = recognizer.detect_faces(frame)
    assert isinstance(faces, list)

    # 可视化：在帧上绘制识别结果，保存到 outputs/ 便于人工检查
    vis = frame.copy()
    results = []

    # This test is intentionally non-flaky: classroom frame may have 0 faces depending on the chosen frame.
    # If faces exist, validate bbox sanity and run recognition without exceptions.
    h, w = frame.shape[:2]
    for face in faces:
        bbox = getattr(face, "bbox", None)
        assert bbox is not None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h

        quality = recognizer.assess_face_quality(face, frame.shape)
        identity, sim = recognizer.recognize_identity(face.embedding, quality, debug=False)
        assert isinstance(identity, str)
        assert isinstance(sim, float)

        result = {
            "bbox": (x1, y1, x2, y2),
            "identity": identity,
            "similarity": float(sim),
            "quality": float(quality),
            "landmarks": getattr(face, "kps", None),
            "det_size": getattr(face, "det_size", None),
            "enhancement": getattr(face, "enhancement", "unknown"),
        }
        results.append(result)
        recognizer._draw(vis, result)

    # 保存输出图片（即使没有检测到人脸，也保存原帧，便于检查帧是否合理）
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_repo = outputs_dir / f"pytest_face_recognition_{video_path.stem}_frame.jpg"
    out_tmp = tmp_path / out_repo.name

    ok1 = cv2.imwrite(str(out_repo), vis)
    ok2 = cv2.imwrite(str(out_tmp), vis)
    assert ok1 and ok2, f"Failed to write output images: {out_repo}, {out_tmp}"

    # 方便在 pytest 输出里看到文件位置
    print(f"Saved visualization to: {out_repo}")
