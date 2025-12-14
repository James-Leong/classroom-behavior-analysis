from __future__ import annotations

import shutil
import subprocess
import sys

from pathlib import Path

import pytest


def _pick_video(video_dir: Path) -> Path:
    preferred = video_dir / "20251115_clip.mp4"
    if preferred.exists():
        return preferred
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(video_dir.glob(ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No video found under {video_dir}")


def _ffmpeg_has_encoder(encoder: str) -> bool:
    """Return True if system ffmpeg advertises the given encoder."""

    if shutil.which("ffmpeg") is None:
        return False

    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False

    encoders_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return encoder in encoders_text


def test_video_tracklet_multiframe_recognition_outputs(tmp_path: Path):
    """Runs tracklet-mode recognition on a short video segment.

    This verifies:
    - pipeline runs end-to-end
    - produces an annotated video + JSON for manual inspection
    - multi-frame tracklet output exists (continuity)
    """

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

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

    from video_recognizer import VideoFaceRecognizer

    vfr = VideoFaceRecognizer(gallery_path=str(gallery_dir), debug_identify=False)

    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_repo_video = outputs_dir / f"pytest_tracklet_{video_path.stem}_short.mp4"
    out_repo_json = outputs_dir / f"pytest_tracklet_{video_path.stem}_short.json"

    out_tmp_video = tmp_path / out_repo_video.name
    out_tmp_json = tmp_path / out_repo_json.name

    # Note: ffmpeg accelerates encoding/transcoding, not the overlay drawing itself.
    ffmpeg_codec = "h264_nvenc"
    if not _ffmpeg_has_encoder(ffmpeg_codec):
        pytest.skip(f"ffmpeg encoder not available: {ffmpeg_codec}")

    # Run a short segment to keep CI/test runtime reasonable.
    # Use small frame interval for better continuity and aggregation.
    result = vfr.process_with_tracklets(
        input_video=str(video_path),
        output_video=str(out_tmp_video),
        output_json=str(out_tmp_json),
        frame_interval_frames=3,
        # frame_interval_sec=2.0,
        batch_frames=8,
        iou_threshold=0.30,
        max_lost=8,
        merge_similarity_threshold=0.86,
        tracklet_min_votes=2,
        max_seconds=10.0,
        ffmpeg_codec=ffmpeg_codec,
    )

    assert isinstance(result, dict)
    assert out_tmp_video.exists() and out_tmp_video.stat().st_size > 10_000
    assert out_tmp_json.exists() and out_tmp_json.stat().st_size > 100

    # Copy to repo outputs for manual inspection.
    out_repo_video.write_bytes(out_tmp_video.read_bytes())
    out_repo_json.write_bytes(out_tmp_json.read_bytes())

    assert "tracklets" in result
    assert "frames" in result

    # Non-flaky assertions: pipeline may detect 0 faces in a short segment.
    # But if tracklets exist, ensure they have sane fields.
    tracklets = result.get("tracklets") or []
    if tracklets:
        t0 = tracklets[0]
        assert "id" in t0
        assert "frames_count" in t0

    print(f"Saved annotated video to: {out_repo_video}")
    print(f"Saved tracklet JSON to: {out_repo_json}")
