from __future__ import annotations

from pathlib import Path
import sys

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


class _DummyActionModel:
    def __init__(self, *args, **kwargs) -> None:
        # Make mapping deterministic: include names that match behavior regex rules.
        self.categories = [
            "writing",
            "reading",
            "talking",
            "talking on cell phone",
            "typing",
        ]

    def predict_proba(self, clip_rgb_uint8, *, topk: int = 10):
        # Always strongly predict "writing".
        scores = {c: 0.01 for c in self.categories}
        scores["writing"] = 0.95
        top = [("writing", 0.95)]

        def _top_obj():
            return type("Top", (), {"categories": top})

        if isinstance(clip_rgb_uint8, (list, tuple)):
            n = len(list(clip_rgb_uint8))
            return [dict(scores) for _ in range(n)], [_top_obj() for _ in range(n)]

        return scores, _top_obj()


class _DummyPersonDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect_persons(self, frame_bgr, conf: float = 0.25):
        # Return a single person bbox that covers the face bbox used in the test.
        from src.behavior.person_detector import PersonDet

        return [PersonDet(bbox=[0, 0, 640, 480], conf=0.99)]


def test_behavior_stats_generation_with_dummy_action_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    video_dir = repo_root / "data" / "video"
    if not video_dir.exists():
        pytest.skip("data/video not found")

    try:
        video_path = _pick_video(video_dir)
    except FileNotFoundError as e:
        pytest.skip(str(e))

    # Patch action model to avoid downloading torchvision weights during tests.
    import src.behavior.pipeline as behavior_pipeline

    monkeypatch.setattr(behavior_pipeline, "TorchvisionVideoActionModel", _DummyActionModel)
    monkeypatch.setattr(behavior_pipeline, "UltralyticsPersonDetector", _DummyPersonDetector)

    from src.behavior.pipeline import BehaviorPipelineConfig, run_behavior_pipeline_on_result

    # Synthetic recognition result: one student is always observed and locked.
    fps = 25.0
    frames = []
    # 0.0s, 0.6s, 1.2s => segment duration > min_duration_seconds.
    for frame_idx in (0, 15, 30):
        frames.append(
            {
                "frame": int(frame_idx),
                "timestamp": float(frame_idx / fps),
                "ts_str": "",
                "detections": [
                    {
                        "bbox": [10, 10, 110, 110],
                        "quality": 0.9,
                        "track_display_identity": "张泽宇",
                        "track_display_similarity": 0.9,
                        "track_is_locked": True,
                    }
                ],
            }
        )

    result = {
        "fps": fps,
        "used_frame_interval": 15,
        "frames": frames,
    }

    cfg = BehaviorPipelineConfig(
        enabled=True,
        target_names=["张泽宇"],
        action_model_name="swin3d_t",
        clip_seconds=1.0,
        clip_num_frames=4,
    )

    behavior_stats = run_behavior_pipeline_on_result(input_video=str(video_path), result=result, cfg=cfg)

    assert isinstance(behavior_stats, dict)
    assert behavior_stats.get("behavior_schema_version") == "v1"
    assert behavior_stats.get("denominator", {}).get("type") == "on_screen_seconds"

    by_student = behavior_stats.get("by_student") or {}
    assert "张泽宇" in by_student

    stu = by_student["张泽宇"]
    assert float(stu.get("total_observed_seconds", 0.0)) > 0.0

    behaviors = stu.get("behaviors") or {}
    assert "writing" in behaviors
    writing = behaviors["writing"]
    assert float(writing.get("total_seconds", 0.0)) > 0.0
    assert isinstance(writing.get("segments"), list)
