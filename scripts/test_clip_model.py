#!/usr/bin/env python
"""Quick test script for CLIP action model."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.behavior.action_model_clip import CLIPVideoActionModel


def test_clip_model():
    """Test CLIP model initialization and inference."""
    print("=== Testing CLIP Action Model ===\n")

    # 1. Test model initialization
    print("1. Initializing CLIP model (ViT-B/32)...")
    model = CLIPVideoActionModel(
        model_name="ViT-B/32",
        device="auto",
        frame_subsample=4,
    )
    print(f"   ✓ Model loaded on device: {model.device}")
    print(f"   ✓ Behaviors: {len(model.behaviors)}")
    print(f"   ✓ Labels: {model.labels}\n")

    # 2. Test single clip inference
    print("2. Testing single clip inference...")
    # Create dummy clip: 16 frames, 224x224, RGB
    clip = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    scores, top = model.predict_proba(clip, topk=5)

    print("   ✓ Inference successful")
    print("   ✓ Top-5 predictions:")
    for label, prob in top.categories:
        print(f"      - {label}: {prob:.3f}")
    print()

    # 3. Test batch inference
    print("3. Testing batch inference...")
    clips = [clip, clip, clip]  # 3 clips
    scores_list, tops_list = model.predict_proba(clips, topk=3)
    print("   ✓ Batch inference successful")
    print(f"   ✓ Processed {len(scores_list)} clips")
    print("   ✓ First clip top-3:")
    for label, prob in tops_list[0].categories:
        print(f"      - {label}: {prob:.3f}")
    print()

    print("=== All Tests Passed! ===")
    print("\nYou can now use CLIP for behavior recognition:")
    print("  python video_recognizer.py data/video/test.mp4 \\")
    print("    --behavior --behavior-model-type clip \\")
    print("    --interval-frames 10 --max-seconds 15")


if __name__ == "__main__":
    try:
        test_clip_model()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
