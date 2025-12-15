#!/bin/bash
# Compare Kinetics vs CLIP behavior recognition performance

set -e

VIDEO="data/video/20251115_clip.mp4"
MAX_SEC=15
INTERVAL=10

echo "=== Behavior Recognition Comparison Test ==="
echo "Video: $VIDEO"
echo "Duration: ${MAX_SEC}s"
echo "Interval: ${INTERVAL} frames"
echo ""

# Test 1: CLIP ViT-B/32 (Fast mode)
echo "----------------------------------------"
echo "Test 1: CLIP ViT-B/32 (subsample=4)"
echo "----------------------------------------"
time python video_recognizer.py "$VIDEO" \
  --behavior \
  --behavior-model-type clip \
  --behavior-clip-model ViT-B/32 \
  --behavior-clip-subsample 4 \
  --interval-frames $INTERVAL \
  --batch-frames 32 \
  --behavior-clip-seconds 1.0 \
  --behavior-person-conf 0.2 \
  --max-seconds $MAX_SEC \
  --output-json outputs/test_clip_vit32_fast.json

echo ""
echo "----------------------------------------"
echo "Test 2: Kinetics r3d_18 (Baseline)"
echo "----------------------------------------"
time python video_recognizer.py "$VIDEO" \
  --behavior \
  --behavior-model-type kinetics \
  --behavior-model r3d_18 \
  --interval-frames $INTERVAL \
  --batch-frames 32 \
  --behavior-clip-seconds 1.0 \
  --behavior-person-conf 0.2 \
  --max-seconds $MAX_SEC \
  --output-json outputs/test_kinetics_r3d18.json

echo ""
echo "=== Comparison Complete ==="
echo ""
echo "Results:"
echo "  CLIP:     outputs/test_clip_vit32_fast.json"
echo "  Kinetics: outputs/test_kinetics_r3d18.json"
echo ""
echo "Compare behaviors for key students (e.g., 张泽宇):"
echo "  jq '.behavior_stats.by_student.\"张泽宇\".behaviors' outputs/test_clip_vit32_fast.json"
echo "  jq '.behavior_stats.by_student.\"张泽宇\".behaviors' outputs/test_kinetics_r3d18.json"
