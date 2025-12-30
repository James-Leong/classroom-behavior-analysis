#!/usr/bin/env python3
"""
Debug Trace Generation Tool for Frontend Visualization.

This script runs the behavior analysis pipeline on a small segment of video/data
and exports detailed intermediate variables (scores, gating status, crops) to a JSON file.
This JSON file is used by the frontend "Simulation Mode" to display a realistic process view.
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.behavior.pipeline import BehaviorPipelineConfig, run_behavior_pipeline_on_result
from src.behavior.debug import DebugTraceCollector
from src.utils.log import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate debug trace for frontend visualization")
    parser.add_argument("--face-json", required=True, help="Face recognition result JSON")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output debug trace JSON")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=30.0, help="End time in seconds")

    # Model args
    parser.add_argument("--model-type", default="clip", choices=["kinetics", "clip"])
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-custom-behaviors", nargs="+", help="Custom behaviors (e.g. listening reading)")
    parser.add_argument("--clip-custom-labels", nargs="+", help="Custom behavior descriptions")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # Load face json
    if not os.path.exists(args.face_json):
        logger.error(f"Face JSON not found: {args.face_json}")
        return

    with open(args.face_json, "r", encoding="utf-8") as f:
        face_data = json.load(f)

    fps = float(face_data.get("fps", 25.0))
    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps)

    logger.info(f"Filtering frames from {start_frame} to {end_frame} ({args.start}-{args.end}s)")

    # Filter frames
    original_frames = face_data.get("frames", [])
    filtered_frames = []
    for fr in original_frames:
        fidx = int(fr.get("frame", 0))
        if start_frame <= fidx <= end_frame:
            filtered_frames.append(fr)

    if not filtered_frames:
        logger.error("No frames found in specified range!")
        return

    logger.info(f"Selected {len(filtered_frames)} frames for processing")

    # Create temp data dict with filtered frames
    subset_data = {
        "fps": fps,
        "frames": filtered_frames,
        "used_frame_interval": face_data.get("used_frame_interval", 1),
    }

    # Config
    cfg = BehaviorPipelineConfig(
        enabled=True,
        model_type=args.model_type,
        clip_model_name=args.clip_model,
        clip_custom_behaviors=args.clip_custom_behaviors,
        clip_custom_labels=args.clip_custom_labels,
        device=args.device,
        enable_smoothing=True,
        # Default standard params
        uncertain_min_prob=0.2,
        uncertain_min_margin=0.02,
    )

    collector = DebugTraceCollector()

    logger.info("Starting pipeline with debug collection...")

    # Run pipeline
    # Note: The pipeline will read video frames. Ensure video path is correct.
    run_behavior_pipeline_on_result(input_video=args.video, result=subset_data, cfg=cfg, debug_collector=collector)

    logger.info(f"Saving trace to {args.output}")
    collector.save_json(args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
