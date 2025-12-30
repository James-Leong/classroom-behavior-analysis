#!/usr/bin/env python3
"""
Prepare YOLO fine-tuning data: Sample frames and auto-label with existing model.
This script filters out all classes except 'person' (class 0) and sets up
a single-class dataset ('student_context') for fine-tuning.
"""

import argparse
import random

from pathlib import Path

import cv2
import yaml
import json
import zipfile
import torch

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO finetuning data")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out-dir", default="data/yolo_finetune", help="Output directory")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to sample")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO model for auto-labeling")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} for auto-labeling...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # prefer CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample random frames
    indices = sorted(random.sample(range(total_frames), min(args.num_images, total_frames)))

    print(f"Sampling {len(indices)} frames from {total_frames} total frames...")

    count = 0
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"frame_{i:06d}"
        img_path = images_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Run inference (explicit device to prefer GPU when available)
        try:
            results = model(frame, device=device, verbose=False)[0]
        except TypeError:
            # Fallback if ultralytics version doesn't accept device here
            results = model(frame, verbose=False)[0]

        txt_path = labels_dir / f"{filename}.txt"

        # Write labels - ONLY keep class 0 (person)
        # But we save it as class 0 in the new dataset (which will be 'student_context')
        with open(txt_path, "w") as f:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # Filter: Only keep persons
                    continue

                xywhn = box.xywhn[0].tolist()
                # Write as class 0 (since our new dataset only has 1 class)
                f.write(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

        count += 1
        if count % 10 == 0:
            print(f"Processed {count} images")

    cap.release()

    # Create dataset.yaml
    # This defines that our new model will only have 1 class: 'student_context'
    dataset_yaml = {
        "path": str(out_dir.absolute()),
        "train": "images",
        "val": "images",  # Use same set for simple fine-tuning demo
        "nc": 1,
        "names": ["student_context"],
    }

    with open(out_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)

    # Create classes.txt for LabelImg (kept for backward compatibility)
    with open(out_dir / "classes.txt", "w") as f:
        f.write("student_context\n")

    # Create Label Studio tasks JSONL (use file:// URLs for local import)
    tasks_path = out_dir / "label_studio_tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as tj:
        for img in sorted(images_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            task = {"data": {"image": f"file://{str(img.resolve())}"}}
            tj.write(json.dumps(task, ensure_ascii=False) + "\n")

    # Write a Label Studio labeling config for rectangle boxes
    labeling_config = (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        '    <Label value="student_context"/>\n'
        "  </RectangleLabels>\n"
        "</View>\n"
    )
    with open(out_dir / "labeling_config.xml", "w", encoding="utf-8") as lc:
        lc.write(labeling_config)

    # Create a zip of images for convenient upload/import
    zip_path = out_dir / "images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img in sorted(images_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            zf.write(img, arcname=img.name)

    print("\nDone!")
    print(f"Data saved to: {out_dir}")
    print(f"- YOLO dataset YAML: {out_dir / 'dataset.yaml'}")
    print(f"- YOLO labels dir: {labels_dir}")
    print(f"- Label Studio tasks: {tasks_path}")
    print(f"- Label Studio config: {out_dir / 'labeling_config.xml'}")
    print(f"- Images archive: {zip_path}")
    print("")
    print("Label Studio usage options:")
    print("  1) Initialize the Label Studio DB:")
    print("     label-studio init")
    print("  2) Start the server (no browser on remote):")
    print("     label-studio start --no-browser --host 0.0.0.0 --port 8080")
    print(f"     Then in the web UI: Import > From JSON > select {tasks_path}")
    print("     Or upload images.zip and import accordingly.")
    print("Examples (run on remote server):")
    print("  # start in background and log output")
    print("  nohup label-studio start --no-browser --host 0.0.0.0 --port 8080 > labelstudio.log 2>&1 &")
    print("  # create SSH tunnel (on your local machine)")
    print("  ssh -L 8080:localhost:8080 user@your-server")
    print("  # then browse: http://localhost:8080")
    print("After labeling, export annotations in COCO/YOLO format for training.")


if __name__ == "__main__":
    main()
