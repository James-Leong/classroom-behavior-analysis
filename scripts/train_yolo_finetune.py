#!/usr/bin/env python3
"""
Train/Fine-tune YOLO model on the custom 'student_context' dataset.
"""

import argparse
import torch

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/yolo_finetune/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model to fine-tune")
    parser.add_argument("--name", default="yolo11n_classroom_context", help="Output project name")
    args = parser.parse_args()

    print(f"Loading base model: {args.model}")
    # Load the pre-trained model
    model = YOLO(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Starting training on {args.data}...")
    # Train the model
    # YOLO will automatically detect that dataset.yaml has nc=1 (vs original 80)
    # and will replace the output head while keeping the backbone weights.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=device,
        project="models",
        name=args.name,
        exist_ok=True,
    )

    print("Training completed.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("\nTo use this model, update your config to point to this path.")


if __name__ == "__main__":
    main()
