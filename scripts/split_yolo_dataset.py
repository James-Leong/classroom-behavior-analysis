#!/usr/bin/env python3
"""Split a YOLO-format dataset into train/val and generate dataset.yaml.

Usage:
  python3 scripts/split_yolo_dataset.py --dataset data/yolo_annotated --ratio 0.8 --copy
"""

import argparse
import random
import shutil
from pathlib import Path


def find_images(img_dir: Path):
    exts = (".jpg", ".jpeg", ".png")
    return [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset root folder containing images/ labels/ classes.txt")
    p.add_argument("--ratio", type=float, default=0.8, help="train fraction (rest -> val)")
    p.add_argument("--copy", action="store_true", help="copy files instead of moving")
    args = p.parse_args()

    root = Path(args.dataset)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"Expected {img_dir} and {lbl_dir} to exist")

    imgs = find_images(img_dir)
    if not imgs:
        raise SystemExit("No images found in " + str(img_dir))

    random.seed(42)
    imgs_shuf = imgs[:]
    random.shuffle(imgs_shuf)
    n_train = int(len(imgs_shuf) * args.ratio)
    train_imgs = imgs_shuf[:n_train]
    val_imgs = imgs_shuf[n_train:]

    # prepare dirs
    images_train = img_dir / "train"
    images_val = img_dir / "val"
    labels_train = lbl_dir / "train"
    labels_val = lbl_dir / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    def copy_pair(img_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
        dst_img = dst_img_dir / img_path.name
        lbl_name = img_path.with_suffix(".txt").name
        src_lbl = lbl_dir / lbl_name
        dst_lbl = dst_lbl_dir / lbl_name
        if args.copy:
            shutil.copy2(img_path, dst_img)
        else:
            shutil.move(str(img_path), str(dst_img))
        if src_lbl.exists():
            if args.copy:
                shutil.copy2(src_lbl, dst_lbl)
            else:
                shutil.move(str(src_lbl), str(dst_lbl))
        else:
            dst_lbl.write_text("")

    for im in train_imgs:
        copy_pair(im, images_train, labels_train)
    for im in val_imgs:
        copy_pair(im, images_val, labels_val)

    # read classes
    names = ["class0"]
    classes_file = root / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]

    nc = len(names)

    yaml_path = root / "dataset.yaml"
    content = (
        f"train: {str(images_train)}\nval: {str(images_val)}\ntest: {str(images_val)}\nnc: {nc}\nnames: {names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")

    print("Wrote", yaml_path)
    print(f"total images: {len(imgs)}, train: {len(train_imgs)}, val: {len(val_imgs)}")


if __name__ == "__main__":
    main()
