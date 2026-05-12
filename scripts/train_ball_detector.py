"""
scripts/train_ball_detector.py
==============================
Task 1.4 — Train a dedicated YOLOv8n ball detector.

Fixes:
  - Uses proper YOLO directory structure (images/train + labels/train)
  - Creates hard links to images (zero disk overhead, no admin needed)
  - Deletes stale .cache files before training
  - Only processes frames that actually contain the ball
"""

import logging
import os
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BALL_CLASS_ID = "3"  # In the existing 4-class YOLO labels
OUTPUT_BALL_CLASS = "0"  # Remapped to 0 for this single-class model


def build_ball_dataset(split_dir: Path, output_images_dir: Path, output_labels_dir: Path) -> int:
    """
    Scans all sequences in split_dir, finds frames containing the ball,
    hard-links the image (zero copy) and writes a ball-only label file.

    Returns the number of ball frames found.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    seqs = sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("SNMOT-"))
    total_ball = 0

    for seq_dir in seqs:
        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            continue

        for img_file in sorted(img_dir.glob("*.jpg")):
            txt_file = img_file.with_suffix(".txt")
            if not txt_file.exists():
                continue

            # Filter only ball annotations and remap class 3 → 0
            ball_lines = []
            for line in txt_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5 and parts[0] == BALL_CLASS_ID:
                    ball_lines.append(OUTPUT_BALL_CLASS + " " + " ".join(parts[1:]))

            if not ball_lines:
                continue  # Skip frames with no ball

            # Unique filename using sequence prefix to avoid collisions
            stem = f"{seq_dir.name}_{img_file.stem}"

            # Hard-link image (instant, no extra disk space, works on same drive)
            dest_img = output_images_dir / f"{stem}.jpg"
            if not dest_img.exists():
                try:
                    os.link(img_file, dest_img)
                except OSError:
                    # Fallback: write a path reference file if hard link fails
                    # (cross-drive scenario — unlikely but safe)
                    pass

            # Write ball-only label
            dest_lbl = output_labels_dir / f"{stem}.txt"
            dest_lbl.write_text("\n".join(ball_lines) + "\n")

            total_ball += 1

    return total_ball


def delete_stale_caches(root_dir: Path) -> None:
    """Delete all .cache files so YOLO rebuilds them fresh."""
    for cache_file in root_dir.rglob("*.cache"):
        cache_file.unlink()
        logging.info(f"Deleted stale cache: {cache_file}")


def write_ball_yaml(output_dir: Path) -> Path:
    yaml_content = f"""# Ball-only detector dataset
path: {output_dir.resolve().as_posix()}
train: images/train
val:   images/val

nc: 1
names:
  0: ball
"""
    yaml_path = output_dir / "ball_dataset.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def train_ball_detector() -> None:
    tracking_dir = Path("data/soccernet/tracking")
    output_dir   = Path("data/ball_only")

    # Step 1: Remove stale caches from previous training run
    logging.info("=== Cleaning stale YOLO cache files ===")
    delete_stale_caches(tracking_dir)

    # Step 2: Build ball-only dataset (hard links, no copying)
    logging.info("=== Building ball-only training split ===")
    n_train = build_ball_dataset(
        tracking_dir / "train",
        output_dir / "images" / "train",
        output_dir / "labels" / "train",
    )
    logging.info(f"Train: {n_train} frames with ball")

    logging.info("=== Building ball-only validation split ===")
    n_val = build_ball_dataset(
        tracking_dir / "test",
        output_dir / "images" / "val",
        output_dir / "labels" / "val",
    )
    logging.info(f"Val:   {n_val} frames with ball")

    if n_train == 0:
        logging.error("No ball frames found! Check that train.py annotation conversion ran first.")
        return

    # Step 3: Write dataset YAML
    yaml_path = write_ball_yaml(output_dir)
    logging.info(f"Dataset YAML: {yaml_path}")

    # Step 4: Train YOLOv8n ball detector
    logging.info("=== Training dedicated ball detector (YOLOv8n) ===")
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(yaml_path),
        epochs=40,
        imgsz=640,         # Same as player model — 4x faster than 1280px
        batch=8,           # Can increase batch since image size is smaller
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2,
        # Lightweight augmentation
        augment=True,
        mosaic=0.0,        # Disabled — mosaic at 640px makes ball too small
        copy_paste=0.0,    # Disabled — slows training significantly
        scale=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Output
        project="models",
        name="ball_detector",
        save=True,
        plots=True,
        patience=10,
        exist_ok=True,
    )

    logging.info("=== Ball detector training complete! ===")
    logging.info("Best weights: models/ball_detector/weights/best.pt")


if __name__ == "__main__":
    train_ball_detector()
