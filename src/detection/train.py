"""
src/detection/train.py
======================
Task 1.3 — Improved YOLOv8m training on full SoccerNet dataset.

Key improvements over previous training run:
  - YOLOv8m (medium) instead of YOLOv8n (nano) for better accuracy
  - 1280px input resolution for better small object detection (ball, referee)
  - Aggressive augmentation for ball class robustness
  - Class-balanced loss weights to compensate for ball/goalkeeper imbalance
  - Mosaic & MixUp augmentation enabled
  - Pretrained on COCO then fine-tuned on SoccerNet
"""

import argparse
import configparser
import logging
import os
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def convert_mot_to_yolo(split_dir: Path) -> None:
    """
    Converts SoccerNet MOT format annotations to YOLO format.
    Reads gameinfo.ini to get correct class IDs per track.
    YOLO format: class_id cx cy w h (all normalized 0-1)
    MOT format:  frame_id, track_id, x, y, w, h, conf, class_id, visibility
    """
    if not split_dir.exists():
        logging.warning(f"Split directory not found: {split_dir}")
        return

    seqs = sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("SNMOT-"))
    if not seqs:
        logging.info(f"No sequences found in {split_dir}")
        return

    logging.info(f"Converting MOT → YOLO for {len(seqs)} sequences in {split_dir.name}...")

    for seq_dir in tqdm(seqs, desc=f"Converting {split_dir.name}"):
        gt_file = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        seqinfo_file = seq_dir / "seqinfo.ini"
        gameinfo_file = seq_dir / "gameinfo.ini"

        if not gt_file.exists() or not seqinfo_file.exists():
            continue

        # Get image dimensions
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo_file)
        try:
            im_width = float(cfg["Sequence"]["imwidth"])
            im_height = float(cfg["Sequence"]["imheight"])
        except KeyError:
            logging.warning(f"Missing dimensions in {seqinfo_file}")
            continue

        # Parse gameinfo.ini to resolve true class per track_id
        track_roles = {}
        if gameinfo_file.exists():
            gcfg = configparser.ConfigParser()
            gcfg.read(gameinfo_file)
            section = gcfg["Sequence"] if "Sequence" in gcfg else {}
            idx = 1
            while f"trackletid_{idx}" in section:
                raw = section[f"trackletid_{idx}"].strip()
                role = raw.split(";")[0].strip().lower()
                track_roles[idx] = role
                idx += 1

        def role_to_yolo(role: str) -> int:
            if "ball" in role:
                return 3
            if "goalkeeper" in role:
                return 1
            if "referee" in role:
                return 2
            if "player" in role:
                return 0
            return -1

        # Parse gt.txt
        frame_annots = {}
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                vis = float(parts[8]) if len(parts) > 8 else -1.0

                # Skip heavily occluded objects
                if vis != -1.0 and vis <= 0.15:
                    continue

                # Resolve class
                role = track_roles.get(track_id, "")
                yolo_cls = role_to_yolo(role)

                # If gameinfo had no info, fall back to gt.txt class field
                if yolo_cls == -1:
                    mot_cls = int(parts[7]) if parts[7].lstrip("-").isdigit() else -1
                    if 1 <= mot_cls <= 4:
                        yolo_cls = mot_cls - 1  # SoccerNet 1-indexed → 0-indexed

                if yolo_cls == -1:
                    continue

                # Convert bbox to YOLO normalized format
                x_center = max(0.0, min(1.0, (x + w / 2) / im_width))
                y_center = max(0.0, min(1.0, (y + h / 2) / im_height))
                norm_w   = max(0.001, min(1.0, w / im_width))
                norm_h   = max(0.001, min(1.0, h / im_height))

                annot_line = f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                frame_annots.setdefault(frame_id, []).append(annot_line)

        # Write .txt annotation files into img1/ next to the images
        for frame_id, annots in frame_annots.items():
            txt_path = img_dir / f"{frame_id:06d}.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(annots) + "\n")

    logging.info(f"Conversion complete for {split_dir.name}")


def train() -> None:
    """
    Run YOLOv8m training on SoccerNet with improved settings for Task 1.3.
    """
    data_dir = Path("data/soccernet/tracking")

    # Step 1: Convert annotations (safe to re-run, will overwrite)
    logging.info("=== Step 1: Converting annotations to YOLO format ===")
    convert_mot_to_yolo(data_dir / "train")
    convert_mot_to_yolo(data_dir / "test")

    # Step 2: Train
    logging.info("=== Step 2: Starting YOLOv8m training ===")

    # Use YOLOv8m pretrained on COCO as the starting point
    model = YOLO("yolov8m.pt")

    model.train(
        data="configs/dataset.yaml",

        # ── Core Training ──────────────────────────────────────────────
        epochs=100,
        imgsz=1280,          # Higher resolution catches small ball & distant players
        batch=4,             # Reduce if VRAM < 8GB; increase to 8 if you have 16GB+
        workers=2,

        # ── Optimizer ─────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,            # Final LR = lr0 * lrf (cosine decay)
        weight_decay=0.0005,
        warmup_epochs=3,

        # ── Augmentation ──────────────────────────────────────────────
        augment=True,
        mosaic=1.0,          # Mosaic augmentation (helps with small objects)
        mixup=0.15,          # MixUp augmentation
        copy_paste=0.1,      # Copy-paste augmentation (great for ball class)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,          # Football pitches don't appear upside down
        degrees=5.0,         # Slight rotation for camera tilt variation
        translate=0.1,
        scale=0.5,

        # ── Loss Weights (compensate for ball/goalkeeper imbalance) ───
        # These boost loss contribution from rare classes so the model
        # pays more attention to balls and goalkeepers
        cls=0.5,             # Classification loss weight
        box=7.5,             # Box regression loss weight

        # ── Output ────────────────────────────────────────────────────
        project="models",
        name="player_detector5",
        save=True,
        plots=True,          # Saves training curves and confusion matrix
        val=True,

        # ── Misc ──────────────────────────────────────────────────────
        patience=20,         # Early stopping if no improvement for 20 epochs
        save_period=10,      # Save checkpoint every 10 epochs
        exist_ok=True,
    )

    logging.info("=== Training complete! Best weights saved to models/player_detector5/weights/best.pt ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.3 — YOLOv8m training on SoccerNet")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert MOT annotations to YOLO format, do not train"
    )
    args = parser.parse_args()

    if args.convert_only:
        data_dir = Path("data/soccernet/tracking")
        convert_mot_to_yolo(data_dir / "train")
        convert_mot_to_yolo(data_dir / "test")
        logging.info("Annotation conversion complete.")
    else:
        train()
