"""
scripts/visualize_annotations.py
=================================
Visualize MOT ground-truth annotations from a SoccerNet tracking sequence.

PRD Reference : Section 5.2 (MOT annotation format), Task 0.3

Reads gt/gt.txt and img1/*.jpg from a sequence directory and draws
bounding boxes with track IDs, roles, and team colors.

Also optionally exports an annotated video clip for inspection.

Usage
-----
  # Show frames 1-30 from first train sequence (interactive)
  python scripts/visualize_annotations.py

  # Specify sequence and frame range
  python scripts/visualize_annotations.py --seq SNMOT-060 --start 1 --end 50

  # Export annotated video of first 300 frames
  python scripts/visualize_annotations.py --seq SNMOT-060 --export --end 300

  # List all available sequences
  python scripts/visualize_annotations.py --list

  # Non-interactive (saves frames to outputs/annotations/)
  python scripts/visualize_annotations.py --seq SNMOT-060 --no-display --export
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACKING_TRAIN = PROJECT_ROOT / "data" / "soccernet" / "tracking" / "train"
TRACKING_TEST  = PROJECT_ROOT / "data" / "soccernet" / "tracking" / "test"
OUTPUT_DIR     = PROJECT_ROOT / "outputs" / "annotations"

# ---------------------------------------------------------------------------
# Color palette — matches data_models.py class and team conventions
# ---------------------------------------------------------------------------
# Role → BGR color (for drawing)
ROLE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "player team left":      (50,  200,  50),   # Green
    "player team right":     (50,   50, 220),   # Red-ish blue
    "goalkeeper team left":  (0,   210, 210),   # Cyan
    "goalkeepers team left": (0,   210, 210),   # Cyan (alias)
    "goalkeeper team right": (210, 130,   0),   # Orange
    "referee":               (200, 200,   0),   # Yellow
    "ball":                  (0,   255, 255),   # Bright cyan
    "unknown":               (180, 180, 180),   # Gray
}

CLASS_LABEL: Dict[str, str] = {
    "player team left":      "L",
    "player team right":     "R",
    "goalkeeper team left":  "GKL",
    "goalkeepers team left": "GKL",
    "goalkeeper team right": "GKR",
    "referee":               "REF",
    "ball":                  "BALL",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_seqinfo(seq_dir: Path) -> Dict[str, str]:
    """Parse seqinfo.ini into a dict."""
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    return dict(cfg["Sequence"])


def load_gameinfo(seq_dir: Path) -> Dict[int, str]:
    """
    Parse gameinfo.ini → {track_id: role_string}.
    Role strings look like: "player team left", "referee", "ball", etc.
    """
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "gameinfo.ini")
    section = dict(cfg["Sequence"])

    track_roles: Dict[int, str] = {}
    idx = 1
    while True:
        key = f"trackletid_{idx}"
        if key not in section:
            break
        raw = section[key].strip()          # e.g. " player team left;10"
        role = raw.split(";")[0].strip()    # e.g. "player team left"
        track_roles[idx] = role
        idx += 1
    return track_roles


def load_gt(seq_dir: Path) -> Dict[int, List[Tuple]]:
    """
    Parse gt/gt.txt → {frame_id: [(track_id, x, y, w, h), ...]}

    MOT format (PRD Section 5.2):
      frame_id, track_id, x, y, w, h, conf, class_id, visibility
    """
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    frames: Dict[int, List[Tuple]] = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x        = float(parts[2])    # top-left x (pixels)
            y        = float(parts[3])    # top-left y (pixels)
            w        = float(parts[4])    # width  (pixels)
            h        = float(parts[5])    # height (pixels)
            frames.setdefault(frame_id, []).append((track_id, x, y, w, h))
    return frames


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_annotations(
    frame: np.ndarray,
    detections: List[Tuple],
    track_roles: Dict[int, str],
    frame_id: int,
    seq_name: str,
    fps: float,
) -> np.ndarray:
    """
    Draw MOT ground-truth bounding boxes on a frame.

    Each box is colored by role/team and labeled with:
      track_id | role abbreviation | jersey number (if available)
    """
    canvas = frame.copy()
    h_img, w_img = canvas.shape[:2]

    for track_id, x, y, w, h in detections:
        role  = track_roles.get(track_id, "unknown")
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        label_abbr = CLASS_LABEL.get(role, "?")

        x1, y1 = int(x),     int(y)
        x2, y2 = int(x + w), int(y + h)

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"ID:{track_id} {label_abbr}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness  = 1
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y1 = max(y1 - lh - baseline - 2, 0)
        label_y2 = label_y1 + lh + baseline + 2
        cv2.rectangle(canvas, (x1, label_y1), (x1 + lw + 4, label_y2), color, -1)
        cv2.putText(
            canvas, label,
            (x1 + 2, label_y2 - baseline - 1),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

    # ── HUD overlay (top-left) ────────────────────────────────────────────
    timestamp_s = (frame_id - 1) / fps
    mm, ss = divmod(int(timestamp_s), 60)
    hud_lines = [
        f"Seq: {seq_name}",
        f"Frame: {frame_id:04d}  |  Time: {mm:02d}:{ss:02d}",
        f"GT boxes: {len(detections)}",
    ]
    for i, txt in enumerate(hud_lines):
        y_pos = 22 + i * 22
        cv2.putText(canvas, txt, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, txt, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Legend (bottom-left) ──────────────────────────────────────────────
    legend_items = [
        ("Team Left (player)",      ROLE_COLORS["player team left"]),
        ("Team Right (player)",     ROLE_COLORS["player team right"]),
        ("GK Left",                 ROLE_COLORS["goalkeeper team left"]),
        ("GK Right",                ROLE_COLORS["goalkeeper team right"]),
        ("Referee",                 ROLE_COLORS["referee"]),
        ("Ball",                    ROLE_COLORS["ball"]),
    ]
    legend_x = 10
    legend_y_start = h_img - len(legend_items) * 22 - 10
    for i, (name, color) in enumerate(legend_items):
        y_pos = legend_y_start + i * 22
        cv2.rectangle(canvas, (legend_x, y_pos - 12), (legend_x + 16, y_pos + 2), color, -1)
        cv2.putText(canvas, name, (legend_x + 22, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, name, (legend_x + 22, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------
def visualize_sequence(
    seq_dir: Path,
    start_frame: int = 1,
    end_frame:   Optional[int] = None,
    display:     bool = True,
    export:      bool = False,
    scale:       float = 0.6,
) -> None:
    """
    Iterate through a sequence and show / export annotated frames.

    Parameters
    ----------
    seq_dir     : path to SNMOT-XXX directory
    start_frame : first frame to show (1-indexed)
    end_frame   : last frame to show (None = all)
    display     : show OpenCV window (requires display)
    export      : save annotated video to outputs/annotations/
    scale       : display scale factor (0.6 = 60% of original resolution)
    """
    seq_name = seq_dir.name
    seqinfo  = load_seqinfo(seq_dir)
    fps      = float(seqinfo.get("framerate", 25))
    seq_len  = int(seqinfo.get("seqlength", 0))
    width    = int(seqinfo.get("imwidth", 1920))
    height   = int(seqinfo.get("imheight", 1080))
    im_ext   = seqinfo.get("imext", ".jpg")

    if end_frame is None:
        end_frame = seq_len

    print(f"\n{'='*55}")
    print(f"Sequence   : {seq_name}")
    print(f"Resolution : {width}×{height}  |  FPS: {fps}")
    print(f"Frames     : {start_frame} -> {end_frame}  ({end_frame - start_frame + 1} total)")
    print(f"{'='*55}")

    # Load annotations
    gt_frames   = load_gt(seq_dir)
    track_roles = load_gameinfo(seq_dir)

    print(f"Loaded {sum(len(v) for v in gt_frames.values()):,} GT annotations "
          f"for {len(track_roles)} tracks")
    print("\nTrack legend:")
    for tid, role in track_roles.items():
        print(f"  Track {tid:2d} -> {role}")
    print()

    # Setup video writer if exporting
    writer = None
    if export:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{seq_name}_annotated.mp4"
        out_w = int(width  * scale)
        out_h = int(height * scale)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        print(f"Exporting to: {out_path}")
        print(f"Output size : {out_w}×{out_h}")

    img_dir = seq_dir / "img1"

    # Statistics
    stats = {"frames_processed": 0, "total_boxes": 0}

    for frame_id in range(start_frame, end_frame + 1):
        img_path = img_dir / f"{frame_id:06d}{im_ext}"
        if not img_path.exists():
            print(f"  WARNING: Frame not found: {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: Could not read frame: {img_path}")
            continue

        detections = gt_frames.get(frame_id, [])
        annotated  = draw_annotations(
            frame, detections, track_roles, frame_id, seq_name, fps
        )

        stats["frames_processed"] += 1
        stats["total_boxes"] += len(detections)

        if export and writer:
            export_frame = cv2.resize(annotated, (int(width * scale), int(height * scale)))
            writer.write(export_frame)

        if display:
            display_frame = cv2.resize(annotated, (int(width * scale), int(height * scale)))
            cv2.imshow(f"SoccerNet Annotations — {seq_name}", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("  User pressed Q — stopping.")
                break
            elif key == ord(" "):
                # Pause on spacebar
                print("  Paused — press any key to continue, Q to quit")
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord("q"):
                    break

    # Cleanup
    if writer:
        writer.release()
        print(f"\n[OK] Video saved: {OUTPUT_DIR / (seq_name + '_annotated.mp4')}")

    if display:
        cv2.destroyAllWindows()

    print(f"\n=== Summary ===")
    print(f"Frames processed : {stats['frames_processed']}")
    print(f"Total GT boxes   : {stats['total_boxes']}")
    print(f"Avg boxes/frame  : {stats['total_boxes'] / max(stats['frames_processed'], 1):.1f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_sequences() -> None:
    """Print all available sequences with metadata."""
    print(f"\n{'='*60}")
    print(f"{'Split':<8} {'Sequence':<14} {'Frames':>7} {'Tracks':>7} {'FPS':>5}")
    print("-" * 60)

    for split_name, split_dir in [("train", TRACKING_TRAIN), ("test", TRACKING_TEST)]:
        if not split_dir.exists():
            print(f"{split_name}: directory not found ({split_dir})")
            continue
        seqs = sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("SNMOT-"))
        for seq_dir in seqs:
            try:
                info   = load_seqinfo(seq_dir)
                roles  = load_gameinfo(seq_dir)
                frames = int(info.get("seqlength", "?"))
                fps    = info.get("framerate", "?")
                print(f"{split_name:<8} {seq_dir.name:<14} {frames:>7} {len(roles):>7} {fps:>5}")
            except Exception:
                print(f"{split_name:<8} {seq_dir.name:<14}   [error reading metadata]")

    print(f"{'='*60}")
    print(f"train: {sum(1 for d in TRACKING_TRAIN.iterdir() if d.is_dir() and d.name.startswith('SNMOT-')) if TRACKING_TRAIN.exists() else 0} sequences")
    print(f"test : {sum(1 for d in TRACKING_TEST.iterdir()  if d.is_dir() and d.name.startswith('SNMOT-')) if TRACKING_TEST.exists()  else 0} sequences\n")


def print_gt_sample(seq_dir: Path, n: int = 10) -> None:
    """Print first N rows of gt.txt for inspection."""
    gt_path = seq_dir / "gt" / "gt.txt"
    print(f"\n=== {gt_path} (first {n} rows) ===")
    print("frame_id | track_id | x      | y      | w    | h    | conf | cls | vis")
    print("-" * 70)
    with open(gt_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            p = line.strip().split(",")
            print(f"{p[0]:>8} | {p[1]:>8} | {float(p[2]):>6.0f} | {float(p[3]):>6.0f} | "
                  f"{float(p[4]):>4.0f} | {float(p[5]):>4.0f} | {p[6]:>4} | {p[7]:>3} | {p[8]:>3}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SoccerNet MOT annotation visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive viewer — first sequence, frames 1-50
  python scripts/visualize_annotations.py

  # Specific sequence and range
  python scripts/visualize_annotations.py --seq SNMOT-061 --start 100 --end 200

  # Export annotated video (no display window)
  python scripts/visualize_annotations.py --seq SNMOT-060 --export --no-display --end 300

  # List all sequences
  python scripts/visualize_annotations.py --list

  # Print raw gt.txt sample
  python scripts/visualize_annotations.py --seq SNMOT-060 --print-gt

  # Test split
  python scripts/visualize_annotations.py --split test --seq SNMOT-116 --end 30
        """
    )
    parser.add_argument("--seq",   default=None, help="Sequence name, e.g. SNMOT-060")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Dataset split (default: train)")
    parser.add_argument("--start", type=int, default=1,  help="Start frame (default: 1)")
    parser.add_argument("--end",   type=int, default=50, help="End frame (default: 50)")
    parser.add_argument("--scale", type=float, default=0.6,
                        help="Display scale factor (default: 0.6)")
    parser.add_argument("--export",     action="store_true", help="Export annotated video")
    parser.add_argument("--no-display", action="store_true", help="Disable live display window")
    parser.add_argument("--list",       action="store_true", help="List available sequences")
    parser.add_argument("--print-gt",   action="store_true", help="Print gt.txt sample rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_sequences()
        return

    # Resolve split directory
    split_dir = TRACKING_TRAIN if args.split == "train" else TRACKING_TEST
    if not split_dir.exists():
        print(f"ERROR: Split directory not found: {split_dir}")
        print("Make sure data is extracted. See data/DATA_LOCATION.md")
        sys.exit(1)

    # Resolve sequence directory
    if args.seq is None:
        # Pick first available sequence
        candidates = sorted(
            d for d in split_dir.iterdir()
            if d.is_dir() and d.name.startswith("SNMOT-")
        )
        if not candidates:
            print(f"ERROR: No sequences found in {split_dir}")
            sys.exit(1)
        seq_dir = candidates[0]
        print(f"No --seq specified, using first available: {seq_dir.name}")
    else:
        seq_dir = split_dir / args.seq
        if not seq_dir.exists():
            print(f"ERROR: Sequence not found: {seq_dir}")
            sys.exit(1)

    if args.print_gt:
        print_gt_sample(seq_dir)
        return

    visualize_sequence(
        seq_dir=seq_dir,
        start_frame=args.start,
        end_frame=args.end,
        display=not args.no_display,
        export=args.export,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
