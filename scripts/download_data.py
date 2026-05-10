
import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root regardless of where script is invoked from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "soccernet"
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Available tasks and their descriptions
# ---------------------------------------------------------------------------
AVAILABLE_TASKS = {
    "tracking": {
        "description": "MOT-format bounding boxes + player IDs per frame",
        "splits":      ["train", "valid", "test"],
        "used_for":    "Tracker training and evaluation",
        "status":      "train + test already extracted; valid missing",
    },
    "tracking-2023": {
        "description": "Updated benchmark with improved annotations",
        "splits":      ["train", "valid", "test"],
        "used_for":    "Primary evaluation benchmark (Phase 1 gate)",
        "status":      "Not downloaded",
    },
    "calibration": {
        "description": "Homography matrices + pitch keypoint ground truth",
        "splits":      ["train", "valid", "test"],
        "used_for":    "Coordinate mapping validation (Task 1.5)",
        "status":      "Not downloaded",
    },
    "ball-2024": {
        "description": "Ball position annotations per frame",
        "splits":      ["train", "valid", "test"],
        "used_for":    "Ball detector training (GOAL-20)",
        "status":      "Not downloaded",
    },
    "action_spotting": {
        "description": "Timestamped event labels (goal, foul, corner, etc.)",
        "splits":      ["train", "valid", "test"],
        "used_for":    "Event correlation (GOAL-19)",
        "status":      "Not downloaded",
    },
}

VIDEO_FILES = ["1_224p.mkv", "2_224p.mkv"]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------
def download_tasks(tasks: list[str], splits: list[str], password: str) -> None:
    """Download specified SoccerNet tasks using the official downloader."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("ERROR: SoccerNet package not installed.")
        print("Run:  pip install SoccerNet>=0.1.50")
        sys.exit(1)

    local_dir = str(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SoccerNet Downloader")
    print(f"Target directory : {local_dir}")
    print(f"Tasks            : {tasks}")
    print(f"Splits           : {splits}")
    print(f"{'='*60}\n")

    dl = SoccerNetDownloader(LocalDirectory=local_dir)
    dl.password = password

    for task in tasks:
        if task not in AVAILABLE_TASKS:
            print(f"WARNING: Unknown task '{task}' — skipping. "
                  f"Valid tasks: {list(AVAILABLE_TASKS.keys())}")
            continue

        info = AVAILABLE_TASKS[task]
        valid_splits = [s for s in splits if s in info["splits"]]
        if not valid_splits:
            print(f"WARNING: No valid splits for task '{task}'. "
                  f"Available: {info['splits']}")
            continue

        print(f"Downloading task='{task}'  splits={valid_splits}")
        print(f"  Description : {info['description']}")
        print(f"  Used for    : {info['used_for']}")
        try:
            dl.downloadDataTask(task=task, split=valid_splits)
            print(f"  [OK] '{task}' downloaded successfully.\n")
        except Exception as e:
            print(f"  [ERR] Error downloading '{task}': {e}\n")


def download_videos(splits: list[str], password: str) -> None:
    """Download raw .mkv video files for fine-tuning."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("ERROR: SoccerNet package not installed.")
        sys.exit(1)

    local_dir = str(DATA_DIR)
    dl = SoccerNetDownloader(LocalDirectory=local_dir)
    dl.password = password

    print(f"Downloading raw videos: {VIDEO_FILES} for splits={splits}")
    try:
        dl.downloadGames(files=VIDEO_FILES, split=splits)
        print("[OK] Videos downloaded.")
    except Exception as e:
        print(f"[ERR] Error downloading videos: {e}")


def print_status() -> None:
    """Print current download status of all tasks."""
    print(f"\n{'='*60}")
    print("SoccerNet Data Status")
    print(f"Root: {DATA_DIR}")
    print(f"{'='*60}")
    print(f"{'Task':<20} {'Status':<15} {'Train':>6} {'Valid':>6} {'Test':>6}")
    print("-" * 60)

    for task, info in AVAILABLE_TASKS.items():
        task_dir = DATA_DIR / task
        train_n  = _count_sequences(task_dir / "train")
        valid_n  = _count_sequences(task_dir / "valid")
        test_n   = _count_sequences(task_dir / "test")
        present  = train_n + valid_n + test_n > 0
        status   = "[OK] present" if present else "[--] missing"
        print(f"{task:<20} {status:<15} {train_n:>6} {valid_n:>6} {test_n:>6}")

    print(f"{'='*60}\n")


def _count_sequences(split_dir: Path) -> int:
    """Count SNMOT-XXX sequence directories in a split directory."""
    if not split_dir.exists():
        return 0
    return sum(1 for d in split_dir.iterdir()
               if d.is_dir() and d.name.startswith("SNMOT-"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SoccerNet dataset downloader for Football Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current download status
  python scripts/download_data.py --status

  # Download missing tracking valid split
  python scripts/download_data.py --tasks tracking --splits valid

  # Download tracking-2023 and calibration (needed for Phase 1 evaluation)
  python scripts/download_data.py --tasks tracking-2023 calibration

  # Download everything for Phase 2
  python scripts/download_data.py --tasks ball-2024 action_spotting

  # Download raw videos for fine-tuning
  python scripts/download_data.py --videos --splits train
        """
    )
    parser.add_argument(
        "--tasks", nargs="+",
        choices=list(AVAILABLE_TASKS.keys()),
        default=["tracking-2023", "calibration"],
        help="SoccerNet tasks to download (default: tracking-2023 calibration)"
    )
    parser.add_argument(
        "--splits", nargs="+",
        choices=["train", "valid", "test"],
        default=["train", "valid", "test"],
        help="Dataset splits to download (default: all)"
    )
    parser.add_argument(
        "--password", type=str,
        default="s0cc3rn3t",
        help="SoccerNet dataset password (default: s0cc3rn3t)"
    )
    parser.add_argument(
        "--videos", action="store_true",
        help="Also download raw .mkv video files"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current download status and exit"
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List all available tasks and exit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        print("\nAvailable SoccerNet tasks:")
        for task, info in AVAILABLE_TASKS.items():
            print(f"\n  {task}")
            print(f"    Description : {info['description']}")
            print(f"    Used for    : {info['used_for']}")
            print(f"    Status      : {info['status']}")
        return

    if args.status:
        print_status()
        return

    print_status()

    download_tasks(
        tasks=args.tasks,
        splits=args.splits,
        password=args.password,
    )

    if args.videos:
        download_videos(splits=args.splits, password=args.password)

    print("\nPost-download status:")
    print_status()


if __name__ == "__main__":
    main()
