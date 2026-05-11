"""
scripts/run_pipeline.py
=======================
Stage 7 of the Football Tracker Pipeline.
The main orchestrator that glues together ingestion, detection, tracking, 
mapping, classification, and analytics.

PRD Reference: Task 1.8
"""

import argparse
import logging
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from src.ingestion.video_reader import VideoReader
from src.detection.detector import PlayerDetector
from src.tracking.tracker import ByteTrackerWrapper
from src.mapping.homography import CoordinateMapper
from src.classification.classifier import TeamClassifier
from src.analytics.metrics import MetricsEngine
from src.data_models import PlayerMetrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_config(config_path: str) -> dict:
    """Load the main pipeline configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    video_path: str,
    config_path: str,
    seq_name: str = "unknown",
    max_frames: int = None,
    output_dir: str = "outputs",
) -> None:
    """
    Run the end-to-end football analytics pipeline.
    """
    cfg = load_config(config_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Pipeline Stages
    logging.info("Initializing Pipeline Stages...")
    
    # Stage 1: Ingestion
    reader = VideoReader(
        video_path=video_path,
        frame_stride=cfg["ingestion"]["frame_stride"],
        max_frames=max_frames
    )
    
    # Stage 2: Detection
    detector = PlayerDetector(
        model_path=cfg["detection"]["model_path"],
        confidence=cfg["detection"]["confidence"],
        iou_threshold=cfg["detection"]["iou_threshold"],
        input_size=cfg["detection"]["input_size"],
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )
    
    # Stage 3: Tracking
    tracker = ByteTrackerWrapper(
        track_thresh=cfg["tracking"]["track_activation_threshold"],
        track_buffer=cfg["tracking"]["lost_track_buffer"],
        match_thresh=cfg["tracking"]["minimum_matching_threshold"],
        frame_rate=int(reader.fps)
    )
    
    # Stage 4: Homography
    mapper = CoordinateMapper(
        calibration_dir=Path("data/soccernet/calibration")
    )
    mapper.load_sequence_calibration(seq_name)
    
    # Stage 5: Classification
    classifier = TeamClassifier(
        history_frames=cfg["teams"]["min_track_frames"],
        update_rate=0.05
    )
    
    # Stage 6: Analytics
    analytics = MetricsEngine(fps=reader.fps)

    # Output Video Writer Setup
    out_video_path = out_dir / f"{seq_name}_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_video_path), 
        fourcc, 
        reader.fps, 
        (reader.width, reader.height)
    )

    logging.info(f"Starting processing for {reader.total_frames} frames...")
    
    # 2. Main Processing Loop
    for frame_id, frame in tqdm(reader.read_frames(), total=reader.max_frames or reader.total_frames):
        # Stage 2: Detection
        detections = detector.detect(frame, frame_id)
        
        # Stage 3: Tracking
        tracks = tracker.update(detections)
        
        # Stage 4: Coordinate Mapping
        mapped_tracks = mapper.map_tracks(tracks, seq_name, frame_id)
        
        # Stage 5: Team Classification
        classified_tracks = classifier.classify(mapped_tracks, frame)
        
        # Stage 6: Analytics Update
        analytics.update(classified_tracks)

        # ---------------------------------------------------------
        # Render Visualization for this frame
        # ---------------------------------------------------------
        canvas = frame.copy()
        for t in classified_tracks:
            x1, y1, x2, y2 = map(int, t.bbox_px)
            
            # Choose color based on team_id
            if t.team_id == 1:
                color = (0, 0, 255)   # Red
            elif t.team_id == 2:
                color = (255, 0, 0)   # Blue
            else:
                color = (0, 255, 255) # Yellow (unknown/ref/ball)
                
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"ID:{t.track_id} T:{t.team_id or '-'}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        # Overlay running metrics for a tracked player (e.g., ID 1) if they exist
        current_metrics = analytics.get_metrics()
        # Just show some generic HUD
        cv2.putText(canvas, f"Frame: {frame_id}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(canvas, f"Active Tracks: {len(classified_tracks)}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    
        writer.write(canvas)

    # 3. Finalize and Export
    analytics.finalize_sprints()
    final_metrics = analytics.get_metrics()
    writer.release()
    
    logging.info(f"Processing complete! Output video saved to {out_video_path}")
    
    # Export CSV
    csv_path = out_dir / f"{seq_name}_metrics.csv"
    import csv
    if final_metrics:
        # Get columns from the first metric's to_csv_row()
        first_metric = list(final_metrics.values())[0]
        fieldnames = list(first_metric.to_csv_row().keys())
        
        with open(csv_path, "w", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            for pm in final_metrics.values():
                csv_writer.writerow(pm.to_csv_row())
        
        logging.info(f"Metrics CSV saved to {csv_path}")
    else:
        logging.warning("No metrics generated to export.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Football Tracker Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to config")
    parser.add_argument("--seq", type=str, default="unknown", help="Sequence name for calibration mapping")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_pipeline(
        video_path=args.video,
        config_path=args.config,
        seq_name=args.seq,
        max_frames=args.max_frames,
        output_dir=args.out
    )
