"""
dashboard/pipeline_runner.py
==============================
Runs the football analytics pipeline on a video file and returns all results
as plain Python dicts/lists suitable for Streamlit display.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import PlayerDetector
from src.tracking.tracker import ByteTrackerWrapper as PlayerTracker
from src.mapping.homography import CoordinateMapper
from src.classification.classifier import TeamClassifier
from src.analytics.metrics import MetricsEngine, MIN_TRACK_FRAMES
from src.analytics.formation import FormationDetector
from src.analytics.pressing import PressingAnalyzer
from src.analytics.possession import PossessionAnalyzer
from src.analytics.offside import OffsideAnalyzer
from src.analytics.voronoi import VoronoiGenerator
from src.analytics.pass_map import PassMapAnalyzer, PassMapGenerator
from src.data_models import Track, TrajectoryPoint, CLASS_BALL, CLASS_PLAYER

logging.basicConfig(level=logging.WARNING)



def run_pipeline(
    video_path: str,
    max_frames: Optional[int] = None,
    conf_threshold: float = 0.35,
    fps_override: int = 25,
    progress_callback=None,
) -> Dict:
    """
    Runs the full analytics pipeline on a video.

    Returns a dict with keys:
        frames         : List of annotated BGR frames (np.ndarray)
        metrics_df     : list of player metric dicts
        heatmap_path   : path to overall heatmap PNG
        sprint_hm_path : path to sprint heatmap PNG
        formations     : {team_id: formation_str}
        pressing       : {team_id: float}
        possession_pct : {team_id: float}
        possession_zone: {team_id: {zone: seconds}}
        offside_alerts : {track_id: int}
        trajectory_data: {track_id: List[TrajectoryPoint]}
        total_frames   : int
    """
    # ── Module init ──────────────────────────────────────────────────────────
    # Use best available player detection model
    model_path = "models/player_detector4/weights/best.pt"
    if not Path(model_path).exists():
        model_path = "models/player_detector3/weights/best.pt"
    if not Path(model_path).exists():
        model_path = "yolov8n.pt"
        logging.warning("Using fallback yolov8n.pt — player detection may be limited")

    # Use dedicated ball model if available (Task 1.4)
    ball_model_path = "models/ball_detector/weights/best.pt"
    if not Path(ball_model_path).exists():
        ball_model_path = None
        logging.info("No dedicated ball model found. Using primary model for ball detection.")
    else:
        logging.info(f"Dedicated ball model loaded: {ball_model_path}")

    detector = PlayerDetector(
        model_path=model_path,
        confidence=conf_threshold,
        iou_threshold=0.45,
        device="cuda" if _cuda_available() else "cpu",
        input_size=640,
        ball_model_path=ball_model_path,
    )
    tracker = PlayerTracker()
    team_classifier = TeamClassifier()
    # Look for calibration ZIP (prefer train split; falls back to default)
    calib_zip = None
    for calib_candidate in [
        "../soccernet_data/calibration/train.zip",
        "soccernet_data/calibration/train.zip",
    ]:
        if Path(calib_candidate).exists():
            calib_zip = calib_candidate
            break

    # Detect actual frame dimensions for the input video
    _cap_probe = cv2.VideoCapture(video_path)
    _vid_w = int(_cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
    _vid_h = int(_cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    _cap_probe.release()

    coord_mapper = CoordinateMapper(
        calibration_zip=calib_zip,
        image_width=_vid_w,
        image_height=_vid_h,
    )
    metrics_engine = MetricsEngine(fps=fps_override)
    formation_det = FormationDetector(fps=fps_override)
    pressing_analyzer = PressingAnalyzer(fps=fps_override)
    possession_analyzer = PossessionAnalyzer(fps=fps_override)
    offside_analyzer = OffsideAnalyzer(fps=fps_override)
    pass_map_analyzer = PassMapAnalyzer(fps=fps_override)

    # Per-track trajectory storage
    trajectories: Dict[int, List[TrajectoryPoint]] = {}
    # Team assignment for aggregation
    track_team:        Dict[int, int] = {}
    # Frame count per track (for ghost-track filtering)
    track_frame_count: Dict[int, int] = {}
    # Best tracks snapshot: frame that had the most valid world-coord players
    best_tracks_snapshot: List[Track] = []
    best_snapshot_score: int = 0

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps_override
    frame_limit = (min(max_frames, total) if total > 0 else max_frames) if max_frames is not None else (total if total > 0 else 999_999)

    annotated_frames: List[np.ndarray] = []
    frame_id = 1

    while cap.isOpened() and frame_id <= frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        detections = detector.detect(frame, frame_id)

        # Tracking
        tracks: List[Track] = tracker.update(detections)

        # Coordinate Mapping (dynamic calibration if available)
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        tracks = coord_mapper.map_tracks(tracks, frame_id=frame_id)

        # Team Classification
        tracks = team_classifier.classify(tracks, frame)

        # Keep track of team assignments and frame counts
        for t in tracks:
            track_frame_count[t.track_id] = track_frame_count.get(t.track_id, 0) + 1
            if t.team_id is not None:
                track_team[t.track_id] = t.team_id

        # Metrics (Players only — must have world coords)
        player_tracks = [
            t for t in tracks
            if t.class_id == CLASS_PLAYER and t.has_world_coords
        ]
        metrics_engine.update(player_tracks)

        # Analytics modules
        formation_det.update(frame_id, tracks)
        pressing_analyzer.update(frame_id, tracks)
        possession_analyzer.update(frame_id, tracks)
        offside_analyzer.update(frame_id, tracks)
        pass_map_analyzer.update(frame_id, tracks)

        # Collect trajectories from metrics engine (only qualified tracks)
        for track_id, traj_list in metrics_engine.trajectories.items():
            trajectories[track_id] = list(traj_list)

        # Track best snapshot for Voronoi (frame with most in-pitch world-coord players)
        snapshot_score = sum(
            1 for t in tracks if t.has_world_coords and t.class_id == CLASS_PLAYER
        )
        if snapshot_score > best_snapshot_score:
            best_snapshot_score = snapshot_score
            best_tracks_snapshot = [t for t in tracks if t.has_world_coords]

        # Compute inverse homography (world→pixel) for offside live overlay
        H_fwd = coord_mapper.get_homography(frame_id)   # pixel→world
        try:
            H_inv = np.linalg.inv(H_fwd)                # world→pixel
        except np.linalg.LinAlgError:
            H_inv = None

        # Annotate frame with bounding boxes
        ann = _annotate_frame(frame.copy(), tracks)

        # Draw live offside lines on frame (both teams)
        h, w = ann.shape[:2]
        for def_team in [1, 2]:
            ann = offside_analyzer.draw_offside_line(
                ann, tracks, H_inv, def_team, w, h
            )
        annotated_frames.append(ann)

        frame_id += 1
        if progress_callback:
            progress_callback(frame_id / frame_limit)

    cap.release()
    coord_mapper.close()  # Release calibration ZIP file handle

    tmp_dir = tempfile.mkdtemp()

    # Pass map
    pass_map_path = str(Path(tmp_dir) / "pass_map.png")
    pass_map_gen = PassMapGenerator(dpi=150)
    pass_map_gen.plot(
        pass_map_analyzer.events,
        pass_map_analyzer._touch_counts,
        pass_map_path,
        title="Pass Map & Possession Events",
    )
    event_summary = pass_map_analyzer.get_event_summary()

    # Voronoi pitch control — use the best-populated frame snapshot
    voronoi_path = str(Path(tmp_dir) / "voronoi.png")
    voronoi_gen = VoronoiGenerator(dpi=150)
    
    valid_snapshot = [
        t for t in best_tracks_snapshot 
        if track_frame_count.get(t.track_id, 0) >= MIN_TRACK_FRAMES
    ]
    
    pitch_control = voronoi_gen.generate(
        valid_snapshot,
        voronoi_path,
        title="Pitch Control — Best Frame"
    )

    # Offside pitch diagram — same best snapshot
    offside_img_path = str(Path(tmp_dir) / "offside.png")
    offside_analyzer.plot_pitch_with_offside(valid_snapshot, offside_img_path)

    # Pressing timeline
    pressing_chart_path = str(Path(tmp_dir) / "pressing.png")
    pressing_analyzer.plot_pressing_timeline(pressing_chart_path)

    # Possession chart
    possession_chart_path = str(Path(tmp_dir) / "possession.png")
    possession_analyzer.plot_possession_breakdown(possession_chart_path)

    # ── Collect results ──────────────────────────────────────────────────────
    player_metrics = metrics_engine.get_metrics()
    metrics_list = []
    for pm in player_metrics.values():
        metrics_list.append({
            "Player ID": pm.track_id,
            "Team": pm.team_id or "?",
            "Distance (km)": round(pm.total_distance_km, 3),
            "Avg Speed (km/h)": round(pm.avg_speed_kmh, 1),
            "Max Speed (km/h)": round(pm.max_speed_kmh, 1),
            "Max Sprint Spd (km/h)": round(pm.max_sprint_speed_kmh, 1),
            "Sprints": pm.sprint_count,
            "Sprint Dist (km)": round(pm.total_sprint_distance_km, 3),
            "Fatigue Index": round(pm.fatigue_index, 3) if pm.fatigue_index is not None else "N/A",
            "Workload Index": round(pm.workload_index, 3),
        })

    formations = {
        1: formation_det.get_majority_formation(1),
        2: formation_det.get_majority_formation(2),
    }

    pressing = {
        1: round(pressing_analyzer.get_pressing_intensity(1), 1),
        2: round(pressing_analyzer.get_pressing_intensity(2), 1),
    }

    return {
        "frames": annotated_frames,
        "metrics": metrics_list,
        "voronoi_path": voronoi_path,
        "pitch_control": pitch_control,
        "pass_map_path": pass_map_path,
        "event_summary": event_summary,
        "offside_img_path": offside_img_path,
        "pressing_chart_path": pressing_chart_path,
        "possession_chart_path": possession_chart_path,
        "formations": formations,
        "pressing": pressing,
        "possession_pct": possession_analyzer.get_possession_percentage(),
        "possession_zones": possession_analyzer.get_possession_stats(),
        "offside_alerts": offside_analyzer.get_alert_counts(),
        "trajectories": trajectories,
        "total_frames": frame_id,
        "fps": actual_fps,
    }


def _annotate_frame(frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
    """Draw bounding boxes with team colours and class labels on a frame."""
    from src.data_models import CLASS_BALL, CLASS_GOALKEEPER, CLASS_REFEREE
    team_colors  = {1: (60, 80, 255), 2: (255, 120, 40), None: (160, 160, 160)}
    class_labels = {0: "P", 1: "GK", 2: "REF", 3: "BALL"}
    class_colors = {2: (50, 220, 80), 3: (0, 220, 255)}   # referee=green, ball=cyan

    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox_px)

        # Colour: class override for ref/ball, else team colour
        if t.class_id in class_colors:
            color = class_colors[t.class_id]
        else:
            color = team_colors.get(t.team_id, (160, 160, 160))

        thickness = 3 if t.class_id == 3 else 2   # thicker box for ball
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        cls_label = class_labels.get(t.class_id, "?")
        label = f"{cls_label} {t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
