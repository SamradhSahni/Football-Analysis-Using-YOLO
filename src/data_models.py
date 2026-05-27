"""
src/data_models.py
==================
Shared dataclasses used as the inter-stage communication contract
across the entire Football Tracker pipeline.

PRD Reference : Section 8
Rule          : Every stage reads/writes ONLY these structures.
                No stage may invent alternative class names or field names.

Class hierarchy
---------------
Detection       — raw YOLOv8 output per bounding box (Stage 2 → Stage 3)
Track           — ByteTrack-assigned identity per box  (Stage 3 → Stage 4/5)
TrajectoryPoint — real-world position + speed per frame (Stage 4/5 → Stage 6)
PlayerMetrics   — final aggregated match stats per player (Stage 5 → Stage 6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CLASS-ID CONSTANTS
# ---------------------------------------------------------------------------
# YOLO class IDs (0-indexed, used throughout the pipeline)
CLASS_PLAYER     = 0
CLASS_GOALKEEPER = 1
CLASS_REFEREE    = 2
CLASS_BALL       = 3

CLASS_ID_TO_NAME: Dict[int, str] = {
    CLASS_PLAYER:     "player",
    CLASS_GOALKEEPER: "goalkeeper",
    CLASS_REFEREE:    "referee",
    CLASS_BALL:       "ball",
}

# SoccerNet MOT gt.txt class IDs (1-indexed) → YOLO class IDs (0-indexed)
SOCCERNET_TO_YOLO_CLASS: Dict[int, int] = {
    1: CLASS_PLAYER,
    2: CLASS_GOALKEEPER,
    3: CLASS_REFEREE,
    4: CLASS_BALL,
}

# Team IDs
TEAM_A    = 0
TEAM_B    = 1
TEAM_REF  = 2   # referee group

TEAM_ID_TO_NAME: Dict[int, str] = {
    TEAM_A:   "team_A",
    TEAM_B:   "team_B",
    TEAM_REF: "referee",
}


# ---------------------------------------------------------------------------
# DETECTION
# Stage 2 output (PlayerDetector) → Stage 3 input (PlayerTracker)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """
    A single bounding box prediction from the YOLOv8 detector for one frame.

    Fields
    ------
    frame_id   : int   — 1-indexed frame number (matches MOT convention)
    bbox_px    : tuple — (x1, y1, x2, y2) pixel coordinates, top-left/bottom-right
    confidence : float — detector confidence score in [0.0, 1.0]
    class_id   : int   — YOLO class index: 0=player 1=goalkeeper 2=referee 3=ball
    class_name : str   — human-readable class label (derived from class_id)

    Units
    -----
    bbox_px  → pixels
    confidence → dimensionless [0, 1]
    """

    frame_id:   int
    bbox_px:    Tuple[float, float, float, float]   # (x1, y1, x2, y2) px
    confidence: float                                # [0.0, 1.0]
    class_id:   int                                  # 0=player … 3=ball
    class_name: str                                  # "player", "goalkeeper", etc.

    # ------------------------------------------------------------------
    # Derived properties (computed, not stored)
    # ------------------------------------------------------------------
    @property
    def width_px(self) -> float:
        """Bounding box width in pixels."""
        return self.bbox_px[2] - self.bbox_px[0]

    @property
    def height_px(self) -> float:
        """Bounding box height in pixels."""
        return self.bbox_px[3] - self.bbox_px[1]

    @property
    def area_px(self) -> float:
        """Bounding box area in square pixels."""
        return self.width_px * self.height_px

    @property
    def centroid_px(self) -> Tuple[float, float]:
        """Centre point (cx, cy) in pixels."""
        x1, y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Detection.confidence must be in [0, 1], got {self.confidence}"
            )
        if self.class_id not in CLASS_ID_TO_NAME:
            raise ValueError(
                f"Detection.class_id must be one of {list(CLASS_ID_TO_NAME.keys())}, "
                f"got {self.class_id}"
            )
        x1, y1, x2, y2 = self.bbox_px
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Detection.bbox_px must satisfy x2>x1 and y2>y1, got {self.bbox_px}"
            )

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------
    @classmethod
    def from_yolo(
        cls,
        frame_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        class_id: int,
    ) -> "Detection":
        """Construct a Detection directly from raw YOLO output values."""
        return cls(
            frame_id=frame_id,
            bbox_px=(x1, y1, x2, y2),
            confidence=confidence,
            class_id=class_id,
            class_name=CLASS_ID_TO_NAME.get(class_id, "unknown"),
        )


# ---------------------------------------------------------------------------
# TRACK
# Stage 3 output (PlayerTracker) → Stage 4/5 input
# ---------------------------------------------------------------------------
@dataclass
class Track:
    """
    A ByteTrack-assigned detection with a persistent identity across frames.

    Fields
    ------
    track_id        : int   — unique persistent player identity (ByteTrack ID)
    frame_id        : int   — 1-indexed frame number
    bbox_px         : tuple — (x1, y1, x2, y2) pixel coordinates
    centroid_px     : tuple — (cx, cy) bounding box centre in pixels
    centroid_world  : tuple — (x_m, y_m) real-world pitch coordinates in metres
                              None until Stage 4 (Homography) runs
    class_id        : int   — 0=player 1=goalkeeper 2=referee 3=ball
    team_id         : int   — 0=team_A 1=team_B 2=referee  (None until Stage 2.1)

    Units
    -----
    centroid_px    → pixels
    centroid_world → metres (FIFA pitch: 105 m × 68 m, origin = top-left corner)
    """

    track_id:       int
    frame_id:       int
    bbox_px:        Tuple[float, float, float, float]           # (x1,y1,x2,y2) px
    centroid_px:    Tuple[float, float]                         # (cx, cy) px
    centroid_world: Optional[Tuple[float, float]] = None        # (x_m, y_m) metres
    class_id:       int = CLASS_PLAYER
    team_id:        Optional[int] = None                        # set by TeamSeparator

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def class_name(self) -> str:
        return CLASS_ID_TO_NAME.get(self.class_id, "unknown")

    @property
    def team_name(self) -> Optional[str]:
        return TEAM_ID_TO_NAME.get(self.team_id) if self.team_id is not None else None

    @property
    def has_world_coords(self) -> bool:
        """True once Stage 4 (Homography) has populated centroid_world."""
        return self.centroid_world is not None

    # ------------------------------------------------------------------
    # Factory helper — build from a Detection + ByteTrack ID
    # ------------------------------------------------------------------
    @classmethod
    def from_detection(cls, detection: Detection, track_id: int) -> "Track":
        """Construct a Track from a Detection and its assigned ByteTrack ID."""
        return cls(
            track_id=track_id,
            frame_id=detection.frame_id,
            bbox_px=detection.bbox_px,
            centroid_px=detection.centroid_px,
            class_id=detection.class_id,
        )


# ---------------------------------------------------------------------------
# TRAJECTORY POINT
# Stage 4/5 output → Stage 6 input
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryPoint:
    """
    One time-stamped real-world position + speed sample on a player's trajectory.

    Created by TrajectoryBuffer after homography transforms real-world coords,
    and speed fields are populated by compute_speed() after Gaussian smoothing.

    Fields
    ------
    frame_id     : int   — 1-indexed frame number
    timestamp_s  : float — elapsed time in seconds since start of clip
    x_m          : float — real-world x position in metres (along pitch length)
    y_m          : float — real-world y position in metres (along pitch width)
    speed_ms     : float — smoothed instantaneous speed in m/s  (None pre-smoothing)
    speed_kmh    : float — speed in km/h = speed_ms × 3.6        (None pre-smoothing)

    Units
    -----
    x_m, y_m   → metres  (FIFA pitch: 0≤x≤105, 0≤y≤68)
    speed_ms    → m/s
    speed_kmh   → km/h
    timestamp_s → seconds
    """

    frame_id:    int
    timestamp_s: float          # seconds from clip start
    x_m:         float          # metres — pitch x-axis (0 → 105 m)
    y_m:         float          # metres — pitch y-axis (0 → 68 m)
    speed_ms:    Optional[float] = None   # m/s  — set after Gaussian smoothing
    speed_kmh:   Optional[float] = None   # km/h — set after Gaussian smoothing

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.timestamp_s < 0:
            raise ValueError(
                f"TrajectoryPoint.timestamp_s must be ≥ 0, got {self.timestamp_s}"
            )
        if self.speed_ms is not None and self.speed_ms < 0:
            raise ValueError(
                f"TrajectoryPoint.speed_ms must be ≥ 0, got {self.speed_ms}"
            )
        # Consistency: if one speed is set, both should be
        if (self.speed_ms is None) != (self.speed_kmh is None):
            raise ValueError(
                "TrajectoryPoint: speed_ms and speed_kmh must both be set or both None"
            )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_track(cls, track: Track, fps: float) -> "TrajectoryPoint":
        """
        Create a TrajectoryPoint from a Track that already has centroid_world set.
        speed fields are left None until compute_speed() runs.
        """
        if track.centroid_world is None:
            raise ValueError(
                "TrajectoryPoint.from_track: track.centroid_world must be set "
                "(run Stage 4 Homography first)"
            )
        x_m, y_m = track.centroid_world
        return cls(
            frame_id=track.frame_id,
            timestamp_s=(track.frame_id - 1) / fps,
            x_m=x_m,
            y_m=y_m,
        )

    def with_speed(self, speed_ms: float) -> "TrajectoryPoint":
        """Return a new TrajectoryPoint with speed fields populated."""
        return TrajectoryPoint(
            frame_id=self.frame_id,
            timestamp_s=self.timestamp_s,
            x_m=self.x_m,
            y_m=self.y_m,
            speed_ms=round(speed_ms, 4),
            speed_kmh=round(speed_ms * 3.6, 4),
        )


# ---------------------------------------------------------------------------
# PLAYER METRICS
# Stage 5 final output → Stage 6 (CSV, PDF, heatmap)
# ---------------------------------------------------------------------------
@dataclass
class PlayerMetrics:
    """
    Aggregated match-level performance statistics for a single tracked player.

    Populated by Stage 5 (metrics/) after the full trajectory has been processed.
    Written to outputs/csv/match_summary.csv by ReportGenerator.

    Fields
    ------
    track_id                 : int   — ByteTrack persistent player ID
    team_id                  : int   — 0=team_A, 1=team_B, 2=referee, None=unknown
    total_distance_km        : float — total path length in km  (GOAL-05)
    avg_speed_kmh            : float — mean speed over tracked frames in km/h (GOAL-04)
    max_speed_kmh            : float — peak instantaneous speed in km/h (GOAL-04)
    sprint_count             : int   — number of sprint events ≥ threshold (GOAL-07)
    max_sprint_speed_kmh     : float — fastest sprint speed in km/h (GOAL-07)
    total_sprint_distance_km : float — cumulative sprint distance in km (GOAL-07)
    time_in_zones            : dict  — zone_name → seconds in that zone (GOAL-11)
    role_adherence_pct       : float — % frames in expected tactical zone (GOAL-12)
    fatigue_index            : float — avg_speed_2nd_half / avg_speed_1st_half (GOAL-13)
    workload_index           : float — composite load score 0.0–1.0 (GOAL-14)

    Units
    -----
    *_km       → kilometres
    *_kmh      → km/h
    time_in_zones values → seconds
    *_pct      → percent (0–100)
    fatigue_index → dimensionless ratio
    workload_index → dimensionless [0, 1]
    """

    track_id:                  int
    team_id:                   Optional[int]

    # Phase 1 core metrics
    total_distance_km:         float           # km          (GOAL-05)
    avg_speed_kmh:             float           # km/h        (GOAL-04)
    max_speed_kmh:             float           # km/h        (GOAL-04)
    sprint_count:              int             # count       (GOAL-07)
    max_sprint_speed_kmh:      float           # km/h        (GOAL-07)
    total_sprint_distance_km:  float           # km          (GOAL-07)
    time_in_zones:             Dict[str, float]  # zone → s  (GOAL-11)

    # Phase 2 advanced metrics (None until computed)
    role_adherence_pct:  Optional[float] = None   # %        (GOAL-12)
    fatigue_index:       Optional[float] = None   # ratio    (GOAL-13)
    workload_index:      Optional[float] = None   # [0,1]    (GOAL-14)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def team_name(self) -> Optional[str]:
        return TEAM_ID_TO_NAME.get(self.team_id) if self.team_id is not None else None

    @property
    def is_fatigued(self) -> Optional[bool]:
        """True if second half speed dropped vs first half (fatigue_index < 1.0)."""
        if self.fatigue_index is None:
            return None
        return self.fatigue_index < 1.0

    # ------------------------------------------------------------------
    # CSV export helper — returns ordered dict matching GOAL-09 column spec
    # ------------------------------------------------------------------
    def to_csv_row(self) -> Dict[str, object]:
        """
        Returns a flat dict matching the exact CSV column order from PRD GOAL-09.
        Columns: track_id, team_id, total_distance_km, avg_speed_kmh,
                 max_speed_kmh, sprint_count, max_sprint_speed_kmh,
                 total_sprint_distance_km, time_defensive_third_s,
                 time_middle_third_s, time_attacking_third_s, role_adherence_pct
        """
        return {
            "track_id":                  self.track_id,
            "team_id":                   self.team_id,
            "total_distance_km":         round(self.total_distance_km, 4),
            "avg_speed_kmh":             round(self.avg_speed_kmh, 4),
            "max_speed_kmh":             round(self.max_speed_kmh, 4),
            "sprint_count":              self.sprint_count,
            "max_sprint_speed_kmh":      round(self.max_sprint_speed_kmh, 4),
            "total_sprint_distance_km":  round(self.total_sprint_distance_km, 4),
            "time_defensive_third_s":    round(self.time_in_zones.get("defensive_third", 0.0), 2),
            "time_middle_third_s":       round(self.time_in_zones.get("middle_third", 0.0), 2),
            "time_attacking_third_s":    round(self.time_in_zones.get("attacking_third", 0.0), 2),
            "role_adherence_pct":        self.role_adherence_pct,
            "fatigue_index":             self.fatigue_index,
            "workload_index":            self.workload_index,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.total_distance_km < 0:
            raise ValueError("PlayerMetrics.total_distance_km must be ≥ 0")
        if self.avg_speed_kmh < 0:
            raise ValueError("PlayerMetrics.avg_speed_kmh must be ≥ 0")
        if self.max_speed_kmh < 0:
            raise ValueError("PlayerMetrics.max_speed_kmh must be ≥ 0")
        if self.sprint_count < 0:
            raise ValueError("PlayerMetrics.sprint_count must be ≥ 0")
        if self.workload_index is not None and not (0.0 <= self.workload_index <= 1.0):
            raise ValueError(
                f"PlayerMetrics.workload_index must be in [0, 1], got {self.workload_index}"
            )
        if self.role_adherence_pct is not None and not (0.0 <= self.role_adherence_pct <= 100.0):
            raise ValueError(
                f"PlayerMetrics.role_adherence_pct must be in [0, 100], "
                f"got {self.role_adherence_pct}"
            )
