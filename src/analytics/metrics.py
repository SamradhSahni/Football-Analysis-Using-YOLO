"""
src/analytics/metrics.py
========================
Stage 6 of the Football Tracker Pipeline.
Calculates physiological and tactical performance metrics for players.

Task 3.1 Fixes:
  - Physics-based speed clamping (max 12 m/s = human sprint limit)
  - Noise gate for implausible position jumps (tracker ID switches)
  - Fatigue index now computed (first-half vs second-half avg speed ratio)
  - Max sprint speed now tracked correctly
  - finalize_sprints() called automatically in get_metrics()
  - Sprint distance accumulated in metres

PRD Reference : Section 6, Task 1.7, GOAL-07, GOAL-08
Outputs       : `data_models.PlayerMetrics` per player
"""

from typing import Dict, List, Optional
import numpy as np

from src.data_models import Track, TrajectoryPoint, PlayerMetrics


# ── Physics constants ──────────────────────────────────────────────────────
MAX_PLAUSIBLE_SPEED_MS   = 12.0   # ~43 km/h — human sprint world record limit
MAX_PLAUSIBLE_JUMP_M     = 8.0    # Max metres a player can move between frames
SPRINT_THRESHOLD_MS      = 5.0    # 18 km/h in m/s (relaxed to capture short bursts)
SPRINT_MIN_DURATION_S    = 0.2    # Sprint must last ≥ 0.2 seconds to be counted
MIN_TRACK_FRAMES         = 25     # Ignore ghost tracks shorter than this (1.0 second at 25fps)


class MetricsEngine:
    """
    Computes rolling and cumulative metrics for players based on their trajectories.
    Maintains per-track history to calculate speed, distance, sprints, zones.
    """

    def __init__(self, fps: float = 25.0) -> None:
        self.fps = fps
        self._dt = 1.0 / fps          # time delta per frame (seconds)
        self._sprint_frames_req = int(SPRINT_MIN_DURATION_S * fps)

        # Per-track state
        self.trajectories:       Dict[int, List[TrajectoryPoint]] = {}
        self.total_distances:    Dict[int, float] = {}   # metres
        self.sprint_counts:      Dict[int, int]   = {}
        self._sprint_frames:     Dict[int, int]   = {}
        self._sprint_dist_m:     Dict[int, float] = {}   # metres during valid sprints
        self._current_sprint_dist_m: Dict[int, float] = {} # buffer for ongoing sprints
        self._max_sprint_spd:    Dict[int, float] = {}   # m/s during any sprint
        self.team_affiliations:  Dict[int, int]   = {}
        self.zone_times:         Dict[int, Dict[str, float]] = {}

        # Pitch zones (x-axis in metres)
        self.ZONE_DEF_MAX = 35.0
        self.ZONE_MID_MAX = 70.0

        # EWMA alpha for speed smoothing (higher = more responsive, more noise)
        self._alpha = 0.60

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, tracks: List[Track]) -> None:
        """Process one frame's player tracks and update all metrics."""
        for t in tracks:
            if not t.has_world_coords:
                continue

            tid = t.track_id
            self._init_track(tid)

            if t.team_id is not None:
                self.team_affiliations[tid] = t.team_id

            tp = TrajectoryPoint.from_track(t, self.fps)
            history = self.trajectories[tid]

            if not history:
                tp = tp.with_speed(speed_ms=0.0)
            else:
                last_tp = history[-1]

                dx = tp.x_m - last_tp.x_m
                dy = tp.y_m - last_tp.y_m
                dist_m = float(np.hypot(dx, dy))

                # ── Noise gate ─────────────────────────────────────────────
                # If the position jump is physically impossible (tracker ID
                # switch or calibration glitch), clamp the contribution.
                if dist_m > MAX_PLAUSIBLE_JUMP_M:
                    dist_m = 0.0   # discard jump but keep the trajectory point

                dt_s = tp.timestamp_s - last_tp.timestamp_s

                if dt_s > 0 and dist_m >= 0:
                    raw_speed_ms = dist_m / dt_s

                    # ── Physics cap ────────────────────────────────────────
                    raw_speed_ms = min(raw_speed_ms, MAX_PLAUSIBLE_SPEED_MS)

                    # EWMA smoothing
                    prev_spd = last_tp.speed_ms or 0.0
                    smoothed = self._alpha * raw_speed_ms + (1 - self._alpha) * prev_spd

                    tp = tp.with_speed(speed_ms=smoothed)
                    self.total_distances[tid] += dist_m if dist_m > 0 else 0.0
                else:
                    tp = tp.with_speed(speed_ms=last_tp.speed_ms or 0.0)

            self.trajectories[tid].append(tp)
            self._update_sprint(tid, tp)
            self._update_zones(tid, tp)

    # ── Sprint tracking ───────────────────────────────────────────────────────

    def _update_sprint(self, tid: int, tp: TrajectoryPoint) -> None:
        """Track consecutive sprint frames and accumulate sprint distance only if valid."""
        speed_ms = tp.speed_ms or 0.0

        if speed_ms >= SPRINT_THRESHOLD_MS:
            self._sprint_frames[tid] += 1
            dist_this_frame = speed_ms * self._dt
            
            if self._sprint_frames[tid] == self._sprint_frames_req:
                # Sprint just became valid: count it and commit the buffered distance
                self.sprint_counts[tid] += 1
                self._sprint_dist_m[tid] += self._current_sprint_dist_m.get(tid, 0.0) + dist_this_frame
            elif self._sprint_frames[tid] > self._sprint_frames_req:
                # Already a valid sprint: commit directly
                self._sprint_dist_m[tid] += dist_this_frame
            else:
                # Not yet valid: buffer it
                self._current_sprint_dist_m[tid] = self._current_sprint_dist_m.get(tid, 0.0) + dist_this_frame

            # Track peak sprint speed
            if speed_ms > self._max_sprint_spd[tid]:
                self._max_sprint_spd[tid] = speed_ms
        else:
            # End of sprint window
            self._sprint_frames[tid] = 0
            self._current_sprint_dist_m[tid] = 0.0

    def finalize_sprints(self) -> None:
        """
        Sprints are now tallied exactly when they reach the required duration,
        so no finalization is needed. Kept for API compatibility.
        """
        pass

    # ── Zone tracking ─────────────────────────────────────────────────────────

    def _update_zones(self, tid: int, tp: TrajectoryPoint) -> None:
        x = tp.x_m
        if x <= self.ZONE_DEF_MAX:
            self.zone_times[tid]["defensive_third"]  += self._dt
        elif x <= self.ZONE_MID_MAX:
            self.zone_times[tid]["middle_third"]     += self._dt
        else:
            self.zone_times[tid]["attacking_third"]  += self._dt

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_track(self, tid: int) -> None:
        if tid not in self.trajectories:
            self.trajectories[tid]    = []
            self.total_distances[tid] = 0.0
            self.sprint_counts[tid]   = 0
            self._sprint_frames[tid]  = 0
            self._sprint_dist_m[tid]  = 0.0
            self._current_sprint_dist_m[tid] = 0.0
            self._max_sprint_spd[tid] = 0.0
            self.zone_times[tid]      = {
                "defensive_third": 0.0,
                "middle_third":    0.0,
                "attacking_third": 0.0,
            }

    # ── Final metrics compilation ─────────────────────────────────────────────

    def get_metrics(self) -> Dict[int, PlayerMetrics]:
        """
        Compile and return PlayerMetrics for all tracks.
        Automatically finalizes any in-progress sprints.
        """
        self.finalize_sprints()

        metrics_map = {}
        for tid, history in self.trajectories.items():
            # Discard ghost tracks (detected for only a few frames)
            if len(history) < MIN_TRACK_FRAMES:
                continue

            if not history:
                continue

            speeds_kmh = [p.speed_kmh for p in history if p.speed_kmh is not None]
            if not speeds_kmh:
                continue

            max_speed_kmh = max(speeds_kmh)

            # Avg speed — exclude the first (always 0) and any zero padding
            active_speeds = [s for s in speeds_kmh[1:] if s > 0.0]
            avg_speed_kmh = float(np.mean(active_speeds)) if active_speeds else 0.0

            dist_km = self.total_distances[tid] / 1000.0
            sprints  = self.sprint_counts[tid]

            # ── Fatigue index ─────────────────────────────────────────────
            # Compare avg speed in first half of clip vs second half.
            # > 1.0 means player got faster (unusual), < 1.0 means fatigue.
            mid = len(history) // 2
            if mid > 5:
                first_half_spd = [p.speed_kmh for p in history[:mid] if p.speed_kmh]
                second_half_spd = [p.speed_kmh for p in history[mid:] if p.speed_kmh]
                if first_half_spd and second_half_spd:
                    fatigue = float(np.mean(second_half_spd)) / max(float(np.mean(first_half_spd)), 0.01)
                    fatigue = round(fatigue, 3)
                else:
                    fatigue = None
            else:
                fatigue = None

            # ── Workload index ────────────────────────────────────────────
            workload = min(1.0, (sprints * 0.1) + (dist_km / 10.0))

            metrics_map[tid] = PlayerMetrics(
                track_id=tid,
                team_id=self.team_affiliations.get(tid),
                total_distance_km=round(dist_km, 3),
                avg_speed_kmh=round(avg_speed_kmh, 2),
                max_speed_kmh=round(max_speed_kmh, 2),
                sprint_count=sprints,
                max_sprint_speed_kmh=round(self._max_sprint_spd[tid] * 3.6, 2),
                total_sprint_distance_km=round(self._sprint_dist_m[tid] / 1000.0, 4),
                time_in_zones=self.zone_times[tid],
                role_adherence_pct=None,
                fatigue_index=fatigue,
                workload_index=round(workload, 3),
            )

        return metrics_map
