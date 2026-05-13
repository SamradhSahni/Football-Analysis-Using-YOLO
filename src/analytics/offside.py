"""
src/analytics/offside.py
========================
Indicative offside line detection and alert counting.

PRD Reference: Goal-22, Task 3.1

⚠️  IMPORTANT: All output from this module is labeled "INDICATIVE ONLY".
    This is not a substitute for VAR or official match officiating.
    Results depend entirely on the accuracy of player detection, tracking,
    and homography calibration.

Logic (PRD Section 15):
    - The offside line is defined by the second-to-last defender
      (the last outfield player before the goalkeeper).
    - Computed separately for each team's defensive half.
    - A player is "potentially offside" if they are ahead of the
      offside line in the opponent's half at the moment the ball
      is played forward.
    - Counts are accumulated as indicative offside alerts.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data_models import Track, CLASS_BALL, CLASS_PLAYER, CLASS_GOALKEEPER


_DISCLAIMER = "⚠ INDICATIVE ONLY — Not for official use"


class OffsideAnalyzer:
    """
    Computes the indicative offside line and detects potential offside
    positions for each team.
    """

    def __init__(self, fps: int = 25, pitch_length: float = 105.0):
        """
        Parameters
        ----------
        fps : int
            Frames per second.
        pitch_length : float
            Length of the pitch in metres (X-axis).
        """
        self.fps = fps
        self.pitch_length = pitch_length

        # Per-player offside alert count
        self._alert_counts: Dict[int, int] = defaultdict(int)

        # Per-frame history: frame_id -> {team_id -> offside_line_x}
        self._offside_line_history: Dict[int, Dict[int, float]] = {}

    def _get_defenders(self, tracks: List[Track], defending_team: int) -> List[Track]:
        """
        Returns all outfield players from the defending team, sorted by
        their X position (closest to their own goal first).
        
        We consider class_id 0 (player) and 1 (goalkeeper) as defenders.
        Team 1 defends the left goal (low X), Team 2 defends the right goal (high X).
        """
        defenders = [
            t for t in tracks
            if t.team_id == defending_team
            and t.centroid_world is not None
            and t.class_id in [CLASS_PLAYER, CLASS_GOALKEEPER]
        ]
        # Sort: Team 1 defenders sort ascending X (closest to goal at X=0)
        #       Team 2 defenders sort descending X (closest to goal at X=105)
        if defending_team == 1:
            defenders.sort(key=lambda t: t.centroid_world[0])
        else:
            defenders.sort(key=lambda t: t.centroid_world[0], reverse=True)
        return defenders

    def get_offside_line(self, tracks: List[Track], defending_team: int) -> Optional[float]:
        """
        Returns the X-position of the offside line for the given defending team.
        
        The offside line is the closest to the defending team's goal line among:
        1. The second-to-last defender
        2. The ball
        3. The halfway line (52.5m)
        """
        defenders = self._get_defenders(tracks, defending_team)
        if len(defenders) < 2:
            return None
            
        # Second-to-last from the goal = index 1 (0=GK or last man, 1=second last)
        second_last_def_x = defenders[1].centroid_world[0]
        
        # Find the ball
        ball_tracks = [t for t in tracks if t.class_id == CLASS_BALL and t.centroid_world is not None]
        ball_x = ball_tracks[0].centroid_world[0] if ball_tracks else None
        
        halfway_line = self.pitch_length / 2.0
        
        if defending_team == 1:
            # Defending left goal (X=0). Line is min(second_last_def, ball, halfway)
            offside_x = min(second_last_def_x, halfway_line)
            if ball_x is not None:
                offside_x = min(offside_x, ball_x)
            return offside_x
        else:
            # Defending right goal (X=105). Line is max(second_last_def, ball, halfway)
            offside_x = max(second_last_def_x, halfway_line)
            if ball_x is not None:
                offside_x = max(offside_x, ball_x)
            return offside_x

    def update(self, frame_id: int, tracks: List[Track]) -> Dict[int, List[int]]:
        """
        Process one frame. Detects potential offside positions and
        accumulates alert counts.

        Parameters
        ----------
        frame_id : int
        tracks : List[Track]

        Returns
        -------
        Dict[team_id → List[track_id]] — players flagged as potentially offside this frame.
        """
        frame_alerts: Dict[int, List[int]] = {1: [], 2: []}
        offside_lines: Dict[int, float] = {}

        for attacking_team in [1, 2]:
            defending_team = 2 if attacking_team == 1 else 1
            offside_x = self.get_offside_line(tracks, defending_team)
            if offside_x is None:
                continue

            offside_lines[attacking_team] = offside_x

            # Check attacking team players
            attackers = [
                t for t in tracks
                if t.team_id == attacking_team
                and t.centroid_world is not None
                and t.class_id == CLASS_PLAYER
            ]

            for attacker in attackers:
                ax = attacker.centroid_world[0]

                # Team 1 attacks toward high X (right goal)
                # Team 2 attacks toward low X (left goal)
                if attacking_team == 1 and ax > offside_x:
                    self._alert_counts[attacker.track_id] += 1
                    frame_alerts[1].append(attacker.track_id)
                elif attacking_team == 2 and ax < offside_x:
                    self._alert_counts[attacker.track_id] += 1
                    frame_alerts[2].append(attacker.track_id)

        self._offside_line_history[frame_id] = offside_lines
        return frame_alerts

    def get_alert_counts(self) -> Dict[int, int]:
        """
        Returns per-player indicative offside alert frame counts.
        """
        return dict(self._alert_counts)

    def draw_offside_line(
        self,
        frame_bgr: np.ndarray,
        tracks: List[Track],
        homography_matrix: Optional[np.ndarray],
        defending_team: int,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Draws a dashed offside line on the raw video frame (pixel space).

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw video frame to annotate.
        tracks : List[Track]
            Active tracks for the frame.
        homography_matrix : np.ndarray or None
            3×3 homography from world→pixel. If None, skip drawing.
        defending_team : int
        image_width, image_height : int
            Frame dimensions.

        Returns
        -------
        np.ndarray : Annotated frame.
        """
        offside_x_m = self.get_offside_line(tracks, defending_team)
        if offside_x_m is None or homography_matrix is None:
            return frame_bgr

        # Project two world points on the offside line to pixel space
        world_pts = np.array([
            [[offside_x_m, 0.0]],
            [[offside_x_m, 68.0]],
        ], dtype=np.float32)

        try:
            pixel_pts = cv2.perspectiveTransform(world_pts, homography_matrix)
        except cv2.error:
            return frame_bgr

        p1 = tuple(int(v) for v in pixel_pts[0][0])
        p2 = tuple(int(v) for v in pixel_pts[1][0])

        # Draw dashed line manually
        color = (0, 200, 255)   # amber
        frame_bgr = _draw_dashed_line(frame_bgr, p1, p2, color, thickness=2, dash_len=20)

        # Disclaimer label
        cv2.putText(
            frame_bgr,
            _DISCLAIMER,
            (10, image_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        return frame_bgr

    def plot_pitch_with_offside(
        self,
        tracks: List[Track],
        output_path: str,
        frame_id: int = 0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        """
        Renders a 2D pitch diagram with offside lines overlaid using mplsoccer.
        Saved as a PNG.
        """
        try:
            from mplsoccer import Pitch
            pitch = Pitch(
                pitch_type="custom",
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                pitch_color="#0d1117",
                line_color="#c8d6e5",
                linewidth=1.5,
                goal_type="box",
            )
            fig, ax = pitch.draw(figsize=(13, 8))
        except ImportError:
            bg_color = "#0d1117"
            fig, ax = plt.subplots(figsize=(13, 8), facecolor=bg_color)
            ax.set_facecolor(bg_color)
            ax.plot([0, 0, pitch_length, pitch_length, 0],
                    [0, pitch_width, pitch_width, 0, 0], color="white", lw=2)
            ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color="white", lw=2)
            ax.add_patch(plt.Circle((pitch_length/2, pitch_width/2), 9.15,
                                    color="white", fill=False, lw=2))

        fig.patch.set_facecolor("#0d1117")

        # Plot players
        team_colors = {1: "#ff4d4d", 2: "#4d9eff", None: "#aaaaaa"}
        for track in tracks:
            if track.centroid_world is None:
                continue
            x, y = track.centroid_world
            if not (0 <= x <= pitch_length and 0 <= y <= pitch_width):
                continue
            color = team_colors.get(track.team_id, "#aaaaaa")
            ax.plot(x, y, 'o', color=color, markersize=11,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            ax.annotate(str(track.track_id), (x, y), color="white",
                        fontsize=7, ha="center", va="center", zorder=6)

        # Draw offside lines for both teams
        offside_drawn = False
        for attacking_team, defending_team, line_color, label in [
            (1, 2, "#ff4d4d", "Team 1 offside line"),
            (2, 1, "#4d9eff", "Team 2 offside line"),
        ]:
            offside_x = self.get_offside_line(tracks, defending_team)
            if offside_x is not None:
                ax.axvline(x=offside_x, color=line_color, linestyle="--",
                           linewidth=2.5, alpha=0.9, label=label, zorder=4)
                # Shade the offside zone
                if attacking_team == 1:
                    ax.axvspan(offside_x, pitch_length, alpha=0.06,
                               color=line_color, zorder=3)
                else:
                    ax.axvspan(0, offside_x, alpha=0.06,
                               color=line_color, zorder=3)
                offside_drawn = True

        if not offside_drawn:
            ax.text(pitch_length/2, pitch_width/2,
                    "Insufficient players\nto compute offside line",
                    color="#888888", fontsize=11, ha="center", va="center")

        # Disclaimer
        ax.text(pitch_length/2, -4.5, _DISCLAIMER, color="#ffaa00",
                fontsize=9, ha="center", style="italic")

        ax.set_xlim(-3, pitch_length + 3)
        ax.set_ylim(-7, pitch_width + 4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Indicative Offside Lines", color="#e6edf3", fontsize=14,
                     fontweight="bold", pad=10)
        if offside_drawn:
            ax.legend(facecolor="#1e1e2e", labelcolor="white",
                      loc="upper right", fontsize=10)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        logging.info(f"Offside diagram saved to {output_path}")


# ---------------------------------------------------------------------------
# Utility: Dashed line on OpenCV frame
# ---------------------------------------------------------------------------

def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 15,
) -> np.ndarray:
    """Draws a dashed line between two pixel-space points on an OpenCV image."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return img

    steps = int(dist / dash_len)
    for i in range(steps):
        if i % 2 == 0:  # draw only even segments
            start = (int(x1 + (x2 - x1) * i / steps),
                     int(y1 + (y2 - y1) * i / steps))
            end = (int(x1 + (x2 - x1) * (i + 1) / steps),
                   int(y1 + (y2 - y1) * (i + 1) / steps))
            cv2.line(img, start, end, color, thickness, cv2.LINE_AA)
    return img
