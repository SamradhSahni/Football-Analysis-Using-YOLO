"""
src/analytics/possession.py
============================
Ball possession zone analysis per team.

PRD Reference: Goal-20, Task 2.9

Logic:
    - Ball position is taken from the CLASS_BALL track (class_id=3) in each frame.
    - The nearest outfield player is identified using Euclidean distance (world coords).
    - That player's team_id is the possessing team for that frame.
    - Pitch is split into 3 horizontal zones (defensive / middle / attacking thirds).
    - Accumulates possession time (seconds) per team per zone.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data_models import Track, CLASS_BALL


# Zone thresholds along X-axis (0–105m)
_ZONE_THRESHOLDS = {
    "defensive_third":  (0.0,  35.0),
    "middle_third":     (35.0, 70.0),
    "attacking_third":  (70.0, 105.0),
}


def _classify_zone(x_m: float) -> str:
    for zone, (lo, hi) in _ZONE_THRESHOLDS.items():
        if lo <= x_m < hi:
            return zone
    return "middle_third"   # fallback


class PossessionAnalyzer:
    """
    Tracks ball possession per team and per pitch zone over a full match.
    """

    def __init__(self, fps: int = 25, max_ball_distance_m: float = 10.0):
        """
        Parameters
        ----------
        fps : int
            Frames per second of the video.
        max_ball_distance_m : float
            Maximum distance (m) for a player to be considered "in possession".
            Increased to 10m to better handle passes.
        """
        self.fps = fps
        self.max_ball_dist = max_ball_distance_m

        # Accumulated frames: team_id -> zone -> frame_count
        self._zone_frames: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_frames: int = 0

        # Per-frame log for timeline charts
        self._timeline: List[Tuple[int, Optional[int], Optional[str]]] = []  # (frame_id, team_id, zone)
        self._last_team: Optional[int] = None
        self._last_zone: Optional[str] = "middle_third"

    def update(self, frame_id: int, tracks: List[Track]) -> None:
        """
        Process one frame. Identifies ball track, finds nearest player,
        and records possession.

        Parameters
        ----------
        frame_id : int
        tracks : List[Track]
        """
        self._total_frames += 1

        # 1. Find the ball track
        ball_track = next((t for t in tracks if t.class_id == CLASS_BALL and t.centroid_world is not None), None)

        if ball_track is None:
            if self._last_team is not None and self._last_zone is not None:
                self._zone_frames[self._last_team][self._last_zone] += 1
            self._timeline.append((frame_id, self._last_team, self._last_zone))
            return

        ball_x, ball_y = ball_track.centroid_world
        zone = _classify_zone(ball_x)

        # 2. Find outfield players with world coords
        players = [t for t in tracks if t.class_id != CLASS_BALL and t.centroid_world is not None and t.team_id in [1, 2]]

        if not players:
            if self._last_team is not None:
                self._zone_frames[self._last_team][zone] += 1
            self._timeline.append((frame_id, self._last_team, zone))
            self._last_zone = zone
            return

        # 3. Nearest player by Euclidean distance
        distances = [
            np.hypot(t.centroid_world[0] - ball_x, t.centroid_world[1] - ball_y)
            for t in players
        ]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]
        nearest = players[min_idx]

        if min_dist <= self.max_ball_dist:
            possessing_team = nearest.team_id
            self._last_team = possessing_team
        else:
            possessing_team = self._last_team

        self._last_zone = zone

        # 4. Accumulate
        if possessing_team is not None:
            self._zone_frames[possessing_team][zone] += 1
        self._timeline.append((frame_id, possessing_team, zone))

    def get_possession_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Returns possession time in seconds per team per zone.

        Returns
        -------
        Dict[team_id → Dict[zone → seconds]]
        """
        stats: Dict[int, Dict[str, float]] = {}
        for team_id in [1, 2]:
            stats[team_id] = {
                zone: self._zone_frames[team_id][zone] / self.fps
                for zone in _ZONE_THRESHOLDS
            }
        return stats

    def get_possession_percentage(self) -> Dict[int, float]:
        """
        Returns overall possession % per team (contested frames excluded).

        Returns
        -------
        Dict[team_id → float (0-100)]
        """
        contested_frames = sum(1 for _, team, _ in self._timeline if team is None)
        effective_frames = self._total_frames - contested_frames

        if effective_frames == 0:
            return {1: 50.0, 2: 50.0}

        result = {}
        for team_id in [1, 2]:
            team_total = sum(self._zone_frames[team_id].values())
            result[team_id] = 100.0 * team_total / effective_frames
        return result

    def plot_possession_breakdown(self, output_path: str) -> None:
        """
        Saves a stacked bar chart of possession time per zone per team.
        """
        stats = self.get_possession_stats()
        zones = list(_ZONE_THRESHOLDS.keys())
        zone_labels = ["Defensive Third", "Middle Third", "Attacking Third"]

        team1_vals = [stats[1][z] for z in zones]
        team2_vals = [stats[2][z] for z in zones]

        x = np.arange(len(zones))
        width = 0.35

        bg_color = "#1e1e1e"
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        bars1 = ax.bar(x - width / 2, team1_vals, width, label="Team 1", color="#ff4d4d", alpha=0.85)
        bars2 = ax.bar(x + width / 2, team2_vals, width, label="Team 2", color="#4d9eff", alpha=0.85)

        # Value labels
        for bar in bars1 + bars2:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}s",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", color="white", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(zone_labels, color="white")
        ax.set_ylabel("Time in Possession (seconds)", color="white")
        ax.set_title("Ball Possession by Zone", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#333333", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        logging.info(f"Possession chart saved to {output_path}")

    def plot_possession_pie(self, output_path: str) -> None:
        """
        Saves an overall possession % pie chart.
        """
        pct = self.get_possession_percentage()
        bg_color = "#1e1e1e"
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        labels = [f"Team 1\n{pct[1]:.1f}%", f"Team 2\n{pct[2]:.1f}%"]
        sizes = [pct[1], pct[2]]
        colors = ["#ff4d4d", "#4d9eff"]

        wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                               startangle=90, textprops={"color": "white", "fontsize": 12})

        ax.set_title("Overall Ball Possession", color="white", fontsize=14)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        logging.info(f"Possession pie chart saved to {output_path}")
