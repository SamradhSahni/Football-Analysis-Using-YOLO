"""
src/analytics/pressing.py
=========================
Calculates pressing intensity and defensive line depth metrics.

PRD Reference: Goal-16 (Pressing Intensity), Goal-17 (Defensive Line Depth)
Tasks 2.5, 2.6
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data_models import Track


class PressingAnalyzer:
    """
    Computes team pressing intensity and defensive line depth.

    Pressing Intensity Formula (PRD Section 15):
        At each frame, identify the N closest defenders (pressing_k) from
        the pressing team to the centroid of the opponent team.
        
        Press score (frame) = 1 / (1 + mean_distance_m)
        
        Normalized to a 0-100 scale:
        pressing_intensity = 100 * mean(press_score per frame)

    Defensive Line Depth:
        Mean X-position of the 4 rearmost players per team per frame.
    """

    def __init__(
        self,
        pressing_k: int = 6,
        fps: int = 25,
        smoothing_window: int = 125,   # 5-second window at 25fps
        half_duration_frames: int = 60 * 45 * 25,
    ):
        """
        Parameters
        ----------
        pressing_k : int
            Number of closest defenders used to compute pressing score.
        fps : int
            Frames per second.
        smoothing_window : int
            Smoothing window size in frames for chart output (5s at 25fps).
        half_duration_frames : int
            Duration of a half match in frames (45min * 60s * fps).
        """
        self.pressing_k = pressing_k
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.half_frames = half_duration_frames

        # Per-frame accumulators
        self._press_score: Dict[int, List[float]] = defaultdict(list)  # team_id -> [press_score]
        self._defense_depth: Dict[int, List[float]] = defaultdict(list)  # team_id -> [x_m]
        self._frame_ids: List[int] = []

    def update(self, frame_id: int, tracks: List[Track]) -> None:
        """
        Process all tracks for one frame.

        Parameters
        ----------
        frame_id : int
            Current frame number.
        tracks : List[Track]
            All active tracks with populated centroid_world.
        """
        self._frame_ids.append(frame_id)

        for pressing_team in [1, 2]:
            opponent_team = 2 if pressing_team == 1 else 1

            pressing_players = [t for t in tracks if t.team_id == pressing_team and t.centroid_world is not None]
            opponent_players = [t for t in tracks if t.team_id == opponent_team and t.centroid_world is not None]

            # --- Pressing Intensity ---
            if pressing_players and opponent_players:
                # Target: centroid of the opponent team
                opp_xs = [t.centroid_world[0] for t in opponent_players]
                opp_ys = [t.centroid_world[1] for t in opponent_players]
                opp_centroid = (np.mean(opp_xs), np.mean(opp_ys))

                # Distances from all pressing team players to opponent centroid
                distances = [
                    np.hypot(t.centroid_world[0] - opp_centroid[0], t.centroid_world[1] - opp_centroid[1])
                    for t in pressing_players
                ]

                # Use the K closest
                k = min(self.pressing_k, len(distances))
                nearest_dists = sorted(distances)[:k]
                mean_dist = np.mean(nearest_dists)

                # Press score: inverse distance, 0-1 scale
                press_score = 1.0 / (1.0 + mean_dist)
                self._press_score[pressing_team].append(press_score)
            else:
                self._press_score[pressing_team].append(0.0)

            # --- Defensive Line Depth ---
            if pressing_players:
                # Rearmost 4 players (minimum X for team attacking right-to-left, max otherwise)
                # We simply take the min X of the 4 lowest-X players (defensive third)
                x_positions = sorted([t.centroid_world[0] for t in pressing_players])
                rear_4 = x_positions[:4] if len(x_positions) >= 4 else x_positions
                self._defense_depth[pressing_team].append(np.mean(rear_4))
            else:
                self._defense_depth[pressing_team].append(np.nan)

    def get_pressing_intensity(self, team_id: int, half: Optional[int] = None) -> float:
        """
        Returns normalized pressing intensity (0-100) for a team.

        Parameters
        ----------
        team_id : int
            Team identifier (1 or 2).
        half : int, optional
            If 1 or 2, returns intensity for that half only.
            
        Returns
        -------
        float : Pressing intensity in range [0, 100].
        """
        scores = self._press_score.get(team_id, [])
        if not scores:
            return 0.0

        if half is not None:
            n = len(scores)
            mid = n // 2
            scores = scores[:mid] if half == 1 else scores[mid:]

        return float(np.mean(scores)) * 100.0

    def get_defensive_line_depth(self, team_id: int) -> float:
        """
        Returns the mean defensive line depth in metres (X-axis).
        """
        depths = [d for d in self._defense_depth.get(team_id, []) if not np.isnan(d)]
        return float(np.mean(depths)) if depths else 0.0

    def _smooth(self, data: List[float]) -> np.ndarray:
        """Rolling mean smoothing."""
        arr = np.array(data, dtype=float)
        if len(arr) < self.smoothing_window:
            return arr
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(arr, kernel, mode='same')

    def plot_pressing_timeline(self, output_path: str) -> None:
        """
        Generates and saves a timeline chart of pressing intensity per team.
        """
        bg_color = "#1e1e1e"
        fig, ax = plt.subplots(figsize=(14, 5), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        colors = {1: "#ff4d4d", 2: "#4d9eff"}
        labels = {1: "Team 1", 2: "Team 2"}

        for team_id in [1, 2]:
            scores = self._press_score.get(team_id, [])
            if not scores:
                continue
            smoothed = self._smooth(scores) * 100.0
            time_s = np.array(self._frame_ids) / self.fps
            ax.plot(time_s, smoothed, color=colors[team_id], linewidth=1.5, label=labels[team_id], alpha=0.9)

        # Halftime line
        ax.axvline(x=self.half_frames / self.fps, color="white", linestyle="--", linewidth=1, alpha=0.5, label="Half Time")

        ax.set_xlabel("Time (seconds)", color="white")
        if self._frame_ids:
            ax.set_xlim(left=0, right=max(self._frame_ids) / self.fps)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Pressing Intensity (0-100)", color="white")
        ax.set_title("Pressing Intensity Over Time", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#333333", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        logging.info(f"Pressing timeline saved to {output_path}")

    def plot_defensive_depth_timeline(self, output_path: str) -> None:
        """
        Generates and saves a timeline chart of defensive line depth per team.
        """
        bg_color = "#1e1e1e"
        fig, ax = plt.subplots(figsize=(14, 5), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        colors = {1: "#ff4d4d", 2: "#4d9eff"}
        labels = {1: "Team 1", 2: "Team 2"}

        for team_id in [1, 2]:
            depths = self._defense_depth.get(team_id, [])
            if not depths:
                continue
            # Replace NaN for smoothing
            arr = np.array(depths, dtype=float)
            nan_mask = np.isnan(arr)
            arr[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), arr[~nan_mask]) if any(~nan_mask) else 0.0
            smoothed = self._smooth(arr.tolist())
            time_s = np.array(self._frame_ids) / self.fps
            ax.plot(time_s, smoothed, color=colors[team_id], linewidth=1.5, label=labels[team_id], alpha=0.9)

        ax.axvline(x=self.half_frames / self.fps, color="white", linestyle="--", linewidth=1, alpha=0.5, label="Half Time")

        ax.set_xlabel("Time (seconds)", color="white")
        if self._frame_ids:
            ax.set_xlim(left=0, right=max(self._frame_ids) / self.fps)
        else:
            ax.set_xlim(left=0)
        ax.set_ylabel("Defensive Line Depth (m)", color="white")
        ax.set_title("Defensive Line Depth Over Time", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#333333", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        logging.info(f"Defensive depth timeline saved to {output_path}")
