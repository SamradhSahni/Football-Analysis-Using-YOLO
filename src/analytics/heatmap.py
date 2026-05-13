"""
src/analytics/heatmap.py
========================
Generates 2D spatial density heatmaps of player movements.

Task 3.2 Update:
  - Replaced hand-drawn pitch lines with mplsoccer Pitch (professional quality)
  - Per-team heatmaps (Team 1 vs Team 2) side-by-side with distinct colours
  - Sprint heatmap uses plasma colormap on the same pitch layout
  - All heatmaps use real-world pitch coordinates (metres) from CoordinateMapper

PRD Reference: Goal-08, Task 2.1 | Goal-23, Task 3.2
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from src.data_models import TrajectoryPoint

try:
    from mplsoccer import Pitch, VerticalPitch
    _HAS_MPLSOCCER = True
except ImportError:
    _HAS_MPLSOCCER = False
    logging.warning("mplsoccer not found — falling back to basic pitch drawing.")


# ── Pitch constants ───────────────────────────────────────────────────────────
PITCH_LENGTH = 105.0  # metres
PITCH_WIDTH  = 68.0   # metres
BG_COLOR     = "#0d1117"
LINE_COLOR   = "#c8d6e5"


def _make_pitch() -> "Pitch":
    """Return a consistently styled mplsoccer Pitch object."""
    if _HAS_MPLSOCCER:
        return Pitch(
            pitch_type="custom",
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            pitch_color=BG_COLOR,
            line_color=LINE_COLOR,
            linewidth=1.5,
            goal_type="box",
        )
    return None


def _draw_basic_pitch(ax: plt.Axes) -> None:
    """Fallback manual pitch drawing (used only if mplsoccer is missing)."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    ax.set_facecolor(BG_COLOR)
    for xs, ys in [
        ([0, 0, L, L, 0], [0, W, W, 0, 0]),
        ([L/2, L/2], [0, W]),
        ([0, 16.5, 16.5, 0], [13.84, 13.84, 54.16, 54.16]),
        ([L, L-16.5, L-16.5, L], [13.84, 13.84, 54.16, 54.16]),
    ]:
        ax.plot(xs, ys, color=LINE_COLOR, linewidth=1.5)
    ax.add_patch(plt.Circle((L/2, W/2), 9.15, color=LINE_COLOR, fill=False, lw=1.5))
    ax.set_xlim(-2, L+2)
    ax.set_ylim(-2, W+2)
    ax.set_aspect("equal")
    ax.axis("off")


def _coords_to_bin_grid(
    x_coords: List[float],
    y_coords: List[float],
    sigma: float = 3.5,
    grid_x: int = 105,
    grid_y: int = 68,
) -> np.ndarray:
    """
    Bin positions into a grid and apply Gaussian smoothing.
    Sigma is adaptive: sparse data gets a wider spread so the heatmap
    is always visible (no invisible pin-dot problem on short clips).
    """
    n = len(x_coords)
    if n < 50:
        effective_sigma = max(sigma, 12.0)
    elif n < 300:
        # Blend from 12 down to the caller-supplied sigma over [50, 300] pts
        t = (n - 50) / 250.0
        effective_sigma = max(sigma, 12.0 * (1 - t) + sigma * t)
    else:
        effective_sigma = sigma

    grid = np.zeros((grid_y, grid_x), dtype=float)
    for x, y in zip(x_coords, y_coords):
        xi = int(np.clip(x, 0, PITCH_LENGTH - 0.01))
        yi = int(np.clip(y, 0, PITCH_WIDTH  - 0.01))
        grid[yi, xi] += 1.0

    smoothed = gaussian_filter(grid, sigma=effective_sigma)
    if smoothed.max() > 0:
        smoothed /= smoothed.max()
    return smoothed


def _save_fig(fig: plt.Figure, output_path: str, dpi: int = 180) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


# ── Main HeatmapGenerator ────────────────────────────────────────────────────

class HeatmapGenerator:
    """
    Generates KDE / binned Gaussian heatmaps for player trajectories,
    overlaid on a professional football pitch (mplsoccer).
    """

    def __init__(
        self,
        pitch_length: float = PITCH_LENGTH,
        pitch_width:  float = PITCH_WIDTH,
        cmap: str = "YlOrRd",
        bandwidth: float = 1.5,
        dpi: int = 180,
    ) -> None:
        self.length    = pitch_length
        self.width     = pitch_width
        self.cmap      = cmap
        self.bandwidth = bandwidth
        self.dpi       = dpi

    def generate(
        self,
        trajectories: List[TrajectoryPoint],
        output_path: str,
        title: str = "Player Movement Heatmap",
        team_color: Optional[str] = None,
    ) -> None:
        """
        Generate and save a single heatmap for a list of trajectory points.

        Parameters
        ----------
        trajectories : List[TrajectoryPoint]
        output_path  : str — path to save PNG
        title        : str — chart title
        team_color   : str — optional colour tint for the title bar
        """
        if not trajectories:
            logging.warning("No trajectory data for heatmap — skipping.")
            return

        x_coords = [p.x_m for p in trajectories if p.x_m is not None]
        y_coords = [p.y_m for p in trajectories if p.y_m is not None]
        if len(x_coords) < 5:
            logging.warning("Insufficient coordinates for heatmap.")
            return

        pitch = _make_pitch()

        if pitch is not None:
            fig, ax = pitch.draw(figsize=(12, 7))
            fig.patch.set_facecolor(BG_COLOR)

            grid = _coords_to_bin_grid(x_coords, y_coords, sigma=self.bandwidth)
            # imshow origin='lower' aligns y=0 at bottom (matches pitch origin)
            ax.imshow(
                grid,
                extent=[0, self.length, 0, self.width],
                origin="lower",
                cmap=self.cmap,
                alpha=0.75,
                aspect="auto",
                zorder=1,
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
            _draw_basic_pitch(ax)
            grid = _coords_to_bin_grid(x_coords, y_coords, sigma=self.bandwidth)
            ax.imshow(
                grid,
                extent=[0, self.length, 0, self.width],
                origin="lower",
                cmap=self.cmap,
                alpha=0.75,
                aspect="auto",
            )

        tc = team_color or "#e6edf3"
        ax.set_title(title, color=tc, fontsize=14, fontweight="bold", pad=10)
        _save_fig(fig, output_path, self.dpi)
        logging.info(f"Heatmap saved → {output_path}")

    def generate_team_comparison(
        self,
        team_a_trajectories: List[TrajectoryPoint],
        team_b_trajectories: List[TrajectoryPoint],
        output_path: str,
    ) -> None:
        """
        Generate a side-by-side heatmap comparing Team 1 and Team 2.
        Team 1 uses red tones, Team 2 uses blue tones.
        """
        pitch = _make_pitch()

        if pitch is not None:
            fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(22, 7))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(22, 7), facecolor=BG_COLOR)
            for ax in axes:
                _draw_basic_pitch(ax)

        fig.patch.set_facecolor(BG_COLOR)

        configs = [
            (team_a_trajectories, axes[0], "Reds",  "#ff6b6b", "Team 1 — Movement Density"),
            (team_b_trajectories, axes[1], "Blues",  "#4d9eff", "Team 2 — Movement Density"),
        ]

        for traj, ax, cmap, tc, title in configs:
            if traj:
                x = [p.x_m for p in traj if p.x_m is not None]
                y = [p.y_m for p in traj if p.y_m is not None]
                if len(x) >= 5:
                    grid = _coords_to_bin_grid(x, y, sigma=self.bandwidth)
                    ax.imshow(
                        grid,
                        extent=[0, self.length, 0, self.width],
                        origin="lower",
                        cmap=cmap,
                        alpha=0.75,
                        aspect="auto",
                        zorder=1,
                    )
            ax.set_title(title, color=tc, fontsize=13, fontweight="bold", pad=8)

        fig.suptitle("Team Heatmap Comparison", color="#e6edf3", fontsize=16, y=1.02)
        _save_fig(fig, output_path, self.dpi)
        logging.info(f"Team comparison heatmap saved → {output_path}")


# ── Sprint HeatmapGenerator ───────────────────────────────────────────────────

class SprintHeatmapGenerator(HeatmapGenerator):
    """
    Extension that filters trajectory points to only sprint frames
    (speed_ms >= sprint_threshold_ms) before plotting.
    """

    def __init__(
        self,
        sprint_threshold_ms: float = 5.56,   # 20 km/h
        pitch_length: float = PITCH_LENGTH,
        pitch_width:  float = PITCH_WIDTH,
        cmap: str = "plasma",
        bandwidth: float = 1.5,
        dpi: int = 180,
    ) -> None:
        super().__init__(pitch_length, pitch_width, cmap, bandwidth, dpi)
        self.sprint_threshold_ms = sprint_threshold_ms

    def _filter_sprints(self, trajectories: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        return [
            p for p in trajectories
            if p.speed_ms is not None and p.speed_ms >= self.sprint_threshold_ms
        ]

    def generate(  # type: ignore[override]
        self,
        trajectories: List[TrajectoryPoint],
        output_path: str,
        title: str = "Sprint Heatmap",
        team_color: Optional[str] = None,
    ) -> None:
        """Generate sprint-filtered heatmap."""
        sprint_pts = self._filter_sprints(trajectories)
        if not sprint_pts:
            logging.info("No sprint frames detected — sprint heatmap skipped.")
            return
        super().generate(sprint_pts, output_path, title, team_color or "#f5a623")

    def generate_player_sprint_heatmap(
        self,
        track_id: int,
        trajectories: List[TrajectoryPoint],
        output_dir: str,
    ) -> Optional[str]:
        out_path = str(Path(output_dir) / f"sprints_{track_id}.png")
        self.generate(
            trajectories,
            out_path,
            title=f"Player {track_id} — Sprint Zones (≥{self.sprint_threshold_ms:.1f} m/s)",
        )
        return out_path if Path(out_path).exists() else None

    def generate_team_sprint_heatmap(
        self,
        team_id: int,
        player_trajectories: Dict[int, List[TrajectoryPoint]],
        output_dir: str,
    ) -> Optional[str]:
        all_pts: List[TrajectoryPoint] = []
        for traj in player_trajectories.values():
            all_pts.extend(self._filter_sprints(traj))
        if not all_pts:
            return None
        out_path = str(Path(output_dir) / f"sprints_team_{team_id}.png")
        super().generate(
            all_pts, out_path,
            title=f"Team {team_id} — Sprint Aggregate",
        )
        return out_path if Path(out_path).exists() else None

    def generate_all(
        self,
        team_trajectories: Dict[int, Dict[int, List[TrajectoryPoint]]],
        output_dir: str,
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        for team_id, player_map in team_trajectories.items():
            for track_id, traj in player_map.items():
                p = self.generate_player_sprint_heatmap(track_id, traj, output_dir)
                if p:
                    paths[f"player_{track_id}"] = p
            p = self.generate_team_sprint_heatmap(team_id, player_map, output_dir)
            if p:
                paths[f"team_{team_id}"] = p
        return paths
