"""
src/analytics/voronoi.py
========================
Generates Voronoi Pitch Control maps to visualise spatial dominance.

Task 4.1 Update:
  - Uses mplsoccer Pitch for professional quality pitch overlay
  - Filled Voronoi regions clipped to pitch boundary
  - Team control percentage computed from region areas
  - Player markers with track ID labels
  - Summary stats (team1_pct / team2_pct) returned alongside the image

PRD Reference: Goal-09, Task 2.2
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi

from src.data_models import Track, CLASS_PLAYER, CLASS_GOALKEEPER

# ── Pitch constants ────────────────────────────────────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
BG_COLOR     = "#0d1117"

TEAM_COLORS = {
    1: ("#ff4d4d", (1.0, 0.30, 0.30, 0.40)),   # red, rgba fill
    2: ("#4d9eff", (0.30, 0.60, 1.0,  0.40)),   # blue, rgba fill
}


def _clip_polygon_to_pitch(polygon: np.ndarray) -> Optional[np.ndarray]:
    """
    Sutherland-Hodgman algorithm to clip a polygon to the pitch rectangle.
    Returns the clipped polygon vertices or None if the polygon is entirely outside.
    """
    def inside(p, edge_start, edge_end):
        return ((edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) -
                (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])) >= 0

    def intersection(p1, p2, edge_start, edge_end):
        dc = [edge_start[0] - edge_end[0], edge_start[1] - edge_end[1]]
        dp = [p1[0] - p2[0], p1[1] - p2[1]]
        n1 = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0]
        n2 = p1[0] * p2[1] - p1[1] * p2[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0] + 1e-12)
        return [(n1 * dp[0] - n2 * dc[0]) * n3,
                (n1 * dp[1] - n2 * dc[1]) * n3]

    # Edges in CCW order so that "inside" = left of directed edge = pitch interior
    edges = [
        ([0, 0],                       [PITCH_LENGTH, 0]),            # bottom
        ([PITCH_LENGTH, 0],            [PITCH_LENGTH, PITCH_WIDTH]),  # right
        ([PITCH_LENGTH, PITCH_WIDTH],  [0, PITCH_WIDTH]),             # top
        ([0, PITCH_WIDTH],             [0, 0]),                       # left
    ]

    output = list(polygon)
    for edge_start, edge_end in edges:
        if not output:
            return None
        input_list = output
        output = []
        for i, pt in enumerate(input_list):
            prev = input_list[i - 1]
            if inside(pt, edge_start, edge_end):
                if not inside(prev, edge_start, edge_end):
                    output.append(intersection(prev, pt, edge_start, edge_end))
                output.append(pt)
            elif inside(prev, edge_start, edge_end):
                output.append(intersection(prev, pt, edge_start, edge_end))

    if not output:
        return None
    return np.array(output)


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class VoronoiGenerator:
    """
    Generates Voronoi pitch control maps using mplsoccer.
    Shows which player 'owns' which zone of the pitch.
    """

    def __init__(
        self,
        pitch_length: float = PITCH_LENGTH,
        pitch_width:  float = PITCH_WIDTH,
        dpi: int = 180,
    ) -> None:
        self.length = pitch_length
        self.width  = pitch_width
        self.dpi    = dpi

    def generate(
        self,
        tracks: List[Track],
        output_path: str,
        title: str = "Voronoi Pitch Control",
    ) -> Dict[str, float]:
        """
        Generates and saves a Voronoi pitch control map.

        Parameters
        ----------
        tracks      : List of active tracks (must have centroid_world and team_id)
        output_path : Path to save PNG
        title       : Chart title

        Returns
        -------
        Dict with keys 'team1_pct' and 'team2_pct' (% of pitch area controlled).
        Returns empty dict if not enough data.
        """
        valid = [
            t for t in tracks
            if t.centroid_world is not None
            and t.team_id in (1, 2)
            and t.class_id in (CLASS_PLAYER, CLASS_GOALKEEPER)
            and 0 <= t.centroid_world[0] <= self.length
            and 0 <= t.centroid_world[1] <= self.width
        ]

        if len(valid) < 3:
            logging.info("Voronoi: fewer than 3 valid players — skipping.")
            return {}

        points = np.array([[t.centroid_world[0], t.centroid_world[1]] for t in valid])
        teams  = [t.team_id for t in valid]
        ids    = [t.track_id for t in valid]

        # Add mirror dummy points far outside to close boundary regions
        margin = 500.0
        dummy  = np.array([
            [-margin, -margin], [-margin, self.width + margin],
            [self.length + margin, self.width + margin],
            [self.length + margin, -margin],
        ])
        all_pts = np.vstack([points, dummy])
        vor = Voronoi(all_pts)

        # ── Draw pitch ───────────────────────────────────────────────────────
        try:
            from mplsoccer import Pitch
            pitch = Pitch(
                pitch_type="custom",
                pitch_length=self.length,
                pitch_width=self.width,
                pitch_color=BG_COLOR,
                line_color="#c8d6e5",
                linewidth=1.5,
                goal_type="box",
            )
            fig, ax = pitch.draw(figsize=(13, 8))
        except ImportError:
            fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG_COLOR)
            ax.set_facecolor(BG_COLOR)

        fig.patch.set_facecolor(BG_COLOR)

        # ── Fill Voronoi regions ─────────────────────────────────────────────
        area_controlled = {1: 0.0, 2: 0.0}

        for point_idx, region_idx in enumerate(vor.point_region):
            if point_idx >= len(valid):   # skip dummy points
                continue

            region = vor.regions[region_idx]
            if -1 in region or len(region) < 3:
                continue   # infinite or degenerate region

            polygon_verts = vor.vertices[region]
            clipped = _clip_polygon_to_pitch(polygon_verts)
            if clipped is None or len(clipped) < 3:
                continue

            team_id = teams[point_idx]
            _, fill_rgba = TEAM_COLORS[team_id]

            patch = Polygon(clipped, closed=True, facecolor=fill_rgba,
                            edgecolor="white", linewidth=0.6, alpha=0.9, zorder=2)
            ax.add_patch(patch)

            area_controlled[team_id] += _polygon_area(clipped)

        # ── Player markers ───────────────────────────────────────────────────
        for i, (pt, team_id, pid) in enumerate(zip(points, teams, ids)):
            hex_color, _ = TEAM_COLORS[team_id]
            ax.plot(pt[0], pt[1], "o", color=hex_color, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            ax.annotate(str(pid), (pt[0], pt[1]),
                        color="white", fontsize=6.5, ha="center", va="center",
                        fontweight="bold", zorder=6)

        # ── Control percentage ────────────────────────────────────────────────
        total_area = area_controlled[1] + area_controlled[2]
        if total_area > 0:
            t1_pct = round(100.0 * area_controlled[1] / total_area, 1)
            t2_pct = round(100.0 * area_controlled[2] / total_area, 1)
        else:
            t1_pct = t2_pct = 0.0

        # ── Legend & title ────────────────────────────────────────────────────
        leg_patches = [
            mpatches.Patch(color=TEAM_COLORS[1][0], label=f"Team 1 — {t1_pct}% pitch control"),
            mpatches.Patch(color=TEAM_COLORS[2][0], label=f"Team 2 — {t2_pct}% pitch control"),
        ]
        ax.legend(handles=leg_patches, facecolor="#1e1e2e", labelcolor="white",
                  loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=2, fontsize=11, framealpha=0.8)

        ax.set_title(title, color="#e6edf3", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlim(-2, self.length + 2)
        ax.set_ylim(-4, self.width + 2)
        ax.set_aspect("equal")
        ax.axis("off")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logging.info(f"Voronoi map saved → {output_path}")

        return {"team1_pct": t1_pct, "team2_pct": t2_pct}
