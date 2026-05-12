"""
src/analytics/events.py
========================
Correlates SoccerNet action spotting events with player tracking data.

PRD Reference: Goal-19, Task 2.8

SoccerNet action spotting annotations are JSON files with the format:
{
    "UrlLocal": "...",
    "annotations": [
        {
            "gameTime": "1 - 12:34",
            "label": "Goal",
            "position": "754000",   <- milliseconds from kick-off
            "team": "home",
            "visibility": "visible"
        }, ...
    ]
}
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data_models import Track


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class MatchEvent:
    """A single SoccerNet action spotting annotation."""
    label: str
    timestamp_ms: int
    game_time: str
    team: str
    visibility: str
    frame_id: Optional[int] = None          # set once mapped to a frame


@dataclass
class EventSnapshot:
    """A match event combined with the player state at that moment."""
    event: MatchEvent
    tracks: List[Track] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event Loader
# ---------------------------------------------------------------------------

class EventLoader:
    """
    Loads SoccerNet action spotting JSON annotations.
    """

    def load(self, json_path: str) -> List[MatchEvent]:
        """
        Parses a SoccerNet Labels-v2.json file into MatchEvent objects.

        Parameters
        ----------
        json_path : str
            Absolute path to the annotation JSON file.

        Returns
        -------
        List[MatchEvent]
        """
        path = Path(json_path)
        if not path.exists():
            logging.warning(f"Event annotation file not found: {json_path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = []
        for ann in data.get("annotations", []):
            try:
                timestamp_ms = int(ann.get("position", 0))
                events.append(MatchEvent(
                    label=ann.get("label", "Unknown"),
                    timestamp_ms=timestamp_ms,
                    game_time=ann.get("gameTime", ""),
                    team=ann.get("team", ""),
                    visibility=ann.get("visibility", ""),
                ))
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping malformed annotation: {ann} — {e}")

        logging.info(f"Loaded {len(events)} events from {json_path}")
        return events

    def map_to_frames(self, events: List[MatchEvent], fps: int = 25) -> List[MatchEvent]:
        """
        Converts event timestamps (ms) to frame IDs.

        Parameters
        ----------
        events : List[MatchEvent]
        fps : int
            Video frames per second.

        Returns
        -------
        List[MatchEvent] with frame_id populated.
        """
        for event in events:
            event.frame_id = int((event.timestamp_ms / 1000.0) * fps)
        return events


# ---------------------------------------------------------------------------
# Event Correlator
# ---------------------------------------------------------------------------

class EventCorrelator:
    """
    Correlates MatchEvents with player positions at each event timestamp.
    """

    def __init__(self, fps: int = 25, snapshot_window_frames: int = 5):
        """
        Parameters
        ----------
        fps : int
            Frames per second.
        snapshot_window_frames : int
            How many frames +/- around the event to use if exact frame missing.
        """
        self.fps = fps
        self.window = snapshot_window_frames

        # Rolling buffer: frame_id -> List[Track]
        self._track_buffer: Dict[int, List[Track]] = {}

    def ingest_frame(self, frame_id: int, tracks: List[Track]) -> None:
        """
        Buffer the tracks for a given frame. Call this every frame.
        """
        self._track_buffer[frame_id] = tracks

        # Prune old frames to avoid unbounded memory growth (keep last 500 frames)
        if len(self._track_buffer) > 500:
            oldest = min(self._track_buffer.keys())
            del self._track_buffer[oldest]

    def correlate(self, events: List[MatchEvent]) -> List[EventSnapshot]:
        """
        For each event, find the closest buffered frame and attach the tracks.

        Parameters
        ----------
        events : List[MatchEvent]
            Events with frame_id populated (use EventLoader.map_to_frames first).

        Returns
        -------
        List[EventSnapshot]
        """
        snapshots = []
        buffered_frames = sorted(self._track_buffer.keys())

        if not buffered_frames:
            logging.warning("No frames buffered. Call ingest_frame() before correlate().")
            return snapshots

        for event in events:
            target = event.frame_id
            if target is None:
                continue

            # Find closest buffered frame within the window
            closest = min(buffered_frames, key=lambda f: abs(f - target))
            if abs(closest - target) > self.window:
                logging.debug(f"Event at frame {target} ({event.label}) — no close frame in buffer (closest={closest})")
                snapshot_tracks = []
            else:
                snapshot_tracks = self._track_buffer[closest]

            snapshots.append(EventSnapshot(event=event, tracks=snapshot_tracks))

        logging.info(f"Correlated {len(snapshots)} events with player snapshots.")
        return snapshots

    def render_event_snapshot(
        self,
        snapshot: EventSnapshot,
        output_path: str,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        """
        Renders an annotated 2D pitch image for a single event.

        Parameters
        ----------
        snapshot : EventSnapshot
        output_path : str
            Path to save the PNG image.
        """
        bg_color = "#1e1e1e"
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=bg_color)
        ax.set_facecolor(bg_color)

        # Draw pitch
        ax.plot([0, 0, pitch_length, pitch_length, 0],
                [0, pitch_width, pitch_width, 0, 0], color="white", linewidth=2)
        ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color="white", linewidth=2)
        center = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color="white", fill=False, linewidth=2)
        ax.add_patch(center)

        # Plot players
        team_colors = {1: "#ff4d4d", 2: "#4d9eff", None: "#aaaaaa"}
        for track in snapshot.tracks:
            if track.centroid_world is None:
                continue
            x, y = track.centroid_world
            color = team_colors.get(track.team_id, "#aaaaaa")
            ax.plot(x, y, 'o', color=color, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(str(track.track_id), (x, y), color="white",
                        fontsize=7, ha="center", va="center")

        # Title with event info
        title = f"{snapshot.event.label} | {snapshot.event.game_time} | {snapshot.event.team.capitalize()}"
        ax.set_title(title, color="white", fontsize=13, pad=10)

        # Legend
        patches = [
            mpatches.Patch(color="#ff4d4d", label="Team 1"),
            mpatches.Patch(color="#4d9eff", label="Team 2"),
        ]
        ax.legend(handles=patches, facecolor="#333333", labelcolor="white", loc="lower right")

        ax.set_xlim(-5, pitch_length + 5)
        ax.set_ylim(-5, pitch_width + 5)
        ax.set_aspect("equal")
        ax.axis("off")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        logging.info(f"Event snapshot saved: {output_path}")

    def render_all_snapshots(self, snapshots: List[EventSnapshot], output_dir: str) -> List[str]:
        """
        Renders PNG snapshot images for all events.

        Returns
        -------
        List[str] — paths to all generated images.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for idx, snap in enumerate(snapshots):
            label_slug = snap.event.label.lower().replace(" ", "_")
            fname = out_dir / f"event_{idx:03d}_{label_slug}.png"
            self.render_event_snapshot(snap, str(fname))
            paths.append(str(fname))

        return paths

    def build_event_table(self, snapshots: List[EventSnapshot]) -> List[Dict]:
        """
        Builds a list of dicts for use in CSV/PDF reports.

        Returns
        -------
        List[Dict] with keys: label, game_time, team, players_visible,
                              avg_speed_kmh, nearest_player_id
        """
        rows = []
        for snap in snapshots:
            n_players = len(snap.tracks)
            speeds = [t.class_id for t in snap.tracks]  # class_id as proxy if speed not available
            
            # Compute avg speed if TrajectoryPoints were attached (future integration)
            rows.append({
                "label": snap.event.label,
                "game_time": snap.event.game_time,
                "team": snap.event.team,
                "frame_id": snap.event.frame_id,
                "players_visible": n_players,
            })
        return rows
