"""
src/analytics/formation.py
==========================
Detects team formations based on spatial clustering of players.

PRD Reference: Goal-15, Task 2.4
"""

import collections
import logging
import warnings
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from src.data_models import Track, CLASS_PLAYER


class FormationDetector:
    """
    Detects the playing formation (e.g., 4-3-3, 4-4-2) of teams dynamically 
    using K-Means clustering on the depth axis (X-axis).
    """

    def __init__(self, fps: int = 25, window_minutes: int = 5):
        """
        Parameters
        ----------
        fps : int
            Frames per second of the video.
        window_minutes : int
            Rolling window duration to calculate the majority formation.
        """
        self.fps = fps
        self.window_frames = window_minutes * 60 * fps
        self.history: Dict[int, collections.deque] = {1: collections.deque(), 2: collections.deque()}

    def detect_instantaneous(self, tracks: List[Track]) -> Dict[int, str]:
        """
        Calculates the exact formation for a single frame.

        Parameters
        ----------
        tracks : List[Track]
            List of active tracks in the current frame.

        Returns
        -------
        Dict[int, str]
            Mapping of team_id to formation string (e.g. {1: "4-3-3", 2: "4-4-2"}).
        """
        formations = {}
        
        for team_id in [1, 2]:
            # Filter outfield players (exclude Goalkeeper if strictly classed, otherwise class_id=0 handles it if GKs are 1)
            # PRD uses class_id=0 for players, 1 for goalkeepers.
            team_tracks = [
                t for t in tracks 
                if t.team_id == team_id 
                and t.centroid_world is not None 
                and t.class_id == CLASS_PLAYER
            ]
            
            if len(team_tracks) < 5:
                continue

            x_coords = np.array([t.centroid_world[0] for t in team_tracks]).reshape(-1, 1)

            # Clamp n_clusters to the number of distinct player positions
            n_clusters = min(3, len(x_coords))
            if n_clusters < 2:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = kmeans.fit_predict(x_coords)

            # Use bincount so counts always has exactly n_clusters entries,
            # even when a cluster ends up empty (avoids IndexError).
            counts = np.bincount(labels, minlength=n_clusters)
            cluster_centers = kmeans.cluster_centers_.flatten()

            # Sort clusters by ascending X (defensive → attacking)
            sorted_indices   = np.argsort(cluster_centers)
            formation_counts = [int(counts[i]) for i in sorted_indices]

            # Normalise to exactly 10 outfield players
            total_detected = sum(formation_counts)
            if total_detected == 0:
                continue
            normalized = [int(round((c / total_detected) * 10)) for c in formation_counts]

            # Fix rounding drift
            diff = 10 - sum(normalized)
            if diff != 0:
                normalized[int(np.argmax(normalized))] += diff

            # Pad to 3 lines if n_clusters == 2
            while len(normalized) < 3:
                normalized.insert(1, 0)

            # If fewest players are at the back, team is attacking left→right; reverse
            if normalized[0] < normalized[-1]:
                normalized.reverse()

            form_str = "-".join(str(n) for n in normalized)
            formations[team_id] = form_str
            
        return formations

    def update(self, frame_id: int, tracks: List[Track]) -> None:
        """
        Updates the rolling history window with instantaneous formations.
        """
        inst = self.detect_instantaneous(tracks)
        for team, form in inst.items():
            self.history[team].append((frame_id, form))
            
            # Clean old frames outside the 5-minute window
            while self.history[team] and (frame_id - self.history[team][0][0]) > self.window_frames:
                self.history[team].popleft()

    def get_majority_formation(self, team_id: int) -> str:
        """
        Returns the most common formation played over the 5-minute window.
        """
        if not self.history.get(team_id):
            return "Unknown"
        forms = [form for _, form in self.history[team_id]]
        most_common = collections.Counter(forms).most_common(1)
        return most_common[0][0]
