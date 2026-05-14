"""
src/classification/classifier.py
================================
Stage 5 of the Football Tracker Pipeline.
Responsible for classifying players into two teams using color clustering.

Task 3.4 — Team Persistence Layer:
  - Per-track vote accumulator: once a player's team assignment is seen
    N frames consistently, it is locked in and no longer changes.
  - Confidence score computed from cluster distance ratio — uncertain
    predictions (ratio near 1.0) are discarded rather than accepted.
  - Decaying vote window: only recent frames count toward the decision.
  - Result: formations, pressing, and offside lines are stable mid-clip.

PRD Reference : Section 6, Task 1.6, GOAL-04
Outputs       : List of `data_models.Track` with updated `team_id`
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.data_models import Track, CLASS_PLAYER, CLASS_GOALKEEPER


# ── Tuning constants ──────────────────────────────────────────────────────────
MIN_PLAYERS_TO_INIT    = 4    # Minimum players needed before K-Means runs
LOCK_VOTE_THRESHOLD    = 8    # Consecutive (or recent) votes needed to lock a team
VOTE_WINDOW            = 20   # Sliding window size for votes (frames)
MIN_CONFIDENCE         = 0.20 # Minimum distance ratio difference to accept a prediction
                               # (ratio = |d1 - d2| / (d1 + d2)  →  0=uncertain, 1=certain)
UPDATE_RATE            = 0.04 # EMA rate for updating cluster centers


class TeamClassifier:
    """
    Classifies players into two teams based on jersey colors (HSV histograms + K-Means).

    Persistence layer:
        Each track ID maintains a sliding window of team votes.
        When the vote tally is decisive enough, the team assignment is *locked*.
        Locked assignments are only overridden if an overwhelming contradiction
        builds up (>= LOCK_VOTE_THRESHOLD votes for the other team).
    """

    def __init__(self) -> None:
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.is_initialized: bool = False
        self.cluster_centers: Optional[np.ndarray] = None

        # Per-track sliding vote window: track_id → deque of (team_id: int)
        self._votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))

        # Per-track locked team: track_id → team_id (1 or 2)
        # Once locked, only a strong contradiction can change it.
        self._locked: Dict[int, int] = {}

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extracts an HSV color histogram from the upper 40% of the bounding box
        (jersey region), masking out green grass pixels.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(32, dtype=np.float32)

        # Upper 40% = jersey (avoid shorts/socks confusing hue)
        y_mid = y1 + max(1, int((y2 - y1) * 0.40))
        crop = frame[y1:y_mid, x1:x2]
        if crop.size == 0:
            return np.zeros(32, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mask out green grass
        not_grass = cv2.bitwise_not(
            cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        )

        hist = cv2.calcHist([hsv], [0], not_grass, [32], [0, 180])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    # ── Prediction with confidence ────────────────────────────────────────────

    def _predict(self, feat: np.ndarray) -> tuple[int, float]:
        """
        Predict team (1 or 2) and a confidence score [0, 1].
        confidence = |d1 - d2| / (d1 + d2)  — 0=ambiguous, 1=very certain
        """
        d1 = float(np.linalg.norm(feat - self.cluster_centers[0]))
        d2 = float(np.linalg.norm(feat - self.cluster_centers[1]))
        total = d1 + d2
        confidence = abs(d1 - d2) / total if total > 1e-9 else 0.0
        team = 1 if d1 < d2 else 2
        return team, confidence

    # ── Cluster center update ─────────────────────────────────────────────────

    def _update_center(self, cluster_idx: int, feat: np.ndarray) -> None:
        self.cluster_centers[cluster_idx] = (
            (1 - UPDATE_RATE) * self.cluster_centers[cluster_idx]
            + UPDATE_RATE * feat
        )

    # ── Team persistence logic ────────────────────────────────────────────────

    def _apply_vote(self, track_id: int, predicted_team: int) -> int:
        """
        Add the predicted team to the vote window for this track.
        If already locked, require a strong reversal to change.
        Returns the final resolved team_id.
        """
        votes = self._votes[track_id]
        votes.append(predicted_team)

        # Count votes in window
        v1 = votes.count(1)
        v2 = votes.count(2)

        if track_id in self._locked:
            locked_team = self._locked[track_id]
            # Override only if the opposite team dominates the full window
            override_team = 2 if locked_team == 1 else 1
            override_votes = v2 if locked_team == 1 else v1
            if override_votes >= LOCK_VOTE_THRESHOLD:
                self._locked[track_id] = override_team
                logging.debug(f"Track {track_id}: team lock changed {locked_team}→{override_team}")
            return self._locked[track_id]

        # Not yet locked — check if we should lock now
        winning = 1 if v1 >= v2 else 2
        winning_count = max(v1, v2)
        if winning_count >= LOCK_VOTE_THRESHOLD:
            self._locked[track_id] = winning
            logging.debug(f"Track {track_id}: team locked as {winning} ({winning_count} votes)")
            return winning

        # Not enough votes — return majority so far (tentative)
        return winning if (v1 + v2) > 0 else predicted_team

    # ── Main classify method ──────────────────────────────────────────────────

    def classify(self, tracks: List[Track], frame: np.ndarray) -> List[Track]:
        """
        Assign team_id to each player track with temporal persistence.

        Non-player classes (referee, ball) are assigned team_id=None.
        Goalkeeper is classified alongside players.
        """
        player_tracks = [
            t for t in tracks
            if t.class_id in (CLASS_PLAYER, CLASS_GOALKEEPER)
        ]

        if len(player_tracks) < 2:
            return tracks

        # Extract features for all player/GK tracks
        features = [self._extract_features(frame, t.bbox_px) for t in player_tracks]
        features_np = np.array(features, dtype=np.float32)

        # ── Initialization ────────────────────────────────────────────────────
        if not self.is_initialized:
            if len(player_tracks) >= MIN_PLAYERS_TO_INIT:
                labels = self.kmeans.fit_predict(features_np)
                self.cluster_centers = self.kmeans.cluster_centers_.copy()
                self.is_initialized = True

                for i, t in enumerate(player_tracks):
                    raw_team = int(labels[i]) + 1
                    t.team_id = self._apply_vote(t.track_id, raw_team)
            return tracks

        # ── Per-track prediction with confidence gating ───────────────────────
        for i, t in enumerate(player_tracks):
            feat = features_np[i]
            predicted_team, confidence = self._predict(feat)

            # If already locked, always use the lock (ignore this prediction)
            if t.track_id in self._locked:
                t.team_id = self._locked[t.track_id]
                # Still do a soft cluster center update for the locked team
                cluster_idx = self._locked[t.track_id] - 1
                self._update_center(cluster_idx, feat)
                continue

            # If confidence is too low, don't vote — just re-use last known
            if confidence < MIN_CONFIDENCE:
                if self._votes[t.track_id]:
                    v1 = list(self._votes[t.track_id]).count(1)
                    v2 = list(self._votes[t.track_id]).count(2)
                    t.team_id = 1 if v1 >= v2 else 2
                # else leave team_id as None until we have more data
                continue

            # Confident prediction — vote and update cluster
            t.team_id = self._apply_vote(t.track_id, predicted_team)
            cluster_idx = predicted_team - 1
            self._update_center(cluster_idx, feat)

        return tracks

    def reset(self) -> None:
        """Reset all state (call when starting a new video)."""
        self.is_initialized = False
        self.cluster_centers = None
        self._votes.clear()
        self._locked.clear()
