"""
src/tracking/reid.py
====================
Player Re-Identification across camera cuts.

PRD Reference: Goal-21, Task 2.10

Pipeline:
    1. CutDetector   — detects scene cuts by comparing consecutive frames
                       (>40% pixel change threshold, PRD Section 15)
    2. AppearanceExtractor — crops each bbox, resizes, computes a lightweight
                             colour-histogram embedding (ResNet50 optional)
    3. ReIDMatcher   — cosine similarity match (threshold 0.75) between
                       pre-cut and post-cut track embeddings
    4. ReIDModule    — orchestrates all three; restores original track IDs
                       and writes a match log
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.data_models import Track


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackEmbedding:
    """Stores the appearance embedding for one track at a specific frame."""
    track_id: int
    frame_id: int
    embedding: np.ndarray   # 1-D float32 vector


@dataclass
class ReIDMatch:
    """Records a successful re-identification across a camera cut."""
    cut_frame: int
    pre_cut_track_id: int
    post_cut_track_id: int
    similarity: float


# ---------------------------------------------------------------------------
# 1. Cut Detector
# ---------------------------------------------------------------------------

class CutDetector:
    """
    Detects abrupt scene (camera) cuts between consecutive frames.

    A cut is detected when the mean absolute pixel difference between
    two consecutive greyscale frames exceeds `threshold_pct` percent
    of the maximum pixel value (255).

    PRD spec: threshold = 40%
    """

    def __init__(self, threshold_pct: float = 40.0):
        self.threshold_pct = threshold_pct
        self._prev_frame: Optional[np.ndarray] = None
        self.cut_frames: List[int] = []

    def update(self, frame_id: int, frame_bgr: np.ndarray) -> bool:
        """
        Ingest a new frame. Returns True if a cut was detected.

        Parameters
        ----------
        frame_id : int
        frame_bgr : np.ndarray
            Full BGR frame (H x W x 3 uint8).
        """
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._prev_frame is None:
            self._prev_frame = grey
            return False

        diff = np.mean(np.abs(grey - self._prev_frame))
        pct_diff = (diff / 255.0) * 100.0

        self._prev_frame = grey
        is_cut = pct_diff >= self.threshold_pct

        if is_cut:
            self.cut_frames.append(frame_id)
            logging.info(f"Camera cut detected at frame {frame_id} (diff={pct_diff:.1f}%)")

        return is_cut


# ---------------------------------------------------------------------------
# 2. Appearance Extractor
# ---------------------------------------------------------------------------

class AppearanceExtractor:
    """
    Extracts a lightweight appearance embedding from a player crop.

    Embedding strategy: 
        - Crop the bounding box from the frame.
        - Resize to 64x128 (standard Re-ID input size).
        - Compute a normalised 3-channel HSV histogram (32 bins per channel).
        - Flatten and L2-normalise → 96-dimensional descriptor.

    This is a CPU-only, zero-dependency baseline. Upgrading to OSNet or
    ResNet50 only requires swapping `_extract_hsv_histogram` with a 
    torch inference call.
    """

    CROP_W, CROP_H = 64, 128
    BINS = 32

    def extract(self, frame_bgr: np.ndarray, bbox_px: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract appearance embedding from a single bounding box.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Full BGR frame.
        bbox_px : Tuple[x1, y1, x2, y2]
            Pixel-space bounding box.

        Returns
        -------
        np.ndarray : L2-normalised 96-D float32 vector, or zeros if crop fails.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox_px)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.BINS * 3, dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.CROP_W, self.CROP_H))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist_parts = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [self.BINS], [0, 256])
            hist = hist.flatten().astype(np.float32)
            hist_parts.append(hist)

        embedding = np.concatenate(hist_parts)

        # L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_tracks(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        tracks: List[Track]
    ) -> List[TrackEmbedding]:
        """
        Extract embeddings for all visible tracks in one frame.
        """
        results = []
        for track in tracks:
            emb = self.extract(frame_bgr, track.bbox_px)
            results.append(TrackEmbedding(track_id=track.track_id, frame_id=frame_id, embedding=emb))
        return results


# ---------------------------------------------------------------------------
# 3. ReID Matcher
# ---------------------------------------------------------------------------

class ReIDMatcher:
    """
    Matches pre-cut track embeddings to post-cut track embeddings
    using cosine similarity (threshold = 0.75, PRD spec).
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.threshold = similarity_threshold

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def match(
        self,
        pre_cut: List[TrackEmbedding],
        post_cut: List[TrackEmbedding],
        cut_frame: int,
    ) -> Tuple[Dict[int, int], List[ReIDMatch]]:
        """
        Greedy one-to-one matching between pre-cut and post-cut embeddings.

        Parameters
        ----------
        pre_cut : List[TrackEmbedding]
            Embeddings from the last frame before the cut.
        post_cut : List[TrackEmbedding]
            Embeddings from the first frame after the cut.
        cut_frame : int
            Frame index of the detected cut.

        Returns
        -------
        id_map : Dict[post_cut_track_id → restored_pre_cut_track_id]
            Empty dict for unmatched tracks.
        matches : List[ReIDMatch]
        """
        if not pre_cut or not post_cut:
            return {}, []

        # Build similarity matrix [post x pre]
        sim_matrix = np.zeros((len(post_cut), len(pre_cut)), dtype=np.float32)
        for pi, post_emb in enumerate(post_cut):
            for pri, pre_emb in enumerate(pre_cut):
                sim_matrix[pi, pri] = self._cosine_similarity(post_emb.embedding, pre_emb.embedding)

        id_map: Dict[int, int] = {}
        matched_pre: set = set()
        re_id_matches: List[ReIDMatch] = []

        # Greedy: assign highest similarity pairs first
        flat_sorted = np.argsort(-sim_matrix, axis=None)   # descending
        for flat_idx in flat_sorted:
            pi = int(flat_idx // len(pre_cut))
            pri = int(flat_idx % len(pre_cut))
            sim = float(sim_matrix[pi, pri])

            if sim < self.threshold:
                break

            post_id = post_cut[pi].track_id
            pre_id = pre_cut[pri].track_id

            if post_id in id_map or pri in matched_pre:
                continue

            id_map[post_id] = pre_id
            matched_pre.add(pri)
            re_id_matches.append(ReIDMatch(
                cut_frame=cut_frame,
                pre_cut_track_id=pre_id,
                post_cut_track_id=post_id,
                similarity=sim,
            ))

        logging.info(f"Re-ID at cut {cut_frame}: {len(re_id_matches)}/{min(len(pre_cut), len(post_cut))} matched")
        return id_map, re_id_matches


# ---------------------------------------------------------------------------
# 4. ReID Module (Orchestrator)
# ---------------------------------------------------------------------------

class ReIDModule:
    """
    Full orchestrator integrating cut detection, appearance extraction,
    and track ID restoration.

    Usage
    -----
    reid = ReIDModule()
    for frame_id, frame_bgr, tracks in video_loop:
        corrected_tracks = reid.process(frame_id, frame_bgr, tracks)
    
    reid.match_log  → List[ReIDMatch]
    """

    def __init__(
        self,
        cut_threshold_pct: float = 40.0,
        similarity_threshold: float = 0.75,
    ):
        self.cut_detector = CutDetector(threshold_pct=cut_threshold_pct)
        self.extractor = AppearanceExtractor()
        self.matcher = ReIDMatcher(similarity_threshold=similarity_threshold)

        # Buffer: track_id -> TrackEmbedding (from last non-cut frame)
        self._last_embeddings: List[TrackEmbedding] = []
        self._id_remapping: Dict[int, int] = {}   # new_id → original_id
        self.match_log: List[ReIDMatch] = []

    def process(
        self,
        frame_id: int,
        frame_bgr: np.ndarray,
        tracks: List[Track],
    ) -> List[Track]:
        """
        Process one frame. Returns the same tracks with corrected track_ids
        if a re-identification was performed after a cut.

        Parameters
        ----------
        frame_id : int
        frame_bgr : np.ndarray
            Full BGR frame for embedding extraction.
        tracks : List[Track]
            Output from the tracker for this frame.

        Returns
        -------
        List[Track] with restored track_ids where possible.
        """
        is_cut = self.cut_detector.update(frame_id, frame_bgr)

        # Extract current embeddings
        current_embeddings = self.extractor.extract_tracks(frame_bgr, frame_id, tracks)

        if is_cut and self._last_embeddings:
            # Match pre-cut (last_embeddings) to post-cut (current_embeddings)
            id_map, matches = self.matcher.match(
                pre_cut=self._last_embeddings,
                post_cut=current_embeddings,
                cut_frame=frame_id,
            )
            self.match_log.extend(matches)

            # Update global remapping
            for new_id, restored_id in id_map.items():
                # Follow the chain: restored_id might itself be a remapped new_id
                self._id_remapping[new_id] = self._id_remapping.get(restored_id, restored_id)

        # Apply remapping to tracks
        restored_tracks = []
        for track in tracks:
            if track.track_id in self._id_remapping:
                import dataclasses
                track = dataclasses.replace(track, track_id=self._id_remapping[track.track_id])
            restored_tracks.append(track)

        # Update buffer for next cut
        self._last_embeddings = current_embeddings
        return restored_tracks
