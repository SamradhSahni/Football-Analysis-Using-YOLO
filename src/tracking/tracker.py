"""
src/tracking/tracker.py
=======================
Stage 3 of the Football Tracker Pipeline.
Responsible for tracking detected objects across frames using ByteTrack.

PRD Reference : Section 6, Task 1.4, GOAL-02
Outputs       : List of `data_models.Track` objects per frame
"""

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np

# We attempt to import a ByteTrack implementation. 
# Ultralytics or supervision both provide them. We'll design this to be easily 
# adaptable to whichever underlying engine is preferred. For now, we will 
# provide a structured wrapper that can use `supervision.ByteTrack` if available.
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    logging.warning("supervision package not found. Tracker will use naive centroid fallback.")

from src.data_models import Detection, Track


class ByteTrackerWrapper:
    """
    Wraps the ByteTrack algorithm to provide persistent IDs for detections.
    Decoupled from the detector: takes `List[Detection]` and returns `List[Track]`.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 120,
        match_thresh: float = 0.8,
        frame_rate: int = 25
    ) -> None:
        """
        Parameters
        ----------
        track_thresh : Confidence threshold for tracking
        track_buffer : Frames to keep a lost track before deleting
        match_thresh : IoU threshold for matching
        frame_rate   : Video FPS (used by tracker for buffer time calculation)
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.tracker = None
        if HAS_SUPERVISION:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_thresh,
                    lost_track_buffer=self.track_buffer,
                    minimum_matching_threshold=self.match_thresh,
                    frame_rate=self.frame_rate
                )
            
        # For naive fallback if supervision isn't installed
        self._next_id = 1
        self._active_tracks: Dict[int, tuple] = {} # id -> (bbox, class_id)

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update the tracker with new detections and return persistent tracks.

        Parameters
        ----------
        detections : Detections for the current frame.

        Returns
        -------
        List of data_models.Track representing the updated identities.
        """
        if not detections:
            return []

        # All detections should belong to the same frame
        frame_id = detections[0].frame_id

        tracks: List[Track] = []

        if HAS_SUPERVISION and self.tracker is not None:
            # 1. Convert our Detections -> Supervision Detections
            # SV format: xyxy (N, 4), confidence (N,), class_id (N,)
            xyxy = np.array([d.bbox_px for d in detections], dtype=np.float32)
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            class_ids = np.array([d.class_id for d in detections], dtype=int)
            
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=class_ids
            )

            # 2. Update ByteTrack
            tracked_dets = self.tracker.update_with_detections(sv_dets)

            # 3. Convert back to our Track objects
            for i in range(len(tracked_dets.xyxy)):
                # tracker_id can be None for detections not yet confirmed
                if tracked_dets.tracker_id is None or tracked_dets.tracker_id[i] is None:
                    continue

                t_bbox = tuple(tracked_dets.xyxy[i].tolist())
                t_cls  = int(tracked_dets.class_id[i])
                t_id   = int(tracked_dets.tracker_id[i])

                # Compute pixel centroid from xyxy
                cx = (t_bbox[0] + t_bbox[2]) / 2.0
                cy = (t_bbox[1] + t_bbox[3]) / 2.0

                track = Track(
                    track_id=t_id,
                    frame_id=frame_id,
                    bbox_px=t_bbox,
                    centroid_px=(cx, cy),
                    class_id=t_cls,
                    team_id=None,        # filled by TeamClassifier
                    centroid_world=None, # filled by CoordinateMapper
                )
                tracks.append(track)
                
        else:
            # Naive fallback: if bounding box center is close, keep ID. 
            # (Just for tests/dev without supervision installed)
            for det in detections:
                assigned_id = None
                det_cx = det.centroid_px[0]
                det_cy = det.centroid_px[1]
                
                best_id = -1
                best_dist = float('inf')
                
                for t_id, (t_bbox, t_cls) in self._active_tracks.items():
                    if t_cls != det.class_id:
                        continue
                    t_cx = (t_bbox[0] + t_bbox[2]) / 2
                    t_cy = (t_bbox[1] + t_bbox[3]) / 2
                    
                    dist = ((t_cx - det_cx)**2 + (t_cy - det_cy)**2)**0.5
                    if dist < 50 and dist < best_dist: # 50px threshold
                        best_dist = dist
                        best_id = t_id
                        
                if best_id != -1:
                    assigned_id = best_id
                else:
                    assigned_id = self._next_id
                    self._next_id += 1
                    
                self._active_tracks[assigned_id] = (det.bbox_px, det.class_id)
                
                track = Track.from_detection(det, track_id=assigned_id)
                tracks.append(track)

        # 4. Handle ball separately if needed
        # The ball often moves too erratically for Kalman filters, or is occluded.
        # Sometimes it's better to pass it through directly or use a specialized logic.
        # This wrapper currently passes everything through ByteTrack/Naive fallback.
        
        return tracks

    def reset(self) -> None:
        """Reset the tracker state (e.g., when starting a new video)."""
        if HAS_SUPERVISION and self.tracker is not None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_thresh,
                    lost_track_buffer=self.track_buffer,
                    minimum_matching_threshold=self.match_thresh,
                    frame_rate=self.frame_rate
                )
        self._next_id = 1
        self._active_tracks.clear()
