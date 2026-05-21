"""
src/mapping/homography.py
=========================
Stage 4 of the Football Tracker Pipeline.
Responsible for transforming 2D pixel coordinates into 2D real-world pitch
coordinates (meters) using homography matrices.

Task 2.2 Update: Now integrates CalibrationParser for per-frame dynamic
homography computed from real SoccerNet calibration data.

PRD Reference : Section 6, Task 1.5, GOAL-03, GOAL-11
Outputs       : List of `data_models.Track` with updated `centroid_world`
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data_models import Track
from src.mapping.calibration_parser import (
    CalibrationParser,
    get_default_homography,
    PITCH_W,
    PITCH_H,
)


class CoordinateMapper:
    """
    Applies homography transformations to tracks.

    Projects the bottom-center of a bounding box (feet of the player)
    from the image plane to the top-down pitch plane (meters).

    With calibration data:
        Uses per-frame homography matrices computed from SoccerNet pitch
        line annotations via CalibrationParser.
    Without calibration data:
        Falls back to a linear scale based on pitch dimensions.
    """

    def __init__(
        self,
        calibration_zip: Optional[str | Path] = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> None:
        """
        Parameters
        ----------
        calibration_zip : Path to a SoccerNet calibration ZIP file
                          (e.g. soccernet_data/calibration/train.zip).
                          If None or not found, uses a fixed default homography.
        image_width     : Frame width in pixels (used for default fallback)
        image_height    : Frame height in pixels (used for default fallback)
        """
        self.image_width  = image_width
        self.image_height = image_height

        # Try to load the CalibrationParser
        self._parser: Optional[CalibrationParser] = None
        if calibration_zip and Path(calibration_zip).exists():
            self._parser = CalibrationParser(calibration_zip)
            logging.info(f"CoordinateMapper: Using dynamic calibration from {calibration_zip}")
        else:
            logging.info(
                "CoordinateMapper: No calibration ZIP found. "
                "Using default static homography (less accurate)."
            )

        # Fallback default matrix (linear scale)
        self._default_h = get_default_homography(image_width, image_height)

        # Frame-level cache for the last resolved H so we don't re-compute
        self._last_frame_id: int = -1
        self._last_H: np.ndarray = self._default_h

    def get_homography(self, frame_id: int) -> np.ndarray:
        """
        Return the best available 3x3 homography (pixel -> world) for this frame.
        Uses CalibrationParser if available, falls back to default.
        """
        if frame_id == self._last_frame_id:
            return self._last_H

        H = None
        if self._parser is not None:
            # SoccerNet calibration frames are 0-indexed
            H = self._parser.get_homography(
                frame_index=frame_id - 1,  # pipeline uses 1-indexed frame_id
                image_width=self.image_width,
                image_height=self.image_height,
            )

        if H is None:
            H = self._default_h

        self._last_frame_id = frame_id
        self._last_H = H
        return H

    def map_tracks(
        self, tracks: List[Track], frame_id: int, seq_name: str = ""
    ) -> List[Track]:
        """
        Apply homography projection to a list of tracks.
        Updates the `centroid_world` property of each track in-place.

        Uses the bottom-center of the bounding box (where feet touch the pitch).

        Parameters
        ----------
        tracks   : List of active tracks in the frame
        frame_id : Current 1-indexed frame number
        seq_name : Unused — kept for API compatibility

        Returns
        -------
        The same list of tracks with `centroid_world` populated.
        """
        if not tracks:
            return tracks

        H = self.get_homography(frame_id)

        # Build vectorized input: bottom-center of each bbox [u, v, 1]
        pts = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox_px
            u = (x1 + x2) / 2.0  # horizontal centre
            v = y2                 # bottom edge (feet position)
            pts.append([u, v, 1.0])

        pts_np = np.array(pts, dtype=np.float64).T  # shape (3, N)

        # Project: world_pt (homogeneous) = H @ pixel_pt
        proj = H @ pts_np  # shape (3, N)

        # Perspective divide (w-normalisation)
        w = proj[2, :]
        w[w == 0] = 1e-9
        
        x_world = proj[0, :] / w
        y_world = proj[1, :] / w

        # Clamp to reasonable pitch bounds to reject wild outliers
        x_world = np.clip(x_world, -15.0, PITCH_W + 15.0)
        y_world = np.clip(y_world, -15.0, PITCH_H + 15.0)

        for i, t in enumerate(tracks):
            t.centroid_world = (float(x_world[i]), float(y_world[i]))

        return tracks

    def close(self) -> None:
        """Release any open ZIP file handles."""
        if self._parser:
            self._parser.close()
