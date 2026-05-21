"""
src/mapping/calibration_parser.py
===================================
Task 2.1 — Parses SoccerNet calibration JSON files and computes a 3x3 homography
matrix from visible pitch line correspondences.

SoccerNet Calibration Format:
    Each JSON file contains named pitch markings (e.g. "Big rect. left main")
    with normalized pixel coordinates (0.0-1.0, relative to image width/height).
    
    We match these to known real-world FIFA pitch coordinates (metres) and
    solve for the homography H such that: pixel_pt = H @ world_pt

FIFA Standard Pitch (metres):
    Origin: top-left corner of pitch
    X-axis: along the length (0 → 105 m)
    Y-axis: along the width  (0 → 68 m)
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# FIFA Pitch keypoints in real-world coordinates (metres)
# Reference: https://en.wikipedia.org/wiki/Football_pitch
# Origin at top-left corner. X = length (105m), Y = width (68m).
# ---------------------------------------------------------------------------

PITCH_W = 105.0   # metres (length)
PITCH_H = 68.0    # metres (width)

# Each entry maps a SoccerNet line key → list of real-world (x_m, y_m) points
# Points must correspond to the order they appear in the JSON (start → end)
SOCCERNET_WORLD_COORDS: Dict[str, List[Tuple[float, float]]] = {
    # Sidelines
    "Side line top":         [(0.0, 0.0),   (PITCH_W, 0.0)],
    "Side line bottom":      [(0.0, PITCH_H), (PITCH_W, PITCH_H)],
    "Side line left":        [(0.0, 0.0),   (0.0, PITCH_H)],
    "Side line right":       [(PITCH_W, 0.0), (PITCH_W, PITCH_H)],

    # Halfway line
    "Middle line":           [(PITCH_W/2, 0.0), (PITCH_W/2, PITCH_H)],

    # Left penalty box (x: 0→16.5)
    "Big rect. left top":    [(0.0, 13.84),  (16.5, 13.84)],
    "Big rect. left main":   [(16.5, 13.84), (16.5, 54.16)],
    "Big rect. left bottom": [(0.0, 54.16),  (16.5, 54.16)],

    # Right penalty box (x: 88.5→105)
    "Big rect. right top":   [(88.5, 13.84), (PITCH_W, 13.84)],
    "Big rect. right main":  [(88.5, 13.84), (88.5, 54.16)],
    "Big rect. right bottom":[(88.5, 54.16), (PITCH_W, 54.16)],

    # Left 6-yard box (x: 0→5.5)
    "Small rect. left top":    [(0.0, 24.84),  (5.5, 24.84)],
    "Small rect. left main":   [(5.5, 24.84),  (5.5, 43.16)],
    "Small rect. left bottom": [(0.0, 43.16),  (5.5, 43.16)],

    # Right 6-yard box (x: 99.5→105)
    "Small rect. right top":    [(99.5, 24.84), (PITCH_W, 24.84)],
    "Small rect. right main":   [(99.5, 24.84), (99.5, 43.16)],
    "Small rect. right bottom": [(99.5, 43.16), (PITCH_W, 43.16)],

    # Goal posts — left goal (x=0)
    "Goal left post left ":    [(0.0, 30.34),  (0.0, 30.34)],
    "Goal left post right":    [(0.0, 37.66),  (0.0, 37.66)],
    "Goal left crossbar":      [(0.0, 30.34),  (0.0, 37.66)],

    # Goal posts — right goal (x=105)
    "Goal right post left ":   [(PITCH_W, 30.34), (PITCH_W, 30.34)],
    "Goal right post right":   [(PITCH_W, 37.66), (PITCH_W, 37.66)],
    "Goal right crossbar":     [(PITCH_W, 30.34), (PITCH_W, 37.66)],
}


def _interpolate_line_endpoints(pts: List[Dict]) -> Tuple[Tuple, Tuple]:
    """Return the first and last point of a line segment."""
    first = (pts[0]["x"], pts[0]["y"])
    last  = (pts[-1]["x"], pts[-1]["y"])
    return first, last


def compute_homography_from_json(
    calib_json: Dict,
    image_width: int = 1920,
    image_height: int = 1080,
    min_correspondences: int = 4,
) -> Optional[np.ndarray]:
    """
    Compute a 3x3 homography matrix from a SoccerNet calibration JSON.

    Parameters
    ----------
    calib_json    : Parsed JSON dict from SoccerNet calibration file
    image_width   : Width of the video frame in pixels
    image_height  : Height of the video frame in pixels
    min_correspondences : Minimum point pairs required to solve for H

    Returns
    -------
    3x3 np.ndarray homography (pixel → world) or None if insufficient data.
    """
    src_pts = []   # pixel coordinates (normalised → pixel)
    dst_pts = []   # real-world pitch coordinates (metres)

    for line_name, world_endpoints in SOCCERNET_WORLD_COORDS.items():
        if line_name not in calib_json:
            continue

        json_pts = calib_json[line_name]
        if not json_pts:
            continue

        img_start, img_end = _interpolate_line_endpoints(json_pts)

        # Convert normalised [0,1] pixel coords → absolute pixel coords
        p1_px = (img_start[0] * image_width,  img_start[1] * image_height)
        p2_px = (img_end[0]   * image_width,  img_end[1]   * image_height)

        w1, w2 = world_endpoints[0], world_endpoints[-1]

        src_pts.extend([p1_px, p2_px])
        dst_pts.extend([w1,    w2])

    if len(src_pts) < min_correspondences:
        logging.debug(
            f"Only {len(src_pts)} correspondences found — need ≥ {min_correspondences}. "
            "Falling back to default homography."
        )
        return None

    src_np = np.array(src_pts, dtype=np.float32)
    dst_np = np.array(dst_pts, dtype=np.float32)

    H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is None:
        logging.warning("cv2.findHomography failed. Falling back to default homography.")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    logging.debug(f"Homography computed with {inliers}/{len(src_pts)} inliers.")

    return H


class CalibrationParser:
    """
    Reads SoccerNet calibration ZIP files and provides per-frame homography matrices.
    
    Usage:
        parser = CalibrationParser("soccernet_data/calibration/train.zip")
        H = parser.get_homography(frame_index=42, image_width=1920, image_height=1080)
    """

    def __init__(self, zip_path: Optional[str | Path] = None) -> None:
        self._zip_path = Path(zip_path) if zip_path else None
        self._cache: Dict[int, Optional[np.ndarray]] = {}
        self._zip = None

        if self._zip_path and self._zip_path.exists():
            try:
                self._zip = zipfile.ZipFile(self._zip_path, "r")
                logging.info(f"CalibrationParser: Loaded {self._zip_path}")
            except Exception as e:
                logging.warning(f"CalibrationParser: Could not open {self._zip_path}: {e}")

    def get_homography(
        self,
        frame_index: int,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> Optional[np.ndarray]:
        """
        Return the homography for a specific frame index (0-based).
        Results are cached.
        """
        if frame_index in self._cache:
            return self._cache[frame_index]

        H = self._load_from_zip(frame_index, image_width, image_height)
        self._cache[frame_index] = H
        return H

    def _load_from_zip(
        self, frame_index: int, width: int, height: int
    ) -> Optional[np.ndarray]:
        if self._zip is None:
            return None

        # SoccerNet calibration files are named 00000.json, 00001.json, etc.
        json_name = f"train/{frame_index:05d}.json"

        try:
            with self._zip.open(json_name) as f:
                data = json.load(f)
            return compute_homography_from_json(data, width, height)
        except KeyError:
            logging.debug(f"Calibration file not found: {json_name}")
            return None
        except Exception as e:
            logging.debug(f"Error reading calibration {json_name}: {e}")
            return None

    def close(self) -> None:
        if self._zip:
            self._zip.close()


# ---------------------------------------------------------------------------
# Default fallback homography (used when calibration is not available)
# Assumes standard 1920×1080 broadcast camera roughly centred on pitch.
# ---------------------------------------------------------------------------

def get_default_homography(image_width: int = 1920, image_height: int = 1080) -> np.ndarray:
    """
    Returns a rough pixel → world homography for a typical broadcast camera.
    Used as fallback when calibration data is unavailable.
    """
    return np.array([
        [PITCH_W / image_width,  0.0,                   0.0],
        [0.0,                    PITCH_H / image_height, 0.0],
        [0.0,                    0.0,                   1.0],
    ], dtype=np.float64)
