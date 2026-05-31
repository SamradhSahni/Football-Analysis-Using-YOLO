"""
tests/test_mapping.py
=====================
Unit tests for src/mapping/homography.py
PRD Reference: Task 1.5
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_models import Track
from src.mapping.homography import CoordinateMapper


@pytest.fixture
def mock_calibration_dir(tmp_path):
    """Create a temporary calibration directory with mock homography matrices."""
    calib_dir = tmp_path / "calibration"
    seq_dir = calib_dir / "SNMOT-000"
    seq_dir.mkdir(parents=True)
    
    # Create a simple scaling matrix mapping pixel to meters
    # Scale x by 0.1, y by 0.1
    # [0.1, 0, 0]
    # [0, 0.1, 0]
    # [0, 0, 1]
    mock_H = [
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 1.0]
    ]
    
    calib_data = {
        "1": mock_H,
        "2": mock_H
    }
    
    with open(seq_dir / "camera_matrices.json", "w") as f:
        json.dump(calib_data, f)
        
    return calib_dir


class TestCoordinateMapper:
    def test_initialization(self):
        mapper = CoordinateMapper()
        assert mapper.calibration_dir is None
        assert np.array_equal(mapper.get_homography("unknown", 1), mapper._default_h)

    def test_load_calibration(self, mock_calibration_dir):
        mapper = CoordinateMapper(calibration_dir=mock_calibration_dir)
        mapper.load_sequence_calibration("SNMOT-000")
        
        H = mapper.get_homography("SNMOT-000", frame_id=1)
        assert H.shape == (3, 3)
        assert H[0, 0] == 0.1
        assert H[1, 1] == 0.1

    def test_map_tracks_empty(self):
        mapper = CoordinateMapper()
        assert mapper.map_tracks([], "SNMOT-000", 1) == []

    def test_map_tracks_projection(self, mock_calibration_dir):
        mapper = CoordinateMapper(calibration_dir=mock_calibration_dir)
        mapper.load_sequence_calibration("SNMOT-000")
        
        # Create a track
        # Bottom center calculation: x_center = (100+200)/2 = 150. y_bottom = 300
        # Expected projection with the mock_H: x_m = 150 * 0.1 = 15.0, y_m = 300 * 0.1 = 30.0
        track = Track(
            track_id=1,
            frame_id=1,
            bbox_px=(100.0, 100.0, 200.0, 300.0),
            centroid_px=(150.0, 200.0),
            class_id=0,
            team_id=None,
            centroid_world=None
        )
        
        mapped_tracks = mapper.map_tracks([track], "SNMOT-000", frame_id=1)
        
        assert len(mapped_tracks) == 1
        assert mapped_tracks[0].centroid_world is not None
        
        x_w, y_w = mapped_tracks[0].centroid_world
        assert x_w == pytest.approx(15.0)
        assert y_w == pytest.approx(30.0)

    def test_map_tracks_fallback_homography(self):
        mapper = CoordinateMapper()
        # Should use the _default_h which scales by 0.05
        track = Track(
            track_id=1,
            frame_id=1,
            bbox_px=(100.0, 100.0, 200.0, 300.0),
            centroid_px=(150.0, 200.0),
            class_id=0,
            team_id=None,
            centroid_world=None
        )
        
        # bottom center is 150, 300
        # Expected: 150 * 0.05 = 7.5, 300 * 0.05 = 15.0
        mapped_tracks = mapper.map_tracks([track], "unknown", frame_id=1)
        
        x_w, y_w = mapped_tracks[0].centroid_world
        assert x_w == pytest.approx(7.5)
        assert y_w == pytest.approx(15.0)
