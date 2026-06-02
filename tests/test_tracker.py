"""
tests/test_tracker.py
=====================
Unit tests for src/tracking/tracker.py
PRD Reference: Task 1.4
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_models import Detection, CLASS_PLAYER
from src.tracking.tracker import ByteTrackerWrapper, HAS_SUPERVISION


@pytest.fixture
def mock_supervision():
    """Mock the supervision package if it's not available, to test the wrapper structure."""
    if not HAS_SUPERVISION:
        # We can't fully test the supervision integration if it's not installed,
        # but we can test the fallback. If we mock it, we can test the SV branch.
        pass
    yield


class TestTracker:
    def test_initialization(self):
        tracker = ByteTrackerWrapper(
            track_thresh=0.5,
            track_buffer=10,
            match_thresh=0.7,
            frame_rate=30
        )
        assert tracker.track_thresh == 0.5
        assert tracker.track_buffer == 10
        assert tracker.match_thresh == 0.7
        assert tracker.frame_rate == 30

    def test_empty_detections(self):
        tracker = ByteTrackerWrapper()
        tracks = tracker.update([])
        assert tracks == []

    def test_naive_fallback_tracking(self):
        """Test the fallback tracker logic (if SV is disabled/mocked out)."""
        # We temporarily disable supervision to test the fallback
        with patch("src.tracking.tracker.HAS_SUPERVISION", False):
            tracker = ByteTrackerWrapper()
            
            # Frame 1: Two players
            det1 = Detection(
                frame_id=1,
                bbox_px=(100.0, 100.0, 150.0, 200.0),
                confidence=0.9,
                class_id=CLASS_PLAYER,
                class_name="player"
            )
            det2 = Detection(
                frame_id=1,
                bbox_px=(300.0, 300.0, 350.0, 400.0),
                confidence=0.8,
                class_id=CLASS_PLAYER,
                class_name="player"
            )
            
            tracks1 = tracker.update([det1, det2])
            assert len(tracks1) == 2
            
            # IDs should be 1 and 2
            t1_id = tracks1[0].track_id
            t2_id = tracks1[1].track_id
            assert t1_id != t2_id
            
            # Frame 2: Players moved slightly
            det3 = Detection(
                frame_id=2,
                bbox_px=(102.0, 102.0, 152.0, 202.0), # Moved +2px
                confidence=0.9,
                class_id=CLASS_PLAYER,
                class_name="player"
            )
            det4 = Detection(
                frame_id=2,
                bbox_px=(305.0, 305.0, 355.0, 405.0), # Moved +5px
                confidence=0.8,
                class_id=CLASS_PLAYER,
                class_name="player"
            )
            
            tracks2 = tracker.update([det4, det3]) # Passed in reverse order
            assert len(tracks2) == 2
            
            # Since det3 is close to det1, and det4 is close to det2, 
            # their IDs should be preserved.
            # det4 should get t2_id, det3 should get t1_id
            
            # Find the track for det3
            track_for_det3 = next(t for t in tracks2 if t.bbox_px == det3.bbox_px)
            track_for_det4 = next(t for t in tracks2 if t.bbox_px == det4.bbox_px)
            
            assert track_for_det3.track_id == t1_id
            assert track_for_det4.track_id == t2_id

    def test_reset(self):
        with patch("src.tracking.tracker.HAS_SUPERVISION", False):
            tracker = ByteTrackerWrapper()
            det1 = Detection(
                frame_id=1,
                bbox_px=(100.0, 100.0, 150.0, 200.0),
                confidence=0.9,
                class_id=CLASS_PLAYER,
                class_name="player"
            )
            
            tracks = tracker.update([det1])
            assert tracks[0].track_id == 1
            
            tracker.reset()
            
            # After reset, ID should start over
            tracks2 = tracker.update([det1])
            assert tracks2[0].track_id == 1
