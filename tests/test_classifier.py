"""
tests/test_classifier.py
========================
Unit tests for src/classification/classifier.py
PRD Reference: Task 1.6
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_models import Track, CLASS_PLAYER, CLASS_REFEREE
from src.classification.classifier import TeamClassifier


class TestTeamClassifier:
    def test_initialization(self):
        classifier = TeamClassifier()
        assert not classifier.is_initialized
        assert classifier.cluster_centers is None

    def test_feature_extraction(self):
        classifier = TeamClassifier()
        
        # Create a synthetic frame: 100x100
        # Upper half is red (jersey), lower half is white (shorts)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[0:50, :] = [0, 0, 255] # Red in BGR
        frame[50:100, :] = [255, 255, 255] # White
        
        bbox = (10, 10, 90, 90)
        
        features = classifier._extract_features(frame, bbox)
        assert features.shape == (32,)
        # It shouldn't be completely zero
        assert np.sum(features) > 0

    def test_classify_few_players(self):
        classifier = TeamClassifier()
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Only 1 player track
        tracks = [
            Track(
                track_id=1, frame_id=1, bbox_px=(10, 10, 50, 50),
                centroid_px=(30, 50), class_id=CLASS_PLAYER, team_id=None, centroid_world=None
            )
        ]
        
        updated_tracks = classifier.classify(tracks, frame)
        # Should return unclassified since < 2 players (and specifically < 4 for init)
        assert updated_tracks[0].team_id is None

    def test_classification_logic(self):
        classifier = TeamClassifier()
        
        # Create a frame with clear red and blue players
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Player 1, 2: Red
        frame[10:50, 10:50] = [0, 0, 255]
        frame[10:50, 60:100] = [0, 0, 255]
        
        # Player 3, 4: Blue
        frame[10:50, 110:150] = [255, 0, 0]
        frame[10:50, 160:200] = [255, 0, 0]
        
        tracks = [
            Track(track_id=1, frame_id=1, bbox_px=(10, 10, 50, 50), centroid_px=(30, 50), class_id=CLASS_PLAYER, team_id=None, centroid_world=None),
            Track(track_id=2, frame_id=1, bbox_px=(60, 10, 100, 50), centroid_px=(80, 50), class_id=CLASS_PLAYER, team_id=None, centroid_world=None),
            Track(track_id=3, frame_id=1, bbox_px=(110, 10, 150, 50), centroid_px=(130, 50), class_id=CLASS_PLAYER, team_id=None, centroid_world=None),
            Track(track_id=4, frame_id=1, bbox_px=(160, 10, 200, 50), centroid_px=(180, 50), class_id=CLASS_PLAYER, team_id=None, centroid_world=None),
            # Add a referee, should be ignored
            Track(track_id=5, frame_id=1, bbox_px=(10, 10, 50, 50), centroid_px=(30, 50), class_id=CLASS_REFEREE, team_id=None, centroid_world=None),
        ]
        
        updated_tracks = classifier.classify(tracks, frame)
        
        assert classifier.is_initialized
        
        t1_team = updated_tracks[0].team_id
        t2_team = updated_tracks[1].team_id
        t3_team = updated_tracks[2].team_id
        t4_team = updated_tracks[3].team_id
        ref_team = updated_tracks[4].team_id
        
        # Red players should have same team ID
        assert t1_team == t2_team
        # Blue players should have same team ID
        assert t3_team == t4_team
        # Red and Blue should be different
        assert t1_team != t3_team
        
        # Referee should still be None
        assert ref_team is None

    def test_reset(self):
        classifier = TeamClassifier()
        classifier.is_initialized = True
        classifier.cluster_centers = np.array([[1, 2], [3, 4]])
        
        classifier.reset()
        
        assert not classifier.is_initialized
        assert classifier.cluster_centers is None
