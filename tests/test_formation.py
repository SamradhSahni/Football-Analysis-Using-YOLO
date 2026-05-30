import pytest

from src.analytics.formation import FormationDetector
from src.data_models import Track, CLASS_PLAYER


def test_formation_initialization():
    detector = FormationDetector(fps=25, window_minutes=5)
    assert detector.fps == 25
    assert detector.window_frames == 25 * 60 * 5


def test_instantaneous_formation():
    detector = FormationDetector()
    
    # Create 10 outfield players for Team 1
    # Let's mock a 4-3-3 formation
    # Defenders at X=20 (4 players)
    # Midfielders at X=40 (3 players)
    # Attackers at X=60 (3 players)
    tracks = []
    
    # Defenders
    for i in range(4):
        tracks.append(Track(track_id=i, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(20.0, 10.0 * i), frame_id=1, team_id=1, class_id=CLASS_PLAYER))
    
    # Midfielders
    for i in range(3):
        tracks.append(Track(track_id=4+i, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(40.0, 15.0 * i), frame_id=1, team_id=1, class_id=CLASS_PLAYER))
        
    # Attackers
    for i in range(3):
        tracks.append(Track(track_id=7+i, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(60.0, 15.0 * i), frame_id=1, team_id=1, class_id=CLASS_PLAYER))
        
    forms = detector.detect_instantaneous(tracks)
    
    assert 1 in forms
    assert forms[1] == "4-3-3"


def test_majority_formation_window():
    detector = FormationDetector(fps=1, window_minutes=1) # 60 frames window
    
    # Feed 40 frames of 4-4-2
    for frame_id in range(40):
        tracks = []
        for i in range(4): tracks.append(Track(track_id=i, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(10.0, 0), frame_id=frame_id, team_id=1, class_id=CLASS_PLAYER))
        for i in range(4): tracks.append(Track(track_id=i+4, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(30.0, 0), frame_id=frame_id, team_id=1, class_id=CLASS_PLAYER))
        for i in range(2): tracks.append(Track(track_id=i+8, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(50.0, 0), frame_id=frame_id, team_id=1, class_id=CLASS_PLAYER))
        detector.update(frame_id, tracks)
        
    assert detector.get_majority_formation(1) == "4-4-2"
    
    # Feed 30 frames of 3-5-2
    for frame_id in range(40, 70):
        tracks = []
        for i in range(3): tracks.append(Track(track_id=i, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(10.0, 0), frame_id=frame_id, team_id=1, class_id=CLASS_PLAYER))
        for i in range(5): tracks.append(Track(track_id=i+3, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(30.0, 0), frame_id=frame_id, team_id=1, class_id=CLASS_PLAYER))
        for i in range(2): tracks.append(Track(track_id=i+8, bbox_px=(0,0,0,0), centroid_px=(0,0), centroid_world=(50.0, 0), frame_id=frame_id, team_id=1, class_id=CLASS_PLAYER))
        detector.update(frame_id, tracks)
        
    # Window is 60 frames. 
    # Current frame is 69. 
    # History contains frames 10 to 69.
    # Frame 10-39 (30 frames) = 4-4-2
    # Frame 40-69 (30 frames) = 3-5-2
    # So it could be either. Let's add 5 more of 3-5-2
    for frame_id in range(70, 75):
        detector.update(frame_id, tracks)
        
    # Now history contains frame 15 to 74.
    # Frames 15-39 (25 frames) = 4-4-2
    # Frames 40-74 (35 frames) = 3-5-2
    assert detector.get_majority_formation(1) == "3-5-2"
