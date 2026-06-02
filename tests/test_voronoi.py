import os
from pathlib import Path

import pytest

from src.analytics.voronoi import VoronoiGenerator
from src.data_models import Track


def test_voronoi_initialization():
    generator = VoronoiGenerator(pitch_length=100.0, pitch_width=50.0, dpi=100)
    assert generator.length == 100.0
    assert generator.width == 50.0
    assert generator.dpi == 100


def test_voronoi_generation_empty(tmp_path):
    generator = VoronoiGenerator()
    out_path = tmp_path / "empty.png"
    
    generator.generate([], str(out_path))
    assert not out_path.exists()


def test_voronoi_generation_insufficient(tmp_path):
    generator = VoronoiGenerator()
    out_path = tmp_path / "insufficient.png"
    
    # Only 2 points, scipy requires >= 3 for a valid 2D voronoi
    tracks = [
        Track(track_id=1, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(10.0, 10.0), frame_id=1, team_id=1),
        Track(track_id=2, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(90.0, 60.0), frame_id=1, team_id=2),
    ]
    
    generator.generate(tracks, str(out_path))
    assert not out_path.exists()


def test_voronoi_generation_valid(tmp_path):
    generator = VoronoiGenerator()
    out_path = tmp_path / "voronoi.png"
    
    # Need at least 3 points
    tracks = [
        Track(track_id=1, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(10.0, 34.0), frame_id=1, team_id=1),
        Track(track_id=2, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(20.0, 20.0), frame_id=1, team_id=1),
        Track(track_id=3, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(80.0, 34.0), frame_id=1, team_id=2),
        Track(track_id=4, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(90.0, 50.0), frame_id=1, team_id=2),
        Track(track_id=5, bbox_px=(0,0,10,10), centroid_px=(5,5), centroid_world=(52.5, 34.0), frame_id=1, team_id=None),
    ]
    
    generator.generate(tracks, str(out_path))
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0
