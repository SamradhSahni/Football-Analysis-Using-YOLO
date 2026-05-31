import os
from pathlib import Path

import pytest

from src.analytics.heatmap import HeatmapGenerator
from src.data_models import TrajectoryPoint


def test_heatmap_initialization():
    generator = HeatmapGenerator(pitch_length=100.0, pitch_width=50.0, cmap="viridis", dpi=100)
    assert generator.length == 100.0
    assert generator.width == 50.0
    assert generator.cmap == "viridis"
    assert generator.dpi == 100


def test_heatmap_generation_empty(tmp_path):
    generator = HeatmapGenerator()
    out_path = tmp_path / "empty.png"
    
    # Should not crash, just warn and return
    generator.generate([], str(out_path))
    assert not out_path.exists()


def test_heatmap_generation_valid(tmp_path):
    generator = HeatmapGenerator()
    out_path = tmp_path / "heatmap.png"
    
    # Synthetic scatter of points
    points = [
        TrajectoryPoint(frame_id=i, timestamp_s=i*0.1, x_m=50.0 + i*0.1, y_m=34.0, speed_ms=2.0, speed_kmh=7.2)
        for i in range(10)
    ]
    
    generator.generate(points, str(out_path))
    
    # File should be created
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_team_comparison_generation(tmp_path):
    generator = HeatmapGenerator()
    out_path = tmp_path / "comparison.png"
    
    team_a = [TrajectoryPoint(frame_id=1, timestamp_s=0.1, x_m=20.0, y_m=30.0, speed_ms=0, speed_kmh=0)]
    team_b = [TrajectoryPoint(frame_id=1, timestamp_s=0.1, x_m=80.0, y_m=30.0, speed_ms=0, speed_kmh=0)]
    
    generator.generate_team_comparison(team_a, team_b, str(out_path))
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0
