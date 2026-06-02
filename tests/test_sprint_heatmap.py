import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before any pyplot import

from pathlib import Path

import pytest

from src.analytics.heatmap import SprintHeatmapGenerator
from src.data_models import TrajectoryPoint


def _pt(x, y, speed_ms):
    return TrajectoryPoint(
        frame_id=1, timestamp_s=0.04,
        x_m=x, y_m=y,
        speed_ms=speed_ms,
        speed_kmh=speed_ms * 3.6,
    )


def test_sprint_filter_excludes_slow_points():
    gen = SprintHeatmapGenerator(sprint_threshold_ms=7.0)
    trajectories = [
        _pt(50.0, 34.0, 3.0),   # walking
        _pt(60.0, 34.0, 7.0),   # exactly threshold → included
        _pt(70.0, 34.0, 9.0),   # sprinting
        _pt(20.0, 34.0, 5.0),   # jogging
    ]
    filtered = gen._filter_sprints(trajectories)
    assert len(filtered) == 2
    assert all(p.speed_ms >= 7.0 for p in filtered)


def test_sprint_filter_excludes_none_speed():
    gen = SprintHeatmapGenerator(sprint_threshold_ms=7.0)
    trajectories = [
        TrajectoryPoint(frame_id=1, timestamp_s=0.04, x_m=50.0, y_m=34.0,
                        speed_ms=None, speed_kmh=None),
        _pt(70.0, 34.0, 8.0),
    ]
    filtered = gen._filter_sprints(trajectories)
    assert len(filtered) == 1


def test_player_sprint_heatmap_no_sprints(tmp_path):
    gen = SprintHeatmapGenerator(sprint_threshold_ms=7.0)
    slow_points = [_pt(50.0 + i, 34.0, 2.0) for i in range(10)]
    result = gen.generate_player_sprint_heatmap(1, slow_points, str(tmp_path))
    assert result is None


def test_player_sprint_heatmap_generates_file(tmp_path):
    gen = SprintHeatmapGenerator(sprint_threshold_ms=7.0, dpi=72)
    # Create a spread of sprint points so KDE has variance
    sprint_points = [_pt(float(20 + i * 3), float(10 + i * 2), 8.0) for i in range(20)]
    result = gen.generate_player_sprint_heatmap(42, sprint_points, str(tmp_path))

    assert result is not None
    assert Path(result).exists()
    assert Path(result).name == "sprints_42.png"
    assert Path(result).stat().st_size > 0


def test_team_sprint_heatmap_generates_file(tmp_path):
    gen = SprintHeatmapGenerator(sprint_threshold_ms=7.0, dpi=72)
    # Give points 2-D spread so KDE has variance in both axes
    player_map = {
        1: [_pt(float(10 + i * 5), float(10 + i * 2), 8.0) for i in range(10)],
        2: [_pt(float(20 + i * 5), float(30 + i * 2), 9.0) for i in range(10)],
    }
    result = gen.generate_team_sprint_heatmap(1, player_map, str(tmp_path))

    assert result is not None
    assert Path(result).exists()
    assert Path(result).name == "sprints_team_1.png"


def test_generate_all_returns_paths(tmp_path):
    gen = SprintHeatmapGenerator(sprint_threshold_ms=7.0, dpi=72)
    team_trajectories = {
        1: {
            10: [_pt(float(10 + i * 4), 20.0, 8.0) for i in range(12)],
            11: [_pt(float(15 + i * 4), 50.0, 7.5) for i in range(12)],
        },
        2: {
            20: [_pt(float(80 - i * 4), 34.0, 9.0) for i in range(12)],
        },
    }
    paths = gen.generate_all(team_trajectories, str(tmp_path))

    # Should have: player_10, player_11, team_1, player_20, team_2
    assert "player_10" in paths
    assert "player_11" in paths
    assert "team_1" in paths
    assert "player_20" in paths
    assert "team_2" in paths
    assert all(Path(p).exists() for p in paths.values())
