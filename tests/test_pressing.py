import numpy as np
import pytest

from src.analytics.pressing import PressingAnalyzer
from src.data_models import Track


def _make_track(track_id, x, y, team_id):
    return Track(
        track_id=track_id,
        bbox_px=(0, 0, 1, 1),
        centroid_px=(0.5, 0.5),
        centroid_world=(x, y),
        frame_id=1,
        team_id=team_id,
    )


def test_pressing_analyzer_initialization():
    analyzer = PressingAnalyzer(pressing_k=6, fps=25)
    assert analyzer.pressing_k == 6
    assert analyzer.fps == 25


def test_pressing_intensity_increases_with_proximity():
    """Team 1 pressing very close to Team 2 should score higher than when far away."""
    analyzer_close = PressingAnalyzer(fps=25)
    analyzer_far = PressingAnalyzer(fps=25)

    # Close scenario: Team 1 crowded around Team 2 at X=80
    close_tracks = [
        _make_track(i, 78.0 + i, 34.0, team_id=1) for i in range(6)
    ] + [
        _make_track(i + 10, 80.0, 34.0 + i * 2, team_id=2) for i in range(4)
    ]

    # Far scenario: Team 1 at X=10, Team 2 at X=90
    far_tracks = [
        _make_track(i, 10.0, 20.0 + i * 5, team_id=1) for i in range(6)
    ] + [
        _make_track(i + 10, 90.0, 20.0 + i * 5, team_id=2) for i in range(4)
    ]

    for _ in range(10):
        analyzer_close.update(1, close_tracks)
        analyzer_far.update(1, far_tracks)

    close_score = analyzer_close.get_pressing_intensity(team_id=1)
    far_score = analyzer_far.get_pressing_intensity(team_id=1)

    assert close_score > far_score


def test_pressing_intensity_range():
    """Pressing intensity should always be between 0 and 100."""
    analyzer = PressingAnalyzer(fps=25)

    tracks = [
        _make_track(i, 30.0, 10.0 * i, team_id=1) for i in range(5)
    ] + [
        _make_track(i + 10, 70.0, 10.0 * i, team_id=2) for i in range(5)
    ]

    for frame in range(50):
        analyzer.update(frame, tracks)

    for team_id in [1, 2]:
        score = analyzer.get_pressing_intensity(team_id)
        assert 0.0 <= score <= 100.0, f"Pressing score out of range: {score}"


def test_defensive_line_depth():
    """Defensive depth should be the mean X of the 4 rearmost players."""
    analyzer = PressingAnalyzer(fps=25)

    # Team 1: 5 players at X positions [5, 10, 15, 20, 80]
    # Rearmost 4 are at [5, 10, 15, 20] -> mean = 12.5
    tracks = [_make_track(i, float(x), 34.0, team_id=1) for i, x in enumerate([5, 10, 15, 20, 80])]

    analyzer.update(1, tracks)
    depth = analyzer.get_defensive_line_depth(team_id=1)

    assert abs(depth - 12.5) < 0.01


def test_no_crash_empty_tracks():
    """Should not crash when no tracks are passed."""
    analyzer = PressingAnalyzer(fps=25)
    analyzer.update(1, [])
    assert analyzer.get_pressing_intensity(1) == 0.0
    assert analyzer.get_defensive_line_depth(1) == 0.0
