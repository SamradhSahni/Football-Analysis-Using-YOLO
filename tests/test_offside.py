import numpy as np
import pytest

from src.analytics.offside import OffsideAnalyzer, _draw_dashed_line
from src.data_models import Track, CLASS_PLAYER, CLASS_GOALKEEPER


def _player(track_id, x, y, team_id, class_id=CLASS_PLAYER):
    return Track(
        track_id=track_id,
        bbox_px=(0, 0, 1, 1),
        centroid_px=(0.5, 0.5),
        centroid_world=(x, y),
        frame_id=1,
        team_id=team_id,
        class_id=class_id,
    )


# ── Offside line calculation ───────────────────────────────────────────────────

def test_offside_line_is_second_to_last_defender():
    analyzer = OffsideAnalyzer()

    # Team 2 defends the right goal (high X)
    # 5 players at X = 95, 80, 70, 60, 50 (sorted descending)
    # 2nd-to-last from goal = index 1 = X=80
    tracks = [
        _player(1, 95.0, 34.0, team_id=2),  # deepest (GK or last man)
        _player(2, 80.0, 34.0, team_id=2),  # second-to-last → offside line
        _player(3, 70.0, 34.0, team_id=2),
        _player(4, 60.0, 34.0, team_id=2),
        _player(5, 50.0, 34.0, team_id=2),
    ]
    offside_x = analyzer.get_offside_line(tracks, defending_team=2)
    assert offside_x == pytest.approx(80.0)


def test_offside_line_insufficient_players():
    analyzer = OffsideAnalyzer()
    tracks = [_player(1, 95.0, 34.0, team_id=2)]  # only 1 player
    result = analyzer.get_offside_line(tracks, defending_team=2)
    assert result is None


# ── Alert counting ─────────────────────────────────────────────────────────────

def test_attacker_beyond_line_gets_alert():
    analyzer = OffsideAnalyzer()

    # Team 2 defending — offside line at X=80 (second-to-last)
    # Team 1 attacker at X=85 → ahead of line → should be alerted
    tracks = [
        _player(10, 95.0, 34.0, team_id=2),
        _player(11, 80.0, 34.0, team_id=2),
        _player(12, 70.0, 34.0, team_id=2),
        _player(20, 85.0, 34.0, team_id=1),  # offside!
        _player(21, 30.0, 34.0, team_id=1),  # onside
    ]
    alerts = analyzer.update(frame_id=1, tracks=tracks)
    assert 20 in alerts[1]
    assert 21 not in alerts[1]


def test_onside_attacker_no_alert():
    analyzer = OffsideAnalyzer()

    tracks = [
        _player(10, 95.0, 34.0, team_id=2),
        _player(11, 80.0, 34.0, team_id=2),
        _player(12, 70.0, 34.0, team_id=2),
        _player(20, 75.0, 34.0, team_id=1),  # behind offside line → onside
    ]
    alerts = analyzer.update(frame_id=1, tracks=tracks)
    assert 20 not in alerts[1]


def test_alert_counts_accumulate():
    analyzer = OffsideAnalyzer()

    tracks = [
        _player(10, 95.0, 34.0, team_id=2),
        _player(11, 80.0, 34.0, team_id=2),
        _player(20, 85.0, 34.0, team_id=1),  # offside every frame
    ]
    for frame in range(10):
        analyzer.update(frame, tracks)

    counts = analyzer.get_alert_counts()
    assert counts[20] == 10


def test_no_crash_empty_tracks():
    analyzer = OffsideAnalyzer()
    alerts = analyzer.update(1, [])
    assert alerts == {1: [], 2: []}


# ── Plot ───────────────────────────────────────────────────────────────────────

def test_plot_pitch_with_offside_creates_file(tmp_path):
    analyzer = OffsideAnalyzer()

    tracks = [
        _player(1, 95.0, 10.0, team_id=2),
        _player(2, 80.0, 34.0, team_id=2),
        _player(3, 70.0, 58.0, team_id=2),
        _player(4, 85.0, 34.0, team_id=1),
    ]
    out = str(tmp_path / "offside.png")
    analyzer.plot_pitch_with_offside(tracks, out)

    from pathlib import Path
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


# ── Dashed line utility ────────────────────────────────────────────────────────

def test_dashed_line_does_not_crash():
    import numpy as np
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = _draw_dashed_line(img, (10, 50), (190, 50), color=(0, 255, 0), thickness=2)
    assert result.shape == img.shape
