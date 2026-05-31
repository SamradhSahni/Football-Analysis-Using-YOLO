import pytest

from src.analytics.possession import PossessionAnalyzer, _classify_zone
from src.data_models import Track, CLASS_BALL, CLASS_PLAYER


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


def _ball(x, y):
    return _player(track_id=99, x=x, y=y, team_id=None, class_id=CLASS_BALL)


# ── Zone classification ────────────────────────────────────────────────────────

def test_zone_classification():
    assert _classify_zone(10.0)  == "defensive_third"
    assert _classify_zone(52.5)  == "middle_third"
    assert _classify_zone(80.0)  == "attacking_third"
    assert _classify_zone(0.0)   == "defensive_third"
    assert _classify_zone(104.9) == "attacking_third"


# ── Basic possession logic ─────────────────────────────────────────────────────

def test_team1_possession_in_attacking_third():
    analyzer = PossessionAnalyzer(fps=25, max_ball_distance_m=5.0)

    # Ball at (80, 34) — attacking third
    # Team 1 player very close, Team 2 far away
    tracks = [
        _ball(80.0, 34.0),
        _player(1, 80.5, 34.0, team_id=1),     # 0.5m from ball
        _player(2, 20.0, 34.0, team_id=2),     # far
    ]
    for frame in range(50):
        analyzer.update(frame, tracks)

    stats = analyzer.get_possession_stats()
    # Team 1 should have 50 frames / 25 fps = 2.0 seconds in attacking_third
    assert stats[1]["attacking_third"] == pytest.approx(2.0, abs=0.1)
    assert stats[2]["attacking_third"] == 0.0


def test_no_ball_no_possession():
    analyzer = PossessionAnalyzer(fps=25)
    tracks = [_player(1, 50.0, 34.0, team_id=1)]

    for frame in range(25):
        analyzer.update(frame, tracks)

    pct = analyzer.get_possession_percentage()
    # No ball detected — both teams should get 50% (fallback)
    assert pct[1] == pytest.approx(50.0)
    assert pct[2] == pytest.approx(50.0)


def test_contested_ball_excluded_from_percentage():
    """If the ball is far from all players, frames are marked contested."""
    analyzer = PossessionAnalyzer(fps=25, max_ball_distance_m=2.0)

    # Ball at (50, 34), all players 20m away → contested
    tracks = [
        _ball(50.0, 34.0),
        _player(1, 20.0, 34.0, team_id=1),
        _player(2, 80.0, 34.0, team_id=2),
    ]
    for frame in range(50):
        analyzer.update(frame, tracks)

    stats = analyzer.get_possession_stats()
    # No one in possession → all zones should be 0 for both teams
    assert stats[1]["middle_third"] == 0.0
    assert stats[2]["middle_third"] == 0.0


def test_possession_percentage_sums_to_100():
    analyzer = PossessionAnalyzer(fps=25, max_ball_distance_m=5.0)

    # First 50 frames: Team 1 in possession
    team1_tracks = [_ball(80.0, 34.0), _player(1, 80.2, 34.0, team_id=1), _player(2, 10.0, 10.0, team_id=2)]
    # Next 50 frames: Team 2 in possession
    team2_tracks = [_ball(20.0, 34.0), _player(1, 80.0, 34.0, team_id=1), _player(2, 20.2, 34.0, team_id=2)]

    for frame in range(50):
        analyzer.update(frame, team1_tracks)
    for frame in range(50, 100):
        analyzer.update(frame, team2_tracks)

    pct = analyzer.get_possession_percentage()
    total = pct[1] + pct[2]
    assert total == pytest.approx(100.0, abs=0.1)


def test_possession_charts_created(tmp_path):
    analyzer = PossessionAnalyzer(fps=25, max_ball_distance_m=5.0)

    tracks = [_ball(80.0, 34.0), _player(1, 80.2, 34.0, team_id=1), _player(2, 10.0, 10.0, team_id=2)]
    for frame in range(25):
        analyzer.update(frame, tracks)

    bar_path = str(tmp_path / "possession_bar.png")
    pie_path = str(tmp_path / "possession_pie.png")

    analyzer.plot_possession_breakdown(bar_path)
    analyzer.plot_possession_pie(pie_path)

    from pathlib import Path
    assert Path(bar_path).exists() and Path(bar_path).stat().st_size > 0
    assert Path(pie_path).exists() and Path(pie_path).stat().st_size > 0
