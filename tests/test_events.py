import json
from pathlib import Path

import pytest

from src.analytics.events import EventLoader, EventCorrelator, MatchEvent, EventSnapshot
from src.data_models import Track


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_annotation_file(tmp_path):
    """Write a minimal Labels-v2.json file to a temp location."""
    data = {
        "UrlLocal": "england_epl/2014-2015/2015-05-17 - 22-00 Man City 2 - 0 QPR/1_720p.mkv",
        "annotations": [
            {"gameTime": "1 - 05:00", "label": "Goal", "position": "300000", "team": "home", "visibility": "visible"},
            {"gameTime": "1 - 10:00", "label": "Yellow card", "position": "600000", "team": "away", "visibility": "visible"},
            {"gameTime": "MALFORMED", "label": "Bad", "position": "NOT_INT", "team": "home", "visibility": "visible"},
        ]
    }
    p = tmp_path / "Labels-v2.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def _make_track(track_id, x, y, team_id):
    return Track(
        track_id=track_id,
        bbox_px=(0, 0, 1, 1),
        centroid_px=(0.5, 0.5),
        centroid_world=(x, y),
        frame_id=1,
        team_id=team_id,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_event_loader_parses_valid_annotations(sample_annotation_file):
    loader = EventLoader()
    events = loader.load(sample_annotation_file)
    
    # Only 2 valid events (third has non-int position)
    assert len(events) == 2
    assert events[0].label == "Goal"
    assert events[0].timestamp_ms == 300_000
    assert events[1].label == "Yellow card"
    assert events[1].team == "away"


def test_event_loader_missing_file():
    loader = EventLoader()
    events = loader.load("/nonexistent/path.json")
    assert events == []


def test_map_to_frames():
    loader = EventLoader()
    events = [
        MatchEvent(label="Goal", timestamp_ms=300_000, game_time="1 - 05:00", team="home", visibility="visible"),
        MatchEvent(label="Card", timestamp_ms=0, game_time="1 - 00:00", team="away", visibility="visible"),
    ]
    result = loader.map_to_frames(events, fps=25)
    
    # 300s * 25fps = 7500
    assert result[0].frame_id == 7500
    # 0ms -> frame 0
    assert result[1].frame_id == 0


def test_correlator_buffers_and_correlates():
    correlator = EventCorrelator(fps=25, snapshot_window_frames=10)
    
    tracks = [_make_track(1, 50.0, 30.0, team_id=1)]
    correlator.ingest_frame(100, tracks)
    correlator.ingest_frame(101, tracks)
    
    events = [
        MatchEvent(label="Goal", timestamp_ms=0, game_time="", team="home", visibility="visible", frame_id=100),
    ]
    
    snapshots = correlator.correlate(events)
    
    assert len(snapshots) == 1
    assert snapshots[0].event.label == "Goal"
    assert len(snapshots[0].tracks) == 1


def test_correlator_out_of_window():
    """Events far from any buffered frame should get empty tracks."""
    correlator = EventCorrelator(fps=25, snapshot_window_frames=5)
    
    correlator.ingest_frame(100, [_make_track(1, 50.0, 30.0, 1)])
    
    events = [
        MatchEvent(label="Goal", timestamp_ms=0, game_time="", team="home", visibility="visible", frame_id=9999),
    ]
    snapshots = correlator.correlate(events)
    
    assert len(snapshots) == 1
    assert snapshots[0].tracks == []


def test_render_event_snapshot(tmp_path):
    correlator = EventCorrelator(fps=25)
    
    tracks = [_make_track(i, float(10 * i + 5), 34.0, team_id=(i % 2) + 1) for i in range(6)]
    event = MatchEvent(label="Goal", timestamp_ms=0, game_time="1 - 05:00", team="home", visibility="visible", frame_id=0)
    snapshot = EventSnapshot(event=event, tracks=tracks)
    
    out = str(tmp_path / "test_event.png")
    correlator.render_event_snapshot(snapshot, out)
    
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_build_event_table():
    correlator = EventCorrelator(fps=25)
    
    tracks = [_make_track(i, float(10 * i), 34.0, team_id=1) for i in range(3)]
    event = MatchEvent(label="Goal", timestamp_ms=300_000, game_time="1 - 05:00", team="home", visibility="visible", frame_id=7500)
    snapshots = [EventSnapshot(event=event, tracks=tracks)]
    
    table = correlator.build_event_table(snapshots)
    
    assert len(table) == 1
    assert table[0]["label"] == "Goal"
    assert table[0]["players_visible"] == 3
    assert table[0]["frame_id"] == 7500
