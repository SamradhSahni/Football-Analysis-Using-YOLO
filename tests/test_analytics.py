"""
tests/test_analytics.py
=======================
Unit tests for src/analytics/metrics.py
PRD Reference: Task 1.7
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_models import Track, CLASS_PLAYER
from src.analytics.metrics import MetricsEngine


@pytest.fixture
def mock_track_factory():
    """Helper to create Tracks easily for tests."""
    def _create(track_id, frame_id, x_m, y_m):
        return Track(
            track_id=track_id,
            frame_id=frame_id,
            bbox_px=(0.0, 0.0, 10.0, 10.0),
            centroid_px=(5.0, 10.0),
            centroid_world=(float(x_m), float(y_m)),
            class_id=CLASS_PLAYER,
            team_id=1
        )
    return _create


class TestMetricsEngine:
    def test_initialization(self):
        engine = MetricsEngine(fps=10.0)
        assert engine.fps == 10.0
        assert engine.SPRINT_FRAMES_REQ == 10  # 1.0s * 10 fps

    def test_distance_and_speed(self, mock_track_factory):
        engine = MetricsEngine(fps=1.0) # 1 frame per second to make math easy
        
        # Frame 1: (0, 0) -> speed 0
        t1 = mock_track_factory(1, 1, 0.0, 0.0)
        engine.update([t1])
        
        # Frame 2: (3, 4) -> distance = 5m, dt = 1s -> raw speed = 5 m/s
        # EWMA with alpha = 0.3: 0.3*5 + 0.7*0 = 1.5 m/s
        t2 = mock_track_factory(1, 2, 3.0, 4.0)
        engine.update([t2])
        
        metrics = engine.get_metrics()
        
        assert 1 in metrics
        pm = metrics[1]
        
        assert pm.total_distance_km == pytest.approx(5.0 / 1000.0)
        assert engine.trajectories[1][-1].speed_ms == pytest.approx(1.5)
        # 1.5 m/s * 3.6 = 5.4 km/h
        assert engine.trajectories[1][-1].speed_kmh == pytest.approx(5.4)

    def test_sprint_counting(self, mock_track_factory):
        engine = MetricsEngine(fps=1.0)
        
        # Sprints > 20 km/h (which is > 5.55 m/s)
        # To get smoothed speed > 20 km/h, we need a high raw speed
        # For simplicity, we just inject points with huge distance jumps
        
        # Frame 1: origin
        engine.update([mock_track_factory(1, 1, 0.0, 0.0)])
        
        # Let's jump 20 meters every second (raw speed = 20 m/s = 72 km/h)
        for i in range(2, 6):
            engine.update([mock_track_factory(1, i, float((i-1)*20), 0.0)])
            
        # At this point, smoothed speed will ramp up quickly and cross 20km/h.
        # Sprint requirement is 1.0 sec (1 frame at 1 fps). 
        # But sprints are registered only when they end or when finalize_sprints is called.
        
        # Stop moving to end the sprint
        engine.update([mock_track_factory(1, 6, 80.0, 0.0)])
        engine.update([mock_track_factory(1, 7, 80.0, 0.0)])
        
        metrics = engine.get_metrics()
        # Should have registered exactly 1 sprint
        assert metrics[1].sprint_count == 1

    def test_zones(self, mock_track_factory):
        engine = MetricsEngine(fps=10.0)
        dt = 0.1
        
        # Def Third (x <= 35)
        engine.update([mock_track_factory(1, 1, 10.0, 34.0)])
        # Mid Third (35 < x <= 70)
        engine.update([mock_track_factory(1, 2, 50.0, 34.0)])
        # Att Third (x > 70)
        engine.update([mock_track_factory(1, 3, 90.0, 34.0)])
        
        metrics = engine.get_metrics()
        
        assert metrics[1].time_in_zones["defensive_third"] == pytest.approx(0.1)
        assert metrics[1].time_in_zones["middle_third"] == pytest.approx(0.1)
        assert metrics[1].time_in_zones["attacking_third"] == pytest.approx(0.1)

    def test_fatigue_index_calculation(self, mock_track_factory):
        engine = MetricsEngine(fps=1.0)
        
        # Manually force internal states to test the formula
        from src.data_models import TrajectoryPoint
        engine.trajectories[1] = [TrajectoryPoint(frame_id=1, timestamp_s=0.0, x_m=0.0, y_m=0.0, speed_ms=0.0, speed_kmh=0.0)]
        engine.total_distances[1] = 5000.0  # 5 km
        engine.sprint_counts[1] = 3
        engine._sprint_frames[1] = 0
        engine.team_affiliations[1] = 1
        engine.zone_times[1] = {"defensive_third": 0.0, "middle_third": 0.0, "attacking_third": 0.0}
        
        # FI = (3 * 0.1) + (5.0 / 10.0) = 0.3 + 0.5 = 0.8
        metrics = engine.get_metrics()
        
        assert metrics[1].workload_index == pytest.approx(0.8)
        # Oh, data_models.py: is_fatigued is True if >= 0.8 maybe?
        # Actually in test_data_models.py we tested exactly 1.0
        # Let's just check workload_index first
        
    def test_workload_clamping(self, mock_track_factory):
        engine = MetricsEngine(fps=1.0)
        
        from src.data_models import TrajectoryPoint
        engine.trajectories[1] = [TrajectoryPoint(frame_id=1, timestamp_s=0.0, x_m=0.0, y_m=0.0, speed_ms=0.0, speed_kmh=0.0)]
        engine.total_distances[1] = 20000.0  # 20 km -> 2.0
        engine.sprint_counts[1] = 50         # 50 -> 5.0
        engine._sprint_frames[1] = 0
        engine.zone_times[1] = {"defensive_third": 0.0, "middle_third": 0.0, "attacking_third": 0.0}
        
        # 7.0 workload clamped to 1.0
        metrics = engine.get_metrics()
        
        assert metrics[1].workload_index == pytest.approx(1.0)
