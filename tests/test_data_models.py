"""
tests/test_data_models.py
=========================
Unit tests for src/data_models.py
PRD Reference: Section 8, Task 0.2

Run with:  pytest tests/test_data_models.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.data_models import (
    Detection, Track, TrajectoryPoint, PlayerMetrics,
    CLASS_PLAYER, CLASS_GOALKEEPER, CLASS_REFEREE, CLASS_BALL,
    CLASS_ID_TO_NAME, SOCCERNET_TO_YOLO_CLASS,
    TEAM_A, TEAM_B, TEAM_REF,
)


# ===========================================================================
# CONSTANTS
# ===========================================================================

class TestConstants:
    def test_class_ids_are_zero_indexed(self):
        assert CLASS_PLAYER == 0
        assert CLASS_GOALKEEPER == 1
        assert CLASS_REFEREE == 2
        assert CLASS_BALL == 3

    def test_class_id_to_name_mapping(self):
        assert CLASS_ID_TO_NAME[0] == "player"
        assert CLASS_ID_TO_NAME[1] == "goalkeeper"
        assert CLASS_ID_TO_NAME[2] == "referee"
        assert CLASS_ID_TO_NAME[3] == "ball"

    def test_soccernet_to_yolo_mapping(self):
        # SoccerNet uses 1-indexed class IDs; YOLO uses 0-indexed
        assert SOCCERNET_TO_YOLO_CLASS[1] == CLASS_PLAYER
        assert SOCCERNET_TO_YOLO_CLASS[2] == CLASS_GOALKEEPER
        assert SOCCERNET_TO_YOLO_CLASS[3] == CLASS_REFEREE
        assert SOCCERNET_TO_YOLO_CLASS[4] == CLASS_BALL

    def test_team_ids(self):
        assert TEAM_A == 0
        assert TEAM_B == 1
        assert TEAM_REF == 2


# ===========================================================================
# DETECTION
# ===========================================================================

class TestDetection:

    def _make(self, **kwargs):
        defaults = dict(
            frame_id=1,
            bbox_px=(100.0, 200.0, 160.0, 380.0),
            confidence=0.85,
            class_id=CLASS_PLAYER,
            class_name="player",
        )
        defaults.update(kwargs)
        return Detection(**defaults)

    # --- field storage ---
    def test_fields_stored_correctly(self):
        d = self._make()
        assert d.frame_id == 1
        assert d.bbox_px == (100.0, 200.0, 160.0, 380.0)
        assert d.confidence == 0.85
        assert d.class_id == CLASS_PLAYER
        assert d.class_name == "player"

    # --- derived properties ---
    def test_width_px(self):
        d = self._make(bbox_px=(100.0, 200.0, 160.0, 380.0))
        assert d.width_px == pytest.approx(60.0)

    def test_height_px(self):
        d = self._make(bbox_px=(100.0, 200.0, 160.0, 380.0))
        assert d.height_px == pytest.approx(180.0)

    def test_area_px(self):
        d = self._make(bbox_px=(100.0, 200.0, 160.0, 380.0))
        assert d.area_px == pytest.approx(60.0 * 180.0)

    def test_centroid_px(self):
        d = self._make(bbox_px=(100.0, 200.0, 160.0, 380.0))
        cx, cy = d.centroid_px
        assert cx == pytest.approx(130.0)
        assert cy == pytest.approx(290.0)

    # --- validation ---
    def test_invalid_confidence_above_1(self):
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=1.5)

    def test_invalid_confidence_below_0(self):
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=-0.1)

    def test_invalid_class_id(self):
        with pytest.raises(ValueError, match="class_id"):
            self._make(class_id=99)

    def test_invalid_bbox_x2_le_x1(self):
        with pytest.raises(ValueError, match="bbox_px"):
            self._make(bbox_px=(200.0, 100.0, 100.0, 300.0))  # x2 <= x1

    def test_invalid_bbox_y2_le_y1(self):
        with pytest.raises(ValueError, match="bbox_px"):
            self._make(bbox_px=(100.0, 300.0, 200.0, 200.0))  # y2 <= y1

    # --- all four class_ids valid ---
    @pytest.mark.parametrize("class_id,name", [
        (0, "player"), (1, "goalkeeper"), (2, "referee"), (3, "ball")
    ])
    def test_all_class_ids_valid(self, class_id, name):
        d = self._make(class_id=class_id, class_name=name)
        assert d.class_id == class_id

    # --- confidence boundary values ---
    def test_confidence_boundary_zero(self):
        d = self._make(confidence=0.0)
        assert d.confidence == 0.0

    def test_confidence_boundary_one(self):
        d = self._make(confidence=1.0)
        assert d.confidence == 1.0

    # --- factory classmethod ---
    def test_from_yolo_factory(self):
        d = Detection.from_yolo(
            frame_id=5,
            x1=50.0, y1=100.0, x2=120.0, y2=250.0,
            confidence=0.72,
            class_id=CLASS_GOALKEEPER,
        )
        assert d.frame_id == 5
        assert d.bbox_px == (50.0, 100.0, 120.0, 250.0)
        assert d.confidence == pytest.approx(0.72)
        assert d.class_id == CLASS_GOALKEEPER
        assert d.class_name == "goalkeeper"

    def test_from_yolo_ball(self):
        d = Detection.from_yolo(1, 0.0, 0.0, 10.0, 10.0, 0.55, CLASS_BALL)
        assert d.class_name == "ball"


# ===========================================================================
# TRACK
# ===========================================================================

class TestTrack:

    def _make(self, **kwargs):
        defaults = dict(
            track_id=7,
            frame_id=10,
            bbox_px=(200.0, 300.0, 260.0, 480.0),
            centroid_px=(230.0, 390.0),
            centroid_world=None,
            class_id=CLASS_PLAYER,
            team_id=None,
        )
        defaults.update(kwargs)
        return Track(**defaults)

    # --- field storage ---
    def test_fields_stored_correctly(self):
        t = self._make()
        assert t.track_id == 7
        assert t.frame_id == 10
        assert t.bbox_px == (200.0, 300.0, 260.0, 480.0)
        assert t.centroid_px == (230.0, 390.0)
        assert t.centroid_world is None
        assert t.class_id == CLASS_PLAYER
        assert t.team_id is None

    # --- derived properties ---
    def test_class_name_player(self):
        assert self._make(class_id=CLASS_PLAYER).class_name == "player"

    def test_class_name_goalkeeper(self):
        assert self._make(class_id=CLASS_GOALKEEPER).class_name == "goalkeeper"

    def test_class_name_referee(self):
        assert self._make(class_id=CLASS_REFEREE).class_name == "referee"

    def test_class_name_ball(self):
        assert self._make(class_id=CLASS_BALL).class_name == "ball"

    def test_team_name_none_when_unset(self):
        assert self._make(team_id=None).team_name is None

    def test_team_name_team_a(self):
        assert self._make(team_id=TEAM_A).team_name == "team_A"

    def test_team_name_team_b(self):
        assert self._make(team_id=TEAM_B).team_name == "team_B"

    def test_team_name_referee(self):
        assert self._make(team_id=TEAM_REF).team_name == "referee"

    def test_has_world_coords_false_when_none(self):
        assert self._make(centroid_world=None).has_world_coords is False

    def test_has_world_coords_true_when_set(self):
        t = self._make(centroid_world=(52.5, 34.0))
        assert t.has_world_coords is True

    def test_world_coords_stored(self):
        t = self._make(centroid_world=(30.0, 15.5))
        assert t.centroid_world == (30.0, 15.5)

    # --- factory from Detection ---
    def test_from_detection_factory(self):
        det = Detection(
            frame_id=3,
            bbox_px=(100.0, 200.0, 160.0, 380.0),
            confidence=0.9,
            class_id=CLASS_PLAYER,
            class_name="player",
        )
        t = Track.from_detection(det, track_id=42)
        assert t.track_id == 42
        assert t.frame_id == 3
        assert t.bbox_px == (100.0, 200.0, 160.0, 380.0)
        assert t.centroid_px == pytest.approx((130.0, 290.0))
        assert t.class_id == CLASS_PLAYER
        assert t.centroid_world is None


# ===========================================================================
# TRAJECTORY POINT
# ===========================================================================

class TestTrajectoryPoint:

    def _make(self, **kwargs):
        defaults = dict(
            frame_id=1,
            timestamp_s=0.0,
            x_m=52.5,
            y_m=34.0,
            speed_ms=None,
            speed_kmh=None,
        )
        defaults.update(kwargs)
        return TrajectoryPoint(**defaults)

    # --- field storage ---
    def test_fields_stored_correctly(self):
        tp = self._make(frame_id=25, timestamp_s=1.0, x_m=20.0, y_m=10.0)
        assert tp.frame_id == 25
        assert tp.timestamp_s == pytest.approx(1.0)
        assert tp.x_m == pytest.approx(20.0)
        assert tp.y_m == pytest.approx(10.0)
        assert tp.speed_ms is None
        assert tp.speed_kmh is None

    # --- speed fields ---
    def test_speed_fields_set_together(self):
        tp = self._make(speed_ms=5.0, speed_kmh=18.0)
        assert tp.speed_ms == pytest.approx(5.0)
        assert tp.speed_kmh == pytest.approx(18.0)

    def test_speed_ms_only_raises(self):
        with pytest.raises(ValueError):
            self._make(speed_ms=5.0, speed_kmh=None)

    def test_speed_kmh_only_raises(self):
        with pytest.raises(ValueError):
            self._make(speed_ms=None, speed_kmh=18.0)

    # --- validation ---
    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError, match="timestamp_s"):
            self._make(timestamp_s=-1.0)

    def test_negative_speed_raises(self):
        with pytest.raises(ValueError, match="speed_ms"):
            self._make(speed_ms=-1.0, speed_kmh=-3.6)

    def test_zero_timestamp_valid(self):
        tp = self._make(timestamp_s=0.0)
        assert tp.timestamp_s == 0.0

    def test_zero_speed_valid(self):
        tp = self._make(speed_ms=0.0, speed_kmh=0.0)
        assert tp.speed_ms == 0.0

    # --- with_speed immutable builder ---
    def test_with_speed_returns_new_object(self):
        tp = self._make()
        tp2 = tp.with_speed(7.5)
        assert tp is not tp2

    def test_with_speed_populates_both_fields(self):
        tp = self._make()
        tp2 = tp.with_speed(10.0)
        assert tp2.speed_ms == pytest.approx(10.0)
        assert tp2.speed_kmh == pytest.approx(36.0)

    def test_with_speed_preserves_position(self):
        tp = self._make(x_m=30.0, y_m=20.0)
        tp2 = tp.with_speed(5.0)
        assert tp2.x_m == pytest.approx(30.0)
        assert tp2.y_m == pytest.approx(20.0)

    # --- from_track factory ---
    def test_from_track_success(self):
        track = Track(
            track_id=1, frame_id=25,
            bbox_px=(0.0, 0.0, 50.0, 100.0),
            centroid_px=(25.0, 50.0),
            centroid_world=(52.5, 34.0),
        )
        tp = TrajectoryPoint.from_track(track, fps=25.0)
        assert tp.frame_id == 25
        assert tp.timestamp_s == pytest.approx(24.0 / 25.0)   # (25-1)/25
        assert tp.x_m == pytest.approx(52.5)
        assert tp.y_m == pytest.approx(34.0)
        assert tp.speed_ms is None

    def test_from_track_raises_without_world_coords(self):
        track = Track(
            track_id=1, frame_id=1,
            bbox_px=(0.0, 0.0, 50.0, 100.0),
            centroid_px=(25.0, 50.0),
            centroid_world=None,
        )
        with pytest.raises(ValueError, match="centroid_world"):
            TrajectoryPoint.from_track(track, fps=25.0)

    def test_timestamp_formula_frame1(self):
        """Frame 1 at 25fps → timestamp_s = 0.0"""
        track = Track(
            track_id=1, frame_id=1,
            bbox_px=(0.0, 0.0, 10.0, 10.0),
            centroid_px=(5.0, 5.0),
            centroid_world=(0.0, 0.0),
        )
        tp = TrajectoryPoint.from_track(track, fps=25.0)
        assert tp.timestamp_s == pytest.approx(0.0)


# ===========================================================================
# PLAYER METRICS
# ===========================================================================

class TestPlayerMetrics:

    def _make(self, **kwargs):
        defaults = dict(
            track_id=5,
            team_id=TEAM_A,
            total_distance_km=10.2,
            avg_speed_kmh=7.5,
            max_speed_kmh=31.2,
            sprint_count=8,
            max_sprint_speed_kmh=30.5,
            total_sprint_distance_km=1.2,
            time_in_zones={
                "defensive_third": 120.0,
                "middle_third":    300.0,
                "attacking_third": 480.0,
            },
            role_adherence_pct=None,
            fatigue_index=None,
            workload_index=None,
        )
        defaults.update(kwargs)
        return PlayerMetrics(**defaults)

    # --- field storage ---
    def test_fields_stored_correctly(self):
        m = self._make()
        assert m.track_id == 5
        assert m.team_id == TEAM_A
        assert m.total_distance_km == pytest.approx(10.2)
        assert m.avg_speed_kmh == pytest.approx(7.5)
        assert m.max_speed_kmh == pytest.approx(31.2)
        assert m.sprint_count == 8
        assert m.max_sprint_speed_kmh == pytest.approx(30.5)
        assert m.total_sprint_distance_km == pytest.approx(1.2)

    # --- time_in_zones ---
    def test_time_in_zones_all_thirds(self):
        m = self._make()
        assert m.time_in_zones["defensive_third"] == pytest.approx(120.0)
        assert m.time_in_zones["middle_third"] == pytest.approx(300.0)
        assert m.time_in_zones["attacking_third"] == pytest.approx(480.0)

    # --- derived properties ---
    def test_team_name_team_a(self):
        assert self._make(team_id=TEAM_A).team_name == "team_A"

    def test_team_name_none(self):
        assert self._make(team_id=None).team_name is None

    def test_is_fatigued_none_when_no_index(self):
        assert self._make(fatigue_index=None).is_fatigued is None

    def test_is_fatigued_true_when_below_1(self):
        assert self._make(fatigue_index=0.88).is_fatigued is True

    def test_is_fatigued_false_when_above_1(self):
        assert self._make(fatigue_index=1.05).is_fatigued is False

    def test_is_fatigued_false_when_exactly_1(self):
        assert self._make(fatigue_index=1.0).is_fatigued is False

    # --- validation ---
    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="total_distance_km"):
            self._make(total_distance_km=-1.0)

    def test_negative_avg_speed_raises(self):
        with pytest.raises(ValueError, match="avg_speed_kmh"):
            self._make(avg_speed_kmh=-0.1)

    def test_negative_max_speed_raises(self):
        with pytest.raises(ValueError, match="max_speed_kmh"):
            self._make(max_speed_kmh=-5.0)

    def test_negative_sprint_count_raises(self):
        with pytest.raises(ValueError, match="sprint_count"):
            self._make(sprint_count=-1)

    def test_workload_above_1_raises(self):
        with pytest.raises(ValueError, match="workload_index"):
            self._make(workload_index=1.1)

    def test_workload_below_0_raises(self):
        with pytest.raises(ValueError, match="workload_index"):
            self._make(workload_index=-0.1)

    def test_role_adherence_above_100_raises(self):
        with pytest.raises(ValueError, match="role_adherence_pct"):
            self._make(role_adherence_pct=101.0)

    def test_role_adherence_below_0_raises(self):
        with pytest.raises(ValueError, match="role_adherence_pct"):
            self._make(role_adherence_pct=-1.0)

    def test_workload_boundary_zero(self):
        m = self._make(workload_index=0.0)
        assert m.workload_index == 0.0

    def test_workload_boundary_one(self):
        m = self._make(workload_index=1.0)
        assert m.workload_index == 1.0

    # --- CSV export ---
    def test_to_csv_row_has_correct_columns(self):
        m = self._make(role_adherence_pct=75.0, fatigue_index=0.92, workload_index=0.65)
        row = m.to_csv_row()
        expected_cols = [
            "track_id", "team_id", "total_distance_km", "avg_speed_kmh",
            "max_speed_kmh", "sprint_count", "max_sprint_speed_kmh",
            "total_sprint_distance_km", "time_defensive_third_s",
            "time_middle_third_s", "time_attacking_third_s",
            "role_adherence_pct", "fatigue_index", "workload_index",
        ]
        for col in expected_cols:
            assert col in row, f"Missing column: {col}"

    def test_to_csv_row_zone_times(self):
        m = self._make()
        row = m.to_csv_row()
        assert row["time_defensive_third_s"] == pytest.approx(120.0)
        assert row["time_middle_third_s"] == pytest.approx(300.0)
        assert row["time_attacking_third_s"] == pytest.approx(480.0)

    def test_to_csv_row_missing_zone_defaults_to_zero(self):
        m = self._make(time_in_zones={})
        row = m.to_csv_row()
        assert row["time_defensive_third_s"] == 0.0
        assert row["time_middle_third_s"] == 0.0
        assert row["time_attacking_third_s"] == 0.0

    def test_to_csv_row_values_rounded(self):
        m = self._make(total_distance_km=10.123456789)
        row = m.to_csv_row()
        # Should be rounded to 4 decimal places
        assert row["total_distance_km"] == pytest.approx(10.1235, abs=1e-4)

    def test_to_csv_row_track_id_correct(self):
        m = self._make(track_id=42)
        assert m.to_csv_row()["track_id"] == 42


# ===========================================================================
# INTEGRATION — Detection → Track → TrajectoryPoint chain
# ===========================================================================

class TestDataModelChain:
    """Verify the Stage 2 → 3 → 4 data flow works end-to-end."""

    def test_detection_to_track_to_trajectory(self):
        # Stage 2: Detection from YOLOv8
        det = Detection.from_yolo(
            frame_id=25,
            x1=900.0, y1=400.0, x2=960.0, y2=580.0,
            confidence=0.91,
            class_id=CLASS_PLAYER,
        )
        assert det.class_name == "player"

        # Stage 3: ByteTrack assigns ID → Track
        track = Track.from_detection(det, track_id=3)
        assert track.track_id == 3
        assert track.has_world_coords is False

        # Stage 4: Homography sets world coords
        track.centroid_world = (45.0, 20.0)
        assert track.has_world_coords is True

        # Stage 5: TrajectoryPoint
        tp = TrajectoryPoint.from_track(track, fps=25.0)
        assert tp.x_m == pytest.approx(45.0)
        assert tp.y_m == pytest.approx(20.0)
        assert tp.timestamp_s == pytest.approx(24.0 / 25.0)

        # Stage 5: Speed computation
        tp2 = tp.with_speed(8.5)
        assert tp2.speed_ms == pytest.approx(8.5)
        assert tp2.speed_kmh == pytest.approx(30.6)

    def test_full_pipeline_produces_valid_metrics(self):
        m = PlayerMetrics(
            track_id=3,
            team_id=TEAM_A,
            total_distance_km=11.4,
            avg_speed_kmh=8.1,
            max_speed_kmh=32.0,
            sprint_count=12,
            max_sprint_speed_kmh=31.2,
            total_sprint_distance_km=1.8,
            time_in_zones={
                "defensive_third": 90.0,
                "middle_third": 360.0,
                "attacking_third": 450.0,
            },
            fatigue_index=0.91,
            workload_index=0.74,
        )
        assert m.is_fatigued is True
        row = m.to_csv_row()
        assert row["track_id"] == 3
        assert row["sprint_count"] == 12
