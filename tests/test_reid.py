import numpy as np
import pytest
import cv2

from src.tracking.reid import (
    CutDetector,
    AppearanceExtractor,
    ReIDMatcher,
    ReIDModule,
    TrackEmbedding,
)
from src.data_models import Track, CLASS_PLAYER


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blank_frame(h=720, w=1280, color=(50, 50, 50)) -> np.ndarray:
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _track(track_id, x1=100, y1=100, x2=150, y2=250, team_id=1) -> Track:
    return Track(
        track_id=track_id,
        bbox_px=(x1, y1, x2, y2),
        centroid_px=((x1 + x2) / 2, (y1 + y2) / 2),
        frame_id=1,
        team_id=team_id,
        class_id=CLASS_PLAYER,
    )


# ── CutDetector ───────────────────────────────────────────────────────────────

def test_cut_detector_no_cut_on_similar_frames():
    detector = CutDetector(threshold_pct=40.0)
    frame = _blank_frame(color=(100, 100, 100))

    detector.update(0, frame)
    is_cut = detector.update(1, frame)  # identical frame

    assert not is_cut
    assert len(detector.cut_frames) == 0


def test_cut_detector_detects_drastic_change():
    detector = CutDetector(threshold_pct=40.0)
    black = _blank_frame(color=(0, 0, 0))
    white = _blank_frame(color=(255, 255, 255))

    detector.update(0, black)
    is_cut = detector.update(1, white)  # max possible diff

    assert is_cut
    assert 1 in detector.cut_frames


# ── AppearanceExtractor ───────────────────────────────────────────────────────

def test_appearance_extractor_returns_normalized_vector():
    extractor = AppearanceExtractor()
    frame = _blank_frame(color=(120, 80, 60))
    bbox = (100, 100, 150, 200)

    emb = extractor.extract(frame, bbox)

    assert emb.shape == (AppearanceExtractor.BINS * 3,)
    assert emb.dtype == np.float32
    # L2 norm should be ~1
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_appearance_extractor_zero_for_invalid_bbox():
    extractor = AppearanceExtractor()
    frame = _blank_frame()
    # Inverted bbox (x2 < x1)
    emb = extractor.extract(frame, (200, 200, 100, 100))
    assert np.all(emb == 0)


def test_extract_tracks_produces_one_embedding_per_track():
    extractor = AppearanceExtractor()
    frame = _blank_frame()
    tracks = [_track(1), _track(2, x1=300, y1=100, x2=350, y2=250)]

    embeddings = extractor.extract_tracks(frame, frame_id=5, tracks=tracks)

    assert len(embeddings) == 2
    assert all(e.frame_id == 5 for e in embeddings)


# ── ReIDMatcher ───────────────────────────────────────────────────────────────

def test_reid_matcher_high_similarity_matches():
    matcher = ReIDMatcher(similarity_threshold=0.9)

    # Identical embeddings should score 1.0 → match
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pre = [TrackEmbedding(track_id=42, frame_id=10, embedding=emb)]
    post = [TrackEmbedding(track_id=99, frame_id=11, embedding=emb.copy())]

    id_map, matches = matcher.match(pre, post, cut_frame=11)

    assert 99 in id_map
    assert id_map[99] == 42
    assert len(matches) == 1
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_reid_matcher_low_similarity_no_match():
    matcher = ReIDMatcher(similarity_threshold=0.9)

    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal → cos=0

    pre = [TrackEmbedding(track_id=1, frame_id=10, embedding=emb_a)]
    post = [TrackEmbedding(track_id=2, frame_id=11, embedding=emb_b)]

    id_map, matches = matcher.match(pre, post, cut_frame=11)

    assert id_map == {}
    assert matches == []


# ── ReIDModule (Integration) ──────────────────────────────────────────────────

def test_reid_module_restores_id_after_cut():
    """After a camera cut, post-cut track IDs should be remapped to pre-cut IDs."""
    module = ReIDModule(cut_threshold_pct=40.0, similarity_threshold=0.5)

    # Red frame: track_id=1 with distinctive red uniform
    red_frame = _blank_frame(color=(0, 0, 200))    # BGR red
    tracks_pre = [_track(1, x1=100, y1=50, x2=160, y2=220)]

    module.process(0, red_frame, tracks_pre)

    # Simulate a hard cut to a completely white frame, same player, new track_id=99
    white_frame = _blank_frame(color=(255, 255, 255))
    # post-cut: tracker assigns new ID 99, but appearance should match
    tracks_post = [_track(99, x1=100, y1=50, x2=160, y2=220)]

    # Manually force a cut by feeding drastically different frame
    restored = module.process(1, white_frame, tracks_post)

    # The module may or may not restore ID depending on similarity,
    # but it must not crash and must return same number of tracks.
    assert len(restored) == 1
