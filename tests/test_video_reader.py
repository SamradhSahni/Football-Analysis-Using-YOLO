"""
tests/test_video_reader.py
==========================
Unit tests for src/ingestion/video_reader.py
PRD Reference: Task 1.1
"""

import sys
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ingestion.video_reader import VideoReader


@pytest.fixture
def synthetic_video(tmp_path):
    """Creates a synthetic 10-frame video for testing."""
    video_path = tmp_path / "test_video.mp4"
    width, height = 64, 64
    fps = 30.0
    num_frames = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a simple frame with changing color to verify frame order
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (i * 20, i * 20, i * 20)
        out.write(frame)

    out.release()
    return video_path, num_frames, width, height, fps


class TestVideoReader:
    def test_initialization(self, synthetic_video):
        video_path, num_frames, width, height, fps = synthetic_video
        reader = VideoReader(video_path)

        assert reader.video_path == video_path
        assert reader.frame_stride == 1
        assert reader.max_frames is None
        assert reader.width == width
        assert reader.height == height
        assert reader.fps == pytest.approx(fps)
        assert reader.total_frames == num_frames

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            VideoReader("non_existent_video.mkv")

    def test_read_all_frames(self, synthetic_video):
        video_path, num_frames, _, _, _ = synthetic_video
        reader = VideoReader(video_path)

        frames = list(reader.read_frames())
        
        assert len(frames) == num_frames
        
        # Check frame IDs (1-indexed)
        expected_ids = list(range(1, num_frames + 1))
        actual_ids = [f[0] for f in frames]
        assert actual_ids == expected_ids

        # Check frame content (we painted them with i*20)
        for i, (frame_id, frame) in enumerate(frames):
            assert abs(int(frame[0, 0, 0]) - (i * 20)) <= 10

    def test_max_frames(self, synthetic_video):
        video_path, num_frames, _, _, _ = synthetic_video
        reader = VideoReader(video_path, max_frames=3)

        frames = list(reader.read_frames())
        
        assert len(frames) == 3
        expected_ids = [1, 2, 3]
        actual_ids = [f[0] for f in frames]
        assert actual_ids == expected_ids

    def test_frame_stride(self, synthetic_video):
        video_path, num_frames, _, _, _ = synthetic_video
        # 10 frames total, stride 2 -> should yield frames 1, 3, 5, 7, 9
        reader = VideoReader(video_path, frame_stride=2)

        frames = list(reader.read_frames())
        
        assert len(frames) == 5
        expected_ids = [1, 3, 5, 7, 9]
        actual_ids = [f[0] for f in frames]
        assert actual_ids == expected_ids
        
        # Verify content matches original frame index (0-indexed)
        # frame_id 1 -> index 0 -> val 0
        # frame_id 3 -> index 2 -> val 40
        for idx, (frame_id, frame) in enumerate(frames):
            expected_val = (expected_ids[idx] - 1) * 20
            # Depending on video compression, exact pixel values might vary slightly, 
            # but for mp4v with simple colors it's usually okay, or we can use approx
            assert abs(int(frame[0, 0, 0]) - expected_val) <= 10

    def test_stride_and_max_frames(self, synthetic_video):
        video_path, num_frames, _, _, _ = synthetic_video
        # 10 frames total, stride 3 -> yields frames 1, 4, 7, 10
        # max_frames 2 -> yields frames 1, 4
        reader = VideoReader(video_path, frame_stride=3, max_frames=2)

        frames = list(reader.read_frames())
        
        assert len(frames) == 2
        expected_ids = [1, 4]
        actual_ids = [f[0] for f in frames]
        assert actual_ids == expected_ids
