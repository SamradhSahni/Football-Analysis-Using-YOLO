"""
tests/test_detector.py
======================
Unit tests for src/detection/detector.py
PRD Reference: Task 1.2
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.detection.detector import PlayerDetector
from src.data_models import CLASS_PLAYER, CLASS_BALL


class MockTensor:
    def __init__(self, data):
        self.data = np.array(data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data
    def __getitem__(self, idx):
        return MockTensor(self.data[idx])

class MockBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = MockTensor([xyxy])
        self.conf = MockTensor([conf])
        self.cls = MockTensor([cls])


class MockResults:
    def __init__(self, boxes):
        self.boxes = boxes


class MockYOLO:
    """Mock YOLO model to simulate detections without loading real weights."""
    def __init__(self, *args, **kwargs):
        self.device_set = None

    def to(self, device):
        self.device_set = device

    def __call__(self, frame, **kwargs):
        # Return synthetic results based on a dummy frame's properties
        # Let's say if the frame is completely black, no detections
        if np.sum(frame) == 0:
            return [MockResults([])]
            
        # Mock some detections
        # Player 1
        box1 = MockBox([100.0, 200.0, 150.0, 300.0], 0.9, CLASS_PLAYER)
        # Player 2
        box2 = MockBox([400.0, 100.0, 450.0, 250.0], 0.8, CLASS_PLAYER)
        # Ball
        box3 = MockBox([300.0, 300.0, 310.0, 310.0], 0.6, CLASS_BALL)
        # Invalid class (should be filtered out)
        box4 = MockBox([0.0, 0.0, 10.0, 10.0], 0.5, 99)
        
        return [MockResults([box1, box2, box3, box4])]


@pytest.fixture
def mock_yolo(monkeypatch):
    monkeypatch.setattr("src.detection.detector.YOLO", MockYOLO)


class TestPlayerDetector:
    def test_initialization(self, mock_yolo, tmp_path):
        model_path = tmp_path / "dummy_model.pt"
        # Create a dummy file so it passes the existence check and loads MockYOLO
        model_path.touch()
        
        detector = PlayerDetector(
            model_path=model_path,
            confidence=0.5,
            iou_threshold=0.6,
            input_size=640,
            device="cpu"
        )
        
        assert detector.model_path == model_path
        assert detector.confidence == 0.5
        assert detector.iou_threshold == 0.6
        assert detector.input_size == 640
        assert detector.device == "cpu"
        assert detector.model is not None
        assert detector.ball_model is None

    def test_detection_empty_frame(self, mock_yolo, tmp_path):
        model_path = tmp_path / "dummy_model.pt"
        model_path.touch()
        detector = PlayerDetector(model_path=model_path)
        
        # Black frame -> MockYOLO returns []
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=5)
        
        assert len(detections) == 0

    def test_detection_with_results(self, mock_yolo, tmp_path):
        model_path = tmp_path / "dummy_model.pt"
        model_path.touch()
        detector = PlayerDetector(model_path=model_path)
        
        # Non-black frame -> MockYOLO returns 4 boxes (one invalid class)
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        detections = detector.detect(frame, frame_id=10)
        
        # Should have 3 valid detections (the class 99 is ignored)
        assert len(detections) == 3
        
        # Check frame_ids
        for det in detections:
            assert det.frame_id == 10
            
        # Check specific mapped detections
        assert detections[0].class_id == CLASS_PLAYER
        assert detections[0].bbox_px == (100.0, 200.0, 150.0, 300.0)
        assert detections[0].confidence == pytest.approx(0.9)
        
        assert detections[1].class_id == CLASS_PLAYER
        assert detections[2].class_id == CLASS_BALL
        assert detections[2].bbox_px == (300.0, 300.0, 310.0, 310.0)
