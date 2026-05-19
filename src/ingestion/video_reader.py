"""
src/ingestion/video_reader.py
=============================
Stage 1 of the Football Tracker Pipeline.
Responsible for reading frames from an input video.

PRD Reference : Section 6, Task 1.1
Outputs       : Yields (frame_id, frame_np) tuples
                where frame_id is 1-indexed (per MOT convention)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


class VideoReader:
    """
    Wraps cv2.VideoCapture to provide a robust frame generator.
    
    Handles:
      - frame_stride: process every Nth frame (useful for speedup/debugging)
      - max_frames  : stop after reading this many frames
      - 1-indexed frame counting (matches MOT tracking ground truth)
    """

    def __init__(
        self,
        video_path: str | Path,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        video_path   : Path to input video file (.mkv, .mp4, etc.)
        frame_stride : Process every Nth frame. Defaults to 1 (all frames).
        max_frames   : Stop reading after yielding this many frames. Defaults to None.
        """
        self.video_path = Path(video_path)
        self.frame_stride = max(1, frame_stride)
        self.max_frames = max_frames

        if not self.video_path.exists() and "%" not in str(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Open just to read properties, then close so we don't hold the handle
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Warn if frame_stride > 1 as it might break tracking
        if self.frame_stride > 1:
            logging.warning(
                f"VideoReader: frame_stride={self.frame_stride}. "
                "Skipping frames can degrade ByteTrack performance."
            )

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields (frame_id, frame_image) tuples.
        frame_id is 1-indexed.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        frames_yielded = 0
        raw_frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                raw_frame_idx += 1

                # Skip frames based on stride
                if (raw_frame_idx - 1) % self.frame_stride != 0:
                    continue

                yield (raw_frame_idx, frame)
                frames_yielded += 1

                # Stop if max_frames reached
                if self.max_frames is not None and frames_yielded >= self.max_frames:
                    break
        finally:
            cap.release()

    def __str__(self) -> str:
        return (
            f"VideoReader({self.video_path.name} | "
            f"{self.width}x{self.height} @ {self.fps:.2f}fps | "
            f"Frames: {self.total_frames})"
        )
