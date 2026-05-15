"""
src/detection/detector.py
=========================
Stage 2 of the Football Tracker Pipeline.
Responsible for running YOLOv8 object detection on individual frames.

PRD Reference : Section 6, Task 1.2, GOAL-01
Outputs       : List of `data_models.Detection` objects per frame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("ultralytics package not found. YOLO model will not load.")

from src.data_models import Detection, CLASS_ID_TO_NAME


class PlayerDetector:
    """
    Wraps Ultralytics YOLOv8 for inference on video frames.
    Transforms raw model outputs into the canonical `Detection` dataclass.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.40,
        iou_threshold: float = 0.50,
        input_size: int = 640,
        device: str = "cpu",
        ball_model_path: Optional[str | Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_path      : Path to trained YOLOv8 model weights (.pt file)
        confidence      : Minimum confidence score for a detection [0.0 - 1.0]
        iou_threshold   : NMS IoU threshold [0.0 - 1.0]
        input_size      : Image size to pass to YOLO
        device          : Device to run on (e.g. "cpu", "cuda:0")
        ball_model_path : Optional path to a secondary model specifically for the ball.
                          (PRD mentions separate ball-specific YOLO model in Section 18 mitigations).
        """
        self.model_path = Path(model_path)
        self.confidence = max(0.0, min(1.0, confidence))
        self.iou_threshold = max(0.0, min(1.0, iou_threshold))
        self.input_size = input_size
        self.device = device
        
        self.ball_model_path = Path(ball_model_path) if ball_model_path else None

        # Try to load models. In tests, we might want to mock this or skip.
        try:
            # Check if model exists, if not, we can load a base model for dev
            if not self.model_path.exists():
                logging.warning(f"Model path {self.model_path} not found. Attempting to load 'yolov8n.pt' for dev fallback.")
                self.model = YOLO("yolov8n.pt") # Base COCO model fallback
            else:
                self.model = YOLO(str(self.model_path))
                
            self.model.to(self.device)
            
            self.ball_model = None
            if self.ball_model_path and self.ball_model_path.exists():
                self.ball_model = YOLO(str(self.ball_model_path))
                self.ball_model.to(self.device)
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.model = None
            self.ball_model = None

    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        """
        Run inference on a single frame and return a list of Detections.

        Parameters
        ----------
        frame    : BGR image array (from OpenCV)
        frame_id : The 1-indexed ID of this frame (to tag the Detections)

        Returns
        -------
        List of data_models.Detection
        """
        if self.model is None:
            # Fallback for when model fails to load (e.g., in some testing environments)
            return []

        # Run primary detection
        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        
        detections: List[Detection] = []
        
        if not results or len(results) == 0:
            return detections
            
        # Parse main model results
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # box.xyxy is a tensor of shape [1, 4]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Map COCO classes if using a generic YOLO model (fallback)
                if class_id == 32: # COCO sports ball
                    class_id = 3
                # COCO person is 0, which correctly maps to CLASS_PLAYER (0)
                
                # Check if this class is supported by our pipeline (0..3)
                if class_id in CLASS_ID_TO_NAME:
                    det = Detection.from_yolo(
                        frame_id=frame_id,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                        class_id=class_id,
                    )
                    detections.append(det)

        # Optional: Secondary pass for ball detection if configured
        if self.ball_model is not None:
            ball_results = self.ball_model(
                frame,
                imgsz=max(1280, self.input_size), # Often higher res for small ball
                conf=self.confidence, # Or specialized ball conf
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            if ball_results and len(ball_results) > 0:
                ball_boxes = ball_results[0].boxes
                if ball_boxes is not None and len(ball_boxes) > 0:
                    for box in ball_boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # In ball_model, if it only predicts ball, it might output class 0.
                        # We force it to CLASS_BALL (3) based on PRD.
                        det = Detection.from_yolo(
                            frame_id=frame_id,
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=conf,
                            class_id=3, # Hardcode ball class
                        )
                        detections.append(det)

        return detections
