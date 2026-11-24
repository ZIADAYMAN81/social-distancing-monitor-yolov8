"""YOLOv8 person detector."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .geometry import Person

PERSON_CLASS_ID = 0  # "person" in the COCO label set used by YOLOv8


class PersonDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model that returns only people."""

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.5, device: Optional[str] = None):
        from ultralytics import YOLO  # imported lazily so the geometry tests need no model

        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device

    def __call__(self, frame: np.ndarray) -> List[Person]:
        # Filter to people after prediction (Ultralytics' built-in class filter is avoided because it
        # breaks non-maximum suppression with some PyTorch releases).
        result = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        keep = result.boxes.cls.cpu().numpy().astype(int) == PERSON_CLASS_ID
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)[keep]
        scores = result.boxes.conf.cpu().numpy()[keep]
        return [Person(int(x1), int(y1), int(x2), int(y2), float(s)) for (x1, y1, x2, y2), s in zip(boxes, scores)]
