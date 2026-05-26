"""
Stage 1: YOLO-based region-of-interest detector.
Finds bounding boxes likely containing traffic lights or signs,
crops them, and passes to MiniMind-O classifier.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# COCO class IDs relevant to driving
TRAFFIC_LIGHT_CLASS = 9
STOP_SIGN_CLASS = 11


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.model = YOLO(model_name)
        self.conf = conf_threshold
        self.target_classes = {TRAFFIC_LIGHT_CLASS, STOP_SIGN_CLASS}

    def detect_rois(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts: {crop, bbox, class_id, confidence}
        Crops are resized to 224x224 for MiniMind-O input.
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        rois = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Add 20% padding around the detection
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (224, 224))
            rois.append({
                "crop": crop_resized,
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "confidence": float(box.conf[0]),
            })

        return rois
