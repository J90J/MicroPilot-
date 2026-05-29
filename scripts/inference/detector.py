"""
Stage 1: Two-model YOLO detector for MicroPilot.

- Speed limit signs: custom YOLOv8 trained on data/speed/ (all 14 classes 10–75 mph)
- Stop signs: COCO YOLOv8n class 11 — very reliable, no custom training needed

Each ROI dict has a 'kind' field:
  'speed_sign'  → pass crop to MiniMind-O for exact speed classification
  'stop_sign'   → announce directly as "Stop sign" (no classifier needed)
"""

import cv2
import numpy as np
from ultralytics import YOLO

COCO_STOP_SIGN_CLASS = 11


class YOLODetector:
    def __init__(
        self,
        speed_model: str = "yolov8n.pt",   # custom-trained speed detector
        coco_model: str = "yolov8n.pt",    # COCO model for stop signs
        conf_threshold: float = 0.30,
    ):
        self.speed_model = YOLO(speed_model)
        # Only load a second model if it's different from the speed model
        self.coco_model = YOLO(coco_model) if coco_model != speed_model else None
        self.conf = conf_threshold
        self._has_custom_speed = speed_model != "yolov8n.pt"

    def _pad_crop(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  pad: float = 0.2) -> tuple:
        pad_x = int((x2 - x1) * pad)
        pad_y = int((y2 - y1) * pad)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)
        return x1, y1, x2, y2

    def detect_rois(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts: {crop, bbox, class_id, confidence, kind}
        kind = 'speed_sign' | 'stop_sign'
        """
        rois = []

        # ── Speed limit signs ──────────────────────────────────────────────
        speed_results = self.speed_model(frame, conf=self.conf, verbose=False)[0]
        for box in speed_results.boxes:
            cls_id = int(box.cls[0])
            # Custom model: all classes are speed limits.
            # COCO fallback: skip (no speed limit class in COCO).
            if not self._has_custom_speed:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = self._pad_crop(frame, x1, y1, x2, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rois.append({
                "crop": cv2.resize(crop, (224, 224)),
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "confidence": float(box.conf[0]),
                "kind": "speed_sign",
            })

        # ── Stop signs (COCO) ──────────────────────────────────────────────
        coco_src = self.coco_model if self.coco_model else self.speed_model
        # Only run a second COCO pass when using a custom speed model
        if self._has_custom_speed and self.coco_model:
            coco_results = coco_src(frame, conf=self.conf, verbose=False)[0]
        elif not self._has_custom_speed:
            coco_results = speed_results  # speed_model IS the COCO model
        else:
            coco_results = None

        if coco_results is not None:
            for box in coco_results.boxes:
                cls_id = int(box.cls[0])
                if cls_id != COCO_STOP_SIGN_CLASS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = self._pad_crop(frame, x1, y1, x2, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                rois.append({
                    "crop": cv2.resize(crop, (224, 224)),
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "confidence": float(box.conf[0]),
                    "kind": "stop_sign",
                })

        return rois
