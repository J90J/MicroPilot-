"""
MicroPilot end-to-end inference pipeline.
Reads a video file (or live webcam), runs YOLO → MiniMind-O, and announces detections via TTS.

Usage:
    # On a video file:
    python scripts/inference/pipeline.py --input data/raw/dashcam.mp4

    # Live webcam (index 0):
    python scripts/inference/pipeline.py --input 0 --live

    # Skip YOLO pre-filter (send full frame to MiniMind-O):
    python scripts/inference/pipeline.py --input data/raw/dashcam.mp4 --no-yolo
"""

import argparse
import subprocess
import time
import cv2
import numpy as np
from pathlib import Path

from scripts.inference.detector import YOLODetector
from scripts.inference.classifier import MiniMindClassifier


# Minimum seconds between repeated announcements of the same label
ANNOUNCE_COOLDOWN = 5.0


def announce(text: str):
    """Non-blocking macOS TTS."""
    subprocess.Popen(["say", text])


class MicroPilotPipeline:
    def __init__(self, model_path: str, lora_path: str = None, use_yolo: bool = True):
        self.classifier = MiniMindClassifier(model_path, lora_path)
        self.detector = YOLODetector() if use_yolo else None
        self._last_announced: dict[str, float] = {}

    def _maybe_announce(self, label: str):
        now = time.time()
        if now - self._last_announced.get(label, 0) >= ANNOUNCE_COOLDOWN:
            announce(label)
            self._last_announced[label] = now
            print(f"[ANNOUNCED] {label}")

    def process_frame(self, frame: np.ndarray) -> list[str]:
        results = []

        if self.detector:
            rois = self.detector.detect_rois(frame)
            if not rois:
                return results
            crops = [roi["crop"] for roi in rois]
        else:
            crops = [cv2.resize(frame, (224, 224))]

        for crop in crops:
            label = self.classifier.classify(crop)
            if label and label.lower() not in ("none", "no sign", ""):
                results.append(label)
                self._maybe_announce(label)

        return results

    def run(self, source, target_fps: float = 1.0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(source_fps / target_fps)))

        print(f"Pipeline running at {target_fps} FPS (every {frame_interval} source frames)")
        frame_idx = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                detections = self.process_frame(frame)
                elapsed = time.time() - t_start
                print(f"[{elapsed:6.1f}s] frame {frame_idx:6d} → {detections or 'nothing detected'}")

            frame_idx += 1

        cap.release()
        print("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Video path or webcam index (0)")
    parser.add_argument("--model_path", default="models/minimind-o")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO pre-filter")
    parser.add_argument("--live", action="store_true", help="Source is a live camera index")
    args = parser.parse_args()

    source = int(args.input) if args.live else args.input

    pipeline = MicroPilotPipeline(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_yolo=not args.no_yolo,
    )
    pipeline.run(source, target_fps=args.fps)
