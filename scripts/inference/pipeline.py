"""
MicroPilot demo pipeline.

Reads a video file, runs YOLO → MiniMind-O, and announces detections via macOS TTS.
This script is for DEMO purposes only — video files are not used for training or evaluation.

Usage:
    python scripts/inference/pipeline.py --input data/raw/demo.mp4
    python scripts/inference/pipeline.py --input data/raw/demo.mp4 --quantize
    python scripts/inference/pipeline.py --input 0 --live   # webcam

Performance flags:
    --quantize     INT8 dynamic quantization (~2x memory reduction, faster on CPU/MPS)
    --batch_size N Continuous batching: accumulate N ROIs before calling classifier
                   (amortizes model load overhead across multiple detections per frame)
    --no_yolo      Skip YOLO pre-filter (send full frame directly to MiniMind-O)

Note on other optimizations:
    PagedAttention: not applicable — KV cache is trivially small at 0.1B / 32 output tokens
    Speculative decoding: not applicable — 32-token output too short to amortize draft overhead
"""

import argparse
import subprocess
import time
import cv2
import numpy as np
from pathlib import Path

from scripts.inference.detector import YOLODetector
from scripts.inference.classifier import MiniMindClassifier


ANNOUNCE_COOLDOWN = 5.0  # seconds between repeated announcements of the same label


def announce(text: str):
    subprocess.Popen(["say", text])


class MicroPilotPipeline:
    def __init__(
        self,
        model_path: str,
        siglip_path: str = "models/siglip2",
        lora_path: str = None,
        use_yolo: bool = True,
        quantize: bool = False,
        batch_size: int = 1,
    ):
        self.classifier = MiniMindClassifier(
            model_path=model_path,
            siglip_path=siglip_path,
            lora_path=lora_path,
            quantize=quantize,
        )
        self.detector = YOLODetector() if use_yolo else None
        self.batch_size = batch_size
        self._last_announced: dict[str, float] = {}

    def _maybe_announce(self, label: str):
        now = time.time()
        if now - self._last_announced.get(label, 0) >= ANNOUNCE_COOLDOWN:
            announce(label)
            self._last_announced[label] = now
            print(f"  [ANNOUNCED] {label}")

    def process_frame(self, frame: np.ndarray) -> list[str]:
        if self.detector:
            rois = self.detector.detect_rois(frame)
            if not rois:
                return []
            crops = [roi["crop"] for roi in rois]
        else:
            crops = [cv2.resize(frame, (224, 224))]

        # Continuous batching: process in groups of batch_size
        all_labels = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i : i + self.batch_size]
            labels = self.classifier.classify_batch(batch)
            all_labels.extend(labels)

        results = []
        for label in all_labels:
            if label and label.lower() not in ("none", "no sign", ""):
                results.append(label)
                self._maybe_announce(label)

        return results

    def run(self, source, target_fps: float = 1.0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(source_fps / target_fps)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nSource: {source}")
        print(f"Processing at {target_fps} FPS (every {frame_interval} frames, ~{total // frame_interval} total)\n")

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
        elapsed = time.time() - t_start
        print(f"\nDone. Processed {frame_idx} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroPilot demo — video inference only, not for training/eval")
    parser.add_argument("--input", required=True, help="Video path or webcam index")
    parser.add_argument("--model_path", default="models/minimind-3o")
    parser.add_argument("--siglip_path", default="models/siglip2")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--quantize", action="store_true", help="INT8 dynamic quantization")
    parser.add_argument("--batch_size", type=int, default=1, help="Continuous batching size for ROIs")
    parser.add_argument("--no_yolo", action="store_true", help="Disable YOLO pre-filter")
    parser.add_argument("--live", action="store_true", help="Source is a webcam index")
    args = parser.parse_args()

    source = int(args.input) if args.live else args.input

    pipeline = MicroPilotPipeline(
        model_path=args.model_path,
        siglip_path=args.siglip_path,
        lora_path=args.lora_path,
        use_yolo=not args.no_yolo,
        quantize=args.quantize,
        batch_size=args.batch_size,
    )
    pipeline.run(source, target_fps=args.fps)
