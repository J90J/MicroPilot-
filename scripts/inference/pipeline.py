"""
MicroPilot demo pipeline.

Reads a video file, runs YOLO → MiniMind-O, displays the video with red bounding
boxes around detected signs, and announces detections via macOS TTS.
This script is for DEMO purposes only — video files are not used for training or evaluation.

Usage:
    python -m scripts.inference.pipeline --input "Demo tesla footage/clip.mp4" --lora_path models/minimind-o-lora
    python -m scripts.inference.pipeline --input 0 --live   # webcam

Controls (when --show is active):
    Q  quit

Performance flags:
    --quantize     INT8 dynamic quantization (~2x memory reduction, faster on CPU/MPS)
    --batch_size N Continuous batching: accumulate N ROIs before calling classifier
    --no_yolo      Skip YOLO pre-filter (send full frame directly to MiniMind-O)
    --no_show      Disable video display (terminal-only output)
"""

import argparse
import csv
import hashlib
import subprocess
import time
import cv2
import numpy as np
from pathlib import Path

from scripts.inference.detector import YOLODetector
from scripts.inference.classifier import MiniMindClassifier

COLLECT_FRAMES_DIR = Path("data/frames")
COLLECT_LABELS_CSV = Path("data/labeled/labels.csv")
COLLECT_MIN_STREAK = 3   # consecutive intervals with same label = confident


DISPLAY_MAX_WIDTH = 1280  # downscale wide frames for display
ABSENT_INTERVALS_TO_RESET = 3  # intervals a label must be gone before re-announcing
BOX_COLOR = (0, 0, 255)   # red (BGR)
BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICKNESS = 2

IGNORE_LABELS = {"none", "no sign", "no_sign", ""}

# Whitelist of valid labels — anything else from MiniMind-O is a garbage output
VALID_LABELS = {
    "stop_sign",
    *(f"speed_limit_{v}" for v in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]),
}


TTS_VOICE = "Daniel"  # British English — closest built-in to Attenborough

def _announce(text: str):
    subprocess.Popen(["say", "-v", TTS_VOICE, text])


def _draw_detection(frame: np.ndarray, bbox: tuple, label: str):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    if label == "stop_sign":
        display_text = "STOP"
    elif label.startswith("speed_limit_"):
        num = label.replace("speed_limit_", "")
        display_text = f"{num} MPH"
    else:
        display_text = label.replace("_", " ").upper()

    (tw, th), baseline = cv2.getTextSize(display_text, FONT, FONT_SCALE, FONT_THICKNESS)
    tag_y1 = max(0, y1 - th - baseline - 6)
    tag_y2 = y1
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), BOX_COLOR, -1)
    cv2.putText(frame, display_text, (x1 + 3, tag_y2 - baseline),
                FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


class MicroPilotPipeline:
    def __init__(
        self,
        model_path: str,
        siglip_path: str = "models/siglip2",
        lora_path: str = None,
        use_yolo: bool = True,
        yolo_model: str = "yolov8n.pt",
        quantize: bool = False,
        batch_size: int = 1,
    ):
        self.classifier = MiniMindClassifier(
            model_path=model_path,
            siglip_path=siglip_path,
            lora_path=lora_path,
            quantize=quantize,
        )
        if use_yolo:
            self.detector = YOLODetector(
                speed_model=yolo_model,
                coco_model="yolov8n.pt",
            )
        else:
            self.detector = None
        self.batch_size = batch_size
        self.collect = False  # enabled via run(..., collect=True)
        # Rising-edge tracker: announce once when label first appears,
        # re-arm only after it has been absent for ABSENT_INTERVALS_TO_RESET intervals.
        self._active_labels: dict[str, int] = {}  # label → consecutive absent intervals
        # Active learning streak tracker: label → (streak_count, last_crop)
        self._streak: dict[str, list] = {}
        self._saved_hashes: set[str] = set()

    def _maybe_announce(self, label: str):
        if label not in self._active_labels:
            if label == "stop_sign":
                speech = "stop"
            elif label.startswith("speed_limit_"):
                speech = label.replace("speed_limit_", "")
            else:
                speech = label.replace("_", " ")
            _announce(speech)
            print(f"  [ANNOUNCED] {label}")
        self._active_labels[label] = 0  # reset absent counter

    def _age_labels(self, seen: set[str]):
        """Age out labels not seen this interval; re-arm after enough absences."""
        for lbl in list(self._active_labels):
            if lbl not in seen:
                self._active_labels[lbl] += 1
                if self._active_labels[lbl] >= ABSENT_INTERVALS_TO_RESET:
                    del self._active_labels[lbl]

    def _update_streak(self, label: str, crop: np.ndarray):
        """Track consecutive detections; save crop once streak reaches threshold."""
        if label not in self._streak:
            self._streak[label] = [0, None]
        self._streak[label][0] += 1
        self._streak[label][1] = crop  # keep most recent crop

        if self._streak[label][0] == COLLECT_MIN_STREAK:
            self._save_crop(label, crop)

    def _reset_streak(self, label: str):
        if label in self._streak:
            del self._streak[label]

    def _save_crop(self, label: str, crop: np.ndarray):
        img_hash = hashlib.md5(crop.tobytes()).hexdigest()[:12]
        if img_hash in self._saved_hashes:
            return
        self._saved_hashes.add(img_hash)

        COLLECT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"dashcam_{label}_{img_hash}.jpg"
        cv2.imwrite(str(COLLECT_FRAMES_DIR / fname), crop)

        with open(COLLECT_LABELS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([fname, label])

        print(f"  [COLLECTED] {fname}")

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Returns list of {label, bbox} for real signs only (no_sign filtered out)."""
        if self.detector:
            rois = self.detector.detect_rois(frame)
            if not rois:
                return []
        else:
            h, w = frame.shape[:2]
            rois = [{"crop": cv2.resize(frame, (224, 224)),
                     "bbox": (0, 0, w, h), "kind": "speed_sign"}]

        detections = []

        # Stop signs: announce directly from YOLO — no classifier needed
        stop_rois = [r for r in rois if r.get("kind") == "stop_sign"]
        for roi in stop_rois:
            detections.append({"label": "stop_sign", "bbox": roi["bbox"]})
            self._maybe_announce("stop_sign")

        # Speed signs: pass crops to MiniMind-O
        speed_rois = [r for r in rois if r.get("kind") != "stop_sign"]
        crops = [r["crop"] for r in speed_rois]
        bboxes = [r["bbox"] for r in speed_rois]

        all_labels = []
        for i in range(0, len(crops), self.batch_size):
            batch_labels = self.classifier.classify_batch(crops[i : i + self.batch_size])
            all_labels.extend(batch_labels)

        active_speed_labels = set()
        for label, bbox, roi in zip(all_labels, bboxes, speed_rois):
            clean = label.lower().strip()
            if clean in IGNORE_LABELS or clean not in VALID_LABELS:
                self._reset_streak(clean)
                continue
            detections.append({"label": clean, "bbox": bbox})
            self._maybe_announce(clean)
            active_speed_labels.add(clean)
            if self.collect:
                self._update_streak(clean, roi["crop"])

        # Reset streaks for speed labels not seen this frame
        if self.collect:
            for lbl in list(self._streak):
                if lbl not in active_speed_labels:
                    self._reset_streak(lbl)

        return detections

    def run(self, source, target_fps: float = 1.0, show: bool = True, collect: bool = False):
        self.collect = collect
        if collect:
            print(f"[COLLECT] Saving confident crops to {COLLECT_FRAMES_DIR} / {COLLECT_LABELS_CSV}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(source_fps / target_fps)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nSource: {source}")
        print(f"Processing at {target_fps} FPS (every {frame_interval} frames, ~{total // frame_interval} to classify)")
        if show:
            print("Press Q in the video window to quit.\n")

        frame_idx = 0
        t_start = time.time()
        last_detections: list[dict] = []  # persist boxes across skipped frames
        stale_counter = 0                  # frames since last classification

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                detections = self.process_frame(frame)
                seen = {d["label"] for d in detections}
                self._age_labels(seen)
                elapsed = time.time() - t_start
                labels = [d["label"] for d in detections]
                print(f"[{elapsed:6.1f}s] frame {frame_idx:6d} → {labels or 'nothing detected'}")

                if detections:
                    last_detections = detections
                    stale_counter = 0
                else:
                    stale_counter += 1
                    if stale_counter >= 3:
                        last_detections = []

            if show:
                display = frame.copy()
                for det in last_detections:
                    _draw_detection(display, det["bbox"], det["label"])

                # Downscale if too wide for screen
                h, w = display.shape[:2]
                if w > DISPLAY_MAX_WIDTH:
                    scale = DISPLAY_MAX_WIDTH / w
                    display = cv2.resize(display, (int(w * scale), int(h * scale)))

                cv2.imshow("MicroPilot", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if show:
            cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        print(f"\nDone. Processed {frame_idx} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroPilot demo — video inference only")
    parser.add_argument("--input", required=True, help="Video path or webcam index")
    parser.add_argument("--model_path", default="models/minimind-3o")
    parser.add_argument("--siglip_path", default="models/siglip2")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to classify")
    parser.add_argument("--quantize", action="store_true", help="INT8 dynamic quantization")
    parser.add_argument("--batch_size", type=int, default=4, help="ROI batch size")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO weights (default: COCO; use custom for speed sign detector)")
    parser.add_argument("--no_yolo", action="store_true", help="Disable YOLO pre-filter")
    parser.add_argument("--no_show", action="store_true", help="Disable video display")
    parser.add_argument("--collect", action="store_true", help="Save confident detections as training data")
    parser.add_argument("--live", action="store_true", help="Source is a webcam index")
    args = parser.parse_args()

    source = int(args.input) if args.live else args.input

    pipeline = MicroPilotPipeline(
        model_path=args.model_path,
        siglip_path=args.siglip_path,
        lora_path=args.lora_path,
        use_yolo=not args.no_yolo,
        yolo_model=args.yolo_model,
        quantize=args.quantize,
        batch_size=args.batch_size,
    )
    pipeline.run(source, target_fps=args.fps, show=not args.no_show, collect=args.collect)
