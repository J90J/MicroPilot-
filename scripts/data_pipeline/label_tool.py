"""
Fast keyboard-driven labeling tool for dashcam frames.
Lets you step through frames and press a key to assign a label.

Controls:
    r  → red_light
    g  → green_light
    y  → yellow_light
    0-9 → speed_limit_X0  (e.g. 2 = speed_limit_20, 3 = speed_limit_30)
    n  → no_sign
    s  → skip (do not label)
    b  → back one frame
    q  → quit and save

Usage:
    python scripts/data_pipeline/label_tool.py --frames data/frames --output data/labeled/labels.csv
"""

import argparse
import csv
import sys
from pathlib import Path
import cv2


SPEED_KEYS = {ord(str(i)): f"speed_limit_{i * 10}" for i in range(1, 10)}
LABEL_KEYS = {
    ord("r"): "red_light",
    ord("g"): "green_light",
    ord("y"): "yellow_light",
    ord("n"): "no_sign",
    **SPEED_KEYS,
}

LABEL_COLOR = {
    "red_light": (0, 0, 220),
    "green_light": (0, 200, 0),
    "yellow_light": (0, 200, 220),
    "no_sign": (160, 160, 160),
}


def load_existing(csv_path: Path) -> dict:
    labeled = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                labeled[row["filename"]] = row["label"]
    return labeled


def save_labels(csv_path: Path, labeled: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for fname, label in sorted(labeled.items()):
            writer.writerow({"filename": fname, "label": label})
    print(f"\nSaved {len(labeled)} labels → {csv_path}")


def run(frames_dir: Path, output_csv: Path):
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"No images found in {frames_dir}")
        sys.exit(1)

    labeled = load_existing(output_csv)
    # Start from first unlabeled frame
    i = 0
    for j, f in enumerate(frames):
        if f.name not in labeled:
            i = j
            break

    print(f"Loaded {len(labeled)} existing labels. Starting at frame {i + 1}/{len(frames)}.")
    print("Keys: r=red  g=green  y=yellow  1-9=speed_limit_X0  n=no_sign  s=skip  b=back  q=quit")

    cv2.namedWindow("MicroPilot Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MicroPilot Labeler", 960, 540)

    while 0 <= i < len(frames):
        frame_path = frames[i]
        img = cv2.imread(str(frame_path))
        if img is None:
            i += 1
            continue

        display = img.copy()
        existing = labeled.get(frame_path.name, "")

        # Overlay: progress + current label
        progress = f"{i + 1}/{len(frames)}  |  labeled: {len(labeled)}"
        cv2.putText(display, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, frame_path.name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if existing:
            color = LABEL_COLOR.get(existing, (255, 255, 0))
            cv2.putText(display, f"Label: {existing}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("MicroPilot Labeler", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("b"):
            i = max(0, i - 1)
        elif key == ord("s"):
            i += 1
        elif key in LABEL_KEYS:
            label = LABEL_KEYS[key]
            labeled[frame_path.name] = label
            color = LABEL_COLOR.get(label, (255, 255, 0))
            cv2.putText(display, f"  → {label}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("MicroPilot Labeler", display)
            cv2.waitKey(150)
            i += 1

    cv2.destroyAllWindows()
    save_labels(output_csv, labeled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/frames", help="Directory of extracted frames")
    parser.add_argument("--output", default="data/labeled/labels.csv")
    args = parser.parse_args()
    run(Path(args.frames), Path(args.output))
