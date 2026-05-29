"""
Convert YOLO-format US speed limit dataset to MicroPilot labels.csv format.

Reads data/speed/images/ + data/speed/labels/, crops each bounding box,
saves crops to data/frames/, and appends rows to data/labeled/labels.csv.

Usage:
    python scripts/data_pipeline/convert_speed_yolo.py
    python scripts/data_pipeline/convert_speed_yolo.py --dry_run
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

SPEED_DIR = Path("data/speed")
IMAGES_DIR = SPEED_DIR / "images"
LABELS_DIR = SPEED_DIR / "labels"
FRAMES_DIR = Path("data/frames")
LABELS_CSV = Path("data/labeled/labels.csv")

# Class ID → speed_limit_XX label (matches speed_limits.yaml)
CLASS_TO_LABEL = {
    0: "speed_limit_10",
    1: "speed_limit_15",
    2: "speed_limit_20",
    3: "speed_limit_25",
    4: "speed_limit_30",
    5: "speed_limit_35",
    6: "speed_limit_40",
    7: "speed_limit_45",
    8: "speed_limit_50",
    9: "speed_limit_55",
    10: "speed_limit_60",
    11: "speed_limit_65",
    12: "speed_limit_70",
    13: "speed_limit_75",
}


def crop_box(img: np.ndarray, cx: float, cy: float, w: float, h: float, pad: float = 0.1) -> np.ndarray:
    H, W = img.shape[:2]
    # Add small padding around the sign
    w_pad = w * (1 + pad)
    h_pad = h * (1 + pad)
    x1 = int(max(0, (cx - w_pad / 2) * W))
    y1 = int(max(0, (cy - h_pad / 2) * H))
    x2 = int(min(W, (cx + w_pad / 2) * W))
    y2 = int(min(H, (cy + h_pad / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def main(dry_run: bool = False):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing CSV filenames to avoid duplicates
    existing = set()
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["filename"])

    image_files = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    new_rows = []
    skipped = 0
    errors = 0

    for img_path in image_files:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            label = CLASS_TO_LABEL.get(class_id)
            if label is None:
                continue

            crop = crop_box(img, cx, cy, w, h)
            if crop is None:
                continue

            out_name = f"speed_{img_path.stem}_{idx}.jpg"
            if out_name in existing:
                skipped += 1
                continue

            out_path = FRAMES_DIR / out_name
            if not dry_run:
                cv2.imwrite(str(out_path), crop)
            new_rows.append({"filename": out_name, "label": label})

    if not dry_run and new_rows:
        with open(LABELS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            for row in new_rows:
                writer.writerow(row)

    print(f"{'[DRY RUN] ' if dry_run else ''}Converted {len(new_rows)} crops, skipped {skipped}, errors {errors}")
    if new_rows:
        label_counts: dict[str, int] = {}
        for row in new_rows:
            label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
        for lbl, cnt in sorted(label_counts.items()):
            print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Count crops without writing files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
