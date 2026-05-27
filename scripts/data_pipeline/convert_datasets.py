"""
Convert GTSRB and S2TLD into MicroPilot's unified label format (data/labeled/labels.csv).

GTSRB class IDs for speed limits: 1=20, 2=30, 3=50, 4=60, 5=70, 6=80, 8=100, 9=120
S2TLD classes: 0=red, 1=yellow, 2=green, 3=off, 4=wait_on

Usage:
    python scripts/data_pipeline/convert_datasets.py --source gtsrb --input data/gtsrb
    python scripts/data_pipeline/convert_datasets.py --source s2tld --input data/s2tld
"""

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


GTSRB_SPEED_LIMIT_CLASSES = {
    1: "speed_limit_20",
    2: "speed_limit_30",
    3: "speed_limit_50",
    4: "speed_limit_60",
    5: "speed_limit_70",
    6: "speed_limit_80",
    8: "speed_limit_100",
    9: "speed_limit_120",
}

S2TLD_CLASS_MAP = {
    "0": "red_light",
    "1": "yellow_light",
    "2": "green_light",
    "3": "no_sign",
    "4": "no_sign",
}


def convert_gtsrb(input_dir: Path, output_frames: Path):
    output_frames.mkdir(parents=True, exist_ok=True)
    rows = []

    for cls_id, label in GTSRB_SPEED_LIMIT_CLASSES.items():
        # Try both zero-padded and plain integer folder names, Train and train
        candidates = [
            input_dir / f"{cls_id:05d}",
            input_dir / str(cls_id),
            input_dir / "Train" / f"{cls_id:05d}",
            input_dir / "Train" / str(cls_id),
            input_dir / "train" / f"{cls_id:05d}",
            input_dir / "train" / str(cls_id),
        ]
        cls_dir = next((c for c in candidates if c.exists()), None)
        if cls_dir is None:
            print(f"  Skipping class {cls_id} — directory not found")
            continue

        images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
        for img_path in images:
            dest = output_frames / f"gtsrb_{cls_id}_{img_path.name}"
            shutil.copy2(img_path, dest)
            rows.append({"filename": dest.name, "label": label})

    print(f"GTSRB: {len(rows)} speed limit samples")
    return rows


def convert_s2tld(input_dir: Path, output_frames: Path):
    output_frames.mkdir(parents=True, exist_ok=True)
    rows = []

    # S2TLD uses Pascal VOC XML annotations
    ann_dir = input_dir / "Annotations"
    img_dir = input_dir / "JPEGImages"
    if not ann_dir.exists():
        # try nested
        for sub in input_dir.rglob("Annotations"):
            ann_dir = sub
            img_dir = sub.parent / "JPEGImages"
            break

    for xml_path in sorted(ann_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_file = root.findtext("filename")
        img_path = img_dir / img_file
        if not img_path.exists():
            continue

        # Take the dominant label from all objects in this frame
        labels_in_frame = []
        for obj in root.findall("object"):
            cls = obj.findtext("name", "3")
            labels_in_frame.append(S2TLD_CLASS_MAP.get(cls, "no_sign"))

        if not labels_in_frame:
            label = "no_sign"
        else:
            # Prefer signal states over no_sign
            for preferred in ("red_light", "green_light", "yellow_light"):
                if preferred in labels_in_frame:
                    label = preferred
                    break
            else:
                label = labels_in_frame[0]

        dest = output_frames / f"s2tld_{xml_path.stem}{img_path.suffix}"
        shutil.copy2(img_path, dest)
        rows.append({"filename": dest.name, "label": label})

    print(f"S2TLD: {len(rows)} traffic light samples")
    return rows


def write_csv(rows: list, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_csv.exists():
        with open(output_csv) as f:
            existing = list(csv.DictReader(f))
        print(f"Merging with {len(existing)} existing labels")

    all_rows = existing + rows
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"labels.csv: {len(all_rows)} total samples → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gtsrb", "s2tld"], required=True)
    parser.add_argument("--input", required=True, help="Path to downloaded dataset directory")
    parser.add_argument("--frames_out", default="data/frames")
    parser.add_argument("--csv_out", default="data/labeled/labels.csv")
    args = parser.parse_args()

    frames_out = Path(args.frames_out)
    csv_out = Path(args.csv_out)

    if args.source == "gtsrb":
        rows = convert_gtsrb(Path(args.input), frames_out)
    else:
        rows = convert_s2tld(Path(args.input), frames_out)

    write_csv(rows, csv_out)
