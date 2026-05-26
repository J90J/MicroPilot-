"""
Prepare a HuggingFace-compatible dataset from labeled frames for LoRA fine-tuning.

Expected label file format (data/labeled/labels.csv):
    filename,label
    frame_000100.jpg,red_light
    frame_000200.jpg,green_light
    frame_000300.jpg,speed_limit_25
    ...

Labels: red_light | green_light | yellow_light | speed_limit_XX | no_sign
"""

import argparse
import csv
import json
from pathlib import Path
from PIL import Image
import base64
import io


LABEL_TO_PROMPT = {
    "red_light":     "What traffic signal is visible? Answer: red light",
    "green_light":   "What traffic signal is visible? Answer: green light",
    "yellow_light":  "What traffic signal is visible? Answer: yellow light",
    "no_sign":       "What traffic signal is visible? Answer: none",
}


def speed_limit_prompt(label: str) -> str:
    speed = label.replace("speed_limit_", "")
    return f"What traffic signal is visible? Answer: speed limit {speed}"


def encode_image(path: Path) -> str:
    with Image.open(path).convert("RGB") as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


def build_dataset(frames_dir: str, labels_csv: str, output_path: str):
    frames_dir = Path(frames_dir)
    records = []

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = frames_dir / row["filename"]
            if not img_path.exists():
                print(f"Skipping missing file: {img_path}")
                continue

            label = row["label"].strip()
            if label.startswith("speed_limit_"):
                prompt = speed_limit_prompt(label)
            else:
                prompt = LABEL_TO_PROMPT.get(label)
                if not prompt:
                    print(f"Unknown label '{label}', skipping")
                    continue

            records.append({
                "image": encode_image(img_path),
                "prompt": "What traffic signal is visible?",
                "answer": prompt.split("Answer: ")[1],
            })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Dataset: {len(records)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--labels", default="data/labeled/labels.csv")
    parser.add_argument("--output", default="data/labeled/dataset.json")
    args = parser.parse_args()

    build_dataset(args.frames, args.labels, args.output)
