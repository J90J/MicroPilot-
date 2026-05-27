"""
Quick sanity check: show raw model output for a handful of samples.
Distinguishes genuine zero-shot failure from a decode_prediction parsing issue.

Usage: python scripts/training/sample_predictions.py [--limit 10]
"""

import argparse, csv, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import torch
from PIL import Image
from pathlib import Path


MODEL_PATH = "models/minimind-3o"
SIGLIP_PATH = "models/siglip2"
LABELS_CSV = "data/labeled/labels.csv"
FRAMES_DIR = "data/frames"


def main(limit: int = 10):
    from transformers import AutoTokenizer, AutoModelForCausalLM, SiglipVisionModel, SiglipImageProcessor

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Running on: {device}\n")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    enc = SiglipVisionModel.from_pretrained(SIGLIP_PATH).eval().to(device)
    proc = SiglipImageProcessor.from_pretrained(SIGLIP_PATH)
    object.__setattr__(model, "vision_encoder", enc)
    object.__setattr__(model, "vision_processor", proc)
    model.eval()

    frames_dir = Path(FRAMES_DIR)
    with open(LABELS_CSV) as f:
        records = list(csv.DictReader(f))

    # Sample evenly across the dataset
    step = max(1, len(records) // limit)
    samples = [records[i] for i in range(0, min(len(records), step * limit), step)][:limit]

    print(f"{'Ground Truth':<30} Raw Model Output")
    print("-" * 90)

    for rec in samples:
        img_path = frames_dir / rec["filename"]
        if not img_path.exists():
            print(f"{rec['label']:<30} [image not found: {rec['filename']}]")
            continue

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        pv = {k: v.to(device) for k, v in proc(images=image, return_tensors="pt").items()}
        input_ids = tok(
            "<image>\nUser: What traffic signal or sign is visible?\nAssistant:",
            return_tensors="pt"
        ).input_ids.to(device)

        out_ids = None
        with torch.no_grad():
            for text_tokens, _ in model.generate(
                input_ids, max_new_tokens=32, temperature=0.1, pixel_values=pv, stream=True
            ):
                if text_tokens is not None:
                    out_ids = text_tokens

        raw = tok.decode(out_ids[0], skip_special_tokens=True) if out_ids is not None else "[None]"
        answer = raw.split("Assistant:")[-1].strip()
        print(f"{rec['label']:<30} {repr(answer)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    main(args.limit)
