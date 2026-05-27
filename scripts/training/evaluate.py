"""
Evaluate MiniMind-O (base or LoRA) on the labeled dataset.
Prints precision, recall, F1 per class and overall accuracy.

Usage:
    # Baseline (no LoRA):
    python scripts/training/evaluate.py --model_id jingyaogong/minimind-3o

    # After fine-tuning:
    python scripts/training/evaluate.py --lora_path models/minimind-o-lora
"""

import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import base64
import io
from tqdm import tqdm


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode_prediction(text: str) -> str:
    t = text.lower()
    if "red" in t:
        return "red_light"
    if "green" in t:
        return "green_light"
    if "yellow" in t:
        return "yellow_light"
    if "speed" in t or "mph" in t or "limit" in t:
        for token in t.split():
            if token.isdigit():
                return f"speed_limit_{token}"
        return "speed_limit_unknown"
    return "no_sign"


def evaluate(args):
    device = _best_device()
    print(f"Evaluating on: {device}")

    from transformers import SiglipVisionModel, SiglipImageProcessor

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    try:
        model = model.to(device)
    except Exception:
        device = torch.device("cpu")
        model = model.to(device)

    # Attach SigLIP2 vision encoder
    vision_encoder = SiglipVisionModel.from_pretrained(args.siglip_path).eval().to(device)
    vision_processor = SiglipImageProcessor.from_pretrained(args.siglip_path)
    object.__setattr__(model, "vision_encoder", vision_encoder)
    object.__setattr__(model, "vision_processor", vision_processor)

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"LoRA adapter: {args.lora_path}")

    model.eval()

    with open(args.dataset) as f:
        records = json.load(f)

    correct = 0
    per_class: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for rec in tqdm(records, desc="Evaluating"):
        img_bytes = base64.b64decode(rec["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))

        pixel_values = {k: v.to(device) for k, v in vision_processor(images=image, return_tensors="pt").items()}

        prompt = "<image>\nUser: What traffic signal or sign is visible?\nAssistant:"
        # Only pass input_ids — attention_mask must NOT go to this model's custom generate()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # stream=True: generator yields (text_tokens, audio_frame); last yield has text_tokens=None
        # collect last non-None text output
        out_ids = None
        with torch.no_grad():
            for text_tokens, _ in model.generate(input_ids, max_new_tokens=32, temperature=0.1,
                                                  pixel_values=pixel_values, stream=True):
                if text_tokens is not None:
                    out_ids = text_tokens
        if out_ids is None:
            continue
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        predicted = decode_prediction(decoded.split("Assistant:")[-1])

        gt = rec["answer"].replace("speed limit ", "speed_limit_").replace(" ", "_")

        if predicted == gt:
            correct += 1
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            per_class[predicted]["fp"] += 1

    accuracy = correct / len(records)
    print(f"\nOverall accuracy: {accuracy:.3f} ({correct}/{len(records)})\n")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 59)
    for cls, c in sorted(per_class.items()):
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{cls:<25} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/labeled/dataset.json")
    parser.add_argument("--model_id", default="models/minimind-3o")
    parser.add_argument("--siglip_path", default="models/siglip2")
    parser.add_argument("--lora_path", default=None)
    args = parser.parse_args()
    evaluate(args)
