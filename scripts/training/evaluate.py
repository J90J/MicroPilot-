"""
Evaluate MiniMind-O (base or LoRA) on the labeled dataset.
Prints precision, recall, F1 per class and overall accuracy.

Usage:
    # Baseline (no LoRA):
    python scripts/training/evaluate.py --model_path models/minimind-o

    # After fine-tuning:
    python scripts/training/evaluate.py --model_path models/minimind-o --lora_path models/minimind-o-lora
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import base64
import io
from tqdm import tqdm


LABELS = ["red_light", "green_light", "yellow_light", "no_sign"]


def decode_prediction(output_text: str) -> str:
    text = output_text.lower()
    if "red" in text:
        return "red_light"
    if "green" in text:
        return "green_light"
    if "yellow" in text:
        return "yellow_light"
    if "speed limit" in text or "mph" in text:
        for token in text.split():
            if token.isdigit():
                return f"speed_limit_{token}"
    return "no_sign"


def evaluate(args):
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"Loaded LoRA adapter from {args.lora_path}")

    model.eval()

    with open(args.dataset) as f:
        records = json.load(f)

    correct = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for rec in tqdm(records, desc="Evaluating"):
        prompt = f"<image>\nUser: {rec['prompt']}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        predicted = decode_prediction(decoded.split("Assistant:")[-1])
        ground_truth = rec["answer"].replace("speed limit ", "speed_limit_").replace(" ", "_")

        if predicted == ground_truth:
            correct += 1
            per_class[ground_truth]["tp"] += 1
        else:
            per_class[ground_truth]["fn"] += 1
            per_class[predicted]["fp"] += 1

    accuracy = correct / len(records)
    print(f"\nOverall accuracy: {accuracy:.3f} ({correct}/{len(records)})\n")

    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 54)
    for cls, counts in sorted(per_class.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{cls:<20} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/labeled/dataset.json")
    parser.add_argument("--model_path", default="models/minimind-o")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    args = parser.parse_args()
    evaluate(args)
