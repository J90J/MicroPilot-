"""
Evaluate MiniMind-O (base or LoRA) on a held-out 20% split of the labeled dataset.

Produces:
  - Accuracy, per-class precision/recall/F1, macro + weighted F1
  - Confusion matrix saved to results/confusion_matrix_<tag>.png
  - Full results written to results/eval_<tag>.md

The 80/20 split is deterministic (seed=42) and matches what train_lora.py excludes
when --holdout 0.2 is passed. Always evaluate on the same 20% regardless of limit.

Usage:
    # Baseline (no LoRA):
    python scripts/training/evaluate.py --tag baseline

    # After fine-tuning:
    python scripts/training/evaluate.py --lora_path models/minimind-o-lora --tag lora

    # Quick sanity check (random 100 samples from the holdout set):
    python scripts/training/evaluate.py --limit 100 --tag quick
"""

import argparse
import csv
import random
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# ── Label mapping ──────────────────────────────────────────────────────────────

SPEED_LABELS = {str(v): f"speed_limit_{v}" for v in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]}

_ALL_LABELS = set(SPEED_LABELS.values()) | {"no_sign", "red_light", "green_light", "yellow_light"}

def decode_prediction(raw: str) -> str:
    """Map free-text model output to a canonical label."""
    t = raw.lower().strip()

    # Direct match — model was fine-tuned to output canonical labels verbatim
    if t in _ALL_LABELS:
        return t

    # Partial canonical match (e.g. trailing punctuation or whitespace)
    for lbl in _ALL_LABELS:
        if t.startswith(lbl):
            return lbl

    # Fallback: parse free-text descriptions (zero-shot / unexpected outputs)
    t = t.replace("km/h", "kmh").replace("mph", " mph ").replace("mi/h", " mph ")
    t_space = t.replace("_", " ")

    if re.search(r"speed|limit|\d+\s*mph", t_space):
        nums = re.findall(r"(?<!\d)(\d+)(?!\d)", t_space)
        for n in nums:
            if n in SPEED_LABELS:
                return SPEED_LABELS[n]
        return "speed_limit_unknown"

    if "red" in t and re.search(r"light|signal|stop|traffic", t):
        return "red_light"
    if "green" in t and re.search(r"light|signal|go|traffic", t):
        return "green_light"
    if "yellow" in t and re.search(r"light|signal|traffic|caution", t):
        return "yellow_light"

    return "no_sign"


# ── Dataset split ──────────────────────────────────────────────────────────────

def load_holdout(labels_csv: str, frames_dir: Path, holdout: float, seed: int) -> list[dict]:
    """Return the held-out 20% of rows that exist on disk."""
    with open(labels_csv) as f:
        all_rows = [r for r in csv.DictReader(f) if (frames_dir / r["filename"]).exists()]

    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n_holdout = max(1, int(len(all_rows) * holdout))
    return all_rows[-n_holdout:]  # last N = holdout; first (1-N) = training


# ── Model loading ──────────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": prec, "recall": rec, "f1": f1,
                          "support": sum(t == cls for t in y_true)}

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    total = len(y_true)
    weighted_f1 = (sum(v["f1"] * v["support"] for v in per_class.values()) / total) if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "classes": classes,
    }


def save_confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed — skipping confusion matrix plot")
        return

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [c.replace("speed_limit_", "") for c in classes]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("MicroPilot — Confusion Matrix", fontsize=13)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7 if n > 10 else 9, color=color)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Confusion matrix → {out_path}")


def save_results_md(metrics: dict, tag: str, lora_path: str | None,
                    n_total: int, n_eval: int, out_path: Path):
    lines = [
        f"# MicroPilot Evaluation — {tag}",
        "",
        f"**Tag:** `{tag}`  ",
        f"**LoRA adapter:** `{lora_path or 'none (zero-shot baseline)'}`  ",
        f"**Eval samples:** {n_eval} / {n_total} total (held-out 20%, seed=42)  ",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Accuracy | {metrics['accuracy']:.3f} ({int(metrics['accuracy']*n_eval)}/{n_eval}) |",
        f"| Macro F1 | {metrics['macro_f1']:.3f} |",
        f"| Weighted F1 | {metrics['weighted_f1']:.3f} |",
        "",
        "## Per-Class Metrics",
        "",
        f"| Class | Precision | Recall | F1 | Support |",
        f"|---|---|---|---|---|",
    ]

    for cls, v in sorted(metrics["per_class"].items()):
        lines.append(
            f"| {cls} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
        )

    lines += [
        "",
        f"![Confusion Matrix](confusion_matrix_{tag}.png)",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Results → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = _best_device()
    dtype = _model_dtype(device)
    print(f"Evaluating on: {device}, dtype={dtype}")

    from transformers import AutoTokenizer, AutoModelForCausalLM, SiglipVisionModel, SiglipImageProcessor
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=dtype
    )
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[WARN] {device} failed ({e}), using CPU")
        device = torch.device("cpu")
        model = model.to(device)

    vision_encoder = SiglipVisionModel.from_pretrained(args.siglip_path).eval().to(device=device, dtype=dtype)
    vision_processor = SiglipImageProcessor.from_pretrained(args.siglip_path)
    object.__setattr__(model, "vision_encoder", vision_encoder)
    object.__setattr__(model, "vision_processor", vision_processor)

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"LoRA adapter: {args.lora_path}")

    model.eval()

    frames_dir = Path(args.frames_dir)
    with open(args.labels_csv) as f:
        all_records = list(csv.DictReader(f))
    n_total = len(all_records)

    records = load_holdout(args.labels_csv, frames_dir, args.holdout, args.seed)
    if args.limit:
        rng = random.Random(args.seed + 1)
        rng.shuffle(records)
        records = records[:args.limit]

    image_token = tok.special_tokens_map.get("image_token", "<|image_pad|>")
    prompt_content = "What traffic signal or sign is visible?\n\n" + image_token * 64

    y_true, y_pred, raw_preds = [], [], []

    for rec in tqdm(records, desc="Evaluating"):
        img_path = frames_dir / rec["filename"]
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        pixel_values = vision_processor(images=image, return_tensors="pt")["pixel_values"] \
                           .to(device=device, dtype=dtype)

        messages = [{"role": "user", "content": prompt_content}]
        inputs_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = torch.tensor(
            tok(inputs_text).data["input_ids"], dtype=torch.long
        ).unsqueeze(0).to(device)

        out_ids = None
        with torch.no_grad():
            for text_tokens, _ in model.generate(
                input_ids, max_new_tokens=32, temperature=0.1,
                pixel_values=pixel_values, stream=True
            ):
                if text_tokens is not None:
                    out_ids = text_tokens

        if out_ids is None:
            continue

        decoded = tok.decode(out_ids[0], skip_special_tokens=True)
        raw = decoded.split("Assistant:")[-1].strip()
        predicted = decode_prediction(raw)

        y_true.append(rec["label"].strip())
        y_pred.append(predicted)
        raw_preds.append(raw)

    if not y_true:
        print("No predictions — check frames_dir and labels_csv paths.")
        return

    metrics = compute_metrics(y_true, y_pred)

    # Print to terminal
    print(f"\n{'='*60}")
    print(f"  Tag: {args.tag}  |  Samples: {len(y_true)}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}  ({int(metrics['accuracy']*len(y_true))}/{len(y_true)})")
    print(f"  Macro F1:    {metrics['macro_f1']:.3f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"\n  {'Class':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>6}")
    print(f"  {'-'*56}")
    for cls, v in sorted(metrics["per_class"].items()):
        print(f"  {cls:<25} {v['precision']:>7.3f} {v['recall']:>7.3f} {v['f1']:>7.3f} {v['support']:>6}")

    # Sample failures
    failures = [(t, p, r) for t, p, r in zip(y_true, y_pred, raw_preds) if t != p][:5]
    if failures:
        print(f"\n  Sample errors (GT → Pred):")
        for t, p, r in failures:
            print(f"    {t} → {p}  (raw: \"{r[:60]}\")")

    # Save outputs
    results_dir = Path("results")
    save_confusion_matrix(
        y_true, y_pred, metrics["classes"],
        results_dir / f"confusion_matrix_{args.tag}.png",
    )
    save_results_md(
        metrics, args.tag, args.lora_path,
        n_total, len(y_true),
        results_dir / f"eval_{args.tag}.md",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="models/minimind-3o")
    parser.add_argument("--siglip_path", default="models/siglip2")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--labels_csv", default="data/labeled/labels.csv")
    parser.add_argument("--frames_dir", default="data/frames")
    parser.add_argument("--tag", default="eval", help="Label for output files (e.g. baseline, lora)")
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction of data held out for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N samples from holdout set")
    args = parser.parse_args()
    evaluate(args)
