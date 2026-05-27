"""
LoRA fine-tuning for MiniMind-O on traffic sign classification.
Loads images directly from disk — no large JSON needed.

Usage:
    python scripts/training/train_lora.py
    python scripts/training/train_lora.py --epochs 5 --lora_r 16
"""

import argparse
import csv
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from tqdm import tqdm


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_dtype(device: torch.device) -> torch.dtype:
    # bfloat16 matmul is unreliable on MPS; float32 is safe everywhere
    return torch.bfloat16 if device.type == "cuda" else torch.float32


class TrafficDataset(Dataset):
    def __init__(self, labels_csv: str, frames_dir: str, tokenizer, vision_processor, max_length: int = 256, train_limit: int = None):
        self.frames_dir = Path(frames_dir)
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.max_length = max_length

        with open(labels_csv) as f:
            all_rows = list(csv.DictReader(f))

        # Only keep rows where the image file exists
        all_valid = [r for r in all_rows if (self.frames_dir / r["filename"]).exists()]

        # Optional subsample for faster training runs
        if train_limit and train_limit < len(all_valid):
            import random; random.seed(42)
            all_valid = random.sample(all_valid, train_limit)

        self.records = all_valid
        print(f"Dataset: {len(self.records)} samples (from {len(all_rows)} total)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = rec["label"].strip()
        image = Image.open(self.frames_dir / rec["filename"]).convert("RGB").resize((256, 256))

        image_token = self.tokenizer.special_tokens_map.get("image_token", "<|image_pad|>")
        prompt_content = f"What traffic signal or sign is visible?\n\n{image_token * 64}"
        messages = [{"role": "user", "content": prompt_content},
                    {"role": "assistant", "content": label}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        pixel_values = self.vision_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "pixel_values": pixel_values,
            "label": label,
        }


def train(args):
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="micropilot", config=vars(args))

    device = _best_device()
    print(f"Training on: {device}")
    if device.type == "cpu":
        print("[WARNING] Training on CPU will be slow (~hours). MPS or CUDA recommended.")

    from transformers import SiglipVisionModel, SiglipImageProcessor

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    vision_processor = SiglipImageProcessor.from_pretrained(args.siglip_path)

    dataset = TrafficDataset(args.labels_csv, args.frames_dir, tok, vision_processor, train_limit=args.train_limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    dtype = _model_dtype(device)
    print(f"Model dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=dtype
    )
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[WARNING] {device} failed ({e}), falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)

    vision_encoder = SiglipVisionModel.from_pretrained(args.siglip_path).eval().to(device=device, dtype=dtype)
    object.__setattr__(model, "vision_encoder", vision_encoder)
    object.__setattr__(model, "vision_processor", vision_processor)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, len(loader) // 10),
        num_training_steps=len(loader) * args.epochs,
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)

            # Custom model has no labels arg — compute cross-entropy from logits manually
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            logits = outputs.logits  # (B, T, vocab)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = input_ids[:, 1:].contiguous().view(-1)
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=pad_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            wandb.log({"train/loss": loss.item()})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/avg_loss": avg_loss})

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"LoRA adapter saved to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="models/minimind-3o")
    parser.add_argument("--siglip_path", default="models/siglip2")
    parser.add_argument("--labels_csv", default="data/labeled/labels.csv")
    parser.add_argument("--frames_dir", default="data/frames")
    parser.add_argument("--output", default="models/minimind-o-lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--train_limit", type=int, default=None, help="Subsample N training examples (faster runs)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    train(args)
