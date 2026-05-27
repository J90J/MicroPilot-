"""
LoRA fine-tuning for MiniMind-O on traffic sign classification.

Model: jingyaogong/minimind-3o
Weights: https://huggingface.co/jingyaogong/minimind-3o

Usage:
    python scripts/training/train_lora.py --dataset data/labeled/dataset.json

MPS note: training on CPU is slow but works. If you have access to a CUDA GPU,
set CUDA_VISIBLE_DEVICES and it will be used automatically.
"""

import argparse
import json
import torch
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
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


class TrafficDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path) as f:
            self.records = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = f"<image>\nUser: What traffic signal or sign is visible?\nAssistant: {rec['answer']}"
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        img_bytes = base64.b64decode(rec["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "image": image,
        }


def train(args):
    wandb.init(project="micropilot", config=vars(args))

    device = _best_device()
    print(f"Training on: {device}")
    if device.type == "cpu":
        print("[WARNING] Training on CPU will be slow. Consider a machine with CUDA or MPS.")

    model_id = args.model_id
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[WARNING] {device} failed ({e}), falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)

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

    dataset = TrafficDataset(args.dataset, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

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
    tokenizer.save_pretrained(str(out_dir))
    print(f"LoRA adapter saved to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/labeled/dataset.json")
    parser.add_argument("--model_id", default="models/minimind-3o",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", default="models/minimind-o-lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()
    train(args)
