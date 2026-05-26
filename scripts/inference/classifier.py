"""
Stage 2: MiniMind-O classifier for traffic sign semantics.
Takes a cropped ROI (numpy BGR array) and returns a text prediction.

Model: jingyaogong/minimind-3o (HuggingFace transformers format)
Weights: https://huggingface.co/jingyaogong/minimind-3o

MPS note: MiniMind-O was developed against CUDA. MPS is attempted first;
if ops fail, we fall back to CPU automatically.
"""

import torch
from PIL import Image
import numpy as np


PROMPT = "What traffic signal or sign is visible? Answer in 5 words or fewer."


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MiniMindClassifier:
    def __init__(self, model_id: str = "jingyaogong/minimind-3o", lora_path: str = None):
        self.device = _best_device()
        print(f"MiniMind-O loading on: {self.device}")

        # Lazy import so the rest of the repo works before weights are downloaded
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # MPS can fail on unsupported ops — fall back to CPU if move raises
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"[WARNING] Could not move model to {self.device}: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)

        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"LoRA adapter loaded from {lora_path}")

        self.model.eval()

    def classify(self, crop: np.ndarray) -> str:
        """crop is a BGR numpy array (OpenCV format)."""
        image = Image.fromarray(crop[..., ::-1]).resize((256, 256))  # BGR→RGB, resize to model input

        prompt_text = f"<image>\nUser: {PROMPT}\nAssistant:"
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                images=[image],  # passed separately per minimind-o API
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return decoded.split("Assistant:")[-1].strip()
