"""
Stage 2: MiniMind-O classifier for traffic sign semantics.
Takes a cropped ROI image and returns a structured prediction.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import numpy as np


PROMPT = "What traffic signal or sign is visible in this image? Be concise."


class MiniMindClassifier:
    def __init__(self, model_path: str, lora_path: str = None):
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"MiniMind-O running on: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)

        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"LoRA adapter loaded from {lora_path}")

        self.model.eval()

    def classify(self, crop: np.ndarray) -> str:
        image = Image.fromarray(crop[..., ::-1])  # BGR → RGB
        prompt_text = f"<image>\nUser: {PROMPT}\nAssistant:"
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = decoded.split("Assistant:")[-1].strip()
        return answer
