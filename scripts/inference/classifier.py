"""
Stage 2: MiniMind-O classifier for traffic sign semantics.
Takes a cropped ROI (numpy BGR array) and returns a text prediction.

Loading pattern confirmed from eval_omni.py:
  1. Load main model via AutoModelForCausalLM (trust_remote_code=True)
  2. Separately attach vision encoder from local SigLIP2 path
  3. Process images via model.vision_processor, pass as pixel_values kwarg to generate()
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
    def __init__(
        self,
        model_path: str = "models/minimind-3o",
        siglip_path: str = "models/siglip2",
        lora_path: str = None,
    ):
        self.device = _best_device()
        print(f"MiniMind-O loading on: {self.device}")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Attach SigLIP2 vision encoder (loaded separately per minimind-o design)
        vision_encoder, vision_processor = _load_vision_encoder(siglip_path, self.device)
        object.__setattr__(self.model, "vision_encoder", vision_encoder)
        object.__setattr__(self.model, "vision_processor", vision_processor)

        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"[WARNING] {self.device} failed ({e}), falling back to CPU.")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)

        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"LoRA adapter loaded from {lora_path}")

        self.model.eval()
        self.vision_processor = vision_processor

    def classify(self, crop: np.ndarray) -> str:
        """crop is a BGR numpy array (OpenCV format)."""
        image = Image.fromarray(crop[..., ::-1])  # BGR → RGB

        pixel_values = {
            k: v.to(self.device)
            for k, v in self.vision_processor(images=image, return_tensors="pt").items()
        }

        prompt_text = f"<image>\nUser: {PROMPT}\nAssistant:"
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=24,
                temperature=0.1,
                pixel_values=pixel_values,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return decoded.split("Assistant:")[-1].strip()


def _load_vision_encoder(siglip_path: str, device: torch.device):
    """Load SigLIP2 vision encoder and processor from local path."""
    from transformers import SiglipVisionModel, SiglipImageProcessor
    import warnings

    try:
        encoder = SiglipVisionModel.from_pretrained(siglip_path)
        processor = SiglipImageProcessor.from_pretrained(siglip_path)
        for p in encoder.parameters():
            p.requires_grad = False
        encoder = encoder.eval().to(device)
        print(f"SigLIP2 vision encoder loaded from {siglip_path}")
        return encoder, processor
    except Exception as e:
        warnings.warn(f"Could not load SigLIP2 from {siglip_path}: {e}. Vision will be disabled.")
        return None, None
