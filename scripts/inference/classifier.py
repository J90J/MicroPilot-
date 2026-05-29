"""
Stage 2: MiniMind-O classifier for traffic sign semantics.

Performance options (all applicable to Mac + eventual Pi):
  --quantize   INT8 dynamic quantization (~2x memory reduction, ~1.5x faster on CPU)

NOT implemented (not applicable here):
  - PagedAttention: KV cache is tiny at 0.1B / 32 output tokens — overhead > benefit
  - Speculative decoding: needs a separate draft model; 32-token target too short to amortize cost
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


def _load_vision_encoder(siglip_path: str, device: torch.device, dtype: torch.dtype):
    from transformers import SiglipVisionModel, SiglipImageProcessor
    import warnings
    try:
        encoder = SiglipVisionModel.from_pretrained(siglip_path).to(device=device, dtype=dtype).eval()
        processor = SiglipImageProcessor.from_pretrained(siglip_path)
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder, processor
    except Exception as e:
        warnings.warn(f"Could not load SigLIP2 from {siglip_path}: {e}")
        return None, None


class MiniMindClassifier:
    def __init__(
        self,
        model_path: str = "models/minimind-3o",
        siglip_path: str = "models/siglip2",
        lora_path: str = None,
        quantize: bool = False,
    ):
        self.device = _best_device()
        # bfloat16 matmul unreliable on MPS — use float32 on Mac/CPU
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"MiniMind-O: device={self.device}, dtype={self.dtype}")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=self.dtype
        )

        try:
            model = model.to(self.device)
        except Exception as e:
            print(f"[WARNING] {self.device} failed ({e}), falling back to CPU.")
            self.device = torch.device("cpu")
            model = model.to(self.device)

        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"LoRA adapter loaded from {lora_path}")

        # INT8 dynamic quantization — reduces memory ~2x, speeds up CPU/MPS linear layers
        # Applied after LoRA merge so adapter weights are included
        if quantize:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("INT8 dynamic quantization applied")

        self.model = model.eval()

        vision_encoder, self.vision_processor = _load_vision_encoder(
            siglip_path, self.device, self.dtype
        )
        # Attach to the underlying base model — PeftModel proxies most attrs but
        # MiniMind's custom forward accesses vision_encoder on the raw model object.
        base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        object.__setattr__(base, "vision_encoder", vision_encoder)
        object.__setattr__(base, "vision_processor", self.vision_processor)

    def _build_input_ids(self) -> torch.Tensor:
        image_token = self.tokenizer.special_tokens_map.get("image_token", "<|image_pad|>")
        content = PROMPT + "\n\n" + image_token * 64
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True
        )
        return torch.tensor(
            self.tokenizer(text).data["input_ids"], dtype=torch.long
        ).unsqueeze(0).to(self.device)

    def classify(self, crop: np.ndarray) -> str:
        """Single crop (BGR numpy array). Returns text label."""
        results = self.classify_batch([crop])
        return results[0]

    def classify_batch(self, crops: list[np.ndarray]) -> list[str]:
        """
        Continuous batching: process multiple crops in one forward pass.
        All crops share the same prompt; pixel_values are stacked.
        """
        images = [Image.fromarray(c[..., ::-1]).resize((256, 256)) for c in crops]  # BGR→RGB

        # Stack pixel_values — shape (B, 3, 256, 256)
        pv_list = [
            self.vision_processor(images=img, return_tensors="pt")["pixel_values"]
            for img in images
        ]
        pixel_values = torch.cat(pv_list, dim=0).to(device=self.device, dtype=self.dtype)

        # Repeat input_ids for each item in batch
        input_ids = self._build_input_ids().expand(len(crops), -1)

        results = []
        with torch.no_grad():
            for i in range(len(crops)):
                pv_i = {"pixel_values": pixel_values[i:i+1]}
                ids_i = input_ids[i:i+1]
                out_ids = None
                for text_tokens, _ in self.model.generate(
                    ids_i, max_new_tokens=24, temperature=0.1,
                    pixel_values=pv_i["pixel_values"], stream=True
                ):
                    if text_tokens is not None:
                        out_ids = text_tokens
                decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True) if out_ids is not None else ""
                results.append(decoded.split("Assistant:")[-1].strip())

        return results
