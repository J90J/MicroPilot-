"""
One-time setup: downloads MiniMind-O and SigLIP2 weights into models/.
Run this once after cloning the repo.

Usage:
    python scripts/setup_models.py
    python scripts/setup_models.py --token hf_your_token_here
"""

import argparse
from pathlib import Path


def download_models(token: str = None):
    from huggingface_hub import snapshot_download, login

    if token:
        login(token=token, add_to_git_credential=False)

    print("=== Downloading MiniMind-O (jingyaogong/minimind-3o) ===")
    if list(Path("models/minimind-3o").glob("pytorch_model.bin")):
        print("Already downloaded, skipping.")
    else:
        snapshot_download(
            "jingyaogong/minimind-3o",
            local_dir="models/minimind-3o",
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print("MiniMind-O done.")

    print("\n=== Downloading SigLIP2 vision encoder (google/siglip2-base-patch32-256) ===")
    if list(Path("models/siglip2").glob("model.safetensors")):
        print("Already downloaded, skipping.")
    else:
        snapshot_download(
            "google/siglip2-base-patch32-256",
            local_dir="models/siglip2",
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        print("SigLIP2 done.")

    print("\n=== Verifying load ===")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, SiglipVisionModel, SiglipImageProcessor

    tok = AutoTokenizer.from_pretrained("models/minimind-3o", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("models/minimind-3o", trust_remote_code=True, torch_dtype=torch.bfloat16)
    enc = SiglipVisionModel.from_pretrained("models/siglip2")
    proc = SiglipImageProcessor.from_pretrained("models/siglip2")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MiniMind-O: {params:.1f}M params — OK")
    print(f"SigLIP2: loaded — OK")
    print("\nAll models ready. You can now run:")
    print("  python scripts/training/evaluate.py          # zero-shot baseline")
    print("  python scripts/training/train_lora.py        # LoRA fine-tuning")
    print("  python scripts/inference/pipeline.py --input data/raw/dashcam.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="HuggingFace token (hf_...)")
    args = parser.parse_args()
    download_models(args.token)
