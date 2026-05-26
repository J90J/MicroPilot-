"""
Download free traffic datasets for baseline training and evaluation.

Downloads:
  - GTSRB  (320 MB) — German Traffic Sign Recognition, includes speed limit signs
  - S2TLD  (~1.4 GB) — SJTU Small Traffic Light Dataset, US-style lights

Usage:
    python scripts/data_pipeline/download_datasets.py --dataset gtsrb
    python scripts/data_pipeline/download_datasets.py --dataset s2tld
    python scripts/data_pipeline/download_datasets.py --dataset all
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


GTSRB_URL = "https://zenodo.org/records/13741936/files/data.zip?download=1"
S2TLD_BASE = "https://huggingface.co/datasets/yangxue/S2TLD/resolve/main"
S2TLD_FILES = [
    "S2TLD_720x1280.zip",
]


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def extract_zip(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    print(f"Extracted to {out_dir}")


def download_gtsrb(base_dir: Path = Path("data/gtsrb")):
    zip_path = base_dir / "gtsrb.zip"
    print("\n=== Downloading GTSRB (~320 MB) ===")
    download_file(GTSRB_URL, zip_path)
    extract_zip(zip_path, base_dir)
    print("GTSRB ready.")


def download_s2tld(base_dir: Path = Path("data/s2tld")):
    print("\n=== Downloading S2TLD (~1.1 GB) ===")
    for fname in S2TLD_FILES:
        url = f"{S2TLD_BASE}/{fname}"
        zip_path = base_dir / fname
        download_file(url, zip_path)
        extract_zip(zip_path, base_dir)
    print("S2TLD ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gtsrb", "s2tld", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("gtsrb", "all"):
        download_gtsrb()
    if args.dataset in ("s2tld", "all"):
        download_s2tld()
