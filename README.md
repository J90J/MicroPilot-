# MicroPilot

**On-Device Driver Intelligence at 0.1B Parameters**

Real-time traffic sign and stoplight detection using [MiniMind-O](https://github.com/jingyaogong/minimind-o), the world's smallest multimodal LLM (0.1B parameters, audio + image + text). Built for EEP 566: Real-Time Intelligence on Embedded Devices.

**Authors:** Jens Jung ([@J90J](https://github.com/J90J)) · Jimmy Yin ([@jimsteryin](https://github.com/jimsteryin))

---

## Architecture

```
Video frame (1–2 FPS)
     │
     ▼
YOLOv8-nano          ← fast region-of-interest detector
(finds signs/lights)
     │
  ROI crops
     │
     ▼
MiniMind-O (LoRA)    ← semantic classification
"red light" / "speed limit 35" / ...
     │
     ▼
macOS TTS (`say`)    ← audio announcement
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/J90J/MicroPilot-.git
cd MicroPilot-
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon users:** PyTorch uses the MPS backend automatically. No extra steps needed.

### 3. Get MiniMind-O weights
```bash
git clone https://github.com/jingyaogong/minimind-o models/minimind-o
```
Then follow the model README to download the actual weights into `models/minimind-o/`.

### 4. Set up wandb (experiment tracking)
```bash
pip install wandb
wandb login   # paste your API key from wandb.ai
```

---

## Usage

### Step 1 — Extract frames from dashcam footage
```bash
python scripts/data_pipeline/extract_frames.py \
  --input data/raw/dashcam.mp4 \
  --output data/frames \
  --fps 1
```

### Step 2 — Run zero-shot baseline evaluation
```bash
python scripts/training/evaluate.py \
  --model_path models/minimind-o \
  --dataset data/labeled/dataset.json
```

### Step 3 — Fine-tune with LoRA
```bash
python scripts/training/train_lora.py \
  --dataset data/labeled/dataset.json \
  --model_path models/minimind-o \
  --output models/minimind-o-lora \
  --epochs 5
```

### Step 4 — Run the full pipeline on a video
```bash
python scripts/inference/pipeline.py \
  --input data/raw/dashcam.mp4 \
  --model_path models/minimind-o \
  --lora_path models/minimind-o-lora \
  --fps 1
```

### Step 4b — Run on live webcam
```bash
python scripts/inference/pipeline.py \
  --input 0 --live \
  --model_path models/minimind-o
```

---

## Labeling format

Create `data/labeled/labels.csv`:
```csv
filename,label
frame_000100.jpg,red_light
frame_000200.jpg,green_light
frame_000300.jpg,speed_limit_25
frame_000400.jpg,no_sign
```

Then build the dataset JSON:
```bash
python scripts/data_pipeline/prepare_dataset.py \
  --frames data/frames \
  --labels data/labeled/labels.csv \
  --output data/labeled/dataset.json
```

---

## Project structure

```
MicroPilot/
├── data/
│   ├── raw/          # dashcam footage (gitignored)
│   ├── frames/       # extracted frames (gitignored)
│   └── labeled/      # labels.csv + dataset.json (gitignored)
├── models/           # model weights (gitignored)
├── scripts/
│   ├── data_pipeline/
│   │   ├── extract_frames.py
│   │   └── prepare_dataset.py
│   ├── training/
│   │   ├── train_lora.py
│   │   └── evaluate.py
│   └── inference/
│       ├── detector.py    # YOLO pre-filter
│       ├── classifier.py  # MiniMind-O wrapper
│       └── pipeline.py    # end-to-end runner
└── notebooks/
    └── baseline_eval.ipynb
```

---

## Target labels (MVP)

| Label | Description |
|---|---|
| `red_light` | Red traffic light visible |
| `green_light` | Green traffic light visible |
| `yellow_light` | Yellow traffic light visible |
| `speed_limit_XX` | Speed limit sign (e.g. `speed_limit_25`) |
| `no_sign` | No relevant sign in frame |
