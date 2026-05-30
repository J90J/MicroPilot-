# MicroPilot

I built this for my EEP 566 final project at university. The idea is simple: take a very small AI model and make it detect speed limit signs and stop signs from a dashcam video, and then announce them out loud.

The long-term goal is to put this on a **Raspberry Pi** and install it in my girlfriend's **2014 Toyota Corolla** as a little assistant that tells her the speed limit when she passes a sign. Nothing fancy, just something useful.

**Built by:** Jens Jung ([@J90J](https://github.com/J90J)) · Jimmy Yin ([@jimsteryin](https://github.com/jimsteryin))

---

## How it works

```
Dashcam video
     │
     ▼
YOLOv8-nano  ←  detects where the sign is in the frame
     │
     ▼
MiniMind-O   ←  tiny 0.1B LLM that reads what the sign says
(with LoRA)
     │
     ▼
Voice output ←  announces "35" or "stop" out loud
```

The model I use is called **MiniMind-O** (`jingyaogong/minimind-3o`) — it was released in May 2026 and is only 0.1 billion parameters. That is the whole point: it is small enough to run on embedded hardware like a Raspberry Pi but still smart enough to recognize signs after fine-tuning.

---

## Results

Zero-shot (no training at all): **0% accuracy** — the model had no idea what to do.

After LoRA fine-tuning on around 7,500 labeled images:

| Metric | Value |
|---|---|
| Accuracy | **91.9%** |
| Weighted F1 | **0.917** |
| Macro F1 | **0.866** |

The model can classify 14 different US speed limit signs (10 to 75 mph) plus stop signs and background. See `results/` for confusion matrices and the full experiment log.

---

## Setup

### Requirements
- Python 3.11+
- Apple Silicon Mac (MPS) or CUDA GPU — CPU works too but is slow
- Around 8GB RAM

```bash
git clone https://github.com/J90J/MicroPilot-.git
cd MicroPilot-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download the models
```bash
python scripts/setup_models.py
```

This downloads MiniMind-O and SigLIP2 from HuggingFace into `models/`.

---

## Run the demo

```bash
python -m scripts.inference.pipeline \
  --input "your_dashcam_video.mp4" \
  --lora_path models/minimind-o-lora-v2 \
  --yolo_model "runs/detect/models/yolo-speed/weights/best.pt" \
  --batch_size 4
```

A video window opens with red boxes around detected signs. The voice says the number ("35", "25") or "stop" when a sign is confirmed. Press **Q** to quit.

The sign must appear in 2 consecutive frames before anything is announced — this filters out most false positives.

---

## Training

If you want to retrain the LoRA on your own data:

```bash
python scripts/training/train_lora.py \
  --model_id models/minimind-3o \
  --siglip_path models/siglip2 \
  --output models/my-lora \
  --epochs 5 \
  --no_wandb
```

Evaluate:
```bash
python scripts/training/evaluate.py \
  --lora_path models/my-lora \
  --tag my_run
```

Results are saved automatically to `results/`.

---

## Active learning

The pipeline has a `--collect` flag. When you use it, every sign detected with high confidence gets saved automatically to the training dataset. So the more footage you run through it, the better it can get over time.

```bash
python -m scripts.inference.pipeline \
  --input video.mp4 \
  --lora_path models/minimind-o-lora-v2 \
  --yolo_model "runs/detect/models/yolo-speed/weights/best.pt" \
  --batch_size 4 \
  --collect
```

---

## Project structure

```
MicroPilot/
├── data/
│   ├── speed/              # YOLO-format US speed limit dataset
│   └── labeled/
│       └── labels.csv      # combined training labels (~7,500 samples)
├── results/                # eval metrics, confusion matrices, experiment log
├── scripts/
│   ├── data_pipeline/
│   │   └── convert_speed_yolo.py
│   ├── training/
│   │   ├── train_lora.py
│   │   └── evaluate.py
│   └── inference/
│       ├── detector.py     # YOLOv8 sign detector
│       ├── classifier.py   # MiniMind-O wrapper
│       └── pipeline.py     # full demo pipeline
└── runs/                   # YOLO training outputs
```

---

## Raspberry Pi plan

The goal is to run this on a **Raspberry Pi 5 (8GB) + AI HAT+ (26 TOPS)**. YOLO runs on the Hailo NPU, MiniMind-O runs on the CPU with INT8 quantization (`--quantize` flag). You mount a small camera on the dash, power it from a USB car charger (needs 27W), and it works as a standalone assistant without any phone or internet connection.

Still working on packaging this cleanly for the Pi — that is the next step after the semester ends.

---

## Known limitations

- 25 mph and 35 mph signs are sometimes confused — they look very similar at a distance
- Signs on the left side of the frame (at an angle) are sometimes missed by the detector
- Other rectangular signs like bicycle route markers can occasionally trigger a false detection
- No red/green/yellow traffic light classification yet
