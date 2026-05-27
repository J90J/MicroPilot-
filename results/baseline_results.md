# MicroPilot Evaluation Results

## Zero-Shot Baseline (MiniMind-O, no fine-tuning)

**Dataset:** 500 samples — GTSRB speed limit signs + S2TLD traffic lights  
**Model:** `jingyaogong/minimind-3o` (0.1B params, no LoRA)  
**Date:** 2026-05-27

| Metric | Value |
|---|---|
| Overall Accuracy | 0.000 (0/500) |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| no_sign | 0.000 | 0.000 | 0.000 |
| red_light | 0.000 | 0.000 | 0.000 |
| speed_limit_20 | 0.000 | 0.000 | 0.000 |

**Sample model outputs (zero-shot):**
- GT: `speed_limit_20` → Model: *"The traffic signal visible in the image is a red-black sign..."*
- GT: `speed_limit_30` → Model: *"...the '3' sign, which is a common feature in traffic signaling..."*
- GT: `speed_limit_80` → Model: *"...a 'WEED' sign, which is a common indicator..."*
- GT: `no_sign` → Model: *"...the 'MAX' sign, which is a common feature in buses..."*

**Interpretation:** The model demonstrates awareness of traffic signs (produces traffic-related descriptions) but cannot reliably classify sign types zero-shot. This is expected for a 0.1B model on a specialized domain and motivates LoRA fine-tuning.

---

## LoRA Fine-Tuned Results

> To be filled after training run completes.

---

## Tesla Dashcam Results

> To be filled after dashcam footage is collected.
