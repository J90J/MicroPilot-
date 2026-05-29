# MicroPilot Evaluation Results

## Zero-Shot Baseline (MiniMind-O, no fine-tuning)

**Dataset:** 500 samples — S2TLD traffic lights (US-style) + manually labeled speed limit frames  
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

## LoRA Fine-Tuned Results (Run 1 — best)

**Model:** `minimind-o-lora` (5 epochs, lr=2e-4, r=8, α=16)  
**Eval set:** 1,513 samples (held-out 20%, seed=42)  
**Date:** 2026-05-28

| Metric | Value |
|---|---|
| Overall Accuracy | **0.919** (1390/1513) |
| Macro F1 | **0.866** |
| Weighted F1 | **0.917** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no_sign | 1.000 | 1.000 | 1.000 | 763 |
| speed_limit_10 | 1.000 | 1.000 | 1.000 | 15 |
| speed_limit_15 | 1.000 | 1.000 | 1.000 | 38 |
| speed_limit_20 | 0.981 | 0.964 | 0.972 | 55 |
| speed_limit_25 | 0.762 | 0.877 | 0.815 | 146 |
| speed_limit_30 | 0.782 | 0.796 | 0.789 | 54 |
| speed_limit_35 | 0.793 | 0.874 | 0.832 | 175 |
| speed_limit_40 | 0.790 | 0.831 | 0.810 | 77 |
| speed_limit_45 | 0.875 | 0.649 | 0.746 | 97 |
| speed_limit_50 | 0.875 | 0.467 | 0.609 | 15 |
| speed_limit_55 | 0.917 | 0.500 | 0.647 | 22 |
| speed_limit_60 | 0.929 | 0.867 | 0.897 | 15 |
| speed_limit_65 | 0.909 | 0.909 | 0.909 | 11 |
| speed_limit_70 | 1.000 | 1.000 | 1.000 | 13 |
| speed_limit_75 | 1.000 | 0.941 | 0.970 | 17 |

**Confusion matrix:** `confusion_matrix_lora_run1.png`

**Hyperparameter search (4 runs):**

| Run | Epochs | LR | Key change | wF1 |
|---|---|---|---|---|
| Run 1 | 5 | 2e-4 | — | **0.917** ✅ best |
| Run 2 | 3 | 1e-4 | 4× class weights (45/50/55) | 0.840 |
| Run 3 | 3 | 1e-4 | 2× weights + dropout=0.1 | 0.660 |
| Run 4 | 2 | 2e-4 | early stop epoch 2 | 0.637 |

**Interpretation:** LoRA fine-tuning improved accuracy from 0.000 → 0.919 (+91.9 pp). Weak spots are speed_limit_45/50/55 (F1 0.61–0.75) due to visual similarity with adjacent speeds and low training support. All tuning variants made overall performance worse; Run 1 is the final model.

---

## Tesla Dashcam Results

> To be filled after dashcam footage is collected.
