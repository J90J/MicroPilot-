# MicroPilot — Experiment Log

**Goal:** Maximize weighted F1 on held-out 20% eval set (1,513 samples).  
**Classes:** `no_sign` (3,785 samples) + 14 US speed limits (3,781 samples total).  
**Hardware:** Apple Silicon MPS, float32.  
**Eval threshold:** Weighted F1 ≥ 0.60 = acceptable · ≥ 0.75 = good · ≥ 0.85 = excellent.

---

## Run 1 — Baseline LoRA (current)

**Status:** Training in progress  
**Config:**
| Param | Value |
|---|---|
| Epochs | 5 |
| Batch size | 4 |
| LR | 2e-4 |
| LoRA r | 8 |
| LoRA α | 16 |
| Train samples | ~6,053 (80% split, seed=42) |
| Warmup | 10% of steps |
| Grad clip | 1.0 |
| Weight decay | 0.01 |

**Results:**

| Metric | Value |
|---|---|
| Accuracy | **0.919** (1390/1513) |
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
| speed_limit_45 | 0.875 | 0.649 | **0.746** ⚠️ | 97 |
| speed_limit_50 | 0.875 | 0.467 | **0.609** ⚠️ | 15 |
| speed_limit_55 | 0.917 | 0.500 | **0.647** ⚠️ | 22 |
| speed_limit_60 | 0.929 | 0.867 | 0.897 | 15 |
| speed_limit_65 | 0.909 | 0.909 | 0.909 | 11 |
| speed_limit_70 | 1.000 | 1.000 | 1.000 | 13 |
| speed_limit_75 | 1.000 | 0.941 | 0.970 | 17 |

**Verdict:** ✅ Excellent. Blows past the ≥0.85 threshold. Weak spots are 45/50/55 mph — all have low recall (confused with adjacent speeds 25/35). Near-miss errors, not random noise.

**Conclusion:** Run 1 is a strong result. Run 2 will target the 45/50/55 weak spot with class-weighted loss.

---

## Run 2 — Class-weighted loss, aggressive (4×)

**Config:** 3 epochs, lr=1e-4, dropout=0.05, class weights: 4× on 45/50/55, 2× on 30/40/60/65/70/75

**Results:** Accuracy 0.835 · Macro F1 0.763 · Weighted F1 0.840

**Verdict:** ❌ Overcorrected. speed_limit_45 recall jumped (0.65→0.96) but precision collapsed (0.875→0.369). speed_limit_35 recall destroyed (0.874→0.440). Net loss vs Run 1.

---

## Run 3 — Class-weighted loss, gentle (2×) + higher dropout

**Config:** 3 epochs, lr=1e-4, dropout=0.1, class weights: 2× on 45/50/55 only

**Results:** Accuracy 0.692 · Macro F1 0.209 · Weighted F1 0.660

**Verdict:** ❌ Much worse. dropout=0.1 + class weighting collapsed 10 classes to zero F1. Too much regularization prevented learning rare classes entirely.

---

## Run 4 — Early stopping at epoch 2 (Run 1 config)

**Config:** 2 epochs, lr=2e-4, dropout=0.05, no class weights (identical to Run 1 but stops before loss near-zero)

**Results:** Accuracy 0.677 · Macro F1 0.194 · Weighted F1 0.637

**Verdict:** ❌ Worst result. 2 epochs is undertrained at lr=2e-4 — many classes have zero F1 (10, 30, 50, 55, 60, 65, 70, 75). The model hasn't seen enough gradient steps to learn the rare classes.

**Key insight:** The near-zero training loss in Run 1 (~0.003) looked like harmful overfitting, but the LoRA adapter (258K / 113M params) is too small to truly memorize visual content. The frozen SigLIP + base model does the heavy lifting; the adapter just learns the label mapping. More epochs = more time to stabilize that mapping. Early stopping at epoch 2 actively hurt generalization.

---

## Final Conclusion

**Run 1 is the best model** (`models/minimind-o-lora`).

The remaining gap on 45/50/55 mph (F1 0.61–0.75) is a data limitation — those classes have 15–97 eval samples and are visually close to adjacent speeds. All tuning attempts shifted errors rather than eliminating them. More labeled data for those classes is the only reliable path to further improvement.

**Final model stats:**
- Accuracy: 0.919 (1390/1513)
- Macro F1: 0.866
- Weighted F1: 0.917
- Zero-shot baseline: 0.000 → **+91.9 percentage points**

---

## Notes

- S2TLD traffic light data converted entirely to `no_sign` (all "off/idle" signal frames) — no red/green/yellow light samples in training data.
- Speed limit class distribution is imbalanced: speed_limit_35 (863) and speed_limit_25 (718) dominate; speed_limit_10/60/65/70/75 each have < 65 samples.
- Zero-shot baseline: 0.000 accuracy (model produces descriptions, not labels).
