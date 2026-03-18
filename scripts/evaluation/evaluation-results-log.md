# Evaluation Results Log

## Run 1 — Corrupted Merged Models (March 8, 2026)
**Issue:** Merged models pushed with transformers 5.2.0, loaded under 4.57.6. Silent weight corruption caused "://" artifacts and template regurgitation.
**Root cause:** Cross-version model serialization incompatibility.
**Status:** INVALID — results discarded.

| Model | Accuracy | Style |
|---|---|---|
| TwinLlama-3.1-8B (SFT) | 1.65 | 1.78 |
| TwinLlama-3.1-8B-DPO (beta=0.5) | 1.59 | 1.72 |
| Meta-Llama-3.1-8B-Instruct | 2.67 | 2.21 |

## Run 2 — Correct Models, DPO beta=0.5 (March 9, 2026)
**Fix:** Re-merged SFT model under transformers 4.57.6. DPO-Merged was already correct.
**Status:** VALID — SFT strong, DPO regressed both metrics.

| Model | Accuracy | Style |
|---|---|---|
| TwinLlama-3.1-8B (SFT) | 2.37 | 2.78 |
| TwinLlama-3.1-8B-DPO (beta=0.5) | 2.05 | 2.47 |
| Meta-Llama-3.1-8B-Instruct | 2.67 | 2.21 |

**Finding:** SFT style (2.78) nturated the 3.0 ceiling. DPO with beta=0.5 had no headroom to improve and degraded both metrics. Decision: retrain DPO with beta=0.1 for more freedom.

## Run 3 — DPO beta=0.1 (March 9, 2026)
**Change:** Beta 0.5 → 0.1. All other hyperparameters identical.
**Status:** VALID — DPO nearly matches SFT. Beta tuning recovered quality.

| Model | Accuracy | Style |
|---|---|---|
| TwinLlama-3.1-8B (SFT) | 2.37 | 2.78 |
| TwinLlama-3.1-8B-DPO (beta=0.1) | 2.31 | 2.74 |
| Meta-Llama-3.1-8B-Instruct | 2.67 | 2.21 |

**Finding:** Beta=0.1 recovered +0.26 accuracy and +0.27 style versus beta=0.5. DPO still slightly below SFT (-0.06 accuracy, -0.04 style) because SFT already saturated the style ceiling at 2.78/3.0. Best model for deployment: TwinLlama-3.1-8B (SFT). DPO demonstrates hyperparameter sensitivity understanding without improving the final model.
