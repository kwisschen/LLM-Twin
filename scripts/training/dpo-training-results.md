# DPO Training Results — TwinLlama-3.1-8B-DPO

**Date:** March 4, 2026
**Comet ML:** https://www.comet.com/christopher-chen/llm-twin/276e192f49c549cc88b46f991307bb96

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | kwisschen/TwinLlama-3.1-8B (SFT LoRA adapter) | Continues from SFT Phase 1 |
| LoRA rank | 32 | Matches SFT for consistency |
| LoRA alpha | 32 | alpha/rank = 1.0 scaling |
| Beta | 0.5 | Higher than default 0.1 — prevents formal language drift |
| Learning rate | 2e-6 | 150x lower than SFT (3e-4) — DPO is refinement |
| Epochs | 1 | Light touch — multiple epochs cause formality |
| Batch size | 2 × 8 grad accum = 16 effective |
| Precision | BF16 (full, no quantization) |
| Optimizer | adamw_8bit |
| Warmup steps | 10 |
| max_length | 1024 |
| max_prompt_length | 1024 |
| Sequence packing | No (DPO handles prompt/chosen/rejected alignment internally) |

## Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | 11,044 | kwisschen/llmtwin-dpo |
| Test | 583 | kwisschen/llmtwin-dpo |
| Columns | prompt, chosen, rejected, category |

Category distribution: repositories 8,429, conversations 2,567, posts 48.

## Training Steps

11,044 samples / 16 effective batch = **690 steps × 1 epoch = 690 total steps**
(691 logged due to rounding)

## Results

### Loss
- Train loss: 3.25 → 0.00002 (near-zero — model fully learned training preferences)
- Eval loss: 0.187 → 0.105 (continuously decreasing, no overfitting)
- Train/eval gap: Eval loss healthy throughout — near-zero train loss not a problem

### Rewards & Margins (DPO-specific metrics)
- Train chosen rewards: 5.03 → 26.37 (increasing — model prefers chosen responses)
- Train rejected rewards: -2.81 → 6.56 (stays relatively low)
- Train margins: 5.58 → 28.71 (widening — model increasingly distinguishes style)
- Eval chosen rewards: 11.67 → 13.29
- Eval rejected rewards: -0.67 → -0.40
- Eval margins: 12.22 → 13.69 (growing conservatively — good generalization)

### Accuracies
- Train: 0.69 → 1.0 (reached 100% gradually over 691 steps — healthy)
- Eval: 0.94 → 0.96 (never hit 100% — dataset not trivially easy)

### Gradient Norm
- Range: 0.0002 → 8.26 (few spikes, generally small — stable training)

## Training Time

- Wall clock: 3,589 seconds (~60 minutes)
- Steps per second: 0.193
- Samples per second: 3.077

## Deliverables

| Artifact | Location | Size |
|----------|----------|------|
| DPO LoRA adapter | kwisschen/TwinLlama-3.1-8B-DPO | ~17MB |
| Merged 16-bit model | kwisschen/TwinLlama-3.1-8B-DPO-Merged | ~16GB |
| Comet ML experiment | christopher-chen/llm-twin project | Full metrics |

## Sanity Check Output

Prompt: "Write a paragraph to introduce supervised fine-tuning."

DPO model response:
> Supervised fine-tuning is a technique where you take a pre-trained model and retrain it on a smaller dataset that is specific to your application. This approach allows you to leverage the knowledge gained from the pre-trained model while adapting it to your particular task, which can improve performance and efficiency.

Observation: Natural, concise, no GPT-isms ("delve into", "it's important to note"). Reads like a human developer explaining a concept, not a chatbot.

## Issues Encountered

### 1. `warnings_issued` AttributeError
**Cause:** Compatibility gap between transformers 5.2.0 and peft 0.18.1. PEFT wrapper doesn't propagate the `warnings_issued` dict that Unsloth's patched DPOTrainer expects.
**Fix:** `model.warnings_issued = {}` before DPOTrainer initialization.
**Prevention:** Added trainer instantiation smoke test to pre-run verification checklist.

### 2. TRL 0.26.1 API Changes from Handbook
**Cause:** `tokenizer` renamed to `processing_class` in DPOTrainer; `beta`, `max_length`, `max_prompt_length` moved from DPOTrainer kwargs into DPOConfig.
**Fix:** Caught by Pre-Run API Verification Mandate (inspect.signature checks in container).
**Impact if missed:** Would have silently used beta=0.1 instead of 0.5 — exactly the formal language failure mode the handbook authors spent 20+ experiments diagnosing.

## Comparison: SFT vs DPO

| Metric | SFT | DPO |
|--------|-----|-----|
| Training time | 1h 30m | 1h 00m |
| Steps | 198 × 3 = 594 | 691 × 1 = 691 |
| Train loss final | 1.08 | ~0 |
| Eval loss final | 1.275 | 0.105 |
| Dataset | 19,124 instruction pairs | 11,044 preference triples |
| Purpose | Teach content & domain | Refine voice & style |
