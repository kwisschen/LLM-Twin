# SFT Training Results — TwinLlama-3.1-8B

## Training Run: March 3, 2026

### Configuration
- Base model: meta-llama/Meta-Llama-3.1-8B
- Method: LoRA (r=32, alpha=32, dropout=0.0)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable parameters: 83,886,080 / 8,114,147,328 (1.03%)
- Precision: BF16 (full, no quantization)
- Dataset: kwisschen/llmtwin — 19,124 train + 2,127 test
- Categories: repositories (13,617), posts (58), conversations (5,449)
- Packing: Enabled — compressed 19,124 samples into 1,044 packed sequences
- Effective batch size: 2 x 8 gradient accumulation = 16
- Epochs: 3 (198 total steps after packing)
- Learning rate: 3e-4 with linear decay, 10-step warmup
- Optimizer: adamw_8bit

### Hardware
- NVIDIA GB10 Grace Blackwell Superchip, 128GB unified memory
- Eager mode (torch.compile disabled — GB10 shared memory limit)
- NGC container: nvcr.io/nvidia/pytorch:25.11-py3

### Results
| Metric | Value |
|---|---|
| Training time | 1h 30m 43s (198 steps, ~27.5s/step) |
| Train loss (start) | 2.725 |
| Train loss (end) | 1.076 (min), 1.340 (final step) |
| Train loss (average) | 1.365 |
| Eval loss (epoch 1) | 1.383 |
| Eval loss (epoch 2) | 1.297 |
| Eval loss (epoch 3) | 1.275 |
| Grad norm (stable range) | 0.16 - 0.30 |

### Observations
- Loss curve: Clear downward trend with expected noise. No plateau, no divergence.
- Eval loss decreased every epoch — no overfitting detected.
- Train-eval gap: ~0.09 (1.18 avg train epoch 3 vs 1.275 eval) — healthy generalization.
- Grad norm spike to 3.9 during warmup (steps 1-10) is expected; settled to 0.2 range.

### Artifacts
- LoRA adapter: https://huggingface.co/kwisschen/TwinLlama-3.1-8B (~336MB)
- Merged 16-bit: https://huggingface.co/kwisschen/TwinLlama-3.1-8B-Merged (~16GB)
- Local adapter: /workspace/outputs/sft_lora_adapter
- Local merged: /workspace/outputs/sft_merged
- Checkpoints: /workspace/outputs/sft_checkpoints (per-epoch)
- Comet ML: https://www.comet.com/christopher-chen/llm-twin/32b5dd75a09145f5a51b82f0319463f9

### Smoke Test Output
Prompt: "Write a paragraph about the importance of data quality in machine learning pipelines."
Model generated coherent, on-topic paragraph with proper EOS termination.
Content was generic (expected — SFT teaches style, DPO will refine quality).
