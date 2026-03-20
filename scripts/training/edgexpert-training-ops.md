# EdgeXpert Training Container Operations

## Overview

Training runs inside a Docker container (`llm-training`) on the EdgeXpert (connected via Tailscale)
using the NVIDIA NGC PyTorch image with Unsloth, TRL, and Comet ML pre-installed.

> If Tailscale is down on LAN, temporarily revert DATABASE_HOST and QDRANT_DATABASE_HOST to the LAN IP.

The container image `llm-training-ready-v2` has all pip packages baked in via `docker commit`.
Volume mounts persist model cache and training outputs across container recreations.

## After Every EdgeXpert Power Cycle

**You MUST recreate the container after a power cycle.** `docker start` on a stopped
container reuses stale GPU device handles from the previous boot, causing
`nvidia-smi: Failed to initialize NVML` inside the container. This is a known
limitation of NVIDIA Container Toolkit.

**Exception:** If the EdgeXpert was NOT power-cycled (just the container stopped),
`docker start llm-training` is safe. Verify with `docker exec llm-training nvidia-smi`
immediately after starting — if GPU works, no recreate needed.

```bash
# 1. Remove the old container
ssh edgexpert "docker rm -f llm-training 2>/dev/null; echo 'Cleaned'"

# 2. Recreate from the committed image (all packages pre-installed)
ssh -t edgexpert "docker run \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env-file \$HOME/training.env \
  -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
  -v \$HOME/training-outputs:/workspace/outputs \
  -v \$HOME/mlops-llm-twin:/workspace/project \
  --name llm-training \
  -it \
  --entrypoint /usr/bin/bash \
  llm-training-ready-v2"

# 3. Verify GPU inside container
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

**What survives** (volume mounts): Model cache, training outputs, project repo.
**What is baked in** (`llm-training-ready-v2`): All pip packages.

## Deploying Scripts

```bash
# From MacBook
scp ~/Downloads/<script>.py edgexpert:~/

# From EdgeXpert host (or via ssh)
ssh edgexpert "docker cp ~/<script>.py llm-training:/workspace/<script>.py"
```

## Pre-Run Verification Checklist

**Run inside the container before every training launch.** Three layers, each catching
a different failure class.

### Layer 1: Import Checks

Verify all required classes are importable. Catches missing packages or broken installs.

```python
python3 -c "
from unsloth import PatchDPOTrainer, FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer  # or SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import TextStreamer
import comet_ml
print('All imports OK')
"
```

### Layer 2: API Signature Inspection

Verify parameter names match the installed library versions. Catches renames
(e.g., `tokenizer` → `processing_class`) and moved kwargs (e.g., `beta` moving
from DPOTrainer to DPOConfig).

```python
python3 -c "
import inspect
from trl import DPOConfig, DPOTrainer

# Check DPOConfig has expected params
sig = inspect.signature(DPOConfig.__init__)
for param in ['beta', 'max_length', 'max_prompt_length', 'learning_rate',
              'eval_strategy', 'bf16']:
    assert param in sig.parameters, f'MISSING from DPOConfig: {param}'

# Check DPOTrainer uses processing_class (not tokenizer)
sig2 = inspect.signature(DPOTrainer.__init__)
assert 'processing_class' in sig2.parameters, 'DPOTrainer missing processing_class'
assert 'tokenizer' not in sig2.parameters, 'DPOTrainer still has old tokenizer param'

print('API signatures verified OK')
"
```

For SFT, check `SFTConfig` for `max_seq_length`, `packing`, `dataset_text_field`
and `SFTTrainer` for `processing_class`.

### Layer 3: Trainer Instantiation Smoke Test

Verify the trainer can be instantiated with a real model and dummy data. Catches
runtime attribute errors from cross-package version mismatches that imports and
signatures cannot detect.

**Why this exists:** The DPO `warnings_issued` bug (March 4, 2026) passed Layers 1
and 2 but crashed at DPOTrainer.__init__ because PEFT 0.18.1 doesn't propagate a
`warnings_issued` dict that Unsloth's patched trainer expects. Only instantiation
with a real model triggers this code path.

```python
# Run AFTER loading your model and tokenizer, BEFORE the real training call
from datasets import Dataset

dummy = Dataset.from_dict({
    "prompt": ["test prompt"],
    "chosen": ["good answer"],
    "rejected": ["bad answer"],
})

try:
    model.warnings_issued = {}  # Fix for PEFT 0.18.1 / transformers 5.2.0 gap
    t = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=dummy,
        args=DPOConfig(output_dir="/tmp/smoke_test", max_steps=0),
    )
    print("SMOKE TEST PASSED: Trainer instantiation OK")
    del t
except Exception as e:
    print(f"SMOKE TEST FAILED: {e}")
    sys.exit(1)
```

For SFT:
```python
dummy = Dataset.from_dict({"text": ["test sample"]})
try:
    t = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dummy,
        args=SFTConfig(output_dir="/tmp/smoke_test", max_steps=0),
    )
    print("SMOKE TEST PASSED")
    del t
except Exception as e:
    print(f"SMOKE TEST FAILED: {e}")
    sys.exit(1)
```

### Layer Summary

| Layer | What it catches | Example failure |
|-------|----------------|-----------------|
| Import | Missing packages, broken installs | `ModuleNotFoundError: No module named 'unsloth'` |
| Signature | Parameter renames, moved kwargs | `beta` silently ignored in DPOTrainer (now in DPOConfig) |
| Smoke test | Runtime attribute errors, cross-package incompatibilities | `AttributeError: 'PeftModelForCausalLM' has no attribute 'warnings_issued'` |

## Running Training Jobs

```bash
# Inside container — always use nohup with python -u for SSH resilience and unbuffered output
cd /workspace && nohup python3 -u <script>.py > <log>.log 2>&1 &

# Monitor (Ctrl+C is safe — only kills tail, not the training process)
tail -f /workspace/<log>.log
```

## Reconnecting After SSH Disconnect

Closing the MacBook lid kills the TCP connection. The container and any `nohup`
processes are unaffected. Just reopen and reconnect:

```bash
ssh edgexpert
docker exec -it llm-training bash
tail -f /workspace/<log>.log
```

## Updating the Committed Image

After installing new packages or making environment changes you want to persist:

```bash
ssh edgexpert "docker commit llm-training llm-training-ready-v2"
```

## Known Issues

### PEFT 0.18.1 / Transformers 5.2.0 warnings_issued Gap
PEFT wrapper doesn't propagate `warnings_issued` dict from transformers. Unsloth's
patched DPOTrainer crashes at `model.warnings_issued["estimate_tokens"] = True`.
Fix: `model.warnings_issued = {}` before trainer instantiation. Already included in
`dpo_train.py`.

### Triton Shared Memory (GB10 Blackwell)
GB10's per-SM shared memory (101KB) is below what torch.compile's auto-tuned Triton
kernels expect (119KB+). Set before any torch imports:
```python
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
```

### Unsloth Import Order
`from unsloth import PatchDPOTrainer` (or PatchSFTTrainer) must come before
trl/transformers/peft imports. The patch modifies trainer classes at import time.

### TRL 0.26.1 API Changes from Handbook
- `tokenizer=` → `processing_class=` (both SFTTrainer and DPOTrainer)
- `beta`, `max_length`, `max_prompt_length` moved into DPOConfig (not DPOTrainer kwargs)
- `max_seq_length`, `dataset_text_field`, `dataset_num_proc`, `packing` moved into SFTConfig
- Set `eos_token=None` in SFTConfig if appending EOS manually

## Completed Training Runs

| Run | Date | Script | Results | Comet ML |
|-----|------|--------|---------|----------|
| SFT | March 3, 2026 | sft_train.py | sft-training-results.md | [link](https://www.comet.com/christopher-chen/llm-twin/32b5dd75a09145f5a51b82f0319463f9) |
| DPO | March 4, 2026 | dpo_train.py | dpo-training-results.md | [link](https://www.comet.com/christopher-chen/llm-twin/276e192f49c549cc88b46f991307bb96) |

## Container Specs

| Component | Value |
|---|---|
| Base image | nvcr.io/nvidia/pytorch:25.11-py3 |
| Committed image | llm-training-ready-v2 |
| PyTorch | 2.10.0a0+b558c986e8.nv25.11 (NGC custom) |
| CUDA Toolkit | 13.0 |
| Triton | 3.5.0 |
| Key packages | transformers 4.57.6, trl 0.26.1, unsloth 2026.2.1, peft 0.18.1, comet_ml 3.56.0 |

## vLLM Inference Container Recreation

**Same power-cycle rule as the training container.** After host power-off/reboot,
always `docker rm` + `docker run`. After container-only stop (host stayed on),
`docker start vllm-inference` is safe.

```bash
# 1. Remove the old container
ssh edgexpert "docker rm -f vllm-inference 2>/dev/null; echo 'Cleaned'"

# 2. Recreate from official vLLM image
ssh edgexpert "docker run -d \
  --gpus all \
  --name vllm-inference \
  -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model kwisschen/TwinLlama-3.1-8B-Merged \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --port 8000"

# 3. Wait ~5 minutes (download + load + compile + CUDA graph capture)
#    Monitor progress:
ssh edgexpert "docker logs --tail 5 -f vllm-inference"
#    Look for: "Capturing CUDA graphs (decode, FULL): 100%"
#    Then Ctrl+C and verify:

curl http://<EDGEXPERT_IP>:8001/v1/models
```

**Port mapping:** vLLM listens on port 8000 inside the container, mapped to 8001
on the host (`-p 8001:8000`). All external references (settings.py, FastAPI, docs)
use port 8001.

**Startup sequence:** Download weights from HuggingFace (~130s) → Load into GPU
(~86s, 14.99 GiB) → torch.compile (~9s) → CUDA graph capture (~40s) → API ready.

**No `docker commit` needed.** Unlike the training container, `vllm-inference` uses
the stock `vllm/vllm-openai:latest` image with no additional pip installs.
