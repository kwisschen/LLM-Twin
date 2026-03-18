#!/usr/bin/env python3
"""
DPO Training Script - TwinLlama-3.1-8B-DPO
============================================
Loads SFT LoRA adapter (kwisschen/TwinLlama-3.1-8B) on top of base
Llama 3.1 8B, trains with DPO on kwisschen/llmtwin-dpo.

Infrastructure: NVIDIA GB10 Blackwell (128GB unified memory)
Container: llm-training (NGC PyTorch + Unsloth/TRL 0.26.1/PEFT)
Launch: nohup python3 /workspace/dpo_train.py > /workspace/dpo_training.log 2>&1 &

API changes from handbook (verified in container 2026-03-04):
- DPOTrainer uses processing_class instead of tokenizer
- beta, max_length, max_prompt_length moved into DPOConfig (not DPOTrainer kwargs)
"""
import os
import sys
import torch

os.environ["COMET_API_KEY"] = os.environ.get("COMET_API_KEY", "")
os.environ["COMET_PROJECT_NAME"] = "llm-twin"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

print("=" * 60)
print("DPO TRAINING - TwinLlama-3.1-8B-DPO")
print("=" * 60)

from unsloth import PatchDPOTrainer
PatchDPOTrainer()

from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer

# --- Configuration ---
SFT_MODEL_NAME = "kwisschen/TwinLlama-3.1-8B"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
BETA = 0.1
LEARNING_RATE = 2e-6
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw_8bit"
OUTPUT_DIR = "/workspace/dpo_output"
DPO_ADAPTER_NAME = "kwisschen/TwinLlama-3.1-8B-DPO"
DPO_MERGED_NAME = "kwisschen/TwinLlama-3.1-8B-DPO-Merged"
DATASET_NAME = "kwisschen/llmtwin-dpo"

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

# --- 4. Load SFT model ---
print(f"Loading SFT model: {SFT_MODEL_NAME}")
print(f"BF16 supported: {is_bfloat16_supported()}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
)

print(f"Model loaded. Device: {model.device}")
print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

# --- 5. Create fresh LoRA adapters for DPO ---
print(f"Creating LoRA adapters: r={LORA_RANK}, alpha={LORA_ALPHA}")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "up_proj", "down_proj",
        "o_proj", "gate_proj",
    ],
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# --- 6. Load and format DPO dataset ---
print(f"Loading dataset: {DATASET_NAME}")
train_dataset = load_dataset(DATASET_NAME, split="train")
eval_dataset = load_dataset(DATASET_NAME, split="test")
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print(f"Columns: {train_dataset.column_names}")

# Dataset has: prompt, chosen, rejected, category
# DPOTrainer uses prompt/chosen/rejected; category is ignored automatically.

EOS_TOKEN = tokenizer.eos_token


def format_samples(example):
    """Apply Alpaca template to prompt, append EOS to chosen/rejected."""
    example["prompt"] = ALPACA_TEMPLATE.format(example["prompt"])
    example["chosen"] = example["chosen"] + EOS_TOKEN
    example["rejected"] = example["rejected"] + EOS_TOKEN
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


print("Formatting dataset with Alpaca template...")
train_dataset = train_dataset.map(format_samples)
eval_dataset = eval_dataset.map(format_samples)

sample = train_dataset[0]
print(f"Sample prompt (200ch): {sample['prompt'][:200]}")
print(f"Sample chosen (200ch): {sample['chosen'][:200]}")

# --- 7. DPO Training ---
# TRL 0.26.1: beta/max_length/max_prompt_length are DPOConfig params.
# tokenizer is now processing_class in DPOTrainer.
print(f"DPOTrainer: beta={BETA}, lr={LEARNING_RATE}, epochs={NUM_EPOCHS}")
print(f"Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

# Fix: PEFT 0.18.1 doesn't propagate warnings_issued from transformers 5.2.0.
# Unsloth's patched DPOTrainer expects model.warnings_issued["estimate_tokens"] to exist.
# Without this, __init__ crashes with AttributeError on PeftModelForCausalLM.
model.warnings_issued = {}

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=DPOConfig(
        # DPO-specific
        beta=BETA,
        max_length=MAX_SEQ_LENGTH // 2,
        max_prompt_length=MAX_SEQ_LENGTH // 2,

        # Learning rate
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",

        # Batch configuration
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Duration
        num_train_epochs=NUM_EPOCHS,

        # Precision
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        # Optimizer
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,

        # Output and logging
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        report_to="comet_ml",
        seed=42,
    ),
)

num_samples = len(train_dataset)
eff_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
steps = num_samples // eff_batch
print(f"Expected: {num_samples}/{eff_batch} = {steps} steps x {NUM_EPOCHS} = {steps * NUM_EPOCHS} total")

print("Starting DPO training...")
print("=" * 60)
trainer.train()
print("=" * 60)
print("DPO training complete!")

# --- 8. Sanity check ---
print("Sanity Check: Generating sample response")
FastLanguageModel.for_inference(model)
test_prompt = ALPACA_TEMPLATE.format(
    "Write a paragraph to introduce supervised fine-tuning."
)
inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
text_streamer = TextStreamer(tokenizer)
print("Response:")
_ = model.generate(
    **inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True
)

# --- 9. Save LoRA adapter ---
print(f"Saving DPO LoRA adapter: {DPO_ADAPTER_NAME}")
model.save_pretrained_merged(DPO_ADAPTER_NAME, tokenizer, save_method="lora")
model.push_to_hub_merged(DPO_ADAPTER_NAME, tokenizer, save_method="lora")
print(f"Pushed: {DPO_ADAPTER_NAME}")

# --- 10. Save merged 16-bit model ---
print(f"Saving merged model: {DPO_MERGED_NAME}")
model.save_pretrained_merged(
    "dpo_merged_model", tokenizer, save_method="merged_16bit"
)
model.push_to_hub_merged(DPO_MERGED_NAME, tokenizer, save_method="merged_16bit")
print(f"Pushed: {DPO_MERGED_NAME}")

print("=" * 60)
print("DPO TRAINING COMPLETE")
print(f"  LoRA adapter: {DPO_ADAPTER_NAME}")
print(f"  Merged model: {DPO_MERGED_NAME}")
print(f"  Comet ML: check llm-twin project")
print("=" * 60)
