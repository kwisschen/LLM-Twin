"""
SFT Fine-tuning: Llama 3.1 8B -> TwinLlama (LoRA)
Dataset: kwisschen/llmtwin (19,124 train + 2,127 test)
Hardware: NVIDIA GB10 Blackwell, 128GB unified memory
"""

import os

# Must be set before any torch/unsloth imports
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Unsloth MUST be imported before trl, transformers, peft
# so that its monkey-patches are applied to all downstream libraries
import unsloth  # noqa: F401

from collections import Counter

from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET_ID = "kwisschen/llmtwin"
OUTPUT_DIR = "/workspace/outputs/sft_checkpoints"
ADAPTER_OUTPUT_DIR = "/workspace/outputs/sft_lora_adapter"
HF_REPO_ID = "kwisschen/TwinLlama-3.1-8B"  # Where to push LoRA adapter

# LoRA hyperparameters
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 2 * 8 = 16
LEARNING_RATE = 3e-4
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 1
SEED = 0

# Alpaca prompt template (must match dataset generation format)
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Response:\n{}"
)

# ============================================================
# 1. Load Model + Inject LoRA
# ============================================================
print("=" * 60)
print("Loading base model...")
print("  Model:", BASE_MODEL)
print("  Max seq length:", MAX_SEQ_LENGTH)
print("  LoRA rank:", LORA_RANK, " alpha:", LORA_ALPHA)
print("  Target modules:", TARGET_MODULES)
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,  # Full BF16 -- we have 128GB, speed > memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
)

EOS_TOKEN = tokenizer.eos_token
print("EOS token: '{}' (id: {})".format(EOS_TOKEN, tokenizer.eos_token_id))

# ============================================================
# 2. Load & Format Dataset
# ============================================================
print("\nLoading dataset from HuggingFace Hub...")
dataset_train = load_dataset(DATASET_ID, split="train")
dataset_test = load_dataset(DATASET_ID, split="test")
print("  Train: {} samples".format(len(dataset_train)))
print("  Test:  {} samples".format(len(dataset_test)))

# Print category distribution
if "category" in dataset_train.column_names:
    cats = Counter(dataset_train["category"])
    print("  Categories:", dict(cats))


def format_alpaca(examples):
    """Format instruction/output pairs into Alpaca template + EOS token."""
    texts = []
    for instruction, output in zip(
        examples["instruction"], examples["output"], strict=False
    ):
        text = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset_train = dataset_train.map(
    format_alpaca,
    batched=True,
    remove_columns=dataset_train.column_names,
    num_proc=8,
)
dataset_test = dataset_test.map(
    format_alpaca,
    batched=True,
    remove_columns=dataset_test.column_names,
    num_proc=8,
)

print("\nFormatted sample (first 300 chars):")
print(dataset_train[0]["text"][:300])
print("...")

# ============================================================
# 3. Configure Trainer
# ============================================================
effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
total_steps = (len(dataset_train) * NUM_EPOCHS) // effective_batch
print("\nTraining configuration:")
print("  Epochs:", NUM_EPOCHS)
print(
    "  Batch size: {} x {} accum = {} effective".format(
        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, effective_batch
    )
)
print("  Estimated total steps:", total_steps)
print("  Learning rate:", LEARNING_RATE, "(linear decay)")
print("  Packing: True")
print("  BF16:", is_bfloat16_supported())
print("  Reporting to: comet_ml")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        # Dataset params (moved to SFTConfig in TRL 0.26+)
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=8,
        packing=True,
        # EOS handling: we already append EOS in format_alpaca,
        # so disable SFTConfig's automatic EOS insertion
        eos_token=None,
        # Training params
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="comet_ml",
        seed=SEED,
    ),
)

# ============================================================
# 4. Train
# ============================================================
print("\n" + "=" * 60)
print("Starting SFT training...")
print("=" * 60)

trainer.train()

# ============================================================
# 5. Smoke Test -- Quick Inference
# ============================================================
print("\n" + "=" * 60)
print("Smoke test: generating sample output...")
print("=" * 60)

model = FastLanguageModel.for_inference(model)
test_prompt = (
    "Write a paragraph about the importance of data quality "
    "in machine learning pipelines."
)
message = ALPACA_TEMPLATE.format(test_prompt, "")
inputs = tokenizer([message], return_tensors="pt").to("cuda")
text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    **inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True
)

# ============================================================
# 6. Save LoRA Adapter + Push to Hub
# ============================================================
print("\n" + "=" * 60)
print("Saving LoRA adapter...")
print("=" * 60)

# Save adapter locally (small -- ~168MB)
model.save_pretrained(ADAPTER_OUTPUT_DIR)
tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
print("  Saved to:", ADAPTER_OUTPUT_DIR)

# Push adapter to HuggingFace Hub
print("  Pushing to:", HF_REPO_ID)
model.push_to_hub(HF_REPO_ID)
tokenizer.push_to_hub(HF_REPO_ID)

# Also save merged 16-bit for direct inference
MERGED_OUTPUT_DIR = "/workspace/outputs/sft_merged"
print("  Saving merged 16-bit model to:", MERGED_OUTPUT_DIR)
model.save_pretrained_merged(
    MERGED_OUTPUT_DIR, tokenizer, save_method="merged_16bit"
)

# Push merged model to a separate repo
MERGED_HF_REPO_ID = "kwisschen/TwinLlama-3.1-8B-Merged"
print("  Pushing merged model to:", MERGED_HF_REPO_ID)
model.push_to_hub_merged(
    MERGED_HF_REPO_ID, tokenizer, save_method="merged_16bit"
)

print("\n" + "=" * 60)
print("SFT training complete!")
print("  LoRA adapter: ", HF_REPO_ID)
print("  Merged model: ", MERGED_HF_REPO_ID)
print("=" * 60)
