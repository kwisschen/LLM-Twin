"""
Answer generation for model evaluation.
Runs inside the llm-training container on EdgeXpert.

Uses transformers + AutoModelForCausalLM for generation.
DO NOT use vLLM — it overwrites NGC's custom PyTorch build.

Usage:
    python generate_answers.py --model-index 0  # Run one model at a time
    python generate_answers.py --model-index 1
    python generate_answers.py --model-index 2
"""

import argparse
import gc

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants (duplicated here to avoid import issues inside container)
MODEL_IDS = [
    "kwisschen/TwinLlama-3.1-8B",
    "kwisschen/TwinLlama-3.1-8B-DPO",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

# Use pre-merged 16-bit models — avoids loading base + adapter separately
MERGED_MODEL_MAP = {
    "kwisschen/TwinLlama-3.1-8B": "kwisschen/TwinLlama-3.1-8B-Merged",
    "kwisschen/TwinLlama-3.1-8B-DPO": "kwisschen/TwinLlama-3.1-8B-DPO-Merged",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# Tokenizer source — merged models were pushed with transformers 5.2.0 whose
# tokenizer config is incompatible with the container's 4.57.6.  The tokenizer
# is unchanged by LoRA fine-tuning, so we always load from the base model.
TOKENIZER_MODEL = "meta-llama/Meta-Llama-3.1-8B"

DATASET_NAME = "kwisschen/llmtwin"
DATASET_SPLIT = "test"
HF_NAMESPACE = "kwisschen"

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(sample):
    """Apply Alpaca template to instruction."""
    return {"prompt": ALPACA_TEMPLATE.format(instruction=sample["instruction"])}


def generate_answers(model_id: str, effective_model_id: str, prompts: list[str]) -> list[str]:
    """Generate answers using transformers."""
    print(f"Loading tokenizer: {TOKENIZER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print(f"Loading model: {effective_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        effective_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    answers = []

    # Process in batches to manage memory
    batch_size = 4  # Small batches to minimize padding overhead on variable-length prompts
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            answer = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            answers.append(answer)

        print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")

    # Free VRAM
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-index", type=int, required=True,
        help="Index into MODEL_IDS (0=SFT, 1=DPO, 2=Instruct baseline)"
    )
    args = parser.parse_args()

    model_id = MODEL_IDS[args.model_index]
    effective_model_id = MERGED_MODEL_MAP[model_id]
    print(f"=== Generating answers for: {model_id} ===")
    print(f"    Using model: {effective_model_id}")

    # Load dataset and format prompts
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    dataset = dataset.map(format_prompt)
    prompts = dataset["prompt"]
    print(f"Loaded {len(prompts)} test prompts")

    # Generate
    answers = generate_answers(model_id, effective_model_id, prompts)

    # Add answers to dataset and push
    if "answers" in dataset.column_names:
        dataset = dataset.remove_columns(["answers"])
    dataset = dataset.add_column("answers", answers)

    repo_name = model_id.split("/")[-1]
    hub_id = f"{HF_NAMESPACE}/{repo_name}-results"
    print(f"Pushing results to {hub_id}")
    try:
        dataset.push_to_hub(hub_id)
    except ValueError:
        from huggingface_hub import HfApi
        print(f"Schema mismatch — deleting {hub_id} and re-pushing")
        HfApi().delete_repo(hub_id, repo_type="dataset")
        dataset.push_to_hub(hub_id)
    print(f"=== Done: {model_id} ===")


if __name__ == "__main__":
    main()
