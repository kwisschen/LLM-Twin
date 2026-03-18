"""Shared constants and utilities for the evaluation pipeline."""

# Models to evaluate
MODEL_IDS = [
    "kwisschen/TwinLlama-3.1-8B",
    "kwisschen/TwinLlama-3.1-8B-DPO",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

# Merged models (for models that are LoRA adapters, use pre-merged versions)
# These avoid loading base + adapter separately, which is simpler and faster.
MERGED_MODEL_MAP = {
    "kwisschen/TwinLlama-3.1-8B": "kwisschen/TwinLlama-3.1-8B-Merged",
    "kwisschen/TwinLlama-3.1-8B-DPO": "kwisschen/TwinLlama-3.1-8B-DPO-Merged",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# Dataset
DATASET_NAME = "kwisschen/llmtwin"
DATASET_SPLIT = "test"

# HuggingFace namespace for results
HF_NAMESPACE = "kwisschen"

# Alpaca prompt template (must match SFT training template)
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

# Judge model
JUDGE_MODEL = "gpt-4.1-mini"

# Evaluation prompt (from handbook Ch. 7, adapted)
EVALUATION_PROMPT = """You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:

1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:
1 (Poor): Contains factual errors or misleading information
2 (Good): Mostly accurate with minor errors or omissions
3 (Excellent): Highly accurate and comprehensive

Style scale:
1 (Poor): Too formal, uses some overly complex words
2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

Instruction: {instruction}
Answer: {answer}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{
        "analysis": "...",
        "score": 0
    }},
    "style": {{
        "analysis": "...",
        "score": 0
    }}
}}
"""
