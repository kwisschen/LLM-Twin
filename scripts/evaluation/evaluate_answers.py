"""
Evaluate generated answers using GPT-4.1-mini as judge.
Runs on MacBook (no GPU needed — only OpenAI API calls).

Usage:
    poetry run python -m scripts.evaluation.evaluate_answers
"""

import argparse
import json
import os
import time
import concurrent.futures
from typing import Optional

from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm.auto import tqdm
from opik import track, opik_context

from scripts.evaluation.evaluation_utils import (
    MODEL_IDS,
    HF_NAMESPACE,
    JUDGE_MODEL,
    EVALUATION_PROMPT,
)


def evaluate_answer(instruction: str, answer: str, client: OpenAI) -> Optional[str]:
    """Call the judge LLM to evaluate a single answer."""
    prompt = EVALUATION_PROMPT.format(instruction=instruction, answer=answer)

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant who evaluates answers based on "
                            "accuracy and style. Provide your response in JSON format with "
                            "a short analysis and score for each criterion."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.8,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429 and attempt < 2:
                sleep_time = 2**attempt
                print(f"Rate limited, retrying in {sleep_time}s (attempt {attempt + 1}/3)")
                time.sleep(sleep_time)
                continue
            print(f"Error evaluating answer: {e}")
            return None


def evaluate_batch(batch: list[tuple[str, str]], start_index: int) -> list[tuple[int, Optional[str]]]:
    """Evaluate a batch of instruction-answer pairs."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return [
        (i, evaluate_answer(instr, ans, client))
        for i, (instr, ans) in enumerate(batch, start=start_index)
    ]


@track(name="evaluate_model", project_name="llm-twin-monitoring")
def evaluate_answers(model_id: str, num_threads: int = 3, batch_size: int = 5) -> Dataset:
    """Orchestrate parallel evaluation of all answers for a model."""
    repo_name = model_id.split("/")[-1]
    hub_id = f"{HF_NAMESPACE}/{repo_name}-results"

    print(f"Loading results from {hub_id}")
    dataset = load_dataset(hub_id, split="test")

    # Create batches
    batches = [
        (i, list(zip(
            dataset["instruction"][i : i + batch_size],
            dataset["answers"][i : i + batch_size],
        )))
        for i in range(0, len(dataset), batch_size)
    ]

    # Parallel evaluation
    evaluations = [None] * len(dataset)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(evaluate_batch, batch, start_index)
            for start_index, batch in batches
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Evaluating {repo_name}",
        ):
            for index, evaluation in future.result():
                evaluations[index] = evaluation

    # Parse scores from evaluations
    accuracy_scores = []
    style_scores = []
    for evaluation in evaluations:
        try:
            eval_dict = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
            accuracy_scores.append(eval_dict["accuracy"]["score"])
            style_scores.append(eval_dict["style"]["score"])
        except (json.JSONDecodeError, KeyError, TypeError):
            accuracy_scores.append(None)
            style_scores.append(None)

    # Rebuild dataset from scratch to avoid Arrow table concatenation issues
    data = {col: dataset[col] for col in dataset.column_names if col not in ("evaluation", "accuracy", "style")}
    data["evaluation"] = evaluations
    data["accuracy"] = accuracy_scores
    data["style"] = style_scores
    dataset = Dataset.from_dict(data)

    # Log summary to Opik
    valid_acc = [s for s in accuracy_scores if s is not None]
    valid_sty = [s for s in style_scores if s is not None]
    opik_context.update_current_trace(
        tags=["evaluation", repo_name],
        metadata={
            "model_id": model_id,
            "judge_model": JUDGE_MODEL,
            "num_samples": len(dataset),
            "num_evaluated": len(valid_acc),
            "mean_accuracy": sum(valid_acc) / len(valid_acc) if valid_acc else 0,
            "mean_style": sum(valid_sty) / len(valid_sty) if valid_sty else 0,
        },
    )

    # Push to Hub
    print(f"Pushing scored results to {hub_id}")
    try:
        dataset.push_to_hub(hub_id, revision="main")
    except ValueError:
        from huggingface_hub import HfApi
        print(f"Schema mismatch — deleting {hub_id} and re-pushing")
        HfApi().delete_repo(hub_id, repo_type="dataset")
        dataset.push_to_hub(hub_id)

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-index", type=int, default=None,
        help="Index into MODEL_IDS (0=SFT, 1=DPO, 2=Instruct). If omitted, run all.",
    )
    args = parser.parse_args()

    # Initialize Opik
    import opik
    opik.configure(
        api_key=os.environ.get("OPIK_API_KEY"),
    )

    models = [MODEL_IDS[args.model_index]] if args.model_index is not None else MODEL_IDS
    for model_id in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*60}")
        evaluate_answers(model_id)


if __name__ == "__main__":
    main()
