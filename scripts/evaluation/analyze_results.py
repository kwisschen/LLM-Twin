"""
Analyze and compare evaluation results across models.

Usage:
    poetry run python -m scripts.evaluation.analyze_results
"""

from datasets import load_dataset
from scripts.evaluation.evaluation_utils import MODEL_IDS, HF_NAMESPACE


def analyze():
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    results = {}
    for model_id in MODEL_IDS:
        repo_name = model_id.split("/")[-1]
        hub_id = f"{HF_NAMESPACE}/{repo_name}-results"
        try:
            dataset = load_dataset(hub_id, split="train")
        except ValueError:
            dataset = load_dataset(hub_id, split="test")

        valid_acc = [s for s in dataset["accuracy"] if s is not None]
        valid_sty = [s for s in dataset["style"] if s is not None]

        mean_acc = sum(valid_acc) / len(valid_acc) if valid_acc else 0
        mean_sty = sum(valid_sty) / len(valid_sty) if valid_sty else 0

        results[repo_name] = {
            "accuracy": mean_acc,
            "style": mean_sty,
            "n_evaluated": len(valid_acc),
            "n_failed": len(dataset) - len(valid_acc),
        }

        print(f"\n{repo_name}:")
        print(f"  Accuracy: {mean_acc:.2f} (n={len(valid_acc)}, failed={len(dataset) - len(valid_acc)})")
        print(f"  Style:    {mean_sty:.2f} (n={len(valid_sty)}, failed={len(dataset) - len(valid_sty)})")

    # Comparison table
    print("\n" + "-" * 70)
    print("COMPARISON (higher is better, max=3.0)")
    print("-" * 70)
    print(f"{'Model':<35} {'Accuracy':>10} {'Style':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<35} {r['accuracy']:>10.2f} {r['style']:>10.2f}")

    # Key findings for interview narrative
    print("\n" + "-" * 70)
    print("KEY FINDINGS FOR INTERVIEW NARRATIVE")
    print("-" * 70)

    sft = results.get("TwinLlama-3.1-8B", {})
    dpo = results.get("TwinLlama-3.1-8B-DPO", {})
    baseline = results.get("Meta-Llama-3.1-8B-Instruct", {})

    if sft and dpo:
        style_delta = dpo.get("style", 0) - sft.get("style", 0)
        acc_delta = dpo.get("accuracy", 0) - sft.get("accuracy", 0)
        print(f"DPO style improvement over SFT: {style_delta:+.2f}")
        print(f"DPO accuracy delta vs SFT:      {acc_delta:+.2f}")

    if dpo and baseline:
        style_vs_base = dpo.get("style", 0) - baseline.get("style", 0)
        acc_vs_base = dpo.get("accuracy", 0) - baseline.get("accuracy", 0)
        print(f"DPO style vs Instruct baseline:  {style_vs_base:+.2f}")
        print(f"DPO accuracy vs Instruct:        {acc_vs_base:+.2f}")


if __name__ == "__main__":
    analyze()
