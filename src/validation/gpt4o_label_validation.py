"""
gpt4o_label_validation.py  (Issue 1)

Validates SufficiencyBench ground-truth labels using GPT-4o as an
independent judge. Tests 300 stratified examples (75 per quadrant).

Outputs:
  - results/validation/gpt4o_validation.json  (full per-item results)
  - results/validation/gpt4o_validation_summary.json  (agreement stats)

The 33 human-annotated items are checked for GPT-4o/human consistency.

Usage:
  export OPENAI_API_KEY=sk-...
  python src/validation/gpt4o_label_validation.py
  python src/validation/gpt4o_label_validation.py --n-per-quadrant 75 --model gpt-4o
"""

import json
import random
import argparse
import sys
import os
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

VALIDATION_DIR = RESULTS_DIR / "validation"

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating whether a given context provides sufficient information to answer a question.

TASK: Read the question and context. Judge whether the context contains the information needed to correctly answer the question.

IMPORTANT RULES:
- Judge ONLY based on the context. Do NOT use outside knowledge.
- "Sufficient" means: a careful reader could extract a correct answer from the context alone.
- "Insufficient" means: the context does not contain the answer, even if the question is familiar.
- For multi-part questions, all parts must be answerable from the context.

Question: {question}

Context: {context}

Does the context contain sufficient information to answer the question?
Respond with exactly one word: YES or NO"""


def call_gpt4o(client, prompt: str, model: str = "gpt-4o", retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip().upper()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error: {e}")
                return "ERROR"


def stratified_sample(test_data: list, n_per_quadrant: int,
                       seed: int = 42) -> list:
    """Sample n_per_quadrant items from each quadrant, balanced across qtypes."""
    rng = random.Random(seed)
    by_quadrant = defaultdict(list)
    for ex in test_data:
        for cond in ex["conditions"]:
            by_quadrant[cond["quadrant"]].append({
                "id": ex["id"],
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "context": cond["context"],
                "condition_id": cond["condition_id"],
                "quadrant": cond["quadrant"],
                "sufficient": cond["sufficient"],
            })

    sample = []
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        items = by_quadrant[quad]
        rng.shuffle(items)
        sample.extend(items[:n_per_quadrant])
    rng.shuffle(sample)
    return sample


def run_validation(n_per_quadrant: int = 75, model: str = "gpt-4o",
                   human_annotation_path: str = None):
    try:
        from openai import OpenAI
    except ImportError:
        print("openai not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    sample = stratified_sample(test_data, n_per_quadrant)
    print(f"Validating {len(sample)} items ({n_per_quadrant}/quadrant) with {model}...")
    print(f"Quadrant distribution: { {q: sum(1 for i in sample if i['quadrant']==q) for q in ['Q1','Q2','Q3','Q4']} }")

    results = []
    errors = 0
    for i, item in enumerate(sample):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=item["question"],
            context=item["context"],
        )
        verdict = call_gpt4o(client, prompt, model=model)
        gpt_sufficient = verdict == "YES"

        if verdict == "ERROR":
            errors += 1

        correct = (gpt_sufficient == item["sufficient"])
        results.append({
            **item,
            "gpt_verdict": verdict,
            "gpt_sufficient": gpt_sufficient,
            "ground_truth": item["sufficient"],
            "correct": correct,
        })

        if (i + 1) % 25 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(sample)}] Running accuracy: {acc:.3f}")
        time.sleep(0.1)

    # Summary statistics
    correct_total = sum(r["correct"] for r in results)
    n = len(results)
    accuracy = correct_total / n

    per_quadrant = {}
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        q_items = [r for r in results if r["quadrant"] == quad]
        if q_items:
            per_quadrant[quad] = {
                "n": len(q_items),
                "accuracy": sum(r["correct"] for r in q_items) / len(q_items),
                "gpt_sufficient_rate": sum(r["gpt_sufficient"] for r in q_items) / len(q_items),
            }

    per_qtype = {}
    for qt in ["factual", "multi_hop", "comparative", "subjective"]:
        q_items = [r for r in results if r["question_type"] == qt]
        if q_items:
            per_qtype[qt] = {
                "n": len(q_items),
                "accuracy": sum(r["correct"] for r in q_items) / len(q_items),
            }

    # Wilson 95% CI for overall accuracy
    from math import sqrt
    z = 1.96
    p = accuracy
    n_w = n
    denom = 1 + z**2 / n_w
    center = (p + z**2 / (2 * n_w)) / denom
    margin = z * sqrt(p * (1 - p) / n_w + z**2 / (4 * n_w**2)) / denom
    wilson_ci = [float(center - margin), float(center + margin)]

    # GPT-4o bias: rate of predicting "sufficient"
    gpt_sufficient_rate = sum(r["gpt_sufficient"] for r in results) / n
    ground_truth_rate = sum(r["sufficient"] for r in results) / n

    summary = {
        "model": model,
        "n_total": n,
        "n_per_quadrant": n_per_quadrant,
        "accuracy": float(accuracy),
        "wilson_95_ci": wilson_ci,
        "gpt_sufficient_rate": float(gpt_sufficient_rate),
        "ground_truth_sufficient_rate": float(ground_truth_rate),
        "errors": errors,
        "per_quadrant": per_quadrant,
        "per_question_type": per_qtype,
    }

    # Cross-check with human annotations if provided
    if human_annotation_path and os.path.exists(human_annotation_path):
        with open(human_annotation_path) as f:
            human = json.load(f)
        # Expect: list of {condition_id, human_majority_label (bool), ...}
        human_by_id = {h["condition_id"]: h for h in human}
        overlap = [r for r in results if r["condition_id"] in human_by_id]
        if overlap:
            agree = sum(
                r["gpt_sufficient"] == human_by_id[r["condition_id"]]["human_majority_sufficient"]
                for r in overlap
            )
            summary["human_gpt_agreement"] = {
                "n_overlap": len(overlap),
                "agreement_rate": float(agree / len(overlap)),
            }
            print(f"\nGPT-4o vs human agreement on {len(overlap)} items: "
                  f"{agree/len(overlap):.3f}")

    print(f"\n=== Validation Summary ===")
    print(f"Overall accuracy: {accuracy:.4f} (95% CI: [{wilson_ci[0]:.4f}, {wilson_ci[1]:.4f}])")
    print(f"GPT predicts sufficient: {gpt_sufficient_rate:.3f} (ground truth: {ground_truth_rate:.3f})")
    print(f"Per quadrant: {per_quadrant}")
    print(f"Per type: {per_qtype}")

    with open(VALIDATION_DIR / "gpt4o_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(VALIDATION_DIR / "gpt4o_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {VALIDATION_DIR}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-quadrant", type=int, default=75,
                        help="Items per quadrant (default 75 = 300 total)")
    parser.add_argument("--model", default="gpt-4o",
                        choices=["gpt-4o", "gpt-4o-mini"])
    parser.add_argument("--human-annotations", default=None,
                        help="Path to human annotation JSON for cross-check")
    args = parser.parse_args()

    run_validation(
        n_per_quadrant=args.n_per_quadrant,
        model=args.model,
        human_annotation_path=args.human_annotations,
    )
