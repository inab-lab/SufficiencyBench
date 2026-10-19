"""
api_cot_test.py

Tests whether chain-of-thought reasoning helps frontier models detect sufficiency.
Uses the same examples as the standard API test but with CoT prompting.

Usage:
  python src/methods/api_cot_test.py --provider openai --n_examples 500
  python src/methods/api_cot_test.py --provider anthropic --n_examples 500
"""

import json, os, sys, time, random, re
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

from dotenv import load_dotenv
load_dotenv("/home/inab/Documents/synthetic_questionnaire/project-a-algorithmic-fidelity/.env")

COT_PROMPT = """You are evaluating whether a given context contains sufficient information to answer a question.

Context: {context}

Question: {question}

Think step by step:
1. What information would be needed to answer this question?
2. Does the context contain that specific information?
3. Am I relying on my own knowledge, or only on what's in the context?

Important: Consider ONLY the information explicitly stated in the context. Do NOT use your own knowledge to fill gaps.

After your reasoning, conclude with exactly one word on its own line: SUFFICIENT or INSUFFICIENT"""


def call_openai_cot(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic_cot(prompt):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def parse_cot_response(response):
    """Extract SUFFICIENT/INSUFFICIENT from CoT response."""
    lines = response.strip().split('\n')
    # Check last few lines for the verdict
    for line in reversed(lines[-5:]):
        line_upper = line.strip().upper()
        if 'INSUFFICIENT' in line_upper:
            return 0.0
        if 'SUFFICIENT' in line_upper:
            return 1.0
    # Fallback: search whole response
    r = response.upper()
    last_suf = r.rfind('SUFFICIENT')
    last_insuf = r.rfind('INSUFFICIENT')
    if last_insuf > last_suf:
        return 0.0
    elif last_suf > -1:
        return 1.0
    return 0.5


def evaluate_method(name, scores, labels, metadata):
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    result = {"method": name, "overall": {}, "per_quadrant": {}}

    if len(set(labels_arr)) > 1:
        result["overall"]["auroc"] = float(roc_auc_score(labels_arr, scores_arr))
    preds = (scores_arr > 0.5).astype(int)
    result["overall"]["accuracy"] = float(accuracy_score(labels_arr, preds))
    result["overall"]["f1"] = float(f1_score(labels_arr, preds, zero_division=0))
    result["overall"]["n"] = len(labels_arr)

    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(metadata) if m["quadrant"] == quad]
        if len(idx) < 5:
            continue
        q_labels = labels_arr[idx]
        q_scores = scores_arr[idx]
        q_preds = (q_scores > 0.5).astype(int)
        entry = {"n": len(idx), "accuracy": float(accuracy_score(q_labels, q_preds))}
        if len(set(q_labels)) > 1:
            entry["auroc"] = float(roc_auc_score(q_labels, q_scores))
        result["per_quadrant"][quad] = entry

    return result


def run_cot_test(provider="openai", n_examples=500, seed=42):
    call_fn = {"openai": call_openai_cot, "anthropic": call_anthropic_cot}[provider]
    model_names = {"openai": "GPT-4o + CoT", "anthropic": "Claude Opus 4 + CoT"}

    print(f"\n{'='*60}")
    print(f"CHAIN-OF-THOUGHT TEST — {model_names[provider]}")
    print(f"{'='*60}")

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question": ex["question"],
                "question_type": ex["question_type"],
            })

    random.seed(seed)
    if n_examples > 0 and n_examples < len(conditions):
        suf = [c for c in conditions if c["sufficient"]]
        insuf = [c for c in conditions if not c["sufficient"]]
        n_per = n_examples // 2
        conditions = random.sample(suf, min(n_per, len(suf))) + \
                     random.sample(insuf, min(n_per, len(insuf)))
        random.shuffle(conditions)

    print(f"Testing {len(conditions)} examples with CoT...")

    scores = []
    errors = 0

    for i, cond in enumerate(tqdm(conditions, desc=f"CoT ({provider})")):
        try:
            prompt = COT_PROMPT.format(
                context=cond["context"][:2000],
                question=cond["question"]
            )
            response = call_fn(prompt)
            score = parse_cot_response(response)
            scores.append(score)
        except Exception as e:
            errors += 1
            scores.append(0.5)
            if errors <= 3:
                print(f"  Error at {i}: {e}")
            if errors > 20:
                print("  Too many errors, stopping.")
                break

        if provider == "anthropic":
            time.sleep(0.5)

    labels = [1 if c["sufficient"] else 0 for c in conditions[:len(scores)]]
    metadata = [{"quadrant": c["quadrant"], "question_type": c["question_type"]}
                for c in conditions[:len(scores)]]

    result = evaluate_method(f"cot_judge_{provider}", scores, labels, metadata)
    result["config"] = {
        "provider": provider,
        "model": model_names[provider],
        "n_examples": len(scores),
        "n_errors": errors,
        "prompt_type": "chain_of_thought",
    }

    # Load standard (non-CoT) results for comparison
    std_file = RESULTS_DIR / "api_models" / f"{'openai' if provider == 'openai' else 'anthropic'}_results.json"
    if std_file.exists():
        with open(std_file) as f:
            std = json.load(f)
        std_auroc = std["llm_judge"]["overall"].get("auroc", "N/A")
        result["comparison"] = {
            "standard_judge_auroc": std_auroc,
            "cot_judge_auroc": result["overall"].get("auroc", "N/A"),
            "improvement": (result["overall"].get("auroc", 0) - std_auroc) if isinstance(std_auroc, float) else "N/A",
        }

    print(f"\nResults ({model_names[provider]}):")
    print(f"  CoT Judge AUROC: {result['overall'].get('auroc', 'N/A')}")
    if "comparison" in result:
        print(f"  Standard Judge AUROC: {result['comparison']['standard_judge_auroc']}")
        print(f"  Improvement: {result['comparison'].get('improvement', 'N/A')}")
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        qd = result["per_quadrant"].get(quad, {})
        print(f"  {quad}: acc={qd.get('accuracy', 'N/A')}")

    out_dir = RESULTS_DIR / "api_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider}_cot_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--n_examples", type=int, default=500)
    args = parser.parse_args()
    run_cot_test(provider=args.provider, n_examples=args.n_examples)
