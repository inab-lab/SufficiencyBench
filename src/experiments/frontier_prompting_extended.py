"""
frontier_prompting_extended.py  (Issue 12)

Tests additional prompting strategies on GPT-4o and Claude Opus 4
to exhaustively test whether any behavioral approach can access
the sufficiency signal.

Strategies:
  1. Few-shot (4 labeled examples, one per quadrant)
  2. Self-consistency (5 verdicts at temp=0.7, majority vote)
  3. Structured reasoning ("identify what's needed, check if present, judge")
  4. Contrastive prompt ("answer with/without context, rate change")
  5. Contrastive CoT (thinking through both answers explicitly)

Expected result: all strategies plateau at ~0.73 AUROC, confirming
the structural limitation is not a prompting artifact.

Usage:
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...
  python src/experiments/frontier_prompting_extended.py --model gpt-4o
  python src/experiments/frontier_prompting_extended.py --all
"""

import json
import numpy as np
import argparse
import sys
import os
import time
import re
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

API_RESULTS_DIR = RESULTS_DIR / "api_models"
N_EXAMPLES = 500
RANDOM_SEED = 42

# ─── Few-shot examples (one per quadrant, hand-crafted) ────────────────────────
FEW_SHOT_EXAMPLES = [
    # Q1: sufficient + model knows
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital city is Paris, "
                   "which is also its largest city and a major European cultural center.",
        "label": "YES",
        "explanation": "The context explicitly states Paris is the capital.",
    },
    # Q2: insufficient + model knows
    {
        "question": "What is the capital of Germany?",
        "context": "Germany is a country in Central Europe known for its engineering "
                   "and automotive industry. It has a rich cultural history.",
        "label": "NO",
        "explanation": "The context discusses Germany but does not mention its capital Berlin.",
    },
    # Q3: sufficient + model doesn't know
    {
        "question": "What year was the Wilhoit Building constructed?",
        "context": "The Wilhoit Building, located in downtown Joplin, Missouri, "
                   "was constructed in 1921 and is listed on the National Register of Historic Places.",
        "label": "YES",
        "explanation": "The context provides the exact year: 1921.",
    },
    # Q4: insufficient + model doesn't know
    {
        "question": "What is the mascot of the Harleysville Savings Bank?",
        "context": "Harleysville Savings Bank was a community bank founded in Pennsylvania. "
                   "It served local businesses and residents for many decades.",
        "label": "NO",
        "explanation": "The context mentions the bank but does not state its mascot.",
    },
]

JUDGE_BASE = (
    "Does the context contain sufficient information to answer the question? "
    "Answer YES or NO only."
)


def build_few_shot_prompt(question: str, context: str) -> str:
    examples = "\n\n".join(
        f"Question: {ex['question']}\nContext: {ex['context']}\n"
        f"Answer: {ex['label']} ({ex['explanation']})"
        for ex in FEW_SHOT_EXAMPLES
    )
    return (
        f"You judge whether a context is sufficient to answer a question.\n\n"
        f"Examples:\n{examples}\n\n"
        f"Now judge:\nQuestion: {question}\nContext: {context}\n"
        f"Answer: (YES or NO)"
    )


def build_structured_prompt(question: str, context: str) -> str:
    return (
        f"Judge whether the context contains sufficient information to answer the question.\n\n"
        f"Step 1 — What does the question require? Identify the specific information needed.\n"
        f"Step 2 — Does the context contain that information? Check explicitly.\n"
        f"Step 3 — Final judgment: YES or NO.\n\n"
        f"Question: {question}\nContext: {context}\n\n"
        f"Step 1:"
    )


def build_contrastive_prompt(question: str, context: str) -> str:
    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Task: Would you answer this question differently with vs. without the context above?\n"
        f"First, what would you answer WITH the context? (one sentence)\n"
        f"Second, what would you answer WITHOUT the context (from memory)? (one sentence)\n"
        f"Third, based on this comparison, does the context contain the answer? YES or NO.\n\n"
        f"With context:"
    )


def extract_yes_no(text: str) -> str:
    text = text.strip().upper()
    if "YES" in text[:20]:
        return "YES"
    if "NO" in text[:20]:
        return "NO"
    # Look anywhere in response
    if re.search(r'\bYES\b', text):
        return "YES"
    if re.search(r'\bNO\b', text):
        return "NO"
    return "UNCLEAR"


def call_openai(client, prompt: str, model: str = "gpt-4o",
                temperature: float = 0.0, max_tokens: int = 150,
                retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "ERROR"


def call_anthropic(client, prompt: str, model: str = "claude-opus-4-5",
                   temperature: float = 0.0, max_tokens: int = 150,
                   retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "ERROR"


def load_sample(n: int = N_EXAMPLES, seed: int = RANDOM_SEED) -> list:
    """Load stratified sample (same as used for existing API results)."""
    rng = np.random.default_rng(seed)
    with open(BENCH_DIR / "test.json") as f:
        data = json.load(f)

    flat = []
    for ex in data:
        for cond in ex["conditions"]:
            flat.append({
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "context": cond["context"],
                "condition_id": cond["condition_id"],
                "quadrant": cond["quadrant"],
                "sufficient": cond["sufficient"],
                "label": int(cond["sufficient"]),
            })

    # Stratified: n/4 per quadrant
    per_quad = n // 4
    sample = []
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        q_items = [x for x in flat if x["quadrant"] == quad]
        idx = rng.choice(len(q_items), size=min(per_quad, len(q_items)),
                         replace=False)
        sample.extend([q_items[i] for i in idx])
    return sample


def run_strategy(items: list, strategy: str, call_fn, model_name: str) -> dict:
    print(f"  Strategy: {strategy} ({len(items)} items)...")
    scores, labels = [], []
    errors = 0

    for i, item in enumerate(items):
        q, ctx, label = item["question"], item["context"], item["label"]

        if strategy == "few_shot":
            prompt = build_few_shot_prompt(q, ctx)
            resp = call_fn(prompt, temperature=0.0)
            verdict = extract_yes_no(resp)
            score = 1.0 if verdict == "YES" else (0.5 if verdict == "UNCLEAR" else 0.0)

        elif strategy == "self_consistency":
            base_prompt = (
                f"Context: {ctx}\n\nQuestion: {q}\n\n{JUDGE_BASE}"
            )
            verdicts = []
            for _ in range(5):
                resp = call_fn(base_prompt, temperature=0.7)
                v = extract_yes_no(resp)
                verdicts.append(v)
                time.sleep(0.05)
            yes_count = verdicts.count("YES")
            score = yes_count / 5.0

        elif strategy == "structured":
            prompt = build_structured_prompt(q, ctx)
            resp = call_fn(prompt, temperature=0.0, max_tokens=300)
            verdict = extract_yes_no(resp)
            score = 1.0 if verdict == "YES" else (0.5 if verdict == "UNCLEAR" else 0.0)

        elif strategy == "contrastive":
            prompt = build_contrastive_prompt(q, ctx)
            resp = call_fn(prompt, temperature=0.0, max_tokens=200)
            verdict = extract_yes_no(resp)
            score = 1.0 if verdict == "YES" else (0.5 if verdict == "UNCLEAR" else 0.0)

        elif strategy == "baseline_judge":
            prompt = f"Context: {ctx}\n\nQuestion: {q}\n\n{JUDGE_BASE}"
            resp = call_fn(prompt, temperature=0.0)
            verdict = extract_yes_no(resp)
            score = 1.0 if verdict == "YES" else (0.5 if verdict == "UNCLEAR" else 0.0)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if resp == "ERROR":
            errors += 1
            score = 0.5

        scores.append(score)
        labels.append(label)
        time.sleep(0.1)

        if (i + 1) % 50 == 0:
            if len(np.unique(labels)) > 1:
                auroc = roc_auc_score(labels, scores)
                print(f"    [{i+1}/{len(items)}] Running AUROC={auroc:.4f}")

    auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
    print(f"    Final AUROC: {auroc:.4f} (errors: {errors})")
    return {"auroc": auroc, "scores": scores, "labels": labels, "errors": errors}


def run_model(model_name: str, strategies: list = None):
    if strategies is None:
        strategies = ["baseline_judge", "few_shot", "self_consistency",
                      "structured", "contrastive"]

    API_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items = load_sample()
    print(f"\n=== Extended Frontier Prompting: {model_name} ({len(items)} items) ===")

    if "gpt" in model_name:
        try:
            from openai import OpenAI
        except ImportError:
            print("pip install openai")
            return
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        def call_fn(prompt, temperature=0.0, max_tokens=150):
            return call_openai(client, prompt, model=model_name,
                               temperature=temperature, max_tokens=max_tokens)
    elif "claude" in model_name:
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            return
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        def call_fn(prompt, temperature=0.0, max_tokens=150):
            return call_anthropic(client, prompt, model=model_name,
                                  temperature=temperature, max_tokens=max_tokens)
    else:
        print(f"Unknown model: {model_name}")
        return

    results = {}
    for strategy in strategies:
        r = run_strategy(items, strategy, call_fn, model_name)
        results[strategy] = r

    out = API_RESULTS_DIR / f"{model_name.replace('/', '_')}_extended.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # Summary table
    print(f"\nSummary for {model_name}:")
    print(f"  {'Strategy':<25s} AUROC")
    for s, r in results.items():
        print(f"  {s:<25s} {r['auroc']:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o",
                        help="Model name: gpt-4o, gpt-4o-mini, claude-opus-4-5")
    parser.add_argument("--all", action="store_true",
                        help="Run all: gpt-4o and claude-opus-4-5")
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline_judge", "few_shot",
                                 "self_consistency", "structured", "contrastive"])
    args = parser.parse_args()

    if args.all:
        for m in ["gpt-4o", "claude-opus-4-5"]:
            run_model(m, args.strategies)
    else:
        run_model(args.model, args.strategies)
