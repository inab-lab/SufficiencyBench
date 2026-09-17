"""
api_model_test.py

Tests whether the sufficiency-confidence confound persists at larger model scales
using API-based models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash).

Runs LLM-judge and verbalized confidence baselines via API calls.

Usage:
  python src/methods/api_model_test.py --provider openai --n_examples 500
  python src/methods/api_model_test.py --provider anthropic --n_examples 500
  python src/methods/api_model_test.py --provider gemini --n_examples 500
"""

import json, os, sys, time, random
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import argparse
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

# API keys are read directly from the environment:
#   export OPENAI_API_KEY=sk-...
#   export ANTHROPIC_API_KEY=sk-ant-...
# Optionally, a .env file at the project root is loaded if python-dotenv is
# installed (real environment variables always take precedence).
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parents[2] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

JUDGE_PROMPT = """You are evaluating whether a given context contains sufficient information to answer a question.

Context: {context}

Question: {question}

Does the context above contain enough information to fully and correctly answer the question? Consider ONLY the information in the context, not your own knowledge.

Answer with only YES or NO."""

CONFIDENCE_PROMPT = """Based on the following context, rate from 0 to 100 how sufficient the context is for answering the question. 0 means the context contains no relevant information, 100 means it fully contains the answer.

Context: {context}

Question: {question}

Score (0-100):"""


def call_openai(prompt, model="gpt-4o-2024-11-20"):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(prompt, model="claude-opus-4-20250514"):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def call_gemini(prompt, model="gemini-2.5-pro"):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gen_model = genai.GenerativeModel(model)
    resp = gen_model.generate_content(prompt)
    return resp.text.strip()


def parse_yes_no(response):
    r = response.upper()[:20]
    if "YES" in r and "NO" not in r:
        return 1.0
    elif "NO" in r:
        return 0.0
    return 0.5


def parse_score(response):
    nums = re.findall(r'\d+', response[:20])
    if nums:
        return min(max(float(nums[0]) / 100, 0), 1)
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


def run_api_test(provider="openai", n_examples=500, seed=42):
    call_fn = {"openai": call_openai, "anthropic": call_anthropic, "gemini": call_gemini}[provider]
    model_names = {
        "openai": "GPT-4o",
        "anthropic": "Claude Opus 4",
        "gemini": "Gemini 2.5 Pro",
    }

    print(f"\n{'='*60}")
    print(f"API MODEL TEST — {model_names[provider]}")
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
                "gold_answer": ex["gold_answer"],
            })

    random.seed(seed)
    if n_examples > 0 and n_examples < len(conditions):
        # Stratified sample
        suf = [c for c in conditions if c["sufficient"]]
        insuf = [c for c in conditions if not c["sufficient"]]
        n_per = n_examples // 2
        conditions = random.sample(suf, min(n_per, len(suf))) + \
                     random.sample(insuf, min(n_per, len(insuf)))
        random.shuffle(conditions)

    print(f"Testing {len(conditions)} examples...")

    judge_scores = []
    conf_scores = []
    errors = 0

    for i, cond in enumerate(tqdm(conditions, desc=f"API ({provider})")):
        try:
            # LLM judge
            judge_resp = call_fn(JUDGE_PROMPT.format(
                context=cond["context"][:2000], question=cond["question"]
            ))
            judge_scores.append(parse_yes_no(judge_resp))

            # Verbalized confidence
            conf_resp = call_fn(CONFIDENCE_PROMPT.format(
                context=cond["context"][:2000], question=cond["question"]
            ))
            conf_scores.append(parse_score(conf_resp))

        except Exception as e:
            errors += 1
            judge_scores.append(0.5)
            conf_scores.append(0.5)
            if errors <= 3:
                print(f"  Error at {i}: {e}")
            if errors > 20:
                print("  Too many errors, stopping.")
                break

        # Rate limiting
        if provider == "anthropic":
            time.sleep(0.5)
        elif provider == "gemini":
            time.sleep(0.2)

    labels = [1 if c["sufficient"] else 0 for c in conditions[:len(judge_scores)]]
    metadata = [{"quadrant": c["quadrant"], "question_type": c["question_type"]}
                for c in conditions[:len(judge_scores)]]

    result_judge = evaluate_method(f"llm_judge_{provider}", judge_scores, labels, metadata)
    result_conf = evaluate_method(f"verbalized_conf_{provider}", conf_scores, labels, metadata)

    result = {
        "provider": provider,
        "model": model_names[provider],
        "n_examples": len(judge_scores),
        "n_errors": errors,
        "llm_judge": result_judge,
        "verbalized_confidence": result_conf,
    }

    print(f"\nResults ({model_names[provider]}):")
    print(f"  LLM Judge AUROC: {result_judge['overall'].get('auroc', 'N/A')}")
    print(f"  Verbalized Conf AUROC: {result_conf['overall'].get('auroc', 'N/A')}")
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        jq = result_judge["per_quadrant"].get(quad, {})
        print(f"  Judge {quad}: acc={jq.get('accuracy', 'N/A')}")

    out_dir = RESULTS_DIR / "api_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{provider}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="gemini",
                        choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--n_examples", type=int, default=500)
    args = parser.parse_args()
    run_api_test(provider=args.provider, n_examples=args.n_examples)
