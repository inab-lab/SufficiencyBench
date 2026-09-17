"""
hotpotqa_behavioral_eval.py

Evaluates behavioral detection methods on HotpotQA entity-linked distractors.
Fills the gap left by hotpotqa_hard_neg_eval.py (which only ran probes).

Methods evaluated:
  - LLM self-judge (YES/NO)
  - Verbalized confidence (0-100 scale)
  - Token entropy (next-token distribution entropy)

Uses the same 300-pair sample (seed=42) as the probe eval so results are
directly comparable.

Usage:
  python src/evaluation/hotpotqa_behavioral_eval.py --model llama --device cuda:0
  python src/evaluation/hotpotqa_behavioral_eval.py --all --device cuda:0
"""

import json
import re
import random
import argparse
import sys
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import RESULTS_DIR, MODEL_CONFIGS

MAX_CONTEXT_CHARS = 1500
N_PAIRS = 300
SEED = 42  # identical to hotpotqa_hard_neg_eval.py

JUDGE_PROMPT = (
    "You are evaluating whether a context contains sufficient information "
    "to answer a question.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Does the context above contain enough information to fully and correctly "
    "answer the question? Consider ONLY the information in the context, not "
    "your own knowledge.\n\nAnswer with only YES or NO."
)

VERBCONF_PROMPT = (
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Before answering, rate from 0 to 100: does the context contain enough "
    "information to answer the question? Consider ONLY the information in "
    "the context.\nReply with ONLY the number.\nScore:"
)

QA_PROMPT = (
    "Based on the following context, answer the question. "
    "If the context does not contain enough information, say "
    "'I cannot answer this from the provided context.'\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)


def build_hotpotqa_pairs(n_pairs=N_PAIRS, seed=SEED):
    print("  Loading HotpotQA distractor split (validation)...")
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)["validation"]
    rng = random.Random(seed)
    pairs, skipped = [], 0
    for ex in ds:
        if len(pairs) >= n_pairs:
            break
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        supporting = set(ex["supporting_facts"]["title"])
        gold_parts, distractor_parts = [], []
        for title, sents in zip(titles, sentences):
            para = " ".join(sents).strip()
            if not para:
                continue
            (gold_parts if title in supporting else distractor_parts).append(para)
        if not gold_parts or not distractor_parts:
            skipped += 1
            continue
        pairs.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "sufficient_context": " ".join(gold_parts)[:MAX_CONTEXT_CHARS],
            "insufficient_context": rng.choice(distractor_parts)[:MAX_CONTEXT_CHARS],
        })
    print(f"  Built {len(pairs)} pairs (skipped {skipped})")
    return pairs


def llm_judge_score(model, tokenizer, context, question, device):
    prompt = JUDGE_PROMPT.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5,
                             temperature=0.0, do_sample=False)
    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().upper()
    if resp.startswith("YES"):
        return 1.0
    elif resp.startswith("NO"):
        return 0.0
    return 0.5


def verbconf_score(model, tokenizer, context, question, device):
    prompt = VERBCONF_PROMPT.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5,
                             temperature=0.0, do_sample=False)
    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    nums = re.findall(r'\d+', resp)
    if nums:
        return min(max(float(nums[0]) / 100.0, 0.0), 1.0)
    return 0.5


def token_entropy_score(model, tokenizer, context, question, device):
    prompt = QA_PROMPT.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return -entropy  # negative entropy: higher = more confident = predicts sufficient


def run_eval(model_key, device="cuda:0"):
    print(f"\n{'='*60}")
    print(f"HOTPOTQA BEHAVIORAL EVAL — {model_key.upper()}")
    print(f"{'='*60}")

    pairs = build_hotpotqa_pairs()

    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    method_scores = {"judge": [], "verbconf": [], "token_entropy": []}
    labels = []

    for i, pair in enumerate(tqdm(pairs, desc=f"[{model_key}]")):
        for ctx_key, label in [("sufficient_context", 1), ("insufficient_context", 0)]:
            ctx = pair[ctx_key]
            q = pair["question"]
            method_scores["judge"].append(
                llm_judge_score(model, tokenizer, ctx, q, device))
            method_scores["verbconf"].append(
                verbconf_score(model, tokenizer, ctx, q, device))
            method_scores["token_entropy"].append(
                token_entropy_score(model, tokenizer, ctx, q, device))
            labels.append(label)

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    labels_arr = np.array(labels)
    results = {
        "model": model_key,
        "dataset": "hotpotqa_distractor",
        "n_pairs": len(pairs),
        "methods": {},
    }
    for name, scores in method_scores.items():
        auroc = roc_auc_score(labels_arr, np.array(scores))
        results["methods"][name] = {"auroc": float(auroc), "n": int(len(labels))}
        print(f"  {name:16s}: AUROC={auroc:.4f}")

    out = RESULTS_DIR / model_key / "hotpotqa_behavioral_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true",
                        help="Run all 3 models sequentially")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.all else [args.model]
    all_results = []
    for m in models:
        r = run_eval(m, args.device)
        all_results.append(r)

    print("\n--- Summary (AUROC on HotpotQA hard negatives) ---")
    for r in all_results:
        print(f"{r['model']:10s}: judge={r['methods']['judge']['auroc']:.4f}  "
              f"verbconf={r['methods']['verbconf']['auroc']:.4f}  "
              f"token_entropy={r['methods']['token_entropy']['auroc']:.4f}")
