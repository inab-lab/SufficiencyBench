"""
llm_judge_baseline.py

LLM-as-judge baseline for sufficiency detection, following Joren et al. (2025).
Prompts the model itself to judge whether the provided context is sufficient
to answer the question (YES/NO).

This is the most directly comparable baseline to our CSP probe method.
Expected behavior: fails on Q3 (insufficient + confident) because the model
uses its parametric knowledge to believe it can answer.

Usage:
  python src/methods/llm_judge_baseline.py --model mistral --device cuda:0
  python src/methods/llm_judge_baseline.py --model llama --device cuda:0
  python src/methods/llm_judge_baseline.py --model qwen --device cuda:0

Batched inference: judge generation runs one generate() call per batch of
--batch-size conditions (default 16), using left-padding so that generated
continuations for every row start at the same offset (input_ids.shape[1]).
Results are numerically identical to the one-at-a-time path under greedy
decoding; verify with --validate.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import argparse
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a given context contains sufficient information to answer a question.

Context: {context}

Question: {question}

Does the context above contain enough information to fully and correctly answer the question? Consider ONLY the information in the context, not your own knowledge.

Answer with only YES or NO."""


def load_model_and_tokenizer(model_key: str, device: str):
    """Load a causal LM in 8-bit quantization."""
    cfg = MODEL_CONFIGS[model_key]
    model_path = cfg["path"]

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for batched decoder-only generation: it keeps
    # every prompt right-aligned so all continuations begin at the same offset
    # (input_ids.shape[1]). For a batch of 1 this is a no-op vs. the old path.
    tokenizer.padding_side = "left"

    print(f"Loading model in 8-bit mode...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def _parse_verdict(response: str) -> tuple[float, str]:
    """
    Map a decoded judge response to (score, raw_response).

    `response` is expected to be the decoded continuation already
    .strip().upper()'d. This is the single source of truth for the verdict
    logic so the batched and single-item paths stay bit-for-bit identical.
    Score: 1.0 = YES (sufficient), 0.0 = NO (insufficient), 0.5 = ambiguous.
    """
    # Parse YES/NO
    if "YES" in response[:10]:
        return 1.0, response
    elif "NO" in response[:10]:
        return 0.0, response
    else:
        # Ambiguous — try harder
        if any(w in response.lower() for w in ["sufficient", "yes", "contains"]):
            return 0.7, response
        elif any(w in response.lower() for w in ["insufficient", "no", "does not", "cannot"]):
            return 0.3, response
        return 0.5, response


def judge_sufficiency(model, tokenizer, context: str, question: str, device: str) -> tuple[float, str]:
    """
    Ask the model to judge whether the context is sufficient (single item).
    Returns (score, raw_response). Kept as the bs=1 reference for --validate.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(context=context, question=question)

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
        )

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip().upper()

    return _parse_verdict(response)


def judge_sufficiency_batch(model, tokenizer, contexts: list, questions: list, device: str) -> tuple[list, list]:
    """
    Batched judge: one generate() call over a batch of (context, question).
    Returns (scores, raw_responses) as lists aligned with the inputs.

    Uses left-padding + attention_mask so all rows' continuations start at
    input_ids.shape[1]; decoding params and verdict logic match the single-item
    path exactly, so under greedy decoding the outputs are numerically identical.
    """
    prompts = [
        JUDGE_PROMPT_TEMPLATE.format(context=c, question=q)
        for c, q in zip(contexts, questions)
    ]

    inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,               # includes attention_mask from the tokenizer
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
        )

    # Left-padding => every prompt is right-aligned to the same width, so the
    # generated continuation for all rows begins at input_ids.shape[1].
    gen = out[:, inputs["input_ids"].shape[1]:]

    scores, raw_responses = [], []
    for row in gen:
        response = tokenizer.decode(
            row, skip_special_tokens=True,
        ).strip().upper()
        score, resp = _parse_verdict(response)
        scores.append(score)
        raw_responses.append(resp)
    return scores, raw_responses


def evaluate_method(method_name: str, scores: list, labels: list, metadata: list) -> dict:
    """Evaluate with per-quadrant and per-question-type breakdown."""
    # Sanitize: replace any non-finite score with the finite mean so a few
    # bad conditions can't NaN-crash roc_auc and lose the whole run.
    scores_arr = np.asarray(scores, dtype=float)
    n_bad = int((~np.isfinite(scores_arr)).sum())
    if n_bad:
        fill = float(np.nanmean(scores_arr[np.isfinite(scores_arr)])) if np.isfinite(scores_arr).any() else 0.0
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, fill)
        print(f"  [warn] {method_name}: imputed {n_bad} non-finite scores with {fill:.4f}")
    labels_arr = np.array(labels)

    result = {
        "method": method_name,
        "overall": {},
        "per_quadrant": {},
        "per_question_type": {},
    }
    result["overall"]["n_imputed"] = n_bad

    if len(set(labels_arr)) > 1:
        try:
            result["overall"]["auroc"] = float(roc_auc_score(labels_arr, scores_arr))
        except Exception as e:
            result["overall"]["error"] = str(e)

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

        entry = {"n": len(idx)}
        if len(set(q_labels)) > 1:
            try:
                entry["auroc"] = float(roc_auc_score(q_labels, q_scores))
            except Exception as e:
                entry["error"] = str(e)
        entry["accuracy"] = float(accuracy_score(q_labels, q_preds))
        entry["f1"] = float(f1_score(q_labels, q_preds, zero_division=0))
        result["per_quadrant"][quad] = entry

    for qtype in ["factual", "multi_hop", "comparative", "subjective"]:
        idx = [i for i, m in enumerate(metadata) if m["question_type"] == qtype]
        if len(idx) < 10 or len(set(labels_arr[idx])) < 2:
            continue
        q_preds = (scores_arr[idx] > 0.5).astype(int)
        qt_entry = {
            "n": len(idx),
            "accuracy": float(accuracy_score(labels_arr[idx], q_preds)),
        }
        try:
            qt_entry["auroc"] = float(roc_auc_score(labels_arr[idx], scores_arr[idx]))
        except Exception as e:
            qt_entry["error"] = str(e)
        result["per_question_type"][qtype] = qt_entry

    return result


def _flatten_conditions(test_data: list) -> list:
    """Flatten per-example conditions into a flat list (order preserved)."""
    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question_type": ex["question_type"],
                "gold_answer": ex["gold_answer"],
                "question": ex["question"],
            })
    return conditions


def validate_batching(model, tokenizer, conditions: list, device: str,
                      batch_size: int = 8, n: int = 16, tol: float = 1e-3):
    """
    Fidelity check: first `n` conditions scored batched (bs=`batch_size`) vs
    unbatched (bs=1). Asserts per-item scores match within `tol` and prints the
    max abs diff plus a per-condition timing comparison. Also reports peak GPU
    memory for a bs=16 batch. Does NOT write results.
    """
    subset = conditions[:n]
    print(f"\n[validate] fidelity: {len(subset)} conditions, "
          f"batched(bs={batch_size}) vs unbatched(bs=1), tol={tol}")

    # --- unbatched reference (bs=1) ---
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    unb_scores = []
    for c in subset:
        s, _ = judge_sufficiency(model, tokenizer, c["context"], c["question"], device)
        unb_scores.append(s)
    torch.cuda.synchronize(device)
    unb_time = time.perf_counter() - t0

    # --- batched ---
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    bat_scores = []
    for start in range(0, len(subset), batch_size):
        b = subset[start:start + batch_size]
        bs, _ = judge_sufficiency_batch(
            model, tokenizer,
            [x["context"] for x in b], [x["question"] for x in b], device,
        )
        bat_scores.extend(bs)
    torch.cuda.synchronize(device)
    bat_time = time.perf_counter() - t0

    diffs = [abs(a - b) for a, b in zip(unb_scores, bat_scores)]
    max_diff = max(diffs) if diffs else 0.0
    print(f"[validate] max abs per-item diff = {max_diff:.6f}")
    print(f"[validate] unbatched: {unb_time:.2f}s ({unb_time/len(subset):.3f}s/cond)  "
          f"batched: {bat_time:.2f}s ({bat_time/len(subset):.3f}s/cond)  "
          f"speedup = {unb_time/bat_time:.2f}x")
    assert max_diff < tol, f"Batching fidelity FAILED: max diff {max_diff} >= {tol}"
    print("[validate] PASS (per-item scores identical within tolerance)")

    # --- peak GPU memory for a bs=16 batch ---
    mem_bs = 16
    mem_subset = conditions[:mem_bs]
    if len(mem_subset) == mem_bs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        _ = judge_sufficiency_batch(
            model, tokenizer,
            [x["context"] for x in mem_subset],
            [x["question"] for x in mem_subset], device,
        )
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f"[validate] peak GPU mem (reserved) at bs={mem_bs}: {peak:.2f} GiB")
    else:
        print(f"[validate] (skipped bs=16 mem probe: fewer than {mem_bs} conditions)")

    return max_diff


def run_llm_judge(
    model_key: str = "mistral",
    device: str = "cuda:0",
    max_examples: int = 0,
    batch_size: int = 16,
    validate: bool = False,
):
    """Run LLM-as-judge sufficiency baseline on SufficiencyBench test set."""
    print(f"\n{'='*60}")
    print(f"LLM-AS-JUDGE BASELINE — {model_key}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_key, device)

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten conditions
    conditions = _flatten_conditions(test_data)

    if max_examples > 0:
        conditions = conditions[:max_examples]
        print(f"  (limited to {max_examples} examples)")

    if validate:
        validate_batching(model, tokenizer, conditions, device,
                          batch_size=batch_size)
        del model
        torch.cuda.empty_cache()
        print("\n[validate] done (no results written).")
        return None

    print(f"Evaluating {len(conditions)} conditions "
          f"(batch_size={batch_size})...\n")

    scores = []
    raw_responses = []
    parse_stats = {"YES": 0, "NO": 0, "ambiguous": 0}

    n_batches = (len(conditions) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(conditions), batch_size),
                      total=n_batches, desc="LLM judge"):
        batch = conditions[start:start + batch_size]
        b_scores, b_responses = judge_sufficiency_batch(
            model, tokenizer,
            [c["context"] for c in batch],
            [c["question"] for c in batch],
            device,
        )
        for score, response in zip(b_scores, b_responses):
            scores.append(score)
            raw_responses.append(response[:100])  # truncate for storage

            if score >= 0.9:
                parse_stats["YES"] += 1
            elif score <= 0.1:
                parse_stats["NO"] += 1
            else:
                parse_stats["ambiguous"] += 1

        torch.cuda.empty_cache()

    labels = [1 if c["sufficient"] else 0 for c in conditions]
    metadata = [
        {"quadrant": c["quadrant"], "question_type": c["question_type"]}
        for c in conditions
    ]

    result = evaluate_method("llm_judge", scores, labels, metadata)
    result["parse_stats"] = parse_stats
    result["config"] = {
        "model": model_key,
        "n_conditions": len(conditions),
        "prompt_template": "joren_style_yes_no",
    }

    # Print summary
    auroc = result["overall"].get("auroc", "N/A")
    acc = result["overall"].get("accuracy", "N/A")
    print(f"\nLLM Judge: Overall AUROC = {auroc}, Accuracy = {acc}")
    print(f"  Parse stats: {parse_stats}")

    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        entry = result["per_quadrant"].get(quad, {})
        q_acc = entry.get("accuracy", "N/A")
        q_n = entry.get("n", 0)
        print(f"  {quad} (n={q_n}): accuracy={q_acc}")

    # Save
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_judge_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_path}")

    del model
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-as-judge sufficiency baseline (Joren et al. style)"
    )
    parser.add_argument(
        "--model", default="mistral",
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=16,
        help="Conditions per generate() call (default 16).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run the batched-vs-unbatched fidelity check on the first 16 "
             "conditions, print max abs diff, and exit without writing results.",
    )
    args = parser.parse_args()

    run_llm_judge(
        model_key=args.model,
        device=args.device,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        validate=args.validate,
    )
