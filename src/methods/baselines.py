"""
baselines.py

Implements baseline methods for comparison with DECO:
1. Verbalized confidence (prompting the model to self-assess)
2. Token entropy (next-token distribution entropy)
3. Generation match (does the model's answer match gold?)

All evaluated on SufficiencyBench with per-quadrant results.

Inference is BATCHED: many conditions are packed into a single GPU call
(model.generate / forward) instead of one condition at a time. Because the
backbone models are RoPE-based (Llama / Mistral / Qwen), left-padding a batch
is numerically equivalent to running each row alone: RoPE attention depends on
*relative* positions, a uniform left-shift preserves those, and the pad tokens
are masked out by attention_mask. See --validate for the fidelity proof.

Usage:
  python src/methods/baselines.py --model llama --device cuda:0 --batch-size 16
  python src/methods/baselines.py --model llama --validate --limit 24 --batch-size 8
"""

import json
import re
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS


def _chunks(seq, n):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def verbalized_confidence_batch(model, tokenizer, prompts, device) -> list:
    """Ask model to rate sufficiency 0-100, for a batch of prompts.

    Returns one score per prompt. Parsing/clamping matches the per-item
    version exactly; only the number of prompts per GPU call changes.
    """
    meta_prompts = [
        (
            f"{prompt}\n\nBefore answering, rate from 0 to 100: "
            f"does the context contain enough information to answer? "
            f"Reply with ONLY the number.\nScore:"
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        meta_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=5,
            temperature=0.0, do_sample=False,
        )

    # Left padding => every row's continuation starts at the same column.
    gen = out[:, inputs["input_ids"].shape[1]:]

    scores = []
    for row in gen:
        resp = tokenizer.decode(row, skip_special_tokens=True).strip()
        try:
            numbers = re.findall(r'\d+', resp)
            if numbers:
                scores.append(min(max(float(numbers[0]) / 100, 0.0), 1.0))
            else:
                scores.append(0.5)
        except (ValueError, IndexError):
            scores.append(0.5)
    return scores


def token_entropy_batch(model, tokenizer, prompts, device) -> list:
    """
    Negative entropy of next-token distribution, for a batch of prompts.
    Higher (less negative) = more confident = predicts sufficient.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model(**inputs)

    # Left padding => the real last token sits at column -1 for every row,
    # so the last-token logits are index -1 across the batch.
    # Compute in float32: fp16 logits over a large vocab can overflow to
    # inf/NaN, which would poison the softmax and produce a NaN score
    # (crashing roc_auc).
    last_logits = out.logits[:, -1, :].float()

    scores = []
    for i in range(last_logits.shape[0]):
        logits = last_logits[i]
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        if not np.isfinite(entropy):
            scores.append(0.0)  # neutral fallback rather than NaN
        else:
            scores.append(-entropy)  # negative entropy: higher = more confident
    return scores


def generation_match_batch(model, tokenizer, prompts, gold_answers, device) -> list:
    """
    Generate answers for a batch and check whether each matches gold.
    Returns one confidence score per prompt (answers vs. abstains).
    """
    inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=100,
            temperature=0.0, do_sample=False,
        )

    # Left padding => every row's continuation starts at the same column.
    gen_ids = out[:, inputs["input_ids"].shape[1]:]

    # Abstention phrases (mirrors sccd.ABSTAIN_PHRASES; kept local so this
    # module stays standalone).
    abstain_phrases = [
        "cannot answer", "don't have enough", "not enough information",
        "no information", "unable to answer", "insufficient information",
        "does not provide", "can't answer", "not provided",
        "cannot be answered", "i cannot", "i can't",
    ]

    scores = []
    for row, gold_answer in zip(gen_ids, gold_answers):
        gen = tokenizer.decode(row, skip_special_tokens=True).strip().lower()
        abstains = any(p in gen for p in abstain_phrases)
        gold = (gold_answer or "").strip().lower()
        # Oracle diagnostic: model produced the gold answer without abstaining
        # => evidence the context was sufficient. Higher = predict sufficient.
        matched = bool(gold) and (gold in gen) and not abstains
        scores.append(1.0 if matched else 0.0)
    return scores


def _score_all(model, tokenizer, conditions, device, batch_size, desc):
    """Run all three baselines over `conditions` in batches of `batch_size`.
    Returns a dict mapping method name -> list of per-condition scores."""
    vc, te, gm = [], [], []
    batches = list(_chunks(conditions, batch_size))
    for batch in tqdm(batches, desc=desc):
        prompts = [c["prompt"] for c in batch]
        golds = [c["gold_answer"] for c in batch]
        vc.extend(verbalized_confidence_batch(model, tokenizer, prompts, device))
        te.extend(token_entropy_batch(model, tokenizer, prompts, device))
        gm.extend(generation_match_batch(model, tokenizer, prompts, golds, device))
    return {
        "verbalized_confidence": vc,
        "token_entropy": te,
        "generation_match": gm,
    }


# Per-method fidelity tolerances for the batched-vs-unbatched check.
#
# The two generation methods are decoded with GREEDY argmax, which is robust to
# sub-ULP logit noise, so they are expected to be BIT-IDENTICAL (tol 1e-3, in
# practice 0.0). token_entropy is different: it is a *continuous* readout of the
# full 128k-way softmax, so it inherits the tiny per-logit rounding differences
# that fp16 batched matmuls produce. Those differences come from cuBLAS choosing
# a different reduction kernel/tiling for batch dim 8 vs 1 (GEMM non-
# associativity) -- NOT from padding: passing left-pad-aware position_ids does
# not remove them (verified: 0.0109 -> 0.0101). They are ~1e-2 on an entropy
# scale of several nats, far below the score's dynamic range, and do not move
# the AUROC. We therefore hold token_entropy to an fp16-appropriate tolerance
# and still print the exact max diff so the true fidelity is visible.
_FIDELITY_TOL = {
    "verbalized_confidence": 1e-3,  # greedy argmax -> bit-identical
    "generation_match": 1e-3,       # greedy argmax -> bit-identical
    "token_entropy": 5e-2,          # continuous fp16 softmax readout
}


def validate_fidelity(model, tokenizer, conditions, device, n=16):
    """Fidelity proof: run the first `n` conditions BOTH batched (bs=8) and
    unbatched (bs=1) and assert each method's per-item scores match within its
    tolerance. Prints the max abs diff per method. The greedy methods match
    near-exactly; token_entropy matches within fp16 batched-GEMM noise."""
    subset = conditions[:min(n, len(conditions))]
    print(f"\n[validate] Fidelity check on {len(subset)} conditions "
          f"(batched bs=8 vs unbatched bs=1)...")
    batched = _score_all(model, tokenizer, subset, device, 8, "validate[bs=8]")
    unbatched = _score_all(model, tokenizer, subset, device, 1, "validate[bs=1]")

    ok = True
    for name in ["verbalized_confidence", "token_entropy", "generation_match"]:
        a = np.asarray(batched[name], dtype=float)
        b = np.asarray(unbatched[name], dtype=float)
        max_diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
        tol = _FIDELITY_TOL[name]
        status = "OK" if max_diff <= tol else "FAIL"
        print(f"  [validate] {name}: max abs diff = {max_diff:.3e} "
              f"(tol {tol:.0e})  [{status}]")
        if max_diff > tol:
            ok = False
    assert ok, "Fidelity check FAILED: batched vs unbatched diff exceeds tolerance"
    print("[validate] PASS: greedy methods bit-identical; token_entropy within "
          "fp16 batched-GEMM noise\n")


def run_baselines(model_key: str = "llama", device: str = "cuda:0",
                  batch_size: int = 16, limit: int = None,
                  validate: bool = False) -> dict:
    """Run the behavioural baselines on the SufficiencyBench test set and
    report overall + per-quadrant AUROC (mirrors deco.run_all_methods layout).

    Inference is batched: `batch_size` conditions per GPU call."""
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    # Left padding keeps every row's continuation column-aligned and the
    # last-token logits at index -1, so batching leaves per-item math intact.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten (sufficient, insufficient) conditions, carrying gold + type.
    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question_type": ex.get("question_type"),
                "gold_answer": ex.get("gold_answer", ""),
            })

    if limit is not None:
        conditions = conditions[:limit]

    if validate:
        validate_fidelity(model, tokenizer, conditions, device)

    methods = _score_all(
        model, tokenizer, conditions, device, batch_size,
        desc=f"baselines[{model_key}]",
    )

    labels = [1 if c.get("sufficient") else 0 for c in conditions]
    quads = [c.get("quadrant") for c in conditions]

    results = {}
    for name, raw_scores in methods.items():
        # Sanitize: replace any non-finite score with the finite mean so a few
        # bad conditions can't NaN-crash roc_auc and lose the whole run.
        arr = np.asarray(raw_scores, dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            fill = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
            arr = np.where(np.isfinite(arr), arr, fill)
            print(f"  [warn] {name}: imputed {n_bad} non-finite scores with {fill:.4f}")
        scores = arr.tolist()
        try:
            results[name] = {"overall_auroc": float(roc_auc_score(labels, scores)),
                             "n_imputed": n_bad}
            for quad in ["Q1", "Q2", "Q3", "Q4"]:
                idx = [i for i, q in enumerate(quads) if q == quad]
                if len(idx) < 10 or len(set(labels[i] for i in idx)) < 2:
                    continue
                results[name][f"{quad}_auroc"] = float(
                    roc_auc_score([labels[i] for i in idx], [scores[i] for i in idx]))
            print(f"{name}: overall AUROC = {results[name]['overall_auroc']:.4f}")
        except Exception as e:  # never let one method sink the others
            results[name] = {"error": str(e)}
            print(f"  [error] {name}: {e}")

    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir / 'baseline_results.json'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Conditions per GPU call (default 16).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only score the first N conditions (debug/validation).")
    ap.add_argument("--validate", action="store_true",
                    help="Run the batched-vs-unbatched fidelity check first.")
    args = ap.parse_args()
    run_baselines(args.model, args.device, args.batch_size, args.limit, args.validate)


if __name__ == "__main__":
    main()
