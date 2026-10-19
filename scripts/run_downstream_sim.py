"""
run_downstream_sim.py

Re-runs the downstream RAG gating simulation for Mistral 7B, producing
consistent numbers for both the probe gate and Mistral VC gate.

Simulation rule (from paper):
  correct = (gate=use_context AND gold_answer in generation)
            OR (gate=abstain AND context is insufficient)

Gate strategies evaluated:
  1. No gate (always use context)
  2. Mistral VC gate (verbalized confidence from baseline_scores.json)
  3. Mistral probe gate (CSP LogReg from probe_scores.json)
  4. Oracle gate (true sufficiency label)

Usage:
  python scripts/run_downstream_sim.py --device cuda:0
  python scripts/run_downstream_sim.py --device cuda:0 --n-per-quad 125
"""

import json
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS


CHECKPOINT_PATH = RESULTS_DIR / "downstream_simulation" / "sim_checkpoint.jsonl"
OUT_PATH        = RESULTS_DIR / "downstream_simulation" / "mistral_sim_results.json"


def load_scores():
    scores_dir = RESULTS_DIR / "mistral" / "scores"
    with open(scores_dir / "probe_scores.json") as f:
        ps = json.load(f)
    with open(scores_dir / "baseline_scores.json") as f:
        bs = json.load(f)

    probe_meta   = ps["CSP (LogReg)"]["metadata"]
    probe_scores = ps["CSP (LogReg)"]["scores"]
    vc_meta      = bs["verbalized_confidence"]["metadata"]
    vc_scores    = bs["verbalized_confidence"]["scores"]

    probe_lookup = {m["id"]: s for m, s in zip(probe_meta, probe_scores)}
    vc_lookup    = {m["condition_id"]: s for m, s in zip(vc_meta, vc_scores)}
    return probe_lookup, vc_lookup


def load_balanced_sample(n_per_quad: int, seed: int = 42):
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten to per-condition examples
    all_examples = []
    for item in test_data:
        for cond in item["conditions"]:
            all_examples.append({
                "condition_id":   cond["condition_id"],
                "question":       item["question"],
                "gold_answer":    item["gold_answer"],
                "question_type":  item["question_type"],
                "sufficient":     cond["sufficient"],
                "quadrant":       cond["quadrant"],
                "prompt":         cond["prompt"],
            })

    by_quad = defaultdict(list)
    for ex in all_examples:
        by_quad[ex["quadrant"]].append(ex)

    rng = np.random.RandomState(seed)
    selected = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        pool = by_quad[q]
        idx  = rng.choice(len(pool), size=min(n_per_quad, len(pool)), replace=False)
        selected.extend([pool[i] for i in idx])

    return selected


def generate_answer(model, tokenizer, prompt: str, device: str, max_tokens: int = 150) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=2048,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def is_correct(gold_answer: str, generation: str) -> bool:
    return gold_answer.lower() in generation.lower()


def simulate_gate(examples, gate_scores, threshold=0.5):
    """
    correct = (gate=1 AND gold in generation) OR (gate=0 AND not sufficient)
    Returns downstream accuracy.
    """
    correct = []
    for ex in examples:
        gate = gate_scores[ex["condition_id"]] > threshold
        if gate:
            correct.append(ex["correct_vanilla"])
        else:
            correct.append(not ex["sufficient"])
    return float(np.mean(correct))


def find_optimal_threshold(examples, score_lookup):
    thresholds = sorted(set(score_lookup[ex["condition_id"]] for ex in examples))
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        acc = simulate_gate(examples, score_lookup, threshold=t)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--n-per-quad",  type=int, default=125)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    # Load pre-computed scores
    print("Loading pre-computed probe and VC scores...")
    probe_lookup, vc_lookup = load_scores()

    # Sample 500 balanced examples
    examples = load_balanced_sample(args.n_per_quad, args.seed)
    print(f"Sampled {len(examples)} examples:")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        n = sum(1 for e in examples if e["quadrant"] == q)
        print(f"  {q}: {n}")

    # Check all have scores
    missing_probe = [e["condition_id"] for e in examples if e["condition_id"] not in probe_lookup]
    missing_vc    = [e["condition_id"] for e in examples if e["condition_id"] not in vc_lookup]
    if missing_probe:
        print(f"WARNING: {len(missing_probe)} examples missing probe scores")
    if missing_vc:
        print(f"WARNING: {len(missing_vc)} examples missing VC scores")

    # Load checkpoint (resume if interrupted)
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["condition_id"]] = rec
        print(f"Resuming from checkpoint: {len(done)} examples already done")

    # Load Mistral 7B
    cfg = MODEL_CONFIGS["mistral"]
    print(f"\nLoading Mistral 7B from {cfg['path']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map=args.device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Generate answers
    ckpt_f = open(CHECKPOINT_PATH, "a")
    for ex in tqdm(examples, desc="Generating"):
        cid = ex["condition_id"]
        if cid in done:
            continue

        gen = generate_answer(model, tokenizer, ex["prompt"], args.device)
        rec = {
            "condition_id":  cid,
            "quadrant":      ex["quadrant"],
            "sufficient":    ex["sufficient"],
            "gold_answer":   ex["gold_answer"],
            "generation":    gen[:400],
            "correct_vanilla": is_correct(ex["gold_answer"], gen),
        }
        done[cid] = rec
        ckpt_f.write(json.dumps(rec) + "\n")
        ckpt_f.flush()

        if len(done) % 50 == 0:
            torch.cuda.empty_cache()

    ckpt_f.close()

    # Attach generation results back to examples
    for ex in examples:
        rec = done[ex["condition_id"]]
        ex["correct_vanilla"] = rec["correct_vanilla"]

    del model
    torch.cuda.empty_cache()

    # ── Simulation ──────────────────────────────────────────────────────────
    print("\n=== Downstream Gating Simulation ===")

    # Oracle gate (true sufficiency label)
    oracle_correct = [
        ex["correct_vanilla"] if ex["sufficient"] else True
        for ex in examples
    ]
    oracle_acc = float(np.mean(oracle_correct))

    # No gate
    no_gate_acc = float(np.mean([ex["correct_vanilla"] for ex in examples]))

    # Probe gate (default threshold 0.5)
    probe_acc = simulate_gate(examples, probe_lookup, threshold=0.5)
    probe_opt_t, probe_opt_acc = find_optimal_threshold(examples, probe_lookup)

    # Mistral VC gate (default threshold 0.5)
    vc_acc = simulate_gate(examples, vc_lookup, threshold=0.5)
    vc_opt_t, vc_opt_acc = find_optimal_threshold(examples, vc_lookup)

    results = {
        "n_per_quad": args.n_per_quad,
        "seed": args.seed,
        "strategies": {
            "No gate (always use context)": round(no_gate_acc, 4),
            "Verbalized confidence (Mistral 7B)": round(vc_acc, 4),
            "Probe gate (Mistral 7B)": round(probe_acc, 4),
            "Oracle gate": round(oracle_acc, 4),
        },
        "optimal_threshold": {
            "probe": {"threshold": round(probe_opt_t, 4), "accuracy": round(probe_opt_acc, 4)},
            "mistral_vc": {"threshold": round(vc_opt_t, 4), "accuracy": round(vc_opt_acc, 4)},
        },
        "probe_vs_vc_pp": round((probe_acc - vc_acc) * 100, 1),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Strategy':<40} {'Accuracy':>10}")
    print("-" * 52)
    for k, v in results["strategies"].items():
        print(f"{k:<40} {v:>10.3f}")
    print(f"\nProbe vs Mistral VC: +{results['probe_vs_vc_pp']} pp")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
