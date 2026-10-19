"""
run_sonnet_vc_sim.py

Computes Sonnet 4.6 verbalized confidence scores for the same 500 balanced
examples used in run_downstream_sim.py, then adds a Sonnet VC row to
mistral_sim_results.json using the same simulation methodology.

Usage:
  python scripts/run_sonnet_vc_sim.py
"""

import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR
from scripts.run_downstream_sim import load_balanced_sample, simulate_gate


CHECKPOINT_PATH = RESULTS_DIR / "downstream_simulation" / "sonnet_vc_checkpoint.jsonl"
OUT_PATH        = RESULTS_DIR / "downstream_simulation" / "mistral_sim_results.json"

VC_PROMPT = """\
You are evaluating whether a context contains enough information to answer a question.

Context: {context}

Question: {question}

On a scale from 0 to 100, how confident are you that the context contains \
sufficient information to answer the question correctly? \
Reply with a single integer from 0 to 100 only."""


def get_context_from_bench(condition_id: str, bench_data: list) -> str:
    for item in bench_data:
        for cond in item["conditions"]:
            if cond["condition_id"] == condition_id:
                return cond["context"]
    return ""


def parse_score(text: str) -> float:
    text = text.strip()
    try:
        val = float(text.split()[0].rstrip("."))
        return max(0.0, min(100.0, val)) / 100.0
    except Exception:
        return 0.5


def main():
    client = anthropic.Anthropic()

    with open(BENCH_DIR / "test.json") as f:
        bench_data = json.load(f)

    # Build context lookup
    context_lookup = {}
    question_lookup = {}
    for item in bench_data:
        for cond in item["conditions"]:
            context_lookup[cond["condition_id"]] = cond["context"]
            question_lookup[cond["condition_id"]] = item["question"]

    # Same 500 examples as the Mistral simulation
    examples = load_balanced_sample(125, 42)
    print(f"Loaded {len(examples)} examples")

    # Load checkpoint
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["condition_id"]] = rec["vc_score"]
        print(f"Resuming: {len(done)} already done")

    ckpt_f = open(CHECKPOINT_PATH, "a")
    errors = 0

    for i, ex in enumerate(examples):
        cid = ex["condition_id"]
        if cid in done:
            continue

        context  = context_lookup.get(cid, "")
        question = question_lookup.get(cid, ex.get("question", ""))

        prompt = VC_PROMPT.format(context=context[:2000], question=question)

        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            score = parse_score(msg.content[0].text)
        except Exception as e:
            print(f"  Error on {cid}: {e}")
            errors += 1
            score = 0.5
            time.sleep(2)

        done[cid] = score
        ckpt_f.write(json.dumps({"condition_id": cid, "vc_score": score}) + "\n")
        ckpt_f.flush()

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/500 done  (errors: {errors})")

        time.sleep(0.1)  # gentle rate limiting

    ckpt_f.close()
    print(f"Done. {errors} errors.")

    # Build score lookup for the simulation
    vc_lookup = done

    # Attach generation results from Mistral sim checkpoint
    mistral_ckpt = RESULTS_DIR / "downstream_simulation" / "sim_checkpoint.jsonl"
    gen_data = {}
    with open(mistral_ckpt) as f:
        for line in f:
            rec = json.loads(line)
            gen_data[rec["condition_id"]] = rec

    for ex in examples:
        cid = ex["condition_id"]
        ex["correct_vanilla"] = gen_data[cid]["correct_vanilla"]
        ex["sufficient"] = gen_data[cid]["sufficient"]

    # Simulate Sonnet VC gate
    sonnet_vc_acc = simulate_gate(examples, vc_lookup, threshold=0.5)

    # Find optimal threshold
    thresholds = sorted(set(vc_lookup[ex["condition_id"]] for ex in examples))
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        acc = simulate_gate(examples, vc_lookup, threshold=t)
        if acc > best_acc:
            best_acc, best_t = acc, t

    print(f"\nSonnet VC gate (t=0.5):    {sonnet_vc_acc:.3f}")
    print(f"Sonnet VC gate (optimal):  {best_acc:.3f}  (t={best_t:.3f})")

    # Update results file
    with open(OUT_PATH) as f:
        results = json.load(f)

    results["strategies"]["Verbalized confidence (Sonnet 4.6)"] = round(sonnet_vc_acc, 4)
    results["optimal_threshold"]["sonnet_vc"] = {
        "threshold": round(best_t, 4),
        "accuracy": round(best_acc, 4),
    }
    # Reorder strategies for display
    ordered = {
        "No gate (always use context)":        results["strategies"]["No gate (always use context)"],
        "Verbalized confidence (Mistral 7B)":  results["strategies"]["Verbalized confidence (Mistral 7B)"],
        "Verbalized confidence (Sonnet 4.6)":  results["strategies"]["Verbalized confidence (Sonnet 4.6)"],
        "Probe gate (Mistral 7B)":             results["strategies"]["Probe gate (Mistral 7B)"],
        "Oracle gate":                         results["strategies"]["Oracle gate"],
    }
    results["strategies"] = ordered

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Full Table ===")
    for k, v in ordered.items():
        print(f"  {k:<40} {v:.3f}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
