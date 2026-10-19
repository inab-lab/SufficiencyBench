"""
evaluate.py

Combines DECO and baseline results into paper-ready comparison tables.
Generates the main results table and per-quadrant breakdown.

Usage:
  python src/evaluation/evaluate.py --model llama
"""

import json
import numpy as np
from pathlib import Path
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import RESULTS_DIR, MODEL_CONFIGS


def load_results(model_key: str) -> dict:
    """Load all results for a model."""
    model_dir = RESULTS_DIR / model_key
    results = {}

    # DECO results
    deco_path = model_dir / "deco_results.json"
    if deco_path.exists():
        with open(deco_path) as f:
            deco = json.load(f)
        # Extract best results for each probing method
        for method in ["standard", "DECO", "confidence"]:
            results[f"probe_{method}"] = {
                "overall": deco["best_overall"].get(method, {}),
                "per_quadrant": deco["best_per_quadrant"].get(method, {}),
                "per_question_type": deco["best_per_question_type"].get(method, {}),
            }
        results["_correlation"] = deco.get("correlation", {})

    # Baseline results
    baseline_path = model_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        for name, data in baselines.items():
            results[name] = data

    return results


def print_main_table(results: dict, model_key: str):
    """Print the main comparison table (Table 1 in paper)."""
    print(f"\n{'='*80}")
    print(f"TABLE 1: Overall Results — {model_key}")
    print(f"{'='*80}")

    header = f"{'Method':<25s} {'AUROC':>8s} {'F1':>8s} {'Accuracy':>10s}"
    print(header)
    print("-" * len(header))

    for method_name in [
        "probe_standard", "probe_DECO", "probe_confidence",
        "verbalized_confidence", "token_entropy", "generation_match",
    ]:
        data = results.get(method_name, {})
        overall = data.get("overall", {})
        auroc = overall.get("auroc", None)
        f1 = overall.get("f1", None)
        acc = overall.get("accuracy", None)

        auroc_s = f"{auroc:.4f}" if auroc is not None else "—"
        f1_s = f"{f1:.4f}" if f1 is not None else "—"
        acc_s = f"{acc:.4f}" if acc is not None else "—"

        # Highlight DECO
        marker = " ***" if method_name == "probe_DECO" else ""
        print(f"{method_name:<25s} {auroc_s:>8s} {f1_s:>8s} {acc_s:>10s}{marker}")


def print_quadrant_table(results: dict, model_key: str):
    """Print per-quadrant breakdown (Table 2 — THE KEY TABLE)."""
    print(f"\n{'='*80}")
    print(f"TABLE 2: Per-Quadrant Accuracy — {model_key} (THE KEY TABLE)")
    print(f"{'='*80}")

    quad_descs = {
        "Q1": "Suf+Conf",
        "Q2": "Suf+Uncert",
        "Q3": "Insuf+Conf",
        "Q4": "Insuf+Uncert",
    }

    methods = [
        "probe_standard", "probe_DECO", "probe_confidence",
        "verbalized_confidence", "token_entropy", "generation_match",
    ]

    # Header
    header = f"{'Method':<25s}"
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        header += f" {quad_descs[quad]:>12s}"
    print(header)
    print("-" * len(header))

    for method_name in methods:
        data = results.get(method_name, {})
        pq = data.get("per_quadrant", {})
        row = f"{method_name:<25s}"
        for quad in ["Q1", "Q2", "Q3", "Q4"]:
            entry = pq.get(quad, {})
            acc = entry.get("accuracy")
            if acc is not None:
                row += f" {acc:>12.3f}"
            else:
                row += f" {'—':>12s}"
        if method_name == "probe_DECO":
            row += "  ***"
        print(row)

    print()
    print("KEY FINDING:")
    print("  If DECO > Standard on Q2 and Q3 → sufficiency decoupled from confidence")
    print("  Q2 (sufficient but uncertain): standard over-refuses, DECO should be better")
    print("  Q3 (insufficient but confident): standard fails to detect, DECO should catch")


def print_correlation(results: dict, model_key: str):
    """Print correlation analysis."""
    corr = results.get("_correlation", {})
    if not corr:
        print("\nNo correlation data available.")
        return

    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS — {model_key}")
    print(f"{'='*80}")

    cos_sc = corr.get("cos_standard_confidence", None)
    cos_dc = corr.get("cos_deco_confidence", None)
    cos_sd = corr.get("cos_standard_deco", None)

    if cos_sc is not None:
        print(f"cos(standard_dir, confidence_dir) = {cos_sc:.4f}")
    if cos_dc is not None:
        print(f"cos(DECO_dir, confidence_dir)     = {cos_dc:.4f}")
    if cos_sd is not None:
        print(f"cos(standard_dir, DECO_dir)       = {cos_sd:.4f}")

    print()
    if cos_sc is not None and abs(cos_sc) > 0.7:
        print("CONFIRMED: Standard probing ≈ confidence detection (cos > 0.7)")
    if cos_dc is not None and abs(cos_dc) < 0.4:
        print("CONFIRMED: DECO decoupled from confidence (cos < 0.4)")


def print_question_type_table(results: dict, model_key: str):
    """Print per-question-type AUROC."""
    print(f"\n{'='*80}")
    print(f"TABLE 3: Per-Question-Type AUROC — {model_key}")
    print(f"{'='*80}")

    qtypes = ["factual", "multi_hop", "comparative", "subjective"]
    methods = ["probe_standard", "probe_DECO", "verbalized_confidence", "token_entropy"]

    header = f"{'Method':<25s}"
    for qt in qtypes:
        header += f" {qt:>12s}"
    print(header)
    print("-" * len(header))

    for method_name in methods:
        data = results.get(method_name, {})
        pqt = data.get("per_question_type", {})
        row = f"{method_name:<25s}"
        for qt in qtypes:
            entry = pqt.get(qt, {})
            auroc = entry.get("auroc")
            if auroc is not None:
                row += f" {auroc:>12.4f}"
            else:
                row += f" {'—':>12s}"
        print(row)


def generate_full_report(model_key: str = "llama"):
    """Generate the full evaluation report."""
    results = load_results(model_key)
    if not results:
        print(f"No results found for {model_key} in {RESULTS_DIR / model_key}")
        return

    print_main_table(results, model_key)
    print_quadrant_table(results, model_key)
    print_correlation(results, model_key)
    print_question_type_table(results, model_key)

    # Save combined results
    out_path = RESULTS_DIR / model_key / "full_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()

    if args.model == "all":
        for key in MODEL_CONFIGS:
            try:
                generate_full_report(key)
            except Exception as e:
                print(f"Error on {key}: {e}")
    else:
        generate_full_report(args.model)
