"""
failure_taxonomy.py

Systematic categorization of WHERE and WHY each method fails on SufficiencyBench.
For each method × model, classifies every misclassified test example into failure modes.

Produces a taxonomy table for the paper: method × failure_mode × count.

Usage:
  python src/evaluation/failure_taxonomy.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, BENCH_DIR, MODEL_CONFIGS


def load_split(model_key, split):
    base = HIDDEN_STATES_DIR / model_key / split
    h_ctx = np.load(base / "h_with_context.npy")
    h_q = np.load(base / "h_question_only.npy")
    labels = np.load(base / "labels.npy")
    with open(base / "metadata.json") as f:
        metadata = json.load(f)
    return h_ctx, h_q, labels, metadata


def load_benchmark_details():
    """Load full benchmark data for error analysis."""
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                "id": cond["condition_id"],
                "question": ex["question"],
                "question_type": ex["question_type"],
                "gold_answer": ex["gold_answer"],
                "context_length": cond.get("context_length", len(cond["context"])),
                "sufficient": cond["sufficient"],
                "model_confident": cond["model_confident"],
                "quadrant": cond["quadrant"],
                "closed_book_confidence": ex.get("closed_book_confidence", 0),
            })
    return conditions


def classify_failure(example, probe_score, probe_correct, conf_score,
                     std_correct, csp_correct):
    """Classify a single failure into a failure mode category."""
    quad = example["quadrant"]
    qtype = example["question_type"]
    suf = example["sufficient"]
    conf = example["model_confident"]
    cb_conf = example["closed_book_confidence"]

    # Borderline: probe score near decision boundary
    if 0.35 < probe_score < 0.65:
        return "borderline_score"

    # Confidence leak: standard gets it wrong on Q2/Q3 but CSP gets it right
    if quad in ["Q2", "Q3"] and not std_correct and csp_correct:
        return "confidence_leak_fixed_by_csp"

    # Confidence leak: both fail on Q3 (confident + insufficient)
    if quad == "Q3" and not probe_correct and conf:
        if cb_conf > 0.95:
            return "extreme_parametric_confidence"
        return "parametric_confidence_interference"

    # Over-refusal: sufficient context but probe says insufficient (Q1/Q2)
    if suf and not probe_correct:
        if qtype == "subjective":
            return "subjective_ambiguity"
        if qtype == "multi_hop":
            return "multi_hop_complexity"
        return "missed_sufficient_context"

    # Under-detection: insufficient context but probe says sufficient (Q3/Q4)
    if not suf and not probe_correct:
        if qtype == "subjective":
            return "subjective_ambiguity"
        return "missed_insufficient_context"

    # Correct prediction — not a failure
    return "correct"


def run_failure_taxonomy():
    print(f"\n{'='*60}")
    print(f"FAILURE MODE TAXONOMY")
    print(f"{'='*60}")

    conditions = load_benchmark_details()
    all_results = {}

    for model_key in ["llama", "mistral", "qwen"]:
        print(f"\n--- {model_key} ---")

        # Load hidden states
        h_ctx_train, h_q_train, y_train, meta_train = load_split(model_key, "train")
        h_ctx_test, h_q_test, y_test, meta_test = load_split(model_key, "test")

        with open(RESULTS_DIR / model_key / "deco_results.json") as f:
            deco_res = json.load(f)
        std_layer = deco_res["best_layers"]["standard"]
        csp_layer = deco_res["best_layers"]["DECO"]

        # Train standard probe
        scaler_std = StandardScaler()
        X_tr_std = scaler_std.fit_transform(h_ctx_train[:, std_layer, :])
        X_te_std = scaler_std.transform(h_ctx_test[:, std_layer, :])
        std_probe = LogisticRegression(max_iter=2000, random_state=42)
        std_probe.fit(X_tr_std, y_train)
        std_probs = std_probe.predict_proba(X_te_std)[:, 1]
        std_preds = std_probe.predict(X_te_std)

        # Train CSP probe
        scaler_csp = StandardScaler()
        diff_train = h_ctx_train[:, csp_layer, :] - h_q_train[:, csp_layer, :]
        diff_test = h_ctx_test[:, csp_layer, :] - h_q_test[:, csp_layer, :]
        X_tr_csp = scaler_csp.fit_transform(diff_train)
        X_te_csp = scaler_csp.transform(diff_test)
        csp_probe = LogisticRegression(max_iter=2000, random_state=42)
        csp_probe.fit(X_tr_csp, y_train)
        csp_probs = csp_probe.predict_proba(X_te_csp)[:, 1]
        csp_preds = csp_probe.predict(X_te_csp)

        # Train confidence probe
        conf_labels_train = np.array([1 if m["model_confident"] else 0 for m in meta_train])
        scaler_conf = StandardScaler()
        X_tr_conf = scaler_conf.fit_transform(h_ctx_train[:, std_layer, :])
        X_te_conf = scaler_conf.transform(h_ctx_test[:, std_layer, :])
        conf_probe = LogisticRegression(max_iter=2000, random_state=42)
        conf_probe.fit(X_tr_conf, conf_labels_train)
        conf_probs = conf_probe.predict_proba(X_te_conf)[:, 1]

        # Classify failures for both methods
        model_results = {"standard_probe": {}, "csp_probe": {}}

        for method_name, preds, probs in [
            ("standard_probe", std_preds, std_probs),
            ("csp_probe", csp_preds, csp_probs),
        ]:
            taxonomy = defaultdict(lambda: {"count": 0, "quadrants": defaultdict(int),
                                             "question_types": defaultdict(int)})
            n_errors = 0
            n_correct = 0

            for i in range(len(y_test)):
                std_correct = (std_preds[i] == y_test[i])
                csp_correct = (csp_preds[i] == y_test[i])
                probe_correct = (preds[i] == y_test[i])

                category = classify_failure(
                    conditions[i], probs[i], probe_correct,
                    conf_probs[i], std_correct, csp_correct
                )

                if category == "correct":
                    n_correct += 1
                    continue

                n_errors += 1
                taxonomy[category]["count"] += 1
                taxonomy[category]["quadrants"][conditions[i]["quadrant"]] += 1
                taxonomy[category]["question_types"][conditions[i]["question_type"]] += 1

            # Convert defaultdicts to regular dicts
            taxonomy_dict = {}
            for cat, info in sorted(taxonomy.items(), key=lambda x: -x[1]["count"]):
                taxonomy_dict[cat] = {
                    "count": info["count"],
                    "fraction_of_errors": info["count"] / max(n_errors, 1),
                    "quadrants": dict(info["quadrants"]),
                    "question_types": dict(info["question_types"]),
                }

            model_results[method_name] = {
                "n_total": len(y_test),
                "n_correct": n_correct,
                "n_errors": n_errors,
                "error_rate": n_errors / len(y_test),
                "taxonomy": taxonomy_dict,
            }

            print(f"\n  {method_name}: {n_errors} errors ({n_errors/len(y_test)*100:.1f}%)")
            for cat, info in taxonomy_dict.items():
                pct = info["fraction_of_errors"] * 100
                print(f"    {cat}: {info['count']} ({pct:.1f}% of errors)")

        all_results[model_key] = model_results

    # Cross-method summary
    print(f"\n{'='*60}")
    print("CROSS-METHOD FAILURE MODE SUMMARY")
    print(f"{'='*60}")

    # Collect all failure modes across models
    all_modes = set()
    for model_key, mr in all_results.items():
        for method, data in mr.items():
            all_modes.update(data["taxonomy"].keys())

    summary_table = {}
    for mode in sorted(all_modes):
        summary_table[mode] = {}
        for model_key in ["llama", "mistral", "qwen"]:
            for method in ["standard_probe", "csp_probe"]:
                key = f"{model_key}_{method}"
                count = all_results[model_key][method]["taxonomy"].get(mode, {}).get("count", 0)
                summary_table[mode][key] = count

    all_results["summary_table"] = summary_table
    all_results["failure_modes"] = sorted(all_modes)

    # Save
    out_dir = RESULTS_DIR / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "failure_taxonomy.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_failure_taxonomy()
