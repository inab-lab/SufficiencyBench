"""
representation_ablation.py

Ablation study comparing different input representations for sufficiency probes:
1. h(q+c)           — Standard probe (baseline)
2. h(q+c) - h(q)    — CSP (contrastive, our method)
3. [h(q+c); h(q)]   — Concatenation (both vectors)
4. h(q)             — Question-only (negative control: no context info)

Tests whether the subtraction in CSP is necessary by comparing alternatives.
Expected: CSP should win on Q3 (confident but insufficient) because subtraction
removes the parametric knowledge signal.

Usage:
  python src/evaluation/representation_ablation.py
  python src/evaluation/representation_ablation.py --models llama mistral qwen
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, BENCH_DIR, MODEL_CONFIGS


def load_split(model_key: str, split: str):
    """Load hidden states and labels for a split."""
    base = HIDDEN_STATES_DIR / model_key / split
    h_ctx = np.load(base / "h_with_context.npy")     # (N, L, D)
    h_q = np.load(base / "h_question_only.npy")       # (N, L, D)
    labels = np.load(base / "labels.npy")              # (N,)
    with open(base / "metadata.json") as f:
        metadata = json.load(f)
    return h_ctx, h_q, labels, metadata


def get_representations(h_ctx, h_q, layer: int):
    """Build all representation variants at a given layer."""
    hc = h_ctx[:, layer, :]    # (N, D)
    hq = h_q[:, layer, :]     # (N, D)

    return {
        "h(q+c)": hc,
        "h(q+c) - h(q)": hc - hq,
        "[h(q+c); h(q)]": np.concatenate([hc, hq], axis=1),
        "h(q) only": hq,
    }


def evaluate_per_quadrant(preds, scores, labels, metadata):
    """Per-quadrant accuracy breakdown."""
    result = {}
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(metadata) if m.get("quadrant") == quad]
        if len(idx) < 5:
            continue
        q_labels = labels[idx]
        q_preds = preds[idx]
        q_scores = scores[idx]

        entry = {"n": len(idx)}
        entry["accuracy"] = float(accuracy_score(q_labels, q_preds))
        if len(set(q_labels)) > 1:
            entry["auroc"] = float(roc_auc_score(q_labels, q_scores))
        result[quad] = entry
    return result


def run_ablation(model_keys=None):
    """Run representation ablation for specified models."""
    if model_keys is None:
        model_keys = ["llama", "mistral", "qwen"]

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"REPRESENTATION ABLATION — {model_key}")
        print(f"{'='*60}")

        # Load data
        h_ctx_train, h_q_train, y_train, meta_train = load_split(model_key, "train")
        h_ctx_val, h_q_val, y_val, meta_val = load_split(model_key, "val")
        h_ctx_test, h_q_test, y_test, meta_test = load_split(model_key, "test")

        n_layers = h_ctx_train.shape[1]
        print(f"  Loaded: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
        print(f"  Layers={n_layers}, hidden_dim={h_ctx_train.shape[2]}")

        model_results = {}

        # For each representation type, find best layer via validation
        representation_names = ["h(q+c)", "h(q+c) - h(q)", "[h(q+c); h(q)]", "h(q) only"]

        for rep_name in representation_names:
            print(f"\n  --- {rep_name} ---")

            best_val_auroc = 0
            best_layer = 0

            # Layer sweep on validation
            for layer in range(n_layers):
                reps_train = get_representations(h_ctx_train, h_q_train, layer)
                reps_val = get_representations(h_ctx_val, h_q_val, layer)

                X_train = reps_train[rep_name]
                X_val = reps_val[rep_name]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                clf = LogisticRegression(max_iter=3000, C=1.0, random_state=42)
                clf.fit(X_train_s, y_train)

                val_scores = clf.decision_function(X_val_s)
                if len(set(y_val)) > 1:
                    val_auroc = roc_auc_score(y_val, val_scores)
                else:
                    val_auroc = 0.5

                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_layer = layer

            # Retrain on best layer and evaluate on test
            print(f"  Best layer: {best_layer} (val AUROC: {best_val_auroc:.4f})")

            reps_train = get_representations(h_ctx_train, h_q_train, best_layer)
            reps_test = get_representations(h_ctx_test, h_q_test, best_layer)

            X_train = reps_train[rep_name]
            X_test = reps_test[rep_name]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=3000, C=1.0, random_state=42)
            clf.fit(X_train_s, y_train)

            test_scores = clf.decision_function(X_test_s)
            test_preds = clf.predict(X_test_s)

            test_auroc = roc_auc_score(y_test, test_scores)
            test_acc = accuracy_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds, zero_division=0)

            per_quad = evaluate_per_quadrant(test_preds, test_scores, y_test, meta_test)

            print(f"  Test AUROC: {test_auroc:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
            for quad, qr in per_quad.items():
                print(f"    {quad} (n={qr['n']}): acc={qr['accuracy']:.4f}", end="")
                if "auroc" in qr:
                    print(f", auroc={qr['auroc']:.4f}", end="")
                print()

            model_results[rep_name] = {
                "best_layer": best_layer,
                "val_auroc": float(best_val_auroc),
                "test_auroc": float(test_auroc),
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1),
                "per_quadrant": per_quad,
                "input_dim": X_train.shape[1],
            }

        all_results[model_key] = model_results

        # Print comparison table
        print(f"\n  Summary for {model_key}:")
        print(f"  {'Representation':<20} {'AUROC':>7} {'Acc':>7} {'Q3 Acc':>7}")
        print(f"  {'-'*45}")
        for rep_name, res in model_results.items():
            q3_acc = res["per_quadrant"].get("Q3", {}).get("accuracy", "N/A")
            if isinstance(q3_acc, float):
                q3_acc = f"{q3_acc:.4f}"
            print(f"  {rep_name:<20} {res['test_auroc']:>7.4f} {res['test_accuracy']:>7.4f} {q3_acc:>7}")

    # Save results
    out_dir = RESULTS_DIR / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "representation_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation ablation for sufficiency probes")
    parser.add_argument("--models", nargs="+", default=["llama", "mistral", "qwen"])
    args = parser.parse_args()
    run_ablation(model_keys=args.models)
