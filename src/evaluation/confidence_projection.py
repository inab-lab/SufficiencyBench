"""
confidence_projection.py

Validates the orthogonality finding by projecting OUT the confidence direction
from h(q+c), then retraining the sufficiency probe. If sufficiency and confidence
are truly orthogonal, removing confidence should NOT affect sufficiency detection.

Also runs the reverse: project out sufficiency, check confidence still works.

This is a CAUSAL validation, not just correlational (like cosine similarity).

Usage:
  python src/evaluation/confidence_projection.py
  python src/evaluation/confidence_projection.py --models llama mistral qwen
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


def load_split(model_key: str, split: str):
    base = HIDDEN_STATES_DIR / model_key / split
    h_ctx = np.load(base / "h_with_context.npy")
    h_q = np.load(base / "h_question_only.npy")
    labels = np.load(base / "labels.npy")
    with open(base / "metadata.json") as f:
        metadata = json.load(f)
    return h_ctx, h_q, labels, metadata


def project_out_direction(X, direction):
    """Remove the component of X along `direction` (rank-1 projection removal)."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    # X_proj = X - (X @ d) * d
    projections = X @ d  # (N,)
    return X - np.outer(projections, d)


def train_and_evaluate(X_train, y_train, X_test, y_test, metadata):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    probe.fit(X_tr, y_train)

    y_prob = probe.predict_proba(X_te)[:, 1]
    y_pred = probe.predict(X_te)

    result = {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Per-quadrant
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(metadata) if m["quadrant"] == quad]
        if len(idx) < 5:
            continue
        q_true = y_test[idx]
        q_pred = y_pred[idx]
        result[f"{quad}_accuracy"] = float(accuracy_score(q_true, q_pred))

    return result, probe.coef_[0]


def run_confidence_projection(model_keys=None):
    if model_keys is None:
        model_keys = ["llama", "mistral", "qwen"]

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"CONFIDENCE PROJECTION — {model_key}")
        print(f"{'='*60}")

        # Load data
        h_ctx_train, h_q_train, y_suf_train, meta_train = load_split(model_key, "train")
        h_ctx_test, h_q_test, y_suf_test, meta_test = load_split(model_key, "test")

        # Get best layer from deco results
        with open(RESULTS_DIR / model_key / "deco_results.json") as f:
            deco_res = json.load(f)
        best_layer = deco_res["best_layers"]["standard"]

        print(f"  Using layer {best_layer}")

        # Extract features at best layer
        H_train = h_ctx_train[:, best_layer, :].astype(np.float32)
        H_test = h_ctx_test[:, best_layer, :].astype(np.float32)

        # Confidence labels from metadata
        y_conf_train = np.array([1 if m["model_confident"] else 0 for m in meta_train])
        y_conf_test = np.array([1 if m["model_confident"] else 0 for m in meta_test])

        print(f"  Train: {len(y_suf_train)} examples, {y_conf_train.sum()} confident")
        print(f"  Test: {len(y_suf_test)} examples, {y_conf_test.sum()} confident")

        model_results = {}

        # --- Step 1: Baseline sufficiency probe (no projection) ---
        print("\n  [1] Baseline sufficiency probe on h(q+c)...")
        res_suf_base, w_suf = train_and_evaluate(
            H_train, y_suf_train, H_test, y_suf_test, meta_test
        )
        model_results["sufficiency_baseline"] = res_suf_base
        print(f"      AUROC = {res_suf_base['auroc']:.4f}")

        # --- Step 2: Baseline confidence probe ---
        print("  [2] Baseline confidence probe on h(q+c)...")
        res_conf_base, w_conf = train_and_evaluate(
            H_train, y_conf_train, H_test, y_conf_test, meta_test
        )
        model_results["confidence_baseline"] = res_conf_base
        print(f"      AUROC = {res_conf_base['auroc']:.4f}")

        # --- Step 3: Project OUT confidence, retrain sufficiency ---
        print("  [3] Sufficiency probe AFTER removing confidence direction...")
        H_train_no_conf = project_out_direction(H_train, w_conf)
        H_test_no_conf = project_out_direction(H_test, w_conf)
        res_suf_projected, _ = train_and_evaluate(
            H_train_no_conf, y_suf_train, H_test_no_conf, y_suf_test, meta_test
        )
        model_results["sufficiency_without_confidence"] = res_suf_projected
        delta_suf = res_suf_projected["auroc"] - res_suf_base["auroc"]
        print(f"      AUROC = {res_suf_projected['auroc']:.4f} (delta = {delta_suf:+.4f})")

        # --- Step 4: Project OUT sufficiency, retrain confidence ---
        print("  [4] Confidence probe AFTER removing sufficiency direction...")
        H_train_no_suf = project_out_direction(H_train, w_suf)
        H_test_no_suf = project_out_direction(H_test, w_suf)
        res_conf_projected, _ = train_and_evaluate(
            H_train_no_suf, y_conf_train, H_test_no_suf, y_conf_test, meta_test
        )
        model_results["confidence_without_sufficiency"] = res_conf_projected
        delta_conf = res_conf_projected["auroc"] - res_conf_base["auroc"]
        print(f"      AUROC = {res_conf_projected['auroc']:.4f} (delta = {delta_conf:+.4f})")

        # --- Step 5: Project out RANDOM direction (control) ---
        print("  [5] Sufficiency probe AFTER removing random direction (control)...")
        rng = np.random.RandomState(42)
        random_deltas = []
        for _ in range(10):
            w_rand = rng.randn(H_train.shape[1]).astype(np.float32)
            H_tr_rand = project_out_direction(H_train, w_rand)
            H_te_rand = project_out_direction(H_test, w_rand)
            res_rand, _ = train_and_evaluate(
                H_tr_rand, y_suf_train, H_te_rand, y_suf_test, meta_test
            )
            random_deltas.append(res_rand["auroc"] - res_suf_base["auroc"])
        model_results["random_projection_delta_mean"] = float(np.mean(random_deltas))
        model_results["random_projection_delta_std"] = float(np.std(random_deltas))
        print(f"      Random delta: {np.mean(random_deltas):+.4f} ± {np.std(random_deltas):.4f}")

        # --- Summary ---
        model_results["summary"] = {
            "layer": best_layer,
            "sufficiency_auroc_baseline": res_suf_base["auroc"],
            "sufficiency_auroc_after_removing_confidence": res_suf_projected["auroc"],
            "delta_sufficiency": float(delta_suf),
            "confidence_auroc_baseline": res_conf_base["auroc"],
            "confidence_auroc_after_removing_sufficiency": res_conf_projected["auroc"],
            "delta_confidence": float(delta_conf),
            "random_projection_delta_mean": float(np.mean(random_deltas)),
            "orthogonality_validated": abs(delta_suf) < 0.02 and abs(delta_conf) < 0.02,
        }

        print(f"\n  SUMMARY for {model_key}:")
        print(f"    Removing confidence from h(q+c): sufficiency delta = {delta_suf:+.4f}")
        print(f"    Removing sufficiency from h(q+c): confidence delta = {delta_conf:+.4f}")
        print(f"    Removing random direction:        delta = {np.mean(random_deltas):+.4f} ± {np.std(random_deltas):.4f}")
        print(f"    Orthogonality validated: {model_results['summary']['orthogonality_validated']}")

        all_results[model_key] = model_results

        # Save per-model
        out_path = RESULTS_DIR / model_key / "confidence_projection_results.json"
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"  Saved to {out_path}")

    # Save combined
    combined_path = RESULTS_DIR / "confidence_projection_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence projection orthogonality validation")
    parser.add_argument("--models", nargs="+", default=["llama", "mistral", "qwen"])
    args = parser.parse_args()
    run_confidence_projection(model_keys=args.models)
