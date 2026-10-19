"""
probe_ablations.py — Probe Architecture & Training Ablations

Experiments:
1. Architecture comparison (LR, MLP, Ridge, RF, NearestCentroid)
   → If LR ≈ MLP, sufficiency is linearly separable
2. Training data efficiency curves
   → AUROC vs % of training data
3. Feature importance overlap across models
   → Jaccard of top-k dimensions

Usage:
  python src/evaluation/probe_ablations.py
"""

import gc
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


def get_best_layer(model_key):
    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        return json.load(f)["best_layers"]["DECO"]


def _load_layer_features(model_key, split, layer):
    """Load DECO features for a single layer using mmap. Memory-safe."""
    import gc
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy", mmap_mode="r")
    h_q = np.load(d / "h_question_only.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)

    X = np.empty((h_qc.shape[0], h_qc.shape[2]), dtype=np.float32)
    chunk = 500
    for i in range(0, h_qc.shape[0], chunk):
        end = min(i + chunk, h_qc.shape[0])
        X[i:end] = h_qc[i:end, layer, :] - h_q[i:end, layer, :]

    del h_qc, h_q
    gc.collect()
    return X, labels, meta


# ---------------------------------------------------------------------------
# Experiment 1: Architecture Comparison
# ---------------------------------------------------------------------------

def run_architecture_comparison():
    """Compare probe architectures across models."""
    print("=" * 70)
    print("EXPERIMENT 1: PROBE ARCHITECTURE COMPARISON")
    print("=" * 70)

    models_list = list(MODEL_CONFIGS.keys())
    results = {}

    for mk in models_list:
        print(f"\n--- {mk.upper()} ---")
        best_layer = get_best_layer(mk)
        X_train, y_tr, _ = _load_layer_features(mk, "train", best_layer)
        X_test, y_te, meta_te = _load_layer_features(mk, "test", best_layer)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "MLP_256": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                     random_state=42, early_stopping=True),
            "MLP_256_256": MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=500,
                                         random_state=42, early_stopping=True),
            "Ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    random_state=42, n_jobs=-1),
            "NearestCentroid": NearestCentroid(),
        }

        print(f"  {'Architecture':<25s} {'AUROC':>8s} {'Acc':>8s} {'F1':>8s}")
        print(f"  {'-'*55}")

        model_results = {}
        for name, clf in classifiers.items():
            clf.fit(X_tr_s, y_tr)

            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_te_s)[:, 1]
                auroc = roc_auc_score(y_te, y_prob)
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_te_s)
                auroc = roc_auc_score(y_te, y_score)
            else:
                y_pred = clf.predict(X_te_s)
                auroc = roc_auc_score(y_te, y_pred)

            y_pred = clf.predict(X_te_s)
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred)

            model_results[name] = {"auroc": float(auroc), "accuracy": float(acc), "f1": float(f1)}
            print(f"  {name:<25s} {auroc:>8.4f} {acc:>8.4f} {f1:>8.4f}")

        results[mk] = model_results
        del X_train, X_test; gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Training Data Efficiency
# ---------------------------------------------------------------------------

def run_training_efficiency():
    """AUROC vs training data fraction."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TRAINING DATA EFFICIENCY")
    print("=" * 70)

    fractions = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    n_repeats = 5  # average over random subsets
    results = {}

    for mk in MODEL_CONFIGS:
        print(f"\n--- {mk.upper()} ---")
        best_layer = get_best_layer(mk)
        X_train, y_tr, _ = _load_layer_features(mk, "train", best_layer)
        X_test, y_te, _ = _load_layer_features(mk, "test", best_layer)

        print(f"  {'Fraction':>10s} {'N_train':>8s} {'AUROC':>8s} {'±':>6s}")
        print(f"  {'-'*38}")

        model_results = {}
        for frac in fractions:
            aurocs = []
            n_sub = max(10, int(frac * len(y_tr)))
            for rep in range(n_repeats):
                rng = np.random.default_rng(42 + rep)
                idx = rng.choice(len(y_tr), n_sub, replace=False)

                scaler = StandardScaler()
                X_sub = scaler.fit_transform(X_train[idx])
                probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                probe.fit(X_sub, y_tr[idx])

                X_te_s = scaler.transform(X_test)
                y_prob = probe.predict_proba(X_te_s)[:, 1]
                aurocs.append(roc_auc_score(y_te, y_prob))

            mean_auroc = np.mean(aurocs)
            std_auroc = np.std(aurocs)
            model_results[str(frac)] = {
                "n_train": n_sub,
                "mean_auroc": float(mean_auroc),
                "std_auroc": float(std_auroc),
                "aurocs": [float(a) for a in aurocs],
            }
            print(f"  {frac:>10.0%} {n_sub:>8d} {mean_auroc:>8.4f} {std_auroc:>6.4f}")

        results[mk] = model_results
        del X_train, X_test; gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Feature Importance Overlap
# ---------------------------------------------------------------------------

def run_feature_importance_overlap():
    """Top-k dimension overlap across models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: FEATURE IMPORTANCE OVERLAP")
    print("=" * 70)

    models_list = list(MODEL_CONFIGS.keys())
    results = {}

    # Train probes and get weight vectors
    weights = {}
    for mk in models_list:
        best_layer = get_best_layer(mk)
        X_train, y_tr, _ = _load_layer_features(mk, "train", best_layer)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        probe.fit(X_scaled, y_tr)

        # Weight in original space
        w = probe.coef_[0] / scaler.scale_
        weights[mk] = w
        del X_train, X_scaled; gc.collect()

    # Same-dim pairs: direct Jaccard overlap
    for k in [50, 100, 200]:
        print(f"\n  Top-{k} dimension overlap (Jaccard):")
        for i, m1 in enumerate(models_list):
            for m2 in models_list[i + 1:]:
                w1, w2 = weights[m1], weights[m2]
                if len(w1) != len(w2):
                    print(f"    {m1} vs {m2}: dim mismatch ({len(w1)} vs {len(w2)})")
                    continue

                top1 = set(np.argsort(np.abs(w1))[-k:])
                top2 = set(np.argsort(np.abs(w2))[-k:])
                jaccard = len(top1 & top2) / len(top1 | top2)
                overlap = len(top1 & top2)
                results[f"{m1}_{m2}_top{k}"] = {
                    "jaccard": float(jaccard), "overlap": overlap, "k": k
                }
                print(f"    {m1} vs {m2}: Jaccard={jaccard:.4f} ({overlap}/{k} overlap)")

    # Sign consistency of top dimensions (for same-dim pairs)
    print(f"\n  Sign consistency of top-100 dims (same direction = same effect):")
    for i, m1 in enumerate(models_list):
        for m2 in models_list[i + 1:]:
            w1, w2 = weights[m1], weights[m2]
            if len(w1) != len(w2):
                continue
            top_dims = list(set(np.argsort(np.abs(w1))[-100:]) &
                            set(np.argsort(np.abs(w2))[-100:]))
            if top_dims:
                signs_match = np.mean(np.sign(w1[top_dims]) == np.sign(w2[top_dims]))
                results[f"{m1}_{m2}_sign_consistency"] = float(signs_match)
                print(f"    {m1} vs {m2}: {signs_match:.3f} sign agreement on shared top dims")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Calibration Analysis
# ---------------------------------------------------------------------------

def run_calibration():
    """Reliability diagrams: binned probe score vs actual sufficiency rate."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: PROBE CALIBRATION (Reliability Diagrams)")
    print("=" * 70)

    results = {}
    n_bins = 10

    for mk in MODEL_CONFIGS:
        print(f"\n--- {mk.upper()} ---")
        best_layer = get_best_layer(mk)
        X_train, y_tr, _ = _load_layer_features(mk, "train", best_layer)
        X_test, y_te, _ = _load_layer_features(mk, "test", best_layer)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        del X_train, X_test; gc.collect()

        probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        probe.fit(X_tr_s, y_tr)
        y_prob = probe.predict_proba(X_te_s)[:, 1]

        # Bin by predicted probability
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_true_rates = []
        bin_counts = []

        print(f"  {'Bin':>12s} {'PredMean':>10s} {'TrueRate':>10s} {'Count':>8s}")
        print(f"  {'-'*45}")

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
            if np.sum(mask) == 0:
                continue

            pred_mean = np.mean(y_prob[mask])
            true_rate = np.mean(y_te[mask])
            count = int(np.sum(mask))

            bin_centers.append(float(pred_mean))
            bin_true_rates.append(float(true_rate))
            bin_counts.append(count)

            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}) {pred_mean:>10.3f} "
                  f"{true_rate:>10.3f} {count:>8d}")

        # ECE (Expected Calibration Error)
        total = sum(bin_counts)
        ece = sum(c * abs(p - t) for c, p, t in
                  zip(bin_counts, bin_centers, bin_true_rates)) / total

        results[mk] = {
            "bin_centers": bin_centers,
            "bin_true_rates": bin_true_rates,
            "bin_counts": bin_counts,
            "ece": float(ece),
        }
        print(f"\n  ECE = {ece:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    all_results = {}

    all_results["architecture"] = run_architecture_comparison()
    all_results["training_efficiency"] = run_training_efficiency()
    all_results["feature_overlap"] = run_feature_importance_overlap()
    all_results["calibration"] = run_calibration()

    # Save
    out_dir = RESULTS_DIR / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_ablations.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nAll results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_all()
