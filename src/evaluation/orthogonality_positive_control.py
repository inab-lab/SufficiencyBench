"""
Positive control for orthogonality finding.

The concern: in 4096 dimensions, random vectors are nearly orthogonal by default
(expected cos ≈ 0, std ≈ 1/√d ≈ 0.016). So cos(sufficiency, confidence) ≈ 0
might be trivially expected.

This script trains probes for MULTIPLE concepts in the same feature space and
shows that some directions ARE correlated (cos >> 0) while sufficiency ⊥ confidence.
This proves the orthogonality is specific, not just high-dimensional noise.

Concepts probed (all on same h(q+c) at best layer):
  1. Sufficiency (sufficient vs insufficient)
  2. Confidence (model_knows vs doesn't)
  3. Question type (multi-hop vs factual vs comparative vs subjective)
  4. Quadrant (Q1 vs Q3, Q2 vs Q4 — same sufficiency, different confidence)
  5. Random labels (negative control — should be orthogonal to everything)

Expected results:
  - cos(sufficiency, question_type) may be > 0 (some question types are easier)
  - cos(sufficiency, random) ≈ 0 (negative control)
  - cos(confidence, question_type) should be > 0 (PK depends on question type)
  - cos(sufficiency, confidence) ≈ 0 (the finding we're validating)

Usage:
  python src/evaluation/orthogonality_positive_control.py
"""

import gc
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from itertools import combinations

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


def load_layer_features(model_key, split, layer):
    """Load h(q+c) features at a single layer using mmap."""
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)

    X = np.empty((h_qc.shape[0], h_qc.shape[2]), dtype=np.float32)
    chunk = 500
    for i in range(0, h_qc.shape[0], chunk):
        end = min(i + chunk, h_qc.shape[0])
        X[i:end] = h_qc[i:end, layer, :]

    del h_qc
    gc.collect()
    return X, labels, meta


def get_best_layer(model_key):
    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        return json.load(f)["best_layers"]["standard"]


def cosine_sim(w1, w2):
    """Cosine similarity between two weight vectors."""
    return float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-10))


def train_binary_probe(X, y, C=1.0):
    """Train logistic regression, return weight vector."""
    probe = LogisticRegression(max_iter=1000, C=C, random_state=42, solver="lbfgs")
    probe.fit(X, y)
    auroc = None  # Computed separately if needed
    return probe.coef_[0], probe


def train_multiclass_probe(X, y, C=1.0):
    """Train one-vs-rest logistic regression, return weight matrix."""
    probe = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=C, random_state=42, solver="lbfgs")
    )
    probe.fit(X, y)
    # Return weight matrix (n_classes, n_features)
    weights = np.array([est.coef_[0] for est in probe.estimators_])
    return weights, probe


def run_positive_control(model_key):
    """Run all concept probes for one model."""
    print(f"\n{'='*60}")
    print(f"  POSITIVE CONTROL: {model_key.upper()}")
    print(f"{'='*60}")

    best_layer = get_best_layer(model_key)
    print(f"Best layer: {best_layer}")

    # Load train and test
    X_train, y_suf_train, meta_train = load_layer_features(model_key, "train", best_layer)
    X_test, y_suf_test, meta_test = load_layer_features(model_key, "test", best_layer)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    del X_train, X_test
    gc.collect()

    # --- Extract all label sets ---
    y_conf_train = np.array([1 if m["model_confident"] else 0 for m in meta_train])
    y_conf_test = np.array([1 if m["model_confident"] else 0 for m in meta_test])

    le = LabelEncoder()
    y_qtype_train = le.fit_transform([m["question_type"] for m in meta_train])
    y_qtype_test = le.transform([m["question_type"] for m in meta_test])
    qtype_classes = le.classes_

    y_quad_train = LabelEncoder().fit_transform([m["quadrant"] for m in meta_train])
    y_quad_test = LabelEncoder().fit_transform([m["quadrant"] for m in meta_test])

    rng = np.random.default_rng(42)
    y_random_train = rng.integers(0, 2, size=len(y_suf_train))
    y_random_test = rng.integers(0, 2, size=len(y_suf_test))

    # --- Train probes ---
    from sklearn.metrics import roc_auc_score, accuracy_score

    probes = {}

    # 1. Sufficiency
    w_suf, p_suf = train_binary_probe(X_tr, y_suf_train)
    y_prob = p_suf.predict_proba(X_te)[:, 1]
    auroc_suf = roc_auc_score(y_suf_test, y_prob)
    probes["sufficiency"] = {"w": w_suf, "auroc": auroc_suf}
    print(f"\nSufficiency probe:    AUROC = {auroc_suf:.4f}")

    # 2. Confidence
    w_conf, p_conf = train_binary_probe(X_tr, y_conf_train)
    y_prob = p_conf.predict_proba(X_te)[:, 1]
    auroc_conf = roc_auc_score(y_conf_test, y_prob)
    probes["confidence"] = {"w": w_conf, "auroc": auroc_conf}
    print(f"Confidence probe:     AUROC = {auroc_conf:.4f}")

    # 3. Question type (binary probes for each type vs rest)
    qtype_weights = {}
    for i, qtype in enumerate(qtype_classes):
        y_qt_tr = (y_qtype_train == i).astype(int)
        y_qt_te = (y_qtype_test == i).astype(int)
        if y_qt_tr.sum() < 10 or y_qt_te.sum() < 10:
            continue
        w_qt, p_qt = train_binary_probe(X_tr, y_qt_tr)
        y_prob = p_qt.predict_proba(X_te)[:, 1]
        auroc_qt = roc_auc_score(y_qt_te, y_prob)
        qtype_weights[qtype] = {"w": w_qt, "auroc": auroc_qt}
        probes[f"qtype_{qtype}"] = {"w": w_qt, "auroc": auroc_qt}
        print(f"Question type '{qtype}': AUROC = {auroc_qt:.4f}")

    # 4. Quadrant probe (Q1 vs Q3: both sufficient=True, differs on confidence)
    # This tests if confidence is separable WITHIN sufficient contexts
    mask_q1q3_tr = np.array([m["quadrant"] in ("Q1", "Q3") for m in meta_train])
    mask_q1q3_te = np.array([m["quadrant"] in ("Q1", "Q3") for m in meta_test])
    if mask_q1q3_tr.sum() > 50 and mask_q1q3_te.sum() > 50:
        y_q1q3_tr = np.array([1 if m["quadrant"] == "Q1" else 0
                              for m in meta_train])[mask_q1q3_tr]
        y_q1q3_te = np.array([1 if m["quadrant"] == "Q1" else 0
                              for m in meta_test])[mask_q1q3_te]
        w_q1q3, p_q1q3 = train_binary_probe(X_tr[mask_q1q3_tr], y_q1q3_tr)
        y_prob = p_q1q3.predict_proba(X_te[mask_q1q3_te])[:, 1]
        auroc_q1q3 = roc_auc_score(y_q1q3_te, y_prob)
        probes["Q1_vs_Q3"] = {"w": w_q1q3, "auroc": auroc_q1q3}
        print(f"Q1 vs Q3 (conf within suf): AUROC = {auroc_q1q3:.4f}")

    # 5. Random labels (negative control)
    w_rand, p_rand = train_binary_probe(X_tr, y_random_train)
    y_prob = p_rand.predict_proba(X_te)[:, 1]
    auroc_rand = roc_auc_score(y_random_test, y_prob)
    probes["random"] = {"w": w_rand, "auroc": auroc_rand}
    print(f"Random labels:        AUROC = {auroc_rand:.4f}")

    # --- Pairwise cosine similarities ---
    print(f"\n--- Pairwise cosine similarities ---")
    probe_names = list(probes.keys())
    cosines = {}
    for a, b in combinations(probe_names, 2):
        cos = cosine_sim(probes[a]["w"], probes[b]["w"])
        key = f"{a}_vs_{b}"
        cosines[key] = cos

    # Print as a matrix for key pairs
    print(f"\n{'Pair':<45s} {'cos':>8s} {'|cos|':>8s}")
    print("-" * 65)
    for key, cos in sorted(cosines.items(), key=lambda x: -abs(x[1])):
        print(f"  {key:<43s} {cos:>8.4f} {abs(cos):>8.4f}")

    # --- Permutation test for non-trivial correlations ---
    # For pairs with |cos| > 0.05, test if significantly different from random
    # Statistical test: compare observed |cos| against the expected
    # distribution for random vectors in d dimensions.
    # For random unit vectors in R^d, cos ~ N(0, 1/d) approximately.
    # So |cos| > k/sqrt(d) is significant at the corresponding z-level.
    from scipy import stats
    dim = len(probes["sufficiency"]["w"])
    print(f"\n--- Statistical significance (analytical, z-test vs random baseline, d={dim}) ---")
    significant_pairs = {}
    null_std = 1.0 / np.sqrt(dim)
    for key, cos in cosines.items():
        z = abs(cos) / null_std
        p_val = 2 * (1 - stats.norm.cdf(z))  # two-tailed
        significant_pairs[key] = {
            "cos": cos,
            "abs_cos": abs(cos),
            "z_score": float(z),
            "p_value": float(p_val),
        }
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        if abs(cos) > 0.03 or "sufficiency_vs_confidence" in key:
            print(f"  {key:<43s} cos={cos:>7.4f}  z={z:>6.2f}  p={p_val:.2e} {sig}")

    # --- Expected cos for random vectors in this dimension ---
    dim = len(probes["sufficiency"]["w"])
    expected_random_cos_std = 1.0 / np.sqrt(dim)
    print(f"\n--- Reference ---")
    print(f"  Dimension: {dim}")
    print(f"  Expected |cos| for random vectors: {expected_random_cos_std:.4f}")
    print(f"  Any |cos| >> {expected_random_cos_std:.4f} is non-trivial")

    # --- Build results ---
    results = {
        "model": model_key,
        "best_layer": best_layer,
        "dimension": dim,
        "expected_random_cos_std": float(expected_random_cos_std),
        "probe_aurocs": {k: float(v["auroc"]) for k, v in probes.items()},
        "cosine_similarities": {k: float(v) for k, v in cosines.items()},
        "permutation_tests": significant_pairs,
    }

    del X_tr, X_te
    gc.collect()
    return results


def main():
    all_results = {}
    for mk in MODEL_CONFIGS:
        results = run_positive_control(mk)
        all_results[mk] = results

    # Save
    out_dir = RESULTS_DIR / "orthogonality_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "positive_control_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'positive_control_results.json'}")

    # --- Summary across models ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY: KEY COSINE SIMILARITIES")
    print(f"{'='*60}")
    key_pairs = [
        "sufficiency_vs_confidence",
        "sufficiency_vs_random",
        "confidence_vs_random",
    ]
    # Also find qtype pairs dynamically
    for mk, res in all_results.items():
        print(f"\n{mk.upper()}:")
        for pair, cos in sorted(res["cosine_similarities"].items(), key=lambda x: -abs(x[1])):
            marker = ""
            if "random" in pair:
                marker = " (negative control)"
            elif "sufficiency_vs_confidence" in pair:
                marker = " ← KEY FINDING"
            elif abs(cos) > 3 * res["expected_random_cos_std"]:
                marker = " ← NON-TRIVIAL"
            print(f"  {pair:<45s} cos={cos:>7.4f}{marker}")


if __name__ == "__main__":
    main()
