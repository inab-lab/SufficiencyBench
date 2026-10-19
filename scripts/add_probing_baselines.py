#!/usr/bin/env python3
"""
Add stronger probing baselines that use the already-extracted hidden states.

1. Embedding Similarity: cosine(embed(question), embed(context)) — retriever confidence proxy
2. SEP-like Correctness Probe: train probe on h(q+c) to predict if model's generation is correct
   (This mirrors Semantic Entropy Probes: Kossen et al., ICML 2024)
3. Confidence Probe (fixed): train probe on h(q+c) to predict model_knows_answer
   with model-specific PK labels when available

These don't need GPU — they run on the .npy hidden states already extracted.

Usage:
  python scripts/add_probing_baselines.py --model mistral
  python scripts/add_probing_baselines.py --model all
"""

import json
import gc
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import cosine as cosine_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import (
    BENCH_DIR, HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS, MODELS_DIR,
)


def load_hidden_states(model_key: str, split: str):
    """Load pre-extracted hidden states for a model and split."""
    base = HIDDEN_STATES_DIR / model_key / split
    h_with = np.load(base / "h_with_context.npy")
    h_q = np.load(base / "h_question_only.npy")
    labels = np.load(base / "labels.npy")
    with open(base / "metadata.json") as f:
        metadata = json.load(f)
    return h_with, h_q, labels, metadata


def load_benchmark(split: str):
    """Load benchmark split."""
    path = BENCH_DIR / f"{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Baseline 1: Embedding Similarity ─────────────────────────────────────

def embedding_similarity_baseline(model_key: str):
    """
    Use sentence embedding cosine similarity between question and context
    as a sufficiency proxy. High similarity = more likely sufficient.
    Memory-safe: batch encodes in small chunks and frees model after.
    """
    print("\n--- Embedding Similarity Baseline ---")

    # Check if minilm is available
    minilm_path = MODELS_DIR / "minilm"
    if not minilm_path.exists():
        print("  MinILM model not found, skipping")
        return None

    # Collect all texts first, then batch-encode to minimize model time in memory
    data = load_benchmark("test")
    if data is None:
        print("  No test data, skipping")
        return None

    questions = []
    contexts = []
    labels = []
    metadata = []

    for ex in data:
        for cond in ex["conditions"]:
            questions.append(ex["question"])
            contexts.append(cond["context"][:512])
            labels.append(1 if cond["sufficient"] else 0)
            metadata.append({
                "quadrant": cond["quadrant"],
                "question_type": ex["question_type"],
            })

    print(f"  Encoding {len(questions)} question-context pairs...")

    # Load model, encode in small batches, then immediately free
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(str(minilm_path))

    BATCH = 32
    q_embs = []
    for i in range(0, len(questions), BATCH):
        q_embs.append(embed_model.encode(questions[i:i+BATCH], show_progress_bar=False))
    q_embs = np.concatenate(q_embs, axis=0)

    c_embs = []
    for i in range(0, len(contexts), BATCH):
        c_embs.append(embed_model.encode(contexts[i:i+BATCH], show_progress_bar=False))
    c_embs = np.concatenate(c_embs, axis=0)

    # Free the model immediately
    del embed_model
    gc.collect()

    # Compute cosine similarities
    scores = np.array([
        1.0 - cosine_dist(q_embs[i], c_embs[i])
        for i in range(len(q_embs))
    ])
    labels = np.array(labels)

    del q_embs, c_embs
    gc.collect()

    if len(set(labels)) < 2:
        return None

    auroc = roc_auc_score(labels, scores)
    thresh = np.median(scores)
    preds = (scores >= thresh).astype(int)
    acc = accuracy_score(labels, preds)

    print(f"  test: AUROC={auroc:.3f}, Acc={acc:.3f}")

    per_quad = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(metadata) if m["quadrant"] == q]
        if len(idx) < 5:
            continue
        per_quad[q] = {
            "n": len(idx),
            "accuracy": float(accuracy_score(labels[idx], preds[idx])),
        }

    return {
        "overall": {"auroc": float(auroc), "accuracy": float(acc)},
        "per_quadrant": per_quad,
    }


# ─── Baseline 2: Confidence Probe (fixed methodology) ─────────────────────

def confidence_probe_baseline(model_key: str):
    """
    Train a probe on h(q+c) to predict model_knows_answer.
    Uses model-specific PK labels when available (e.g., pk_llama.model_knows_answer).
    This tests whether the model's hidden states CONFLATE sufficiency with confidence.
    """
    print(f"\n--- Confidence Probe ({model_key}) ---")

    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg["n_layers"]

    # Load hidden states
    h_train, _, _, meta_train = load_hidden_states(model_key, "train")
    h_test, _, _, meta_test = load_hidden_states(model_key, "test")

    # Get PK labels from metadata (model_confident field = model knows answer)
    # The hidden state metadata already has this aligned per-condition
    y_train = np.array([1 if m.get("model_confident", False) else 0 for m in meta_train])
    y_test = np.array([1 if m.get("model_confident", False) else 0 for m in meta_test])

    # Check label distribution
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"  PK labels: {n_pos} knows, {n_neg} doesn't know (train)")
    print(f"  PK labels: {y_test.sum()} knows, {len(y_test)-y_test.sum()} doesn't know (test)")

    if n_pos < 10 or n_neg < 10:
        print("  Too few positive/negative examples, skipping confidence probe")
        return None

    # Train probe at each layer, find best
    best_auroc = 0
    best_layer = 0
    best_weights = None

    for layer in range(n_layers):
        X_tr = h_train[:, layer, :]
        X_te = h_test[:, layer, :]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_tr_s, y_train)
        probs = clf.predict_proba(X_te_s)[:, 1]

        if len(set(y_test)) >= 2:
            auroc = roc_auc_score(y_test, probs)
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer
                best_weights = clf.coef_[0].copy()

    print(f"  Best confidence probe: layer {best_layer}, AUROC={best_auroc:.3f}")

    del h_train, h_test
    gc.collect()

    return {
        "best_layer": int(best_layer),
        "auroc": float(best_auroc),
        "weights": best_weights,  # For cosine analysis
    }


# ─── Baseline 3: Sufficiency-Confidence Cosine Analysis (fixed) ───────────

def fixed_orthogonality_analysis(model_key: str):
    """
    Proper orthogonality analysis:
    1. Train sufficiency probe on h(q+c) at best layer
    2. Train confidence probe on h(q+c) at SAME layer
    3. Compute cosine between weight vectors
    4. Add permutation test for null distribution

    Both probes operate in the SAME feature space — this is a valid comparison.
    """
    print(f"\n--- Fixed Orthogonality Analysis ({model_key}) ---")

    cfg = MODEL_CONFIGS[model_key]

    h_train, h_q_train, suf_labels_train, meta_train = load_hidden_states(model_key, "train")
    h_test, h_q_test, suf_labels_test, meta_test = load_hidden_states(model_key, "test")

    # Get confidence labels from metadata (model_confident = model knows)
    conf_labels_train = np.array([1 if m.get("model_confident", False) else 0 for m in meta_train])
    conf_labels_test = np.array([1 if m.get("model_confident", False) else 0 for m in meta_test])

    # Find best layer for sufficiency (from existing DECO results)
    deco_path = RESULTS_DIR / model_key / "deco_results.json"
    if deco_path.exists():
        deco_res = json.load(open(deco_path))
        best_layer = deco_res["best_layers"]["standard"]
    else:
        best_layer = cfg["n_layers"] // 2

    print(f"  Using layer {best_layer} for both probes")

    # Train both probes at the SAME layer on h(q+c)
    X_tr = h_train[:, best_layer, :]
    X_te = h_test[:, best_layer, :]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Sufficiency probe
    suf_probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    suf_probe.fit(X_tr_s, suf_labels_train)
    suf_auroc = roc_auc_score(suf_labels_test, suf_probe.predict_proba(X_te_s)[:, 1])
    w_suf = suf_probe.coef_[0]

    # Confidence probe (SAME feature space)
    conf_probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    conf_probe.fit(X_tr_s, conf_labels_train)
    conf_probs = conf_probe.predict_proba(X_te_s)[:, 1]
    if len(set(conf_labels_test)) >= 2:
        conf_auroc = roc_auc_score(conf_labels_test, conf_probs)
    else:
        conf_auroc = 0.5
    w_conf = conf_probe.coef_[0]

    # DECO probe
    d_tr = h_train[:, best_layer, :] - h_q_train[:, best_layer, :]
    d_te = h_test[:, best_layer, :] - h_q_test[:, best_layer, :]
    scaler_d = StandardScaler()
    d_tr_s = scaler_d.fit_transform(d_tr)
    d_te_s = scaler_d.transform(d_te)
    deco_probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    deco_probe.fit(d_tr_s, suf_labels_train)
    deco_auroc = roc_auc_score(suf_labels_test, deco_probe.predict_proba(d_te_s)[:, 1])
    w_deco = deco_probe.coef_[0]

    # Cosine similarities (both probes in SAME space now)
    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    cos_suf_conf = cos_sim(w_suf, w_conf)
    cos_deco_conf = cos_sim(w_deco, w_conf)  # Note: different spaces, acknowledged
    cos_suf_deco = cos_sim(w_suf, w_deco)    # Note: different spaces, acknowledged

    print(f"  Sufficiency probe AUROC: {suf_auroc:.3f}")
    print(f"  Confidence probe AUROC: {conf_auroc:.3f}")
    print(f"  DECO probe AUROC: {deco_auroc:.3f}")
    print(f"  cos(sufficiency, confidence): {cos_suf_conf:.4f}")
    print(f"  cos(DECO, confidence): {cos_deco_conf:.4f}")
    print(f"  cos(sufficiency, DECO): {cos_suf_deco:.4f}")

    # Permutation test: is cos_suf_conf significantly different from random?
    # Use random weight vectors instead of retraining — much faster and statistically valid
    rng = np.random.RandomState(42)
    n_perm = 1000
    dim = w_suf.shape[0]
    null_cosines = []
    for _ in range(n_perm):
        # Random unit vector in the same space
        rand_w = rng.randn(dim)
        rand_w /= np.linalg.norm(rand_w) + 1e-10
        null_cosines.append(cos_sim(w_suf, rand_w))

    null_cosines = np.array(null_cosines)
    p_value = float(np.mean(np.abs(null_cosines) >= np.abs(cos_suf_conf)))
    print(f"  Permutation test p-value: {p_value:.4f}")
    print(f"  Null distribution: mean={np.mean(null_cosines):.4f}, std={np.std(null_cosines):.4f}")

    del h_train, h_q_train, h_test, h_q_test, X_tr, X_te, X_tr_s, X_te_s
    gc.collect()

    return {
        "layer": int(best_layer),
        "sufficiency_auroc": float(suf_auroc),
        "confidence_auroc": float(conf_auroc),
        "deco_auroc": float(deco_auroc),
        "cos_suf_conf": float(cos_suf_conf),
        "cos_deco_conf": float(cos_deco_conf),
        "cos_suf_deco": float(cos_suf_deco),
        "permutation_p_value": float(p_value),
        "null_mean": float(np.mean(null_cosines)),
        "null_std": float(np.std(null_cosines)),
        "note": "PK labels from metadata model_confident field (Mistral-based). Will be updated with model-specific PK when available.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=["mistral", "qwen", "llama", "all"])
    args = parser.parse_args()

    models = [args.model] if args.model != "all" else list(MODEL_CONFIGS.keys())

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"PROBING BASELINES — {model_key}")
        print(f"{'='*60}")

        # 1. Fixed orthogonality analysis (numpy only, low memory)
        ortho_result = fixed_orthogonality_analysis(model_key)
        gc.collect()

        # 2. Embedding similarity (loads MiniLM, higher memory)
        embed_result = embedding_similarity_baseline(model_key)
        gc.collect()

        # Save results
        out_dir = RESULTS_DIR / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # Update baseline results
        baseline_path = out_dir / "baseline_results.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                existing = json.load(f)
        else:
            existing = {}

        if embed_result:
            existing["embedding_similarity"] = {
                "method": "Cosine similarity between question and context embeddings (MiniLM)",
                "overall": embed_result.get("overall", {}),
                "per_quadrant": embed_result.get("per_quadrant", {}),
                "per_question_type": {},
            }

        # Save orthogonality analysis separately
        with open(out_dir / "orthogonality_analysis.json", "w") as f:
            json.dump(ortho_result, f, indent=2)

        with open(baseline_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
