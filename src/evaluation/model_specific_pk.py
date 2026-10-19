"""
model_specific_pk.py — Model-Specific Parametric Knowledge Analysis

Compares Mistral-based PK labels (used for all models) with model-specific
PK labels (already stored in the benchmark as pk_llama / pk_qwen).

For each model:
1. Load the benchmark and extract model-specific PK labels
2. Compute agreement rate between Mistral PK and model-specific PK
3. Count how many examples change quadrant
4. Re-run orthogonality analysis using model-specific confidence labels
   (sufficiency probe vs. confidence probe with model-specific PK)

Usage:
  python src/evaluation/model_specific_pk.py --model llama --device cuda:0
  python src/evaluation/model_specific_pk.py --model qwen --device cuda:0
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import (
    BENCH_DIR, HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS,
)


def load_benchmark(split: str = "test") -> list:
    """Load SufficiencyBench split."""
    path = BENCH_DIR / f"{split}.json"
    with open(path) as f:
        return json.load(f)


def build_condition_lookup(bench_data: list, model_key: str) -> dict:
    """
    Build a mapping from condition_id to model-specific PK info.

    Each entry contains:
      - mistral_confident: bool (original Mistral-based label)
      - model_confident: bool (model-specific PK label)
      - mistral_quadrant: str
      - model_quadrant: str
      - sufficient: bool
    """
    pk_field = f"pk_{model_key}"
    quad_field = f"quadrant_{model_key}"

    lookup = {}
    for ex in bench_data:
        pk_info = ex.get(pk_field)
        if pk_info is None:
            continue

        model_knows = pk_info["model_knows_answer"]

        for cond in ex["conditions"]:
            cid = cond["condition_id"]
            model_quad = cond.get(quad_field)

            lookup[cid] = {
                "mistral_confident": cond["model_confident"],
                "model_confident": model_knows,
                "mistral_quadrant": cond["quadrant"],
                "model_quadrant": model_quad,
                "sufficient": cond["sufficient"],
                "model_pk_confidence": pk_info.get("confidence"),
                "closed_book_answer": pk_info.get("closed_book_answer", ""),
            }

    return lookup


def compute_agreement(lookup: dict) -> dict:
    """Compute agreement between Mistral PK and model-specific PK."""
    n_total = len(lookup)
    n_agree_confident = 0
    n_agree_quadrant = 0
    quadrant_changes = {"Q1->Q2": 0, "Q2->Q1": 0, "Q3->Q4": 0, "Q4->Q3": 0}

    mistral_confident_count = 0
    model_confident_count = 0

    for cid, info in lookup.items():
        mc = info["mistral_confident"]
        sc = info["model_confident"]

        if mc:
            mistral_confident_count += 1
        if sc:
            model_confident_count += 1

        if mc == sc:
            n_agree_confident += 1

        mq = info["mistral_quadrant"]
        sq = info["model_quadrant"]
        if mq == sq:
            n_agree_quadrant += 1
        else:
            change_key = f"{mq}->{sq}"
            if change_key in quadrant_changes:
                quadrant_changes[change_key] += 1

    return {
        "n_total": n_total,
        "confidence_agreement": n_agree_confident / n_total if n_total > 0 else 0,
        "quadrant_agreement": n_agree_quadrant / n_total if n_total > 0 else 0,
        "n_quadrant_changed": n_total - n_agree_quadrant,
        "quadrant_change_breakdown": quadrant_changes,
        "mistral_confident_rate": mistral_confident_count / n_total if n_total > 0 else 0,
        "model_confident_rate": model_confident_count / n_total if n_total > 0 else 0,
    }


def load_hidden_states(model_key: str, split: str):
    """Load hidden states, labels, and metadata for a model/split."""
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy")
    h_q = np.load(d / "h_question_only.npy")
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    return h_qc, h_q, labels, metadata


def get_best_layer(model_key: str) -> int:
    """Read the best DECO layer from saved results."""
    results_path = RESULTS_DIR / model_key / "deco_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        return results["best_layers"]["DECO"]
    # Fallback: use layer 8 for llama/mistral, 7 for qwen
    cfg = MODEL_CONFIGS[model_key]
    return cfg["n_layers"] // 4


def run_orthogonality_with_model_pk(
    model_key: str, lookup: dict
) -> dict:
    """
    Re-run the orthogonality analysis using model-specific PK labels
    instead of Mistral-based PK labels.

    Returns cosine similarities between:
      - sufficiency direction and confidence direction (model-specific)
      - DECO direction and confidence direction (model-specific)
    For comparison, also computes with original Mistral-based labels.
    """
    print("\nLoading hidden states...")
    h_qc_train, h_q_train, y_train, meta_train = load_hidden_states(model_key, "train")
    h_qc_test, h_q_test, y_test, meta_test = load_hidden_states(model_key, "test")

    best_layer = get_best_layer(model_key)
    print(f"Using best DECO layer: L{best_layer}")
    print(f"Data: {len(y_train)} train, {len(y_test)} test")

    # Build model-specific confidence labels by matching condition IDs
    def get_confidence_labels(metadata, use_model_specific: bool):
        labels = []
        for m in metadata:
            cid = m["id"]
            if cid in lookup:
                if use_model_specific:
                    labels.append(1 if lookup[cid]["model_confident"] else 0)
                else:
                    labels.append(1 if lookup[cid]["mistral_confident"] else 0)
            else:
                # Fallback to metadata's model_confident (Mistral-based)
                labels.append(1 if m["model_confident"] else 0)
        return np.array(labels)

    # Also build model-specific quadrant labels for per-quadrant analysis
    def get_model_quadrants(metadata):
        quads = []
        for m in metadata:
            cid = m["id"]
            if cid in lookup and lookup[cid]["model_quadrant"] is not None:
                quads.append(lookup[cid]["model_quadrant"])
            else:
                quads.append(m["quadrant"])
        return quads

    y_conf_mistral_train = get_confidence_labels(meta_train, use_model_specific=False)
    y_conf_mistral_test = get_confidence_labels(meta_test, use_model_specific=False)
    y_conf_model_train = get_confidence_labels(meta_train, use_model_specific=True)
    y_conf_model_test = get_confidence_labels(meta_test, use_model_specific=True)

    model_quadrants_test = get_model_quadrants(meta_test)

    print(f"\nConfidence label distribution:")
    print(f"  Mistral PK — train: {y_conf_mistral_train.sum()}/{len(y_conf_mistral_train)} confident")
    print(f"  Model PK   — train: {y_conf_model_train.sum()}/{len(y_conf_model_train)} confident")
    print(f"  Mistral PK — test:  {y_conf_mistral_test.sum()}/{len(y_conf_mistral_test)} confident")
    print(f"  Model PK   — test:  {y_conf_model_test.sum()}/{len(y_conf_model_test)} confident")

    # Extract features at best layer
    X_qc_train = h_qc_train[:, best_layer, :]
    X_qc_test = h_qc_test[:, best_layer, :]
    X_diff_train = h_qc_train[:, best_layer, :] - h_q_train[:, best_layer, :]
    X_diff_test = h_qc_test[:, best_layer, :] - h_q_test[:, best_layer, :]

    results = {}

    for label_name, y_conf_tr, y_conf_te in [
        ("mistral_pk", y_conf_mistral_train, y_conf_mistral_test),
        ("model_specific_pk", y_conf_model_train, y_conf_model_test),
    ]:
        print(f"\n--- Orthogonality with {label_name} ---")

        # Check if confidence labels have both classes
        if len(set(y_conf_tr)) < 2:
            print(f"  WARNING: {label_name} train labels have only one class. Skipping.")
            results[label_name] = {"error": "single class in training labels"}
            continue

        scaler = StandardScaler()

        # Standard sufficiency probe: h(q+c) -> sufficiency
        X_tr = scaler.fit_transform(X_qc_train)
        X_te = scaler.transform(X_qc_test)
        probe_suf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        probe_suf.fit(X_tr, y_train)
        dir_suf = probe_suf.coef_[0]
        dir_suf = dir_suf / np.linalg.norm(dir_suf)

        suf_auroc = roc_auc_score(
            y_test, probe_suf.predict_proba(X_te)[:, 1]
        ) if len(set(y_test)) > 1 else 0.5

        # DECO sufficiency probe: h(q+c) - h(q) -> sufficiency
        X_tr_d = scaler.fit_transform(X_diff_train)
        X_te_d = scaler.transform(X_diff_test)
        probe_deco = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        probe_deco.fit(X_tr_d, y_train)
        dir_deco = probe_deco.coef_[0]
        dir_deco = dir_deco / np.linalg.norm(dir_deco)

        deco_auroc = roc_auc_score(
            y_test, probe_deco.predict_proba(X_te_d)[:, 1]
        ) if len(set(y_test)) > 1 else 0.5

        # Confidence probe: h(q+c) -> model_confident
        X_tr_c = scaler.fit_transform(X_qc_train)
        X_te_c = scaler.transform(X_qc_test)
        probe_conf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        probe_conf.fit(X_tr_c, y_conf_tr)
        dir_conf = probe_conf.coef_[0]
        norm_conf = np.linalg.norm(dir_conf)
        if norm_conf > 1e-10:
            dir_conf = dir_conf / norm_conf
        else:
            print("  WARNING: confidence direction has zero norm")
            dir_conf = np.zeros_like(dir_conf)

        conf_auroc = roc_auc_score(
            y_conf_te, probe_conf.predict_proba(X_te_c)[:, 1]
        ) if len(set(y_conf_te)) > 1 else 0.5

        # Cosine similarities
        cos_std_conf = float(np.dot(dir_suf, dir_conf))
        cos_deco_conf = float(np.dot(dir_deco, dir_conf))
        cos_std_deco = float(np.dot(dir_suf, dir_deco))

        print(f"  cos(standard, confidence):  {cos_std_conf:.4f}")
        print(f"  cos(DECO, confidence):      {cos_deco_conf:.4f}")
        print(f"  cos(standard, DECO):        {cos_std_deco:.4f}")
        print(f"  Sufficiency AUROC (std):    {suf_auroc:.4f}")
        print(f"  Sufficiency AUROC (DECO):   {deco_auroc:.4f}")
        print(f"  Confidence AUROC:           {conf_auroc:.4f}")

        results[label_name] = {
            "cos_standard_confidence": cos_std_conf,
            "cos_deco_confidence": cos_deco_conf,
            "cos_standard_deco": cos_std_deco,
            "sufficiency_auroc_standard": suf_auroc,
            "sufficiency_auroc_deco": deco_auroc,
            "confidence_auroc": conf_auroc,
            "analysis_layer": best_layer,
        }

    # Model-specific quadrant distribution
    from collections import Counter
    model_quad_counts = Counter(model_quadrants_test)
    mistral_quad_counts = Counter(m["quadrant"] for m in meta_test)

    results["quadrant_distribution"] = {
        "mistral": dict(mistral_quad_counts),
        "model_specific": dict(model_quad_counts),
    }

    print(f"\nQuadrant distribution (test set):")
    print(f"  Mistral:        {dict(mistral_quad_counts)}")
    print(f"  Model-specific: {dict(model_quad_counts)}")

    return results


def run_model_specific_pk(model_key: str = "llama", device: str = "cuda:0"):
    """
    Run model-specific PK analysis: agreement, quadrant changes, and
    orthogonality re-test.
    """
    if model_key not in ("llama", "qwen"):
        print(f"Error: --model must be 'llama' or 'qwen' (not '{model_key}')")
        print("Mistral is the reference model; its PK is already the baseline.")
        sys.exit(1)

    cfg = MODEL_CONFIGS[model_key]

    print(f"\n{'='*70}")
    print(f"MODEL-SPECIFIC PK ANALYSIS — {model_key}")
    print(f"{'='*70}")

    # Step 1: Load benchmark and build lookup
    print("\nStep 1: Loading benchmark and building condition lookup...")
    bench_test = load_benchmark("test")
    bench_train = load_benchmark("train")

    lookup_test = build_condition_lookup(bench_test, model_key)
    lookup_train = build_condition_lookup(bench_train, model_key)
    lookup_all = {**lookup_train, **lookup_test}

    print(f"  Test conditions with model-specific PK: {len(lookup_test)}")
    print(f"  Train conditions with model-specific PK: {len(lookup_train)}")

    # Step 2: Agreement analysis
    print(f"\nStep 2: Computing agreement between Mistral PK and {model_key} PK...")
    agreement_test = compute_agreement(lookup_test)
    agreement_train = compute_agreement(lookup_train)

    print(f"\n  TEST SET:")
    print(f"    Confidence agreement: {agreement_test['confidence_agreement']:.4f}")
    print(f"    Quadrant agreement:   {agreement_test['quadrant_agreement']:.4f}")
    print(f"    Quadrants changed:    {agreement_test['n_quadrant_changed']} / {agreement_test['n_total']}")
    print(f"    Mistral confident rate: {agreement_test['mistral_confident_rate']:.4f}")
    print(f"    {model_key} confident rate:  {agreement_test['model_confident_rate']:.4f}")
    print(f"    Change breakdown:     {agreement_test['quadrant_change_breakdown']}")

    print(f"\n  TRAIN SET:")
    print(f"    Confidence agreement: {agreement_train['confidence_agreement']:.4f}")
    print(f"    Quadrant agreement:   {agreement_train['quadrant_agreement']:.4f}")
    print(f"    Quadrants changed:    {agreement_train['n_quadrant_changed']} / {agreement_train['n_total']}")

    # Step 3: Orthogonality re-test with model-specific labels
    print(f"\nStep 3: Orthogonality re-test with model-specific PK labels...")
    ortho_results = run_orthogonality_with_model_pk(model_key, lookup_all)

    # Step 4: Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    mistral_cos = ortho_results.get("mistral_pk", {}).get("cos_standard_confidence", "N/A")
    model_cos = ortho_results.get("model_specific_pk", {}).get("cos_standard_confidence", "N/A")
    mistral_deco_cos = ortho_results.get("mistral_pk", {}).get("cos_deco_confidence", "N/A")
    model_deco_cos = ortho_results.get("model_specific_pk", {}).get("cos_deco_confidence", "N/A")

    print(f"\n  cos(standard, confidence):")
    print(f"    Mistral PK labels:        {mistral_cos}")
    print(f"    {model_key}-specific PK labels: {model_cos}")

    print(f"\n  cos(DECO, confidence):")
    print(f"    Mistral PK labels:        {mistral_deco_cos}")
    print(f"    {model_key}-specific PK labels: {model_deco_cos}")

    ortho_holds = True
    if isinstance(model_cos, float) and isinstance(model_deco_cos, float):
        if abs(model_cos) < 0.3 and abs(model_deco_cos) < 0.3:
            print(f"\n  FINDING: Orthogonality HOLDS with {model_key}-specific PK labels.")
            print(f"  Both standard and DECO directions are orthogonal to confidence.")
        elif abs(model_cos) > 0.5:
            ortho_holds = False
            print(f"\n  FINDING: With {model_key}-specific PK labels, standard probe")
            print(f"  shows higher correlation with confidence (conflation detected).")
        else:
            print(f"\n  FINDING: Results are intermediate with {model_key}-specific PK labels.")
    else:
        print(f"\n  Could not determine orthogonality (missing results).")

    # Save results
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "model": model_key,
        "agreement": {
            "test": agreement_test,
            "train": agreement_train,
        },
        "orthogonality": ortho_results,
        "orthogonality_holds_with_model_pk": ortho_holds,
    }

    out_path = out_dir / "model_specific_pk.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model-specific PK analysis: compare Mistral-based vs "
                    "model-specific parametric knowledge labels"
    )
    parser.add_argument(
        "--model", default="llama", choices=["llama", "qwen"],
        help="Model to analyze (llama or qwen; mistral is the reference)"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device (not used for probe training, kept for interface consistency)"
    )
    args = parser.parse_args()
    run_model_specific_pk(model_key=args.model, device=args.device)
