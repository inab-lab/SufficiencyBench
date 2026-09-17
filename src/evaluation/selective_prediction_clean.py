"""
selective_prediction_clean.py — Re-run selective prediction with clean-trained probe.

Filters mislabeled training examples (gold_leak + eli5_prefix) from train/val,
uses clean best layers from layer_analysis_clean.json, evaluates on the FULL
unchanged test set. Saves results to results/{model}/selective_prediction_clean.json
and a comparison summary to results/selective_prediction_clean_summary.json.

Usage:
  python src/evaluation/selective_prediction_clean.py --model all
  python src/evaluation/selective_prediction_clean.py --model mistral
"""

import json
import re
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, BENCH_DIR, MODEL_CONFIGS

VALIDATION_DIR = RESULTS_DIR / "validation"


def build_filter_ids():
    with open(VALIDATION_DIR / "context_verification.json") as f:
        verif = json.load(f)
    gold_leak_ids = set(x["id"] for x in verif
                        if "gold_still_in_insufficient" in x.get("flags", []))
    entity_leak_only_ids = set(x["id"] for x in verif
                               if any("entity_leak" in f for f in x.get("flags", []))
                               and "gold_still_in_insufficient" not in x.get("flags", []))

    filter_by_split = {"train": set(), "val": set()}
    for split in ("train", "val"):
        with open(BENCH_DIR / f"{split}.json") as f:
            data = json.load(f)
        for ex in data:
            insuf = next((c["context"] for c in ex["conditions"] if not c["sufficient"]), "")
            gold = ex["gold_answer"].lower().strip()
            if ex["id"] in gold_leak_ids:
                filter_by_split[split].add(ex["id"] + "_insuf")
            elif ex["id"] in entity_leak_only_ids:
                parts = re.split(r"(?<=[.!?])\s+", gold)
                first_sent = parts[0] if parts else ""
                has_prefix = len(gold[:50].strip()) > 15 and gold[:50] in insuf.lower()
                has_first = len(first_sent) > 20 and first_sent in insuf.lower()
                if has_prefix or has_first:
                    filter_by_split[split].add(ex["id"] + "_insuf")
    return filter_by_split


def load_split(model_key, split, filter_ids=None):
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy")
    h_q  = np.load(d / "h_question_only.npy")
    y    = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)
    if filter_ids:
        mask = np.array([m["id"] not in filter_ids for m in meta])
        return h_qc[mask], h_q[mask], y[mask], [m for m, k in zip(meta, mask) if k]
    return h_qc, h_q, y, meta


def evaluate_model(model_key, filter_by_split, clean_layers):
    print(f"\n{'='*60}")
    print(f"CLEAN SELECTIVE PREDICTION — {model_key.upper()}")

    h_qc_tr, h_q_tr, y_tr, meta_tr = load_split(model_key, "train", filter_by_split["train"])
    h_qc_te, h_q_te, y_te, meta_te = load_split(model_key, "test")  # unchanged

    best_layer = clean_layers[model_key]["best_val_layer"]
    print(f"Clean best layer: {best_layer}, train size: {len(y_tr)}, test size: {len(y_te)}")

    X_tr = (h_qc_tr[:, best_layer, :] - h_q_tr[:, best_layer, :]).astype(np.float32)
    X_te = (h_qc_te[:, best_layer, :] - h_q_te[:, best_layer, :]).astype(np.float32)

    scaler = StandardScaler()
    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(scaler.fit_transform(X_tr), y_tr)
    deco_scores = probe.predict_proba(scaler.transform(X_te))[:, 1]

    quadrants = [m["quadrant"] for m in meta_te]
    q3_mask   = np.array([q == "Q3" for q in quadrants])
    safe_mask = np.array([q in ("Q1", "Q2") for q in quadrants])

    # Sufficiency AUROC on full test
    auroc = float(roc_auc_score(y_te, deco_scores))
    print(f"Sufficiency AUROC (full test, clean probe): {auroc:.4f}")

    # Q3 separation AUROC: can the clean probe tell safe (Q1+Q2) from
    # dangerous (Q3, insufficient+confident) examples?
    is_safe = (~q3_mask).astype(int)
    sel = q3_mask | safe_mask
    if sel.sum() > 0 and len(set(is_safe[sel])) > 1:
        q3_sep_auroc = float(roc_auc_score(is_safe[sel], deco_scores[sel]))
    else:
        q3_sep_auroc = float("nan")
    print(f"Q3 separation AUROC (safe vs Q3, clean probe):  {q3_sep_auroc:.4f}")

    # Per-quadrant score distributions
    per_quadrant = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, qd in enumerate(quadrants) if qd == q]
        if not idx:
            continue
        qs = deco_scores[idx]
        per_quadrant[q] = {
            "n": len(idx),
            "mean": float(np.mean(qs)),
            "std": float(np.std(qs)),
            "median": float(np.median(qs)),
        }

    return {
        "model": model_key,
        "best_val_layer": int(best_layer),
        "n_train_clean": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "sufficiency_auroc": auroc,
        "q3_separation_auroc": q3_sep_auroc,
        "per_quadrant_scores": per_quadrant,
    }


def load_clean_layers():
    """Load clean best-layer selection produced by retrain_clean_probe.py."""
    with open(RESULTS_DIR / "layer_analysis_clean.json") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    print("Building filter ID sets (gold_leak + eli5_prefix)...")
    filter_by_split = build_filter_ids()
    for split, ids in filter_by_split.items():
        print(f"  {split}: {len(ids)} insufficient examples to filter")

    clean_layers = load_clean_layers()

    summary = {}
    for mk in models:
        try:
            res = evaluate_model(mk, filter_by_split, clean_layers)
            out_dir = RESULTS_DIR / mk
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "selective_prediction_clean.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Saved {out_path}")
            summary[mk] = res
        except Exception as e:
            print(f"Error on {mk}: {e}")
            import traceback
            traceback.print_exc()

    if summary:
        summary_path = RESULTS_DIR / "selective_prediction_clean_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCross-model summary saved to {summary_path}")

        print(f"\n{'Model':<10} {'Suf AUROC':>10} {'Q3 sep AUROC':>14}")
        for mk, res in summary.items():
            print(f"{mk:<10} {res['sufficiency_auroc']:>10.4f} "
                  f"{res['q3_separation_auroc']:>14.4f}")


if __name__ == "__main__":
    main()