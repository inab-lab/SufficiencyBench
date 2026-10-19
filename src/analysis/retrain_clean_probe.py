"""
retrain_clean_probe.py — Retrain CSP probe after filtering mislabeled training examples.

Filters two categories of mislabeled "insufficient" examples from train/val:
  1. gold_still_in_insufficient: gold answer verbatim in replacement context (324 total)
  2. eli5_prefix: first sentence of gold answer present in replacement context (209 total)

Test set is kept UNCHANGED for apples-to-apples AUROC comparison.

Saves results/layer_analysis_clean.json alongside original layer_analysis.json.

Usage:
  python src/analysis/retrain_clean_probe.py
"""

import gc
import json
import sys
import os
import re
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, BENCH_DIR, MODEL_CONFIGS

VALIDATION_DIR = RESULTS_DIR / "validation"


def _get_filter_ids():
    """Return {split: set of _insuf metadata IDs to remove from train and val}."""
    with open(VALIDATION_DIR / "context_verification.json") as f:
        verif = json.load(f)

    gold_leak_ids = set(x["id"] for x in verif
                        if "gold_still_in_insufficient" in x.get("flags", []))
    entity_leak_only_ids = set(x["id"] for x in verif
                               if any("entity_leak" in f for f in x.get("flags", []))
                               and "gold_still_in_insufficient" not in x.get("flags", []))

    filter_by_split = {"train": set(), "val": set()}

    for split in ("train", "val"):
        split_file = BENCH_DIR / f"{split}.json"
        with open(split_file) as f:
            data = json.load(f)
        for ex in data:
            insuf_ctx = next(
                (c["context"] for c in ex["conditions"] if not c["sufficient"]), ""
            )
            gold = ex["gold_answer"].lower().strip()
            ex_id = ex["id"]

            if ex_id in gold_leak_ids:
                filter_by_split[split].add(ex_id + "_insuf")
                continue

            if ex_id in entity_leak_only_ids:
                parts = re.split(r"(?<=[.!?])\s+", gold)
                first_sent = parts[0] if parts else ""
                has_prefix = len(gold[:50].strip()) > 15 and gold[:50] in insuf_ctx.lower()
                has_first = len(first_sent) > 20 and first_sent in insuf_ctx.lower()
                if has_prefix or has_first:
                    filter_by_split[split].add(ex_id + "_insuf")

    return filter_by_split


def _load_mask(meta_path, filter_ids):
    """Return boolean keep-mask for a split's metadata list."""
    with open(meta_path) as f:
        meta = json.load(f)
    mask = np.array([item["id"] not in filter_ids for item in meta])
    return mask


def run_clean_layer_curve(filter_by_split):
    results = {}

    for mk, cfg in MODEL_CONFIGS.items():
        n_layers = cfg["n_layers"]
        print(f"\n{mk.upper()} — sweeping {n_layers} layers (clean train/val)")

        d_tr = HIDDEN_STATES_DIR / mk / "train"
        d_te = HIDDEN_STATES_DIR / mk / "test"
        d_va = HIDDEN_STATES_DIR / mk / "val"

        mask_tr = _load_mask(d_tr / "metadata.json", filter_by_split["train"])
        mask_va = _load_mask(d_va / "metadata.json", filter_by_split["val"])
        # test mask = keep all
        n_te = len(json.load(open(d_te / "metadata.json")))
        mask_te = np.ones(n_te, dtype=bool)

        print(f"  Train: {mask_tr.sum()} / {len(mask_tr)} kept "
              f"(filtered {(~mask_tr).sum()})")
        print(f"  Val:   {mask_va.sum()} / {len(mask_va)} kept "
              f"(filtered {(~mask_va).sum()})")
        print(f"  Test:  {mask_te.sum()} (unchanged)")

        h_qc_tr = np.load(d_tr / "h_with_context.npy")[mask_tr]
        h_q_tr  = np.load(d_tr / "h_question_only.npy")[mask_tr]
        y_tr    = np.load(d_tr / "labels.npy")[mask_tr]

        h_qc_te = np.load(d_te / "h_with_context.npy")
        h_q_te  = np.load(d_te / "h_question_only.npy")
        y_te    = np.load(d_te / "labels.npy")

        h_qc_va = np.load(d_va / "h_with_context.npy")[mask_va]
        h_q_va  = np.load(d_va / "h_question_only.npy")[mask_va]
        y_va    = np.load(d_va / "labels.npy")[mask_va]

        test_aurocs = []
        val_aurocs  = []

        for layer in range(n_layers):
            X_tr = (h_qc_tr[:, layer, :] - h_q_tr[:, layer, :]).astype(np.float32)
            X_te = (h_qc_te[:, layer, :] - h_q_te[:, layer, :]).astype(np.float32)
            X_va = (h_qc_va[:, layer, :] - h_q_va[:, layer, :]).astype(np.float32)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            X_va_s = scaler.transform(X_va)

            probe = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            probe.fit(X_tr_s, y_tr)

            te_auc = float(roc_auc_score(y_te, probe.predict_proba(X_te_s)[:, 1]))
            va_auc = float(roc_auc_score(y_va, probe.predict_proba(X_va_s)[:, 1]))
            test_aurocs.append(te_auc)
            val_aurocs.append(va_auc)
            print(f"  Layer {layer:2d}: val={va_auc:.4f}  test={te_auc:.4f}")

        best_val_layer = int(np.argmax(val_aurocs))
        results[mk] = {
            "test_aurocs":    test_aurocs,
            "val_aurocs":     val_aurocs,
            "best_val_layer": best_val_layer,
            "best_val_auroc": float(val_aurocs[best_val_layer]),
            "test_at_best":   float(test_aurocs[best_val_layer]),
            "n_train_clean":  int(mask_tr.sum()),
            "n_train_filtered": int((~mask_tr).sum()),
        }
        print(f"  Best val layer: {best_val_layer}  "
              f"(val={val_aurocs[best_val_layer]:.4f}, "
              f"test={test_aurocs[best_val_layer]:.4f})")

        del h_qc_tr, h_q_tr, h_qc_te, h_q_te, h_qc_va, h_q_va
        gc.collect()

    return results


def main():
    print("Building filter ID sets...")
    filter_by_split = _get_filter_ids()
    for split, ids in filter_by_split.items():
        print(f"  {split}: {len(ids)} examples to filter")

    results = run_clean_layer_curve(filter_by_split)

    out_path = RESULTS_DIR / "layer_analysis_clean.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print comparison
    orig_path = RESULTS_DIR / "layer_analysis.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        print("\n=== AUROC comparison (same test set) ===")
        print(f"{'Model':<10} {'Original':>10} {'Clean':>10} {'Delta':>8}")
        for mk in results:
            orig_auc = orig[mk]["test_at_best"]
            clean_auc = results[mk]["test_at_best"]
            print(f"{mk:<10} {orig_auc:>10.4f} {clean_auc:>10.4f} {clean_auc-orig_auc:>+8.4f}")


if __name__ == "__main__":
    main()
