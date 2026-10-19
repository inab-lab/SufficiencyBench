"""
deco.py — DECO / CSP: Decoupled (Context-Subtraction) Sufficiency Probing.

The core method contribution of Paper 1 (SufficiencyBench).

  Standard probe : probe(h(q+c))
      Problem: conflates context-sufficiency with the model's baseline
      (parametric) confidence.

  DECO / CSP     : probe(d), where d = h(q+c) - h(q)
      Subtracting the question-only state isolates what the context ADDS.
      Useful information added -> sufficient; noise / nothing added -> insufficient.
      The model's baseline confidence is subtracted out.

Both probes come in two classifier variants: "logreg" (sklearn LogisticRegression)
and "neural" (a small MLP).  Features are z-scored (StandardScaler) inside a
Pipeline so the trained probe can be applied to raw hidden states downstream
(e.g. in deco_rag.py).  A per-layer sweep is run; the best layer is chosen on the
VALIDATION split and the corresponding TEST AUROC is reported.

Public API (imported elsewhere, see README / deco_rag.py):
  from methods.deco import train_deco_probe, evaluate_deco

Usage:
  python src/methods/deco.py --model mistral
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


def make_classifier(kind: str = "logreg"):
    """Return an (unfitted) sklearn Pipeline: StandardScaler -> classifier.

    Two variants per the paper:
      "logreg" -> LogisticRegression
      "neural" -> small MLP
    Wrapping the scaler in the pipeline lets callers apply the probe directly to
    raw hidden states (deco_rag.py passes unscaled features).
    """
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif kind == "neural":
        # TODO(reconstruct): exact MLP architecture used for the paper's "Neural"
        # variant is not recoverable from the surviving code; this is a small,
        # reasonable default (single hidden layer). Tune if reproducing the
        # Neural-probe numbers specifically.
        clf = MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown classifier kind: {kind}")
    return make_pipeline(StandardScaler(), clf)


def load_data(model_key: str, split: str):
    """Load cached hidden states, labels, and metadata for one split."""
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy")   # (N, n_layers, hidden_dim)
    h_q = np.load(d / "h_question_only.npy")    # (N, n_layers, hidden_dim)
    labels = np.load(d / "labels.npy")           # (N,)
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    return h_qc, h_q, labels, metadata


def train_deco_probe(model_key: str = "mistral", clf_kind: str = "logreg"):
    """Train the DECO and Standard probes; select the best layer on validation.

    Returns (deco_probe, std_probe, best_layer) — the signature used by
    deco_rag.py. Both probes are Pipelines (scaler + classifier) fitted at the
    same `best_layer`, chosen by DECO validation AUROC so the delta representation
    drives layer selection. `deco_probe` expects the delta d = h(q+c) - h(q);
    `std_probe` expects h(q+c).
    """
    h_qc_train, h_q_train, y_train, _ = load_data(model_key, "train")
    h_qc_val, h_q_val, y_val, _ = load_data(model_key, "val")
    n_layers = h_qc_train.shape[1]

    best_layer, best_val_auroc = 0, -1.0
    for layer in range(n_layers):
        d_tr = h_qc_train[:, layer, :] - h_q_train[:, layer, :]
        d_val = h_qc_val[:, layer, :] - h_q_val[:, layer, :]
        probe = make_classifier(clf_kind)
        probe.fit(d_tr, y_train)
        auroc = roc_auc_score(y_val, probe.predict_proba(d_val)[:, 1])
        if auroc > best_val_auroc:
            best_val_auroc, best_layer = auroc, layer

    deco_probe = make_classifier(clf_kind)
    deco_probe.fit(
        h_qc_train[:, best_layer, :] - h_q_train[:, best_layer, :], y_train
    )
    std_probe = make_classifier(clf_kind)
    std_probe.fit(h_qc_train[:, best_layer, :], y_train)

    print(f"[train_deco_probe] {model_key}: best layer {best_layer} "
          f"(val AUROC={best_val_auroc:.4f})")
    return deco_probe, std_probe, best_layer


def evaluate_deco(probe, d_test, labels_test) -> float:
    """Test AUROC for a fitted DECO probe on delta features (README API)."""
    y_prob = probe.predict_proba(d_test)[:, 1]
    return float(roc_auc_score(labels_test, y_prob))


def evaluate_probe(name, X_train, y_train, X_test, y_test,
                   test_metadata, clf_kind: str = "logreg"):
    """Train a probe and evaluate overall + per-quadrant + per-question-type."""
    probe = make_classifier(clf_kind)
    probe.fit(X_train, y_train)

    y_prob = probe.predict_proba(X_test)[:, 1]
    y_pred = probe.predict(X_test)

    results = {
        "method": name,
        "overall": {
            "auroc": float(roc_auc_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        },
        "per_quadrant": {},
        "per_question_type": {},
    }

    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(test_metadata) if m["quadrant"] == quad]
        if len(idx) < 10 or len(set(y_test[idx])) < 2:
            results["per_quadrant"][quad] = {"n": len(idx), "note": "too few or single class"}
            continue
        q_true, q_prob, q_pred = y_test[idx], y_prob[idx], y_pred[idx]
        results["per_quadrant"][quad] = {
            "n": len(idx),
            "auroc": float(roc_auc_score(q_true, q_prob)) if len(set(q_true)) > 1 else None,
            "f1": float(f1_score(q_true, q_pred, zero_division=0)),
            "accuracy": float(accuracy_score(q_true, q_pred)),
        }

    for qtype in ["factual", "multi_hop", "comparative", "subjective"]:
        idx = [i for i, m in enumerate(test_metadata) if m["question_type"] == qtype]
        if len(idx) < 10 or len(set(y_test[idx])) < 2:
            continue
        results["per_question_type"][qtype] = {
            "n": len(idx),
            "auroc": float(roc_auc_score(y_test[idx], y_prob[idx])),
            "f1": float(f1_score(y_test[idx], y_pred[idx], zero_division=0)),
        }

    return results, probe, None


def run_all_methods(model_key: str = "mistral", clf_kind: str = "logreg"):
    """Full per-layer sweep for Standard / DECO / confidence-only probes.

    Best layer per method is selected on the validation split; the reported
    numbers (per_quadrant, correlation, layer_sweep, ...) are on the test split.
    """
    h_qc_train, h_q_train, y_train, meta_train = load_data(model_key, "train")
    h_qc_val, h_q_val, y_val, meta_val = load_data(model_key, "val")
    h_qc_test, h_q_test, y_test, meta_test = load_data(model_key, "test")

    n_layers = h_qc_train.shape[1]
    layers_to_eval = list(range(n_layers))

    # layer_results[layer][method] -> TEST results dict; val_auroc for selection.
    layer_results = {}
    val_auroc = {}
    print(f"\n=== Evaluating {model_key} ({clf_kind}) ===")
    for l in layers_to_eval:
        feats = {
            "standard": (
                h_qc_train[:, l, :], h_qc_val[:, l, :], h_qc_test[:, l, :]),
            "DECO": (
                h_qc_train[:, l, :] - h_q_train[:, l, :],
                h_qc_val[:, l, :] - h_q_val[:, l, :],
                h_qc_test[:, l, :] - h_q_test[:, l, :]),
            "confidence": (
                h_q_train[:, l, :], h_q_val[:, l, :], h_q_test[:, l, :]),
        }
        layer_results[l] = {}
        val_auroc[l] = {}
        for method, (X_tr, X_val, X_te) in feats.items():
            res_test, _, _ = evaluate_probe(
                f"{method}_L{l}", X_tr, y_train, X_te, y_test, meta_test, clf_kind)
            res_val, _, _ = evaluate_probe(
                f"{method}_L{l}", X_tr, y_train, X_val, y_val, meta_val, clf_kind)
            layer_results[l][method] = res_test
            val_auroc[l][method] = res_val["overall"]["auroc"]
        print(f"Layer {l:2d}  "
              f"Std={layer_results[l]['standard']['overall']['auroc']:.4f}  "
              f"DECO={layer_results[l]['DECO']['overall']['auroc']:.4f}")

    # Best layer chosen on VALIDATION AUROC.
    best_std_layer = max(layers_to_eval, key=lambda l: val_auroc[l]["standard"])
    best_deco_layer = max(layers_to_eval, key=lambda l: val_auroc[l]["DECO"])
    best_conf_layer = max(layers_to_eval, key=lambda l: val_auroc[l]["confidence"])

    best_std = layer_results[best_std_layer]["standard"]
    best_deco = layer_results[best_deco_layer]["DECO"]
    best_conf = layer_results[best_conf_layer]["confidence"]

    print(f"\nBest layers (val-selected): "
          f"standard=L{best_std_layer}, DECO=L{best_deco_layer}, "
          f"confidence=L{best_conf_layer}")
    print(f"Test AUROC: standard={best_std['overall']['auroc']:.4f}  "
          f"DECO={best_deco['overall']['auroc']:.4f}  "
          f"confidence={best_conf['overall']['auroc']:.4f}")

    # ---- THE KEY COMPARISON: per-quadrant ----
    print(f"\n{'='*70}")
    print("2x2 QUADRANT RESULTS (Main finding)")
    print(f"{'='*70}")

    quad_descs = {
        "Q1": "Suf+Conf (easy)",
        "Q2": "Suf+Uncert (hard)",
        "Q3": "Insuf+Conf (DANGEROUS)",
        "Q4": "Insuf+Uncert (easy)",
    }

    header = f"{'Quadrant':<30s} {'Standard':>10s} {'DECO':>10s} {'Conf-Only':>12s} {'n':>6s}"
    print(header)
    print("-" * len(header))

    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        def get_acc(res, q):
            entry = res.get("per_quadrant", {}).get(q, {})
            acc = entry.get("accuracy")
            return f"{acc:.3f}" if acc is not None else "N/A"

        def get_n(res, q):
            return str(res.get("per_quadrant", {}).get(q, {}).get("n", 0))

        print(f"{quad_descs[quad]:<30s} "
              f"{get_acc(best_std, quad):>10s} "
              f"{get_acc(best_deco, quad):>10s} "
              f"{get_acc(best_conf, quad):>12s} "
              f"{get_n(best_std, quad):>6s}")

    # ---- CORRELATION ANALYSIS ----
    print(f"\n{'='*70}")
    print("SUFFICIENCY <-> CONFIDENCE DIRECTION CORRELATION")
    print(f"{'='*70}")

    # Use best DECO layer for correlation analysis
    analysis_layer = best_deco_layer

    # Build confidence labels from metadata (model_confident, not sufficiency)
    y_conf_train = np.array([
        1 if m["model_confident"] else 0 for m in meta_train
    ])
    y_conf_test = np.array([
        1 if m["model_confident"] else 0 for m in meta_test
    ])

    # Train probes to get weight vectors (directions)
    scaler = StandardScaler()

    # Standard sufficiency direction: trained on h(q+c) to predict sufficiency
    X = scaler.fit_transform(h_qc_train[:, analysis_layer, :])
    probe_std = LogisticRegression(max_iter=1000, random_state=42)
    probe_std.fit(X, y_train)
    dir_std = probe_std.coef_[0]
    dir_std = dir_std / np.linalg.norm(dir_std)

    # DECO sufficiency direction: trained on h(q+c)-h(q) to predict sufficiency
    X = scaler.fit_transform(
        h_qc_train[:, analysis_layer, :] - h_q_train[:, analysis_layer, :]
    )
    probe_deco = LogisticRegression(max_iter=1000, random_state=42)
    probe_deco.fit(X, y_train)
    dir_deco = probe_deco.coef_[0]
    dir_deco = dir_deco / np.linalg.norm(dir_deco)

    # Confidence direction: trained on h(q+c) to predict model_confident
    # This is the direction that captures "does the model know this already?"
    X = scaler.fit_transform(h_qc_train[:, analysis_layer, :])
    probe_conf = LogisticRegression(max_iter=1000, random_state=42)
    probe_conf.fit(X, y_conf_train)
    dir_conf = probe_conf.coef_[0]
    norm_conf = np.linalg.norm(dir_conf)
    if norm_conf > 1e-10:
        dir_conf = dir_conf / norm_conf
    else:
        print("  WARNING: confidence direction has zero norm — cannot compute correlation")
        dir_conf = np.zeros_like(dir_conf)

    # Also train confidence probe on h(q) alone for comparison
    X_q = scaler.fit_transform(h_q_train[:, analysis_layer, :])
    probe_conf_q = LogisticRegression(max_iter=1000, random_state=42)
    probe_conf_q.fit(X_q, y_conf_train)
    # Check how well h(q) predicts confidence
    conf_q_auroc = roc_auc_score(
        y_conf_test,
        probe_conf_q.predict_proba(
            scaler.transform(h_q_test[:, analysis_layer, :])
        )[:, 1]
    ) if len(set(y_conf_test)) > 1 else 0.5
    print(f"\n  Confidence probe on h(q) alone AUROC: {conf_q_auroc:.4f}")
    print(f"  (This measures how well question-only states predict parametric knowledge)")

    cos_std_conf = float(np.dot(dir_std, dir_conf))
    cos_deco_conf = float(np.dot(dir_deco, dir_conf))
    cos_std_deco = float(np.dot(dir_std, dir_deco))

    print(f"cos(standard, confidence):  {cos_std_conf:.4f}  "
          f"{'<-- HIGH = conflation!' if abs(cos_std_conf) > 0.5 else ''}")
    print(f"cos(DECO, confidence):      {cos_deco_conf:.4f}  "
          f"{'<-- LOW = decoupled!' if abs(cos_deco_conf) < 0.5 else ''}")
    print(f"cos(standard, DECO):        {cos_std_deco:.4f}")

    print(f"\nINTERPRETATION:")
    if abs(cos_std_conf) > 0.7:
        print("  [CONFIRMED] Standard probing direction ≈ confidence direction.")
        print("  Standard probes detect confidence, NOT sufficiency.")
    elif abs(cos_std_conf) > 0.5:
        print("  [PARTIAL] Standard probing has moderate correlation with confidence.")
    else:
        print("  [UNEXPECTED] Standard probing is NOT strongly correlated with confidence.")
        print("  The conflation hypothesis may be weaker than expected.")

    if abs(cos_deco_conf) < 0.4:
        print("  [CONFIRMED] DECO successfully decouples from confidence!")
    elif abs(cos_deco_conf) < 0.6:
        print("  [PARTIAL] DECO reduces confidence correlation but doesn't fully decouple.")
    else:
        print("  [WARNING] DECO still correlated with confidence. Subtraction may be insufficient.")

    # ---- Save results ----
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "model": model_key,
        "best_layers": {
            "standard": best_std_layer,
            "DECO": best_deco_layer,
            "confidence": best_conf_layer,
        },
        "best_overall": {
            "standard": best_std["overall"],
            "DECO": best_deco["overall"],
            "confidence": best_conf["overall"],
        },
        "best_per_quadrant": {
            "standard": best_std["per_quadrant"],
            "DECO": best_deco["per_quadrant"],
            "confidence": best_conf["per_quadrant"],
        },
        "best_per_question_type": {
            "standard": best_std.get("per_question_type", {}),
            "DECO": best_deco.get("per_question_type", {}),
            "confidence": best_conf.get("per_question_type", {}),
        },
        "correlation": {
            "cos_standard_confidence": cos_std_conf,
            "cos_deco_confidence": cos_deco_conf,
            "cos_standard_deco": cos_std_deco,
            "analysis_layer": analysis_layer,
        },
        "layer_sweep": {
            str(l): {
                method: layer_results[l][method]["overall"]
                for method in ["standard", "DECO", "confidence"]
            }
            for l in layers_to_eval
        },
    }

    # Write a clf-specific file so logreg and neural runs don't clobber each
    # other (Table 1 reports both variants). For backward compatibility the
    # logreg run also writes the canonical all_results.json / correlation.json
    # that downstream figure/table code reads.
    with open(out_dir / f"all_results_{clf_kind}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    if clf_kind == "logreg":
        with open(out_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        with open(out_dir / "correlation.json", "w") as f:
            json.dump(all_results["correlation"], f, indent=2)

    print(f"\nSaved results to {out_dir} (all_results_{clf_kind}.json)")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--clf", default="logreg", choices=["logreg", "neural"])
    args = parser.parse_args()
    run_all_methods(args.model, args.clf)


if __name__ == "__main__":
    main()