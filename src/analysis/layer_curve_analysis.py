"""
layer_curve_analysis.py — Probe AUROC at every residual-stream layer.

Trains a CSP logistic-regression probe at each layer for Mistral, Llama,
and Qwen, reports AUROC on the held-out test split, marks the validation-
selected best layer, and saves:
  results/layer_analysis.json
  results/figures/layer_curve.pdf

Usage:
  python src/analysis/layer_curve_analysis.py
"""

import gc
import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, FIGURES_DIR, MODEL_CONFIGS

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

MODEL_COLORS = {"llama": "#1a2a52", "mistral": "#4a90c4", "qwen": "#8fa830"}
MODEL_LABELS = {"llama": "Llama 3.1 8B", "mistral": "Mistral 7B", "qwen": "Qwen 2.5 7B"}


def _load_csp_features(model_key, split, layer):
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy", mmap_mode="r")
    h_q  = np.load(d / "h_question_only.npy", mmap_mode="r")
    X = (h_qc[:, layer, :] - h_q[:, layer, :]).astype(np.float32)
    y = np.load(d / "labels.npy")
    return X, y


def run_layer_curve():
    results = {}

    for mk, cfg in MODEL_CONFIGS.items():
        n_layers = cfg["n_layers"]
        print(f"\n{mk.upper()} — sweeping {n_layers} layers")

        # Load full arrays once (train + test) to avoid repeated disk I/O
        d_tr = HIDDEN_STATES_DIR / mk / "train"
        d_te = HIDDEN_STATES_DIR / mk / "test"
        d_va = HIDDEN_STATES_DIR / mk / "val"

        h_qc_tr = np.load(d_tr / "h_with_context.npy")
        h_q_tr  = np.load(d_tr / "h_question_only.npy")
        y_tr    = np.load(d_tr / "labels.npy")
        h_qc_te = np.load(d_te / "h_with_context.npy")
        h_q_te  = np.load(d_te / "h_question_only.npy")
        y_te    = np.load(d_te / "labels.npy")
        h_qc_va = np.load(d_va / "h_with_context.npy")
        h_q_va  = np.load(d_va / "h_question_only.npy")
        y_va    = np.load(d_va / "labels.npy")

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
            "best_val_auroc": val_aurocs[best_val_layer],
            "test_at_best":   test_aurocs[best_val_layer],
        }
        print(f"  Best val layer: {best_val_layer}  "
              f"(val={val_aurocs[best_val_layer]:.4f}, test={test_aurocs[best_val_layer]:.4f})")

        del h_qc_tr, h_q_tr, h_qc_te, h_q_te, h_qc_va, h_q_va
        gc.collect()

    return results


def plot_layer_curve(results):
    fig, ax = plt.subplots(figsize=(7, 4))

    for mk, res in results.items():
        aurocs = res["test_aurocs"]
        best   = res["best_val_layer"]
        color  = MODEL_COLORS[mk]
        label  = MODEL_LABELS[mk]

        ax.plot(range(len(aurocs)), aurocs, color=color, linewidth=1.8, label=label)
        ax.scatter([best], [aurocs[best]], color=color, s=80, zorder=5,
                   marker="*", edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Residual-stream layer")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Probe AUROC by Layer (CSP logistic regression)")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    note = "★ = validation-selected best layer"
    ax.text(0.01, 0.02, note, transform=ax.transAxes,
            fontsize=8, color="grey", va="bottom")

    fig.tight_layout()
    out = FIGURES_DIR / "layer_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")
    return fig


def main():
    results = run_layer_curve()

    out_json = RESULTS_DIR / "layer_analysis.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved to {out_json}")

    plot_layer_curve(results)

    # Print summary table
    print("\nSummary (val-selected layer, test AUROC):")
    print(f"{'Model':<12} {'Best layer':>12} {'Val AUROC':>12} {'Test AUROC':>12}")
    for mk, res in results.items():
        print(f"{MODEL_LABELS[mk]:<12} {res['best_val_layer']:>12} "
              f"{res['best_val_auroc']:>12.4f} {res['test_at_best']:>12.4f}")


if __name__ == "__main__":
    main()
