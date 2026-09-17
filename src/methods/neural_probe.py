"""
neural_probe.py

Implements a neural network probe following the Probing-RAG (NAACL 2025) architecture.
Tests whether nonlinear probes improve sufficiency detection over logistic regression.

Architecture: LayerNorm → FC(512) → SiLU → Dropout → FC(1) → Sigmoid

Usage:
  python src/methods/neural_probe.py
  python src/methods/neural_probe.py --models llama mistral qwen
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


class ImprovedProbe(nn.Module):
    """Probing-RAG style neural probe."""
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(model_key, split):
    base = HIDDEN_STATES_DIR / model_key / split
    h_ctx = np.load(base / "h_with_context.npy")
    h_q = np.load(base / "h_question_only.npy")
    labels = np.load(base / "labels.npy")
    with open(base / "metadata.json") as f:
        metadata = json.load(f)
    return h_ctx, h_q, labels, metadata


def train_neural_probe(X_train, y_train, X_val, y_val, input_dim,
                       epochs=50, lr=1e-3, batch_size=256, patience=7):
    """Train neural probe with early stopping on val AUROC."""
    device = "cpu"
    model = ImprovedProbe(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_np = y_val

    best_val_auroc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
            if len(set(y_val_np)) > 1:
                val_auroc = roc_auc_score(y_val_np, val_probs)
            else:
                val_auroc = 0.5

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_auroc


def evaluate_per_quadrant(y_true, y_prob, y_pred, metadata):
    result = {}
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(metadata) if m["quadrant"] == quad]
        if len(idx) < 5:
            continue
        q_true = y_true[idx]
        q_pred = y_pred[idx]
        q_prob = y_prob[idx]
        entry = {"n": len(idx), "accuracy": float(accuracy_score(q_true, q_pred))}
        if len(set(q_true)) > 1:
            entry["auroc"] = float(roc_auc_score(q_true, q_prob))
        result[quad] = entry
    return result


def run_neural_probe(model_keys=None):
    if model_keys is None:
        model_keys = ["llama", "mistral", "qwen"]

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"NEURAL PROBE — {model_key}")
        print(f"{'='*60}")

        h_ctx_train, h_q_train, y_train, meta_train = load_split(model_key, "train")
        h_ctx_val, h_q_val, y_val, meta_val = load_split(model_key, "val")
        h_ctx_test, h_q_test, y_test, meta_test = load_split(model_key, "test")

        with open(RESULTS_DIR / model_key / "deco_results.json") as f:
            deco_res = json.load(f)

        model_results = {}

        # Test on two representations at their best layers
        representations = {
            "h(q+c)": {
                "layer": deco_res["best_layers"]["standard"],
                "train": lambda l: h_ctx_train[:, l, :],
                "val": lambda l: h_ctx_val[:, l, :],
                "test": lambda l: h_ctx_test[:, l, :],
            },
            "h(q+c)-h(q)": {
                "layer": deco_res["best_layers"]["DECO"],
                "train": lambda l: h_ctx_train[:, l, :] - h_q_train[:, l, :],
                "val": lambda l: h_ctx_val[:, l, :] - h_q_val[:, l, :],
                "test": lambda l: h_ctx_test[:, l, :] - h_q_test[:, l, :],
            },
        }

        for rep_name, rep_info in representations.items():
            layer = rep_info["layer"]
            print(f"\n  --- {rep_name} (layer {layer}) ---")

            X_train = rep_info["train"](layer).astype(np.float32)
            X_val = rep_info["val"](layer).astype(np.float32)
            X_test = rep_info["test"](layer).astype(np.float32)

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            input_dim = X_train_s.shape[1]

            # --- LogReg baseline ---
            logreg = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            logreg.fit(X_train_s, y_train)
            lr_prob = logreg.predict_proba(X_test_s)[:, 1]
            lr_pred = logreg.predict(X_test_s)
            lr_auroc = roc_auc_score(y_test, lr_prob)
            lr_quads = evaluate_per_quadrant(y_test, lr_prob, lr_pred, meta_test)
            print(f"  LogReg: AUROC={lr_auroc:.4f}")

            # --- Neural probe ---
            probe, val_auroc = train_neural_probe(
                X_train_s, y_train, X_val_s, y_val, input_dim
            )
            probe.eval()
            with torch.no_grad():
                test_logits = probe(torch.FloatTensor(X_test_s)).numpy()
                test_probs = 1 / (1 + np.exp(-test_logits))
            test_preds = (test_probs > 0.5).astype(int)
            nn_auroc = roc_auc_score(y_test, test_probs)
            nn_quads = evaluate_per_quadrant(y_test, test_probs, test_preds, meta_test)
            print(f"  Neural: AUROC={nn_auroc:.4f} (val={val_auroc:.4f})")

            model_results[rep_name] = {
                "layer": layer,
                "logistic_regression": {
                    "auroc": float(lr_auroc),
                    "accuracy": float(accuracy_score(y_test, lr_pred)),
                    "f1": float(f1_score(y_test, lr_pred, zero_division=0)),
                    "per_quadrant": lr_quads,
                },
                "neural_probe": {
                    "auroc": float(nn_auroc),
                    "accuracy": float(accuracy_score(y_test, test_preds)),
                    "f1": float(f1_score(y_test, test_preds, zero_division=0)),
                    "val_auroc": float(val_auroc),
                    "per_quadrant": nn_quads,
                },
                "delta_auroc": float(nn_auroc - lr_auroc),
            }

        all_results[model_key] = model_results

        # Print summary
        print(f"\n  Summary for {model_key}:")
        for rep_name, res in model_results.items():
            lr_a = res["logistic_regression"]["auroc"]
            nn_a = res["neural_probe"]["auroc"]
            print(f"    {rep_name}: LogReg={lr_a:.4f}, Neural={nn_a:.4f}, delta={nn_a-lr_a:+.4f}")

        out_path = RESULTS_DIR / model_key / "neural_probe_results.json"
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"  Saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama", "mistral", "qwen"])
    args = parser.parse_args()
    run_neural_probe(model_keys=args.models)
