"""
store_baseline_scores.py  (Issue 2 — prerequisite)

Re-runs the behavioral / generation baseline methods and stores the raw
per-example (score, label) pairs alongside the existing summary stats, plus
per-example probe scores obtained by refitting the probes on the train split.

These caches are consumed downstream:
  - scripts/bootstrap_ci.py       — needs per-example scores to bootstrap CIs
                                     for the behavioral / generation baselines.
  - scripts/run_downstream_sim.py — reads results/{model}/scores/probe_scores.json
                                     ("CSP (LogReg)") and baseline_scores.json
                                     ("verbalized_confidence").

Outputs (per model):
  results/{model}/scores/probe_scores.json
      { "CSP (LogReg)":      {"metadata": [{"id": ...}, ...], "scores": [...]},
        "Standard (LogReg)": {"metadata": [...],              "scores": [...]} }
  results/{model}/scores/baseline_scores.json
      { "verbalized_confidence": {"metadata": [{"condition_id": ...}, ...],
                                  "scores": [...], "labels": [...]},
        "token_entropy":        {...},
        "generation_match":     {...} }

The probe scores are always recoverable (they only need the cached hidden
states). The behavioral baselines require loading the model, so that stage is
gated behind --with-baselines and a --device.

Usage:
  python src/analysis/store_baseline_scores.py --model all
  python src/analysis/store_baseline_scores.py --model mistral --with-baselines --device cuda:0
"""

import json
import argparse
import sys
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, BENCH_DIR, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Hidden-state loading
# ---------------------------------------------------------------------------

def load_split(model_key: str, split: str):
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy")
    h_q = np.load(d / "h_question_only.npy")
    y = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)
    return h_qc, h_q, y, meta


def load_best_layers(model_key: str):
    """Best DECO/standard layers, read from the canonical probe results file
    (all_results_logreg.json; deco_results.json is a legacy path that may be
    absent in the reconstructed layout)."""
    for fname in ("deco_results.json", "all_results_logreg.json", "all_results.json"):
        path = RESULTS_DIR / model_key / fname
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            return cfg["best_layers"]["DECO"], cfg["best_layers"]["standard"]
    raise FileNotFoundError(f"No best-layer file found for {model_key}")


# ---------------------------------------------------------------------------
# Probe scores (per-example) — refit on train, score test
# ---------------------------------------------------------------------------

def compute_probe_scores(model_key: str) -> dict:
    h_qc_tr, h_q_tr, y_tr, _ = load_split(model_key, "train")
    h_qc_te, h_q_te, y_te, meta_te = load_split(model_key, "test")
    deco_layer, std_layer = load_best_layers(model_key)

    # CSP / DECO probe: logistic regression on h(q+c) - h(q)
    X_tr = (h_qc_tr[:, deco_layer, :] - h_q_tr[:, deco_layer, :]).astype(np.float32)
    X_te = (h_qc_te[:, deco_layer, :] - h_q_te[:, deco_layer, :]).astype(np.float32)
    scaler = StandardScaler()
    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(scaler.fit_transform(X_tr), y_tr)
    csp_scores = probe.predict_proba(scaler.transform(X_te))[:, 1]

    # Standard probe: logistic regression on h(q+c)
    Xs_tr = h_qc_tr[:, std_layer, :].astype(np.float32)
    Xs_te = h_qc_te[:, std_layer, :].astype(np.float32)
    scaler_s = StandardScaler()
    probe_s = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe_s.fit(scaler_s.fit_transform(Xs_tr), y_tr)
    std_scores = probe_s.predict_proba(scaler_s.transform(Xs_te))[:, 1]

    metadata = [{"id": m["id"], "quadrant": m.get("quadrant"),
                 "label": int(y_te[i])} for i, m in enumerate(meta_te)]

    return {
        "CSP (LogReg)":      {"metadata": metadata, "scores": csp_scores.tolist()},
        "Standard (LogReg)": {"metadata": metadata, "scores": std_scores.tolist()},
    }


# ---------------------------------------------------------------------------
# Behavioral / generation baselines (per-example) — requires the model
# ---------------------------------------------------------------------------

def compute_baseline_scores(model_key: str, device: str) -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.methods.baselines import (
        verbalized_confidence, token_entropy, generation_match,
    )

    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], device_map=device, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    methods = {
        "verbalized_confidence": {"metadata": [], "scores": [], "labels": []},
        "token_entropy":         {"metadata": [], "scores": [], "labels": []},
        "generation_match":      {"metadata": [], "scores": [], "labels": []},
    }

    for ex in test_data:
        for cond in ex["conditions"]:
            prompt = cond["prompt"]
            meta = {
                "condition_id": cond["condition_id"],
                "quadrant": cond["quadrant"],
                "question_type": ex["question_type"],
            }
            label = int(cond["sufficient"])

            vc = verbalized_confidence(model, tokenizer, prompt, device)
            te = token_entropy(model, tokenizer, prompt, device)
            gm = generation_match(model, tokenizer, prompt, ex["gold_answer"], device)

            for name, val in (("verbalized_confidence", vc),
                              ("token_entropy", te),
                              ("generation_match", gm)):
                methods[name]["metadata"].append(meta)
                methods[name]["scores"].append(float(val) if val is not None else 0.5)
                methods[name]["labels"].append(label)

    return methods


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def store_for_model(model_key: str, with_baselines: bool, device: str):
    out_dir = RESULTS_DIR / model_key / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{model_key}] computing probe scores...")
    probe_scores = compute_probe_scores(model_key)
    with open(out_dir / "probe_scores.json", "w") as f:
        json.dump(probe_scores, f, indent=2)
    print(f"[{model_key}] saved probe_scores.json")

    if with_baselines:
        print(f"[{model_key}] computing behavioral baseline scores (device={device})...")
        baseline_scores = compute_baseline_scores(model_key, device)
        with open(out_dir / "baseline_scores.json", "w") as f:
            json.dump(baseline_scores, f, indent=2)
        print(f"[{model_key}] saved baseline_scores.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--with-baselines", action="store_true",
                        help="Also re-run the model-dependent behavioral baselines")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    for mk in models:
        try:
            store_for_model(mk, args.with_baselines, args.device)
        except Exception as e:
            print(f"Error on {mk}: {e}")
            import traceback
            traceback.print_exc()
