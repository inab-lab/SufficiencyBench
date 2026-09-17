"""
real_retrieval_eval.py

Tests whether sufficiency probes generalize beyond SufficiencyBench to real data.
Uses SQuAD v2 validation set:
  - Sufficient: gold paragraph (contains the answer)
  - Insufficient: random non-gold paragraph from another question

Loads trained probe from SufficiencyBench hidden states, extracts hidden states
on SQuAD examples, and evaluates probe AUROC on this out-of-distribution data.

Usage:
  python src/evaluation/real_retrieval_eval.py --model mistral --device cuda:0
"""

import json
import torch
import numpy as np
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from datasets import load_from_disk
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import (
    HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS,
    DATASETS_DIR, BENCH_DIR,
)


PROMPT_TEMPLATE = (
    "Based on the following context, answer the question. "
    "If the context does not contain enough information, say "
    "'I cannot answer this from the provided context.'\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\nAnswer:"
)

NO_CONTEXT_TEMPLATE = (
    "Based on the following context, answer the question. "
    "If the context does not contain enough information, say "
    "'I cannot answer this from the provided context.'\n\n"
    "Context: [No context provided]\n\n"
    "Question: {question}\n\nAnswer:"
)


def train_probe_from_bench(model_key):
    """Train standard and CSP probes on SufficiencyBench training data."""
    d = HIDDEN_STATES_DIR / model_key / "train"
    h_ctx = np.load(d / "h_with_context.npy")
    h_q = np.load(d / "h_question_only.npy")
    labels = np.load(d / "labels.npy")

    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        deco_res = json.load(f)
    std_layer = deco_res["best_layers"]["standard"]
    csp_layer = deco_res["best_layers"]["DECO"]

    # Standard probe
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(h_ctx[:, std_layer, :])
    probe_std = LogisticRegression(max_iter=2000, random_state=42)
    probe_std.fit(X_std, labels)

    # CSP probe
    scaler_csp = StandardScaler()
    diff = h_ctx[:, csp_layer, :] - h_q[:, csp_layer, :]
    X_csp = scaler_csp.fit_transform(diff)
    probe_csp = LogisticRegression(max_iter=2000, random_state=42)
    probe_csp.fit(X_csp, labels)

    return {
        "standard": (probe_std, scaler_std, std_layer),
        "csp": (probe_csp, scaler_csp, csp_layer),
    }


def extract_hidden_state(model, tokenizer, prompt, device, layer):
    """Extract last-token hidden state at a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is tuple of (n_layers+1, batch, seq, hidden_dim)
    h = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
    last_token_h = h[0, -1, :].float().cpu().numpy()
    return last_token_h


def build_squad_pairs(n_pairs=300, seed=42):
    """Build sufficient/insufficient pairs from SQuAD v2."""
    ds = load_from_disk(str(DATASETS_DIR / "squad_v2"))["validation"]
    random.seed(seed)

    # Filter to answerable questions (have non-empty answers)
    answerable = [ex for ex in ds if len(ex["answers"]["text"]) > 0]
    print(f"  SQuAD v2 answerable: {len(answerable)}")

    # Sample pairs
    sampled = random.sample(answerable, min(n_pairs, len(answerable)))

    # For each, create sufficient (gold context) and insufficient (random other context)
    all_contexts = [ex["context"] for ex in answerable]
    pairs = []

    for ex in sampled:
        question = ex["question"]
        gold_context = ex["context"]
        gold_answer = ex["answers"]["text"][0]

        # Pick a random non-gold context
        while True:
            rand_ctx = random.choice(all_contexts)
            if rand_ctx != gold_context:
                break

        pairs.append({
            "question": question,
            "gold_answer": gold_answer,
            "sufficient_context": gold_context,
            "insufficient_context": rand_ctx,
        })

    return pairs


def run_real_retrieval(model_key="mistral", device="cuda:0", n_pairs=300):
    print(f"\n{'='*60}")
    print(f"REAL RETRIEVAL EVALUATION — {model_key}")
    print(f"{'='*60}")

    # Train probes on SufficiencyBench
    print("\n  Training probes from SufficiencyBench...")
    probes = train_probe_from_bench(model_key)

    # Build SQuAD pairs
    print("  Building SQuAD v2 pairs...")
    pairs = build_squad_pairs(n_pairs)
    print(f"  Created {len(pairs)} pairs")

    # Load model
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], quantization_config=bnb_config,
        device_map=device, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Extract hidden states and predict
    results_per_probe = {name: {"scores": [], "labels": []} for name in probes}

    for i, pair in enumerate(tqdm(pairs, desc="Real retrieval")):
        for condition, label in [("sufficient_context", 1), ("insufficient_context", 0)]:
            context = pair[condition]
            prompt = PROMPT_TEMPLATE.format(context=context, question=pair["question"])
            prompt_no_ctx = NO_CONTEXT_TEMPLATE.format(question=pair["question"])

            for probe_name, (probe, scaler, layer) in probes.items():
                h_ctx = extract_hidden_state(model, tokenizer, prompt, device, layer)

                if probe_name == "csp":
                    h_q = extract_hidden_state(model, tokenizer, prompt_no_ctx, device, layer)
                    features = (h_ctx - h_q).reshape(1, -1)
                else:
                    features = h_ctx.reshape(1, -1)

                features_scaled = scaler.transform(features.astype(np.float32))
                score = probe.predict_proba(features_scaled)[0, 1]

                results_per_probe[probe_name]["scores"].append(float(score))
                results_per_probe[probe_name]["labels"].append(label)

        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    # Evaluate
    result = {"model": model_key, "n_pairs": len(pairs), "probes": {}}

    for probe_name, data in results_per_probe.items():
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])
        preds = (scores > 0.5).astype(int)

        auroc = roc_auc_score(labels, scores)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)

        result["probes"][probe_name] = {
            "auroc": float(auroc),
            "accuracy": float(acc),
            "f1": float(f1),
            "n": len(labels),
        }
        print(f"\n  {probe_name}: AUROC={auroc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Save
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_pairs", type=int, default=300)
    args = parser.parse_args()
    run_real_retrieval(model_key=args.model, device=args.device, n_pairs=args.n_pairs)
