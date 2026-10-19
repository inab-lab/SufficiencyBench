"""
redeep_baseline.py

Implements the ReDeEP (ICLR 2025) concept on SufficiencyBench.
Two scores for sufficiency detection:
  1. ECS (External Context Score): Measures how much attention goes to context tokens
  2. PKS (Parametric Knowledge Score): KL-divergence between logits with and without context

Higher ECS → model attends to context → predicts sufficient
Higher PKS → context changes model output → predicts sufficient

Usage:
  python src/methods/redeep_baseline.py --model mistral --device cuda:0
  python src/methods/redeep_baseline.py --model llama --device cuda:1
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS


def load_model_and_tokenizer(model_key: str, device: str):
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Required for output_attentions=True
    )
    model.eval()
    return model, tokenizer


def find_context_span(tokenizer, prompt: str, context: str):
    """Find the token index range of the context within the prompt."""
    # Tokenize full prompt
    full_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Find context start by tokenizing prefix
    ctx_start_str = "Context: "
    prefix_end = prompt.find(ctx_start_str)
    if prefix_end == -1:
        # Fallback: assume context starts after first 20 tokens
        return 20, len(full_ids) - 20

    prefix = prompt[:prefix_end + len(ctx_start_str)]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ctx_start = len(prefix_ids)

    # Find context end
    question_marker = "\n\nQuestion:"
    q_pos = prompt.find(question_marker, prefix_end)
    if q_pos == -1:
        ctx_end = len(full_ids) - 10
    else:
        prefix_to_q = prompt[:q_pos]
        prefix_to_q_ids = tokenizer.encode(prefix_to_q, add_special_tokens=False)
        ctx_end = len(prefix_to_q_ids)

    return max(ctx_start, 0), min(ctx_end, len(full_ids))


def compute_ecs(model, tokenizer, prompt: str, context: str, device: str,
                layer_range=(16, 24)):
    """
    External Context Score: Average attention from last token to context tokens.
    Uses upper layers where attention is more semantic.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    ctx_start, ctx_end = find_context_span(tokenizer, prompt, context)
    ctx_start = min(ctx_start, seq_len - 1)
    ctx_end = min(ctx_end, seq_len)

    if ctx_end <= ctx_start:
        return 0.0

    # Average attention to context tokens from last token, across upper layers
    n_layers = len(outputs.attentions)
    layer_lo = min(layer_range[0], n_layers - 1)
    layer_hi = min(layer_range[1], n_layers)

    ecs_scores = []
    for layer_idx in range(layer_lo, layer_hi):
        attn = outputs.attentions[layer_idx]  # (1, heads, seq, seq)
        # Attention from last token to context span
        last_to_ctx = attn[0, :, -1, ctx_start:ctx_end]  # (heads, ctx_len)
        # Average across heads and context positions
        ecs = last_to_ctx.mean().item()
        ecs_scores.append(ecs)

    return float(np.mean(ecs_scores))


def compute_pks(model, tokenizer, prompt_with_ctx: str, prompt_no_ctx: str, device: str):
    """
    Parametric Knowledge Score: KL-divergence between logit distributions
    with and without context. Higher = context changes model's mind more.
    """
    # With context
    inputs_ctx = tokenizer(
        prompt_with_ctx, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    with torch.no_grad():
        out_ctx = model(**inputs_ctx)
    logits_ctx = out_ctx.logits[0, -1, :]  # (vocab,)

    # Without context
    inputs_no = tokenizer(
        prompt_no_ctx, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    with torch.no_grad():
        out_no = model(**inputs_no)
    logits_no = out_no.logits[0, -1, :]  # (vocab,)

    # KL divergence: KL(P_ctx || P_no)
    p_ctx = F.softmax(logits_ctx, dim=-1)
    p_no = F.softmax(logits_no, dim=-1)

    # Symmetric KL (Jensen-Shannon divergence)
    m = 0.5 * (p_ctx + p_no)
    jsd = 0.5 * F.kl_div(m.log(), p_ctx, reduction="sum") + \
          0.5 * F.kl_div(m.log(), p_no, reduction="sum")

    return float(jsd.item())


def evaluate_method(method_name, scores, labels, metadata):
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    result = {"method": method_name, "overall": {}, "per_quadrant": {}, "per_question_type": {}}

    if len(set(labels_arr)) > 1:
        result["overall"]["auroc"] = float(roc_auc_score(labels_arr, scores_arr))

    preds = (scores_arr > np.median(scores_arr)).astype(int)
    result["overall"]["accuracy"] = float(accuracy_score(labels_arr, preds))
    result["overall"]["f1"] = float(f1_score(labels_arr, preds, zero_division=0))
    result["overall"]["n"] = len(labels_arr)

    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(metadata) if m["quadrant"] == quad]
        if len(idx) < 5:
            continue
        q_labels = labels_arr[idx]
        q_preds = preds[idx]
        q_scores = scores_arr[idx]
        entry = {"n": len(idx), "accuracy": float(accuracy_score(q_labels, q_preds))}
        if len(set(q_labels)) > 1:
            entry["auroc"] = float(roc_auc_score(q_labels, q_scores))
        result["per_quadrant"][quad] = entry

    for qtype in ["factual", "multi_hop", "comparative", "subjective"]:
        idx = [i for i, m in enumerate(metadata) if m["question_type"] == qtype]
        if len(idx) < 10 or len(set(labels_arr[idx])) < 2:
            continue
        result["per_question_type"][qtype] = {
            "n": len(idx),
            "auroc": float(roc_auc_score(labels_arr[idx], scores_arr[idx])),
        }

    return result


def make_question_only_prompt(question):
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, say "
        f"'I cannot answer this from the provided context.'\n\n"
        f"Context: [No context provided]\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def run_redeep(model_key="mistral", device="cuda:0", max_examples=0):
    print(f"\n{'='*60}")
    print(f"ReDeEP-STYLE BASELINE — {model_key}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_key, device)
    n_layers = MODEL_CONFIGS[model_key]["n_layers"]

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question_type": ex["question_type"],
                "gold_answer": ex["gold_answer"],
                "question": ex["question"],
            })

    if max_examples > 0:
        conditions = conditions[:max_examples]
        print(f"  (limited to {max_examples} examples)")

    print(f"Evaluating {len(conditions)} conditions...\n")

    ecs_scores = []
    pks_scores = []

    for i, cond in enumerate(tqdm(conditions, desc="ReDeEP")):
        # ECS: attention to context tokens
        ecs = compute_ecs(
            model, tokenizer, cond["prompt"], cond["context"], device,
            layer_range=(n_layers * 2 // 3, n_layers)  # upper third of layers
        )
        ecs_scores.append(ecs)

        # PKS: logit divergence with vs without context
        prompt_no_ctx = make_question_only_prompt(cond["question"])
        pks = compute_pks(model, tokenizer, cond["prompt"], prompt_no_ctx, device)
        pks_scores.append(pks)

        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    labels = [1 if c["sufficient"] else 0 for c in conditions]
    metadata = [{"quadrant": c["quadrant"], "question_type": c["question_type"]}
                for c in conditions]

    # Evaluate individual scores
    result_ecs = evaluate_method("redeep_ecs", ecs_scores, labels, metadata)
    result_pks = evaluate_method("redeep_pks", pks_scores, labels, metadata)

    # Combined: train logistic regression on [ECS, PKS]
    X_combined = np.column_stack([ecs_scores, pks_scores])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    # Use leave-one-out or just fit on all (since we're comparing to other methods)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, labels)
    combined_probs = lr.predict_proba(X_scaled)[:, 1]
    combined_preds = lr.predict(X_scaled)
    result_combined = evaluate_method("redeep_combined", combined_probs.tolist(), labels, metadata)

    # Full result
    result = {
        "model": model_key,
        "n_examples": len(conditions),
        "ecs": result_ecs,
        "pks": result_pks,
        "combined": result_combined,
        "ecs_stats": {
            "mean": float(np.mean(ecs_scores)),
            "std": float(np.std(ecs_scores)),
            "min": float(np.min(ecs_scores)),
            "max": float(np.max(ecs_scores)),
        },
        "pks_stats": {
            "mean": float(np.mean(pks_scores)),
            "std": float(np.std(pks_scores)),
            "min": float(np.min(pks_scores)),
            "max": float(np.max(pks_scores)),
        },
    }

    # Print summary
    print(f"\nResults:")
    print(f"  ECS AUROC: {result_ecs['overall'].get('auroc', 'N/A')}")
    print(f"  PKS AUROC: {result_pks['overall'].get('auroc', 'N/A')}")
    print(f"  Combined AUROC: {result_combined['overall'].get('auroc', 'N/A')}")
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        e = result_combined["per_quadrant"].get(quad, {})
        print(f"  {quad}: acc={e.get('accuracy', 'N/A')}")

    # Save
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "redeep_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReDeEP-style baseline for SufficiencyBench")
    parser.add_argument("--model", default="mistral", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()
    run_redeep(model_key=args.model, device=args.device, max_examples=args.max_examples)
