"""
deco_rag.py — Apply the DECO / CSP sufficiency probe to a RAG abstention /
contrastive-decoding decision.

Given a fitted DECO probe (from methods.deco.train_deco_probe), at inference time
we extract h(q+c) and h(q) at the probe's best layer, form the delta
d = h(q+c) - h(q), and let the probe predict whether the context is SUFFICIENT.
That prediction gates the model's behaviour:

  1. Vanilla       — standard generation (no gating).
  2. DECO-Abstain  — abstain when the probe predicts INSUFFICIENT.
  3. DECO-CAD      — apply Context-Aware Decoding (CAD) only when INSUFFICIENT.
  4. Always-CAD    — always apply CAD (baseline).

CAD (Shi et al. 2023) contrasts logits with and without context:
  logits = (1 + alpha) * logits(q+c) - alpha * logits(q).

Usage:
  python src/methods/deco_rag.py --model mistral --device cuda:0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS
from methods.deco import train_deco_probe

MAX_SEQ_LEN = 2048


def format_question_only(question: str) -> str:
    """Question-only prompt (no context) — matches extraction.extract_states."""
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, "
        f"say 'I cannot answer this from the provided context.'\n\n"
        f"Context: [No context provided]\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def extract_hidden_at_layer(model, tokenizer, prompt, device, layer):
    """Last-token hidden state at a single layer. Returns (hidden_dim,) float32.

    `layer` is 0-indexed over transformer blocks (as in methods.deco), i.e. it
    maps to outputs.hidden_states[layer + 1] (index 0 is the embedding layer).
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer + 1][0, -1, :].float().cpu().numpy()


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=100):
    """Greedy generation; returns the decoded continuation."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def generate_with_cad(model, tokenizer, prompt, q_only_prompt, device,
                      alpha=0.5, max_new_tokens=100):
    """Context-Aware Decoding (Shi et al. 2023), greedy.

    At each step: logits = (1 + alpha) * logits(q+c) - alpha * logits(q_only).
    The two streams are decoded in lock-step, appending the same chosen token to
    both so their key/value contexts stay aligned.
    """
    ctx = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(device)
    base = tokenizer(
        q_only_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(device)

    ctx_ids = ctx["input_ids"]
    base_ids = base["input_ids"]
    generated = []
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits_ctx = model(ctx_ids).logits[0, -1, :]
            logits_base = model(base_ids).logits[0, -1, :]
            cad_logits = (1 + alpha) * logits_ctx - alpha * logits_base
            next_id = int(torch.argmax(cad_logits).item())
            if next_id == eos_id:
                break
            generated.append(next_id)
            nxt = torch.tensor([[next_id]], device=device)
            ctx_ids = torch.cat([ctx_ids, nxt], dim=1)
            base_ids = torch.cat([base_ids, nxt], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


ABSTAIN_PHRASES = [
    "cannot answer", "don't have enough", "not enough information",
    "i cannot", "i don't know", "unable to answer",
    "the context does not", "no information",
    "cannot be determined", "not provided",
    "not mentioned",
]


def check_abstain(text):
    text_lower = text.lower()
    return any(p in text_lower for p in ABSTAIN_PHRASES)


def run_deco_rag(
    model_key: str = "mistral",
    device: str = "cuda:0",
    n_samples: int = 200,
    cad_alpha: float = 0.5,
    sufficiency_threshold: float = 0.5,
):
    """
    Run the full DECO-RAG evaluation.

    Methods compared:
    1. Vanilla: standard generation
    2. DECO-Abstain: abstain when DECO probe predicts insufficient
    3. DECO-CAD: apply CAD when DECO probe predicts insufficient
    4. Always-CAD: always apply CAD (baseline)
    """
    print(f"\n{'='*70}")
    print(f"DECO-RAG EVALUATION — {model_key}")
    print(f"{'='*70}")

    # Step 1: Train probes
    deco_probe, std_probe, best_layer = train_deco_probe(model_key)

    # Step 2: Load model
    cfg = MODEL_CONFIGS[model_key]
    model_path = cfg["path"]

    print(f"Loading {model_key} in 8-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map=device, low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Step 3: Load test data
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten
    examples = []
    for ex in test_data:
        for cond in ex["conditions"]:
            examples.append({
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "model_knows": ex["model_knows_answer"],
                "prompt": cond["prompt"],
                "sufficient": cond["sufficient"],
                "quadrant": cond["quadrant"],
                "question_type": ex["question_type"],
            })

    # Sample balanced across quadrants
    np.random.seed(42)
    by_quad = {}
    for e in examples:
        by_quad.setdefault(e["quadrant"], []).append(e)
    selected = []
    per_quad = max(n_samples // 4, 10)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        pool = by_quad.get(q, [])
        n = min(per_quad, len(pool))
        idx = np.random.choice(len(pool), n, replace=False)
        selected.extend([pool[i] for i in idx])

    print(f"Selected {len(selected)} examples")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        print(f"  {q}: {sum(1 for e in selected if e['quadrant'] == q)}")

    # Step 4: Run evaluation
    results = []
    for ex in tqdm(selected, desc="DECO-RAG"):
        # Extract hidden state for DECO probe
        h_ctx = extract_hidden_at_layer(
            model, tokenizer, ex["prompt"], device, best_layer
        )
        q_only_prompt = format_question_only(ex["question"])
        h_q = extract_hidden_at_layer(
            model, tokenizer, q_only_prompt, device, best_layer
        )

        # DECO prediction
        deco_feat = (h_ctx - h_q).reshape(1, -1)
        deco_prob = deco_probe.predict_proba(deco_feat)[0, 1]
        deco_pred_sufficient = deco_prob > sufficiency_threshold

        # Standard probe prediction
        std_feat = h_ctx.reshape(1, -1)
        std_prob = std_probe.predict_proba(std_feat)[0, 1]
        std_pred_sufficient = std_prob > sufficiency_threshold

        result = {
            "quadrant": ex["quadrant"],
            "sufficient": ex["sufficient"],
            "model_knows": ex["model_knows"],
            "gold_answer": ex["gold_answer"],
            "question_type": ex["question_type"],
            "deco_prob": float(deco_prob),
            "std_prob": float(std_prob),
            "deco_pred_sufficient": bool(deco_pred_sufficient),
        }

        # Method 1: Vanilla generation
        gen_vanilla = generate_answer(model, tokenizer, ex["prompt"], device)
        result["gen_vanilla"] = gen_vanilla[:500]
        result["abstain_vanilla"] = check_abstain(gen_vanilla)
        result["correct_vanilla"] = (
            ex["gold_answer"].lower() in gen_vanilla.lower()
            and not result["abstain_vanilla"]
        )

        # Method 2: DECO-Abstain
        if deco_pred_sufficient:
            result["gen_deco_abstain"] = gen_vanilla[:500]
            result["abstain_deco_abstain"] = result["abstain_vanilla"]
            result["correct_deco_abstain"] = result["correct_vanilla"]
        else:
            result["gen_deco_abstain"] = "I cannot answer this from the provided context."
            result["abstain_deco_abstain"] = True
            result["correct_deco_abstain"] = False

        # Method 3: DECO-CAD (apply CAD only when probe says insufficient)
        if deco_pred_sufficient:
            result["gen_deco_cad"] = gen_vanilla[:500]
            result["abstain_deco_cad"] = result["abstain_vanilla"]
            result["correct_deco_cad"] = result["correct_vanilla"]
        else:
            gen_cad = generate_with_cad(
                model, tokenizer, ex["prompt"], q_only_prompt,
                device, alpha=cad_alpha,
            )
            result["gen_deco_cad"] = gen_cad[:500]
            result["abstain_deco_cad"] = check_abstain(gen_cad)
            result["correct_deco_cad"] = (
                ex["gold_answer"].lower() in gen_cad.lower()
                and not result["abstain_deco_cad"]
            )

        # Method 4: Always-CAD (baseline)
        gen_always_cad = generate_with_cad(
            model, tokenizer, ex["prompt"], q_only_prompt,
            device, alpha=cad_alpha,
        )
        result["gen_always_cad"] = gen_always_cad[:500]
        result["abstain_always_cad"] = check_abstain(gen_always_cad)
        result["correct_always_cad"] = (
            ex["gold_answer"].lower() in gen_always_cad.lower()
            and not result["abstain_always_cad"]
        )

        results.append(result)

        if len(results) % 20 == 0:
            torch.cuda.empty_cache()

    # Step 5: Analysis
    methods = ["vanilla", "deco_abstain", "deco_cad", "always_cad"]

    # TODO(reconstruct): the per-method / per-quadrant aggregation, printing and
    # results-saving that followed were lost to the deletion damage and cannot be
    # faithfully recovered from the surviving body. Return the raw per-example
    # results plus the method list so callers can aggregate.
    _ = methods
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--cad-alpha", type=float, default=0.5)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_deco_rag(
        model_key=args.model, device=args.device, n_samples=args.n_samples,
        cad_alpha=args.cad_alpha, sufficiency_threshold=args.sufficiency_threshold,
    )


if __name__ == "__main__":
    main()
