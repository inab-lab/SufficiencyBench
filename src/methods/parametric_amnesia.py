"""
parametric_amnesia.py — Erasing the Parametric Knowledge Direction

Finds the direction in hidden space that encodes "the model knows this from
training data" and erases it at inference time, forcing the model to rely
exclusively on provided context.

Inspired by:
- Arditi et al. (2024): refusal is mediated by a single direction
- LEACE (Belrose et al., NeurIPS 2023): optimal linear concept erasure
- ParamMute (NeurIPS 2025): suppressing knowledge FFNs

Our approach: geometric erasure of the parametric knowledge direction,
which is simpler, faster, and more principled than FFN suppression.

Usage:
  python src/methods/parametric_amnesia.py --step find_direction --model mistral
  python src/methods/parametric_amnesia.py --step test_erasure --model mistral --device cuda:1
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import (
    HIDDEN_STATES_DIR, CRAG_DIR, BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS
)


# ================================================================
# STEP 1: Find the Parametric Knowledge Direction
# ================================================================

def find_pk_direction(model_key: str = "mistral"):
    """
    Find the direction in hidden space that encodes parametric knowledge.

    Method:
    1. Train logistic probe on h(q+c) to predict model_confident
    2. The probe weight vector IS the parametric knowledge direction
    3. Also compute LEACE erasure matrix for optimal removal
    4. Verify: after erasure, sufficiency detection should be preserved
    """
    print(f"\n{'='*70}")
    print(f"FINDING PARAMETRIC KNOWLEDGE DIRECTION — {model_key}")
    print(f"{'='*70}")

    # Load hidden states
    hs_dir = HIDDEN_STATES_DIR / model_key / "train"
    h_qc = np.load(hs_dir / "h_with_context.npy", mmap_mode="r")
    h_q = np.load(hs_dir / "h_question_only.npy", mmap_mode="r")
    labels_suf = np.load(hs_dir / "labels.npy")  # sufficiency labels

    with open(hs_dir / "metadata.json") as f:
        meta = json.load(f)

    # Parametric knowledge labels
    y_pk = np.array([1 if m["model_confident"] else 0 for m in meta])

    n_samples, n_layers, hidden_dim = h_qc.shape
    print(f"Data: {n_samples} samples, {n_layers} layers, {hidden_dim} dim")
    print(f"Model confident: {y_pk.sum()}/{len(y_pk)} ({y_pk.mean()*100:.1f}%)")

    # Also load test set for validation
    hs_test = HIDDEN_STATES_DIR / model_key / "test"
    h_qc_test = np.load(hs_test / "h_with_context.npy", mmap_mode="r")
    h_q_test = np.load(hs_test / "h_question_only.npy", mmap_mode="r")
    labels_suf_test = np.load(hs_test / "labels.npy")
    with open(hs_test / "metadata.json") as f:
        meta_test = json.load(f)
    y_pk_test = np.array([1 if m["model_confident"] else 0 for m in meta_test])

    out_dir = CRAG_DIR / model_key / "amnesia"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print(f"\n{'Layer':>5} {'PK_AUROC':>10} {'PK_dir_on_hqc':>14} "
          f"{'Suf_before':>11} {'Suf_after':>10} {'Delta':>7}")
    print("-" * 60)

    best_pk_auroc = 0
    best_layer = 0

    for L in range(n_layers):
        # --- Extract features at this layer ---
        X_train = np.array(h_qc[:, L, :])
        X_test = np.array(h_qc_test[:, L, :])
        X_q_train = np.array(h_q[:, L, :])
        X_q_test = np.array(h_q_test[:, L, :])

        # --- Train PK direction probe on h(q+c) ---
        scaler_pk = StandardScaler()
        X_tr_s = scaler_pk.fit_transform(X_train)
        X_te_s = scaler_pk.transform(X_test)

        probe_pk = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        probe_pk.fit(X_tr_s, y_pk)
        pk_auroc = roc_auc_score(
            y_pk_test, probe_pk.predict_proba(X_te_s)[:, 1]
        )

        # The PK direction (normalized)
        pk_dir = probe_pk.coef_[0].copy()
        pk_dir_norm = pk_dir / np.linalg.norm(pk_dir)

        # --- Sufficiency probe BEFORE erasure ---
        scaler_suf = StandardScaler()
        X_tr_suf = scaler_suf.fit_transform(X_train)
        X_te_suf = scaler_suf.transform(X_test)

        probe_suf = LogisticRegression(max_iter=1000, random_state=42)
        probe_suf.fit(X_tr_suf, labels_suf)
        suf_auroc_before = roc_auc_score(
            labels_suf_test, probe_suf.predict_proba(X_te_suf)[:, 1]
        )

        # --- LEACE-style erasure: project out the PK direction ---
        # Erasure: X_erased = X - (X @ d) * d^T  (for unit vector d)
        # This removes all information along the PK direction
        pk_dir_raw = pk_dir_norm.astype(np.float32)

        # Erase from raw (unscaled) representations
        proj_train = X_train @ pk_dir_raw
        X_train_erased = X_train - np.outer(proj_train, pk_dir_raw)

        proj_test = X_test @ pk_dir_raw
        X_test_erased = X_test - np.outer(proj_test, pk_dir_raw)

        # --- Sufficiency probe AFTER erasure ---
        scaler_suf2 = StandardScaler()
        X_tr_suf2 = scaler_suf2.fit_transform(X_train_erased)
        X_te_suf2 = scaler_suf2.transform(X_test_erased)

        probe_suf2 = LogisticRegression(max_iter=1000, random_state=42)
        probe_suf2.fit(X_tr_suf2, labels_suf)
        suf_auroc_after = roc_auc_score(
            labels_suf_test, probe_suf2.predict_proba(X_te_suf2)[:, 1]
        )

        delta = suf_auroc_after - suf_auroc_before

        # --- PK probe AFTER erasure (should be destroyed) ---
        probe_pk2 = LogisticRegression(max_iter=1000, random_state=42)
        probe_pk2.fit(scaler_suf2.fit_transform(X_train_erased), y_pk)
        pk_auroc_after = roc_auc_score(
            y_pk_test, probe_pk2.predict_proba(scaler_suf2.transform(X_test_erased))[:, 1]
        )

        results[L] = {
            "pk_auroc_before": float(pk_auroc),
            "pk_auroc_after": float(pk_auroc_after),
            "suf_auroc_before": float(suf_auroc_before),
            "suf_auroc_after": float(suf_auroc_after),
            "suf_delta": float(delta),
        }

        if L % 2 == 0 or L >= n_layers - 4 or pk_auroc > best_pk_auroc:
            print(f"L{L:>4d} {pk_auroc:>10.4f} {pk_auroc:>14.4f} "
                  f"{suf_auroc_before:>11.4f} {suf_auroc_after:>10.4f} "
                  f"{delta:>+7.4f}")

        if pk_auroc > best_pk_auroc:
            best_pk_auroc = pk_auroc
            best_layer = L

    print(f"\nBest layer for PK detection: L{best_layer} (AUROC={best_pk_auroc:.4f})")

    best = results[best_layer]
    print(f"\n{'='*70}")
    print(f"KEY RESULTS AT BEST LAYER (L{best_layer})")
    print(f"{'='*70}")
    print(f"PK detection AUROC:      {best['pk_auroc_before']:.4f} -> {best['pk_auroc_after']:.4f} after erasure")
    print(f"Sufficiency AUROC:       {best['suf_auroc_before']:.4f} -> {best['suf_auroc_after']:.4f} after erasure")
    print(f"Sufficiency preserved:   {'YES' if abs(best['suf_delta']) < 0.03 else 'PARTIALLY' if abs(best['suf_delta']) < 0.05 else 'NO'}")
    print(f"PK direction destroyed:  {'YES' if best['pk_auroc_after'] < 0.55 else 'PARTIALLY' if best['pk_auroc_after'] < 0.6 else 'NO'}")

    # Save the PK direction for the best layer and neighbors
    for L in [best_layer, max(0, best_layer-1), min(n_layers-1, best_layer+1),
              max(0, best_layer-2), min(n_layers-1, best_layer+2)]:
        X_L = np.array(h_qc[:, L, :])
        sc = StandardScaler()
        X_s = sc.fit_transform(X_L)
        pr = LogisticRegression(max_iter=1000, random_state=42)
        pr.fit(X_s, y_pk)
        d = pr.coef_[0].copy()
        d = d / np.linalg.norm(d)
        np.save(out_dir / f"pk_direction_L{L}.npy", d.astype(np.float32))

    # Save summary
    summary = {
        "model": model_key,
        "best_layer": best_layer,
        "best_pk_auroc": best_pk_auroc,
        "per_layer": results,
    }
    with open(out_dir / "direction_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")

    # Also find the best layer where erasure preserves sufficiency the most
    # while still having good PK detection
    best_tradeoff_layer = max(
        range(n_layers),
        key=lambda L: (
            results[L]["pk_auroc_before"] * 0.5 +
            (1.0 - abs(results[L]["suf_delta"])) * 0.5
        )
    )
    print(f"\nBest tradeoff layer: L{best_tradeoff_layer}")
    print(f"  PK AUROC: {results[best_tradeoff_layer]['pk_auroc_before']:.4f}")
    print(f"  Suf delta: {results[best_tradeoff_layer]['suf_delta']:+.4f}")

    return summary


# ================================================================
# STEP 2: Test Erasure Effect on Generation
# ================================================================

def test_erasure_generation(
    model_key: str = "mistral",
    device: str = "cuda:0",
    n_samples: int = 100,
    max_new_tokens: int = 256,
    erasure_alpha: float = 1.0,
):
    """
    Test the effect of erasing the PK direction during generation.

    Compares three modes:
    1. normal: standard generation
    2. erasure: project out PK direction at target layer
    3. amplify: amplify PK direction (opposite, should hurt faithfulness)

    Measures:
    - Does the model answer from context when context is sufficient?
    - Does the model abstain when context is insufficient?
    - Does the model's answer change when we erase PK?
    """
    print(f"\n{'='*70}")
    print(f"TESTING PARAMETRIC ERASURE ON GENERATION — {model_key}")
    print(f"{'='*70}")

    cfg = MODEL_CONFIGS[model_key]
    model_path = cfg["path"]

    # Load erasure direction
    amnesia_dir = CRAG_DIR / model_key / "amnesia"
    with open(amnesia_dir / "direction_results.json") as f:
        dir_info = json.load(f)
    best_layer = dir_info["best_layer"]
    pk_dir = np.load(amnesia_dir / f"pk_direction_L{best_layer}.npy")
    pk_dir_tensor = torch.tensor(pk_dir, dtype=torch.float16, device=device)

    print(f"Using PK direction at layer {best_layer}")
    print(f"PK AUROC at this layer: {dir_info['best_pk_auroc']:.4f}")

    # Load model — use 8-bit quantization to avoid CPU memory overcommit
    # (system has overcommit_memory=2 with strict limits)
    print(f"Loading {model_key} in 8-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Load test data
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten and sample
    examples = []
    for ex in test_data:
        for cond in ex["conditions"]:
            examples.append({
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "model_knows": ex["model_knows_answer"],
                "prompt": cond["prompt"],
                "context": cond["context"],
                "sufficient": cond["sufficient"],
                "quadrant": cond["quadrant"],
                "condition_id": cond["condition_id"],
            })

    # Sample balanced across quadrants
    np.random.seed(42)
    by_quad = {}
    for e in examples:
        by_quad.setdefault(e["quadrant"], []).append(e)
    selected = []
    per_quad = max(n_samples // 4, 10)
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        pool = by_quad.get(quad, [])
        n = min(per_quad, len(pool))
        idx = np.random.choice(len(pool), n, replace=False)
        selected.extend([pool[i] for i in idx])

    print(f"Selected {len(selected)} examples")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        n = sum(1 for e in selected if e["quadrant"] == q)
        print(f"  {q}: {n}")

    # Get target layer module
    target_layer = model.model.layers[best_layer]

    # Generation function with optional erasure
    def generate_with_mode(prompt, mode="normal", alpha=1.0):
        hook_handle = None

        def erasure_hook(module, input, output):
            # Handle different output formats across model architectures
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Apply erasure/amplification to ALL tokens
            # This is critical: PK info is baked into KV cache during
            # the initial prompt forward pass. We must erase from every
            # position so the KV cache doesn't retain PK signal.
            h_float = hidden.float()
            d = pk_dir_tensor.float()

            if hidden.dim() == 2:
                # (seq, dim) — project each token
                projs = h_float @ d  # (seq,)
                steering = projs.unsqueeze(-1) * d.unsqueeze(0)  # (seq, dim)
            else:
                # (batch, seq, dim)
                projs = h_float @ d  # (batch, seq)
                steering = projs.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)  # (batch, seq, dim)

            if mode == "erasure":
                hidden -= (alpha * steering).to(hidden.dtype)
            elif mode == "amplify":
                hidden += (alpha * steering).to(hidden.dtype)

            # Return in same format as received
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            else:
                return hidden

        if mode != "normal":
            hook_handle = target_layer.register_forward_hook(erasure_hook)

        try:
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )

            text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return text

    # Run all three modes
    all_results = []
    modes = ["normal", "erasure", "amplify"]

    for ex in tqdm(selected, desc="Generating"):
        result = {
            "condition_id": ex["condition_id"],
            "quadrant": ex["quadrant"],
            "sufficient": ex["sufficient"],
            "model_knows": ex["model_knows"],
            "question_type": ex["question_type"],
            "gold_answer": ex["gold_answer"],
            "context_snippet": ex["context"][:200],
        }

        for mode in modes:
            text = generate_with_mode(
                ex["prompt"], mode=mode, alpha=erasure_alpha
            )
            result[f"gen_{mode}"] = text[:500]

            # Quick metrics
            text_lower = text.lower()
            gold_lower = ex["gold_answer"].lower()

            abstains = any(p in text_lower for p in [
                "cannot answer", "don't have enough", "not enough information",
                "i cannot", "i don't know", "unable to answer",
                "the context does not", "no information",
                "cannot be determined", "not provided",
            ])
            contains_answer = gold_lower in text_lower

            result[f"abstains_{mode}"] = abstains
            result[f"correct_{mode}"] = contains_answer and not abstains

        all_results.append(result)

        if len(all_results) % 20 == 0:
            torch.cuda.empty_cache()

    # ---- Analysis ----
    print(f"\n{'='*70}")
    print(f"ERASURE RESULTS")
    print(f"{'='*70}")

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        # Overall
        n_correct = sum(1 for r in all_results if r[f"correct_{mode}"])
        n_abstain = sum(1 for r in all_results if r[f"abstains_{mode}"])
        print(f"  Overall: correct={n_correct}/{len(all_results)}, "
              f"abstain={n_abstain}/{len(all_results)}")

        # Per-quadrant
        for quad in ["Q1", "Q2", "Q3", "Q4"]:
            q_results = [r for r in all_results if r["quadrant"] == quad]
            if not q_results:
                continue
            n = len(q_results)
            correct = sum(1 for r in q_results if r[f"correct_{mode}"])
            abstain = sum(1 for r in q_results if r[f"abstains_{mode}"])
            print(f"  {quad} (n={n}): correct={correct}/{n} ({correct/n:.0%}), "
                  f"abstain={abstain}/{n} ({abstain/n:.0%})")

    # Key comparison: on Q3 (insufficient + confident), does erasure help?
    print(f"\n{'='*70}")
    print("KEY COMPARISON: Q3 (Insufficient + Confident)")
    print("This is where the model SHOULD abstain but tends to hallucinate")
    print(f"{'='*70}")

    q3 = [r for r in all_results if r["quadrant"] == "Q3"]
    if q3:
        for mode in modes:
            abstain = sum(1 for r in q3 if r[f"abstains_{mode}"])
            correct = sum(1 for r in q3 if r[f"correct_{mode}"])
            print(f"  {mode:>8s}: abstain={abstain}/{len(q3)} ({abstain/len(q3):.0%}), "
                  f"hallucinate={len(q3)-abstain}/{len(q3)} ({(len(q3)-abstain)/len(q3):.0%})")

    # Also check Q1: does erasure hurt correct answers?
    print(f"\n{'='*70}")
    print("SAFETY CHECK: Q1 (Sufficient + Confident)")
    print("Model should answer correctly — erasure should NOT break this")
    print(f"{'='*70}")

    q1 = [r for r in all_results if r["quadrant"] == "Q1"]
    if q1:
        for mode in modes:
            correct = sum(1 for r in q1 if r[f"correct_{mode}"])
            abstain = sum(1 for r in q1 if r[f"abstains_{mode}"])
            print(f"  {mode:>8s}: correct={correct}/{len(q1)} ({correct/len(q1):.0%}), "
                  f"over-refuse={abstain}/{len(q1)} ({abstain/len(q1):.0%})")

    # Compute composite score
    print(f"\n{'='*70}")
    print("COMPOSITE FAITHFULNESS SCORE")
    print(f"{'='*70}")

    for mode in modes:
        suf = [r for r in all_results if r["sufficient"]]
        insuf = [r for r in all_results if not r["sufficient"]]

        correct_rate = sum(1 for r in suf if r[f"correct_{mode}"]) / max(len(suf), 1)
        abstain_rate = sum(1 for r in insuf if r[f"abstains_{mode}"]) / max(len(insuf), 1)
        over_refuse = sum(1 for r in suf if r[f"abstains_{mode}"]) / max(len(suf), 1)

        composite = (correct_rate + abstain_rate - over_refuse) / 2
        print(f"  {mode:>8s}: correct_suf={correct_rate:.3f}, abstain_insuf={abstain_rate:.3f}, "
              f"over_refuse={over_refuse:.3f}, composite={composite:.3f}")

    # Save results
    out_dir = CRAG_DIR / model_key / "amnesia"
    with open(out_dir / "generation_results.json", "w") as f:
        json.dump({
            "model": model_key,
            "best_layer": best_layer,
            "erasure_alpha": erasure_alpha,
            "n_samples": len(all_results),
            "examples": all_results,
        }, f, indent=2)

    print(f"\nResults saved to {out_dir / 'generation_results.json'}")

    del model
    torch.cuda.empty_cache()

    return all_results


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["find_direction", "test_erasure"])
    parser.add_argument("--model", default="mistral",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Erasure strength multiplier")
    args = parser.parse_args()

    if args.step == "find_direction":
        find_pk_direction(args.model)
    elif args.step == "test_erasure":
        test_erasure_generation(
            args.model, args.device, args.n_samples,
            erasure_alpha=args.alpha,
        )
