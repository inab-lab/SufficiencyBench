"""
crag.py — Contrastive Representation-guided Adaptive Grounding

Uses the Context Contribution Direction (CCD) = h(q+c) - h(q) to:
1. Detect per-token grounding during generation
2. Adaptively steer the model toward context faithfulness

Key insight: the direction from question-only to question+context hidden states
captures "what the context contributes." Projecting generated tokens onto this
direction yields a per-token grounding score. When the score drops, steering
pushes the model back toward context-informed generation.

Usage:
  # Step 1a: Compute CCD
  python src/methods/crag.py --step compute_ccd --model mistral

  # Step 1b: Pilot validation
  python src/methods/crag.py --step pilot --model mistral --device cuda:0

  # Step 2: Full generation with steering
  python src/methods/crag.py --step generate --model mistral --device cuda:0 --mode adaptive
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
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
# STEP 1a: Compute Context Contribution Direction (CCD)
# ================================================================

def compute_ccd(model_key: str = "mistral"):
    """
    Compute the Context Contribution Direction from existing hidden states.
    CCD[L] = normalize(mean(h_with_context - h_question_only)) per layer.
    Select best layer by AUROC of 1D projection for sufficient/insufficient.
    """
    print(f"\n{'='*70}")
    print(f"COMPUTING CCD — {model_key}")
    print(f"{'='*70}")

    # Load existing hidden states (memory-mapped to avoid OOM)
    hs_dir = HIDDEN_STATES_DIR / model_key / "train"
    h_qc = np.load(hs_dir / "h_with_context.npy", mmap_mode="r")
    h_q = np.load(hs_dir / "h_question_only.npy", mmap_mode="r")
    labels = np.load(hs_dir / "labels.npy")

    n_samples, n_layers, hidden_dim = h_qc.shape
    print(f"Data: {n_samples} samples, {n_layers} layers, {hidden_dim} dim")
    print(f"Sufficient: {labels.sum()}, Insufficient: {(1-labels).sum()}")

    out_dir = CRAG_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    best_auroc = 0
    best_layer = 0

    print(f"\n{'Layer':>5} {'AUROC':>8} {'Separation':>12} {'Suf_mean':>10} {'Insuf_mean':>12}")
    print("-" * 50)

    for L in range(n_layers):
        # Load one layer at a time to avoid OOM
        delta_L = np.array(h_qc[:, L, :]) - np.array(h_q[:, L, :])  # (N, hidden_dim)

        # CCD = normalized mean of delta
        ccd_raw = delta_L.mean(axis=0)  # (hidden_dim,)
        norm = np.linalg.norm(ccd_raw)
        if norm < 1e-10:
            continue
        ccd = ccd_raw / norm

        # Project all examples onto CCD
        projections = delta_L @ ccd  # (N,)

        # AUROC for classifying sufficient vs insufficient
        auroc = roc_auc_score(labels, projections)

        # Separation (Cohen's d)
        suf_projs = projections[labels == 1]
        insuf_projs = projections[labels == 0]
        pooled_std = np.sqrt(
            (suf_projs.var() + insuf_projs.var()) / 2
        )
        separation = (suf_projs.mean() - insuf_projs.mean()) / max(pooled_std, 1e-10)

        results[L] = {
            "auroc": float(auroc),
            "separation": float(separation),
            "suf_mean": float(suf_projs.mean()),
            "insuf_mean": float(insuf_projs.mean()),
            "ccd_norm": float(norm),
        }

        if L % 2 == 0 or L == n_layers - 1 or L >= n_layers - 4:
            print(f"L{L:>4d} {auroc:>8.4f} {separation:>12.3f} "
                  f"{suf_projs.mean():>10.3f} {insuf_projs.mean():>12.3f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = L

    print(f"\nBest layer: L{best_layer} (AUROC={best_auroc:.4f})")

    # Save CCD for the best layer (and a few neighbors)
    layers_to_save = set([best_layer, max(0, best_layer-1), min(n_layers-1, best_layer+1)])
    for L in layers_to_save:
        delta_L = np.array(h_qc[:, L, :]) - np.array(h_q[:, L, :])
        ccd_raw = delta_L.mean(axis=0)
        ccd = ccd_raw / np.linalg.norm(ccd_raw)
        np.save(out_dir / f"ccd_L{L}.npy", ccd)

    # Save results
    summary = {
        "model": model_key,
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "per_layer": results,
    }
    with open(out_dir / "ccd_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCCD saved to {out_dir}")

    # GO/NO-GO
    if best_auroc > 0.7:
        print(f"\nGO: CCD AUROC {best_auroc:.4f} > 0.7 threshold")
    else:
        print(f"\nWARNING: CCD AUROC {best_auroc:.4f} < 0.7. Signal may be weak.")

    return summary


# ================================================================
# STEP 1b: Pilot Validation — Per-Token Grounding Scores
# ================================================================

def validate_ccd_pilot(
    model_key: str = "mistral",
    device: str = "cuda:0",
    n_samples: int = 200,
    max_new_tokens: int = 128,
):
    """
    Pilot test: generate answers and measure per-token grounding scores.
    Validates that CCD projections during generation separate sufficient
    from insufficient conditions.
    """
    print(f"\n{'='*70}")
    print(f"PILOT VALIDATION — {model_key}")
    print(f"{'='*70}")

    cfg = MODEL_CONFIGS[model_key]
    model_path = cfg["path"]

    # Load CCD
    crag_dir = CRAG_DIR / model_key
    with open(crag_dir / "ccd_results.json") as f:
        ccd_info = json.load(f)
    best_layer = ccd_info["best_layer"]
    ccd = np.load(crag_dir / f"ccd_L{best_layer}.npy")
    ccd_tensor = torch.tensor(ccd, dtype=torch.float16, device=device)

    print(f"Using CCD at layer {best_layer} (AUROC={ccd_info['best_auroc']:.4f})")

    # Load model
    print(f"Loading {model_key}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Load test data
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Sample balanced set
    examples = []
    for ex in test_data:
        for cond in ex["conditions"]:
            examples.append({
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "prompt": cond["prompt"],
                "context": cond["context"],
                "sufficient": cond["sufficient"],
                "quadrant": cond["quadrant"],
                "condition_id": cond["condition_id"],
            })

    suf_examples = [e for e in examples if e["sufficient"]]
    insuf_examples = [e for e in examples if not e["sufficient"]]
    n_each = min(n_samples // 2, len(suf_examples), len(insuf_examples))

    np.random.seed(42)
    selected = (
        [suf_examples[i] for i in np.random.choice(len(suf_examples), n_each, replace=False)] +
        [insuf_examples[i] for i in np.random.choice(len(insuf_examples), n_each, replace=False)]
    )
    print(f"Selected {len(selected)} examples ({n_each} sufficient, {n_each} insufficient)")

    # Question-only prompt template
    def format_question_only(question: str) -> str:
        return (
            f"Based on the following context, answer the question. "
            f"If the context does not contain enough information, "
            f"say 'I cannot answer this from the provided context.'\n\n"
            f"Context: [No context provided]\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    # Generate and track grounding scores
    all_results = []

    for ex in tqdm(selected, desc="Pilot generation"):
        # 1. Get h_q baseline (question-only forward pass)
        q_only_prompt = format_question_only(ex["question"])
        q_inputs = tokenizer(
            q_only_prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(device)

        with torch.no_grad():
            q_outputs = model(**q_inputs, output_hidden_states=True)
        h_q = q_outputs.hidden_states[best_layer + 1][0, -1, :]  # +1 to skip embedding

        # 2. Generate with full prompt, tracking per-token grounding
        inputs = tokenizer(
            ex["prompt"], return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(device)

        generated_tokens = []
        grounding_scores = []
        past_key_values = None

        # Initial forward pass for the prompt
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values

        # Get grounding score for the last prompt token
        h_current = outputs.hidden_states[best_layer + 1][0, -1, :]
        h_delta = h_current - h_q
        score = torch.dot(h_delta.float(), ccd_tensor.float()).item()
        grounding_scores.append(score)

        # Get first generated token
        next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        # Auto-regressive generation
        for step in range(max_new_tokens):
            token_id = next_token[0, 0].item()
            if token_id == tokenizer.eos_token_id:
                break
            generated_tokens.append(token_id)

            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            # Grounding score
            h_current = outputs.hidden_states[best_layer + 1][0, -1, :]
            h_delta = h_current - h_q
            score = torch.dot(h_delta.float(), ccd_tensor.float()).item()
            grounding_scores.append(score)

            next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        all_results.append({
            "condition_id": ex["condition_id"],
            "sufficient": ex["sufficient"],
            "quadrant": ex["quadrant"],
            "question_type": ex["question_type"],
            "generated_text": generated_text[:500],
            "gold_answer": ex["gold_answer"],
            "grounding_scores": grounding_scores,
            "mean_grounding": float(np.mean(grounding_scores)),
            "std_grounding": float(np.std(grounding_scores)),
            "min_grounding": float(np.min(grounding_scores)),
            "n_tokens": len(generated_tokens),
        })

        # Memory cleanup
        if len(all_results) % 50 == 0:
            torch.cuda.empty_cache()

    # ---- Analysis ----
    print(f"\n{'='*70}")
    print("PILOT RESULTS")
    print(f"{'='*70}")

    suf_scores = [r["mean_grounding"] for r in all_results if r["sufficient"]]
    insuf_scores = [r["mean_grounding"] for r in all_results if not r["sufficient"]]

    print(f"\nMean grounding score:")
    print(f"  Sufficient context:   {np.mean(suf_scores):.4f} +/- {np.std(suf_scores):.4f}")
    print(f"  Insufficient context: {np.mean(insuf_scores):.4f} +/- {np.std(insuf_scores):.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(suf_scores, insuf_scores)
    print(f"\n  t-test: t={t_stat:.4f}, p={p_value:.2e}")

    # AUROC
    all_scores = suf_scores + insuf_scores
    all_labels = [1] * len(suf_scores) + [0] * len(insuf_scores)
    auroc = roc_auc_score(all_labels, all_scores)
    print(f"  AUROC (mean grounding -> sufficient): {auroc:.4f}")

    # Per-quadrant
    print(f"\nPer-quadrant mean grounding:")
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        q_scores = [r["mean_grounding"] for r in all_results if r["quadrant"] == quad]
        if q_scores:
            print(f"  {quad}: {np.mean(q_scores):.4f} +/- {np.std(q_scores):.4f} (n={len(q_scores)})")

    # Save results
    out_dir = CRAG_DIR / model_key
    with open(out_dir / "pilot_results.json", "w") as f:
        json.dump({
            "n_samples": len(all_results),
            "suf_mean": float(np.mean(suf_scores)),
            "insuf_mean": float(np.mean(insuf_scores)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "auroc": float(auroc),
            "best_layer": best_layer,
            "examples": all_results,
        }, f, indent=2)

    # GO/NO-GO
    print(f"\n{'='*70}")
    if p_value < 0.01 and auroc > 0.55:
        print(f"GO: Grounding scores separate conditions (p={p_value:.2e}, AUROC={auroc:.4f})")
    elif p_value < 0.05:
        print(f"MARGINAL: Weak separation (p={p_value:.2e}, AUROC={auroc:.4f}). Consider tweaks.")
    else:
        print(f"NO-GO: No separation (p={p_value:.2e}, AUROC={auroc:.4f}). Need different approach.")
    print(f"{'='*70}")

    del model
    torch.cuda.empty_cache()

    return all_results


# ================================================================
# STEP 2: CRAGGenerator — Adaptive Steering During Generation
# ================================================================

class CRAGGenerator:
    """
    Generation wrapper with per-token grounding detection and adaptive steering.

    Modes:
      - no_steering: just generate, record grounding scores
      - static_steering: always add CCD to hidden states (like ContextFocus)
      - adaptive_steering: add CCD only when grounding score drops below threshold
    """

    def __init__(
        self,
        model,
        tokenizer,
        ccd: np.ndarray,
        layer_idx: int,
        device: str = "cuda:0",
        threshold: float = 0.0,
        alpha: float = 1.0,
        mode: str = "adaptive",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode

        self.ccd_tensor = torch.tensor(
            ccd, dtype=torch.float16, device=device
        )

        # State for the current generation
        self._h_q = None
        self._current_scores = []
        self._hook_handle = None

    def _compute_question_baseline(self, question: str) -> torch.Tensor:
        """Forward pass on question-only prompt, return h_q at target layer."""
        prompt = (
            f"Based on the following context, answer the question. "
            f"If the context does not contain enough information, "
            f"say 'I cannot answer this from the provided context.'\n\n"
            f"Context: [No context provided]\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        return outputs.hidden_states[self.layer_idx + 1][0, -1, :]

    def _steering_hook(self, module, input, output):
        """Hook applied to target layer during generation."""
        hidden_states = output[0]  # (batch, seq_len, hidden_dim)
        h_current = hidden_states[0, -1, :]  # last token

        # Compute grounding score
        h_delta = h_current.float() - self._h_q.float()
        score = torch.dot(h_delta, self.ccd_tensor.float()).item()
        self._current_scores.append(score)

        # Apply steering based on mode
        if self.mode == "static_steering":
            hidden_states[0, -1, :] += self.alpha * self.ccd_tensor
        elif self.mode == "adaptive_steering":
            deficit = self.threshold - score
            if deficit > 0:
                hidden_states[0, -1, :] += self.alpha * deficit * self.ccd_tensor

        return (hidden_states,) + output[1:]

    def generate(
        self, prompt: str, question: str, max_tokens: int = 256,
    ) -> dict:
        """
        Generate with grounding detection and optional steering.
        Returns dict with generated text, per-token scores, and metadata.
        """
        # Cache h_q
        self._h_q = self._compute_question_baseline(question)
        self._current_scores = []

        # Determine the target layer module
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_layer = self.model.model.layers[self.layer_idx]
        else:
            raise ValueError(f"Cannot find layers in model architecture")

        # Register hook
        if self.mode != "no_steering":
            self._hook_handle = target_layer.register_forward_hook(
                self._steering_hook
            )

        try:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(self.device)

            generated_tokens = []
            past_key_values = None

            # Initial forward pass
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            # Record prompt grounding score (if no hook, compute manually)
            if self.mode == "no_steering":
                h_current = outputs.hidden_states[self.layer_idx + 1][0, -1, :]
                h_delta = h_current.float() - self._h_q.float()
                score = torch.dot(h_delta, self.ccd_tensor.float()).item()
                self._current_scores.append(score)

            next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

            # Auto-regressive generation
            for step in range(max_tokens):
                token_id = next_token[0, 0].item()
                if token_id == self.tokenizer.eos_token_id:
                    break
                generated_tokens.append(token_id)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values

                # Record grounding score (if no hook, compute manually)
                if self.mode == "no_steering":
                    h_current = outputs.hidden_states[self.layer_idx + 1][0, -1, :]
                    h_delta = h_current.float() - self._h_q.float()
                    score = torch.dot(h_delta, self.ccd_tensor.float()).item()
                    self._current_scores.append(score)

                next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        finally:
            # Remove hook
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None

        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return {
            "generated_text": generated_text,
            "grounding_scores": self._current_scores,
            "mean_grounding": float(np.mean(self._current_scores)) if self._current_scores else 0.0,
            "n_tokens": len(generated_tokens),
            "mode": self.mode,
        }


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["compute_ccd", "pilot"])
    parser.add_argument("--model", default="mistral",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    if args.step == "compute_ccd":
        compute_ccd(args.model)
    elif args.step == "pilot":
        validate_ccd_pilot(args.model, args.device, args.n_samples)
