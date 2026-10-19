"""
redeep.py

ReDeEP hallucination / context-sufficiency detector for SufficiencyBench.

Reference
---------
Sun et al. (2024/2025), "ReDeEP: Detecting Hallucination in Retrieval-Augmented
Generation via Mechanistic Interpretability", ICLR 2025. arXiv:2410.11414.
https://arxiv.org/abs/2410.11414

Formulation (this module)
-------------------------
ReDeEP decouples an LLM's reliance on *external context* (routed through
attention / "copying heads") from its reliance on *parametric knowledge*
(injected by the FFN / "knowledge neurons"). We implement a token-level
version over the model's own greedily-generated response tokens r, using a
single forward pass over [prompt | response] with attentions, hidden states
and FFN-output hooks.

External Context Score (ECS)  -- paper Eq. 3-4, "copying heads":
    For a response query token n, layer l, head h, let I(n,l,h) be the top-k%
    (k=10%) *context* key tokens with the highest attention weight. With the
    final-layer hidden state x^L,
        e            = mean_{j in I(n,l,h)} x_j^L
        ECS(n,l,h)   = cos( x_n^L , e )
    ECS(r) = mean over response tokens n, heads h and layers l.
    Higher ECS  => the model is copying/attending to the retrieved context
                   => context is being *used* => context is SUFFICIENT.

Parametric Knowledge Score (PKS)  -- paper Eq. 5-6, "knowledge FFNs":
    For response token n, layer l, with LogitLens q(x)=softmax(lm_head(norm(x))),
    let x_mid = residual stream *before* the FFN and x_post = x_mid + FFN(.)
    the residual stream *after* the FFN. Then
        PKS(n,l)     = JSD( q(x_mid) || q(x_post) )
    PKS(r) = mean over response tokens n and layers l.
    Higher PKS  => the FFN is overwriting the residual stream with parametric
                   knowledge => the model relies on internal memory rather than
                   the context => context is INSUFFICIENT (hallucination-prone).

Combined  -- paper Eq. (Sec 4.1), hallucination score H = a*PKS - b*ECS with
regression coefficients a,b>0 (higher H = more hallucination). Because no
labelled fit set is available here, we use the standardized equal-weight
combination and *flip the sign* so higher = sufficient (matching ECS/PKS):
        ReDeEP-Combined = zscore(ECS) - zscore(PKS)   ( = -H, standardized )

Sign convention for ALL three scores: **higher => context SUFFICIENT**, so
that roc_auc_score(sufficient_label, score) is maximised.

Caveats
-------
- The paper first *searches* for a small set of specific "copying heads" and
  "knowledge FFN layers" per model. We have no such per-model selection here,
  so ECS/PKS are averaged over ALL heads/layers (a model-agnostic
  approximation). This will not reproduce the paper's exact AUROC numbers.
- The paper regresses a,b on a labelled set (RAGTruth); we substitute a
  standardized equal-weight combination.

Usage
-----
  python src/methods/redeep.py --model llama --device cuda:0
  python src/methods/redeep.py --model llama --device cuda:1 --limit 5   # smoke
"""

import json
import argparse
import sys
import os

# Reduce allocator fragmentation from the per-layer attention tensors below.
# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Repo root + src on path (mirror baselines.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS

# --- Hyperparameters (defaults follow the paper where applicable) -----------
TOP_K_FRAC = 0.10        # top-k% attended context tokens for ECS (paper: 10%)
MAX_KK = 64              # cap on #top tokens kept, to bound memory
MAX_RESP_TOKENS = 32     # response tokens analysed per example
GEN_MAX_NEW_TOKENS = 32  # greedy generation length


def _logit_lens_probs(model, hidden):
    """q(x) = softmax(lm_head(final_norm(x))).  hidden: [N, D] -> probs [N, V]."""
    norm = model.model.norm            # final RMSNorm (Llama/Mistral/Qwen)
    lm_head = model.get_output_embeddings()
    x = norm(hidden)
    logits = lm_head(x)
    return F.softmax(logits.float(), dim=-1)


def _jsd(p, q, eps=1e-12):
    """Jensen-Shannon divergence per row.  p,q: [N, V] -> [N]."""
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def _context_token_positions(prompt, context, tokenizer, max_length):
    """Token indices (into the truncated prompt) that fall inside `context`.

    Uses fast-tokenizer offset mapping. Returns list[int]; falls back to all
    prompt tokens if the context span can't be located.
    """
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_length, return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    n_tok = len(offsets)
    start = prompt.find(context) if context else -1
    if start < 0:
        # Fallback: treat everything except the last 20 tokens (question/instr)
        # as context so a score can still be produced.
        return list(range(max(0, n_tok - 20)))
    end = start + len(context)
    pos = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
    return pos if pos else list(range(max(0, n_tok - 20)))


def redeep_scores(model, tokenizer, prompt, context, device, max_length,
                  ffn_out_store):
    """Return (ecs, pks) for one prompt.

    ffn_out_store: dict mutated by the registered MLP hooks (layer_idx -> tensor).
    """
    ctx_pos = _context_token_positions(prompt, context, tokenizer, max_length)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # 1) Greedy response.
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_ids = gen[0]
    seq_len = full_ids.shape[0]
    if seq_len <= prompt_len:            # empty generation -> neutral
        return 0.0, 0.0

    # Response query positions (cap for cost).
    resp_pos = list(range(prompt_len, seq_len))[:MAX_RESP_TOKENS]

    del inputs, gen
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 2) Single forward over [prompt|response] with internals.
    ffn_out_store.clear()
    with torch.no_grad():
        out = model(
            input_ids=full_ids.unsqueeze(0),
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    attentions = out.attentions            # tuple[L] each [1,H,S,S]
    hidden_states = out.hidden_states       # tuple[L+1] each [1,S,D]
    n_layers = len(attentions)

    # Valid context positions inside this (possibly truncated) sequence.
    ctx_pos = [c for c in ctx_pos if c < seq_len]

    # ---- ECS: attention/copying-head reliance on context --------------------
    ecs_val = 0.0
    if ctx_pos:
        H_L = hidden_states[-1][0]                      # [S, D] final layer
        ctx_hidden = H_L[ctx_pos]                       # [|C|, D]
        x_r = H_L[resp_pos]                             # [|R|, D]
        kk = max(1, min(int(len(ctx_pos) * TOP_K_FRAC), MAX_KK, len(ctx_pos)))
        ecs_accum, ecs_count = 0.0, 0
        for l in range(n_layers):
            att = attentions[l][0]                       # [H, S, S]
            # attention from each response token to each context token
            att_rc = att[:, resp_pos][:, :, ctx_pos]     # [H, |R|, |C|]
            _, topi = att_rc.topk(kk, dim=-1)            # [H, |R|, kk] idx into C
            e = ctx_hidden[topi].mean(dim=2)             # [H, |R|, D]
            cos = F.cosine_similarity(
                e.float(), x_r.float().unsqueeze(0), dim=-1)  # [H, |R|]
            ecs_accum += cos.mean().item()
            ecs_count += 1
            del att, att_rc, topi, e, cos
        ecs_val = ecs_accum / max(ecs_count, 1)

    # ---- PKS: FFN / parametric-knowledge injection --------------------------
    pks_accum, pks_count = 0.0, 0
    for l in range(n_layers):
        ffn_out = ffn_out_store.get(l)
        if ffn_out is None:
            continue
        x_post = hidden_states[l + 1][0][resp_pos]       # residual after FFN
        x_ffn = ffn_out[0][resp_pos]                     # FFN contribution
        x_mid = x_post - x_ffn                           # residual before FFN
        p_mid = _logit_lens_probs(model, x_mid)
        p_post = _logit_lens_probs(model, x_post)
        pks_accum += _jsd(p_mid, p_post).mean().item()
        pks_count += 1
        del x_post, x_ffn, x_mid, p_mid, p_post
    pks_val = pks_accum / max(pks_count, 1)

    del out, attentions, hidden_states
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return float(ecs_val), float(pks_val)


def _zscore(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    if s < 1e-9:
        return np.zeros_like(a)
    return (a - a.mean()) / s


# Per-model load dtype. Llama/Mistral are numerically stable in fp16 and keep
# it so their existing results stay bit-for-bit valid. Qwen2.5, however, was
# trained in bf16 and carries extreme "massive activations": its final decoder
# layer (attn + SwiGLU MLP over the 18944-dim intermediate) produces values
# that exceed the fp16 max (65504) and overflow to +/-inf. That inf lands in
# the final hidden state and the layer-27 FFN output, so LogitLens -> NaN and
# the ECS cosine -> NaN, turning *every* condition's score non-finite (they
# then get imputed to a constant => chance AUROC 0.5). The overflow happens
# inside the forward pass, so it cannot be recovered by casting a downstream
# op -- the tensors are already inf when redeep sees them. Loading Qwen in its
# native bf16 (which has the dynamic range for these activations, and is what
# extract_states.py / deco_rag.py already use) keeps every value finite. See
# https://arxiv.org/abs/2402.17762 (massive activations).
_MODEL_DTYPES = {"qwen": torch.bfloat16}


def run_redeep(model_key="llama", device="cuda:0", limit=0, max_length=1024):
    cfg = MODEL_CONFIGS[model_key]
    dtype = _MODEL_DTYPES.get(model_key, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=dtype, device_map=device,
        output_attentions=True,
    )
    model.eval()
    # NB: leaving output_attentions on in the config makes generate() materialise
    # full attention tensors at every decode step (OOM). We disable it globally
    # and re-request attentions only in the explicit analysis forward pass.
    model.config.output_attentions = False

    # Register hooks capturing each decoder layer's FFN (MLP) output.
    ffn_out_store = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def _hook(mod, inp, outp, idx=i):
            ffn_out_store[idx] = outp.detach()
        handles.append(layer.mlp.register_forward_hook(_hook))

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten conditions exactly like baselines.py.
    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question_type": ex.get("question_type"),
                "gold_answer": ex.get("gold_answer", ""),
            })
    if limit and limit > 0:
        conditions = conditions[:limit]

    ecs_list, pks_list = [], []
    for cond in tqdm(conditions, desc=f"redeep[{model_key}]"):
        ecs, pks = redeep_scores(
            model, tokenizer, cond["prompt"], cond.get("context", ""),
            device, max_length, ffn_out_store,
        )
        ecs_list.append(ecs)
        pks_list.append(pks)

    for h in handles:
        h.remove()

    labels = [1 if c.get("sufficient") else 0 for c in conditions]
    quads = [c.get("quadrant") for c in conditions]

    # Sign convention: higher => sufficient.
    #   ECS as-is (more context reliance = sufficient)
    #   PKS negated (more parametric reliance = insufficient)
    #   Combined = z(ECS) - z(PKS)  ( = -H_hallucination, standardized )
    combined = (_zscore(ecs_list) - _zscore(pks_list)).tolist()
    method_scores = {
        "redeep_ecs": list(ecs_list),
        "redeep_pks": [-p for p in pks_list],
        "redeep_combined": combined,
    }

    results = {}
    both_classes = len(set(labels)) >= 2
    for name, scores in method_scores.items():
        # Sanitize: replace any non-finite score (e.g. a degenerate zscore) with
        # the finite mean so a few bad conditions can't NaN-crash roc_auc and
        # lose the whole run.
        arr = np.asarray(scores, dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            fill = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
            arr = np.where(np.isfinite(arr), arr, fill)
            print(f"  [warn] {name}: imputed {n_bad} non-finite scores with {fill:.4f}")
        scores = arr.tolist()
        entry = {"n_imputed": n_bad}
        try:
            if both_classes:
                entry["overall_auroc"] = float(roc_auc_score(labels, scores))
            else:
                entry["overall_auroc"] = float("nan")
            for quad in ["Q1", "Q2", "Q3", "Q4"]:
                idx = [i for i, q in enumerate(quads) if q == quad]
                if len(idx) < 10 or len(set(labels[i] for i in idx)) < 2:
                    continue
                entry[f"{quad}_auroc"] = float(
                    roc_auc_score([labels[i] for i in idx], [scores[i] for i in idx]))
            print(f"{name}: overall AUROC = {entry['overall_auroc']:.4f}")
        except Exception as e:  # never let one method sink the others
            entry["error"] = str(e)
            print(f"  [error] {name}: {e}")
        results[name] = entry

    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "redeep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir / 'redeep_results.json'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0,
                    help="Truncate to N conditions for smoke testing (0 = all).")
    ap.add_argument("--max_length", type=int, default=1024,
                    help="Max prompt tokens. output_attentions memory scales "
                         "with seq^2 (~2.3GB @1024, ~8.6GB @2048 for an 8B "
                         "model); raise only if the GPU has headroom.")
    args = ap.parse_args()
    run_redeep(args.model, args.device, args.limit, args.max_length)


if __name__ == "__main__":
    main()
