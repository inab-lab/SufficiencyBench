"""
contrastive_baselines.py  (Issue 7)

Three new behavioral baselines that attempt to access the sufficiency
signal via context-vs-no-context comparison:

  A. Response Delta:  cosine similarity between model answers with/without context.
     Low similarity → context changed the answer → context was informative (sufficient).

  B. Token Probability Delta:  log P(gold | q+c) - log P(gold | q).
     High delta → context substantially raised confidence in the answer → sufficient.

  C. Verbalized Comparison Prompt:  "Answer with context, then without.
     Rate 0-100 how much the context changed your answer."

These are the "obvious fixes" a reviewer might suggest. If they still
plateau at ~0.73, that strongly confirms the structural limitation claim.

Usage:
  python src/methods/contrastive_baselines.py --model llama --device cuda:0
"""

import json
import torch
import numpy as np
import re
import argparse
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS


def encode_text(tokenizer, text: str, device: str, max_len: int = 2048):
    return tokenizer(text, return_tensors="pt",
                     truncation=True, max_length=max_len).to(device)


def get_generation(model, tokenizer, inputs, max_new_tokens: int = 80,
                   device: str = "cuda:0") -> str:
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.0, do_sample=False,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def response_delta(model, tokenizer, prompt_with_ctx: str,
                   question: str, embed_model, device: str) -> float:
    """
    Generate answer with and without context.
    Low similarity → context changed the answer → sufficient.
    Returns score where HIGH = more likely sufficient.
    """
    # Answer with context
    inputs_ctx = encode_text(tokenizer, prompt_with_ctx, device)
    ans_with = get_generation(model, tokenizer, inputs_ctx, device=device)

    # Answer without context (question only, same prompt format)
    prompt_no_ctx = (
        f"Answer the following question. If you don't know, say 'I don't know'.\n\n"
        f"Question: {question}\nAnswer:"
    )
    inputs_no_ctx = encode_text(tokenizer, prompt_no_ctx, device)
    ans_without = get_generation(model, tokenizer, inputs_no_ctx, device=device)

    if not ans_with or not ans_without:
        return 0.5

    # Embed both answers
    emb_with = embed_model.encode(ans_with)
    emb_without = embed_model.encode(ans_without)
    cos_sim = float(
        (emb_with @ emb_without) /
        ((emb_with @ emb_with) ** 0.5 * (emb_without @ emb_without) ** 0.5 + 1e-8)
    )
    # Low similarity = context changed answer = sufficient
    return 1.0 - max(0.0, cos_sim)


def token_prob_delta(model, tokenizer, prompt_with_ctx: str,
                     question: str, gold_answer: str, device: str) -> float:
    """
    log P(gold | q+c) - log P(gold | q alone).
    High delta → context substantially helped → sufficient.
    """
    def log_prob_answer(prompt: str, answer: str) -> float:
        full_text = prompt + " " + answer
        inp = encode_text(tokenizer, full_text, device)
        ans_inp = encode_text(tokenizer, prompt + " ", device)
        n_prompt = ans_inp["input_ids"].shape[1]

        with torch.no_grad():
            out = model(**inp)

        logits = out.logits[0, n_prompt - 1:-1, :]
        tgt = inp["input_ids"][0, n_prompt:]
        if len(tgt) == 0:
            return 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(tgt)), tgt
        ]
        return float(token_log_probs.mean().item())

    prompt_no_ctx = (
        f"Answer the following question.\n\nQuestion: {question}\nAnswer:"
    )

    lp_ctx = log_prob_answer(prompt_with_ctx, gold_answer)
    lp_no_ctx = log_prob_answer(prompt_no_ctx, gold_answer)
    delta = lp_ctx - lp_no_ctx
    # Positive delta → context helped → likely sufficient
    # Normalize to [0, 1] range: sigmoid
    return float(1 / (1 + np.exp(-delta)))


def verbalized_comparison(model, tokenizer, prompt_with_ctx: str,
                           question: str, device: str) -> float:
    """
    Prompt: answer with context, then without, then rate how much context changed things.
    """
    compare_prompt = (
        f"{prompt_with_ctx}\n\n"
        f"Now answer the same question WITHOUT using the context above, "
        f"from memory only:\nMemory answer:"
    )
    inputs = encode_text(tokenizer, compare_prompt, device)
    ans_memory = get_generation(model, tokenizer, inputs, max_new_tokens=60,
                                device=device)

    rate_prompt = (
        f"{compare_prompt}\n{ans_memory}\n\n"
        f"On a scale of 0 to 100, how much did the context change your answer? "
        f"(0=no change, 100=completely different). Reply with ONLY a number.\nScore:"
    )
    inputs2 = encode_text(tokenizer, rate_prompt, device)
    score_str = get_generation(model, tokenizer, inputs2, max_new_tokens=5,
                                device=device)

    nums = re.findall(r'\d+', score_str)
    if nums:
        return min(max(float(nums[0]) / 100, 0.0), 1.0)
    return 0.5


# ---------------------------------------------------------------------------
# BATCHED implementations.
#
# These reproduce the per-item MATH of response_delta / token_prob_delta /
# verbalized_comparison exactly, but run generation / forward passes / encoding
# in batches for a large wall-time / heat reduction.
#
#   * GENERATION (response_delta, verbalized_comparison): tokenizer.padding_side
#     == "left", so every row's prompt is right-aligned and the continuation
#     starts at input_ids.shape[1] for every row.
#   * FORWARD PASS (token_prob_delta): _batch_log_prob uses RIGHT-padding so real
#     tokens stay front-aligned at indices [0:L] and the answer slice is the
#     exact [n_prompt-1 : L-1] indexing of the original log_prob_answer().
#   * ENCODING: SentenceTransformer.encode() is called on whole lists.
#
# FIDELITY CAVEAT (important) -- with load_in_8bit (bitsandbytes LLM.int8) the
# outlier feature-columns kept in fp16 are selected from statistics gathered
# over the WHOLE batch, so any batch_size > 1 shifts the fp16/int8 split for
# every token. Batched results therefore are NOT bit-identical to the per-item
# path: token_prob_delta differs by ~1e-1 on some items, and greedy generation
# occasionally flips an argmax (changing a whole continuation). This is a
# property of 8-bit batched inference, not of the math here -- it is independent
# of padding side (left- and right-padding diverge by the same ~0.1). Running
# with --batch-size 1 goes through this same code and is bit-identical to the
# original per-condition script (verified: max abs diff 0.0), so bs=1 is the
# exact-reproduction mode and bs>1 trades a small per-item numeric shift for a
# ~5x speedup. See the accompanying report for measured diffs.
# ---------------------------------------------------------------------------

def _batch_generate(model, tokenizer, prompts, max_new_tokens, batch_size,
                    device):
    """Greedy-generate continuations for a list of prompts, in batches.
    Returns a list of decoded (stripped) continuation strings, one per prompt,
    in the same order."""
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        start = enc["input_ids"].shape[1]  # left-padding => same for all rows
        for row in gen[:, start:]:
            outs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return outs


def _batch_log_prob(model, tokenizer, prompts, answers, batch_size, device):
    """Batched equivalent of token_prob_delta's inner log_prob_answer().
    Returns mean log P(answer | prompt) for each (prompt, answer) pair.

    Uses RIGHT-padding for this pure forward pass: real tokens occupy indices
    [0:L] exactly as in the unpadded reference (pads land at future positions,
    masked out by causal attention + attention_mask), so the answer-token slice
    is [n_prompt-1 : L-1] — the exact indexing of the original log_prob_answer(),
    and no position_ids fix-up is needed. NB: under load_in_8bit this is still
    not bit-identical to the per-item path for batch_size > 1 (batch-global
    outlier selection, see module header); the divergence is the same magnitude
    for left- vs right-padding, so right-padding is chosen purely because it
    reproduces the original front-aligned indexing."""
    scores = []
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for i in range(0, len(prompts), batch_size):
            pchunk = prompts[i:i + batch_size]
            achunk = answers[i:i + batch_size]
            full_texts = [p + " " + a for p, a in zip(pchunk, achunk)]
            # n_prompt from "prompt + ' '" tokenised alone, exactly as original.
            n_prompts = [
                len(tokenizer(p + " ", truncation=True,
                              max_length=2048)["input_ids"])
                for p in pchunk
            ]
            enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(device)
            attn = enc["attention_mask"]
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=attn)
            for r in range(len(pchunk)):
                L = int(attn[r].sum().item())      # real (unpadded) length
                n_prompt = n_prompts[r]
                if L - n_prompt <= 0:              # no answer tokens survived
                    scores.append(0.0)
                    continue
                logits = out.logits[r, n_prompt - 1:L - 1, :]
                tgt = enc["input_ids"][r, n_prompt:L]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[torch.arange(len(tgt)), tgt]
                scores.append(float(token_log_probs.mean().item()))
    finally:
        tokenizer.padding_side = saved_side
    return scores


def response_delta_batched(model, tokenizer, prompts, questions, embed_model,
                           batch_size, device):
    """Batched response_delta over aligned lists of prompts/questions."""
    no_ctx_prompts = [
        (f"Answer the following question. If you don't know, say 'I don't know'."
         f"\n\nQuestion: {q}\nAnswer:")
        for q in questions
    ]
    ans_with = _batch_generate(model, tokenizer, prompts, 80, batch_size, device)
    ans_without = _batch_generate(model, tokenizer, no_ctx_prompts, 80,
                                  batch_size, device)
    # Batched SentenceTransformer encode (accepts a list).
    emb_with = embed_model.encode(ans_with)
    emb_without = embed_model.encode(ans_without)
    scores = []
    for aw, awo, ew, ewo in zip(ans_with, ans_without, emb_with, emb_without):
        if not aw or not awo:
            scores.append(0.5)
            continue
        cos_sim = float(
            (ew @ ewo) /
            ((ew @ ew) ** 0.5 * (ewo @ ewo) ** 0.5 + 1e-8)
        )
        scores.append(1.0 - max(0.0, cos_sim))
    return scores


def token_prob_delta_batched(model, tokenizer, prompts, questions, golds,
                             batch_size, device):
    """Batched token_prob_delta over aligned lists."""
    no_ctx_prompts = [
        f"Answer the following question.\n\nQuestion: {q}\nAnswer:"
        for q in questions
    ]
    lp_ctx = _batch_log_prob(model, tokenizer, prompts, golds, batch_size,
                             device)
    lp_no_ctx = _batch_log_prob(model, tokenizer, no_ctx_prompts, golds,
                                batch_size, device)
    scores = []
    for a, b in zip(lp_ctx, lp_no_ctx):
        scores.append(float(1 / (1 + np.exp(-(a - b)))))
    return scores


def verbalized_comparison_batched(model, tokenizer, prompts, questions,
                                  batch_size, device):
    """Batched verbalized_comparison over aligned lists (two batched stages)."""
    compare_prompts = [
        (f"{p}\n\n"
         f"Now answer the same question WITHOUT using the context above, "
         f"from memory only:\nMemory answer:")
        for p in prompts
    ]
    ans_memory = _batch_generate(model, tokenizer, compare_prompts, 60,
                                 batch_size, device)
    rate_prompts = [
        (f"{cp}\n{am}\n\n"
         f"On a scale of 0 to 100, how much did the context change your answer? "
         f"(0=no change, 100=completely different). Reply with ONLY a number."
         f"\nScore:")
        for cp, am in zip(compare_prompts, ans_memory)
    ]
    score_strs = _batch_generate(model, tokenizer, rate_prompts, 5, batch_size,
                                 device)
    scores = []
    for ss in score_strs:
        nums = re.findall(r'\d+', ss)
        if nums:
            scores.append(min(max(float(nums[0]) / 100, 0.0), 1.0))
        else:
            scores.append(0.5)
    return scores


def run_contrastive_baselines(model_key: str, device: str = "cuda:0",
                              batch_size: int = 16, limit: int = None):
    from sentence_transformers import SentenceTransformer

    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"CONTRASTIVE BASELINES — {model_key}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding => batched continuations all start at input_ids.shape[1]
    # and real tokens end at index -1 (needed for the forward-pass slicing).
    tokenizer.padding_side = "left"

    # fp16 (NOT 8-bit): matches baselines.py / llm_judge_baseline.py and makes
    # batched inference numerically faithful. 8-bit matmul is batch-size
    # sensitive, so left-padding flipped greedy tokens and drifted the forward
    # pass (validation showed 0.67 / 0.10 diffs) — unacceptable for reproduction.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)
    if limit is not None:
        test_data = test_data[:limit]

    # Flatten to per-condition lists (exact same order as the original loop, so
    # labels / metadata / scores stay aligned).
    prompts, questions, golds = [], [], []
    labels, meta = [], []
    for ex in test_data:
        for cond in ex["conditions"]:
            labels.append(int(cond["sufficient"]))
            prompts.append(cond["prompt"])
            questions.append(ex["question"])
            golds.append(ex["gold_answer"])
            meta.append({
                "id": ex["id"],
                "condition_id": cond["condition_id"],
                "quadrant": cond["quadrant"],
                "question_type": ex["question_type"],
            })

    n_cond = len(prompts)
    print(f"  {n_cond} conditions | batch_size={batch_size}")

    # Batched inference, method by method (each batch is homogeneous).
    print("  [1/3] Response Delta ...")
    rd_scores = response_delta_batched(model, tokenizer, prompts, questions,
                                       embed_model, batch_size, device)
    print("  [2/3] Token Prob Delta ...")
    tpd_scores = token_prob_delta_batched(model, tokenizer, prompts, questions,
                                          golds, batch_size, device)
    print("  [3/3] Verbalized Comparison ...")
    vc_scores = verbalized_comparison_batched(model, tokenizer, prompts,
                                              questions, batch_size, device)

    labels_arr = np.array(labels)
    results = {}
    for name, scores in [
        ("Response Delta", rd_scores),
        ("Token Prob Delta", tpd_scores),
        ("Verbalized Comparison", vc_scores),
    ]:
        # Sanitize: replace any non-finite score with the finite mean so a few
        # bad conditions can't NaN-crash roc_auc and lose the whole run.
        arr = np.asarray(scores, dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            fill = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
            arr = np.where(np.isfinite(arr), arr, fill)
            print(f"  [warn] {name}: imputed {n_bad} non-finite scores with {fill:.4f}")
        scores = arr.tolist()
        entry = {
            "n_imputed": n_bad,
            "scores": scores,
            "labels": labels,
            "metadata": meta,
        }
        try:
            entry["auroc"] = float(roc_auc_score(labels_arr, arr))
            print(f"  {name}: AUROC={entry['auroc']:.4f}")
        except Exception as e:  # never let one method sink the others
            entry["error"] = str(e)
            print(f"  [error] {name}: {e}")
        results[name] = entry

    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "contrastive_baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also store scores for bootstrap
    scores_dir = RESULTS_DIR / model_key / "scores"
    scores_dir.mkdir(exist_ok=True)
    contrastive_scores = {
        name: {"scores": results[name]["scores"],
               "labels": results[name]["labels"],
               "metadata": meta,
               "auroc": results[name].get("auroc")}
        for name in results
    }
    with open(scores_dir / "contrastive_scores.json", "w") as f:
        json.dump(contrastive_scores, f, indent=2)

    print(f"\nSaved to {out_dir / 'contrastive_baselines.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama",
                        choices=["llama", "mistral", "qwen"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Conditions per inference batch.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to the first N test examples (debug).")
    args = parser.parse_args()
    run_contrastive_baselines(args.model, args.device,
                              batch_size=args.batch_size, limit=args.limit)
