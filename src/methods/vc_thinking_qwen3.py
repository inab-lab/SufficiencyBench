"""
vc_thinking_qwen3.py

Tests whether Qwen3-8B's "thinking mode" lets verbalized confidence (VC)
escape the output-based ceiling documented in baselines.py's
verbalized_confidence_batch. Four variants of the SAME sufficiency-rating
question, run on the TEST split of SufficiencyBench:

  1. raw          - baselines.py's exact VC prompt (base answer prompt +
                    the 0-100 rating instruction), as raw text, with NO
                    chat template. This is parity with the other
                    (non-Qwen3) backbones in baselines.py.
  2. chat-nothink  - the same instruction as (1), wrapped as a user turn in
                    the Qwen3 chat template with enable_thinking=False.
  3. chat-think    - the same instruction as (1)/(2), chat template with
                    enable_thinking=True. Generation budget is
                    max_new_tokens=768. Greedy by default; --sampling
                    recommended switches to Qwen3's documented "thinking
                    mode" sampling (temperature=0.6, top_p=0.95, top_k=20).
                    The score is parsed from the text AFTER the closing
                    </think> tag; if no </think> token is produced within
                    budget the condition is marked `truncated` and the
                    score falls back to 0.5. Thinking length (in tokens,
                    counted exactly via Qwen3's dedicated <think>/</think>
                    token ids -- see THINK_START_ID/THINK_END_ID) is
                    recorded for every condition.
  4. cot-nothink  - chat template, enable_thinking=False (i.e. the model's
                    OWN hidden thinking channel is off), but the user
                    instruction itself asks for visible step-by-step
                    reasoning before a final numeric line (mirrors the
                    structure of api_cot_test.py's CoT judge prompt, which
                    asks for reasoning then a one-line verdict). The score
                    is parsed as the LAST integer in the response.

Batching: batch size 1 is used for chat-think regardless of --batch-size,
because thinking-chain lengths vary hugely per example and left-padding to
the longest chain in a batch would waste most of the compute budget on pad
columns for every other row. The other three variants are short, bounded
generations and are batched normally (left-padding, same fidelity argument
as baselines.py: RoPE attention is shift-invariant and pad columns are
masked out).

Resume: each variant writes a per-condition JSONL cache under
  results/qwen3/vc_thinking/<variant>.jsonl
A rerun loads this cache and skips any condition_id already present, so an
interrupted run (e.g. the 15+ hour chat-think pass) can be restarted safely.

GPU time estimate: the dominant cost is chat-think. At ~30 tok/s and a
768-token budget, a condition that uses the full budget costs ~25.6s;
across 2,224 conditions (1,112 questions x 2 conditions each) that is up to
~15.8 GPU-hours if every condition maxes out the budget (in practice many
finish thinking well before 768 tokens, so real wall-clock will be lower).
The other three variants are batched with short budgets (<=300 new tokens)
and are cheap by comparison.

Usage:
  conda run -n csp_qwen3 python src/methods/vc_thinking_qwen3.py --device cuda:0
  conda run -n csp_qwen3 python src/methods/vc_thinking_qwen3.py \\
      --variants chat-think --sampling recommended --device cuda:0
  # CPU-only sanity check (no model weights loaded):
  conda run -n csp_qwen3 python src/methods/vc_thinking_qwen3.py \\
      --dry-run --device cpu --limit 2
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS

# Qwen3's chat template registers <think> / </think> as dedicated single
# tokens (verified against data/models/qwen3-8b/tokenizer.json: id 151667
# and 151668 respectively, both above the base vocab_size of 151643), so the
# thinking/answer boundary can be located exactly in the generated id
# sequence -- no decoding/offset-mapping heuristics needed.
THINK_START_ID = 151667
THINK_END_ID = 151668

VARIANTS = ["raw", "chat-nothink", "chat-think", "cot-nothink"]

# Per-variant generation budgets (new tokens). raw matches baselines.py's
# verbalized_confidence_batch exactly (max_new_tokens=5). chat-nothink gets
# a small buffer since chat-formatted answers occasionally add a token of
# punctuation/filler before the number. chat-think is the 768-token spec
# budget. cot-nothink mirrors api_cot_test.py's CoT judge (max_tokens=300).
MAX_NEW_TOKENS = {
    "raw": 5,
    "chat-nothink": 8,
    "chat-think": 768,
    "cot-nothink": 512,
}

QUESTION_TYPES = ["factual", "multi_hop", "subjective", "comparative"]

# Identical wording to baselines.py's verbalized_confidence_batch meta-prompt.
VC_SUFFIX = (
    "\n\nBefore answering, rate from 0 to 100: "
    "does the context contain enough information to answer? "
    "Reply with ONLY the number.\nScore:"
)

# Mirrors api_cot_test.py's COT_PROMPT structure (reason step by step, then
# emit a single-line verdict) but adapted to the 0-100 numeric VC score
# instead of a SUFFICIENT/INSUFFICIENT label.
COT_SUFFIX = (
    "\n\nThink step by step about whether the context contains enough "
    "information to answer. Keep the reasoning under 100 words. Then end your "
    "reply with a final line of exactly this form:\nSCORE: <number from 0 to 100>"
)


def _chunks(seq, n):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conditions(limit=None):
    """Flatten (sufficient, insufficient) conditions from the test split,
    carrying the fields needed to build every prompt variant and to score
    results (question id for paired bootstrap, question_type, gold label)."""
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)
    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                "question_id": ex["id"],
                "question_type": ex.get("question_type"),
                "sufficient": bool(cond["sufficient"]),
                "condition_id": cond["condition_id"],
                "base_prompt": cond["prompt"],
            })
    if limit is not None:
        conditions = conditions[:limit]
    return conditions


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_text(variant, tokenizer, base_prompt):
    """Return (text, add_special_tokens) for one condition under `variant`.

    add_special_tokens is False for chat-templated variants (the template
    already renders every special token as literal text) and True for
    `raw` (plain text handed straight to the tokenizer, exactly like
    baselines.py)."""
    if variant == "raw":
        return base_prompt + VC_SUFFIX, True

    if variant == "chat-nothink":
        user_msg = base_prompt + VC_SUFFIX
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return text, False

    if variant == "chat-think":
        user_msg = base_prompt + VC_SUFFIX
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        return text, False

    if variant == "cot-nothink":
        user_msg = base_prompt + COT_SUFFIX
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return text, False

    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_first_score(text):
    """First integer in `text`, scaled to [0, 1]. Fallback 0.5 (matches
    baselines.py's verbalized_confidence_batch exactly)."""
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            return min(max(float(numbers[0]) / 100, 0.0), 1.0)
        except ValueError:
            return 0.5
    return 0.5


def parse_last_score(text):
    """Score for cot-nothink. Prefers the anchored ``SCORE: n`` line, because a
    free-form "last integer" parse picks up years and other digits from the
    reasoning. Falls back to the last integer, then to 0.5."""
    anchored = re.findall(r'SCORE\s*:\s*(\d+)', text, flags=re.IGNORECASE)
    if anchored:
        try:
            return min(max(float(anchored[-1]) / 100, 0.0), 1.0)
        except ValueError:
            return 0.5
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            return min(max(float(numbers[-1]) / 100, 0.0), 1.0)
        except ValueError:
            return 0.5
    return 0.5


def parse_think_output(tokenizer, gen_ids):
    """Split a chat-think generation into (thinking, answer) using the exact
    </think> token id, and parse the score from the answer half only.

    gen_ids: python list[int] of newly generated token ids for ONE example
    (already stripped of any trailing pad tokens).

    Returns a dict: score, truncated, thinking_tokens, answer_text.
    If THINK_END_ID never appears within the budget, the whole budget was
    spent thinking: truncated=True, score falls back to 0.5, and
    thinking_tokens = len(gen_ids).
    """
    if THINK_END_ID in gen_ids:
        end_idx = gen_ids.index(THINK_END_ID)
        start_idx = 1 if gen_ids and gen_ids[0] == THINK_START_ID else 0
        thinking_tokens = max(end_idx - start_idx, 0)
        answer_ids = gen_ids[end_idx + 1:]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return {
            "score": parse_first_score(answer_text),
            "truncated": False,
            "thinking_tokens": thinking_tokens,
            "answer_text": answer_text,
        }
    return {
        "score": 0.5,
        "truncated": True,
        "thinking_tokens": len(gen_ids),
        "answer_text": "",
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, texts, device, max_new_tokens,
                    add_special_tokens, sampling):
    """Run one batched (or size-1) generate() call. Left-padded, like every
    other batched method in this repo (see baselines.py's module docstring
    for the RoPE shift-invariance argument)."""
    import torch  # deferred: keeps --dry-run usable without CUDA/torch churn

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=4096, add_special_tokens=add_special_tokens,
    ).to(device)

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if sampling == "recommended":
        gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    else:
        gen_kwargs.update(do_sample=False, temperature=0.0)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Left padding => every row's continuation starts at the same column.
    return out[:, inputs["input_ids"].shape[1]:]


# ---------------------------------------------------------------------------
# Resume cache
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict:
    """Load a per-variant JSONL cache into {condition_id: record}."""
    done = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["condition_id"]] = rec
    return done


def run_variant(variant, model, tokenizer, conditions, device, batch_size,
                 sampling, cache_dir: Path):
    """Score every condition under `variant`, resuming from the JSONL cache
    and appending newly computed records to it as they are produced."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{variant}.jsonl"
    done = load_cache(cache_path)
    todo = [c for c in conditions if c["condition_id"] not in done]
    print(f"[{variant}] {len(done)} cached, {len(todo)} to run "
          f"(cache: {cache_path})")

    max_new = MAX_NEW_TOKENS[variant]
    eff_batch_size = 1 if variant == "chat-think" else max(1, batch_size)
    eff_sampling = sampling if variant == "chat-think" else "greedy"

    if todo:
        pad_id = tokenizer.pad_token_id
        with open(cache_path, "a") as f:
            batches = list(_chunks(todo, eff_batch_size))
            for batch in tqdm(batches, desc=f"vc_thinking[{variant}]"):
                built = [build_text(variant, tokenizer, c["base_prompt"]) for c in batch]
                texts = [t for t, _ in built]
                add_special = built[0][1]
                gen = generate_batch(model, tokenizer, texts, device, max_new,
                                      add_special, eff_sampling)

                for i, c in enumerate(batch):
                    row = gen[i]
                    if variant == "chat-think":
                        ids_list = [t for t in row.tolist() if t != pad_id]
                        parsed = parse_think_output(tokenizer, ids_list)
                        rec = {
                            "condition_id": c["condition_id"],
                            "question_id": c["question_id"],
                            "question_type": c["question_type"],
                            "sufficient": c["sufficient"],
                            "score": parsed["score"],
                            "truncated": parsed["truncated"],
                            "thinking_tokens": parsed["thinking_tokens"],
                            "answer_text": parsed["answer_text"][:300],
                        }
                    else:
                        text = tokenizer.decode(row, skip_special_tokens=True).strip()
                        score = (parse_last_score(text) if variant == "cot-nothink"
                                 else parse_first_score(text))
                        rec = {
                            "condition_id": c["condition_id"],
                            "question_id": c["question_id"],
                            "question_type": c["question_type"],
                            "sufficient": c["sufficient"],
                            "score": score,
                            "truncated": False,
                            "thinking_tokens": 0,
                            "answer_text": text[:300],
                        }
                    f.write(json.dumps(rec) + "\n")
                    done[c["condition_id"]] = rec
                f.flush()

    return [done[c["condition_id"]] for c in conditions if c["condition_id"] in done]


# ---------------------------------------------------------------------------
# Scoring / summary
# ---------------------------------------------------------------------------

def bootstrap_auroc_by_question(y_true, y_score, question_ids, n_boot=1000, seed=42):
    """95% bootstrap CI for AUROC, resampling question ids with replacement
    (both conditions of a resampled question travel together), matching
    answerability_baselines.py's paired-structure bootstrap."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    q_to_idx = {}
    for i, q in enumerate(question_ids):
        q_to_idx.setdefault(q, []).append(i)
    unique_qs = list(q_to_idx.keys())
    n_q = len(unique_qs)

    rng = np.random.RandomState(seed)
    aurocs = []
    for _ in range(n_boot):
        sampled = rng.randint(0, n_q, n_q)
        idx = []
        for j in sampled:
            idx.extend(q_to_idx[unique_qs[j]])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, ys))

    point = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    if aurocs:
        lo, hi = (float(x) for x in np.percentile(aurocs, [2.5, 97.5]))
    else:
        lo, hi = float("nan"), float("nan")
    return point, lo, hi


def summarize_variant(name, records, n_boot=1000, seed=42):
    y_true = np.array([1 if r["sufficient"] else 0 for r in records])
    y_score = np.array([r["score"] for r in records])
    qids = [r["question_id"] for r in records]

    point, lo, hi = bootstrap_auroc_by_question(y_true, y_score, qids, n_boot, seed)
    trunc = [bool(r.get("truncated")) for r in records]
    think_tok = [r.get("thinking_tokens", 0) for r in records]

    result = {
        "variant": name,
        "n": len(records),
        "auroc": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_half": (hi - lo) / 2 if not (math.isnan(hi) or math.isnan(lo)) else None,
        "truncation_rate": float(np.mean(trunc)) if trunc else 0.0,
        "mean_thinking_tokens": float(np.mean(think_tok)) if think_tok else 0.0,
        "per_question_type": {},
    }
    for qt in QUESTION_TYPES:
        idx = [i for i, r in enumerate(records) if r["question_type"] == qt]
        if len(idx) < 10 or len(set(y_true[idx])) < 2:
            continue
        result["per_question_type"][qt] = {
            "auroc": float(roc_auc_score(y_true[idx], y_score[idx])),
            "n": len(idx),
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Verbalized-confidence thinking-mode variants (Qwen3-8B) "
                     "on SufficiencyBench TEST")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
                     help="Which of the four VC variants to run.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only score the first N conditions (debug/CPU test).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16,
                     help="Batch size for raw/chat-nothink/cot-nothink "
                          "(chat-think always uses batch size 1).")
    ap.add_argument("--sampling", default="greedy", choices=["greedy", "recommended"],
                     help="Generation strategy for chat-think ONLY: 'greedy' "
                          "(default, comparable to the other variants) or "
                          "'recommended' (Qwen3 thinking-mode sampling: "
                          "temperature=0.6, top_p=0.95, top_k=20). The other "
                          "three variants are always greedy.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                     help="Build and print one example prompt per requested "
                          "variant (chat-template rendering check) and exit "
                          "WITHOUT loading model weights. For CPU testing.")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    cfg = MODEL_CONFIGS["qwen3"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = load_conditions(args.limit)
    n_q = len(set(c["question_id"] for c in conditions))
    print(f"Loaded {len(conditions)} conditions ({n_q} questions) "
          f"from {BENCH_DIR / 'test.json'}")

    if args.dry_run:
        for variant in args.variants:
            text, add_special = build_text(variant, tokenizer, conditions[0]["base_prompt"])
            print(f"\n{'=' * 70}\n{variant}  (add_special_tokens={add_special}, "
                  f"max_new_tokens={MAX_NEW_TOKENS[variant]})\n{'=' * 70}")
            print(text)
        return

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    cache_dir = RESULTS_DIR / "qwen3" / "vc_thinking"
    summary = {}
    for variant in args.variants:
        t0 = time.time()
        records = run_variant(variant, model, tokenizer, conditions, args.device,
                               args.batch_size, args.sampling, cache_dir)
        summary[variant] = summarize_variant(variant, records, args.n_boot, args.seed)
        elapsed = time.time() - t0
        r = summary[variant]
        print(f"[{variant}] AUROC={r['auroc']:.4f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  n={r['n']}  "
              f"trunc_rate={r['truncation_rate']:.3f}  "
              f"mean_think_tok={r['mean_thinking_tokens']:.1f}  "
              f"({elapsed:.1f}s)")

    out_path = RESULTS_DIR / "qwen3" / "vc_thinking_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
