"""
build_longcontext.py — construct the long-context test sets (manifest only).

For N=300 test questions (stratified by question_type, seed 0) we bury the
question's target passage -- the sufficient one in one condition, the
insufficient one in the other -- inside a pile of distractor passages drawn from
the *train* split, at 4k / 16k / 32k / 64k / 100k tokens (llama tokenizer).

Design points that the analysis depends on:

  * The sufficient and insufficient condition of a question share the EXACT same
    ordered distractor list at a given length.  The only thing that differs
    between the two conditions is the target passage, so any AUROC drop is
    attributable to the distractors, not to a different haystack.
  * Distractor lists are NESTED across lengths (the 16k list extends the 4k
    list, etc.), so the length axis is a clean "add more noise" manipulation.
  * The target position (first / middle / last) is fixed per question and shared
    by both conditions and all lengths, and is balanced 100/100/100 overall.
  * Distractors that leak the answer are removed: same question id, gold-answer
    substring, or >=50% sentence overlap with either of the target contexts.

Nothing but ids is written to disk: longcontext/common.assemble_context()
rebuilds any context string deterministically from train/test json.

Usage:
  python src/longcontext/build_longcontext.py
  python src/longcontext/build_longcontext.py --n 300 --seed 0 --out <path>
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longcontext.common import (  # noqa: E402
    BUILD_SEED, BUILD_TOKENIZER, LENGTHS, LENGTH_TOLERANCE, MANIFEST_PATH,
    N_QUESTIONS, PASSAGE_SEP, POSITIONS, POSITION_LABELS,
    format_with_context, load_split, load_tokenizer,
)

#: sentences shorter than this are ignored when measuring passage overlap
MIN_SENT_CHARS = 20
#: a distractor sharing this fraction of sentences with a target is dropped
OVERLAP_THRESHOLD = 0.5
#: gold answers shorter than this are too generic for a substring leak test
MIN_GOLD_CHARS = 3
#: token cost charged for each PASSAGE_SEP when filling the budget
SEP_TOKENS = 1

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(test_data, n, seed):
    """n questions, question_type proportions matching the test split.

    Largest-remainder apportionment so the per-type counts sum to exactly n.
    """
    by_type = defaultdict(list)
    for ex in test_data:
        by_type[ex["question_type"]].append(ex)
    total = len(test_data)

    types = sorted(by_type)
    exact = {t: n * len(by_type[t]) / total for t in types}
    quota = {t: int(np.floor(exact[t])) for t in types}
    remainder = n - sum(quota.values())
    for t in sorted(types, key=lambda t: (-(exact[t] - quota[t]), t))[:remainder]:
        quota[t] += 1

    rng = np.random.default_rng(seed)
    sampled = []
    for t in types:
        pool = by_type[t]
        k = min(quota[t], len(pool))
        pick = rng.choice(len(pool), size=k, replace=False)
        sampled.extend(pool[i] for i in sorted(pick))

    # stable, reproducible order for the manifest
    sampled.sort(key=lambda ex: ex["id"])
    return sampled, quota


def assign_positions(sampled, seed):
    """Balanced first/middle/last assignment, one position per question."""
    rng = np.random.default_rng(seed + 1)
    n = len(sampled)
    pos = np.array([POSITIONS[i % len(POSITIONS)] for i in range(n)])
    rng.shuffle(pos)
    return {ex["id"]: float(p) for ex, p in zip(sampled, pos)}


# ---------------------------------------------------------------------------
# Distractor eligibility
# ---------------------------------------------------------------------------

def sentence_set(text):
    out = set()
    for s in _SENT_SPLIT.split(text):
        s = _WS.sub(" ", s).strip().lower()
        if len(s) >= MIN_SENT_CHARS:
            out.add(s)
    return out


def overlap_fraction(a, b):
    """|a & b| / min(|a|, |b|) -- symmetric, so a distractor that *contains* the
    target passage is caught as well as one contained by it."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def eligible_distractors(target, pool, pool_sents):
    """Indices of pool entries usable as distractors for one target question."""
    gold = (target["gold_answer"] or "").strip().lower()
    gold_ok = len(gold) >= MIN_GOLD_CHARS
    tgt_sents = [sentence_set(c["context"]) for c in target["conditions"]]

    keep = []
    for i, cand in enumerate(pool):
        if cand["question_id"] == target["id"]:
            continue
        if gold_ok and gold in cand["context_lower"]:
            continue
        if any(overlap_fraction(pool_sents[i], ts) >= OVERLAP_THRESHOLD
               for ts in tgt_sents):
            continue
        keep.append(i)
    return keep


# ---------------------------------------------------------------------------
# Budget filling
# ---------------------------------------------------------------------------

def fill_to_budget(selected, available, pool_tokens, base_tokens, budget):
    """Greedily extend ``selected`` (in permutation order) until no remaining
    passage fits under ``budget`` tokens.

    ``available`` is consumed in place, so calling this for increasing budgets
    yields NESTED distractor lists: each length's list is the previous one plus
    whatever else fits.  Passages skipped at a small budget stay in the pool and
    can be picked up at a larger one, which keeps the realised length within a
    few tokens of the nominal target instead of up to one passage short.
    """
    used = base_tokens + sum(pool_tokens[i] for i in selected) \
        + SEP_TOKENS * len(selected)
    progress = True
    while progress:
        progress = False
        for pos, i in enumerate(available):
            cost = pool_tokens[i] + SEP_TOKENS
            if used + cost <= budget:
                selected.append(i)
                available.pop(pos)
                used += cost
                progress = True
                break
    return used


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(n=N_QUESTIONS, seed=BUILD_SEED, out_path=MANIFEST_PATH,
          lengths=LENGTHS, exact_tokens=True):
    test_data = load_split("test")
    train_data = load_split("train")
    tok = load_tokenizer(BUILD_TOKENIZER)

    def ntok(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    # ---- distractor pool = sufficient-condition contexts of TRAIN questions --
    pool = []
    for ex in train_data:
        for cond in ex["conditions"]:
            if cond["sufficient"]:
                pool.append({
                    "condition_id": cond["condition_id"],
                    "question_id": ex["id"],
                    "context": cond["context"],
                    "context_lower": cond["context"].lower(),
                })
    print(f"[build] distractor pool: {len(pool)} train sufficient contexts")

    pool_tokens = np.array([ntok(p["context"]) for p in tqdm(pool, desc="tok(pool)")])
    pool_sents = [sentence_set(p["context"]) for p in pool]
    print(f"[build] pool tokens: total={pool_tokens.sum()} "
          f"mean={pool_tokens.mean():.1f} max={pool_tokens.max()}")

    # ---- sample targets ----------------------------------------------------
    sampled, quota = stratified_sample(test_data, n, seed)
    positions = assign_positions(sampled, seed)
    print(f"[build] sampled {len(sampled)} questions, quota={dict(quota)}")
    print(f"[build] positions: {Counter(positions.values())}")

    numeric_lengths = [int(x) for x in lengths if x != "base"]
    numeric_lengths.sort()

    entries = []
    warnings = []
    for qi, ex in enumerate(tqdm(sampled, desc="build")):
        conds = {("sufficient" if c["sufficient"] else "insufficient"): c
                 for c in ex["conditions"]}
        if len(conds) != 2:
            warnings.append(f"{ex['id']}: expected 2 conditions, got {len(conds)}")
            continue

        cond_ids = {k: c["condition_id"] for k, c in conds.items()}
        tgt_tokens = {k: ntok(c["context"]) for k, c in conds.items()}
        position = positions[ex["id"]]

        # per-question deterministic permutation of the eligible pool
        keep = eligible_distractors(ex, pool, pool_sents)
        rng = np.random.default_rng([seed, qi])
        perm = [keep[i] for i in rng.permutation(len(keep))]

        # ---- base: the original short context, no distractors --------------
        base_entry = _make_entry(ex, cond_ids, position, [], 0, "base",
                                 pool, tgt_tokens, tok, exact_tokens)
        entries.append(base_entry)

        # ---- nested fills at increasing budgets ----------------------------
        selected, available = [], list(perm)
        for L in numeric_lengths:
            est = fill_to_budget(selected, available, pool_tokens,
                                 tgt_tokens["sufficient"], L)
            k = len(selected)
            insert_index = int(round(position * k))
            entry = _make_entry(
                ex, cond_ids, position, [pool[i]["condition_id"] for i in selected],
                insert_index, str(L), pool, tgt_tokens, tok, exact_tokens,
                estimated_tokens=est,
            )
            entries.append(entry)

            if est < L * (1 - LENGTH_TOLERANCE):
                warnings.append(
                    f"{ex['id']}@{L}: only {est} ctx tokens "
                    f"({len(available)} distractors left) -- pool exhausted")

    manifest = {
        "meta": {
            "seed": seed,
            "n_questions": len(sampled),
            "lengths": list(lengths),
            "positions": POSITIONS,
            "position_labels": {str(k): v for k, v in POSITION_LABELS.items()},
            "tokenizer": BUILD_TOKENIZER,
            "passage_sep": PASSAGE_SEP,
            "distractor_pool_size": len(pool),
            "question_type_quota": dict(quota),
            "exclusion_rules": {
                "same_question_id": True,
                "gold_answer_substring": True,
                "sentence_overlap_threshold": OVERLAP_THRESHOLD,
                "min_sentence_chars": MIN_SENT_CHARS,
            },
            "n_warnings": len(warnings),
        },
        "entries": entries,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f)
    print(f"[build] wrote {len(entries)} entries -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    if warnings:
        print(f"[build] {len(warnings)} warnings; first 5:")
        for w in warnings[:5]:
            print(f"  {w}")

    _report(manifest)
    return manifest


def _make_entry(ex, cond_ids, position, distractor_ids, insert_index, length,
                pool, tgt_tokens, tok, exact_tokens, estimated_tokens=None):
    entry = {
        "question_id": ex["id"],
        "question": ex["question"],
        "question_type": ex["question_type"],
        "gold_answer": ex.get("gold_answer", ""),
        "length": length,
        "position": position,
        "position_label": POSITION_LABELS[position],
        "condition_ids": cond_ids,
        "distractors": list(distractor_ids),
        "insert_index": insert_index,
        "n_distractors": len(distractor_ids),
    }
    if estimated_tokens is not None:
        entry["estimated_context_tokens"] = int(estimated_tokens)

    if not exact_tokens:
        entry["n_context_tokens"] = {k: int(v) for k, v in tgt_tokens.items()}
        entry["n_prompt_tokens"] = {k: int(v) for k, v in tgt_tokens.items()}
        return entry

    # Exact llama-tokenizer count of the assembled PROMPT -- that is the number
    # the position-embedding guard in extraction cares about.  Re-tokenising the
    # bare context as well would double the cost on 100k-token strings, so the
    # context count is derived by subtracting the template overhead measured
    # exactly on this question's "base" entry (stable to +-2 tokens).
    from longcontext.common import assemble_context
    n_prompt = {}
    for cond in ("sufficient", "insufficient"):
        prompt = format_with_context(assemble_context(entry, cond), ex["question"])
        n_prompt[cond] = len(tok(prompt)["input_ids"])
    entry["n_prompt_tokens"] = n_prompt

    if length == "base":
        entry["n_context_tokens"] = {k: int(v) for k, v in tgt_tokens.items()}
        for cond in n_prompt:
            _TEMPLATE_OVERHEAD[(ex["id"], cond)] = \
                n_prompt[cond] - int(tgt_tokens[cond])
    else:
        entry["n_context_tokens"] = {
            cond: n_prompt[cond] - _TEMPLATE_OVERHEAD.get((ex["id"], cond), 0)
            for cond in n_prompt
        }
    return entry


#: (question_id, condition) -> prompt_tokens - context_tokens, measured once on
#: the "base" entry and reused for the long entries (stable to +-2 tokens).
_TEMPLATE_OVERHEAD = {}


def _report(manifest):
    """Print realised length / position / type distributions."""
    by_len = defaultdict(list)
    for e in manifest["entries"]:
        by_len[e["length"]].append(e)
    print("\n[build] realised prompt lengths (llama tokenizer)")
    print(f"  {'length':>8s} {'n':>5s} {'mean':>9s} {'min':>8s} {'max':>8s} {'off%':>7s}")
    for ln in manifest["meta"]["lengths"]:
        es = by_len.get(ln, [])
        if not es:
            continue
        toks = np.array([e["n_prompt_tokens"]["sufficient"] for e in es])
        if ln == "base":
            off = float("nan")
        else:
            off = 100 * (toks.mean() - int(ln)) / int(ln)
        print(f"  {ln:>8s} {len(es):>5d} {toks.mean():>9.0f} {toks.min():>8d} "
              f"{toks.max():>8d} {off:>7.2f}")
    qtypes = Counter(e["question_type"] for e in by_len["base"])
    poss = Counter(e["position_label"] for e in by_len["base"])
    print(f"  question types: {dict(qtypes)}")
    print(f"  positions:      {dict(poss)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_QUESTIONS)
    ap.add_argument("--seed", type=int, default=BUILD_SEED)
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    ap.add_argument("--lengths", nargs="+", default=LENGTHS)
    ap.add_argument("--no-exact-tokens", action="store_true",
                    help="skip the exact re-tokenisation of every assembled "
                         "prompt (fast, but token counts are estimates)")
    args = ap.parse_args()
    build(args.n, args.seed, Path(args.out), args.lengths,
          exact_tokens=not args.no_exact_tokens)


if __name__ == "__main__":
    main()
