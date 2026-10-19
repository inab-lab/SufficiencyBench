"""
answerability_baselines.py

Cheap, LLM-free "standalone" baselines for context-sufficiency detection on
SufficiencyBench, for comparison against the paper's 395M SufficiencyProbe
(0.754 AUROC) and verbalized confidence (0.67-0.73):

  (a) SQuAD2 answerability -- deepset/deberta-v3-large-squad2 extractive QA.
      Score = best non-null context span logit minus the null/[CLS] logit
      (the model's own answerability signal). Contexts longer than the
      model's max length are handled with the standard SQuAD2 sliding
      window (doc_stride=128), taking the max answerability over windows.
      Higher score => more answerable => predicted sufficient.

  (b) MiniCheck entailment (ORACLE) -- lytang/MiniCheck-Flan-T5-Large scores
      whether the context entails the claim "The answer to the question
      '{question}' is {gold_answer}." This uses the gold answer, so it is an
      ORACLE-style baseline (like "Generation Match" in the paper), not a
      fair deployable method -- it is reported and labeled as such.

  (c) Length-only logistic regression (sanity check) -- 5-fold CV logistic
      regression on [context_length_chars, n_capitalized_tokens, word_count].
      The paper reports ~0.508 for this; it should stay near chance.

All AUROCs are computed on the full TEST split (2,224 conditions across 1,112
questions) with 95% bootstrap CIs (1000 resamples BY QUESTION -- both
conditions of a resampled question are kept together, matching
scripts/bootstrap_ci.py's general approach but adapted to this pair
structure), plus a per-question_type breakdown.

Usage:
  python src/methods/answerability_baselines.py --device cuda:0 --batch-size 16
  python src/methods/answerability_baselines.py --device cuda:0 --batch-size 16 --limit 100
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import re
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

QUESTION_TYPES = ["factual", "multi_hop", "subjective", "comparative"]
SQUAD2_MODEL_NAME = "deepset/deberta-v3-large-squad2"
MINICHECK_MODEL_NAME = "flan-t5-large"  # -> lytang/MiniCheck-Flan-T5-Large


def _chunks(seq, n):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_conditions(limit=None):
    """Flatten (sufficient, insufficient) conditions from the test split,
    carrying question/gold/type so each row is self-contained."""
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)
    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                "question_id": ex["id"],
                "question": ex["question"],
                "gold_answer": ex.get("gold_answer", ""),
                "question_type": ex.get("question_type"),
                "context": cond["context"],
                "sufficient": bool(cond["sufficient"]),
                "condition_id": cond["condition_id"],
            })
    if limit is not None:
        conditions = conditions[:limit]
    return conditions


# ---------------------------------------------------------------------------
# (a) SQuAD2 answerability
# ---------------------------------------------------------------------------

def load_squad2_model(device):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    # HF_HUB_OFFLINE must be unset for this first call so the model can be
    # downloaded; it is then cached under ~/.cache/huggingface/hub.
    tokenizer = AutoTokenizer.from_pretrained(SQUAD2_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(SQUAD2_MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


def squad2_answerability_batch(tokenizer, model, questions, contexts, device,
                                max_length=512, doc_stride=128, max_answer_length=30):
    """Answerability score per (question, context) pair.

    For each sliding-window chunk: answerability = (best non-null span logit,
    restricted to context tokens) - (null/[CLS] logit). We take the max over
    all chunks belonging to the same (question, context) pair, per the
    standard SQuAD2 sliding-window protocol. Higher => more answerable.
    """
    import torch

    enc = tokenizer(
        questions, contexts,
        max_length=max_length, truncation="only_second", stride=doc_stride,
        return_overflowing_tokens=True, padding="max_length", return_tensors="pt",
    )
    sample_map = enc["overflow_to_sample_mapping"].tolist()
    model_inputs = {k: v.to(device) for k, v in enc.items()
                     if k in ("input_ids", "attention_mask", "token_type_ids")}

    with torch.no_grad():
        out = model(**model_inputs)
    start_logits = out.start_logits.float().cpu().numpy()
    end_logits = out.end_logits.float().cpu().numpy()

    cls_id = tokenizer.cls_token_id
    n = len(questions)
    best_per_q = np.full(n, -np.inf)
    n_windows = start_logits.shape[0]

    for w in range(n_windows):
        qi = sample_map[w]
        seq_ids = enc.sequence_ids(w)  # None=special, 0=question, 1=context
        context_mask = np.array([sid == 1 for sid in seq_ids])

        s = start_logits[w]
        e = end_logits[w]

        # Null-answer position: the [CLS]/BOS token (standard SQuAD2
        # convention). Look it up robustly rather than assuming index 0.
        window_ids = enc["input_ids"][w]
        cls_matches = (window_ids == cls_id).nonzero(as_tuple=True)[0]
        cls_idx = int(cls_matches[0]) if len(cls_matches) else 0
        null_score = float(s[cls_idx] + e[cls_idx])

        if not context_mask.any():
            answerability = 0.0  # no context tokens survived this window
        else:
            s_masked = np.where(context_mask, s, -np.inf)
            e_masked = np.where(context_mask, e, -np.inf)
            K = max_answer_length
            padded_e = np.concatenate([e_masked, np.full(K - 1, -np.inf)])
            end_max = sliding_window_view(padded_e, K).max(axis=1)
            cand = s_masked + end_max
            finite = np.isfinite(cand)
            best_span = float(np.max(cand[finite])) if finite.any() else null_score
            answerability = best_span - null_score

        if answerability > best_per_q[qi]:
            best_per_q[qi] = answerability

    best_per_q = np.where(np.isfinite(best_per_q), best_per_q, 0.0)
    return best_per_q.tolist()


def run_squad2(conditions, device, batch_size):
    tokenizer, model = load_squad2_model(device)
    scores = []
    for batch in tqdm(list(_chunks(conditions, batch_size)), desc="squad2_answerability"):
        qs = [c["question"] for c in batch]
        ctxs = [c["context"] for c in batch]
        scores.extend(squad2_answerability_batch(tokenizer, model, qs, ctxs, device))
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------------
# (b) MiniCheck entailment (ORACLE: uses gold_answer)
# ---------------------------------------------------------------------------

def run_minicheck(conditions, batch_size):
    from minicheck.minicheck import MiniCheck
    model = MiniCheck(model_name=MINICHECK_MODEL_NAME,
                       cache_dir="/tmp/minicheck_cache", batch_size=batch_size)
    scores = []
    for batch in tqdm(list(_chunks(conditions, batch_size)), desc="minicheck_oracle"):
        docs = [c["context"] for c in batch]
        claims = [f"The answer to the question '{c['question']}' is {c['gold_answer']}."
                  for c in batch]
        _, probs, _, _ = model.score(docs=docs, claims=claims)
        scores.extend(float(p) for p in probs)
    return scores


# ---------------------------------------------------------------------------
# (c) Length-only logistic regression (sanity check)
# ---------------------------------------------------------------------------

_CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z]*\b")


def _n_capitalized_tokens(text):
    """Simple capitalised-token count as a crude entity-density proxy."""
    return len(_CAP_TOKEN_RE.findall(text))


def run_length_baseline(conditions, n_splits=5, seed=42):
    X = np.array([
        [len(c["context"]), _n_capitalized_tokens(c["context"]), len(c["context"].split())]
        for c in conditions
    ], dtype=float)
    y = np.array([1 if c["sufficient"] else 0 for c in conditions])

    pipe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    return oof_scores.tolist()


# ---------------------------------------------------------------------------
# Bootstrap AUROC: resample QUESTIONS, keep both conditions of a pair together
# ---------------------------------------------------------------------------

def bootstrap_auroc_by_question(y_true, y_score, question_ids, n_boot=1000, seed=42):
    """95% bootstrap CI for AUROC, resampling question ids with replacement
    (both conditions of a resampled question travel together), matching the
    paired structure of SufficiencyBench."""
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

    point = float(roc_auc_score(y_true, y_score))
    if aurocs:
        lo, hi = (float(x) for x in np.percentile(aurocs, [2.5, 97.5]))
    else:
        lo, hi = float("nan"), float("nan")
    return point, lo, hi


def summarize_method(name, scores, conditions, n_boot, seed, oracle=False):
    y_true = np.array([1 if c["sufficient"] else 0 for c in conditions])
    y_score = np.array(scores, dtype=float)
    qids = [c["question_id"] for c in conditions]

    point, lo, hi = bootstrap_auroc_by_question(y_true, y_score, qids, n_boot, seed)
    result = {
        "auroc": point, "ci_lo": lo, "ci_hi": hi, "ci_half": (hi - lo) / 2,
        "n": len(conditions), "n_questions": len(set(qids)), "oracle": oracle,
        "per_question_type": {},
    }
    for qt in QUESTION_TYPES:
        idx = [i for i, c in enumerate(conditions) if c["question_type"] == qt]
        if len(idx) < 10 or len(set(y_true[idx])) < 2:
            continue
        result["per_question_type"][qt] = {
            "auroc": float(roc_auc_score(y_true[idx], y_score[idx])),
            "n": len(idx),
        }

    tag = " [ORACLE]" if oracle else ""
    print(f"{name}{tag}: AUROC = {point:.4f}  [{lo:.4f}, {hi:.4f}]  n={len(conditions)}")
    for qt, r in result["per_question_type"].items():
        print(f"    {qt:12s} AUROC = {r['auroc']:.4f}  n={r['n']}")
    return result


def print_compact_table(results):
    print("\n" + "=" * 78)
    print("SUMMARY: LLM-free standalone baselines on SufficiencyBench TEST (n=2224)")
    print("=" * 78)
    header = f"{'Method':<34s} {'AUROC':>8s} {'95% CI':>18s} {'n':>6s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        tag = " (oracle)" if r.get("oracle") else ""
        label = f"{name}{tag}"
        print(f"{label:<34s} {r['auroc']:8.4f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  {r['n']:6d}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="LLM-free standalone baselines for SufficiencyBench")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                     help="Only score the first N conditions (debug).")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-squad2", action="store_true")
    ap.add_argument("--skip-minicheck", action="store_true")
    ap.add_argument("--skip-length", action="store_true")
    args = ap.parse_args()

    conditions = load_conditions(args.limit)
    n_q = len(set(c["question_id"] for c in conditions))
    print(f"Loaded {len(conditions)} conditions ({n_q} questions) "
          f"from {BENCH_DIR / 'test.json'}")

    results = {}

    if not args.skip_squad2:
        t0 = time.time()
        scores = run_squad2(conditions, args.device, args.batch_size)
        results["squad2_answerability"] = summarize_method(
            "SQuAD2 Answerability (deberta-v3-large-squad2)",
            scores, conditions, args.n_boot, args.seed)
        print(f"  [squad2] done in {time.time() - t0:.1f}s\n")

    if not args.skip_minicheck:
        t0 = time.time()
        scores = run_minicheck(conditions, args.batch_size)
        results["minicheck_entailment_oracle"] = summarize_method(
            "MiniCheck Entailment (MiniCheck-Flan-T5-Large)",
            scores, conditions, args.n_boot, args.seed, oracle=True)
        print(f"  [minicheck] done in {time.time() - t0:.1f}s\n")

    if not args.skip_length:
        t0 = time.time()
        scores = run_length_baseline(conditions, seed=args.seed)
        results["length_only_logreg"] = summarize_method(
            "Length-only LogReg (5-fold CV)",
            scores, conditions, args.n_boot, args.seed)
        print(f"  [length] done in {time.time() - t0:.1f}s\n")

    print_compact_table(results)

    out_path = RESULTS_DIR / "answerability_baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
