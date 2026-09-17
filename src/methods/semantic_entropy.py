"""
semantic_entropy.py

Semantic Entropy baseline for detecting context sufficiency on
SufficiencyBench.

Method (faithful reproduction of the original)
-----------------------------------------------
Semantic Entropy (SE) measures the *meaning-level* uncertainty of a language
model's answer to a prompt. Two answers that are phrased differently but mean
the same thing ("Paris" vs. "It is Paris") should not count as disagreement.
SE therefore clusters sampled generations by semantic equivalence and computes
the entropy over the resulting clusters rather than over surface tokens.

Algorithm (per condition):
  1. Sample K generations from the prompt at temperature T > 0 (do_sample=True).
  2. Cluster the K generations by *bidirectional entailment*: two generations
     a, b belong to the same semantic cluster iff an NLI/entailment model
     predicts ENTAILMENT for the pair (a -> b) AND for the pair (b -> a)
     (mutual entailment). Clustering is greedy: each generation joins the first
     existing cluster whose representative it is mutually-entailed with, else it
     starts a new cluster. The NLI input for each direction is the concatenation
     "question + answer", following Kuhn et al.
  3. Estimate cluster probabilities from cluster sizes (the "discrete" semantic
     entropy estimator of Farquhar et al., which assumes each sampled answer is
     equally probable and uses cluster frequencies): p_c = |c| / K.
  4. Semantic entropy = -sum_c p_c * log(p_c).

Score sign
----------
Sufficient context should yield a more certain, semantically-concentrated set
of answers (fewer clusters => lower entropy), whereas insufficient context
should produce scattered / contradictory answers (more clusters => higher
entropy). We therefore report

    score = -semantic_entropy

so that HIGHER score => context SUFFICIENT, which is the orientation the
AUROC-vs-sufficient-label evaluation expects (mirrors token_entropy in
baselines.py, which also negates entropy).

Cost / configurability
-----------------------
K (number of samples) and temperature are CLI-configurable (defaults K=10,
temperature=0.7). This is the most expensive baseline: runtime is dominated by
K generations per condition plus O(clusters) NLI comparisons per generation, so
lowering K speeds the run up roughly LINEARLY (K=5 ~ 2x faster than K=10).

NLI model
---------
Uses microsoft/deberta-large-mnli (the standard DeBERTa-MNLI used for semantic
entropy). Loaded on the SAME device as generation. The entailment label index
is read from the model config (no hard-coding).

Usage
-----
  python src/methods/semantic_entropy.py --model llama --device cuda:0
  # smoke test:
  python src/methods/semantic_entropy.py --model llama --device cuda:1 \
      --limit 3 --k 4

Citations
---------
- Kuhn, Gal & Farquhar (2023), "Semantic Uncertainty: Linguistic Invariances
  for Uncertainty Estimation in Natural Language Generation", ICLR 2023.
- Farquhar, Kossen, Kuhn & Gal (2024), "Detecting hallucinations in large
  language models using semantic entropy", Nature 630, 625-630.
"""

import json
import math
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS

# Standard DeBERTa-MNLI entailment model for semantic clustering.
NLI_MODEL = "microsoft/deberta-large-mnli"
# Max tokens generated per sample (answers on SufficiencyBench are short).
MAX_NEW_TOKENS = 64
# DeBERTa forward-pass mini-batch size for the (batched) entailment checks.
# Kept modest because NLI runs in float32 (fp16 is unsafe for DeBERTa's
# disentangled attention) at sequence length up to 512.
NLI_BATCH_SIZE = 32


def sample_generations(model, tokenizer, prompt, device, k, temperature):
    """Sample K generations from the prompt at temperature > 0.

    Returns a list of K generated strings (prompt stripped)."""
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )

    plen = inputs["input_ids"].shape[1]
    gens = [
        tokenizer.decode(seq[plen:], skip_special_tokens=True).strip()
        for seq in out
    ]
    return gens


def _entails(nli_model, nli_tokenizer, premise, hypothesis, device, ent_idx):
    """True iff the NLI model predicts ENTAILMENT for premise -> hypothesis."""
    inputs = nli_tokenizer(
        premise, hypothesis, return_tensors="pt",
        truncation=True, max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits[0]
    return int(torch.argmax(logits).item()) == ent_idx


def cluster_by_entailment(
    generations, question, nli_model, nli_tokenizer, device, ent_idx
):
    """Greedy bidirectional-entailment clustering (Kuhn et al.).

    Two answers share a cluster iff they mutually entail each other, using
    "question + answer" as the NLI sequence. Returns a list of cluster sizes."""
    clusters = []  # each entry is a representative answer string
    sizes = []
    for gen in generations:
        seq_g = f"{question} {gen}".strip()
        placed = False
        for i, rep in enumerate(clusters):
            seq_r = f"{question} {rep}".strip()
            # Bidirectional (mutual) entailment.
            if _entails(nli_model, nli_tokenizer, seq_r, seq_g, device, ent_idx) \
               and _entails(nli_model, nli_tokenizer, seq_g, seq_r, device, ent_idx):
                sizes[i] += 1
                placed = True
                break
        if not placed:
            clusters.append(gen)
            sizes.append(1)
    return sizes


def semantic_entropy_score(cluster_sizes):
    """Discrete semantic entropy from cluster sizes, negated so that
    higher score => lower uncertainty => context sufficient."""
    k = sum(cluster_sizes)
    if k == 0:
        return 0.0
    entropy = 0.0
    for s in cluster_sizes:
        p = s / k
        if p > 0:
            entropy -= p * math.log(p)
    return -entropy  # negate: higher = more certain = predict sufficient


# ---------------------------------------------------------------------------
# BATCHED implementations (same math, fewer/larger forward passes).
#
# The clustering *decisions* are byte-for-byte identical to the one-pair-at-a-
# time path above: greedy bidirectional-entailment placement depends only on
# the per-pair mutual-entailment predicate, which is deterministic and
# independent of the order in which the NLI forward passes are executed.
# Batching therefore changes only *how many* pairs we score per forward pass,
# never the resulting cluster assignment (and hence never the entropy).
# Generation, by contrast, is stochastic (do_sample=True), so per-item scores
# are NOT expected to match the unbatched path exactly.
# ---------------------------------------------------------------------------


def sample_generations_batch(model, tokenizer, prompts, device, k, temperature):
    """Sample K generations for EACH of the B prompts in one padded batch.

    Prompts are LEFT-padded (decoder-only) so every completion starts at the
    same column and the prompt can be sliced off uniformly. Returns a list of
    B lists, each holding K generated strings (prompt stripped). The math for
    a single condition is identical to sample_generations(); only the batching
    differs."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            list(prompts), return_tensors="pt",
            padding=True, truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        tokenizer.padding_side = prev_side

    # out has shape (B*k, L); rows for prompt b occupy [b*k : (b+1)*k].
    plen = inputs["input_ids"].shape[1]  # uniform (left-padded) prompt length
    batch_gens = []
    for b in range(len(prompts)):
        gens = [
            tokenizer.decode(out[b * k + s][plen:], skip_special_tokens=True).strip()
            for s in range(k)
        ]
        batch_gens.append(gens)
    return batch_gens


def _batch_entails(
    nli_model, nli_tokenizer, premises, hypotheses, device, ent_idx,
    nli_batch_size=NLI_BATCH_SIZE,
):
    """Vectorised entailment: returns a bool list, one per (premise, hypothesis)
    pair, True iff the NLI model predicts ENTAILMENT. Runs in DeBERTa mini-
    batches of nli_batch_size, float32 (fp16 unsafe)."""
    results = []
    for i in range(0, len(premises), nli_batch_size):
        p = premises[i:i + nli_batch_size]
        h = hypotheses[i:i + nli_batch_size]
        inputs = nli_tokenizer(
            p, h, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)
        results.extend((preds == ent_idx).tolist())
    return results


def _greedy_clusters_from_matrix(ent):
    """Greedy bidirectional-entailment clustering from a precomputed directed
    entailment matrix. ent[a][b] = (seq_a entails seq_b). Two generations share
    a cluster iff they mutually entail (ent[rep][g] and ent[g][rep]). This is
    the SAME greedy rule as cluster_by_entailment(): each generation joins the
    first existing cluster (identified by its founding generation) it mutually
    entails, else starts a new cluster. Returns cluster sizes."""
    n = len(ent)
    reps = []   # indices of the generation that founded each cluster
    sizes = []
    for i in range(n):
        placed = False
        for ci, r in enumerate(reps):
            if ent[r][i] and ent[i][r]:
                sizes[ci] += 1
                placed = True
                break
        if not placed:
            reps.append(i)
            sizes.append(1)
    return sizes


def cluster_batch_by_entailment(
    batch_gens, questions, nli_model, nli_tokenizer, device, ent_idx,
    nli_batch_size=NLI_BATCH_SIZE,
):
    """Cluster each condition's K generations, but score ALL directed answer-
    pairs across the WHOLE batch in shared DeBERTa mini-batches instead of one
    pair at a time. Returns a list of cluster-size lists (one per condition).

    Equivalence: for a condition we precompute the full directed entailment
    matrix over its K generations, then run the identical greedy placement in
    Python. Because the greedy decision only reads matrix entries, the cluster
    assignment (and entropy) is identical to cluster_by_entailment()."""
    premises, hypotheses, index_map = [], [], []
    seqs_per_cond = []
    for bi, (gens, q) in enumerate(zip(batch_gens, questions)):
        seqs = [f"{q} {g}".strip() for g in gens]
        seqs_per_cond.append(seqs)
        n = len(seqs)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                premises.append(seqs[a])
                hypotheses.append(seqs[b])
                index_map.append((bi, a, b))

    flags = _batch_entails(
        nli_model, nli_tokenizer, premises, hypotheses, device, ent_idx,
        nli_batch_size)

    # Rebuild a directed entailment matrix per condition.
    mats = [
        [[False] * len(seqs) for _ in range(len(seqs))]
        for seqs in seqs_per_cond
    ]
    for (bi, a, b), f in zip(index_map, flags):
        mats[bi][a][b] = bool(f)

    return [_greedy_clusters_from_matrix(m) for m in mats]


def run_semantic_entropy(
    model_key="llama", device="cuda:0", k=10, temperature=0.7, limit=0,
    batch_size=8,
):
    """Run semantic entropy on SufficiencyBench and report overall +
    per-quadrant AUROC (mirrors baselines.run_baselines layout)."""
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    # NLI/entailment model for semantic clustering, on the same device.
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    # DeBERTa's disentangled attention is not fp16-safe, so keep the NLI model
    # in float32 (it is small, ~0.4B params).
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, torch_dtype=torch.float32, device_map=device,
    )
    nli_model.eval()
    # Read the entailment class index from the model config (no hard-coding).
    ent_idx = nli_model.config.label2id.get(
        "ENTAILMENT",
        nli_model.config.label2id.get("entailment", 2),
    )

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    # Flatten (sufficient, insufficient) conditions, carrying the question so
    # the NLI clustering can condition on it (as in Kuhn et al.).
    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question": ex.get("question", ""),
                "question_type": ex.get("question_type"),
                "gold_answer": ex.get("gold_answer", ""),
            })

    if limit and limit > 0:
        conditions = conditions[:limit]

    bs = max(1, int(batch_size))
    scores = []
    for start in tqdm(
        range(0, len(conditions), bs),
        desc=f"semantic_entropy[{model_key}]",
    ):
        chunk = conditions[start:start + bs]
        prompts = [c["prompt"] for c in chunk]
        questions = [c.get("question", "") for c in chunk]
        # Batched generation: B*K sequences in one padded generate() call.
        batch_gens = sample_generations_batch(
            model, tokenizer, prompts, device, k, temperature)
        # Batched NLI: all directed answer-pairs across the chunk scored in
        # shared DeBERTa mini-batches.
        batch_sizes = cluster_batch_by_entailment(
            batch_gens, questions,
            nli_model, nli_tokenizer, device, ent_idx)
        for sizes in batch_sizes:
            scores.append(semantic_entropy_score(sizes))

    labels = [1 if c.get("sufficient") else 0 for c in conditions]
    quads = [c.get("quadrant") for c in conditions]

    # Sanitize: replace any non-finite score with the finite mean so a few bad
    # conditions can't NaN-crash roc_auc and lose the whole run.
    arr = np.asarray(scores, dtype=float)
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        fill = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
        print(f"  [warn] semantic_entropy: imputed {n_bad} non-finite scores with {fill:.4f}")
    scores = arr.tolist()

    results = {"semantic_entropy": {}}
    entry = results["semantic_entropy"]
    entry["n_imputed"] = n_bad
    try:
        entry["overall_auroc"] = float(roc_auc_score(labels, scores))
        for quad in ["Q1", "Q2", "Q3", "Q4"]:
            idx = [i for i, q in enumerate(quads) if q == quad]
            if len(idx) < 10 or len(set(labels[i] for i in idx)) < 2:
                continue
            entry[f"{quad}_auroc"] = float(
                roc_auc_score([labels[i] for i in idx], [scores[i] for i in idx]))
        print(f"semantic_entropy: overall AUROC = {entry['overall_auroc']:.4f}")
    except Exception as e:  # still write the output JSON below
        entry["error"] = str(e)
        print(f"  [error] semantic_entropy: {e}")

    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "semantic_entropy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir / 'semantic_entropy_results.json'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=10,
                    help="number of samples per condition (lower = ~linearly faster)")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature (>0)")
    ap.add_argument("--limit", type=int, default=0,
                    help="truncate to first N conditions for smoke testing (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="conditions per batch; generation issues B*K sequences "
                         "per generate() call and NLI is scored in shared "
                         "mini-batches (peak GPU mem scales with B*K)")
    args = ap.parse_args()
    run_semantic_entropy(
        args.model, args.device, args.k, args.temperature, args.limit,
        args.batch_size)


if __name__ == "__main__":
    main()
