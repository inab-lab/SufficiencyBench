"""
internal_baselines.py

Two INTERNAL-SIGNAL hallucination / sufficiency detectors from the
uncertainty-quantification literature, added as baselines so the paper can
show hidden-state probes beat other *internal-state* methods, not only
output-based ones (verbalized confidence, token entropy, generation match,
semantic entropy, ReDeEP).

(a) INSIDE / EigenScore
------------------------
Chen, Fu, Wang, Xu, Yang, Jia & Wu (2024), "INSIDE: LLMs' Internal States
Retain the Power of Hallucination Detection", ICLR 2024. arXiv:2402.03744.

For each condition, sample K generations at temperature > 0 and take the
LAST-TOKEN hidden state of each sampled response at a fixed middle layer
(the paper's "sentence embedding" surrogate). Stack the K vectors into a
K x d matrix Z, centre it, and compute the differential-entropy surrogate

    EigenScore(Z) = (1/K) * sum_i log(lambda_i)

where lambda_i are the eigenvalues of the regularised covariance-like matrix
Sigma = (1/K) Z_c Z_c^T + alpha*I (we use the K x K Gram matrix, which shares
the informative eigenvalues with the d x d covariance but is far cheaper
since K << d; alpha=1e-3 regularises against a singular Gram matrix). Higher
EigenScore = more semantic diversity across samples = less confident = context
LESS likely sufficient, so we report the sufficiency score as -EigenScore.

Layer choice: this repo already has a per-model "best layer" selected for the
hidden-state probes (llama=10, mistral=12, qwen=16, qwen3=18 -- see
LAYER_INDEX below). We report EigenScore at that layer AND at the paper's own
default of a "middle" layer (n_layers // 2 of the actual loaded model), since
these do not always coincide (for qwen3 they happen to be the same layer).

(b) Lookback Lens
------------------
Chuang, Xie, Luo, Kim, Glass & He (2024), "Lookback Lens: Detecting and
Mitigating Contextual Hallucinations in Large Language Models Using Only
Attention Maps", EMNLP 2024. arXiv:2407.07071.

During GREEDY generation of the answer, for every attention head (layer l,
head h) and every generated token n we compute the lookback ratio

    LR(n, l, h) = A_context(n,l,h) / (A_context(n,l,h) + A_new(n,l,h))

where A_context is the attention mass the query token n places on the
CONTEXT span of the prompt and A_new is the attention mass it places on the
response tokens generated so far (both read directly off the incremental
attention row HF's `generate(output_attentions=True, return_dict_in_generate=
True)` returns at each decode step -- no separate forward pass is needed).
Averaging LR over the generated tokens gives one scalar per head, i.e. an
(n_layers * n_heads)-dim feature vector per condition. Following the original
paper, a linear (logistic-regression) classifier is TRAINED on TRAIN-split
feature vectors to predict sufficiency and evaluated (AUROC) on TEST.

Cost note: generating with output_attentions=True in eager mode for every
condition is the expensive part (see module docstring in redeep.py for the
analogous cost trade-off), so features are cached to
    data/experiments/lookback/<model>/<split>_features.npz
with per-condition resume, and `--train-subset N` (default 800 QUESTIONS,
i.e. ~2N conditions, sampled proportionally stratified by question_type via
_stratified_subset) bounds the train-split cost.

Sign convention for BOTH methods (matches baselines.py / semantic_entropy.py
/ redeep.py): **higher score => context predicted SUFFICIENT**.

Usage
-----
  python src/methods/internal_baselines.py --model llama --device cuda:0 \
      --methods inside lookback --batch-size 4
  # smoke test (CPU, tiny model, meaningless numbers -- verified end-to-end):
  python src/methods/internal_baselines.py --model llama --device cpu \
      --model-path hf-internal-testing/tiny-random-LlamaForCausalLM \
      --limit 4 --batch-size 2 --methods inside lookback \
      --splits train test --train-subset 2
"""

import json
import argparse
import sys
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, EXPERIMENTS_DIR, MODEL_CONFIGS

# --- Hyperparameters (defaults follow the papers where applicable) ----------
K_SAMPLES = 10            # INSIDE: number of sampled responses per condition
INSIDE_TEMPERATURE = 0.7  # INSIDE: sampling temperature
MAX_NEW_TOKENS = 32       # both methods: response length (same as baselines.py's
                          # answer prompt; short answers on SufficiencyBench)
LOOKBACK_MAX_LENGTH = 1024  # prompt truncation for lookback: attention memory
                            # scales with seq^2 in eager mode (mirrors redeep.py)
INSIDE_MAX_LENGTH = 2048    # INSIDE only needs hidden states, not attentions,
                            # so it can afford the full baselines.py truncation
EIGEN_ALPHA = 1e-3
TRAIN_SUBSET_DEFAULT = 800  # questions (examples), not conditions

# Per-model "best layer" used by the hidden-state probes elsewhere in this
# project (see scripts/bootstrap_ci.py:BEST_LAYERS and the CSP probe).
# v1.1 benchmark: validation-selected Standard-probe layers (zero-based block index as in deco.py)
LAYER_INDEX = {"llama": 14, "mistral": 13, "qwen": 19, "qwen3": 20}


# =============================================================================
# Model loading (8-bit bitsandbytes, eager attention -- eager is REQUIRED for
# output_attentions=True, which Lookback Lens needs; INSIDE only needs hidden
# states but shares the same loaded model). Mirrors extract_states.py's 8-bit
# loading pattern. On CPU (only used for the tiny-model smoke test:
# bitsandbytes 8-bit requires CUDA) we fall back to plain float32.
# =============================================================================

def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(low_cpu_mem_usage=True, attn_implementation="eager")
    if device.startswith("cuda"):
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = device
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer


# =============================================================================
# Shared helpers
# =============================================================================

def _flatten_conditions(examples, limit=None):
    """Flatten (sufficient, insufficient) conditions, carrying question-level
    metadata (mirrors baselines.py / redeep.py's flattening)."""
    conditions = []
    for ex in examples:
        for cond in ex["conditions"]:
            conditions.append({
                **cond,
                "question_id": ex.get("id"),
                "question_type": ex.get("question_type"),
                "gold_answer": ex.get("gold_answer", ""),
            })
    if limit is not None:
        conditions = conditions[:limit]
    return conditions


def _stratified_subset(examples, n, seed=42):
    """Select ~n examples from `examples`, stratified proportionally by
    question_type (largest-remainder allocation so per-stratum counts sum to
    exactly n), rather than a plain prefix slice. Returns all examples
    unchanged if n is None or already >= len(examples)."""
    if n is None or n >= len(examples):
        return list(examples)
    by_type = {}
    for ex in examples:
        by_type.setdefault(ex.get("question_type"), []).append(ex)
    types = sorted(by_type.keys(), key=str)
    total = len(examples)

    raw = {t: n * len(by_type[t]) / total for t in types}
    counts = {t: int(np.floor(raw[t])) for t in types}
    remainder = n - sum(counts.values())
    for t in sorted(types, key=lambda t: raw[t] - counts[t], reverse=True)[:remainder]:
        counts[t] += 1

    rng = np.random.RandomState(seed)
    selected = []
    for t in types:
        pool = by_type[t]
        k = min(counts[t], len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        selected.extend(pool[i] for i in idx)
    rng.shuffle(selected)
    return selected


def _context_token_positions(prompt, context, tokenizer, max_length):
    """Token indices (into the truncated prompt) that fall inside `context`.

    Uses fast-tokenizer offset mapping; falls back to "everything except the
    last 20 tokens" if the context span can't be located. (Duplicated from
    redeep.py to keep this module self-contained.)
    """
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_length, return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    n_tok = len(offsets)
    start = prompt.find(context) if context else -1
    if start < 0:
        return list(range(max(0, n_tok - 20)))
    end = start + len(context)
    pos = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
    return pos if pos else list(range(max(0, n_tok - 20)))


def bootstrap_auroc_by_question(labels, scores, question_ids, n_boot=1000, seed=42):
    """Bootstrap 95% CI for AUROC, resampling whole QUESTIONS (with all their
    conditions) with replacement rather than individual conditions, so paired
    sufficient/insufficient conditions of the same question move together.
    Returns (point_estimate, ci_lo, ci_hi)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    qids = np.asarray([str(q) for q in question_ids])
    uniq_q = np.unique(qids)
    q_to_idx = {q: np.where(qids == q)[0] for q in uniq_q}

    if len(set(labels.tolist())) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(labels, scores))

    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        sampled_q = rng.choice(uniq_q, size=len(uniq_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in sampled_q])
        yt, ys = labels[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, ys))
    if not boots:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _per_question_type_auroc(labels, scores, qtypes):
    out = {}
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    qtypes = np.asarray([str(q) for q in qtypes])
    for qt in sorted(set(qtypes.tolist())):
        idx = np.where(qtypes == qt)[0]
        if len(idx) < 10 or len(set(labels[idx].tolist())) < 2:
            continue
        out[qt] = float(roc_auc_score(labels[idx], scores[idx]))
    return out


# =============================================================================
# (a) INSIDE / EigenScore
# =============================================================================

def eigen_score(Z: np.ndarray, alpha: float = EIGEN_ALPHA) -> float:
    """EigenScore(Z) = (1/K) * sum_i log(lambda_i) of the regularised
    covariance-like matrix of the centred K x d embedding matrix Z. Uses the
    K x K Gram matrix (equivalent nonzero spectrum to the d x d covariance,
    far cheaper since K << d)."""
    Z = np.asarray(Z, dtype=np.float64)
    K = Z.shape[0]
    if K < 2:
        return 0.0
    Zc = Z - Z.mean(axis=0, keepdims=True)
    G = (Zc @ Zc.T) / K + alpha * np.eye(K)
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.clip(eigvals, 1e-12, None)  # numerical floor, never exactly 0
    return float(np.mean(np.log(eigvals)))


def _last_content_positions(gen_ids_2d, pad_id):
    """For each row of GENERATED-part token ids, the index (relative to the
    generated part) of the last non-pad token; 0 if the row is all pad."""
    out = []
    for row in gen_ids_2d:
        nz = (row != pad_id).nonzero(as_tuple=True)[0]
        out.append(int(nz[-1].item()) if len(nz) else 0)
    return out


def sample_hidden_states_batch(model, tokenizer, prompts, device, k, temperature,
                                max_new_tokens, layers):
    """Sample K generations for EACH of the B prompts (one padded batch, as in
    semantic_entropy.py's sample_generations_batch) and return, for each
    requested layer index, an array (B, K, hidden_dim) holding the LAST-TOKEN
    hidden state of each sampled response.

    Layer-index convention matches extract_states.py: layer L refers to
    hidden_states[L+1] (hidden_states[0] is the embedding output)."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            list(prompts), return_tensors="pt", padding=True,
            truncation=True, max_length=INSIDE_MAX_LENGTH,
        )
        enc = {kk: v.to(device) for kk, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=1.0, num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        tokenizer.padding_side = prev_side

    B = len(prompts)
    total_len = gen.shape[1]
    gen_len = max(total_len - prompt_len, 0)
    pad_id = tokenizer.pad_token_id

    # Left-pad mask (from the prompt) repeated per sample, plus an all-ones
    # mask for the generated span. Trailing pad tokens from early stopping
    # don't need to be zeroed: causal attention means later padding cannot
    # affect the earlier real-token hidden states we actually read.
    left_mask_rep = enc["attention_mask"].repeat_interleave(k, dim=0)
    if gen_len > 0:
        gen_mask = torch.ones((B * k, gen_len), dtype=left_mask_rep.dtype, device=device)
        full_mask = torch.cat([left_mask_rep, gen_mask], dim=1)
    else:
        full_mask = left_mask_rep

    gen_only = gen[:, prompt_len:]
    last_rel = _last_content_positions(gen_only, pad_id) if gen_len > 0 else [0] * (B * k)
    abs_idx = [min(prompt_len + r, total_len - 1) for r in last_rel]

    with torch.no_grad():
        out = model(
            input_ids=gen, attention_mask=full_mask,
            output_hidden_states=True, use_cache=False,
        )
    hs = out.hidden_states  # tuple length n_transformer_layers + 1
    n_avail = len(hs) - 1

    result = {}
    for layer in layers:
        li = min(max(layer, 0), n_avail - 1)  # clip to what THIS model actually has
        h = hs[li + 1]  # [B*K, S, D]
        vecs = torch.stack([h[i, abs_idx[i], :] for i in range(B * k)])
        vecs = vecs.float().cpu().numpy().reshape(B, k, -1)
        result[layer] = vecs

    del out, hs
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_inside_eval(model, tokenizer, conditions, device, batch_size, k,
                     temperature, max_new_tokens, best_layer, middle_layer):
    """Score every condition with EigenScore at both `best_layer` and
    `middle_layer`, returning overall + per-question-type AUROC (bootstrapped
    over questions) for each."""
    layers = sorted(set([best_layer, middle_layer]))
    raw_scores = {layer: [] for layer in layers}

    for start in tqdm(range(0, len(conditions), batch_size), desc="inside"):
        batch = conditions[start:start + batch_size]
        prompts = [c["prompt"] for c in batch]
        hid = sample_hidden_states_batch(
            model, tokenizer, prompts, device, k, temperature, max_new_tokens, layers)
        for layer in layers:
            for b in range(len(batch)):
                es = eigen_score(hid[layer][b])
                raw_scores[layer].append(-es)  # higher => more sufficient

    labels = np.array([1 if c.get("sufficient") else 0 for c in conditions])
    qids = [c.get("question_id") for c in conditions]
    qtypes = [c.get("question_type") for c in conditions]

    out = {}
    layer_names = {best_layer: "best_layer", middle_layer: "middle_layer"}
    for layer, name in layer_names.items():
        arr = np.asarray(raw_scores[layer], dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            fill = float(np.nanmean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
            arr = np.where(np.isfinite(arr), arr, fill)
            print(f"  [warn] inside[{name}]: imputed {n_bad} non-finite scores with {fill:.4f}")
        point, lo, hi = bootstrap_auroc_by_question(labels, arr, qids)
        entry = {
            "layer": int(layer),
            "overall_auroc": point, "ci_lo": lo, "ci_hi": hi,
            "n": int(len(labels)), "n_imputed": n_bad,
            "per_question_type_auroc": _per_question_type_auroc(labels, arr, qtypes),
        }
        out[name] = entry
        print(f"inside[{name}=L{layer}]: overall AUROC = {point:.4f} "
              f"[{lo:.4f}, {hi:.4f}] (n={len(labels)})")
    return out


# =============================================================================
# (b) Lookback Lens
# =============================================================================

def lookback_features(model, tokenizer, prompt, context, device, max_length,
                       max_new_tokens):
    """Greedily generate the answer and return the (n_layers * n_heads,)
    lookback-ratio feature vector for this one condition, or None if nothing
    was generated."""
    ctx_pos_all = _context_token_positions(prompt, context, tokenizer, max_length)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    ctx_pos = [c for c in ctx_pos_all if c < prompt_len]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, output_attentions=True,
            return_dict_in_generate=True, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    attentions = out.attentions  # tuple[n_steps] of tuple[n_layers] of [1,H,Qlen,Klen]
    n_steps = len(attentions)
    if n_steps == 0:
        return None
    n_layers = len(attentions[0])
    n_heads = attentions[0][0].shape[1]
    ratio_accum = np.zeros((n_layers, n_heads), dtype=np.float64)

    for t in range(n_steps):
        step = attentions[t]
        for l in range(n_layers):
            att = step[l][0].float()          # [H, Qlen, Klen]
            row = att[:, -1, :]                # newly-generated token's attention row
            klen = row.shape[-1]
            ctx_idx = [p for p in ctx_pos if p < klen]
            gen_idx = list(range(prompt_len, klen))  # response tokens so far (incl. self-key)
            ctx_mass = row[:, ctx_idx].sum(dim=-1) if ctx_idx else torch.zeros(n_heads, device=row.device)
            gen_mass = row[:, gen_idx].sum(dim=-1) if gen_idx else torch.zeros(n_heads, device=row.device)
            denom = ctx_mass + gen_mass
            ratio = torch.where(denom > 1e-8, ctx_mass / denom, torch.full_like(denom, 0.5))
            ratio_accum[l] += ratio.cpu().numpy()

    ratio_mean = ratio_accum / n_steps
    del out, attentions
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ratio_mean.flatten().astype(np.float32)  # (n_layers * n_heads,)


def _load_feature_cache(cache_path):
    if not cache_path.exists():
        return {}
    d = np.load(cache_path, allow_pickle=True)
    cache = {}
    for cid, feat, lab, qt, qid in zip(
        d["condition_ids"], d["features"], d["labels"], d["question_types"], d["question_ids"]
    ):
        cache[str(cid)] = (feat, int(lab), str(qt), str(qid))
    return cache


def _save_feature_cache(cache_path, cache):
    if not cache:
        return
    ids = list(cache.keys())
    feats = np.stack([cache[i][0] for i in ids])
    labels = np.array([cache[i][1] for i in ids])
    qtypes = np.array([cache[i][2] for i in ids])
    qids = np.array([cache[i][3] for i in ids])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # NB: np.savez silently APPENDS ".npz" to any filename that doesn't
    # already end in it, so a tmp name like "*.npz.tmp" actually gets written
    # as "*.npz.tmp.npz" and the plain rename below would raise
    # FileNotFoundError. Keep the tmp name ending in ".npz" so savez writes
    # exactly the path we pass it.
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez(tmp, condition_ids=np.array(ids), features=feats,
              labels=labels, question_types=qtypes, question_ids=qids)
    tmp.replace(cache_path)


def extract_lookback_split(model, tokenizer, conditions, device, max_length,
                            max_new_tokens, cache_path, save_every=50):
    """Resumable feature extraction: conditions already in the cache (by
    condition_id) are skipped. Saves incrementally every `save_every` new
    conditions and once more at the end."""
    cache = _load_feature_cache(cache_path)
    todo = [c for c in conditions if c["condition_id"] not in cache]
    if todo:
        print(f"  [lookback] {cache_path.parent.name}/{cache_path.stem}: "
              f"{len(cache)} cached, {len(todo)} to extract")
    n_new = 0
    for cond in tqdm(todo, desc=f"lookback[{cache_path.stem}]"):
        feat = lookback_features(
            model, tokenizer, cond["prompt"], cond.get("context", ""),
            device, max_length, max_new_tokens)
        if feat is None:
            continue
        cache[cond["condition_id"]] = (
            feat, 1 if cond.get("sufficient") else 0,
            str(cond.get("question_type")), str(cond.get("question_id")),
        )
        n_new += 1
        if n_new % save_every == 0:
            _save_feature_cache(cache_path, cache)
    _save_feature_cache(cache_path, cache)
    return cache


def run_lookback_eval(model, tokenizer, model_key, device, limit, train_subset,
                       splits, max_length, max_new_tokens):
    lookback_dir = EXPERIMENTS_DIR / "lookback" / model_key
    caches = {}
    for split in splits:
        data_path = BENCH_DIR / f"{split}.json"
        if not data_path.exists():
            print(f"  [lookback] {data_path} not found, skipping split '{split}'")
            continue
        with open(data_path) as f:
            examples = json.load(f)
        if split == "train" and train_subset:
            examples = _stratified_subset(examples, train_subset)
        conditions = _flatten_conditions(examples, limit=limit)
        cache_path = lookback_dir / f"{split}_features.npz"
        caches[split] = extract_lookback_split(
            model, tokenizer, conditions, device, max_length, max_new_tokens, cache_path)

    result = {
        "n_train": len(caches.get("train", {})),
        "n_test": len(caches.get("test", {})),
    }
    if not (caches.get("train") and caches.get("test")):
        print("  [lookback] missing train and/or test feature cache; run with "
              "--splits train test to compute both before the classifier can be evaluated")
        return result

    train_ids = list(caches["train"].keys())
    X_train = np.stack([caches["train"][i][0] for i in train_ids])
    y_train = np.array([caches["train"][i][1] for i in train_ids])
    if len(set(y_train.tolist())) < 2:
        print("  [lookback] train split has only one class; cannot fit classifier")
        return result

    test_ids = list(caches["test"].keys())
    X_test = np.stack([caches["test"][i][0] for i in test_ids])
    y_test = np.array([caches["test"][i][1] for i in test_ids])
    q_test = [caches["test"][i][2] for i in test_ids]
    qid_test = [caches["test"][i][3] for i in test_ids]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)
    y_score = clf.predict_proba(X_test_s)[:, 1]

    point, lo, hi = bootstrap_auroc_by_question(y_test, y_score, qid_test)
    result.update({
        "overall_auroc": point, "ci_lo": lo, "ci_hi": hi,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "per_question_type_auroc": _per_question_type_auroc(y_test, y_score, q_test),
    })
    print(f"lookback_lens: overall AUROC = {point:.4f} [{lo:.4f}, {hi:.4f}] "
          f"(n_train={len(y_train)}, n_test={len(y_test)}, dim={X_train.shape[1]})")
    return result


# =============================================================================
# Orchestration
# =============================================================================

def run_internal_baselines(model_key="llama", device="cuda:0", batch_size=4,
                            limit=None, methods=("inside", "lookback"),
                            splits=("train", "test"), train_subset=TRAIN_SUBSET_DEFAULT,
                            k=K_SAMPLES, temperature=INSIDE_TEMPERATURE,
                            max_new_tokens=MAX_NEW_TOKENS,
                            lookback_max_length=LOOKBACK_MAX_LENGTH,
                            model_path=None, stage_marker=None):
    cfg = MODEL_CONFIGS[model_key]
    path = model_path or cfg["path"]
    model, tokenizer = load_model(path, device)

    # "True middle layer" is defined on the ACTUALLY LOADED model (matters for
    # --model-path overrides, e.g. the CPU tiny-model smoke test, whose layer
    # count differs from MODEL_CONFIGS); fall back to the static config value
    # if the loaded model's config doesn't expose a standard layer-count field.
    n_layers_actual = getattr(model.config, "num_hidden_layers", None) or cfg.get("n_layers")
    middle_layer = n_layers_actual // 2
    best_layer = LAYER_INDEX.get(model_key)
    if best_layer is None:
        best_layer = middle_layer
        print(f"  [warn] no literature best-layer configured for '{model_key}' "
              f"(LAYER_INDEX has {sorted(LAYER_INDEX.keys())}); falling back to "
              f"n_layers // 2 = {best_layer} for BOTH the 'best' and 'middle' report")

    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "internal_baselines.json"
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)

    settings = {
        "model": model_key, "model_path": path, "device": device,
        "k": k, "temperature": temperature, "max_new_tokens": max_new_tokens,
        "best_layer": int(best_layer), "middle_layer": int(middle_layer),
        "lookback_max_length": lookback_max_length,
        "train_subset": train_subset, "batch_size": batch_size,
        "limit": limit, "methods": list(methods), "splits": list(splits),
    }

    if "inside" in methods:
        with open(BENCH_DIR / "test.json") as f:
            test_data = json.load(f)
        conditions = _flatten_conditions(test_data, limit=limit)
        results["inside"] = run_inside_eval(
            model, tokenizer, conditions, device, batch_size, k, temperature,
            max_new_tokens, best_layer, middle_layer)

    if "lookback" in methods:
        results["lookback_lens"] = run_lookback_eval(
            model, tokenizer, model_key, device, limit, train_subset,
            splits, lookback_max_length, max_new_tokens)

    results["settings"] = settings
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    if stage_marker:
        from pathlib import Path
        Path(stage_marker).parent.mkdir(parents=True, exist_ok=True)
        Path(stage_marker).write_text("done\n")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4,
                     help="Conditions per batch for INSIDE generation (each "
                          "issues batch_size * k sequences); lookback always "
                          "uses batch size 1 (per-condition greedy generation "
                          "with output_attentions).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap conditions per split (debug/smoke test).")
    ap.add_argument("--methods", nargs="+", default=["inside", "lookback"],
                     choices=["inside", "lookback"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                     choices=["train", "val", "test"],
                     help="lookback: which splits to extract/cache features "
                          "for in this invocation (evaluation runs only once "
                          "both 'train' and 'test' caches exist on disk).")
    ap.add_argument("--train-subset", type=int, default=TRAIN_SUBSET_DEFAULT,
                     help="lookback: number of TRAIN questions (examples, not "
                          "conditions) to use, bounding extraction cost.")
    ap.add_argument("--k", type=int, default=K_SAMPLES,
                     help="INSIDE: number of sampled responses per condition.")
    ap.add_argument("--temperature", type=float, default=INSIDE_TEMPERATURE,
                     help="INSIDE: sampling temperature.")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                     help="Response length for both INSIDE sampling and "
                          "lookback greedy generation.")
    ap.add_argument("--lookback-max-length", type=int, default=LOOKBACK_MAX_LENGTH,
                     help="Prompt truncation for lookback (attention memory "
                          "scales with seq^2 in eager mode).")
    ap.add_argument("--model-path", default=None,
                     help="[testing only] override the model path/HF id "
                          "instead of MODEL_CONFIGS[--model]['path'] -- lets "
                          "a CPU smoke test use e.g. sshleifer/tiny-gpt2 while "
                          "still exercising the real --model layer settings.")
    ap.add_argument("--stage-marker", default=None,
                     help="[driver use] touch this file on successful "
                          "completion, so a thermal-guarded driver can treat "
                          "it as the resumability marker for this stage.")
    args = ap.parse_args()

    run_internal_baselines(
        model_key=args.model, device=args.device, batch_size=args.batch_size,
        limit=args.limit, methods=tuple(args.methods), splits=tuple(args.splits),
        train_subset=args.train_subset, k=args.k, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        lookback_max_length=args.lookback_max_length,
        model_path=args.model_path, stage_marker=args.stage_marker,
    )


if __name__ == "__main__":
    main()
