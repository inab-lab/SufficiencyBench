#!/usr/bin/env python
"""
retrieved_context_eval.py -- context-sufficiency detectors on REAL retrieved contexts.

Answers the reviewer objection "the paper contains no retrieval": instead of the
benchmark's edited gold passages, the context is the BM25 top-k over a pool of
HotpotQA paragraphs, and the sufficiency label is "every supporting-fact title
is among the top-k retrieved titles".

Subcommands
  build     sample HotpotQA questions (SufficiencyBench questions excluded), index the
            paragraph pool (pyserini Lucene BM25; rank_bm25 fallback), retrieve, label,
            format prompts with the benchmark's own templates.
  extract   one model, one split: last-token hidden states at all layers for the
            with-context and question-only prompts, verbalized confidence, closed-book
            parametric knowledge (benchmark rule), with-context answer + EM/F1.
            Checkpointed every 200 records, resumable.
  analyse   transfer probes (trained on SufficiencyBench train states), retrained probes
            (trained on retrieved train states), layer sweep, VC, post-retrieval QPP,
            e5 cosine, oracle answer-in-context; AUROC + CI, PK-stratified AUROC, AURC,
            coverage at risk, Kendall tau vs answer F1, paired bootstrap vs VC, gating.
  summarise aggregate the analysed models.

Reused code (not modified)
  extraction.extract_states : load_model, extract_last_token_hidden, format_question_only
  data.build_sufficiency_bench : format_prompt (with-context template),
                                 test_parametric_knowledge (PK rule; fidelity check only)
  methods.baselines : verbalized_confidence_batch
  analysis.store_baseline_scores : load_best_layers (probe hyper-parameters mirrored
                                   from compute_probe_scores)

Usage: see scripts/run_retrieved_ctx.sh
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.paths import (  # noqa: E402
    BENCH_DIR, DATASETS_DIR, EXPERIMENTS_DIR, HIDDEN_STATES_DIR, MODELS_DIR,
    MODEL_CONFIGS, RESULTS_DIR,
)

RC_DIR = EXPERIMENTS_DIR / "retrieved_ctx"
JAVA21 = "/usr/lib/jvm/java-21-openjdk-amd64"
CHUNK = 200            # checkpoint granularity (records)
GEN_BS = 8             # batch size for VC / PK / answer generation
K_REPORT = (1, 3, 5, 10)
N_RANK_DEPTH = 100     # depth for gold_ranks
SEED_PROBE = 42


# ----------------------------------------------------------------------------
# text utilities
# ----------------------------------------------------------------------------

def norm_question(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def clean_ws(s: str) -> str:
    """SufficiencyBench v1.1 collapses every whitespace run inside contexts to one space."""
    return " ".join(s.split())


def squad_normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_em(pred: str, gold: str) -> float:
    return float(squad_normalize(pred) == squad_normalize(gold))


def squad_f1(pred: str, gold: str) -> float:
    p, g = squad_normalize(pred).split(), squad_normalize(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if len(p) == 0 or len(g) == 0:
        return float(p == g)
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_in_text(answer: str, text: str):
    a = squad_normalize(answer)
    if a in ("yes", "no") or not a:
        return None
    return f" {a} " in f" {squad_normalize(text)} "


def first_line(s: str) -> str:
    s = s.strip()
    return s.split("\n")[0].strip() if s else s


CTX_ABSTAIN = ["cannot answer", "can't answer", "not enough information", "does not contain enough",
               "unable to answer", "cannot be answered", "not provided in the context", "does not provide"]


def _abstains(s: str) -> bool:
    sl = s.lower()
    return any(p in sl for p in CTX_ABSTAIN)


def first_sentence(s: str) -> str:
    s = first_line(s)
    return re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0]


# ----------------------------------------------------------------------------
# Closed-book PK rule. Copied verbatim from build_sufficiency_bench.test_parametric_knowledge
# (the rule is inline there, not a function). `extract --check-pk N` verifies that the
# batched version reproduces that function's "knows" on N questions.
# ----------------------------------------------------------------------------

PK_ABSTAIN = [
    "i don't know", "i'm not sure", "i cannot",
    "don't have enough", "not sure", "i do not know",
    "unable to", "no information", "cannot determine",
]
PK_MAX_NEW_TOKENS = 60
PK_MAX_LEN = 512


def pk_prompt(question: str) -> str:
    return (
        f"Answer this question in one sentence. "
        f"If you don't know, say 'I don't know.'\n\n"
        f"Question: {question}\nAnswer:"
    )


def pk_knows(response: str, answer: str) -> bool:
    rl = response.lower()
    abstains = any(p in rl for p in PK_ABSTAIN)
    return (answer.lower() in rl) and not abstains


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------

def _stratified_sample(indices, types, n, rng):
    """Sample n of `indices` stratified by type, natural proportions."""
    indices = np.asarray(indices)
    tarr = np.asarray([types[i] for i in indices])
    cats = sorted(set(tarr.tolist()))
    n = min(n, len(indices))
    props = {c: (tarr == c).mean() for c in cats}
    alloc = {c: int(np.floor(n * props[c])) for c in cats}
    rem = n - sum(alloc.values())
    for c in sorted(cats, key=lambda c: -(n * props[c] - alloc[c]))[:rem]:
        alloc[c] += 1
    out = []
    for c in cats:
        pool = indices[tarr == c]
        out.extend(rng.choice(pool, size=alloc[c], replace=False).tolist())
    rng.shuffle(out)
    return [int(i) for i in out]


def _bench_questions():
    s = set()
    for sp in ("train", "val", "test"):
        with open(BENCH_DIR / f"{sp}.json") as f:
            for ex in json.load(f):
                s.add(norm_question(ex["question"]))
    return s


class _Retriever:
    def __init__(self, docs, index_dir, prefer="pyserini", threads=4):
        """docs: list of (docid, title, text)."""
        self.docs = docs
        self.title_of = {d[0]: d[1] for d in docs}
        self.backend = None
        self.error = None
        if prefer == "pyserini":
            try:
                self._init_pyserini(index_dir, threads)
                self.backend = "pyserini-lucene-bm25(k1=0.9,b=0.4)"
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                print(f"[build] pyserini failed ({self.error}); falling back to rank_bm25")
        if self.backend is None:
            self._init_rankbm25()
            self.backend = "rank_bm25.BM25Okapi(k1=1.5,b=0.75)"

    def _init_pyserini(self, index_dir, threads):
        os.environ.setdefault("OPENAI_API_KEY", "unused")  # pyserini.encode builds a client at import
        if Path(JAVA21).exists():
            os.environ["JAVA_HOME"] = JAVA21           # jnius must start a JVM >= 21
        from pyserini.index.lucene import LuceneIndexer
        from pyserini.search.lucene import LuceneSearcher
        index_dir = Path(index_dir)
        marker = index_dir / "_ndocs.txt"   # index dir name already encodes the pool content hash
        if not (marker.exists() and int(marker.read_text()) == len(self.docs)):
            import shutil
            if index_dir.exists():
                shutil.rmtree(index_dir)
            index_dir.mkdir(parents=True)
            idx = LuceneIndexer(str(index_dir), threads=threads)
            batch = []
            for did, title, text in self.docs:
                batch.append({"id": did, "contents": f"{title} {text}"})
                if len(batch) == 5000:
                    idx.add_batch_dict(batch); batch = []
            if batch:
                idx.add_batch_dict(batch)
            idx.close()
            marker.write_text(str(len(self.docs)))
        self.searcher = LuceneSearcher(str(index_dir))
        self.threads = threads
        self.kind = "pyserini"

    def _init_rankbm25(self):
        from rank_bm25 import BM25Okapi
        self.tok = lambda s: re.findall(r"\w+", s.lower())
        self.bm25 = BM25Okapi([self.tok(f"{t} {x}") for _, t, x in self.docs])
        self.kind = "rank_bm25"

    def search(self, queries, k):
        """Returns list of [(title, score), ...] per query."""
        out = []
        if self.kind == "pyserini":
            qids = [str(i) for i in range(len(queries))]
            res = self.searcher.batch_search(queries, qids, k=k, threads=self.threads)
            for q in qids:
                out.append([(self.title_of[h.docid], float(h.score)) for h in res[q]])
        else:
            for q in queries:
                sc = self.bm25.get_scores(self.tok(q))
                top = np.argsort(-sc)[:k]
                out.append([(self.docs[i][1], float(sc[i])) for i in top])
        return out


def _make_context(hits_k, ptext):
    return "\n\n".join(f"{t}: {ptext[t]}" for t, _ in hits_k)


def cmd_build(a):
    from datasets import load_from_disk
    from src.data.build_sufficiency_bench import format_prompt
    from src.extraction.extract_states import format_question_only

    tag = a.tag or f"{a.retriever}_k{a.k}"
    out_dir = RC_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    ds = load_from_disk(str(DATASETS_DIR / "hotpotqa"))
    val, tr = ds["validation"], ds["train"]
    bench = _bench_questions()
    print(f"[build] SufficiencyBench questions to exclude: {len(bench)}")

    vq, vt = val["question"], val["type"]
    v_elig = [i for i, q in enumerate(vq) if norm_question(q) not in bench]
    test_idx = _stratified_sample(v_elig, vt, a.n_test, rng)
    test_norm = {norm_question(vq[i]) for i in test_idx}
    print(f"[build] validation eligible {len(v_elig)}/{len(vq)} (excluded {len(vq)-len(v_elig)}); "
          f"sampled test {len(test_idx)}")

    n_train_note = None
    if a.train_source == "validation":
        # train and test both from the eligible validation questions, disjoint (no level shift)
        test_set = set(test_idx)
        t_elig = [i for i in v_elig if i not in test_set and norm_question(vq[i]) not in test_norm]
        n_tr = a.n_train
        if n_tr > len(t_elig):
            n_train_note = (f"requested n_train={a.n_train} but only {len(t_elig)} eligible validation "
                            f"questions remain after the {len(test_idx)} test questions; using {len(t_elig)}")
            print(f"[build] NOTE: {n_train_note}")
            n_tr = len(t_elig)
        train_idx = _stratified_sample(t_elig, vt, n_tr, rng)
        tr_sel = val.select(train_idx)
        print(f"[build] train source = validation: eligible {len(t_elig)}; sampled train {len(train_idx)}")
    else:
        tq, tt = tr["question"], tr["type"]
        t_elig = [i for i, q in enumerate(tq)
                  if norm_question(q) not in bench and norm_question(q) not in test_norm]
        train_idx = _stratified_sample(t_elig, tt, a.n_train, rng)
        tr_sel = tr.select(train_idx)
        print(f"[build] train source = hotpot_train: eligible {len(t_elig)}/{len(tq)}; "
              f"sampled train {len(train_idx)}")

    # --- paragraph pool: all validation paragraphs (+ paragraphs of sampled HotpotQA-train questions)
    ptext = {}
    for ctx in val["context"]:
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title not in ptext:
                ptext[title] = clean_ws(" ".join(sents))
    if a.train_source == "hotpot_train":
        for ctx in tr_sel["context"]:
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                if title not in ptext:
                    ptext[title] = clean_ws(" ".join(sents))
    docs = [(f"d{i}", t, x) for i, (t, x) in enumerate(ptext.items())]
    print(f"[build] pool: {len(docs)} unique paragraphs (deduped by title)")

    if a.retriever != "bm25":
        raise SystemExit("only --retriever bm25 is implemented")
    import hashlib
    sig = hashlib.sha1("\n".join(f"{t}\t{x}" for _, t, x in docs).encode()).hexdigest()[:10]
    retr = _Retriever(docs, out_dir / f"index_{sig}", prefer="rank_bm25" if a.no_pyserini else "pyserini")
    print(f"[build] retriever backend: {retr.backend}")
    retr_index_dir = str((out_dir / f"index_{sig}").relative_to(ROOT)) if retr.kind == "pyserini" else None

    depth = max(N_RANK_DEPTH, a.k, max(K_REPORT))
    split_rows = {"test": val.select(test_idx), "train": tr_sel}
    report = {"tag": tag, "train_source": a.train_source, "n_train_note": n_train_note,
              "index_dir": retr_index_dir, "retriever": a.retriever, "backend": retr.backend,
              "pyserini_error": retr.error, "k": a.k, "seed": a.seed,
              "pool_size": len(docs), "n_bench_questions_excluded": len(bench),
              "n_validation_eligible": len(v_elig), "n_train_eligible": len(t_elig),
              "splits": {}}
    all_records = {}
    for split, rows in split_rows.items():
        qs = rows["question"]
        hits = retr.search(qs, depth)
        recs, pos_k = [], {k: [] for k in K_REPORT}
        for j, ex in enumerate(rows):
            gold = list(dict.fromkeys(ex["supporting_facts"]["title"]))
            h = hits[j]
            titles = [t for t, _ in h]
            gold_ranks = {g: (titles.index(g) + 1 if g in titles else None) for g in gold}
            for k in K_REPORT:
                pos_k[k].append(all(r is not None and r <= k for r in gold_ranks.values()))
            hk = h[:a.k]
            ctx = _make_context(hk, ptext)
            suff = all(r is not None and r <= a.k for r in gold_ranks.values())
            recs.append({
                "id": ex["id"],
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "prompt": format_prompt(ex["question"], ctx),
                "prompt_q_only": format_question_only(ex["question"]),
                # benchmark-format rendering of the same passages: no titles, single-space join
                "prompt_plain": format_prompt(ex["question"], clean_ws(" ".join(ptext[t] for t, _ in hk))),
                "sufficient": int(suff),
                "gold_ranks": gold_ranks,
                "answer_in_context": answer_in_text(ex["answer"], " ".join(ptext[t] for t, _ in hk)),
                "topk_titles": [t for t, _ in hk],
                "topk_scores": [s for _, s in hk],
                "context_chars": len(ctx),
            })
        all_records[split] = recs
        y = np.array([r["sufficient"] for r in recs])
        aic = [r["answer_in_context"] for r in recs]
        types = Counter(r["type"] for r in recs)
        report["splits"][split] = {
            "n": len(recs),
            "positive_rate": float(y.mean()),
            "positive_rate_by_k": {str(k): float(np.mean(pos_k[k])) for k in K_REPORT},
            "positive_rate_by_type_at_k": {
                t: float(np.mean([r["sufficient"] for r in recs if r["type"] == t])) for t in types},
            "type_distribution": dict(types),
            "level_distribution": dict(Counter(r["level"] for r in recs)),
            "yes_no_answers": int(sum(v is None for v in aic)),
            "answer_in_context_rate_sufficient": _safe_mean(
                [v for v, l in zip(aic, y) if v is not None and l == 1]),
            "answer_in_context_rate_insufficient": _safe_mean(
                [v for v, l in zip(aic, y) if v is not None and l == 0]),
            "mean_context_chars": float(np.mean([r["context_chars"] for r in recs])),
            "max_context_chars": int(max(r["context_chars"] for r in recs)),
        }
    report["build_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[build] positive rate (all gold titles in top-k), pool={len(docs)}, backend={retr.backend}")
    print("  k    " + "  ".join(f"{s:>7s}" for s in ("train", "test")))
    for k in K_REPORT:
        print(f"  {k:<4d} " + "  ".join(
            f"{report['splits'][s]['positive_rate_by_k'][str(k)]:7.3f}" for s in ("train", "test")))
    pr = report["splits"]["test"]["positive_rate"]
    if not (0.2 <= pr <= 0.8):
        raise SystemExit(
            f"[build] ABORT: test positive rate at k={a.k} is {pr:.3f}, outside [0.2, 0.8]. "
            f"Records NOT written; build_report.json written to {out_dir}. Choose k from the table above.")
    for split, recs in all_records.items():
        with open(out_dir / f"{split}.json", "w") as f:
            json.dump(recs, f, indent=1)
    print(f"[build] wrote {out_dir}/{{train,test}}.json and build_report.json "
          f"({report['build_seconds']} s)")


def _safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else None


# ----------------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------------

def _greedy_batch(model, tok, prompts, device, max_new_tokens, max_length):
    import torch
    inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_length).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, inp["input_ids"].shape[1]:]
    return [tok.decode(r, skip_special_tokens=True).strip() for r in gen]


def _paths(tag, model, split):
    hs = RC_DIR / tag / "hidden_states" / model / split
    res = RESULTS_DIR / model / f"retrieved_ctx_{tag}"
    return hs, res


def _check_pk(model, tok, recs, device, n):
    """Fidelity: batched PK vs the benchmark's own unbatched test_parametric_knowledge."""
    from src.data.build_sufficiency_bench import test_parametric_knowledge
    sub = recs[:n]
    ref = test_parametric_knowledge(
        model, tok, [{"id": r["id"], "question": r["question"], "answer": r["gold_answer"]} for r in sub],
        device=device)
    got = []
    for i in range(0, len(sub), GEN_BS):
        b = sub[i:i + GEN_BS]
        resp = _greedy_batch(model, tok, [pk_prompt(r["question"]) for r in b], device,
                             PK_MAX_NEW_TOKENS, PK_MAX_LEN)
        got += [(pk_knows(x, r["gold_answer"]), x) for x, r in zip(resp, b)]
    agree = sum(ref[r["id"]]["knows"] == g[0] for r, g in zip(sub, got))
    same_txt = sum(ref[r["id"]]["closed_book_answer"] == g[1][:300] for r, g in zip(sub, got))
    same_first = sum(first_sentence(ref[r["id"]]["closed_book_answer"]) == first_sentence(g[1]) for r, g in zip(sub, got))
    div = []
    for r, g in zip(sub, got):
        x, y = ref[r["id"]]["closed_book_answer"], g[1][:300]
        i = next((i for i, (c1, c2) in enumerate(zip(x, y)) if c1 != c2), min(len(x), len(y)))
        div.append(None if x == y else i)
    print(f"[check-pk] first-sentence identical {same_first}/{len(sub)}; divergence char offsets {div}")
    print(f"[check-pk] knows agreement {agree}/{len(sub)}; identical closed-book text {same_txt}/{len(sub)}; "
          f"knows rate ref={np.mean([ref[r['id']]['knows'] for r in sub]):.3f}")
    return {"n": len(sub), "knows_agree": agree, "text_identical": same_txt,
            "first_sentence_identical": same_first, "divergence_char_offsets": div,
            "examples": [{"ref": ref[r["id"]]["closed_book_answer"][:160], "batched": g[1][:160]}
                         for r, g in list(zip(sub, got))[:4]]}


def cmd_extract(a):
    import torch
    from src.extraction.extract_states import load_model, extract_last_token_hidden, MAX_SEQ_LEN
    from src.methods.baselines import verbalized_confidence_batch

    tag = a.tag
    with open(RC_DIR / tag / f"{a.split}.json") as f:
        recs = json.load(f)
    if a.limit:
        recs = recs[:a.limit]
    N = len(recs)
    hs_dir, res_dir = _paths(tag, a.model, a.split)
    parts = hs_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    scores_path = res_dir / f"{a.split}_scores.json"

    meta_path = hs_dir / "metadata.json"
    if meta_path.exists() and scores_path.exists() and (hs_dir / "h_with_context.npy").exists():
        with open(meta_path) as f:
            m = json.load(f)
        if m.get("n") == N and [r["id"] for r in m["records"]] == [r["id"] for r in recs]:
            print(f"[extract] {a.model}/{a.split}: final outputs exist for N={N}; skipping")
            return

    cfg = MODEL_CONFIGS[a.model]
    print(f"[extract] loading {a.model} ({cfg['path']}) on {a.device}")
    t_load = time.time()
    model, tok = load_model(cfg, a.device)
    tok.padding_side = "left"   # batched generation (VC/PK/answers) needs left padding
    print(f"[extract] loaded in {time.time()-t_load:.0f}s")
    torch.cuda.reset_peak_memory_stats(a.device)

    pk_check = None
    if a.check_pk:
        pk_check = _check_pk(model, tok, recs, a.device, a.check_pk)

    timing = []
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        stem = f"part_{start:06d}_{end:06d}"
        if (parts / f"{stem}.npz").exists() and (parts / f"{stem}.json").exists():
            with open(parts / f"{stem}.json") as f:
                ids_done = [r["id"] for r in json.load(f)["rows"]]
            if ids_done == [r["id"] for r in recs[start:end]]:
                print(f"[extract] {stem} exists, resuming past it")
                continue
            print(f"[extract] {stem} exists but its ids do not match the current records (data rebuilt?); redoing")
        t0 = time.time()
        chunk = recs[start:end]
        hqc = np.zeros((len(chunk), cfg["n_layers"], cfg["hidden_dim"]), np.float16)
        hq = np.zeros_like(hqc)
        hqcp = np.zeros_like(hqc)   # with-context, benchmark (plain) rendering
        rows = []
        t_fwd = t_vc = t_pk = t_ans = 0.0
        for b0 in range(0, len(chunk), GEN_BS):
            b = chunk[b0:b0 + GEN_BS]
            ts = time.time()
            ntok = []
            for j, r in enumerate(b):
                hqc[b0 + j] = extract_last_token_hidden(model, tok, r["prompt"], a.device).astype(np.float16)
                hq[b0 + j] = extract_last_token_hidden(model, tok, r["prompt_q_only"], a.device).astype(np.float16)
                hqcp[b0 + j] = extract_last_token_hidden(model, tok, r["prompt_plain"], a.device).astype(np.float16)
                ntok.append(len(tok(r["prompt"])["input_ids"]))
            torch.cuda.synchronize(a.device); t_fwd += time.time() - ts
            ts = time.time()
            vc = verbalized_confidence_batch(model, tok, [r["prompt"] for r in b], a.device)
            torch.cuda.synchronize(a.device); t_vc += time.time() - ts
            ts = time.time()
            cb = _greedy_batch(model, tok, [pk_prompt(r["question"]) for r in b], a.device,
                               PK_MAX_NEW_TOKENS, PK_MAX_LEN)
            torch.cuda.synchronize(a.device); t_pk += time.time() - ts
            ts = time.time()
            ans = _greedy_batch(model, tok, [r["prompt"] for r in b], a.device,
                                a.max_new_tokens, MAX_SEQ_LEN)
            torch.cuda.synchronize(a.device); t_ans += time.time() - ts
            for j, r in enumerate(b):
                line = first_line(ans[j])
                pred = first_sentence(ans[j])   # 24 greedy tokens run past the answer; score its first sentence
                rows.append({
                    "id": r["id"], "label": int(r["sufficient"]), "type": r["type"],
                    "level": r["level"], "vc": float(vc[j]),
                    # primary: gold in the FIRST sentence and no abstention phrase in that sentence
                    "pk": int(pk_knows(first_sentence(cb[j]), r["gold_answer"])),
                    "pk_benchmark_rule": int(pk_knows(cb[j], r["gold_answer"])),
                    "cb_answer": cb[j][:300],
                    "ctx_answer": pred, "ctx_answer_raw": ans[j][:300],
                    "em": squad_em(pred, r["gold_answer"]), "f1": squad_f1(pred, r["gold_answer"]),
                    "em_line": squad_em(line, r["gold_answer"]), "f1_line": squad_f1(line, r["gold_answer"]),
                    "abstained": int(_abstains(line)),
                    "contains_gold": int((not _abstains(line)) and
                                         f" {squad_normalize(r['gold_answer'])} " in f" {squad_normalize(line)} "),
                    "n_tokens_prompt": ntok[j], "truncated": int(ntok[j] > MAX_SEQ_LEN - 40),
                })
        dt = time.time() - t0
        timing.append({"part": stem, "n": len(chunk), "seconds": dt, "per_record": dt / len(chunk),
                       "fwd": t_fwd, "vc": t_vc, "pk_gen": t_pk, "ans_gen": t_ans})
        with open(parts / f"{stem}.npz.tmp", "wb") as f:
            np.savez(f, hqc=hqc, hq=hq, hqcp=hqcp)
        os.replace(parts / f"{stem}.npz.tmp", parts / f"{stem}.npz")
        with open(parts / f"{stem}.json.tmp", "w") as f:
            json.dump({"rows": rows, "timing": timing[-1]}, f)
        os.replace(parts / f"{stem}.json.tmp", parts / f"{stem}.json")
        mem = torch.cuda.max_memory_allocated(a.device) / 2**30
        print(f"[extract] {a.model}/{a.split} {end}/{N}  {dt/len(chunk):.2f}s/rec "
              f"(fwd {t_fwd/len(chunk):.2f}, vc {t_vc/len(chunk):.2f}, pk {t_pk/len(chunk):.2f}, "
              f"ans {t_ans/len(chunk):.2f})  peak alloc {mem:.1f} GiB", flush=True)
        torch.cuda.empty_cache()

    # ---- assemble
    H1, H2, H3, rows_all, timings = [], [], [], [], []
    for start in range(0, N, CHUNK):
        stem = f"part_{start:06d}_{min(start + CHUNK, N):06d}"
        z = np.load(parts / f"{stem}.npz")
        H1.append(z["hqc"]); H2.append(z["hq"]); H3.append(z["hqcp"])
        with open(parts / f"{stem}.json") as f:
            p = json.load(f)
        rows_all += p["rows"]; timings.append(p["timing"])
    assert [r["id"] for r in rows_all] == [r["id"] for r in recs], "part/record misalignment"
    np.save(hs_dir / "h_with_context.npy", np.concatenate(H1))
    np.save(hs_dir / "h_question_only.npy", np.concatenate(H2))
    np.save(hs_dir / "h_with_context_plain.npy", np.concatenate(H3))
    np.save(hs_dir / "labels.npy", np.array([r["label"] for r in rows_all]))
    peak = torch.cuda.max_memory_allocated(a.device) / 2**30
    peak_res = torch.cuda.max_memory_reserved(a.device) / 2**30
    tot_s = sum(t["seconds"] for t in timings)
    meta = {"model": a.model, "split": a.split, "tag": tag, "n": N, "dtype": "float16",
            "layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"],
            "quantization": "bitsandbytes 8-bit (extract_states.load_model)",
            "peak_mem_allocated_gib_this_session": peak, "peak_mem_reserved_gib_this_session": peak_res,
            "seconds_total_parts": tot_s, "seconds_per_record": tot_s / max(N, 1),
            "timing_parts": timings, "pk_fidelity_check": pk_check,
            "n_truncated": int(sum(r["truncated"] for r in rows_all)),
            "records": [{"id": r["id"], "label": r["label"], "type": r["type"],
                         "n_tokens_prompt": r["n_tokens_prompt"]} for r in rows_all]}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    with open(scores_path, "w") as f:
        json.dump(rows_all, f, indent=1)
    y = np.array([r["label"] for r in rows_all])
    print(f"[extract] done {a.model}/{a.split}: N={N}, pos={y.mean():.3f}, "
          f"PK={np.mean([r['pk'] for r in rows_all]):.3f} (benchmark rule {np.mean([r['pk_benchmark_rule'] for r in rows_all]):.3f}), "
          f"EM={np.mean([r['em'] for r in rows_all]):.3f}, F1={np.mean([r['f1'] for r in rows_all]):.3f}, "
          f"{tot_s/max(N,1):.2f}s/rec, peak alloc {peak:.1f} GiB / reserved {peak_res:.1f} GiB, "
          f"truncated {meta['n_truncated']}")


# ----------------------------------------------------------------------------
# analyse
# ----------------------------------------------------------------------------

def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _groups_desc(s):
    """Unique thresholds in descending order with group boundaries for sorted-desc arrays."""
    order = np.argsort(-s, kind="mergesort")
    ss = s[order]
    ends = np.r_[np.nonzero(np.diff(ss))[0] + 1, len(ss)]
    return order, ss, ends


def risk_coverage(y, s):
    """Decision 'sufficient' for covered (top-scored) items; risk = fraction of covered with y=0.
    Ties handled by expected value under random tie-breaking (AURC) and by whole-group
    thresholds (coverage at risk)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n = len(y)
    order, ss, ends = _groups_desc(s)
    err = 1 - y[order]
    aurc_sum, c0, e0, start = 0.0, 0, 0.0, 0
    pts = []
    for e in ends:
        m = e - start; eg = err[start:e].sum()
        j = np.arange(1, m + 1)
        aurc_sum += float(((e0 + j * eg / m) / (c0 + j)).sum())
        c0 += m; e0 += eg; start = e
        pts.append((c0 / n, e0 / c0, ss[e - 1]))
    out = {"aurc": aurc_sum / n}
    for r in (0.05, 0.10):
        ok = [c for c, rk, _ in pts if rk <= r + 1e-12]
        out[f"cov_at_risk_{int(r*100)}"] = max(ok) if ok else 0.0
    return out


def threshold_for_precision(y, s, target=0.9, min_cov=0.05):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    order, ss, ends = _groups_desc(s)
    yy = y[order]
    cum = np.cumsum(yy)
    best = None
    for e in ends:
        prec = cum[e - 1] / e
        if prec >= target:
            best = (ss[e - 1], e / len(y), prec)   # later groups = more coverage
    if best is not None:
        return {"threshold": float(best[0]), "train_cov": best[1], "train_prec": float(best[2]), "target_met": True}
    cand = [(cum[e - 1] / e, ss[e - 1], e / len(y)) for e in ends if e / len(y) >= min_cov]
    if not cand:
        return {"threshold": float("inf"), "train_cov": 0.0, "train_prec": None, "target_met": False}
    p, t, c = max(cand)
    return {"threshold": float(t), "train_cov": c, "train_prec": float(p), "target_met": False}


def _fit_lr(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # identical hyper-parameters to store_baseline_scores.compute_probe_scores
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, C=1.0, random_state=SEED_PROBE))
    clf.fit(Xtr, ytr)
    return clf


def _oof(X, y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    k = int(min(n_splits, np.bincount(y, minlength=2).min()))
    if k < 2:
        return np.full(len(y), 0.5)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED_PROBE).split(X, y):
        out[te] = _fit_lr(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def _qpp(recs):
    F = []
    for r in recs:
        s = np.asarray(r["topk_scores"], float)
        F.append([s[0], s[0] - (s[1] if len(s) > 1 else 0.0), s.mean(), s.std(), s.sum()])
    return np.asarray(F)


QPP_NAMES = ["qpp_top1", "qpp_gap12", "qpp_mean", "qpp_std_nqc", "qpp_sum"]


def _e5_scores(recs, cache_path, device):
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    todo = [r for r in recs if r["id"] not in cache]
    if todo:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(str(MODELS_DIR / "e5-base-v2"), device=device)
        qe = m.encode([f"query: {r['question']}" for r in todo], batch_size=64,
                      normalize_embeddings=True, show_progress_bar=False)
        ctx =[f"passage: {_ctx_from_prompt(r['prompt'])}" for r in todo]
        pe = m.encode(ctx, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        for r, a_, b_ in zip(todo, qe, pe):
            cache[r["id"]] = float(np.dot(a_, b_))
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return np.array([cache[r["id"]] for r in recs])


def _ctx_from_prompt(p):
    a = p.index("Context: ") + len("Context: ")
    b = p.rindex("\n\nQuestion: ")
    return p[a:b]


def _load_split(tag, model, split):
    hs_dir, res_dir = _paths(tag, model, split)
    with open(res_dir / f"{split}_scores.json") as f:
        sc = json.load(f)
    with open(RC_DIR / tag / f"{split}.json") as f:
        recs = {r["id"]: r for r in json.load(f)}
    recs = [recs[r["id"]] for r in sc]
    hqc = np.load(hs_dir / "h_with_context.npy", mmap_mode="r")
    hq = np.load(hs_dir / "h_question_only.npy", mmap_mode="r")
    pp = hs_dir / "h_with_context_plain.npy"
    hqcp = np.load(pp, mmap_mode="r") if pp.exists() else None
    assert hqc.shape[0] == len(sc)
    return sc, recs, hqc, hq, hqcp


def _fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


LABELS = ("support", "answer", "strict")
LABEL_DESC = {
    "support": "every supporting-fact title in the top-k",
    "answer": "normalised gold answer string in the concatenated top-k context (yes/no answers excluded)",
    "strict": "support label, restricted to questions where support and answer labels agree (yes/no excluded)",
}


def _label_arrays(recs, label):
    sup = np.array([r["sufficient"] for r in recs], int)
    aic = np.array([-1 if r["answer_in_context"] is None else int(r["answer_in_context"]) for r in recs])
    if label == "support":
        mask = np.ones(len(recs), bool); y = sup
    elif label == "answer":
        mask = aic >= 0; y = np.clip(aic, 0, 1)
    elif label == "strict":
        mask = (aic >= 0) & (aic == sup); y = sup
    else:
        raise ValueError(label)
    return mask, y[mask]


def _metrics_block(test_scores, train_scores, kinds, y_te, y_tr, pk_te, em_te, f1_te, acc_te, n_boot):
    from scipy.stats import kendalltau
    rng = np.random.default_rng(0)
    n = len(y_te)
    B = rng.integers(0, n, size=(n_boot, n))
    boot = {name: np.array([auroc(y_te[b], s[b]) for b in B]) for name, s in test_scores.items()}
    vc_b = boot["Verbalized confidence"]
    vc_auc = auroc(y_te, test_scores["Verbalized confidence"])
    em_all, f1_all, acc_all = float(em_te.mean()), float(f1_te.mean()), float(acc_te.mean())
    methods = {}
    for name, s in test_scores.items():
        auc = auroc(y_te, s)
        lo, hi = np.nanpercentile(boot[name], [2.5, 97.5])
        m = {"kind": kinds[name], "auroc": auc, "ci95": [float(lo), float(hi)],
             "auroc_pk0": auroc(y_te[pk_te == 0], s[pk_te == 0]) if (pk_te == 0).sum() >= 10 else None,
             "auroc_pk1": auroc(y_te[pk_te == 1], s[pk_te == 1]) if (pk_te == 1).sum() >= 10 else None,
             **risk_coverage(y_te, s)}
        tau, p_tau = kendalltau(s, f1_te)
        m["kendall_tau_f1"] = float(tau) if np.isfinite(tau) else None
        m["kendall_tau_f1_p"] = float(p_tau) if np.isfinite(p_tau) else None
        if name != "Verbalized confidence":
            d = boot[name] - vc_b
            d = d[np.isfinite(d)]
            m["delta_vs_vc"] = auc - vc_auc
            m["delta_vs_vc_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            m["delta_vs_vc_p"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        thr = threshold_for_precision(y_tr, train_scores[name])
        cov = s >= thr["threshold"]
        m["gating"] = {**thr, "test_coverage": float(cov.mean()),
                       "test_suff_precision": float(y_te[cov].mean()) if cov.any() else None,
                       "em_covered": float(em_te[cov].mean()) if cov.any() else None,
                       "f1_covered": float(f1_te[cov].mean()) if cov.any() else None,
                       "acc_contains_covered": float(acc_te[cov].mean()) if cov.any() else None,
                       "em_all": em_all, "f1_all": f1_all, "acc_contains_all": acc_all}
        methods[name] = m
    return methods


def cmd_analyse(a):
    from src.analysis.store_baseline_scores import load_best_layers

    tag, model = a.tag, a.model
    labels = LABELS if a.label == "all" else (a.label,)
    t0 = time.time()
    sc_tr, rec_tr, hqc_tr, hq_tr, hqcp_tr = _load_split(tag, model, "train")
    sc_te, rec_te, hqc_te, hq_te, hqcp_te = _load_split(tag, model, "test")
    deco_layer, std_layer = load_best_layers(model)
    print(f"[analyse] {model}: train N={len(sc_tr)}, test N={len(sc_te)}; "
          f"best layers standard={std_layer} CSP={deco_layer}; labels={labels}")

    f32 = lambda x: np.asarray(x, dtype=np.float32)
    def feat(key, hqc, hq):
        return f32(hqc[:, std_layer, :]) if key == "std" else f32(hqc[:, deco_layer, :]) - f32(hq[:, deco_layer, :])
    F = {(k, sp): feat(k, *hh) for k in ("std", "csp")
         for sp, hh in (("tr", (hqc_tr, hq_tr)), ("te", (hqc_te, hq_te)))}
    if hqcp_tr is not None and hqcp_te is not None:
        for k in ("std", "csp"):
            F[(k, "trp")] = feat(k, hqcp_tr, hq_tr); F[(k, "tep")] = feat(k, hqcp_te, hq_te)

    # ---- label-independent scores (computed once on all records)
    full_te, full_tr, kinds = {}, {}, {}
    bdir = HIDDEN_STATES_DIR / model / "train"
    b_hqc = np.load(bdir / "h_with_context.npy", mmap_mode="r")
    b_hq = np.load(bdir / "h_question_only.npy", mmap_mode="r")
    b_y = np.load(bdir / "labels.npy")
    for key, nm in (("std", "Standard probe"), ("csp", "CSP probe")):
        clf = _fit_lr(feat(key, b_hqc, b_hq), b_y)   # trained on SufficiencyBench train states
        full_te[f"{nm} (transfer)"] = clf.predict_proba(F[(key, "te")])[:, 1]
        full_tr[f"{nm} (transfer)"] = clf.predict_proba(F[(key, "tr")])[:, 1]
        kinds[f"{nm} (transfer)"] = "probe"
        if (key, "tep") in F:
            full_te[f"{nm} (transfer, plain ctx)"] = clf.predict_proba(F[(key, "tep")])[:, 1]
            full_tr[f"{nm} (transfer, plain ctx)"] = clf.predict_proba(F[(key, "trp")])[:, 1]
            kinds[f"{nm} (transfer, plain ctx)"] = "probe"
    del b_hqc, b_hq
    full_te["Verbalized confidence"] = np.array([r["vc"] for r in sc_te], float)
    full_tr["Verbalized confidence"] = np.array([r["vc"] for r in sc_tr], float)
    kinds["Verbalized confidence"] = "baseline"
    full_te["Model answers (no abstention)"] = 1.0 - np.array([r["abstained"] for r in sc_te], float)
    full_tr["Model answers (no abstention)"] = 1.0 - np.array([r["abstained"] for r in sc_tr], float)
    kinds["Model answers (no abstention)"] = "baseline"
    Q_tr, Q_te = _qpp(rec_tr), _qpp(rec_te)
    for i, nm in enumerate(QPP_NAMES):
        full_te[nm] = Q_te[:, i]; full_tr[nm] = Q_tr[:, i]; kinds[nm] = "qpp"
    try:
        e5dir = RC_DIR / tag / "e5_cache"; e5dir.mkdir(exist_ok=True)
        full_te["e5 cosine(q, ctx)"] = _e5_scores(rec_te, e5dir / "test.json", a.e5_device)
        full_tr["e5 cosine(q, ctx)"] = _e5_scores(rec_tr, e5dir / "train.json", a.e5_device)
        kinds["e5 cosine(q, ctx)"] = "baseline"
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] e5 failed: {e}")
    order = ["Standard probe (transfer)", "CSP probe (transfer)",
             "Standard probe (transfer, plain ctx)", "CSP probe (transfer, plain ctx)",
             "Standard probe (retrained)", "CSP probe (retrained)", "Verbalized confidence",
             "Model answers (no abstention)", *QPP_NAMES, "qpp_logreg(5 feats)", "e5 cosine(q, ctx)"]

    pk_all = np.array([r["pk"] for r in sc_te]); pkb_all = np.array([r["pk_benchmark_rule"] for r in sc_te])
    em_all = np.array([r["em"] for r in sc_te], float); f1_all = np.array([r["f1"] for r in sc_te], float)
    acc_all = np.array([r["contains_gold"] for r in sc_te], float)
    abst_all = np.array([r["abstained"] for r in sc_te], float)
    res_dir = RESULTS_DIR / model / f"retrieved_ctx_{tag}"
    res_dir.mkdir(parents=True, exist_ok=True)
    outs, mds = {}, []
    for lab in labels:
        m_tr, y_tr = _label_arrays(rec_tr, lab)
        m_te, y_te = _label_arrays(rec_te, lab)
        # drop prompts that hit the 2048-token truncation (right-truncation cuts the question)
        keep_tr = np.array([not r.get("truncated", 0) for r in sc_tr])
        keep_te = np.array([not r.get("truncated", 0) for r in sc_te])
        y_tr = y_tr[keep_tr[m_tr]]; m_tr = m_tr & keep_tr
        y_te = y_te[keep_te[m_te]]; m_te = m_te & keep_te
        te = {k: v[m_te] for k, v in full_te.items()}
        tr = {k: v[m_tr] for k, v in full_tr.items()}
        kd = dict(kinds)
        for key, nm in (("std", "Standard probe (retrained)"), ("csp", "CSP probe (retrained)")):
            Xtr, Xte = F[(key, "tr")][m_tr], F[(key, "te")][m_te]
            te[nm] = _fit_lr(Xtr, y_tr).predict_proba(Xte)[:, 1]
            tr[nm] = _oof(Xtr, y_tr)                   # out-of-fold, threshold selection only
            kd[nm] = "probe"
        te["qpp_logreg(5 feats)"] = _fit_lr(Q_tr[m_tr], y_tr).predict_proba(Q_te[m_te])[:, 1]
        tr["qpp_logreg(5 feats)"] = _oof(Q_tr[m_tr], y_tr); kd["qpp_logreg(5 feats)"] = "qpp"
        te = {k: te[k] for k in order if k in te}
        sweep = {}
        if not a.no_sweep:
            idx_tr, idx_te = np.nonzero(m_tr)[0], np.nonzero(m_te)[0]
            for L in range(hqc_tr.shape[1]):
                s = _fit_lr(f32(hqc_tr[idx_tr, L, :]), y_tr).predict_proba(f32(hqc_te[idx_te, L, :]))[:, 1]
                sweep[str(L)] = auroc(y_te, s)
        pk_te = pk_all[m_te]
        methods = _metrics_block(te, tr, kd, y_te, y_tr, pk_te, em_all[m_te], f1_all[m_te],
                                 acc_all[m_te], a.n_boot)
        aic = np.array([np.nan if r["answer_in_context"] is None else float(r["answer_in_context"])
                        for r in rec_te])[m_te]
        ok = np.isfinite(aic)
        oracle = {"n_non_yesno": int(ok.sum()),
                  "auroc_answer_in_context": auroc(y_te[ok], aic[ok]) if lab == "support" else None,
                  "rate_given_sufficient": _safe_mean(aic[ok & (y_te == 1)].tolist()),
                  "rate_given_insufficient": _safe_mean(aic[ok & (y_te == 0)].tolist())}
        sup_te = np.array([r["sufficient"] for r in rec_te])
        out = {"model": model, "tag": tag, "label": lab, "label_definition": LABEL_DESC[lab],
               "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
               "n_train_total": len(sc_tr), "n_test_total": len(sc_te),
               "n_truncated_dropped": {"train": int((~keep_tr).sum()), "test": int((~keep_te).sum())},
               "pos_rate_train": float(y_tr.mean()), "pos_rate_test": float(y_te.mean()),
               "support_answer_agreement_test": float(np.mean(
                   [(r["answer_in_context"] == bool(r["sufficient"])) for r in rec_te
                    if r["answer_in_context"] is not None])),
               "best_layers": {"standard": int(std_layer), "csp": int(deco_layer)},
               "pk_rate_test": float(pk_te.mean()), "pk_benchmark_rule_rate_test": float(pkb_all[m_te].mean()),
               "n_pk1_test": int(pk_te.sum()),
               "em_all": float(em_all[m_te].mean()), "f1_all": float(f1_all[m_te].mean()),
               "acc_contains_all": float(acc_all[m_te].mean()), "abstention_rate": float(abst_all[m_te].mean()),
               "em_by_label": {"sufficient": _safe_mean(em_all[m_te][y_te == 1].tolist()),
                               "insufficient": _safe_mean(em_all[m_te][y_te == 0].tolist())},
               "support_label_rate_in_subset": float(sup_te[m_te].mean()),
               "n_boot": a.n_boot, "methods": methods, "oracle_answer_in_context": oracle,
               "layer_sweep_retrained_standard": sweep,
               "notes": ["transfer probes: trained on SufficiencyBench train states (benchmark labels), label-independent",
                         "'plain ctx' rows: same transfer probes on the same passages rendered in benchmark format "
                         "(no titles, single-space join, whitespace collapsed)",
                         "retrained probes / qpp_logreg are trained on the retrieved train split with THIS label; "
                         "their train scores are 5-fold out-of-fold (threshold selection only)",
                         "gating threshold: max-coverage threshold with train precision >= 0.90 on 'sufficient'",
                         "pk = gold answer in the first sentence of the closed-book generation and no abstention "
                         "phrase in that sentence; pk_benchmark_rule = original rule over the whole generation",
                         "em/f1 = SQuAD EM/F1 of the first sentence of the 24-token greedy answer; "
                         "acc_contains = normalised gold in first line and no abstention"]}
        with open(res_dir / f"analysis_{lab}.json", "w") as f:
            json.dump(out, f, indent=2)
        md = _analysis_md(out)
        (res_dir / f"analysis_{lab}.md").write_text(md)
        outs[lab] = out; mds.append(md)
        print(md)
    if len(labels) > 1:
        with open(res_dir / "analysis.json", "w") as f:
            json.dump(outs, f, indent=2)
        (res_dir / "analysis.md").write_text("\n".join(mds))
    print(f"[analyse] done in {time.time()-t0:.0f}s -> {res_dir}")


def _analysis_md(o):
    L = [f"## {o['model']} ({o['tag']}), label = {o['label']}", "",
         f"Label: {o['label_definition']}. Test N={o['n_test']}/{o['n_test_total']} (sufficient {o['pos_rate_test']:.3f}); "
         f"train N={o['n_train']}/{o['n_train_total']} (sufficient {o['pos_rate_train']:.3f}). "
         f"PK rate {o['pk_rate_test']:.3f} (n PK=1: {o['n_pk1_test']}; benchmark rule {o['pk_benchmark_rule_rate_test']:.3f}). "
         f"With-context EM {o['em_all']:.3f}, F1 {o['f1_all']:.3f}, contains-gold {o['acc_contains_all']:.3f}, "
         f"abstention {o['abstention_rate']:.3f}. Best layers std={o['best_layers']['standard']} "
         f"CSP={o['best_layers']['csp']}; bootstrap B={o['n_boot']}.", "",
         "| Method | AUROC [95% CI] | AUROC PK=0 | AUROC PK=1 | AURC | Cov@risk5% | Cov@risk10% | tau(F1) "
         "| dAUROC vs VC [95% CI], p | Gate cov | Gate EM (all) | Gate contains-acc (all) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in o["methods"].items():
        d = ("--" if "delta_vs_vc" not in m else
             f"{m['delta_vs_vc']:+.3f} [{m['delta_vs_vc_ci95'][0]:+.3f}, {m['delta_vs_vc_ci95'][1]:+.3f}], p={m['delta_vs_vc_p']:.3f}")
        g = m["gating"]
        L.append(f"| {name} | {_fmt(m['auroc'])} [{_fmt(m['ci95'][0])}, {_fmt(m['ci95'][1])}] | "
                 f"{_fmt(m['auroc_pk0'])} | {_fmt(m['auroc_pk1'])} | {_fmt(m['aurc'])} | "
                 f"{_fmt(m['cov_at_risk_5'])} | {_fmt(m['cov_at_risk_10'])} | {_fmt(m['kendall_tau_f1'])} | {d} | "
                 f"{_fmt(g['test_coverage'])}{'' if g['target_met'] else '*'} | "
                 f"{_fmt(g['em_covered'])} ({_fmt(g['em_all'])}) | "
                 f"{_fmt(g['acc_contains_covered'])} ({_fmt(g['acc_contains_all'])}) |")
    oc = o["oracle_answer_in_context"]
    L += ["", "\\* train precision 0.90 not attainable; threshold = max train precision at >= 5% coverage. "
          "PK strata need >= 10 questions.",
          f"Answer string in retrieved context ({oc['n_non_yesno']} non-yes/no): rate | sufficient "
          f"{_fmt(oc['rate_given_sufficient'])}, | insufficient {_fmt(oc['rate_given_insufficient'])}"
          + (f"; oracle AUROC {_fmt(oc['auroc_answer_in_context'])} (reference, not a method)."
             if oc["auroc_answer_in_context"] is not None else ".")]
    if o["layer_sweep_retrained_standard"]:
        sw = o["layer_sweep_retrained_standard"]
        L += ["", "Layer sweep (retrained Standard, test AUROC): " +
              ", ".join(f"L{k}={v:.3f}" for k, v in sw.items())]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# summarise
# ----------------------------------------------------------------------------

def cmd_summarise(a):
    tag = a.tag
    labels = LABELS if a.label == "all" else (a.label,)
    L = [f"# Retrieved-context evaluation summary ({tag})", ""]
    summ = {"tag": tag, "labels": {}}
    for lab in labels:
        res = {}
        for m in ("llama", "mistral", "qwen"):
            p = RESULTS_DIR / m / f"retrieved_ctx_{tag}" / f"analysis_{lab}.json"
            if p.exists():
                with open(p) as f:
                    res[m] = json.load(f)
        if not res:
            print(f"[summarise] no analysis_{lab}.json for tag {tag}; skipping label {lab}")
            continue
        models = list(res)
        names = list(dict.fromkeys(n for m in models for n in res[m]["methods"]))
        L += [f"## Label = {lab}: {LABEL_DESC[lab]}", "",
              "| Model | N test | sufficient | PK rate (n PK=1) | EM | F1 | contains-acc |",
              "|---|---|---|---|---|---|---|"]
        for m in models:
            o = res[m]
            L.append(f"| {m} | {o['n_test']} | {o['pos_rate_test']:.3f} | {o['pk_rate_test']:.3f} ({o['n_pk1_test']}) | "
                     f"{o['em_all']:.3f} | {o['f1_all']:.3f} | {o['acc_contains_all']:.3f} |")
        L += ["", "AUROC [95% CI]; dAUROC vs verbalized confidence (p) underneath.", "",
              "| Method | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        s_lab = {"models": models, "per_model": {}, "methods": {}}
        for n in names:
            cells = []
            s_lab["methods"][n] = {}
            for m in models:
                mm = res[m]["methods"].get(n)
                if mm is None:
                    cells.append("--"); continue
                s_lab["methods"][n][m] = {k: mm.get(k) for k in (
                    "auroc", "ci95", "auroc_pk0", "auroc_pk1", "aurc", "cov_at_risk_5", "cov_at_risk_10",
                    "delta_vs_vc", "delta_vs_vc_ci95", "delta_vs_vc_p", "kendall_tau_f1", "gating")}
                c = f"{_fmt(mm['auroc'])} [{_fmt(mm['ci95'][0])}, {_fmt(mm['ci95'][1])}]"
                if "delta_vs_vc" in mm:
                    c += f"<br>{mm['delta_vs_vc']:+.3f} (p={mm['delta_vs_vc_p']:.3f})"
                cells.append(c)
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L += ["", "Gating (train threshold for precision 0.90): test coverage / EM covered (EM all)", "",
              "| Method | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for n in names:
            cells = []
            for m in models:
                mm = res[m]["methods"].get(n)
                g = mm["gating"] if mm else None
                cells.append("--" if g is None else
                             f"{_fmt(g['test_coverage'])}{'' if g['target_met'] else '*'} / "
                             f"{_fmt(g['em_covered'])} ({_fmt(g['em_all'])})")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L.append("")
        for m in models:
            s_lab["per_model"][m] = {k: res[m][k] for k in (
                "n_test", "n_train", "pos_rate_test", "pk_rate_test", "n_pk1_test", "em_all", "f1_all",
                "acc_contains_all", "best_layers", "oracle_answer_in_context")}
        summ["labels"][lab] = s_lab
    if not summ["labels"]:
        raise SystemExit(f"no analysis files found for tag {tag}")
    out = RESULTS_DIR / f"retrieved_ctx_{tag}_summary"
    with open(f"{out}.json", "w") as f:
        json.dump(summ, f, indent=2)
    Path(f"{out}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--retriever", default="bm25", choices=["bm25"])
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--n-train", type=int, default=3000)
    b.add_argument("--n-test", type=int, default=1500)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tag", default=None, help="default: <retriever>_k<k>")
    b.add_argument("--no-pyserini", action="store_true", help="force the rank_bm25 fallback")
    b.add_argument("--train-source", default="validation", choices=["validation", "hotpot_train"],
                   help="validation: train and test disjoint samples of eligible HotpotQA validation questions")
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True, choices=["llama", "mistral", "qwen"])
    e.add_argument("--split", required=True, choices=["train", "test"])
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--tag", default="bm25_k5")
    e.add_argument("--max-new-tokens", type=int, default=24, help="with-context answer generation")
    e.add_argument("--check-pk", type=int, default=0,
                   help="verify batched PK against build_sufficiency_bench.test_parametric_knowledge on N records")
    n = sub.add_parser("analyse")
    n.add_argument("--model", required=True, choices=["llama", "mistral", "qwen"])
    n.add_argument("--tag", default="bm25_k5")
    n.add_argument("--n-boot", type=int, default=2000)
    n.add_argument("--no-sweep", action="store_true")
    n.add_argument("--e5-device", default="cpu")
    n.add_argument("--label", default="all", choices=["all", *LABELS])
    s = sub.add_parser("summarise")
    s.add_argument("--tag", default="bm25_k5")
    s.add_argument("--label", default="all", choices=["all", *LABELS])
    a = ap.parse_args()
    {"build": cmd_build, "extract": cmd_extract, "analyse": cmd_analyse, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
