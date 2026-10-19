#!/usr/bin/env python
"""Document-swap control for SufficiencyBench (test split).

Builds a "natural-data style" counterpart where the insufficient context is simply the SUFFICIENT context of a
different test question (BM25-nearest, gold answer absent, not same source), then scores the same predictors on the
matched (original) and swap sets. Subcommands:
  build    -> data/experiments/sufficiency_bench_docswap/test.json (+ build_stats.json)
  vc       --model M : verbalized confidence on swap prompts (raw prompt, baselines.verbalized_confidence_batch)
  extract  --model M : last-token hidden states of swap prompts (whitespace-normalised, as the stored train/test states)
  report   : AUROC + question-clustered bootstrap CIs -> results/docswap_control{,_limitN}.json/.md
--limit N restricts vc/extract/report to the first N questions and writes *_limitN outputs.
"""
import argparse, json, os, re, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from configs.paths import BENCH_DIR, EXPERIMENTS_DIR, HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS, MINILM_PATH

SWAP_DIR = EXPERIMENTS_DIR / "sufficiency_bench_docswap"
HS_SWAP = EXPERIMENTS_DIR / "hidden_states_docswap"
OUT_DIR = RESULTS_DIR / "docswap_control"
MODELS = ["llama", "mistral", "qwen"]
B = 1000


def format_prompt(question, context):  # identical to src/data/build_sufficiency_bench.py
    return (f"Based on the following context, answer the question. "
            f"If the context does not contain enough information, "
            f"say 'I cannot answer this from the provided context.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\nAnswer:")


def norm_ws(t):
    return re.sub(r"\s+", " ", t).strip()


def sfx(limit):
    return f"_limit{limit}" if limit else ""


# ------------------------------------------------------------------ build
def sentences(t):
    return {s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) >= 40}


def build():
    from rank_bm25 import BM25Okapi
    data = json.load(open(BENCH_DIR / "test.json"))
    suf = [next(c for c in ex["conditions"] if c["sufficient"]) for ex in data]
    tok = lambda s: re.findall(r"\w+", s.lower())
    bm = BM25Okapi([tok(c["context"]) for c in suf])
    sents = [sentences(c["context"]) for c in suf]
    out, stats = [], dict(n=len(data), top1_rejected_answer=0, top1_rejected_source=0, top1_rejected_any=0,
                          rejected_answer_total=0, rejected_source_total=0, rank_chosen=[], no_candidate=0)
    for i, ex in enumerate(data):
        scores = bm.get_scores(tok(ex["question"] + " " + suf[i]["context"]))
        order = np.argsort(-scores)
        gold = (ex["gold_answer"] or "").strip().lower()
        base = ex["conditions"][0]["condition_id"].rsplit("_", 1)[0]
        chosen, rank, top1_flags = None, 0, None
        for r, j in enumerate([int(k) for k in order if int(k) != i]):
            ans = bool(gold) and gold in suf[j]["context"].lower()
            src = (data[j]["id"] == ex["id"]) or bool(sents[i] & sents[j])
            if r == 0:
                top1_flags = (ans, src)
            stats["rejected_answer_total"] += int(ans); stats["rejected_source_total"] += int(src and not ans)
            if not ans and not src:
                chosen, rank = j, r; break
        if chosen is None:
            stats["no_candidate"] += 1; raise RuntimeError(f"no valid swap for {ex['id']}")
        stats["top1_rejected_answer"] += int(top1_flags[0]); stats["top1_rejected_source"] += int(top1_flags[1] and not top1_flags[0])
        stats["top1_rejected_any"] += int(top1_flags[0] or top1_flags[1])
        stats["rank_chosen"].append(rank)
        sc, oc = suf[i], next(c for c in ex["conditions"] if not c["sufficient"])
        ctx = suf[chosen]["context"]
        ins = dict(oc, context=ctx, context_length=len(ctx), prompt=format_prompt(ex["question"], ctx),
                   swap_source_id=data[chosen]["id"], swap_source_condition=suf[chosen]["condition_id"])
        out.append(dict(ex, conditions=[dict(sc), ins]))
    rk = np.array(stats.pop("rank_chosen"))
    L = lambda cs: float(np.mean([len(c) for c in cs]))
    stats.update(mean_rank_chosen=float(rk.mean()), frac_rank0=float((rk == 0).mean()),
                 mean_len_swap_ctx=L([e["conditions"][1]["context"] for e in out]),
                 mean_len_matched_insuf_ctx=L([next(c for c in ex["conditions"] if not c["sufficient"])["context"] for ex in data]),
                 mean_len_sufficient_ctx=L([e["conditions"][0]["context"] for e in out]),
                 n_distinct_swap_sources=len({e["conditions"][1]["swap_source_id"] for e in out}))
    SWAP_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(SWAP_DIR / "test.json", "w"), indent=1)
    json.dump(stats, open(SWAP_DIR / "build_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def load_pair():
    orig = json.load(open(BENCH_DIR / "test.json")); swap = json.load(open(SWAP_DIR / "test.json"))
    return orig, swap


# ------------------------------------------------------------------ vc
def run_vc(model_key, device, limit, bs):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.methods.baselines import verbalized_confidence_batch, _quant_kwargs
    out = OUT_DIR / f"vc_{model_key}{sfx(limit)}.json"
    _, swap = load_pair()
    swap = swap[:limit] if limit else swap
    conds = [(c["condition_id"], c["prompt"], int(c["sufficient"])) for ex in swap for c in ex["conditions"]]
    cfg = MODEL_CONFIGS[model_key]
    tk = AutoTokenizer.from_pretrained(cfg["path"]); tk.padding_side = "left"
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    m = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.float16, device_map=device, **_quant_kwargs(cfg)).eval()
    sc = []
    for i in range(0, len(conds), bs):
        sc += verbalized_confidence_batch(m, tk, [c[1] for c in conds[i:i + bs]], device)
        if (i // bs) % 25 == 0: print(f"vc {model_key} {i}/{len(conds)}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"ids": [c[0] for c in conds], "labels": [c[2] for c in conds], "scores": sc}, open(out, "w"))
    print("saved", out)


# ------------------------------------------------------------------ extract
def run_extract(model_key, device, limit):
    from src.extraction.extract_states import load_model, extract_last_token_hidden
    _, swap = load_pair()
    swap = swap[:limit] if limit else swap
    conds = [c for ex in swap for c in ex["conditions"]]
    d = HS_SWAP / model_key / f"test{sfx(limit)}"; d.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(MODEL_CONFIGS[model_key], device)
    H = None; t0 = time.time()
    for i, c in enumerate(conds):
        ex = swap[i // 2]
        p = format_prompt(ex["question"], norm_ws(c["context"]))
        h = extract_last_token_hidden(model, tok, p, device)
        if H is None: H = np.zeros((len(conds),) + h.shape, dtype=np.float16)
        H[i] = h.astype(np.float16)
        if i % 200 == 0: print(f"extract {model_key} {i}/{len(conds)} {time.time()-t0:.0f}s", flush=True)
    np.save(d / "h_with_context.npy", H)
    json.dump([c["condition_id"] for c in conds], open(d / "ids.json", "w"))
    json.dump({"note": "swap-set prompts, context whitespace-normalised (matches stored hidden_states), h(q) reused from original test"}, open(d / "PROVENANCE.json", "w"))
    print("saved", d)


# ------------------------------------------------------------------ report
def fast_auc(y, s):
    from scipy.stats import rankdata
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(sa, sb, nq, strata_mask, rng):
    """sa, sb: (nq,2) score matrices [suf, insuf] (sb may be None). Question-clustered bootstrap within mask.
    Returns point(a), ci(a), point(b), ci(b), point(diff b-a), ci(diff)."""
    idx = np.where(strata_mask)[0]
    y = np.tile([1, 0], len(idx))
    def au(s, ii): return fast_auc(y, s[ii].reshape(-1))
    pa, pb = au(sa, idx), au(sb, idx)
    A, Bv = [], []
    for _ in range(B):
        ii = rng.choice(idx, len(idx), replace=True)
        A.append(au(sa, ii)); Bv.append(au(sb, ii))
    A, Bv = np.array(A), np.array(Bv); D = Bv - A
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return dict(matched=dict(auroc=float(pa), ci=ci(A)), swap=dict(auroc=float(pb), ci=ci(Bv)),
                delta_swap_minus_matched=dict(value=float(pb - pa), ci=ci(D)), n_questions=int(len(idx)))


def cv_length(lens_flat, groups, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_predict(clf, np.asarray(lens_flat, float)[:, None], y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]


def probe_scores(model_key, hs_swap_path, layers, nq):
    from methods.deco import make_classifier
    base = HIDDEN_STATES_DIR / model_key
    hq_tr = np.load(base / "train" / "h_question_only.npy", mmap_mode="r")
    hc_tr = np.load(base / "train" / "h_with_context.npy", mmap_mode="r")
    y_tr = np.load(base / "train" / "labels.npy")
    hc_te = np.load(base / "test" / "h_with_context.npy", mmap_mode="r")
    hq_te = np.load(base / "test" / "h_question_only.npy", mmap_mode="r")
    hc_sw = np.load(hs_swap_path, mmap_mode="r")
    n = 2 * nq
    res = {}
    for name, L in layers.items():
        if name == "standard":
            f = lambda hc, hq: np.asarray(hc[:, L, :], np.float32)
        else:
            f = lambda hc, hq: np.asarray(hc[:, L, :], np.float32) - np.asarray(hq[:, L, :], np.float32)
        clf = make_classifier("logreg"); clf.fit(f(hc_tr, hq_tr), y_tr)
        # swap sets reuse the original test h(q) (same questions, same order)
        res[name] = (clf.predict_proba(f(hc_te[:n], hq_te[:n]))[:, 1].reshape(nq, 2),
                     clf.predict_proba(f(hc_sw[:n], hq_te[:n]))[:, 1].reshape(nq, 2))
    return res


def report(limit, do_emb=True):
    from sklearn.metrics import roc_auc_score
    orig, swap = load_pair()
    if limit: orig, swap = orig[:limit], swap[:limit]
    nq = len(orig)
    for ex, sw in zip(orig, swap):
        assert [c["condition_id"] for c in ex["conditions"]] == [c["condition_id"] for c in sw["conditions"]] and ex["conditions"][0]["sufficient"]
    rng = np.random.default_rng(0)
    y = np.tile([1, 0], nq); groups = np.repeat(np.arange(nq), 2)
    lens = lambda d: np.array([[len(c["context"]) for c in ex["conditions"]] for ex in d], float)
    Lo, Ls = lens(orig), lens(swap)
    lo = cv_length(Lo.reshape(-1), groups, y).reshape(nq, 2); ls = cv_length(Ls.reshape(-1), groups, y).reshape(nq, 2)
    emb = None
    if do_emb:
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(str(MINILM_PATH), device="cpu")
            def cos(d):
                q = st.encode([ex["question"] for ex in d], normalize_embeddings=True, show_progress_bar=False)
                c = st.encode([c["context"] for ex in d for c in ex["conditions"]], batch_size=64, normalize_embeddings=True, show_progress_bar=False).reshape(nq, 2, -1)
                return (c * q[:, None, :]).sum(-1)
            emb = (cos(orig), cos(swap))
        except Exception as e:
            print("embedding baseline skipped:", e)
    allq = np.ones(nq, bool)
    out = {"B": B, "n_questions": nq, "limit": limit, "models": {},
           "construction": json.load(open(SWAP_DIR / "build_stats.json")),
           "mean_context_chars_report_subset": {"swap_insufficient": float(Ls[:, 1].mean()), "matched_insufficient": float(Lo[:, 1].mean()), "sufficient": float(Lo[:, 0].mean())}}
    for mk in MODELS:
        pk = json.load(open(BENCH_DIR / f"pk_cache_{mk}.json"))
        pk0 = np.array([not pk[ex["id"]].get("knows") for ex in orig])
        pred = {"length_only": (lo, ls)}
        if emb: pred["embedding_cosine"] = emb
        vcf = OUT_DIR / f"vc_{mk}{sfx(limit)}.json"
        if vcf.exists():
            v = json.load(open(vcf)); sv = np.array(v["scores"]).reshape(nq, 2)
            st = json.load(open(RESULTS_DIR / mk / "scores" / "baseline_scores.json"))["verbalized_confidence"]
            so = np.array(st["scores"][:2 * nq]).reshape(nq, 2)
            pred["verbalized_confidence"] = (so, sv)
            out.setdefault("sanity", {})[mk] = {"vc_sufficient_scores_identical_to_stored": float(np.mean(sv[:, 0] == so[:, 0]))}
        hsf = HS_SWAP / mk / f"test{sfx(limit)}" / "h_with_context.npy"
        if hsf.exists():
            best = json.load(open(RESULTS_DIR / mk / "all_results_logreg.json"))["best_layers"]
            ps = probe_scores(mk, hsf, {"standard": best["standard"], "csp": best["DECO"]}, nq)
            pred["standard_probe"], pred["csp_probe"] = ps["standard"], ps["csp"]
            out.setdefault("sanity", {}).setdefault(mk, {})["layers"] = {"standard": best["standard"], "csp": best["DECO"]}
            sp = json.load(open(RESULTS_DIR / mk / "scores" / "probe_scores.json"))
            if "CSP (LogReg)" in sp:
                stored = np.array(sp["CSP (LogReg)"]["scores"][:2 * nq]).reshape(nq, 2)
                out["sanity"][mk]["csp_matched_auroc_stored_scores"] = float(fast_auc(y, stored.reshape(-1)))
        me = {}
        for pn, (a, b) in pred.items():
            me[pn] = {"overall": boot(a, b, nq, allq, rng), "pk0": boot(a, b, nq, pk0, rng)}
            print(mk, pn, me[pn]["overall"]["matched"]["auroc"], me[pn]["overall"]["swap"]["auroc"], flush=True)
        out["models"][mk] = me
    name = f"docswap_control{sfx(limit)}"
    json.dump(out, open(RESULTS_DIR / f"{name}.json", "w"), indent=1)
    lines = [f"# Document-swap control (n_questions={nq}, B={B}, question-clustered bootstrap, 95% CI)", "",
             "AUROC [CI]. Matched = original SufficiencyBench test; Swap = insufficient context replaced by a BM25-nearest other-question sufficient context. "
             "Delta = swap - matched (paired bootstrap).", "",
             "| Model | Predictor | Stratum | Matched | Swap | Delta (swap - matched) |", "|---|---|---|---|---|---|"]
    f = lambda d: f"{d['auroc']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
    for mk, me in out["models"].items():
        for pn, r in me.items():
            for sn in ("overall", "pk0"):
                x = r[sn]; dd = x["delta_swap_minus_matched"]
                lines.append(f"| {mk} | {pn} | {sn} (n={x['n_questions']}q) | {f(x['matched'])} | {f(x['swap'])} | {dd['value']:+.3f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}] |")
    c = out["construction"]
    lines += ["", "## Construction", "", "```", json.dumps(c, indent=1), "```", "", f"Mean chars (report subset): {out['mean_context_chars_report_subset']}"]
    open(RESULTS_DIR / f"{name}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "vc", "extract", "report"])
    ap.add_argument("--model", default="llama"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-embedding", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build": build()
    elif a.cmd == "vc": run_vc(a.model, a.device, a.limit, min(a.batch_size, 8))
    elif a.cmd == "extract": run_extract(a.model, a.device, a.limit)
    else: report(a.limit, not a.no_embedding)
