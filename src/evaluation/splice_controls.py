"""
splice_controls.py - two reviewer controls on SufficiencyBench v1.1.

Control A (context-only probe): LR probe on h(c), last-token residual of a prompt that contains the
  context but NOT the question.
Control B (spliced-but-sufficient): for every test question, replace one NON-answer sentence of the
  SUFFICIENT context by a length-matched same-topic sentence (replacement pool + filters of
  build_sufficiency_bench.py), keep the label sufficient, and score with the already-trained
  Standard / CSP probes.

Subcommands:
  build    (CPU) build data/experiments/sufficiency_bench_splice_control/test.json
  extract  (GPU, run through scripts/run_methods_guarded.sh) --model M --target context_only|splice
  eval     (CPU) write results/splice_controls.{json,md}
Use --limit N on every subcommand for a smoke test (separate *_limitN output dirs/files).
"""
import argparse, json, os, re, sys, gc, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from configs.paths import (BENCH_DIR, HIDDEN_STATES_DIR, EXPERIMENTS_DIR, RESULTS_DIR,
                           DATASETS_DIR, MINILM_PATH, MODEL_CONFIGS)

BENCH_V10 = EXPERIMENTS_DIR / "sufficiency_bench_v1.0"
SPLICE_DIR = EXPERIMENTS_DIR / "sufficiency_bench_splice_control"
CTX_ONLY_DIR = EXPERIMENTS_DIR / "hidden_states_context_only"
SPLICE_HS_DIR = EXPERIMENTS_DIR / "hidden_states_splice_control"
MODELS = ["llama", "mistral", "qwen"]


def norm_ws(t):  # identical to scripts/build_v11.py
    return re.sub(r"\s+", " ", t).strip()


def format_context_only(context):
    """format_prompt (build_sufficiency_bench.py) with the 'Question: ...' line removed."""
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, "
        f"say 'I cannot answer this from the provided context.'\n\n"
        f"Context: {context}\n\n"
        f"Answer:"
    )


def sfx(limit):
    return f"_limit{limit}" if limit else ""


# ------------------------------------------------------------------ build
def build(limit=None, split='test'):
    from sentence_transformers import SentenceTransformer
    import build_sufficiency_bench as B  # src/data is not on path; add below
    random.seed(42); np.random.seed(42)  # same as the benchmark builder before loading questions
    qs = B.load_squad_questions(max_n=1500) + B.load_hotpotqa_questions(max_n=1500) + B.load_eli5_questions(max_n=500)
    embed = SentenceTransformer(str(MINILM_PATH), device="cpu")
    pools = B.build_replacement_pools(qs, embed)
    del embed
    qmap = {q["id"]: q for q in qs}

    test11 = json.load(open(BENCH_DIR / f"{split}.json"))
    test10 = {e["id"]: e for e in json.load(open(BENCH_V10 / f"{split}.json"))}
    if limit:
        test11 = test11[:limit]
    rng = random.Random(0)
    out, skipped, repro_ok, repro_n = [], [], 0, 0
    for ex in test11:
        qid = ex["id"]
        suf11 = [c for c in ex["conditions"] if c["sufficient"]][0]
        ins11 = [c for c in ex["conditions"] if not c["sufficient"]][0]
        raw = test10[qid]["conditions"][0]["context"]
        assert test10[qid]["conditions"][0]["sufficient"]
        assert norm_ws(raw) == suf11["context"], qid
        pool = pools.get(qid, [])
        # sanity: does the rebuilt pool reproduce the stored insufficient context?
        rebuilt = B.replace_answer_sentence(qmap[qid]["context"], ex["gold_answer"], pool) if qid in qmap else None
        repro_n += 1; repro_ok += int(rebuilt is not None and norm_ws(rebuilt) == ins11["context"])
        sents = B.split_sentences(raw)
        al = ex["gold_answer"].lower()
        ans_idx = [i for i, s in enumerate(sents) if al in s.lower()]
        if not ans_idx and len(sents) > 1:  # same fallback as replace_answer_sentence
            ans_idx = [len(sents) // 2] if len(sents) > 2 else [0]
        cand = [i for i in range(len(sents)) if i not in ans_idx]
        if not cand or not pool:
            skipped.append((qid, "no non-answer sentence or empty pool")); continue
        dist = {i: min(abs(i - a) for a in ans_idx) for i in cand}
        dmax = max(dist.values())
        # farthest from the answer sentence; ties broken at random (seed 0)
        order = [i for i in cand if dist[i] == dmax]; rng.shuffle(order)
        done = None
        # try farthest first, then next farthest, so that a failed filter does not lose the question
        for i in order + sorted([j for j in cand if j not in order], key=lambda j: -dist[j]):
            tgt = sents[i]
            # pool filter of the benchmark (non-answer, len>20) already applied in build_replacement_pools;
            # additionally: not already in the context, does not contain the gold answer
            reps = [r for r in pool if r not in raw and al not in r.lower() and norm_ws(r) != norm_ws(tgt)]
            if not reps:
                continue
            rep = sorted(reps, key=lambda s: abs(len(s) - len(tgt)))[0]
            new = raw.replace(tgt, rep, 1)
            ratio = len(new) / max(len(raw), 1)
            if ratio < 0.5 or ratio > 2.0:  # benchmark's length-ratio filter
                continue
            done = (i, tgt, rep, norm_ws(new)); break
        if done is None:
            skipped.append((qid, "no replacement passes filters")); continue
        i, tgt, rep, new = done
        assert new != suf11["context"]
        out.append({
            "id": qid, "question": ex["question"], "gold_answer": ex["gold_answer"],
            "question_type": ex["question_type"], "model_knows_answer": ex["model_knows_answer"],
            "conditions": [{
                "condition_id": f"{qid}_spl", "context": new, "sufficient": True, "spliced": True,
                "model_confident": suf11["model_confident"], "quadrant": suf11["quadrant"],
                "answer_sentence_indices": ans_idx, "replaced_sentence_index": i,
                "replaced_sentence": norm_ws(tgt), "replacement_sentence": norm_ws(rep),
                "original_context": suf11["context"], "context_length": len(new),
                "prompt": B.format_prompt(ex["question"], new),
            }]})
    SPLICE_DIR.mkdir(parents=True, exist_ok=True)
    p = SPLICE_DIR / f"{split}{sfx(limit)}.json"
    json.dump(out, open(p, "w"), indent=1)
    rep = {"n_test_questions": len(test11), "n_spliced": len(out), "n_skipped": len(skipped),
           "skipped": skipped, "pool_reproduces_stored_insufficient": f"{repro_ok}/{repro_n}"}
    json.dump(rep, open(SPLICE_DIR / (f"build_report{sfx(limit)}.json" if split == "test" else f"build_report_{split}{sfx(limit)}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "skipped"}), "->", p)


# ------------------------------------------------------------------ extract
def extract(model_key, device, target, limit=None, split='test'):
    from extraction.extract_states import load_model, extract_last_token_hidden
    from tqdm import tqdm
    cfg = MODEL_CONFIGS[model_key]
    model, tok = load_model(cfg, device)
    jobs = []  # (out_dir, [(cond_id, text)])
    if target == "context_only":
        for split in ["train", "val", "test"]:
            ex = json.load(open(BENCH_DIR / f"{split}.json"))
            if limit: ex = ex[:limit]
            items = [(c["condition_id"], format_context_only(c["context"])) for e in ex for c in e["conditions"]]
            jobs.append((CTX_ONLY_DIR / f"{model_key}{sfx(limit)}" / split, items, "h_context_only"))
    else:
        ex = json.load(open(SPLICE_DIR / f"{split}{sfx(limit)}.json"))
        items = [(c["condition_id"], c["prompt"]) for e in ex for c in e["conditions"]]
        # h(q) of a spliced question is identical to the stored h_question_only (same question-only prompt)
        jobs.append((SPLICE_HS_DIR / f"{model_key}{sfx(limit)}" / split, items, "h_with_context"))
    for out_dir, items, fname in jobs:
        out_dir.mkdir(parents=True, exist_ok=True)
        H, ids = [], []
        for cid, text in tqdm(items, desc=str(out_dir.name)):
            H.append(extract_last_token_hidden(model, tok, text, device)); ids.append(cid)
        np.save(out_dir / f"{fname}.npy", np.stack(H).astype(np.float16))
        json.dump(ids, open(out_dir / "ids.json", "w"))
        print("saved", out_dir, len(ids))
    del model; gc.collect()


# ------------------------------------------------------------------ eval
def qid_of(cid):
    return re.sub(r"_(suf|insuf|spl)$", "", cid)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def cluster_boot(fn, groups, B=1000, seed=0):
    """fn(row_index_array) -> float. Resample question clusters with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        try: vals.append(fn(idx))
        except ValueError: pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bal_acc_threshold(y, p):
    best, bt = -1, 0.5
    for t in np.unique(np.concatenate([p, [0.5]])):
        pred = p >= t
        ba = 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
        if ba > best: best, bt = ba, float(t)
    return bt


def load_split(model, split, sfxs=""):
    d = HIDDEN_STATES_DIR / model / split
    meta = json.load(open(d / "metadata.json"))
    return (np.load(d / "h_with_context.npy", mmap_mode="r"), np.load(d / "h_question_only.npy", mmap_mode="r"),
            np.load(d / "labels.npy"), [m["id"] for m in meta])


def eval_model(model, limit, B):
    from methods.deco import make_classifier
    from sklearn.metrics import roc_auc_score
    best = json.load(open(RESULTS_DIR / model / "all_results_logreg.json"))["best_layers"]
    Ls, Lc = best["standard"], best["DECO"]
    tr = load_split(model, "train"); va = load_split(model, "val"); te = load_split(model, "test")
    res = {"layers": {"standard": Ls, "csp": Lc}}

    # ---- trained Standard / CSP probes
    std = make_classifier("logreg").fit(np.asarray(tr[0][:, Ls]), tr[2])
    csp = make_classifier("logreg").fit(np.asarray(tr[0][:, Lc] - tr[1][:, Lc]), tr[2])
    f_std = lambda hqc, hq: std.predict_proba(np.asarray(hqc[:, Ls]))[:, 1]
    f_csp = lambda hqc, hq: csp.predict_proba(np.asarray(hqc[:, Lc] - hq[:, Lc]))[:, 1]
    thr = {"standard": bal_acc_threshold(va[2], f_std(va[0], va[1])), "csp": bal_acc_threshold(va[2], f_csp(va[0], va[1]))}
    res["thresholds"] = thr

    # ---------------- Control A
    te_ids = te[3]; te_q = np.array([qid_of(i) for i in te_ids]); y_te = te[2]
    if limit:  # restrict to the smoke-test questions
        keep_q = {e["id"] for e in json.load(open(BENCH_DIR / "test.json"))[:limit]}
        sel = np.array([q in keep_q for q in te_q])
    else:
        sel = np.ones(len(te_ids), bool)
    ctxd = CTX_ONLY_DIR / f"{model}{sfx(limit)}"
    def load_ctx(split):
        ids = json.load(open(ctxd / split / "ids.json")); H = np.load(ctxd / split / "h_context_only.npy", mmap_mode="r")
        return H, ids
    Htr, idtr = load_ctx("train"); Hva, idva = load_ctx("val"); Hte, idte = load_ctx("test")
    def lab(ids, ref):  # labels via stored label arrays aligned by id
        m = {i: l for i, l in zip(ref[3], ref[2])}; return np.array([m[i] for i in ids])
    ytr, yva, yte = lab(idtr, tr), lab(idva, va), lab(idte, te)
    qte = np.array([qid_of(i) for i in idte])
    n_layers = Htr.shape[1]
    val_sweep = []
    for l in range(n_layers):
        p = make_classifier("logreg").fit(np.asarray(Htr[:, l]), ytr)
        val_sweep.append(auc(yva, p.predict_proba(np.asarray(Hva[:, l]))[:, 1]))
    Lbest = int(np.argmax(val_sweep))
    def ctx_test(l):
        p = make_classifier("logreg").fit(np.asarray(Htr[:, l]), ytr)
        return p.predict_proba(np.asarray(Hte[:, l]))[:, 1]
    A = {}
    for name, l in [("context_only@standard_layer", Ls), ("context_only@own_best_layer", Lbest)]:
        s = ctx_test(l)
        A[name] = {"layer": int(l), "auroc": auc(yte, s),
                   "ci95": cluster_boot(lambda ix: auc(yte[ix], s[ix]), qte, B)}
    # reference probes on the same test rows (aligned by id)
    pos = {i: k for k, i in enumerate(te_ids)}; order = np.array([pos[i] for i in idte])
    for name, f in [("standard", f_std), ("csp", f_csp)]:
        s_all = f(te[0], te[1])[order]
        A[name + "_reference"] = {"layer": int(Ls if name == "standard" else Lc), "auroc": auc(yte, s_all),
                                  "ci95": cluster_boot(lambda ix: auc(yte[ix], s_all[ix]), qte, B)}
    A["context_only_val_auroc_by_layer"] = [float(v) for v in val_sweep]
    A["n_test_conditions"] = int(len(yte))
    res["control_A"] = A

    # ---------------- Control B
    spl = json.load(open(SPLICE_DIR / f"test{sfx(limit)}.json"))
    hd = SPLICE_HS_DIR / f"{model}{sfx(limit)}" / "test"
    Hs = np.load(hd / "h_with_context.npy", mmap_mode="r"); ids_s = json.load(open(hd / "ids.json"))
    qs = [qid_of(i) for i in ids_s]
    ix_suf = np.array([pos[q + "_suf"] for q in qs]); ix_ins = np.array([pos[q + "_insuf"] for q in qs])
    hq = te[1][ix_suf]  # h(q): same question-only prompt as the stored one
    Hs = np.asarray(Hs, dtype=np.float32)
    P = {}
    for name, f, L in [("standard", f_std, Ls), ("csp", f_csp, Lc)]:
        p_suf = f(te[0][ix_suf], te[1][ix_suf]); p_ins = f(te[0][ix_ins], te[1][ix_ins])
        p_spl = f(Hs, hq)
        t = thr[name]; g = np.array(qs); n = len(qs)
        def a2(pos_s, neg_s, ix): return auc(np.r_[np.ones(len(ix)), np.zeros(len(ix))], np.r_[pos_s[ix], neg_s[ix]])
        mk = lambda fn: {"value": fn(np.arange(n)), "ci95": cluster_boot(fn, g, B)}
        P[name] = {
            "layer": int(L), "threshold": t, "n": n,
            "mean_prob_original_sufficient": mk(lambda ix: float(p_suf[ix].mean())),
            "mean_prob_spliced_sufficient": mk(lambda ix: float(p_spl[ix].mean())),
            "mean_prob_insufficient": mk(lambda ix: float(p_ins[ix].mean())),
            "auroc_orig_suf_vs_spliced_suf": mk(lambda ix: a2(p_suf, p_spl, ix)),
            "auroc_spliced_suf_vs_insufficient": mk(lambda ix: a2(p_spl, p_ins, ix)),
            "auroc_orig_suf_vs_insufficient_same_questions": mk(lambda ix: a2(p_suf, p_ins, ix)),
            "frac_spliced_below_threshold": mk(lambda ix: float((p_spl[ix] < t).mean())),
            "frac_original_sufficient_below_threshold": mk(lambda ix: float((p_suf[ix] < t).mean())),
            "frac_insufficient_below_threshold": mk(lambda ix: float((p_ins[ix] < t).mean())),
        }
    rep = json.load(open(SPLICE_DIR / f"build_report{sfx(limit)}.json"))
    P["n_spliced"] = len(qs); P["n_skipped"] = rep["n_skipped"]; P["n_test_questions"] = rep["n_test_questions"]
    res["control_B"] = P
    return res


def f3(x): return f"{x:.3f}"
def ci(d): return f"{d['value']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def render(R, B):
    L = ["# Splice controls on SufficiencyBench v1.1 (test split)", "",
         f"95% CIs: question-clustered bootstrap, B={B}, seed 0.", "",
         "## Control A: context-only probe, test AUROC (probe on h(c); no question in prompt)", "",
         "| Model | Context-only @ Standard layer | Context-only @ own best layer (val) | Standard probe h(q+c) | CSP probe |",
         "|---|---|---|---|---|"]
    for m, r in R.items():
        a = r["control_A"]
        c = lambda k: f"{a[k]['auroc']:.3f} [{a[k]['ci95'][0]:.3f}, {a[k]['ci95'][1]:.3f}]"
        L.append(f"| {m} | {c('context_only@standard_layer')} (L{a['context_only@standard_layer']['layer']}) | "
                 f"{c('context_only@own_best_layer')} (L{a['context_only@own_best_layer']['layer']}) | "
                 f"{c('standard_reference')} (L{a['standard_reference']['layer']}) | {c('csp_reference')} (L{a['csp_reference']['layer']}) |")
    L += ["", "## Control B: spliced-but-sufficient (one non-answer sentence replaced; label stays sufficient)", ""]
    for m, r in R.items():
        b = r["control_B"]
        L.append(f"{m}: {b['n_spliced']} spliced of {b['n_test_questions']} test questions; {b['n_skipped']} skipped (no replaceable non-answer sentence under the filters).")
    L += ["", "| Model | Probe | mean p orig-suf | mean p spliced-suf | mean p insuf | AUROC orig-suf vs spliced-suf | AUROC spliced-suf vs insuf | (ref) AUROC orig-suf vs insuf | frac spliced < val bal-acc thr | (ref) frac orig-suf < thr |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in R.items():
        for pn in ["standard", "csp"]:
            p = r["control_B"][pn]
            L.append(f"| {m} | {pn} | {ci(p['mean_prob_original_sufficient'])} | {ci(p['mean_prob_spliced_sufficient'])} | {ci(p['mean_prob_insufficient'])} | "
                     f"{ci(p['auroc_orig_suf_vs_spliced_suf'])} | {ci(p['auroc_spliced_suf_vs_insufficient'])} | {ci(p['auroc_orig_suf_vs_insufficient_same_questions'])} | "
                     f"{ci(p['frac_spliced_below_threshold'])} | {ci(p['frac_original_sufficient_below_threshold'])} |")
    L += ["", "AUROC orig-suf vs spliced-suf uses original as the positive class (0.5 = probe ignores the splice).",
          "AUROC spliced-suf vs insuf uses spliced-sufficient as the positive class (stays near the paper's value if the probe reads sufficiency).",
          "", "Context-only prompt: `Based on the following context, answer the question. If the context does not contain enough information, say 'I cannot answer this from the provided context.'\\n\\nContext: {context}\\n\\nAnswer:`"]
    return "\n".join(L) + "\n"


def main():
    sys.path.insert(0, str(ROOT / "src" / "data"))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "extract", "eval"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--target", choices=["context_only", "splice"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.limit, a.split)
    elif a.cmd == "extract":
        extract(a.model, a.device, a.target, a.limit, a.split)
    else:
        R = {m: eval_model(m, a.limit, a.B) for m in (MODELS if not a.model else [a.model])}
        s = sfx(a.limit)
        json.dump(R, open(RESULTS_DIR / f"splice_controls{s}.json", "w"), indent=1)
        open(RESULTS_DIR / f"splice_controls{s}.md", "w").write(render(R, a.B))
        print(render(R, a.B))


if __name__ == "__main__":
    main()
