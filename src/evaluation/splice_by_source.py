"""Per-source (squad/hotpot/eli5) breakdown of the splice controls A and B (CPU)."""
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "data"))
from evaluation.splice_controls import *  # noqa
from evaluation.splice_controls import qid_of, auc, cluster_boot, bal_acc_threshold, load_split, norm_ws
from methods.deco import make_classifier

B = 1000
SRC = ["squad", "hotpot", "eli5"]


def ci(v, lo_hi): return f"{v:.3f} [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def hotpot_support(spl):
    from datasets import load_from_disk
    ds = load_from_disk(str(DATASETS_DIR / "hotpotqa")); split = "validation" if "validation" in ds else list(ds.keys())[0]
    d = ds[split]; n = hit = 0; strict_hit = 0
    for e in spl:
        if not e["id"].startswith("hotpot"): continue
        ex = d[int(e["id"].split("_")[1])]
        titles = ex["context"]["title"]; sents = ex["context"]["sentences"]
        sf = list(zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]))
        sup = []
        for t, i in sf:
            if t in titles:
                ss = sents[titles.index(t)]
                if i < len(ss): sup.append(norm_ws(ss[i]))
        rs = e["conditions"][0]["replaced_sentence"]
        n += 1
        if any(s and (s in rs or rs in s) for s in sup): hit += 1
        # strict: replaced sentence shares >=50% chars with a supporting sentence
        elif any(s and len(set(s.split()) & set(rs.split())) >= 0.5 * min(len(s.split()), len(rs.split())) for s in sup): strict_hit += 1
    return {"n_hotpot_spliced": n, "replaced_is_supporting_sentence": hit, "partial_word_overlap_ge50pct": strict_hit,
            "note": "all hotpot contexts consist solely of the gold supporting PARAGRAPHS, so the replaced sentence is always from a gold paragraph; the counts above are for the sentence-level supporting facts"}


def run_model(m):
    best = json.load(open(RESULTS_DIR / m / "all_results_logreg.json"))["best_layers"]; Ls, Lc = best["standard"], best["DECO"]
    tr = load_split(m, "train"); va = load_split(m, "val"); te = load_split(m, "test")
    std = make_classifier("logreg").fit(np.asarray(tr[0][:, Ls]), tr[2]); csp = make_classifier("logreg").fit(np.asarray(tr[0][:, Lc] - tr[1][:, Lc]), tr[2])
    f = {"standard": lambda a, q: std.predict_proba(np.asarray(a[:, Ls]))[:, 1], "csp": lambda a, q: csp.predict_proba(np.asarray(a[:, Lc] - q[:, Lc]))[:, 1]}
    thr = {k: bal_acc_threshold(va[2], f[k](va[0], va[1])) for k in f}
    ids = te[3]; pos = {i: k for k, i in enumerate(ids)}; y = te[2]; src = np.array([i.split("_")[0] for i in ids]); q = np.array([qid_of(i) for i in ids])
    # Control A
    ctxd = CTX_ONLY_DIR / m
    Htr = np.load(ctxd / "train/h_context_only.npy", mmap_mode="r"); idtr = json.load(open(ctxd / "train/ids.json"))
    Hte = np.load(ctxd / "test/h_context_only.npy", mmap_mode="r"); idte = json.load(open(ctxd / "test/ids.json"))
    lm = dict(zip(tr[3], tr[2])); ytr = np.array([lm[i] for i in idtr])
    cp = make_classifier("logreg").fit(np.asarray(Htr[:, Ls]), ytr)
    order = np.array([pos[i] for i in idte]); s_ctx = cp.predict_proba(np.asarray(Hte[:, Ls]))[:, 1]
    yte = y[order]; qte = q[order]; ste = src[order]
    s_std = f["standard"](te[0], te[1])[order]; s_csp = f["csp"](te[0], te[1])[order]
    A = {}
    for s in SRC + ["all"]:
        mk = (ste == s) if s != "all" else np.ones(len(ste), bool); ix0 = np.where(mk)[0]
        row = {"n_conditions": int(mk.sum())}
        for name, sc in [("context_only", s_ctx), ("standard", s_std), ("csp", s_csp)]:
            fn = lambda ix, sc=sc: auc(yte[ix0][ix], sc[ix0][ix]); g = qte[ix0]
            row[name] = {"auroc": fn(np.arange(len(ix0))), "ci95": cluster_boot(fn, g, B)}
        A[s] = row
    # Control B
    spl = json.load(open(SPLICE_DIR / "test.json")); hd = SPLICE_HS_DIR / m / "test"
    Hs = np.asarray(np.load(hd / "h_with_context.npy"), dtype=np.float32); idss = json.load(open(hd / "ids.json"))
    qs = np.array([qid_of(i) for i in idss]); ssrc = np.array([i.split("_")[0] for i in qs])
    i_suf = np.array([pos[x + "_suf"] for x in qs]); i_ins = np.array([pos[x + "_insuf"] for x in qs])
    Bres = {}
    for name in f:
        p_suf = f[name](te[0][i_suf], te[1][i_suf]); p_ins = f[name](te[0][i_ins], te[1][i_ins]); p_spl = f[name](Hs, te[1][i_suf]); t = thr[name]
        for s in SRC + ["all"]:
            ix0 = np.where(ssrc == s)[0] if s != "all" else np.arange(len(qs)); g = qs[ix0]
            def a2(pp, pn, ix): return auc(np.r_[np.ones(len(ix)), np.zeros(len(ix))], np.r_[pp[ix0][ix], pn[ix0][ix]])
            def mm(fn): return {"value": fn(np.arange(len(ix0))), "ci95": cluster_boot(fn, g, B)}
            Bres.setdefault(name, {})[s] = {
                "n": int(len(ix0)),
                "mean_p_orig_suf": mm(lambda ix: float(p_suf[ix0][ix].mean())), "mean_p_spliced_suf": mm(lambda ix: float(p_spl[ix0][ix].mean())),
                "mean_p_insuf": mm(lambda ix: float(p_ins[ix0][ix].mean())),
                "auroc_orig_vs_spliced": mm(lambda ix: a2(p_suf, p_spl, ix)), "auroc_spliced_vs_insuf": mm(lambda ix: a2(p_spl, p_ins, ix)),
                "auroc_orig_vs_insuf": mm(lambda ix: a2(p_suf, p_ins, ix)),
                "frac_spliced_below_thr": mm(lambda ix: float((p_spl[ix0][ix] < t).mean())), "frac_orig_below_thr": mm(lambda ix: float((p_suf[ix0][ix] < t).mean())),
            }
    return {"A": A, "B": Bres, "thresholds": thr}


def main():
    R = {m: run_model(m) for m in MODELS}
    spl = json.load(open(SPLICE_DIR / "test.json"))
    R["hotpot_supporting_sentence_check"] = hotpot_support(spl)
    json.dump(R, open(RESULTS_DIR / "splice_controls_by_source.json", "w"), indent=1)
    L = ["# Splice controls by source dataset (test split)", "", f"95% CI: question-clustered bootstrap B={B}, seed 0. Layers as in splice_controls.md.", "",
         "## Control A: test AUROC by source", "", "| Model | Source | n cond. | Context-only | Standard | CSP |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for s in SRC + ["all"]:
            r = R[m]["A"][s]; L.append(f"| {m} | {s} | {r['n_conditions']} | " + " | ".join(ci(r[k]["auroc"], r[k]["ci95"]) for k in ["context_only", "standard", "csp"]) + " |")
    L += ["", "## Control B by source", "", "| Model | Probe | Source | n | p orig-suf | p spliced-suf | p insuf | AUROC orig vs spliced | AUROC spliced vs insuf | (ref) orig vs insuf | frac spliced < thr | (ref) frac orig < thr |", "|" + "---|" * 12]
    for m in MODELS:
        for pn in ["standard", "csp"]:
            for s in SRC + ["all"]:
                r = R[m]["B"][pn][s]; c = lambda k: ci(r[k]["value"], r[k]["ci95"])
                L.append(f"| {m} | {pn} | {s} | {r['n']} | " + " | ".join(c(k) for k in ["mean_p_orig_suf", "mean_p_spliced_suf", "mean_p_insuf", "auroc_orig_vs_spliced", "auroc_spliced_vs_insuf", "auroc_orig_vs_insuf", "frac_spliced_below_thr", "frac_orig_below_thr"]) + " |")
    h = R["hotpot_supporting_sentence_check"]
    L += ["", "## HotpotQA: was the replaced sentence a supporting fact?", "", json.dumps(h, indent=1)]
    open(RESULTS_DIR / "splice_controls_by_source.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

main()
