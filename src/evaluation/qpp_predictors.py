"""Standard post-retrieval QPP predictors (WIG, NQC, SMV, Clarity, sigma_max) for the retrieved-context
experiment (tag bm25_k5), evaluated against the 'support' label, plus logreg over all predictors and paired
deltas vs the transferred Standard probes. Writes results/retrieved_ctx_qpp_proper.{json,md}.

Formulas (q = analysed query tokens, |q| = their count with repeats, N = #docs, L_C = total tokens,
cf/df = collection / document frequency of the analysed term, s(d) = Lucene BM25 score, k1=0.9, b=0.4):
  idf(t)      = ln(1 + (N - df + 0.5)/(df + 0.5))                                  (Lucene)
  s(C)        = sum_{t in q} idf(t) * cf (k1+1) / (cf + k1 (1 - b + b L_C/avgdl)),  avgdl = L_C/N
                (the collection as ONE document: tf = cf, length = L_C)
  WIG@k       = mean_{d in top-k} (s(d) - s(C)) / sqrt(|q|)
  NQC@k       = std_{d in top-k}(s(d)) / s(C)                  (population std)
  SMV@k       = [ mean_d s(d) |ln(s(d)/mu)| ] / s(C),   mu = mean_{top-k} s(d)
  Clarity@k   = sum_{w in V_k} P(w|R) log2(P(w|R)/P(w|C)),  V_k = vocab of top-k docs
                P(w|R) = sum_d P(w|d) P(q|d)/sum_d' P(q|d'),  P(q|d) = prod_{t in q} P(t|d)
                P(w|d) = (tf(w,d) + mu P(w|C))/(|d| + mu), mu = 1000, P(w|C) = cf/L_C (query terms with cf=0 skipped)
  sigma_max@k = max_{2<=j<=k} std of the top-j scores
"""
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.evaluation import retrieved_context_eval as R
from src.analysis.store_baseline_scores import load_best_layers

TAG, B, SEED, LAB = "bm25_k5", 2000, 0, "support"
KS = [5, 10, 20]
MODELS = ["llama", "mistral", "qwen"]
PRED = ["WIG", "NQC", "SMV", "Clarity", "sigma_max"]
K1, BB, MU = 0.9, 0.4, 1000.0
D = R.RC_DIR / TAG


class Stats:
    def __init__(self):
        os.environ["JAVA_HOME"] = R.JAVA21; os.environ.setdefault("OPENAI_API_KEY", "unused")
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.index.lucene import LuceneIndexReader
        rep = json.load(open(D / "build_report.json"))
        ip = str(ROOT / rep["index_dir"])
        self.searcher = LuceneSearcher(ip)
        self.searcher.set_bm25(K1, BB)
        self.reader = LuceneIndexReader(ip)
        st = self.reader.stats()
        self.N, self.LC = st["documents"], st["total_terms"]
        from datasets import load_from_disk
        val = load_from_disk(str(R.DATASETS_DIR / "hotpotqa"))["validation"]
        ptext = {}
        for ctx in val["context"]:
            for t, s in zip(ctx["title"], ctx["sentences"]):
                if t not in ptext: ptext[t] = R.clean_ws(" ".join(s))
        self.docs = [(f"d{i}", t, x) for i, (t, x) in enumerate(ptext.items())]
        assert len(self.docs) == self.N, (len(self.docs), self.N)
        self.byid = {d[0]: d for d in self.docs}
        self._tf, self._cf = {}, {}

    def doc_tf(self, did):
        if did not in self._tf:
            _, t, x = self.byid[did]
            toks = self.reader.analyze(f"{t} {x}")
            self._tf[did] = (Counter(toks), len(toks))
        return self._tf[did]

    def term(self, t):
        if t not in self._cf:
            df, cf = self.reader.get_term_counts(t, analyzer=None)
            self._cf[t] = (df, cf)
        return self._cf[t]

    def corpus_score(self, q):
        avgdl = self.LC / self.N; s = 0.0
        for t in q:
            df, cf = self.term(t)
            if cf == 0: continue
            idf = np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            s += idf * cf * (K1 + 1) / (cf + K1 * (1 - BB + BB * self.LC / avgdl))
        return s

    def clarity(self, q, dids):
        qs = [t for t in q if self.term(t)[1] > 0]
        if not qs: return np.nan
        pc = lambda t: self.term(t)[1] / self.LC
        tfs = [self.doc_tf(d) for d in dids]
        lq = np.array([sum(np.log((tf.get(t, 0) + MU * pc(t)) / (n + MU)) for t in qs) for tf, n in tfs])
        w = np.exp(lq - lq.max()); w /= w.sum()
        vocab = set().union(*[tf.keys() for tf, _ in tfs])
        c = 0.0
        for v in vocab:
            p = pc(v)
            if p == 0: continue
            pr = sum(wi * (tf.get(v, 0) + MU * p) / (n + MU) for wi, (tf, n) in zip(w, tfs))
            c += pr * np.log2(pr / p)
        return float(c)

    def features(self, recs, kmax=20):
        res = self.searcher.batch_search([r["question"] for r in recs], [str(i) for i in range(len(recs))], k=kmax, threads=4)
        out = {k: [] for k in KS}; top5 = []
        for i, r in enumerate(recs):
            hits = res[str(i)]
            sc = np.array([h.score for h in hits]); ids = [h.docid for h in hits]
            top5.append(sc[:5])
            q = self.reader.analyze(r["question"]); sC = self.corpus_score(q)
            for k in KS:
                s = sc[:k]; mu = s.mean()
                out[k].append([np.mean((s - sC) / np.sqrt(len(q))), s.std() / sC,
                               np.mean(s * np.abs(np.log(s / mu))) / sC,
                               self.clarity(q, ids[:k]), max(sc[:j].std() for j in range(2, k + 1)), s.std()])
        return {k: np.asarray(v) for k, v in out.items()}, np.asarray(top5)


def boot_ci(y, s, idx):
    a = np.array([R.auroc(y[b], s[b]) for b in idx]); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); out[i] = min(1.0, run)
    return out


def pval(d):
    d = d[np.isfinite(d)]
    return float(min(1.0, max(2 * (d <= 0).mean(), 1.0 / len(d))))


def transfer_scores(model, mask_te_recs):
    """Transferred Standard probe scores on the test set, exactly as scripts/retrieved_ctx_paired_vs_baselines.py."""
    sc_te, rec_te, hqc_te, hq_te, _ = R._load_split(TAG, model, "test")
    deco, std = load_best_layers(model)
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    bdir = R.HIDDEN_STATES_DIR / model / "train"
    b_hqc = np.load(bdir / "h_with_context.npy", mmap_mode="r"); b_hq = np.load(bdir / "h_question_only.npy", mmap_mode="r")
    b_y = np.load(bdir / "labels.npy")
    m_te, y_te = R._label_arrays(rec_te, LAB)
    keep = np.array([not r.get("truncated", 0) for r in sc_te]); y_te = y_te[keep[m_te]]; m_te &= keep
    clf = R._fit_lr(f32(b_hqc[:, std, :]), b_y)
    s = clf.predict_proba(f32(hqc_te[:, std, :]))[:, 1][m_te]
    ids = [r["id"] for r in rec_te]
    return s, y_te, [i for i, m in zip(ids, m_te) if m]


def main():
    st = Stats()
    tr = json.load(open(D / "train.json")); te = json.load(open(D / "test.json"))
    F, T5 = {}, {}
    for nm, recs in (("train", tr), ("test", te)):
        F[nm], T5[nm] = st.features(recs)
        stored = np.array([np.asarray(r["topk_scores"], float)[:5] for r in recs])
        print(nm, "max |re-retrieved top5 - stored|:", np.abs(T5[nm] - stored).max(), flush=True)
    tr_idx = {r["id"]: i for i, r in enumerate(tr)}; te_idx = {r["id"]: i for i, r in enumerate(te)}
    ytr_all = np.array([int(r["sufficient"]) for r in tr]); yte_all = np.array([int(r["sufficient"]) for r in te])
    cols = PRED + ["raw_std"]
    # ---- predictors individually, all test questions (support label)
    n = len(te); idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    out = {"n_test": n, "n_train": len(tr), "B": B, "seed": SEED, "predictors": {}, "sanity": {}}
    for k in KS:
        for j, c in enumerate(cols):
            s = F["test"][k][:, j]; ok = np.isfinite(s)
            a = R.auroc(yte_all[ok], s[ok])
            out["predictors"].setdefault(c, {})[k] = {"auroc": a, "ci95": boot_ci(yte_all, np.where(ok, s, np.nanmedian(s)), idx) if ok.all() else None, "n_nan": int((~ok).sum())}
    out["sanity"]["raw_std_k5_auroc"] = out["predictors"]["raw_std"][5]["auroc"]
    # ---- logreg over all QPP predictors (fit on train)
    from sklearn.model_selection import StratifiedKFold
    def fe(k, split):
        X = F[split][k][:, :5].copy(); return X
    def imp(Xtr, X):
        med = np.nanmedian(Xtr, 0); X = X.copy(); r, c = np.where(~np.isfinite(X)); X[r, c] = med[c]; return X
    lr, cv = {}, {}
    for k in KS:
        Xtr = imp(fe(k, "train"), fe(k, "train")); Xte = imp(fe(k, "train"), fe(k, "test"))
        oof = np.zeros(len(tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=R.SEED_PROBE).split(Xtr, ytr_all):
            oof[b] = R._fit_lr(Xtr[a], ytr_all[a]).predict_proba(Xtr[b])[:, 1]
        cv[k] = R.auroc(ytr_all, oof)
        s = R._fit_lr(Xtr, ytr_all).predict_proba(Xte)[:, 1]; lr[k] = s
        out.setdefault("logreg", {})[k] = {"auroc": R.auroc(yte_all, s), "ci95": boot_ci(yte_all, s, idx), "train_cv_auroc": cv[k]}
    best_k = max(KS, key=lambda k: cv[k]); out["best_k_by_train_cv"] = best_k
    # reference: paper's 5-feature raw-score logreg
    Q = lambda recs: R._qpp(recs)
    q5 = R._fit_lr(Q(tr), ytr_all).predict_proba(Q(te))[:, 1]
    out["sanity"]["paper_qpp_logreg_5feat_auroc"] = R.auroc(yte_all, q5)
    # ---- paired deltas vs transferred Standard probe
    comps_md = {}
    out["paired"] = {}
    for model in MODELS:
        s_probe, y_m, ids_m = transfer_scores(model, None)
        ii = np.array([te_idx[i] for i in ids_m]); assert (yte_all[ii] == y_m).all()
        nm = len(ii); idm = np.random.default_rng(SEED).integers(0, nm, size=(B, nm))
        cand = {"Standard probe (transfer)": s_probe}
        for k in sorted({5, best_k}):
            cand[f"QPP logreg k={k}"] = lr[k][ii]
        for c in PRED:
            for k in KS: cand[f"{c}@{k}"] = np.nan_to_num(F["test"][k][ii, PRED.index(c)], nan=np.nanmedian(F["test"][k][:, PRED.index(c)]))
        cand["raw std@5"] = F["test"][5][ii, 5]
        bt = {nme: np.array([R.auroc(y_m[b], v[b]) for b in idm]) for nme, v in cand.items()}
        au = {nme: R.auroc(y_m, v) for nme, v in cand.items()}
        rows = []
        for nme in cand:
            if nme.startswith("Standard"): continue
            d = bt["Standard probe (transfer)"] - bt[nme]; d = d[np.isfinite(d)]
            rows.append({"comparison": f"Standard (transfer) vs {nme}", "delta": au["Standard probe (transfer)"] - au[nme],
                         "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": pval(d)})
        hp = holm(np.array([r["p"] for r in rows]))
        for r, h in zip(rows, hp): r["p_holm"] = float(h)
        out["paired"][model] = {"n": nm, "auroc": au, "rows": rows}
    json.dump(out, open(ROOT / "results/retrieved_ctx_qpp_proper.json", "w"), indent=2, default=float)
    # ---- markdown
    f = lambda a, c: f"{a:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if c else f"{a:.3f}"
    md = [f"Retrieved-context bm25_k5, support label. Test N={n}, train N={len(tr)}. 95% bootstrap CI over questions (B={B}, seed {SEED}).\n",
          "| Predictor | k=5 | k=10 | k=20 |", "|---|---|---|---|"]
    for c in cols:
        label = "raw top-k score std (reference)" if c == "raw_std" else c
        md.append(f"| {label} | " + " | ".join(f(out['predictors'][c][k]["auroc"], out['predictors'][c][k]["ci95"]) for k in KS) + " |")
    md.append("| Logreg over WIG, NQC, SMV, Clarity, sigma_max (fit on train) | " + " | ".join(f(out['logreg'][k]["auroc"], out['logreg'][k]["ci95"]) for k in KS) + " |")
    md.append(f"\nBest k for the logreg by 5-fold train CV AUROC ({', '.join(f'k={k}: {cv[k]:.3f}' for k in KS)}): k={best_k}. "
              f"Sanity: raw std@5 = {out['sanity']['raw_std_k5_auroc']:.4f} (paper 0.584); paper 5-feature raw logreg = {out['sanity']['paper_qpp_logreg_5feat_auroc']:.4f} (paper 0.671).\n")
    for model in MODELS:
        p = out["paired"][model]
        md.append(f"### Paired Delta AUROC: transferred Standard probe minus QPP predictor, {model} (N={p['n']}, probe AUROC {p['auroc']['Standard probe (transfer)']:.3f})\n")
        md.append("| Comparison | Delta | 95% CI | p | Holm p |\n|---|---|---|---|---|")
        for r in p["rows"]:
            md.append(f"| {r['comparison']} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.4f} | {r['p_holm']:.4f} |")
        md.append("")
    md.append("p = 2*frac(delta<=0) (floor 1/B), Holm over all comparisons within a model.")
    (ROOT / "results/retrieved_ctx_qpp_proper.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
