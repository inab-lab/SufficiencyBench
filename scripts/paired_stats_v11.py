"""Question-clustered bootstrap CIs, paired tests and risk-coverage from stored per-example scores (v1.1 test, 1069 questions)."""
import json, os, numpy as np
from scipy.stats import rankdata, norm
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["llama", "mistral", "qwen", "qwen3", "qwen14", "deepseek32"]
B, SEED = 2000, 0
STD, CSP, VC = "Standard Probe", "CSP Probe", "Verbalized Confidence"
J = lambda *p: json.load(open(os.path.join(R, *p)))
test = J("data/experiments/sufficiency_bench/test.json")
QIDS = [q["id"] for q in test]
CIDS = [c["condition_id"] for q in test for c in q["conditions"]]
LAB = np.array([int(c["sufficient"]) for q in test for c in q["conditions"]])
QIDX = np.array([i for i, q in enumerate(test) for _ in q["conditions"]])
NQ = len(QIDS)
ROWS = [np.where(QIDX == i)[0] for i in range(NQ)]

def auroc(s, y):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def delong_p(s1, s2, y):
    pos, neg = y == 1, y == 0
    def comps(s):
        a, b = s[pos], s[neg]
        psi = (a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])
        return psi.mean(1), psi.mean(0)
    v1, w1 = comps(s1); v2, w2 = comps(s2)
    d = v1.mean() - v2.mean()
    var = (np.var(v1 - v2, ddof=1) / len(v1) + np.var(w1 - w2, ddof=1) / len(w1))
    return float(2 * norm.sf(abs(d) / np.sqrt(var))) if var > 0 else float("nan")

def load_scores(m):
    """Return {method: np.array aligned with CIDS} and notes on skipped items."""
    out, notes = {}, []
    def add(name, ids, sc):
        if ids != CIDS: notes.append(f"{name}: ids do not match v1.1 test split ({len(ids)} examples); skipped"); return
        out[name] = np.array(sc, float)
    def f(*p): return os.path.join(R, "results", m, *p)
    if os.path.exists(f("scores", "probe_scores.json")):
        for k, n in [("Standard (LogReg)", STD), ("CSP (LogReg)", CSP)]:
            d = J("results", m, "scores", "probe_scores.json").get(k)
            if d: add(n, [x["id"] for x in d["metadata"]], d["scores"])
    if os.path.exists(f("scores", "baseline_scores.json")):
        d = J("results", m, "scores", "baseline_scores.json").get("verbalized_confidence")
        if d: add(VC, [x["condition_id"] for x in d["metadata"]], d["scores"])
    if os.path.exists(f("scores", "contrastive_scores.json")):
        for k, d in J("results", m, "scores", "contrastive_scores.json").items():
            add(k, [x["condition_id"] for x in d["metadata"]], d["scores"])
    vt = f("vc_thinking")
    if os.path.isdir(vt):  # qwen3: per-condition VC under thinking/CoT variants
        for fn in sorted(os.listdir(vt)):
            rows = [json.loads(l) for l in open(os.path.join(vt, fn))]
            add(f"VC ({fn[:-6]})", [r["condition_id"] for r in rows], [r["score"] for r in rows])
    return out, notes

def risk_coverage(s, y):
    o = np.argsort(-s, kind="stable"); s, y = s[o], y[o]
    last = np.r_[s[1:] != s[:-1], True]  # threshold sweep: predict sufficient if score >= t
    k = np.arange(1, len(s) + 1)[last]; wrong = np.cumsum(1 - y)[last]
    cov, risk = k / len(s), wrong / k
    aurc = float(np.trapezoid(np.r_[risk[0], risk], np.r_[0, cov]))  # curve padded back to coverage 0
    cv = lambda a: float(cov[risk <= a].max()) if (risk <= a).any() else 0.0
    return {"AURC": aurc, "coverage_at_risk<=5%": cv(.05), "coverage_at_risk<=10%": cv(.10)}

def holm(p):
    p = np.array(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (len(p) - i) * p[j])); out[j] = run
    return out

def main():
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([ROWS[i] for i in rng.integers(0, NQ, NQ)]) for _ in range(B)]
    res, md = {}, ["# Paired statistics on SufficiencyBench v1.1 test (1,069 questions, 2,138 conditions)\n",
        f"Question-clustered bootstrap: {B} resamples of questions (both conditions move together), seed {SEED}. "
        "Delta = AUROC(method) - AUROC(reference); p = two-sided bootstrap p (2*min tail fraction, floor 1/B). "
        "Holm is across methods within a model, for the comparison against the Standard probe. "
        "DeLong p (in JSON) treats examples as independent and ignores pairing by question.\n"]
    aurc_rows = []
    for m in MODELS:
        sc, notes = load_scores(m)
        if not sc: res[m] = {"_notes": notes + ["no per-example scores"]}; continue
        pk = None
        pf = os.path.join(R, "data/experiments/sufficiency_bench", f"pk_cache_{m}.json")
        if os.path.exists(pf):
            c = J("data/experiments/sufficiency_bench", f"pk_cache_{m}.json")
            if all(q in c for q in QIDS): pk = np.array([int(bool(c[QIDX_q]["knows"])) for QIDX_q in np.array(QIDS)[QIDX]])
        else: notes.append("no PK cache")
        bs = {n: np.array([auroc(s[ix], LAB[ix]) for ix in boots]) for n, s in sc.items()}
        res[m] = {"_notes": notes}
        for n, s in sc.items():
            e = {"auroc": auroc(s, LAB), "ci95_clustered": list(np.nanpercentile(bs[n], [2.5, 97.5]))}
            if pk is not None:
                for v in (0, 1):
                    k = pk == v; e[f"auroc_pk{v}"] = auroc(s[k], LAB[k])
            if n in (STD, CSP, VC): e["risk_coverage"] = risk_coverage(s, LAB)
            res[m][n] = e
        refs = {"vs_standard": STD, "vs_vc": VC if VC in sc else ("VC (chat-nothink)" if "VC (chat-nothink)" in sc else None)}
        for key, ref in refs.items():
            if ref not in sc: continue
            names = [n for n in sc if n != ref]; ps = []
            for n in names:
                d = bs[n] - bs[ref]; ok = ~np.isnan(d); d = d[ok]
                p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B)
                res[m][n][key] = {"reference": ref, "delta": res[m][n]["auroc"] - res[m][ref]["auroc"],
                                  "ci95": list(np.percentile(d, [2.5, 97.5])), "p_bootstrap": float(p),
                                  "p_delong_independent": delong_p(sc[n], sc[ref], LAB)}
                ps.append(p)
            if key == "vs_standard":
                for n, h in zip(names, holm(ps)): res[m][n][key]["p_holm"] = float(h)
        md.append(f"\n## {m}\n")
        md += [f"- note: {x}" for x in notes]
        md.append("\n| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |\n|---|---|---|---|---|")
        for n in sc:
            e = res[m][n]; cell = lambda k, h: "-" if k not in e else f"{e[k]['delta']:+.3f} [{e[k]['ci95'][0]:+.3f}, {e[k]['ci95'][1]:+.3f}], " + (f"{e[k]['p_holm']:.3g}" if h else f"{e[k]['p_bootstrap']:.3g}")
            md.append(f"| {n} | {e['auroc']:.3f} | [{e['ci95_clustered'][0]:.3f}, {e['ci95_clustered'][1]:.3f}] | {cell('vs_standard', 1)} | {cell('vs_vc', 0)} |")
        if pk is not None:
            md.append("\nPK strata AUROC: " + "; ".join(f"{n}: PK0 {res[m][n]['auroc_pk0']:.3f}, PK1 {res[m][n]['auroc_pk1']:.3f}" for n in sc))
        for n in (STD, CSP, VC):
            if n in sc: r = res[m][n]["risk_coverage"]; aurc_rows.append(f"| {m} | {n} | {r['AURC']:.4f} | {r['coverage_at_risk<=5%']:.3f} | {r['coverage_at_risk<=10%']:.3f} |")
    md += ["\n## Risk-coverage\n", "Decision: predict sufficient if score >= t; coverage = fraction predicted sufficient; risk = fraction of those that are insufficient. AURC = trapezoid area over coverage.\n",
           "| Model | Method | AURC | Coverage @ risk<=5% | Coverage @ risk<=10% |\n|---|---|---|---|---|"] + aurc_rows
    json.dump(res, open(os.path.join(R, "results/paired_stats_v11.json"), "w"), indent=1)
    open(os.path.join(R, "results/paired_stats_v11.md"), "w").write("\n".join(md) + "\n")

if __name__ == "__main__": main()
