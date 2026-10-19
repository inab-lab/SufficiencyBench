"""Paired bootstrap of hidden-state probes vs E5 cosine and QPP logreg (retrieved-context, bm25_k5, support label).
Reuses functions from src/evaluation/retrieved_context_eval.py. Writes
results/retrieved_ctx_paired_vs_baselines.{json,md}."""
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.evaluation import retrieved_context_eval as R
from src.analysis.store_baseline_scores import load_best_layers

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--tag", default="bm25_k5"); _args = _ap.parse_args()
TAG, B, SEED, LAB = _args.tag, 2000, 0, "support"
SFX = "" if TAG == "bm25_k5" else "_" + TAG
MODELS = ["llama", "mistral", "qwen"]
PROBES = ["Standard (transfer)", "CSP (transfer)", "Standard (retrained)", "CSP (retrained)"]
BASES = {"E5": "E5 cosine", "QPP": "QPP logreg"}


def scores_for(model):
    sc_tr, rec_tr, hqc_tr, hq_tr, _ = R._load_split(TAG, model, "train")
    sc_te, rec_te, hqc_te, hq_te, _ = R._load_split(TAG, model, "test")
    deco, std = load_best_layers(model)
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    def feat(key, hqc, hq):
        return f32(hqc[:, std, :]) if key == "std" else f32(hqc[:, deco, :]) - f32(hq[:, deco, :])
    bdir = R.HIDDEN_STATES_DIR / model / "train"
    b_hqc = np.load(bdir / "h_with_context.npy", mmap_mode="r")
    b_hq = np.load(bdir / "h_question_only.npy", mmap_mode="r")
    b_y = np.load(bdir / "labels.npy")
    m_tr, y_tr = R._label_arrays(rec_tr, LAB); m_te, y_te = R._label_arrays(rec_te, LAB)
    keep_tr = np.array([not r.get("truncated", 0) for r in sc_tr]); keep_te = np.array([not r.get("truncated", 0) for r in sc_te])
    y_tr = y_tr[keep_tr[m_tr]]; m_tr &= keep_tr
    y_te = y_te[keep_te[m_te]]; m_te &= keep_te
    S = {}
    for key, nm in (("std", "Standard"), ("csp", "CSP")):
        Ftr, Fte = feat(key, hqc_tr, hq_tr), feat(key, hqc_te, hq_te)
        clf = R._fit_lr(feat(key, b_hqc, b_hq), b_y)
        S[f"{nm} (transfer)"] = clf.predict_proba(Fte)[:, 1][m_te]
        S[f"{nm} (retrained)"] = R._fit_lr(Ftr[m_tr], y_tr).predict_proba(Fte[m_te])[:, 1]
    Qtr, Qte = R._qpp(rec_tr), R._qpp(rec_te)
    S["QPP logreg"] = R._fit_lr(Qtr[m_tr], y_tr).predict_proba(Qte[m_te])[:, 1]
    e5dir = R.RC_DIR / TAG / "e5_cache"
    S["E5 cosine"] = R._e5_scores(rec_te, e5dir / "test.json", "cpu")[m_te]
    S["VC"] = np.array([r["vc"] for r in sc_te], float)[m_te]
    return S, y_te


def boot_auc(y, s, idx):
    return np.array([R.auroc(y[b], s[b]) for b in idx])


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); out[i] = min(1.0, run)
    return out


def pval(d, twosided_min=False):
    d = d[np.isfinite(d)]
    f = min((d <= 0).mean(), (d >= 0).mean()) if twosided_min else (d <= 0).mean()
    return float(min(1.0, max(2 * f, 1.0 / len(d))))


# paper AUROC names in analysis_support.json
NAME = {"Standard (transfer)": "Standard probe (transfer)", "CSP (transfer)": "CSP probe (transfer)",
        "Standard (retrained)": "Standard probe (retrained)", "CSP (retrained)": "CSP probe (retrained)",
        "QPP logreg": "qpp_logreg(5 feats)", "E5 cosine": "e5 cosine(q, ctx)", "VC": "Verbalized confidence"}

out, md = {}, []
for model in MODELS:
    S, y = scores_for(model)
    ref = json.load(open(R.RESULTS_DIR / model / f"retrieved_ctx_{TAG}" / f"analysis_{LAB}.json"))["methods"]
    aucs = {k: R.auroc(y, v) for k, v in S.items()}
    check = {k: {"mine": aucs[k], "ref": ref[NAME[k]]["auroc"], "diff": aucs[k] - ref[NAME[k]]["auroc"]} for k in S}
    bad = {k: c for k, c in check.items() if abs(c["diff"]) > 0.002}
    print(model, {k: round(c["diff"], 5) for k, c in check.items()})
    if bad:
        print("MISMATCH", model, bad); sys.exit(1)
    n = len(y)
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    bt = {k: boot_auc(y, v, idx) for k, v in S.items()}
    rows = []
    for bk, bn in BASES.items():
        for p in PROBES:
            d = bt[p] - bt[bn]; d = d[np.isfinite(d)]
            rows.append({"comparison": f"{p} vs {bn}", "family": "probe_vs_baseline", "delta": aucs[p] - aucs[bn],
                         "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": pval(d)})
    hp = holm(np.array([r["p"] for r in rows]))
    for r, h in zip(rows, hp): r["p_holm"] = float(h)
    extra = []
    for a_, b_ in (("E5 cosine", "VC"), ("QPP logreg", "VC")):
        d = bt[a_] - bt[b_]; d = d[np.isfinite(d)]
        extra.append({"comparison": f"{a_} vs VC", "family": "baseline_vs_vc", "delta": aucs[a_] - aucs[b_],
                      "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": pval(d, True), "p_holm": None})
    out[model] = {"n_test": n, "B": B, "seed": SEED, "auroc": aucs, "auroc_check_vs_analysis": check,
                  "comparisons": rows + extra}
    md.append(f"### {model} ({TAG}, support label, N={n}, B={B})\n")
    md.append("AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + "\n")
    md.append("| Comparison | Delta AUROC | 95% CI | p | Holm p |\n|---|---|---|---|---|")
    for r in rows + extra:
        h = "--" if r["p_holm"] is None else f"{r['p_holm']:.4f}"
        md.append(f"| {r['comparison']} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.4f} | {h} |")
    md.append("")
json.dump(out, open(ROOT / f"results/retrieved_ctx_paired_vs_baselines{SFX}.json", "w"), indent=2)
open(ROOT / f"results/retrieved_ctx_paired_vs_baselines{SFX}.md", "w").write(
    "Paired bootstrap over questions (B=2000, seed 0). p = 2*frac(delta<=0), floor 1/B. Holm within the 8 probe-vs-baseline comparisons per model; baseline-vs-VC rows use p = 2*min(frac<=0, frac>=0) (direction not assumed, as in analysis.md) and are not Holm-adjusted.\n\n" + "\n".join(md))
