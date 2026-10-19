"""Writes results/retrieved_ctx_e5_summary.md from the e5_k5 analysis_support.json files (models present)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
rows = [("Retriever-score logreg (5 feats)", "qpp_logreg(5 feats)"), ("Retriever top-1 score", "qpp_top1"),
        ("E5 cosine(q, ctx)", "e5 cosine(q, ctx)"), ("Verbalized confidence", "Verbalized confidence"),
        ("Model answers (no abstention)", "Model answers (no abstention)"),
        ("Standard probe (transfer)", "Standard probe (transfer)"), ("CSP probe (transfer)", "CSP probe (transfer)"),
        ("Standard probe (retrained)", "Standard probe (retrained)"), ("CSP probe (retrained)", "CSP probe (retrained)")]
res = {m: json.load(open(p)) for m in ("llama", "mistral", "qwen")
       if (p := R / m / "retrieved_ctx_e5_k5" / "analysis_support.json").exists()}
pr = {m: json.load(open(p)) for m in res if (p := R / "retrieved_ctx_paired_vs_baselines_e5_k5.json").exists()} if (R / "retrieved_ctx_paired_vs_baselines_e5_k5.json").exists() else {}
paired = json.load(open(R / "retrieved_ctx_paired_vs_baselines_e5_k5.json")) if pr else {}
ms = list(res)
L = ["# Retrieved-context, E5 dense retriever (e5_k5), support label", "",
     "Sufficiency rate (all gold titles in top-5), test: " + ", ".join(f"{m} {res[m]['pos_rate_test']:.3f} (N={res[m]['n_test']})" for m in ms)
     + "; BM25 reference 0.445. Retrieval-level rate by k (test, E5): k=1 0.000, k=3 0.563, k=5 0.662, k=10 0.765 (BM25: 0.000, 0.330, 0.445, 0.587).", "",
     "AUROC [95% CI], support label.", "", "| Predictor | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
for lab, k in rows:
    L.append(f"| {lab} | " + " | ".join(
        f"{res[m]['methods'][k]['auroc']:.3f} [{res[m]['methods'][k]['ci95'][0]:.3f}, {res[m]['methods'][k]['ci95'][1]:.3f}]" for m in ms) + " |")
L += ["", "Paired delta AUROC of transferred probes (95% CI; paired bootstrap, B=2000).", "",
      "| Comparison | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
for p in ("Standard", "CSP"):
    L.append(f"| {p} (transfer) - VC | " + " | ".join(
        "{d:+.3f} [{a:+.3f}, {b:+.3f}], p={q:.3f}".format(d=res[m]['methods'][f'{p} probe (transfer)']['delta_vs_vc'],
        a=res[m]['methods'][f'{p} probe (transfer)']['delta_vs_vc_ci95'][0], b=res[m]['methods'][f'{p} probe (transfer)']['delta_vs_vc_ci95'][1],
        q=res[m]['methods'][f'{p} probe (transfer)']['delta_vs_vc_p']) for m in ms) + " |")
for p in ("Standard", "CSP"):
    cells = []
    for m in ms:
        c = next(c for c in paired[m]["comparisons"] if c["comparison"] == f"{p} (transfer) vs E5 cosine")
        cells.append(f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], Holm p={c['p_holm']:.4f}")
    L.append(f"| {p} (transfer) - E5 cosine | " + " | ".join(cells) + " |")
(R / "retrieved_ctx_e5_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
