#!/usr/bin/env python
"""Build the LaTeX table for Section "Sufficiency Prediction on Retrieved Contexts" from
results/<model>/retrieved_ctx_<tag>/analysis.json (written by src/evaluation/retrieved_context_eval.py analyse).

Usage: python scripts/make_retrieved_ctx_table.py --tag bm25_k5 --models llama mistral qwen \
           --out paper/tois_v1_src/table_retrieved.tex --stats results/retrieved_ctx_bm25_k5_table_stats.json
"""
import argparse, json, os, sys

ROWS = [
    ("group", r"\emph{Retriever scores (post-retrieval QPP)}"),
    ("qpp_top1", "Top-1 BM25 score"),
    ("qpp_gap12", "Score gap (rank 1 $-$ rank 2)"),
    ("qpp_mean", "Mean top-$k$ score"),
    ("qpp_std_nqc", "Score std.\\ dev.\\ of the top-$k$ (unnormalised NQC)"),
    ("qpp_logreg(5 feats)", "Logistic regression over the five"),
    ("group", r"\emph{Query--context similarity}"),
    ("e5 cosine(q, ctx)", "E5 cosine (question, context)"),
    ("group", r"\emph{Output-based}"),
    ("Verbalized confidence", "Verbalized confidence"),
    ("Model answers (no abstention)", "Model does not abstain"),
    ("group", r"\emph{Hidden-state probes, transferred from \bench{}}"),
    ("Standard probe (transfer)", "Standard probe"),
    ("CSP probe (transfer)", "CSP probe"),
    ("group", r"\emph{Hidden-state probes, retrained on retrieved contexts}"),
    ("Standard probe (retrained)", "Standard probe"),
    ("CSP probe (retrained)", "CSP probe"),
]
NAMES = {"llama": "Llama~3.1~8B", "mistral": "Mistral~7B", "qwen": "Qwen~2.5~7B"}

def load(model, tag, label):
    p = f"results/{model}/retrieved_ctx_{tag}/analysis.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))[label]

def fmt(m):
    lo, hi = m["ci95"]
    return f"{m['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bm25_k5")
    ap.add_argument("--models", nargs="+", default=["llama", "mistral", "qwen"])
    ap.add_argument("--label", default="support")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    data = {m: load(m, a.tag, a.label) for m in a.models}
    models = [m for m in a.models if data[m] is not None]
    missing = [m for m in a.models if data[m] is None]
    if missing:
        print("WARNING missing:", missing, file=sys.stderr)
    cols = " & ".join(NAMES[m] for m in models)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Sufficiency prediction on BM25-retrieved contexts (top-5 passages, pooled HotpotQA paragraphs; 1,500 test questions, " +
                 f"{100*data[models[0]]['pos_rate_test']:.1f}" + r"\% sufficient). AUROC with 95\% question-clustered bootstrap CI (2,000 resamples). " +
                 r"Transferred probes are the \bench{}-trained probes at their selected layers, applied without retraining; retrained probes are trained on 3,000 disjoint retrieved contexts with the same recipe. Retriever-score and similarity predictors do not read the generator, so one value spans the model columns; every other predictor reads the listed model.}")
    lines.append(r"\label{tab:retrieved}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(models) + "}")
    lines.append(r"\toprule")
    lines.append(r"Predictor & " + cols + r" \\")
    lines.append(r"\midrule")
    for key, label in ROWS:
        if key == "group":
            lines.append(r"\multicolumn{" + str(len(models) + 1) + r"}{l}{" + label + r"} \\")
            continue
        if key.startswith("qpp_") or key.startswith("e5"):
            # generator-independent predictors: one value spanning the model columns
            vals = {fmt(data[m]["methods"][key]) for m in models}
            assert len(vals) == 1, (key, vals)
            cells = r"\multicolumn{" + str(len(models)) + r"}{c}{" + vals.pop() + "}"
        else:
            cells = " & ".join(fmt(data[m]["methods"][key]) for m in models)
        lines.append(label + " & " + cells + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(a.out, "w").write("\n".join(lines) + "\n")
    # stats for the text
    st = {}
    for m in models:
        d = data[m]
        M = d["methods"]
        st[m] = {
            "n_test": d["n_test"], "pos_rate_test": d["pos_rate_test"], "pk_rate_test": d["pk_rate_test"],
            "n_pk1_test": d["n_pk1_test"], "em_all": d["em_all"], "f1_all": d["f1_all"],
            "abstention_rate": d["abstention_rate"], "best_layers": d["best_layers"],
            "support_answer_agreement_test": d["support_answer_agreement_test"],
            "oracle": d["oracle_answer_in_context"],
            "methods": {k: {"auroc": v["auroc"], "ci95": v["ci95"], "aurc": v["aurc"], "auroc_pk0": v["auroc_pk0"], "auroc_pk1": v["auroc_pk1"],
                            "delta_vs_vc": v.get("delta_vs_vc"), "delta_vs_vc_ci95": v.get("delta_vs_vc_ci95"), "delta_vs_vc_p": v.get("delta_vs_vc_p"),
                            "tau": v["kendall_tau_f1"], "cov5": v["cov_at_risk_5"], "cov10": v["cov_at_risk_10"]} for k, v in M.items()},
            "layer_sweep": d["layer_sweep_retrained_standard"],
        }
        # strict / answer label variants for the text
        for lab in ("strict", "answer"):
            dd = load(m, a.tag, lab)
            if dd:
                st[m][lab] = {"n_test": dd["n_test"], "pos_rate_test": dd["pos_rate_test"],
                              "methods": {k: {"auroc": v["auroc"], "ci95": v["ci95"], "delta_vs_vc": v.get("delta_vs_vc"), "delta_vs_vc_p": v.get("delta_vs_vc_p")} for k, v in dd["methods"].items()}}
    json.dump(st, open(a.stats, "w"), indent=1)
    print("wrote", a.out, a.stats, "models:", models)

if __name__ == "__main__":
    main()
