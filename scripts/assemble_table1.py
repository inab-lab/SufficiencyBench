#!/usr/bin/env python
"""assemble_table1.py — Assemble the 14-method Table 1 from result JSONs and
compare each AUROC against the paper's reported value.

Reads results/<model>/{all_results_logreg,all_results_neural,baseline_results,
contrastive_baselines,llm_judge_results,redeep_results,semantic_entropy_results}.json
and the paper's Table 1 values, prints a per-method mine-vs-paper table with a
PASS/FLAG marker (FLAG if |mine-paper| exceeds the method's ~CI tolerance).

Missing files are reported as 'pending' so this can be run mid-run.
"""
import json
from pathlib import Path

ROOT = Path("/home/inab/Documents/MSD_Project/paper1")
RES = ROOT / "results"
MODELS = ["mistral", "llama", "qwen"]

# Paper Table 1 AUROC (tab:main_results), per model.
PAPER = {
    "Standard Probe (LogReg)": {"mistral": 0.922, "llama": 0.923, "qwen": 0.946},
    "CSP Probe (LogReg)":      {"mistral": 0.910, "llama": 0.918, "qwen": 0.949},
    "Standard Probe (Neural)": {"mistral": 0.931, "llama": 0.928, "qwen": 0.953},
    "CSP Probe (Neural)":      {"mistral": 0.943, "llama": 0.933, "qwen": 0.956},
    "ReDeEP-ECS":              {"mistral": 0.631, "llama": 0.573, "qwen": 0.602},
    "ReDeEP-PKS":              {"mistral": 0.678, "llama": 0.720, "qwen": 0.716},
    "ReDeEP-Combined":         {"mistral": 0.684, "llama": 0.717, "qwen": 0.716},
    "Verbalized Confidence":   {"mistral": 0.670, "llama": 0.686, "qwen": 0.730},
    "Token Entropy":           {"mistral": 0.593, "llama": 0.521, "qwen": 0.401},
    "Token Prob. Delta":       {"mistral": 0.670, "llama": 0.657, "qwen": 0.655},
    "Generation Match":        {"mistral": 0.720, "llama": 0.615, "qwen": 0.656},
    "Semantic Entropy":        {"mistral": 0.558, "llama": 0.521, "qwen": 0.562},
    "LLM Self-Judge":          {"mistral": 0.657, "llama": 0.488, "qwen": 0.660},
}
# CI half-widths in the paper are ~0.009-0.024; allow a little slack for
# reconstruction (ELI5 mirror etc.). FLAG only if clearly outside.
TOL = 0.04


def load(model, name):
    p = RES / model / f"{name}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p) as f:
        return json.load(f)


def dig(d, *paths):
    """Return the first path that resolves to a float, else None.
    Each path is a tuple of keys."""
    for path in paths:
        cur = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


def auroc_for(model, method):
    """Extract this method's AUROC for the model, or None if not available."""
    lr = load(model, "all_results_logreg")
    nr = load(model, "all_results_neural")
    base = load(model, "baseline_results")
    con = load(model, "contrastive_baselines")
    judge = load(model, "llm_judge_results")
    redeep = load(model, "redeep_results")
    se = load(model, "semantic_entropy_results")

    m = method
    if m == "Standard Probe (LogReg)" and lr:
        return dig(lr, ("best_overall", "standard", "auroc"))
    if m == "CSP Probe (LogReg)" and lr:
        return dig(lr, ("best_overall", "DECO", "auroc"))
    if m == "Standard Probe (Neural)" and nr:
        return dig(nr, ("best_overall", "standard", "auroc"))
    if m == "CSP Probe (Neural)" and nr:
        return dig(nr, ("best_overall", "DECO", "auroc"))
    if m == "Verbalized Confidence" and base:
        return dig(base, ("verbalized_confidence", "overall_auroc"))
    if m == "Token Entropy" and base:
        return dig(base, ("token_entropy", "overall_auroc"))
    if m == "Generation Match" and base:
        return dig(base, ("generation_match", "overall_auroc"))
    if m == "Token Prob. Delta" and con:
        return dig(con, ("Token Prob Delta", "auroc"))
    if m == "LLM Self-Judge" and judge:
        return dig(judge, ("overall", "auroc"), ("overall_auroc",))
    if m == "ReDeEP-ECS" and redeep:
        return dig(redeep, ("redeep_ecs", "overall_auroc"))
    if m == "ReDeEP-PKS" and redeep:
        return dig(redeep, ("redeep_pks", "overall_auroc"))
    if m == "ReDeEP-Combined" and redeep:
        return dig(redeep, ("redeep_combined", "overall_auroc"))
    if m == "Semantic Entropy" and se:
        return dig(se, ("semantic_entropy", "overall_auroc"))
    return None


def main():
    hdr = f"{'Method':26} " + " ".join(f"{m:^17}" for m in MODELS)
    print(hdr)
    print(f"{'':26} " + " ".join(f"{'mine / paper':^17}" for _ in MODELS))
    print("-" * len(hdr))
    n_flag = n_have = n_total = 0
    for method, paper_vals in PAPER.items():
        cells = []
        for model in MODELS:
            n_total += 1
            mine = auroc_for(model, method)
            paper = paper_vals[model]
            if mine is None:
                cells.append(f"{'-- / ' + f'{paper:.3f}':^17}")
                continue
            n_have += 1
            flag = "" if abs(mine - paper) <= TOL else " !"
            if flag:
                n_flag += 1
            cells.append(f"{f'{mine:.3f}/{paper:.3f}{flag}':^17}")
        print(f"{method:26} " + " ".join(cells))
    print("-" * len(hdr))
    print(f"populated {n_have}/{n_total} cells; {n_flag} FLAGGED (|mine-paper|>{TOL})")
    print("'!' = outside tolerance; '--' = result not yet produced")


if __name__ == "__main__":
    main()
