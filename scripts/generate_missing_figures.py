#!/usr/bin/env python
"""generate_missing_figures.py — regenerate the SufficiencyBench paper figures that
were lost, writing each to BOTH results/figures/ and the Overleaf figures dir with
the EXACT filename main_tacl.tex references.

Produces:
  A (data-driven, from results/*.json):
    fig_main_comparison.{png,pdf}   14-method AUROC comparison (llama/mistral/qwen)
    fig_layer_curve.{png,pdf}       CSP probe AUROC per layer, star at val-best layer
    fig_qtype_breakdown.{png,pdf}   probe vs verbalized-confidence AUROC by qtype
    results/per_qtype_auroc.json    + per-model results/{model}/per_qtype_auroc.json
  B (schematic redraws matching tex captions):
    fig_construction_example.{png}  paired sufficient/insufficient construction
    fig_methods_overview.{png}      where each method family reads from the transformer

Run: ~/miniconda3/envs/csp/bin/python scripts/generate_missing_figures.py
"""
import json
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/inab/Documents/MSD_Project/paper1")
RES = ROOT / "results"
FIG_RESULTS = RES / "figures"
FIG_PAPER = ROOT / "paper" / "sufficiencybench_overleaf_v10" / "figures"
DATA = ROOT / "data" / "experiments" / "sufficiency_bench"
MODELS = ["llama", "mistral", "qwen"]
MODEL_LABEL = {"llama": "Llama 3.1 8B", "mistral": "Mistral 7B", "qwen": "Qwen 2.5 7B"}
MODEL_MARK = {"llama": "o", "mistral": "s", "qwen": "^"}
N_LAYERS = {"llama": 32, "mistral": 32, "qwen": 28}

sys.path.insert(0, str(ROOT / "scripts"))
import assemble_table1 as A  # reuse the method->file AUROC extraction

for d in (FIG_RESULTS, FIG_PAPER):
    d.mkdir(parents=True, exist_ok=True)


def savefig(fig, name, pdf=True):
    exts = ["png", "pdf"] if pdf else ["png"]
    for base in (FIG_RESULTS, FIG_PAPER):
        for ext in exts:
            fig.savefig(base / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                        facecolor="white")
    print(f"  saved {name}.{'/'.join(exts)} -> results/figures + paper/figures")


# ---------------------------------------------------------------------------
# Family definitions (order + colours), and paper Table 1 CI half-widths.
# ---------------------------------------------------------------------------
FAMILY = {
    "Hidden-state probe":   ("#1f4e8c", ["Standard Probe (LogReg)", "CSP Probe (LogReg)",
                                          "Standard Probe (Neural)", "CSP Probe (Neural)"]),
    "Mechanistic (attn.)":  ("#6a3d9a", ["ReDeEP-ECS", "ReDeEP-PKS", "ReDeEP-Combined"]),
    "Generation-based":     ("#e08214", ["Verbalized Confidence", "Token Entropy",
                                          "Token Prob. Delta", "Generation Match",
                                          "Semantic Entropy"]),
    "Behavioural (output)": ("#b2182b", ["LLM Self-Judge", "Embedding Similarity"]),
}
METHOD_FAMILY = {m: fam for fam, (_, ms) in FAMILY.items() for m in ms}
METHOD_COLOR = {m: c for _, (c, ms) in FAMILY.items() for m in ms}
ORACLE = {"Token Prob. Delta", "Generation Match"}

# Paper Table 1 CI half-widths (0.5 * reported +/-), per model.
PAPER_CI = {
    "Standard Probe (LogReg)": (0.011, 0.011, 0.009),
    "CSP Probe (LogReg)":      (0.011, 0.012, 0.008),
    "Standard Probe (Neural)": (0.010, 0.010, 0.009),
    "CSP Probe (Neural)":      (0.011, 0.011, 0.009),
    "ReDeEP-ECS":              (0.024, 0.023, 0.023),
    "ReDeEP-PKS":              (0.021, 0.022, 0.021),
    "ReDeEP-Combined":         (0.021, 0.022, 0.021),
    "Verbalized Confidence":   (0.022, 0.022, 0.021),
    "Token Entropy":           (0.024, 0.023, 0.023),
    "Token Prob. Delta":       (0.023, 0.022, 0.022),
    "Generation Match":        (0.023, 0.021, 0.023),
    "Semantic Entropy":        (0.024, 0.024, 0.024),
    "LLM Self-Judge":          (0.024, 0.023, 0.022),
    "Embedding Similarity":    (0.022, 0.022, 0.022),
}
# order of models inside PAPER_CI tuples:
CI_ORDER = ["llama", "mistral", "qwen"]
# Embedding Similarity has no regenerated result (constant paper baseline); use paper value.
PAPER_EMBED = {"llama": 0.662, "mistral": 0.662, "qwen": 0.662}


# ---------------------------------------------------------------------------
# A1. fig_main_comparison
# ---------------------------------------------------------------------------
def fig_main_comparison():
    rows = []  # (method, {model: auroc}, {model: ci})
    for method in PAPER_CI:
        vals, cis = {}, {}
        for i, model in enumerate(CI_ORDER):
            if method == "Embedding Similarity":
                v = PAPER_EMBED[model]
            else:
                v = A.auroc_for(model, method)
            if v is None:
                continue
            vals[model] = v
            cis[model] = PAPER_CI[method][i]
        if vals:
            rows.append((method, vals, cis))
    # sort by mean AUROC ascending (best on top of horizontal plot)
    rows.sort(key=lambda r: np.mean(list(r[1].values())))

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    yticklabels = []
    for y, (method, vals, cis) in enumerate(rows):
        color = METHOD_COLOR[method]
        xs = [vals[m] for m in CI_ORDER if m in vals]
        ax.plot([min(xs), max(xs)], [y, y], color=color, lw=2.2, alpha=0.45,
                solid_capstyle="round", zorder=2)
        for model in CI_ORDER:
            if model not in vals:
                continue
            ax.errorbar(vals[model], y, xerr=cis[model], fmt=MODEL_MARK[model],
                        color=color, ecolor=color, elinewidth=1.1, capsize=2.5,
                        markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                        zorder=3)
        star = "$^*$" if method in ORACLE else ""
        yticklabels.append(method + star)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_xlabel("AUROC (SufficiencyBench test)", fontsize=11)
    ax.set_xlim(0.38, 1.0)
    ax.axvline(0.5, color="#999999", ls=":", lw=0.8, zorder=1)
    ax.set_title("Sufficiency detection: 14 methods across 3 models", fontsize=12,
                 pad=10)
    ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    from matplotlib.lines import Line2D
    fam_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=9, label=fam) for fam, (c, _) in FAMILY.items()]
    mod_handles = [Line2D([0], [0], marker=MODEL_MARK[m], color="#444444",
                          linestyle="none", markersize=8, label=MODEL_LABEL[m])
                   for m in CI_ORDER]
    leg1 = ax.legend(handles=fam_handles, title="Method family", loc="lower right",
                     fontsize=8.5, title_fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=mod_handles, title="Model", loc="lower right",
              bbox_to_anchor=(0.72, 0.0), fontsize=8.5, title_fontsize=9,
              framealpha=0.95)
    ax.text(0.01, -0.065, "$^*$Oracle methods (require gold-answer access): "
            "diagnostic upper bounds only.", transform=ax.transAxes, fontsize=8,
            color="#555555")
    fig.tight_layout()
    savefig(fig, "fig_main_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A2. fig_layer_curve
# ---------------------------------------------------------------------------
def fig_layer_curve():
    colors = {"llama": "#1f77b4", "mistral": "#2ca02c", "qwen": "#d62728"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for model in MODELS:
        d = json.load(open(RES / model / "all_results_logreg.json"))
        sweep = d["layer_sweep"]
        best = d["best_layers"]["DECO"]
        layers = sorted(int(k) for k in sweep)
        aurocs = [sweep[str(l)]["DECO"]["auroc"] for l in layers]
        # normalise x to network depth fraction for comparability
        frac = [l / (N_LAYERS[model] - 1) for l in layers]
        ax.plot(frac, aurocs, "-", color=colors[model], lw=1.8, alpha=0.9,
                label=f"{MODEL_LABEL[model]} ({N_LAYERS[model]} layers)")
        bx = best / (N_LAYERS[model] - 1)
        by = sweep[str(best)]["DECO"]["auroc"]
        ax.plot(bx, by, "*", color=colors[model], markersize=17,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        ax.annotate(f"L{best}", (bx, by), textcoords="offset points",
                    xytext=(4, 8), fontsize=8.5, color=colors[model],
                    fontweight="bold")
    ax.axhline(0.73, color="#888888", ls="--", lw=1.0)
    ax.text(0.015, 0.735, "behavioural ceiling (0.73)", fontsize=8,
            color="#666666")
    ax.set_xlabel("Residual-stream layer (fraction of network depth)", fontsize=11)
    ax.set_ylabel("CSP probe AUROC", fontsize=11)
    ax.set_title("CSP direction is strongest at mid-depth", fontsize=12)
    ax.set_ylim(0.60, 1.0)
    ax.set_xlim(-0.01, 1.01)
    ax.legend(loc="lower center", fontsize=9, framealpha=0.95)
    ax.grid(lw=0.4, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig_layer_curve")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A3. per_qtype_auroc.json  +  fig_qtype_breakdown
# ---------------------------------------------------------------------------
QTYPES = ["factual", "multi_hop", "comparative", "subjective"]
QLABEL = {"factual": "Factual", "multi_hop": "Multi-hop",
          "comparative": "Comparative", "subjective": "Subjective"}


def load_qtype_map():
    """base condition id (e.g. 'squad_133') -> question_type"""
    test = json.load(open(DATA / "test.json"))
    return {it["id"]: it["question_type"] for it in test}


def build_per_qtype():
    qmap = load_qtype_map()
    per_model = {}
    for model in MODELS:
        # probe (Standard LogReg): AUROC per qtype already computed at best layer
        lr = json.load(open(RES / model / "all_results_logreg.json"))
        probe_q = {qt: lr["best_per_question_type"]["standard"][qt]["auroc"]
                   for qt in QTYPES}
        # verbalized confidence: compute per-qtype AUROC from stored per-example scores
        vc = json.load(open(RES / model / "scores" / "baseline_scores.json"))[
            "verbalized_confidence"]
        scores, labels, meta = vc["scores"], vc["labels"], vc["metadata"]
        by_qt = {qt: ([], []) for qt in QTYPES}  # (scores, labels)
        for s, y, m in zip(scores, labels, meta):
            cid = m["condition_id"]
            base = cid.rsplit("_", 1)[0]  # strip _suf/_insuf
            qt = qmap.get(base)
            if qt in by_qt:
                by_qt[qt][0].append(s)
                by_qt[qt][1].append(y)
        vc_q = {}
        for qt in QTYPES:
            ss, yy = by_qt[qt]
            vc_q[qt] = float(roc_auc_score(yy, ss)) if len(set(yy)) > 1 else None
        per_model[model] = {"standard_probe": probe_q,
                            "verbalized_confidence": vc_q}
        with open(RES / model / "per_qtype_auroc.json", "w") as f:
            json.dump(per_model[model], f, indent=2)
        print(f"  wrote results/{model}/per_qtype_auroc.json")
    with open(RES / "per_qtype_auroc.json", "w") as f:
        json.dump(per_model, f, indent=2)
    print("  wrote results/per_qtype_auroc.json")
    return per_model


def fig_qtype_breakdown(per_model):
    probe = {qt: [per_model[m]["standard_probe"][qt] for m in MODELS] for qt in QTYPES}
    vc = {qt: [per_model[m]["verbalized_confidence"][qt] for m in MODELS
               if per_model[m]["verbalized_confidence"][qt] is not None]
          for qt in QTYPES}
    p_avg = np.array([np.mean(probe[qt]) for qt in QTYPES])
    p_std = np.array([np.std(probe[qt]) for qt in QTYPES])
    v_avg = np.array([np.mean(vc[qt]) for qt in QTYPES])
    v_std = np.array([np.std(vc[qt]) for qt in QTYPES])

    x = np.arange(len(QTYPES))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w/2, p_avg, w, yerr=p_std, capsize=4, color="#1f4e8c", alpha=0.9,
           error_kw={"elinewidth": 1.1, "ecolor": "#333"}, label="Linear probe (Standard)")
    ax.bar(x + w/2, v_avg, w, yerr=v_std, capsize=4, color="#b2182b", alpha=0.85,
           error_kw={"elinewidth": 1.1, "ecolor": "#333"}, label="Verbalized confidence")
    for i in range(len(QTYPES)):
        gap = p_avg[i] - v_avg[i]
        ax.annotate(f"+{gap:.2f}", (x[i], max(p_avg[i], v_avg[i]) + p_std[i] + 0.02),
                    ha="center", va="bottom", fontsize=9, color="#1a4d7a",
                    fontweight="bold")
    ax.axhline(0.5, color="#aaaaaa", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([QLABEL[qt] for qt in QTYPES])
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.4, 1.08)
    ax.set_title("Probe advantage grows with question complexity\n(averaged across "
                 "Llama, Mistral, Qwen)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig_qtype_breakdown")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B1. fig_construction_example  (schematic)
# ---------------------------------------------------------------------------
def fig_construction_example():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    question = ("Q: Who was in charge of the papal army\n"
                "     in the War of Barbastro?")
    gold = "Gold answer: William of Montreuil"
    common = ("The War of Barbastro (1064) was a pan-European\n"
              "military expedition, sanctioned by Pope Alexander II,\n"
              "to take the city of Barbastro. Knights came from\n"
              "many regions to join the campaign.")
    suf_sent = ("William of Montreuil led the papal contingent\n"
                "and commanded the army during the siege.")
    insuf_sent = ("The campaign is considered a precursor to the\n"
                  "later Reconquista and the Crusades.")

    def panel(ax, title, tcolor, sentence, scolor, sbg, tag):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                     boxstyle="round,pad=0.01", linewidth=1.6,
                     edgecolor=tcolor, facecolor="white"))
        ax.text(0.5, 0.93, title, ha="center", va="top", fontsize=12.5,
                fontweight="bold", color=tcolor)
        ax.text(0.06, 0.82, question, ha="left", va="top", fontsize=9.5,
                fontweight="bold")
        ax.text(0.06, 0.63, common, ha="left", va="top", fontsize=9,
                color="#333333")
        # highlighted swapped sentence
        ax.add_patch(FancyBboxPatch((0.05, 0.17), 0.90, 0.20,
                     boxstyle="round,pad=0.008", linewidth=1.0,
                     edgecolor=scolor, facecolor=sbg, alpha=0.9))
        ax.text(0.08, 0.34, tag, ha="left", va="top", fontsize=8.2,
                style="italic", color=scolor, fontweight="bold")
        ax.text(0.08, 0.29, sentence, ha="left", va="top", fontsize=9,
                color="#222222")
        ax.text(0.06, 0.09, gold, ha="left", va="top", fontsize=8.5,
                color="#555555", style="italic")

    panel(axes[0], "Sufficient context", "#1a7a3a", suf_sent, "#1a7a3a",
          "#d7f0dd", "answer-bearing sentence (kept)")
    panel(axes[1], "Insufficient context", "#b2182b", insuf_sent, "#b2182b",
          "#f7d7d7", "topically-similar swap (answer removed)")
    fig.suptitle("Paired construction: identical context except the answer sentence "
                 "(passes all 3 quality filters)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_construction_example", pdf=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B2. fig_methods_overview  (schematic)
# ---------------------------------------------------------------------------
def fig_methods_overview():
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # transformer stack (bottom -> top)
    layer_x, layer_w = 1.2, 2.4
    ys = np.linspace(1.2, 7.6, 5)
    for i, y in enumerate(ys):
        ax.add_patch(FancyBboxPatch((layer_x, y), layer_w, 1.0,
                     boxstyle="round,pad=0.02", linewidth=1.2,
                     edgecolor="#333", facecolor="#eef2f8"))
        ax.text(layer_x + layer_w/2, y + 0.5, f"Transformer block {i+1}",
                ha="center", va="center", fontsize=8.5)
    # residual stream arrow
    ax.add_patch(FancyArrowPatch((layer_x + layer_w/2, 0.7),
                 (layer_x + layer_w/2, 8.8), arrowstyle="-|>",
                 mutation_scale=18, lw=2.2, color="#1f4e8c"))
    ax.text(layer_x + layer_w/2, 9.0, "residual stream", ha="center",
            fontsize=9, color="#1f4e8c", fontweight="bold")
    ax.text(layer_x + layer_w/2, 0.35, "input: question + context", ha="center",
            fontsize=8.5, color="#555")
    # output tokens box
    ax.add_patch(FancyBboxPatch((layer_x, 8.9), layer_w, 0.0,
                 boxstyle="round", linewidth=0))
    ax.text(layer_x + layer_w/2, 9.45, "output tokens", ha="center", fontsize=9,
            color="#b2182b", fontweight="bold")

    def reader(y, color, title, desc, source_xy):
        bx = 6.1
        ax.add_patch(FancyBboxPatch((bx, y - 0.55), 3.6, 1.1,
                     boxstyle="round,pad=0.03", linewidth=1.4,
                     edgecolor=color, facecolor="white"))
        ax.text(bx + 0.15, y + 0.22, title, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=color)
        ax.text(bx + 0.15, y - 0.22, desc, ha="left", va="center", fontsize=7.8,
                color="#444")
        ax.add_patch(FancyArrowPatch(source_xy, (bx, y), arrowstyle="-|>",
                     mutation_scale=14, lw=1.6, color=color,
                     connectionstyle="arc3,rad=-0.15"))

    mid = layer_x + layer_w
    reader(8.9, "#b2182b", "Behavioural / VC / LLM-judge",
           "read OUTPUT TOKENS only", (mid, 8.9))
    reader(6.6, "#e08214", "Generation-based (entropy,\nsemantic entropy, gen-match)",
           "read output-token distributions", (mid, 7.2))
    reader(4.4, "#6a3d9a", "ReDeEP (mechanistic)",
           "reads ATTENTION patterns", (mid, 4.6))
    reader(2.0, "#1f4e8c", "Hidden-state probe (ours)",
           "reads RESIDUAL STREAM directly", (mid, 2.2))

    # "mixing point" annotation
    ax.axhline(5.9, xmin=0.11, xmax=0.36, color="#888", ls="--", lw=1.0)
    ax.text(mid + 0.1, 5.95, "contextual + parametric signals mixed above here",
            fontsize=7.6, color="#666", style="italic")
    ax.set_title("Where each detection-method family reads from the transformer",
                 fontsize=12.5, pad=12)
    ax.text(5.0, 0.1, "All methods except the hidden-state probe are DOWNSTREAM of the "
            "point where contextual and parametric signals mix.",
            ha="center", fontsize=8.3, color="#555")
    fig.tight_layout()
    savefig(fig, "fig_methods_overview", pdf=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. fig_steering_mistral  (from results/steering_mistral.json)
# ---------------------------------------------------------------------------
def fig_steering_mistral():
    d = json.load(open(RES / "steering_mistral.json"))
    alphas = d["alphas"]
    qcolors = {"Q1": "#1f77b4", "Q2": "#2ca02c", "Q3": "#d62728", "Q4": "#9467bd"}
    qlab = {"Q1": "Q1 (suf, knows)", "Q2": "Q2 (suf, no-PK)",
            "Q3": "Q3 (insuf, knows)", "Q4": "Q4 (insuf, no-PK)"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        ax1.plot(alphas, d["correctness"][q], "-o", color=qcolors[q], lw=1.8,
                 markersize=6, label=qlab[q])
        ax2.plot(alphas, d["abstention"][q], "-o", color=qcolors[q], lw=1.8,
                 markersize=6, label=qlab[q])
    for ax in (ax1, ax2):
        ax.axvline(0, color="#bbbbbb", ls=":", lw=0.9)
        ax.grid(lw=0.4, alpha=0.4)
        ax.set_ylim(-0.03, 1.03)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax1.set_ylabel("Generation correctness", fontsize=11)
    ax2.set_ylabel("Abstention rate", fontsize=11)
    ax2.set_xlabel(r"Steering coefficient $\alpha$  "
                   r"($-$ insufficient pole  $\rightarrow$  $+$ sufficient pole)",
                   fontsize=10.5)
    ax1.set_xticks(alphas)
    ax1.set_title("Activation steering of the CSP direction (Mistral 7B, layer 12,\n"
                  "5-layer window) — graded, symmetric, collapse at $|\\alpha|=2$",
                  fontsize=11)
    ax1.legend(loc="lower center", ncol=2, fontsize=8, framealpha=0.95)
    fig.tight_layout()
    savefig(fig, "fig_steering_mistral")
    plt.close(fig)


if __name__ == "__main__":
    print("[A1] fig_main_comparison"); fig_main_comparison()
    print("[A2] fig_layer_curve"); fig_layer_curve()
    print("[A3] per_qtype_auroc + fig_qtype_breakdown")
    pm = build_per_qtype(); fig_qtype_breakdown(pm)
    print("[B1] fig_construction_example"); fig_construction_example()
    print("[B2] fig_methods_overview"); fig_methods_overview()
    if (RES / "steering_mistral.json").exists():
        print("[C] fig_steering_mistral"); fig_steering_mistral()
    print("DONE")
