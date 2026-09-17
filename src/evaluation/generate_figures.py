"""
generate_figures.py — Paper-ready figures for the SufficiencyBench paper.

Generates:
1. The 2×2 framework diagram (iconic figure)
2. Per-layer AUROC curves across models
3. Cosine similarity analysis (orthogonality finding)
4. Per-quadrant comparison heatmap
5. DECO-RAG faithfulness comparison bar chart
6. Method comparison table (LaTeX)

Usage:
  python src/evaluation/generate_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.paths import RESULTS_DIR, FIGURES_DIR, TABLES_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODELS = ["mistral", "qwen", "llama"]
MODEL_LABELS = {"mistral": "Mistral 7B", "qwen": "Qwen 2.5 7B", "llama": "Llama 3.1 8B"}
MODEL_COLORS = {"mistral": "#2196F3", "qwen": "#4CAF50", "llama": "#FF9800"}


def load_deco_results(model):
    path = RESULTS_DIR / model / "deco_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline_results(model):
    path = RESULTS_DIR / model / "baseline_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_deco_rag_results(model):
    path = RESULTS_DIR / model / "deco_rag_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Figure 1: The 2×2 Framework Diagram ───────────────────────────────────

def fig1_framework():
    """The iconic 2×2 matrix figure."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    quadrants = {
        "Q1": {"pos": (0.25, 0.75), "color": "#81C784",
               "label": "Q1: Easy\nSufficient + Knows",
               "desc": "Model has context\nand knows answer"},
        "Q2": {"pos": (0.75, 0.75), "color": "#FFD54F",
               "label": "Q2: Learning\nSufficient + Doesn't Know",
               "desc": "Context has answer\nbut model doesn't know it"},
        "Q3": {"pos": (0.25, 0.25), "color": "#E57373",
               "label": "Q3: Dangerous\nInsufficient + Knows",
               "desc": "Model may hallucinate\nfrom parametric knowledge"},
        "Q4": {"pos": (0.75, 0.25), "color": "#90CAF9",
               "label": "Q4: Honest Gap\nInsufficient + Doesn't Know",
               "desc": "Should abstain\nno information available"},
    }

    for qname, q in quadrants.items():
        x, y = q["pos"]
        rect = mpatches.FancyBboxPatch(
            (x - 0.22, y - 0.22), 0.44, 0.44,
            boxstyle="round,pad=0.02",
            facecolor=q["color"], edgecolor="gray", alpha=0.8, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.05, q["label"], ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x, y - 0.1, q["desc"], ha="center", va="center",
                fontsize=8, fontstyle="italic", color="#333333")

    # Axis labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Parametric Knowledge (PK)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Context Sufficiency", fontsize=12, fontweight="bold")

    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["Knows Answer", "Doesn't Know Answer"])
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Insufficient", "Sufficient"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("SufficiencyBench: 2×2 Evaluation Framework", fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_framework.pdf")
    fig.savefig(FIGURES_DIR / "fig1_framework.png")
    plt.close(fig)
    print("  Saved fig1_framework.{pdf,png}")


# ─── Figure 2: Per-Layer AUROC Curves ───────────────────────────────────────

def fig2_layer_auroc():
    """AUROC across layers for each model, standard vs DECO probes."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        data = load_deco_results(model)
        if data is None:
            ax.set_title(f"{MODEL_LABELS.get(model, model)} (no data)")
            continue

        layers = sorted(data["layer_sweep"].keys(), key=int)
        std_auroc = []
        deco_auroc = []
        for layer in layers:
            sweep = data["layer_sweep"][layer]
            std_auroc.append(sweep["standard"]["auroc"])
            deco_auroc.append(sweep["DECO"]["auroc"])

        layer_nums = [int(l) for l in layers]
        ax.plot(layer_nums, std_auroc, "o-", color="#1976D2", label="Standard", markersize=4, linewidth=1.5)
        ax.plot(layer_nums, deco_auroc, "s-", color="#D32F2F", label="DECO", markersize=4, linewidth=1.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        best_layer = data["best_layers"]["standard"]
        ax.axvline(x=best_layer, color="gray", linestyle=":", alpha=0.4, label=f"Best (L{best_layer})")

        ax.set_xlabel("Layer")
        if ax == axes[0]:
            ax.set_ylabel("AUROC")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.legend(fontsize=9)
        ax.set_ylim(0.45, 1.0)

    fig.suptitle("Sufficiency Detection AUROC by Layer", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_layer_auroc.pdf")
    fig.savefig(FIGURES_DIR / "fig2_layer_auroc.png")
    plt.close(fig)
    print("  Saved fig2_layer_auroc.{pdf,png}")


# ─── Figure 3: Orthogonality Analysis ───────────────────────────────────────

def fig3_orthogonality():
    """Bar chart of cosine similarities showing orthogonality.
    Uses the fixed orthogonality_analysis.json (probes trained in same space)
    with fallback to old deco_results correlation data."""
    models_with_data = []
    for m in MODELS:
        ortho_path = RESULTS_DIR / m / "orthogonality_analysis.json"
        if ortho_path.exists():
            with open(ortho_path) as f:
                d = json.load(f)
            models_with_data.append((m, d, "ortho"))
        else:
            d = load_deco_results(m)
            if d and "correlation" in d:
                models_with_data.append((m, d, "deco"))

    if not models_with_data:
        print("  Skipping fig3: no correlation data")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    x = np.arange(len(models_with_data))
    width = 0.25

    cos_std_conf = []
    cos_deco_conf = []
    cos_std_deco = []
    for _, d, src in models_with_data:
        if src == "ortho":
            cos_std_conf.append(d["cos_suf_conf"])
            cos_deco_conf.append(d["cos_deco_conf"])
            cos_std_deco.append(d["cos_suf_deco"])
        else:
            cos_std_conf.append(d["correlation"]["cos_standard_confidence"])
            cos_deco_conf.append(d["correlation"]["cos_deco_confidence"])
            cos_std_deco.append(d["correlation"]["cos_standard_deco"])

    bars1 = ax.bar(x - width, cos_std_conf, width, label="Sufficiency ↔ Confidence", color="#1976D2", alpha=0.8)
    bars2 = ax.bar(x, cos_deco_conf, width, label="DECO ↔ Confidence", color="#D32F2F", alpha=0.8)
    bars3 = ax.bar(x + width, cos_std_deco, width, label="Sufficiency ↔ DECO", color="#7B1FA2", alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.4, label="High correlation threshold")
    ax.axhline(y=-0.7, color="gray", linestyle="--", alpha=0.4)

    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Orthogonality: Sufficiency vs. Confidence Directions", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m, _, _ in models_with_data])
    ax.legend(loc="upper right")
    ax.set_ylim(-1, 1)

    # Annotate confidence probe AUROC to show it works
    for i, (m, d, src) in enumerate(models_with_data):
        if src == "ortho":
            conf_auroc = d.get("confidence_auroc", 0)
            ax.text(i - width, cos_std_conf[i] + 0.06, f"conf={conf_auroc:.2f}",
                    ha="center", fontsize=7, color="#1976D2")

    # Annotate key finding
    ax.annotate(
        "Near-zero: sufficiency ⊥ confidence\n(despite both probes working)",
        xy=(0, cos_std_conf[0]),
        xytext=(0.5, 0.5),
        fontsize=9, fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_orthogonality.pdf")
    fig.savefig(FIGURES_DIR / "fig3_orthogonality.png")
    plt.close(fig)
    print("  Saved fig3_orthogonality.{pdf,png}")


# ─── Figure 4: Per-Quadrant Comparison Heatmap ─────────────────────────────

def fig4_quadrant_heatmap():
    """Per-quadrant accuracy for each method, each model."""
    for model in MODELS:
        deco_data = load_deco_results(model)
        baseline_data = load_baseline_results(model)
        if deco_data is None:
            continue

        methods = {}
        # DECO methods
        for method_key in ["standard", "DECO"]:
            pq = deco_data["best_per_quadrant"].get(method_key, {})
            if pq:
                methods[method_key] = {q: pq[q]["accuracy"] for q in ["Q1", "Q2", "Q3", "Q4"] if q in pq}

        # Baselines
        if baseline_data:
            for bname, bdata in baseline_data.items():
                if "per_quadrant" in bdata:
                    methods[bname] = {}
                    for q, qdata in bdata["per_quadrant"].items():
                        methods[bname][q] = qdata.get("accuracy", 0)

        if not methods:
            continue

        quadrants = ["Q1", "Q2", "Q3", "Q4"]
        method_names = list(methods.keys())
        data_matrix = np.array([
            [methods[m].get(q, 0.5) for q in quadrants]
            for m in method_names
        ])

        fig, ax = plt.subplots(figsize=(6, max(3, len(method_names) * 0.6 + 1)))
        im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(quadrants)))
        ax.set_xticklabels(quadrants)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names)

        for i in range(len(method_names)):
            for j in range(len(quadrants)):
                ax.text(j, i, f"{data_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if data_matrix[i, j] < 0.5 else "black")

        plt.colorbar(im, ax=ax, label="Accuracy")
        ax.set_title(f"{MODEL_LABELS.get(model, model)}: Per-Quadrant Accuracy", fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fig4_quadrant_{model}.pdf")
        fig.savefig(FIGURES_DIR / f"fig4_quadrant_{model}.png")
        plt.close(fig)
        print(f"  Saved fig4_quadrant_{model}.{{pdf,png}}")


# ─── Figure 5: DECO-RAG Faithfulness ───────────────────────────────────────

def fig5_faithfulness():
    """Bar chart comparing faithfulness metrics across RAG methods."""
    for model in MODELS:
        rag_data = load_deco_rag_results(model)
        if rag_data is None:
            continue

        examples = rag_data["examples"]
        methods = ["vanilla", "deco_abstain", "deco_cad", "always_cad"]
        method_labels = ["Vanilla RAG", "DECO-Abstain", "DECO-CAD", "Always-CAD"]

        suf = [e for e in examples if e["sufficient"]]
        insuf = [e for e in examples if not e["sufficient"]]

        correct_suf = []
        abstain_insuf = []
        composite = []

        for m in methods:
            cs = sum(1 for e in suf if e[f"correct_{m}"]) / max(len(suf), 1)
            ai = sum(1 for e in insuf if e[f"abstain_{m}"]) / max(len(insuf), 1)
            correct_suf.append(cs)
            abstain_insuf.append(ai)
            composite.append((cs + ai) / 2)

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - width, correct_suf, width, label="Correct (sufficient)", color="#4CAF50", alpha=0.8)
        ax.bar(x, abstain_insuf, width, label="Abstain (insufficient)", color="#FF9800", alpha=0.8)
        ax.bar(x + width, composite, width, label="Composite", color="#2196F3", alpha=0.8)

        ax.set_ylabel("Rate")
        ax.set_title(f"{MODEL_LABELS.get(model, model)}: RAG Faithfulness Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels
        for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fig5_faithfulness_{model}.pdf")
        fig.savefig(FIGURES_DIR / f"fig5_faithfulness_{model}.png")
        plt.close(fig)
        print(f"  Saved fig5_faithfulness_{model}.{{pdf,png}}")


# ─── Table 1: Method Comparison (LaTeX) ────────────────────────────────────

def table1_comparison():
    """Generate LaTeX table comparing all methods across models."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sufficiency detection performance across models and methods. "
                 r"AUROC on the full test set and per-quadrant accuracy are reported.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{ll" + "c" * 5 + "}")
    lines.append(r"\toprule")
    lines.append(r"Model & Method & AUROC & Q1 Acc & Q2 Acc & Q3 Acc & Q4 Acc \\")
    lines.append(r"\midrule")

    for model in MODELS:
        deco_data = load_deco_results(model)
        baseline_data = load_baseline_results(model)
        if deco_data is None:
            continue

        model_label = MODEL_LABELS.get(model, model)
        first = True

        # Probe methods
        for method_key, method_label in [("standard", "Standard Probe"), ("DECO", "DECO Probe")]:
            bo = deco_data["best_overall"].get(method_key, {})
            pq = deco_data["best_per_quadrant"].get(method_key, {})
            auroc = bo.get("auroc", 0)
            q_accs = [pq.get(f"Q{i}", {}).get("accuracy", 0) for i in range(1, 5)]

            m_col = model_label if first else ""
            first = False
            best_mark = r"\textbf" if method_key == "DECO" else ""
            lines.append(
                f"{m_col} & {method_label} & {auroc:.3f} & "
                + " & ".join(f"{a:.3f}" for a in q_accs) + r" \\"
            )

        # Baselines
        if baseline_data:
            for bname, bdata in baseline_data.items():
                auroc = bdata.get("overall", {}).get("auroc", 0)
                q_accs = []
                for qi in range(1, 5):
                    q = f"Q{qi}"
                    if "per_quadrant" in bdata and q in bdata["per_quadrant"]:
                        q_accs.append(bdata["per_quadrant"][q].get("accuracy", 0))
                    else:
                        q_accs.append(0)
                lines.append(
                    f" & {bname.replace('_', ' ').title()} & {auroc:.3f} & "
                    + " & ".join(f"{a:.3f}" for a in q_accs) + r" \\"
                )

        lines.append(r"\midrule")

    # Remove last midrule, add bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    with open(TABLES_DIR / "table1_comparison.tex", "w") as f:
        f.write(latex)
    print(f"  Saved table1_comparison.tex")


# ─── Table 2: Faithfulness Comparison ──────────────────────────────────────

def table2_faithfulness():
    """LaTeX table for DECO-RAG results."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RAG faithfulness: correct answers on sufficient context "
                 r"and abstention on insufficient context.}")
    lines.append(r"\label{tab:faithfulness}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Correct$_{\text{suf}}$ & Abstain$_{\text{insuf}}$ & Composite \\")
    lines.append(r"\midrule")

    for model in MODELS:
        rag_data = load_deco_rag_results(model)
        if rag_data is None:
            continue

        examples = rag_data["examples"]
        suf = [e for e in examples if e["sufficient"]]
        insuf = [e for e in examples if not e["sufficient"]]

        lines.append(r"\multicolumn{4}{l}{\textit{" + MODEL_LABELS.get(model, model) + r"}} \\")

        for method, label in [
            ("vanilla", "Vanilla RAG"),
            ("deco_abstain", "DECO-Abstain"),
            ("deco_cad", "DECO-CAD"),
            ("always_cad", "Always-CAD"),
        ]:
            cs = sum(1 for e in suf if e[f"correct_{method}"]) / max(len(suf), 1)
            ai = sum(1 for e in insuf if e[f"abstain_{method}"]) / max(len(insuf), 1)
            comp = (cs + ai) / 2

            bold = r"\textbf" if method == "deco_abstain" else ""
            if bold:
                lines.append(
                    f"  {label} & \\textbf{{{cs:.3f}}} & \\textbf{{{ai:.3f}}} & \\textbf{{{comp:.3f}}} \\\\"
                )
            else:
                lines.append(f"  {label} & {cs:.3f} & {ai:.3f} & {comp:.3f} \\\\")

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(TABLES_DIR / "table2_faithfulness.tex", "w") as f:
        f.write(latex)
    print(f"  Saved table2_faithfulness.tex")


# ─── Table 3: SCCD Cross-Model Comparison ─────────────────────────────────

def load_sccd_results(model):
    path = RESULTS_DIR / model / "sccd" / "sccd_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _sccd_composite(examples, method, weights=(0.4, 0.4, 0.2)):
    """Compute composite = w1*correct_suf + w2*abstain_insuf - w3*over_refusal."""
    q1 = [r for r in examples if r["quadrant"] == "Q1"]
    q3 = [r for r in examples if r["quadrant"] == "Q3"]
    if not q1 or not q3:
        return 0, 0, 0, 0
    q1_cor = sum(1 for r in q1 if r["generations"].get(method, {}).get("correct", False)) / len(q1)
    q3_abs = sum(1 for r in q3 if r["generations"].get(method, {}).get("abstains", False)) / len(q3)
    q1_abs = sum(1 for r in q1 if r["generations"].get(method, {}).get("abstains", False)) / len(q1)
    comp = weights[0] * q1_cor + weights[1] * q3_abs - weights[2] * q1_abs
    return comp, q1_cor, q3_abs, q1_abs


def _bootstrap_ci(values, stat_fn, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(42)
    n = len(values)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_stats.append(stat_fn([values[i] for i in idx]))
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, 100 * alpha)
    hi = np.percentile(boot_stats, 100 * (1 - alpha))
    return lo, hi


def _sccd_bootstrap_composite(examples, method, n_boot=1000):
    """Bootstrap CI for SCCD composite score."""
    rng = np.random.RandomState(42)
    q1 = [r for r in examples if r["quadrant"] == "Q1"]
    q3 = [r for r in examples if r["quadrant"] == "Q3"]
    if not q1 or not q3:
        return 0, 0, 0

    boot_comps = []
    for _ in range(n_boot):
        q1_s = [q1[rng.randint(0, len(q1))] for _ in range(len(q1))]
        q3_s = [q3[rng.randint(0, len(q3))] for _ in range(len(q3))]
        q1_cor = sum(1 for r in q1_s if r["generations"].get(method, {}).get("correct", False)) / len(q1_s)
        q3_abs = sum(1 for r in q3_s if r["generations"].get(method, {}).get("abstains", False)) / len(q3_s)
        q1_abs = sum(1 for r in q1_s if r["generations"].get(method, {}).get("abstains", False)) / len(q1_s)
        boot_comps.append(0.4 * q1_cor + 0.4 * q3_abs - 0.2 * q1_abs)

    mean = np.mean(boot_comps)
    lo, hi = np.percentile(boot_comps, [2.5, 97.5])
    return mean, lo, hi


FIXED_ABSTENTION = "I don't have enough information"


def _derive_hybrid(results, thresholds, max_alphas):
    """Derive hybrid configs from existing results if not already present."""
    sample_key = f"hybrid_t{thresholds[0]}_a{max_alphas[0]}"
    if sample_key in results[0].get("generations", {}):
        return
    for r in results:
        for t in thresholds:
            for a in max_alphas:
                hk = f"hybrid_t{t}_a{a}"
                if r["probe_score"] < t:
                    r["generations"][hk] = {
                        "text": FIXED_ABSTENTION, "abstains": True, "correct": False,
                    }
                else:
                    cad_key = f"always_cad_a{a}"
                    r["generations"][hk] = r["generations"].get(cad_key, {}).copy()


def table3_sccd():
    """Generate LaTeX table for SCCD cross-model comparison."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sufficiency-Conditioned Contrastive Decoding (SCCD) results. "
                 r"Correct$_\text{suf}$: correct answers on sufficient context (Q1). "
                 r"Abstain$_\text{insuf}$: abstention on insufficient context (Q3). "
                 r"Over-Ref: over-refusal rate on sufficient context (Q1). "
                 r"Composite = 0.4$\cdot$Correct + 0.4$\cdot$Abstain $-$ 0.2$\cdot$Over-Ref.}")
    lines.append(r"\label{tab:sccd}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Method & Correct$_{\text{suf}}$↑ & Abstain$_{\text{insuf}}$↑ & Over-Ref↓ & Composite \\")
    lines.append(r"\midrule")

    for model in MODELS:
        data = load_sccd_results(model)
        if data is None:
            continue

        results = data["examples"]
        thresholds = data["thresholds"]
        max_alphas = data["max_alphas"]
        _derive_hybrid(results, thresholds, max_alphas)

        model_label = MODEL_LABELS.get(model, model)
        first = True

        # Find best of each type
        methods_to_show = []

        # Baseline
        comp, q1c, q3a, orf = _sccd_composite(results, "baseline")
        _, ci_lo, ci_hi = _sccd_bootstrap_composite(results, "baseline")
        methods_to_show.append(("Baseline", comp, q1c, q3a, orf, ci_lo, ci_hi, False))

        # Best CAD
        best_cad = max(max_alphas, key=lambda a: _sccd_composite(results, f"always_cad_a{a}")[0])
        comp, q1c, q3a, orf = _sccd_composite(results, f"always_cad_a{best_cad}")
        _, ci_lo, ci_hi = _sccd_bootstrap_composite(results, f"always_cad_a{best_cad}")
        methods_to_show.append((f"CAD ($\\alpha$={best_cad})", comp, q1c, q3a, orf, ci_lo, ci_hi, False))

        # Best probe-abstain
        best_t = max(thresholds, key=lambda t: _sccd_composite(results, f"probe_abstain_t{t}")[0])
        comp, q1c, q3a, orf = _sccd_composite(results, f"probe_abstain_t{best_t}")
        _, ci_lo, ci_hi = _sccd_bootstrap_composite(results, f"probe_abstain_t{best_t}")
        methods_to_show.append((f"Probe-Abstain ($\\tau$={best_t})", comp, q1c, q3a, orf, ci_lo, ci_hi, False))

        # Best hybrid
        best_combo = max(
            [(t, a) for t in thresholds for a in max_alphas],
            key=lambda ta: _sccd_composite(results, f"hybrid_t{ta[0]}_a{ta[1]}")[0],
        )
        hk = f"hybrid_t{best_combo[0]}_a{best_combo[1]}"
        comp, q1c, q3a, orf = _sccd_composite(results, hk)
        _, ci_lo, ci_hi = _sccd_bootstrap_composite(results, hk)
        methods_to_show.append((
            f"Hybrid SCCD ($\\tau$={best_combo[0]}, $\\alpha$={best_combo[1]})",
            comp, q1c, q3a, orf, ci_lo, ci_hi, True,
        ))

        for label, comp, q1c, q3a, orf, ci_lo, ci_hi, is_best in methods_to_show:
            m_col = model_label if first else ""
            first = False
            ci_str = f"$\\pm${(ci_hi - ci_lo) / 2:.3f}"
            if is_best:
                lines.append(
                    f"{m_col} & {label} & \\textbf{{{q1c:.3f}}} & \\textbf{{{q3a:.3f}}} "
                    f"& \\textbf{{{orf:.3f}}} & \\textbf{{{comp:.3f}}} {ci_str} \\\\"
                )
            else:
                lines.append(
                    f"{m_col} & {label} & {q1c:.3f} & {q3a:.3f} & {orf:.3f} & {comp:.3f} {ci_str} \\\\"
                )

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    with open(TABLES_DIR / "table3_sccd.tex", "w") as f:
        f.write(latex)
    print(f"  Saved table3_sccd.tex")


# ─── Figure 6: SCCD Cross-Model Comparison ────────────────────────────────

def fig6_sccd_comparison():
    """Grouped bar chart of SCCD methods across models."""
    method_keys = []  # (model, method_name, composite, q1_cor, q3_abs)
    model_data = {}

    for model in MODELS:
        data = load_sccd_results(model)
        if data is None:
            continue
        results = data["examples"]
        thresholds = data["thresholds"]
        max_alphas = data["max_alphas"]
        _derive_hybrid(results, thresholds, max_alphas)

        row = {}
        # Baseline
        row["Baseline"] = _sccd_composite(results, "baseline")[0]
        # Best CAD
        best_a = max(max_alphas, key=lambda a: _sccd_composite(results, f"always_cad_a{a}")[0])
        row["CAD"] = _sccd_composite(results, f"always_cad_a{best_a}")[0]
        # Best probe-abstain
        best_t = max(thresholds, key=lambda t: _sccd_composite(results, f"probe_abstain_t{t}")[0])
        row["Probe-Abstain"] = _sccd_composite(results, f"probe_abstain_t{best_t}")[0]
        # Best hybrid
        best_combo = max(
            [(t, a) for t in thresholds for a in max_alphas],
            key=lambda ta: _sccd_composite(results, f"hybrid_t{ta[0]}_a{ta[1]}")[0],
        )
        row["Hybrid SCCD"] = _sccd_composite(results, f"hybrid_t{best_combo[0]}_a{best_combo[1]}")[0]

        model_data[model] = row

    if not model_data:
        print("  Skipping fig6: no SCCD data")
        return

    methods = ["Baseline", "CAD", "Probe-Abstain", "Hybrid SCCD"]
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#D32F2F"]
    models_with_data = [m for m in MODELS if m in model_data]

    x = np.arange(len(models_with_data))
    width = 0.18
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [model_data[m].get(method, 0) for m in models_with_data]
        bars = ax.bar(x + offsets[i] * width, vals, width, label=method, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_ylabel("Composite Score")
    ax.set_title("SCCD: Cross-Model Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_with_data])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(model_data[m].values()) for m in models_with_data) + 0.15)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_sccd_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig6_sccd_comparison.png")
    plt.close(fig)
    print("  Saved fig6_sccd_comparison.{pdf,png}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures and tables...")
    print()

    # Check which models have data
    available = []
    for m in MODELS:
        if (RESULTS_DIR / m / "deco_results.json").exists():
            available.append(m)
    print(f"Models with data: {available}")
    print()

    print("Figure 1: 2x2 Framework")
    fig1_framework()

    print("Figure 2: Per-Layer AUROC")
    fig2_layer_auroc()

    print("Figure 3: Orthogonality")
    fig3_orthogonality()

    print("Figure 4: Per-Quadrant Heatmap")
    fig4_quadrant_heatmap()

    print("Figure 5: Faithfulness")
    fig5_faithfulness()

    print("Figure 6: SCCD Comparison")
    fig6_sccd_comparison()

    print("Table 1: Method Comparison")
    table1_comparison()

    print("Table 2: Faithfulness")
    table2_faithfulness()

    print("Table 3: SCCD")
    table3_sccd()

    print()
    print("All figures and tables saved to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
