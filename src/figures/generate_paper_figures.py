"""
generate_paper_figures.py — Orchestrate generation of the paper's main figures.

Regenerates the SufficiencyBench paper figures from results/*.json in one pass:
  1. Framework figure   (src/figures/generate_framework_figure.py)
  2. Overview figure    (src/figures/generate_overview_figure.py)
  3. Frontier figure    (src/figures/generate_frontier_figure.py)
  4. Per-question-type AUROC bar chart (probe vs verbalized confidence)

Only figure (4) is produced directly here; (1)-(3) are delegated to their
dedicated sibling scripts so their layouts stay authoritative.

Usage:
  python src/figures/generate_paper_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import RESULTS_DIR, FIGURES_DIR, MODEL_CONFIGS

OUT = FIGURES_DIR

C_PROBE = "#1f77b4"
C_BEH   = "#c0392b"

# Question types, in paper order.
QUESTION_TYPES = ["factual", "multi_hop", "comparative", "subjective"]
QTYPE_LABELS = {
    "factual": "Factual",
    "multi_hop": "Multi-hop",
    "comparative": "Comparative",
    "subjective": "Subjective",
}


# ---------------------------------------------------------------------------
# Data loading for the per-question-type figure
# ---------------------------------------------------------------------------

def load_per_qtype_auroc():
    """Load per-question-type AUROC for the probe and verbalized-confidence
    baselines, averaged across models.

    Reads results/{model}/per_qtype_auroc.json, expected shape:
        {"standard_probe": {"factual": 0.91, ...},
         "verbalized_confidence": {"factual": 0.68, ...}}

    TODO(reconstruct): confirm the exact JSON key names / source file — the
    original loader was lost. If the file/keys differ, adjust here.
    Returns (qtypes, probe_avg, probe_std, vc_avg, vc_std).
    """
    probe_by_qtype = {qt: [] for qt in QUESTION_TYPES}
    vc_by_qtype = {qt: [] for qt in QUESTION_TYPES}

    for model_key in MODEL_CONFIGS:
        path = RESULTS_DIR / model_key / "per_qtype_auroc.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        probe = data.get("standard_probe", {})
        vc = data.get("verbalized_confidence", {})
        for qt in QUESTION_TYPES:
            if qt in probe:
                probe_by_qtype[qt].append(probe[qt])
            if qt in vc:
                vc_by_qtype[qt].append(vc[qt])

    qtypes = [qt for qt in QUESTION_TYPES if probe_by_qtype[qt] and vc_by_qtype[qt]]
    probe_avg = np.array([np.mean(probe_by_qtype[qt]) for qt in qtypes])
    probe_std = np.array([np.std(probe_by_qtype[qt]) for qt in qtypes])
    vc_avg = np.array([np.mean(vc_by_qtype[qt]) for qt in qtypes])
    vc_std = np.array([np.std(vc_by_qtype[qt]) for qt in qtypes])
    return qtypes, probe_avg, probe_std, vc_avg, vc_std


# ---------------------------------------------------------------------------
# Figure: per-question-type AUROC (probe vs verbalized confidence)
# ---------------------------------------------------------------------------

def draw_per_qtype_figure():
    qtypes, probe_avg, probe_std, vc_avg, vc_std = load_per_qtype_auroc()
    if not qtypes:
        print("  [SKIP] per-question-type figure: no per_qtype_auroc.json found")
        return

    x      = np.arange(len(qtypes))
    width  = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.2))

    bars_probe = ax.bar(x - width/2, probe_avg, width,
                        yerr=probe_std, capsize=4,
                        color=C_PROBE, alpha=0.85,
                        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
                        label="Linear probe (Standard)")
    bars_vc    = ax.bar(x + width/2, vc_avg, width,
                        yerr=vc_std, capsize=4,
                        color=C_BEH, alpha=0.75,
                        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
                        label="Verbalized confidence")

    # Annotate gap on each pair
    for i, (pv, vv) in enumerate(zip(probe_avg, vc_avg)):
        gap = pv - vv
        ax.annotate(f"+{gap:.2f}",
                    xy=(x[i], max(pv, vv) + 0.025),
                    ha="center", va="bottom",
                    fontsize=8, color="#1a4d7a", fontweight="bold")

    ax.axhline(0.5, color="#aaaaaa", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([QTYPE_LABELS.get(qt, qt) for qt in qtypes])
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Sufficiency detection by question type", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    plt.tight_layout(pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = OUT / f"fig_per_qtype.{ext}"
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all():
    OUT.mkdir(parents=True, exist_ok=True)

    # Delegate to the dedicated sibling generators.
    for mod_name in ("generate_framework_figure",
                     "generate_overview_figure",
                     "generate_frontier_figure"):
        try:
            mod = __import__(mod_name)
            mod.draw()
        except Exception as e:
            print(f"  [SKIP] {mod_name}: {e}")

    draw_per_qtype_figure()


if __name__ == "__main__":
    generate_all()
