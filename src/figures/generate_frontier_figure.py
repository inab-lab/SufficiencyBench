"""
Figure: Frontier API model results with PK stratification.
Two-panel layout (one per model) for maximum clarity.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

OUT = Path(__file__).resolve().parents[2] / "results/figures"

C_OVERALL = "#6b7280"      # neutral gray (overall AUROC)
C_PK0     = "#1a2a52"      # INAB Navy Blue (PK=0 — honest estimate, primary result)
C_PK1     = "#c07840"      # INAB warm amber (PK=1)
C_PROBE   = "#4a90c4"      # INAB Sky Blue (probe ceiling band label)

PROBE_LO, PROBE_HI = 0.910, 0.973

# Data per model: list of (method, overall, pk0, pk1)
OPUS = [
    ("LLM Judge",   0.717, 0.695, 0.757),
    ("Verb. Conf.", 0.802, 0.774, 0.851),
    ("CoT Judge",   0.729, None,  None),
]
GPT = [
    ("LLM Judge",   0.727, 0.711, 0.765),
    ("Verb. Conf.", 0.752, 0.742, 0.763),
]


def draw_panel(ax, rows, title, xlim=(0.63, 0.98)):
    n = len(rows)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)

    # Probe ceiling band
    ax.axvspan(PROBE_LO, min(PROBE_HI, xlim[1]),
               color="#c5dff8", alpha=0.50, zorder=0)

    # Chance reference
    ax.axvline(0.5, color="#cccccc", lw=0.8, ls=":", zorder=1)

    ax.set_title(title, fontsize=12, fontweight="bold",
                 color="#1e3a5f", pad=8)

    for i, (method, overall, pk0, pk1) in enumerate(rows):
        y = i

        # PK range line
        if pk0 is not None and pk1 is not None:
            ax.plot([pk0, pk1], [y, y],
                    color="#cccccc", lw=7,
                    solid_capstyle="round", zorder=2, alpha=0.6)

        # Dots
        ax.scatter(overall, y, s=100, color=C_OVERALL,
                   zorder=4, edgecolors="white", linewidths=1.2, marker="o")
        if pk0 is not None:
            ax.scatter(pk0, y, s=130, color=C_PK0,
                       zorder=5, edgecolors="white", linewidths=1.2, marker="s")
        if pk1 is not None:
            ax.scatter(pk1, y, s=90, color=C_PK1,
                       zorder=5, edgecolors="white", linewidths=1.2, marker="^")

        # Label PK=0 value only (the honest estimate)
        if pk0 is not None:
            ax.text(pk0 - 0.006, y - 0.28, f"{pk0:.3f}",
                    ha="right", va="center",
                    fontsize=9, color=C_PK0, fontweight="bold")
        else:
            # CoT: just label overall
            ax.text(overall + 0.005, y + 0.22, f"{overall:.3f}",
                    ha="left", va="center",
                    fontsize=9, color=C_OVERALL)


    ax.set_xlabel("AUROC", fontsize=10)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw():
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(10, 3.8),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.38}
    )
    fig.patch.set_facecolor("white")

    draw_panel(ax_l, OPUS, "Claude Opus 4.7")
    draw_panel(ax_r, GPT,  "GPT-5.5")

    # Remove y-label from right panel (shared axis range)
    ax_r.set_xlabel("AUROC", fontsize=10)

    # Shared legend below both panels
    handles = [
        plt.scatter([], [], s=100, color=C_OVERALL, marker="o",
                    label="Overall AUROC"),
        plt.scatter([], [], s=130, color=C_PK0,     marker="s",
                    label="PK=0  (honest: model cannot answer without context)"),
        plt.scatter([], [], s=90,  color=C_PK1,     marker="^",
                    label="PK=1  (model knows answer from memory)"),
        mpatches.Patch(color="#dbeafe", alpha=0.7,
                       label="Probe ceiling  (0.910–0.973, open-weight models)"),
    ]
    fig.legend(handles=handles, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, framealpha=0.92)

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    for ext in ["pdf", "png"]:
        path = OUT / f"fig_frontier.{ext}"
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    draw()
