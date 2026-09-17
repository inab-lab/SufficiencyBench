"""
Figure 1: SufficiencyBench 2×2 Evaluation Framework — INAB branding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/figures"

NAVY  = "#1a2a52"
SKY   = "#94bff6"
BONE  = "#f4f0eb"
GREEN = "#C9D678"


def draw():
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Each quadrant covers exactly half the axis range in both dimensions.
    # (x_left, y_bottom, bg, fg, title, subtitle, desc)
    quads = [
        (0,   0.5, GREEN,  NAVY,    "Q1: Easy",
         "Sufficient + Knows",
         "Model has context\nand knows the answer"),
        (0.5, 0.5, SKY,    NAVY,    "Q2: Learning",
         "Sufficient + Doesn't Know",
         "Context contains the answer\nbut model lacks the knowledge"),
        (0,   0,   NAVY,   "white", "Q3: Dangerous",
         "Insufficient + Knows",
         "Model may confabulate\nfrom parametric memory"),
        (0.5, 0,   BONE,   NAVY,    "Q4: Honest Gap",
         "Insufficient + Doesn't Know",
         "System should abstain;\nno information available"),
    ]

    for x, y, bg, fg, title, subtitle, desc in quads:
        ax.add_patch(mpatches.Rectangle(
            (x, y), 0.5, 0.5,
            facecolor=bg, edgecolor=NAVY, linewidth=2.5, zorder=1,
        ))
        cx, cy = x + 0.25, y + 0.25

        ax.text(cx, cy + 0.10, title,
                ha="center", va="center", fontsize=13.5,
                fontweight="bold", color=fg, zorder=3)
        ax.text(cx, cy - 0.01, subtitle,
                ha="center", va="center", fontsize=10.5,
                color=fg, alpha=0.88, zorder=3)

        desc_color = "#90a8c8" if bg == NAVY else "#505050"
        ax.text(cx, cy - 0.13, desc,
                ha="center", va="center", fontsize=9,
                fontstyle="italic", color=desc_color,
                linespacing=1.5, zorder=3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(
        ["Knows Answer  (PK = 1)", "Doesn't Know  (PK = 0)"],
        fontsize=11, fontweight="bold", color=NAVY,
    )
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(
        ["Insufficient", "Sufficient"],
        fontsize=11, fontweight="bold", color=NAVY,
    )
    ax.tick_params(length=0, pad=8)

    ax.set_xlabel("Parametric Knowledge (PK)", fontsize=13,
                  fontweight="bold", color=NAVY, labelpad=12)
    ax.set_ylabel("Context Sufficiency", fontsize=13,
                  fontweight="bold", color=NAVY, labelpad=12)
    ax.set_title("SufficiencyBench: 2×2 Evaluation Framework",
                 fontsize=15, fontweight="bold", color=NAVY, pad=16)

    plt.tight_layout(pad=1.2)

    for stem in ["fig1_framework", "fig2_framework"]:
        for ext in ["pdf", "png"]:
            p = OUT / f"{stem}.{ext}"
            plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
            print(f"Saved {p}")
    plt.close()


if __name__ == "__main__":
    draw()
