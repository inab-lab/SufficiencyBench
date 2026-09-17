"""
Overview figure: AUROC across all model scales (7B → 72B → frontier API).
Probe rises with scale. Behavioural ceiling stays flat.
Even frontier models cannot close the gap.
The uncontrolled benchmark range is shown as a reference band.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/figures"

C_PROBE = "#1f77b4"
C_BEH   = "#c0392b"
C_BAND  = "#bbbbbb"

# ── Data ─────────────────────────────────────────────────────────────────────
# (model_label, size_group, probe_auroc, vc_auroc, is_frontier)
# size_group: 0=7B-8B, 1=32B, 2=72B, 3=frontier
MODELS = [
    ("Llama 3.1\n8B",       0,  0.923, 0.686, False),
    ("Mistral\n7B",          0,  0.910, 0.670, False),
    ("Qwen 2.5\n7B",         0,  0.946, 0.730, False),
    ("DeepSeek-R1\n32B",     1,  0.963, 0.560, False),
    ("Qwen 2.5\n72B",        2,  0.973, 0.774, False),
    ("Claude\nOpus 4.7",     3,  None,  0.774, True),   # PK=0 VC
    ("GPT-5.5",              3,  None,  0.742, True),   # PK=0 VC
]

# Frontier models: VC is the PK=0 (honest) number
# Uncontrolled benchmark range (prior work on HotpotQA / SQuAD)
UNCONTROLLED_LO = 0.78
UNCONTROLLED_HI = 0.91


def draw():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("white")

    # ── x positions: group models with gaps between size classes ─────────────
    group_centers = {0: 1.0, 1: 3.2, 2: 4.4, 3: 6.0}
    group_offsets = {
        0: [-0.42, 0.0, 0.42],   # three 7B models
        1: [0.0],
        2: [0.0],
        3: [-0.3, 0.3],
    }
    group_counts  = {0: 0, 1: 0, 2: 0, 3: 0}

    x_probe, y_probe, lbl_probe = [], [], []
    x_vc,    y_vc,    lbl_vc    = [], [], []
    x_front, y_front, lbl_front = [], [], []

    for (label, grp, probe, vc, is_front) in MODELS:
        idx = group_counts[grp]
        xc  = group_centers[grp] + group_offsets[grp][idx]
        group_counts[grp] += 1

        if probe is not None:
            x_probe.append(xc)
            y_probe.append(probe)
            lbl_probe.append(label)

        x_vc.append(xc)
        y_vc.append(vc)
        lbl_vc.append(label)

        if is_front:
            x_front.append(xc)
            y_front.append(vc)
            lbl_front.append(label)

    # ── Background bands ──────────────────────────────────────────────────────
    ax.set_xlim(-0.1, 7.6)
    ax.set_ylim(0.44, 1.02)

    # Uncontrolled benchmark band
    ax.axhspan(UNCONTROLLED_LO, UNCONTROLLED_HI,
               color="#e8e8e8", alpha=0.9, zorder=0)
    ax.text(7.5, (UNCONTROLLED_LO + UNCONTROLLED_HI) / 2,
            "Uncontrolled\nbenchmarks\n0.78 – 0.91",
            ha="right", va="center",
            fontsize=8, color="#777777", style="italic")

    # Probe band
    ax.axhspan(0.89, 0.975,
               color="#dbeafe", alpha=0.45, zorder=0)

    # Behavioural ceiling band
    ax.axhspan(0.55, 0.80,
               color="#fee2e2", alpha=0.35, zorder=0)

    # Chance line
    ax.axhline(0.5, color="#aaaaaa", lw=0.8, ls=":", zorder=1)
    ax.text(-0.08, 0.502, "Chance", ha="left", va="bottom",
            fontsize=7.5, color="#aaaaaa")

    # ── Group dividers and labels ─────────────────────────────────────────────
    dividers = [2.1, 3.8, 5.1]
    for xd in dividers:
        ax.axvline(xd, color="#dddddd", lw=1.0, ls="--", zorder=1)

    group_label_x = [1.0, 3.2, 4.4, 6.0]
    group_labels  = ["7B – 8B", "32B", "72B", "Frontier API\n(no probe access)"]
    for xg, gl in zip(group_label_x, group_labels):
        ax.text(xg, 1.005, gl, ha="center", va="bottom",
                fontsize=8.5, color="#444444", fontweight="bold")

    # ── Plot probe dots (blue circles) ───────────────────────────────────────
    ax.scatter(x_probe, y_probe,
               s=110, color=C_PROBE, zorder=5,
               edgecolors="white", linewidths=1.5,
               label="Linear probe (hidden states)")

    # ── Plot VC dots ──────────────────────────────────────────────────────────
    # Non-frontier: filled red squares
    x_vc_open  = [x for x, (_, _, _, _, f) in zip(x_vc, MODELS) if not f]
    y_vc_open  = [y for y, (_, _, _, _, f) in zip(y_vc, MODELS) if not f]
    ax.scatter(x_vc_open, y_vc_open,
               s=90, color=C_BEH, marker="s", zorder=5,
               edgecolors="white", linewidths=1.2,
               label="Verbalized confidence (output)")

    # Frontier: red diamond markers (larger, distinct)
    ax.scatter(x_front, y_front,
               s=160, color=C_BEH, marker="D", zorder=6,
               edgecolors="#7b1111", linewidths=1.8,
               label="Verbalized confidence — frontier API ★")

    # ── Trend lines ───────────────────────────────────────────────────────────
    # Probe trend (7B → 72B only, frontier has no probe)
    ax.plot(x_probe, y_probe,
            color=C_PROBE, lw=1.4, ls="--", alpha=0.5, zorder=3)
    # VC trend (all open-weight)
    ax.plot(x_vc_open, y_vc_open,
            color=C_BEH, lw=1.4, ls="--", alpha=0.4, zorder=3)

    # ── Model labels ─────────────────────────────────────────────────────────
    for xi, yi, lb in zip(x_probe, y_probe, lbl_probe):
        ax.text(xi, yi + 0.018, lb,
                ha="center", va="bottom", fontsize=7, color=C_PROBE)
    for xi, yi, lb in zip(x_vc, y_vc, lbl_vc):
        dy = -0.025 if yi > 0.76 else 0.018
        ax.text(xi, yi + dy, lb,
                ha="center", va="top" if dy < 0 else "bottom",
                fontsize=7, color=C_BEH)

    # ── Band annotations ─────────────────────────────────────────────────────
    ax.annotate("",
                xy=(2.0, 0.965), xytext=(2.0, 0.805),
                arrowprops=dict(arrowstyle="<->", color=C_PROBE,
                                lw=1.5, mutation_scale=10))
    ax.text(2.08, 0.886, "Gap\n0.19–0.27",
            ha="left", va="center",
            fontsize=8, color="#1a4d7a", fontweight="bold")

    ax.text(0.0, 0.935, "Probe band", ha="left", va="center",
            fontsize=8, color=C_PROBE, style="italic")
    ax.text(0.0, 0.69, "Behavioural ceiling", ha="left", va="center",
            fontsize=8, color=C_BEH, style="italic")

    # Frontier annotation
    ax.annotate("Scale and prompting\ncannot close this gap →",
                xy=(5.7, 0.758), xytext=(4.3, 0.84),
                fontsize=8, color="#444444", ha="center",
                arrowprops=dict(arrowstyle="->", color="#555555",
                                lw=1.2, connectionstyle="arc3,rad=-0.2"))

    # ── Axes cosmetics ────────────────────────────────────────────────────────
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92,
              bbox_to_anchor=(0.0, 0.01))

    plt.tight_layout(pad=0.5)
    for ext in ["pdf", "png"]:
        path = OUT / f"fig_overview.{ext}"
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    draw()
