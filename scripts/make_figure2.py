"""Paired construction example (Figure 2), print-legible version.
Usage: python make_figure2_tmlr.py OUTPUT.pdf
Canvas 4.94 x 3.2 in (printed at native size); all text >= 7 pt."""
import sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42

W, H = 4.94, 3.2
FS_TXT, FS_LAB = 7.0, 8.0
NAVY = "#0f2350"

def main(out):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=0.8, ls="-", r=0.05):
        ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                     boxstyle=f"round,pad={r}", fc=fc, ec=ec, lw=lw, ls=ls))

    # question box
    qx, qw, qy, qh = 0.55, 3.84, 2.76, 0.40
    box(qx, qy, qw, qh, NAVY, NAVY)
    ax.text(W / 2, qy + qh / 2, "Question: Which fault can produce a magnitude 8.0 earthquake?\n"
            "Answer: \"San Andreas\"", ha="center", va="center", color="white",
            fontsize=FS_TXT, linespacing=1.3)
    # connectors
    lx, rx = 1.20, 3.74
    ax.plot([W / 2, W / 2], [qy, 2.62], color=NAVY, lw=0.8)
    ax.plot([lx, rx], [2.62, 2.62], color=NAVY, lw=0.8)
    for x in (lx, rx):
        ax.plot([x, x], [2.62, 2.50], color=NAVY, lw=0.8)

    cw, cx = 2.28, (0.06, 2.60)
    for x0, x1, lab in ((cx[0], lx, "Sufficient context (label = 1)"),
                        (cx[1], rx, "Insufficient context (label = 0)")):
        bw = 2.05
        box(x1 - bw / 2, 2.22, bw, 0.28, "white", NAVY, lw=0.8)
        ax.text(x1, 2.36, lab, ha="center", va="center", fontsize=FS_LAB,
                fontweight="bold", color=NAVY)
        ax.plot([x1, x1], [2.22, 2.17], color=NAVY, lw=0.8)

    def wrap(s, n): return textwrap.fill(s, n)
    other = "Other faults include the San Jacinto Fault, the Puente Hills Fault, and the Elsinore Zone."
    usgs = "The USGS has released a California Earthquake forecast which models earthquake occurrence in California."
    suf = ("Many faults are able to produce a magnitude 6.7+ earthquake, such as the San Andreas "
           "Fault, which can produce a magnitude 8.0 event.")
    ins = ("In southern California there are also twelve cities with more than 200,000 "
           "residents and 34 cities over 100,000 in population.")
    N = 42
    for i, x0 in enumerate(cx):
        # outer frame: solid for sufficient, dashed for insufficient (grayscale-safe)
        box(x0, 0.46, cw, 1.71, "white", NAVY, lw=1.2, ls="-" if i == 0 else (0, (3, 2)))
        # highlighted sentence: grey fill + thick border + bold
        box(x0 + 0.07, 1.42, cw - 0.14, 0.67, "#e4e4e4", "black",
            lw=1.4, ls="-" if i == 0 else (0, (3, 2)))
        ax.text(x0 + 0.14, 1.80, wrap(suf if i == 0 else ins, 40), ha="left", va="center",
                fontsize=FS_TXT, fontweight="bold", color="black", linespacing=1.25)
        if i == 1:
            ax.text(x0 + cw - 0.12, 1.46, "topically related replacement", ha="right",
                    va="bottom", fontsize=FS_TXT, style="italic", color="#222222")
        for y, t in ((1.02, other), (0.60, usgs)):
            box(x0 + 0.07, y - 0.03, cw - 0.14, 0.37, "#f2f2f2", "#999999", lw=0.5)
            ax.text(x0 + 0.14, y + 0.155, wrap(t, N), ha="left", va="center",
                    fontsize=FS_TXT, color="#111111", linespacing=1.2)
    # relation symbols between panels
    xm = (cx[0] + cw + cx[1]) / 2
    for y, s in ((1.71, "≠"), (1.19, "="), (0.77, "=")):
        ax.text(xm, y, s, ha="center", va="center", fontsize=FS_LAB + 3, color=NAVY)

    ax.text(W / 2, 0.31, "Quality filters applied to every pair:", ha="center", va="center",
            fontsize=FS_LAB, fontweight="bold", color=NAVY)
    ax.text(W / 2, 0.11, "(a) gold answer absent from replacement;  (b) length ratio 0.5–2.0;  "
            "(c) cosine similarity ≥ 0.5", ha="center", va="center",
            fontsize=FS_TXT, color="#111111")
    fig.savefig(out, format="pdf")

if __name__ == "__main__":
    main(sys.argv[1])
