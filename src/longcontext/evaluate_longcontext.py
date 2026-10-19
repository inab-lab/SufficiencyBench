"""
evaluate_longcontext.py — analysis of the long-context extension.

Three readouts per (model, length):

  (a) TRANSFER probe — the probe the paper actually ships: Standard and DECO
      logistic regressions trained on the SHORT-context train split
      (hidden_states/<model>/train) at the paper's best layers, applied unchanged
      to the long-context states.  This is the question a practitioner has: does
      my probe survive a longer retrieval window?
  (b) IN-DISTRIBUTION probe — 5-fold GroupKFold CV (group = question id, so both
      conditions of a question always land in the same fold) on the long-context
      states themselves.  Separates "the signal is gone" from "the signal moved".
  (c) VERBALIZED CONFIDENCE — the behavioural baseline, from vc_scores.json.

Each is reported at the fixed best layer and over a full layer sweep, with
bootstrap 95% CIs resampled over QUESTIONS (not rows), plus breakdowns by target
position (first/middle/last) and question type.  Length "base" is the reference:
it is the original short context, so it should reproduce the main-paper numbers.

Outputs:
  results/longcontext/<model>/summary.json
  results/longcontext/summary_table.md
  results/longcontext/layer_curves.json
  results/longcontext/fig_longcontext.pdf

Usage:
  python src/longcontext/evaluate_longcontext.py
  python src/longcontext/evaluate_longcontext.py --models llama --n-boot 200
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import HIDDEN_STATES_DIR  # noqa: E402
from longcontext.common import (  # noqa: E402
    LC_RESULTS_DIR, LENGTHS, MODEL_CONFIGS, POSITION_LABELS, hq_index_path,
    hq_path, length_dir,
)

warnings.filterwarnings("ignore")

#: best layers from the main paper's validation sweep (index into the stored
#: array, i.e. the output of decoder layer i+1) — see methods/deco.py.
def _load_best_layers():
    """Validation-selected layers of the current benchmark version (results/<m>/all_results_logreg.json);
    the v1.0 values are kept as a fallback for models without a probe run."""
    import json as _json, os as _os
    out = {"standard": {"llama": 10, "mistral": 12, "qwen": 16, "qwen3": 17},
           "DECO": {"llama": 9, "mistral": 11, "qwen": 16, "qwen3": 20}}
    for m in list(out["standard"]):
        f = f"/home/inab/Documents/MSD_Project/paper1/results/{m}/all_results_logreg.json"
        if _os.path.exists(f):
            bl = _json.load(open(f))["best_layers"]; out["standard"][m] = int(bl["standard"]); out["DECO"][m] = int(bl["DECO"])
    return out
BEST_LAYERS = _load_best_layers()
METHODS = ["standard", "DECO"]
QUESTION_TYPES = ["factual", "multi_hop", "comparative", "subjective"]
N_FOLDS = 5


def make_probe():
    """StandardScaler + LogisticRegression, identical to methods/deco.py."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=42))


def features(h_qc, h_q, layer, method):
    if method == "standard":
        return h_qc[:, layer, :]
    if method == "DECO":
        return h_qc[:, layer, :] - h_q[:, layer, :]
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_short_train(model_key):
    d = HIDDEN_STATES_DIR / model_key / "train"
    return (np.load(d / "h_with_context.npy"),
            np.load(d / "h_question_only.npy"),
            np.load(d / "labels.npy"))


def load_long(model_key, length):
    """(h_qc, h_q, labels, metadata) for one length, or None if not extracted.

    h(q) is stored once per question and re-expanded to one row per condition so
    the DECO delta is a plain subtraction, as in the short-context pipeline.
    """
    d = length_dir(model_key, length)
    if not (d / "h_with_context.npy").exists():
        return None
    h_qc = np.load(d / "h_with_context.npy")
    labels = np.load(d / "labels.npy")
    meta = json.load(open(d / "metadata.json"))
    hq = np.load(hq_path(model_key))
    hq_index = json.load(open(hq_index_path(model_key)))
    rows = [hq_index[m["question_id"]] for m in meta]
    return h_qc, hq[rows], labels, meta


def load_vc(model_key, length):
    p = length_dir(model_key, length) / "vc_scores.json"
    if not p.exists():
        return None
    return json.load(open(p))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc_ci(y, scores, groups, n_boot=1000, seed=0):
    """Point AUROC + percentile bootstrap 95% CI, resampled over QUESTIONS.

    Resampling questions (not rows) keeps the paired sufficient/insufficient
    conditions of a question together, which is the unit of independence here.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if len(set(y)) < 2:
        return {"auroc": None, "ci": [None, None], "n": int(len(y))}
    point = float(roc_auc_score(y, scores))

    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in pick])
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    if not boots:
        return {"auroc": point, "ci": [None, None], "n": int(len(y))}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": point, "ci": [float(lo), float(hi)],
            "n": int(len(y)), "n_questions": int(len(uniq))}


def cv_scores(X, y, groups, n_folds=N_FOLDS):
    """Out-of-fold predicted probabilities from grouped CV."""
    out = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        probe = make_probe()
        probe.fit(X[tr], y[tr])
        out[te] = probe.predict_proba(X[te])[:, 1]
    return out


def subgroup_auroc(y, scores, groups, mask, n_boot, seed):
    if mask.sum() < 10 or len(set(np.asarray(y)[mask])) < 2:
        return {"auroc": None, "ci": [None, None], "n": int(mask.sum())}
    return auroc_ci(np.asarray(y)[mask], np.asarray(scores)[mask],
                    np.asarray(groups)[mask], n_boot, seed)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_key, lengths, n_boot, seed, layer_stride, do_sweep=True):
    print(f"\n{'=' * 70}\n{model_key}\n{'=' * 70}")
    h_qc_tr, h_q_tr, y_tr = load_short_train(model_key)
    n_layers = h_qc_tr.shape[1]
    sweep_layers = list(range(0, n_layers, layer_stride))

    transfer_probes = {}   # (method, layer) -> fitted pipeline

    def transfer_probe(method, layer):
        key = (method, layer)
        if key not in transfer_probes:
            probe = make_probe()
            probe.fit(features(h_qc_tr, h_q_tr, layer, method), y_tr)
            transfer_probes[key] = probe
        return transfer_probes[key]

    summary = {
        "model": model_key,
        "best_layers": {m: BEST_LAYERS[m][model_key] for m in METHODS},
        "n_layers": int(n_layers),
        "n_boot": n_boot,
        "n_train_short": int(len(y_tr)),
        "lengths": {},
    }
    curves = {}

    # Lengths a backbone could not represent simply have no output directory
    # (extract_longcontext.py drops and logs them), so absence is the filter.
    for length in [str(x) for x in lengths]:
        data = load_long(model_key, length)
        if data is None:
            print(f"  [{length}] no hidden states — skipped")
            continue
        h_qc, h_q, y, meta = data
        groups = np.array([m["question_id"] for m in meta])
        qtypes = np.array([m["question_type"] for m in meta])
        positions = np.array([m["position"] for m in meta])
        ntoks = np.array([m["n_tokens"] for m in meta])
        print(f"  [{length}] n={len(y)} questions={len(np.unique(groups))} "
              f"tokens~{ntoks.mean():.0f}")

        entry = {
            "n_rows": int(len(y)),
            "n_questions": int(len(np.unique(groups))),
            "n_tokens_mean": float(ntoks.mean()),
            "n_tokens_max": int(ntoks.max()),
            "transfer": {}, "cv": {}, "by_position": {}, "by_question_type": {},
        }
        score_cols = {}

        # ---- (a) transfer, (b) in-distribution CV, at the fixed best layer --
        for method in METHODS:
            layer = BEST_LAYERS[method][model_key]
            X = features(h_qc, h_q, layer, method)

            s_tr = transfer_probe(method, layer).predict_proba(X)[:, 1]
            entry["transfer"][method] = {"layer": layer,
                                         **auroc_ci(y, s_tr, groups, n_boot, seed)}
            score_cols[f"transfer_{method}"] = s_tr

            s_cv = cv_scores(X, y, groups)
            entry["cv"][method] = {"layer": layer,
                                   **auroc_ci(y, s_cv, groups, n_boot, seed)}
            score_cols[f"cv_{method}"] = s_cv
            print(f"      {method:<9s} L{layer:<3d} "
                  f"transfer={entry['transfer'][method]['auroc']:.4f}  "
                  f"cv={entry['cv'][method]['auroc']:.4f}")

        # ---- (c) verbalized confidence ------------------------------------
        vc = load_vc(model_key, length)
        if vc:
            cids = [m["condition_id"] for m in meta]
            have = np.array([c in vc for c in cids])
            if have.sum() >= 10:
                s_vc = np.array([vc[c]["score"] if c in vc else np.nan
                                 for c in cids])
                entry["vc"] = {
                    "n_scored": int(have.sum()),
                    **auroc_ci(y[have], s_vc[have], groups[have], n_boot, seed),
                }
                score_cols["vc"] = np.where(have, s_vc, np.nan)
                print(f"      {'VC':<9s} {'':<4s} auroc={entry['vc']['auroc']:.4f} "
                      f"(n={int(have.sum())})")
            else:
                entry["vc"] = {"auroc": None, "ci": [None, None],
                               "note": f"only {int(have.sum())} scores"}
        else:
            entry["vc"] = {"auroc": None, "ci": [None, None],
                           "note": "vc_scores.json missing"}
            if (length_dir(model_key, length) / "vc_oom.txt").exists():
                entry["vc"]["note"] = "OOM"

        # ---- (d) breakdowns ------------------------------------------------
        for pos, label in POSITION_LABELS.items():
            mask = positions == pos
            entry["by_position"][label] = {
                name: subgroup_auroc(y, s, groups, mask & np.isfinite(s), n_boot, seed)
                for name, s in score_cols.items()
            }
        for qt in QUESTION_TYPES:
            mask = qtypes == qt
            if mask.sum() == 0:
                continue
            entry["by_question_type"][qt] = {
                name: subgroup_auroc(y, s, groups, mask & np.isfinite(s), n_boot, seed)
                for name, s in score_cols.items()
            }

        # ---- layer sweeps ---------------------------------------------------
        if do_sweep:
            curve = {}
            for method in METHODS:
                t_curve, c_curve = [], []
                for layer in sweep_layers:
                    X = features(h_qc, h_q, layer, method)
                    t_curve.append(float(roc_auc_score(
                        y, transfer_probe(method, layer).predict_proba(X)[:, 1])))
                    c_curve.append(float(roc_auc_score(y, cv_scores(X, y, groups))))
                curve[f"transfer_{method}"] = t_curve
                curve[f"cv_{method}"] = c_curve
                print(f"      sweep {method}: transfer max={max(t_curve):.4f} "
                      f"@L{sweep_layers[int(np.argmax(t_curve))]}  "
                      f"cv max={max(c_curve):.4f} "
                      f"@L{sweep_layers[int(np.argmax(c_curve))]}")
            curve["layers"] = sweep_layers
            curves[length] = curve

        summary["lengths"][length] = entry

    out_dir = LC_RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out_dir / 'summary.json'}")
    return summary, curves


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(cell):
    if not cell or cell.get("auroc") is None:
        return cell.get("note", "—") if cell else "—"
    lo, hi = cell.get("ci", [None, None])
    if lo is None:
        return f"{cell['auroc']:.3f}"
    return f"{cell['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"


def write_table(summaries, path):
    lines = [
        "# Long-context extension — AUROC (95% bootstrap CI over questions)",
        "",
        "`transfer` = probe trained on the SHORT-context train split at the paper's",
        "best layer, applied unchanged. `CV` = 5-fold grouped CV on the long-context",
        "set itself. `VC` = verbalized confidence. `base` = original short context.",
        "",
        "| Model | Length | Tokens | n | Transfer Standard | Transfer DECO | "
        "CV Standard | CV DECO | VC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model_key, s in summaries.items():
        for length, e in s["lengths"].items():
            lines.append(
                f"| {model_key} | {length} | {e['n_tokens_mean']:.0f} | {e['n_rows']} | "
                f"{_fmt(e['transfer'].get('standard'))} | {_fmt(e['transfer'].get('DECO'))} | "
                f"{_fmt(e['cv'].get('standard'))} | {_fmt(e['cv'].get('DECO'))} | "
                f"{_fmt(e.get('vc'))} |")

    lines += ["", "## Breakdown by target position (DECO transfer / DECO CV / VC)", "",
              "| Model | Length | Position | Transfer DECO | CV DECO | VC |",
              "|---|---|---|---|---|---|"]
    for model_key, s in summaries.items():
        for length, e in s["lengths"].items():
            for label, cells in e["by_position"].items():
                lines.append(
                    f"| {model_key} | {length} | {label} | "
                    f"{_fmt(cells.get('transfer_DECO'))} | {_fmt(cells.get('cv_DECO'))} | "
                    f"{_fmt(cells.get('vc'))} |")

    lines += ["", "## Breakdown by question type (DECO transfer / DECO CV / VC)", "",
              "| Model | Length | Question type | Transfer DECO | CV DECO | VC |",
              "|---|---|---|---|---|---|"]
    for model_key, s in summaries.items():
        for length, e in s["lengths"].items():
            for qt, cells in e["by_question_type"].items():
                lines.append(
                    f"| {model_key} | {length} | {qt} | "
                    f"{_fmt(cells.get('transfer_DECO'))} | {_fmt(cells.get('cv_DECO'))} | "
                    f"{_fmt(cells.get('vc'))} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# Colour = readout (probe-transfer / probe-CV / verbalized confidence);
# line style = feature set (DECO solid, Standard dashed). Identity is therefore
# never carried by colour alone. INAB brand palette (matches
# src/figures/generate_frontier_figure.py / generate_framework_figure.py).
# Palette checked with the CVD validator (CIELAB dE76): worst adjacent pair
# (SKY vs NAVY) is dE76=42.6 (normal), 37.5 (deutan/protan) — all well above
# the ~10 "clearly distinguishable" threshold.
C_TRANSFER = "#4a90c4"   # INAB Sky Blue  — probe transfer
C_CV = "#1a2a52"         # INAB Navy Blue — probe in-distribution
C_VC = "#c07840"         # INAB warm amber — verbalized confidence

# Ordered so matplotlib's column-major legend fill (ncol=3) reads
# CSP row / Standard row.
SERIES = [
    ("transfer_DECO", "Probe transfer (CSP)", C_TRANSFER, "-"),
    ("transfer_standard", "Probe transfer (Standard)", C_TRANSFER, "--"),
    ("cv_DECO", "Probe in-distribution (CSP)", C_CV, "-"),
    ("cv_standard", "Probe in-distribution (Standard)", C_CV, "--"),
    ("vc", "Verbalized confidence", C_VC, "-"),
]


def _series_cell(entry, name):
    if name == "vc":
        return entry.get("vc")
    kind, method = name.split("_", 1)
    return entry.get("transfer" if kind == "transfer" else "cv", {}).get(method)


def make_figure(summaries, path, lengths=LENGTHS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summaries)
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.4),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    xticks = [ln for ln in lengths]
    xpos = {ln: i for i, ln in enumerate(xticks)}

    for ax, model_key in zip(axes, models):
        s = summaries[model_key]
        for name, label, color, style in SERIES:
            xs, ys, los, his = [], [], [], []
            for ln in xticks:
                cell = _series_cell(s["lengths"].get(ln, {}), name)
                if not cell or cell.get("auroc") is None:
                    continue
                xs.append(xpos[ln])
                ys.append(cell["auroc"])
                lo, hi = cell.get("ci", [None, None])
                los.append(lo if lo is not None else cell["auroc"])
                his.append(hi if hi is not None else cell["auroc"])
            if not xs:
                continue
            ax.plot(xs, ys, style, color=color, linewidth=2,
                    marker="o" if style == "-" else "s", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label)
            ax.fill_between(xs, los, his, color=color, alpha=0.13, linewidth=0)

        ax.axhline(0.5, color="#9aa0a6", linewidth=1, linestyle=":", zorder=0)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels([("base" if ln == "base" else f"{int(ln) // 1000}k")
                            for ln in xticks])
        ax.set_xlabel("Context length (tokens)")
        ax.set_title(MODEL_CONFIGS[model_key]["name"].split("/")[-1], fontsize=10)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("AUROC (sufficient vs. insufficient)")
    axes[0].set_ylim(0.4, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--lengths", nargs="+", default=LENGTHS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="evaluate every k-th layer in the sweep (1 = all)")
    ap.add_argument("--no-layer-sweep", action="store_true")
    ap.add_argument("--device", default=None,
                    help="ignored (this stage is CPU-only); accepted so the "
                         "scripts/run_methods_guarded.sh driver can run it")
    args = ap.parse_args()

    summaries, all_curves = {}, {}
    for model_key in args.models:
        if not (HIDDEN_STATES_DIR / model_key / "train").exists():
            print(f"[skip] {model_key}: no short-context train states")
            continue
        if model_key not in BEST_LAYERS["standard"]:
            print(f"[skip] {model_key}: no probe layer recorded")
            continue
        s, curves = evaluate_model(model_key, args.lengths, args.n_boot,
                                   args.seed, args.layer_stride,
                                   do_sweep=not args.no_layer_sweep)
        if s["lengths"]:
            summaries[model_key] = s
            all_curves[model_key] = curves

    if not summaries:
        print("Nothing to report — no long-context states found.")
        return

    LC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_table(summaries, LC_RESULTS_DIR / "summary_table.md")
    with open(LC_RESULTS_DIR / "layer_curves.json", "w") as f:
        json.dump({"best_layers": BEST_LAYERS, "curves": all_curves}, f, indent=2)
    print(f"-> {LC_RESULTS_DIR / 'layer_curves.json'}")
    present = [ln for ln in args.lengths
               if any(ln in s["lengths"] for s in summaries.values())]
    make_figure(summaries, LC_RESULTS_DIR / "fig_longcontext.pdf", present)


if __name__ == "__main__":
    main()
