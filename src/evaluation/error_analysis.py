"""
error_analysis.py -- Qualitative Error Analysis Table for Paper

Selects 2-3 interesting examples per quadrant (8-12 total) that illustrate
where DECO succeeds/fails relative to baselines. Outputs a LaTeX table
and a JSON file with the selected examples.

Each example shows: question (truncated), question_type, quadrant,
DECO_score, verbalized_confidence_score, DECO_correct, baseline_correct.

Usage:
  python src/evaluation/error_analysis.py
  python src/evaluation/error_analysis.py --model llama
"""

import json
import gc
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import (
    BENCH_DIR, HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS,
)

# ── Quadrant labels for display ──────────────────────────────────────
QUAD_LABELS = {
    "Q1": "Suf+Conf",
    "Q2": "Suf+Uncert",
    "Q3": "Insuf+Conf",
    "Q4": "Insuf+Uncert",
}


# ── Helpers ──────────────────────────────────────────────────────────

def truncate(text: str, max_chars: int = 50) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    for ch in ["&", "%", "$", "#", "_", "{", "}"]:
        text = text.replace(ch, f"\\{ch}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    return text


def load_benchmark_test() -> list:
    """Load test.json and flatten conditions, keeping question text."""
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                "condition_id": cond["condition_id"],
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "sufficient": cond["sufficient"],
                "model_confident": cond["model_confident"],
                "quadrant": cond["quadrant"],
                "context_length": cond.get("context_length", 0),
            })
    return conditions


def load_hidden_states(model_key: str, split: str):
    """Load hidden states with mmap for memory efficiency."""
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy", mmap_mode="r")
    h_q = np.load(d / "h_question_only.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    return h_qc, h_q, labels, metadata


def train_deco_probe(model_key: str, best_layer: int):
    """
    Train a DECO probe on train split and return predictions on test split.
    Returns (test_probs, test_preds, test_labels, test_metadata).
    """
    print(f"  Loading train hidden states for {model_key} ...")
    h_qc_tr, h_q_tr, y_train, _ = load_hidden_states(model_key, "train")
    X_train = np.array(
        h_qc_tr[:, best_layer, :] - h_q_tr[:, best_layer, :]
    )
    del h_qc_tr, h_q_tr
    gc.collect()

    print(f"  Loading test hidden states for {model_key} ...")
    h_qc_te, h_q_te, y_test, meta_test = load_hidden_states(model_key, "test")
    X_test = np.array(
        h_qc_te[:, best_layer, :] - h_q_te[:, best_layer, :]
    )
    del h_qc_te, h_q_te
    gc.collect()

    print(f"  Training DECO probe (layer {best_layer}) ...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    del X_train, X_test
    gc.collect()

    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_tr_s, y_train)

    y_prob = probe.predict_proba(X_te_s)[:, 1]
    y_pred = probe.predict(X_te_s)

    del X_tr_s, X_te_s
    gc.collect()

    return y_prob, y_pred, y_test, meta_test


def train_standard_probe(model_key: str, best_layer: int):
    """Train a standard probe on h(q+c) and return test predictions."""
    h_qc_tr, _, y_train, _ = load_hidden_states(model_key, "train")
    X_train = np.array(h_qc_tr[:, best_layer, :])
    del h_qc_tr
    gc.collect()

    h_qc_te, _, y_test, meta_test = load_hidden_states(model_key, "test")
    X_test = np.array(h_qc_te[:, best_layer, :])
    del h_qc_te
    gc.collect()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    del X_train, X_test
    gc.collect()

    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_tr_s, y_train)

    y_prob = probe.predict_proba(X_te_s)[:, 1]
    y_pred = probe.predict(X_te_s)

    del X_tr_s, X_te_s
    gc.collect()

    return y_prob, y_pred


# ── Example selection logic ──────────────────────────────────────────

def select_examples(
    conditions: list,
    meta_test: list,
    deco_probs: np.ndarray,
    deco_preds: np.ndarray,
    std_probs: np.ndarray,
    std_preds: np.ndarray,
    labels: np.ndarray,
) -> list:
    """
    Select 2-3 interesting examples per quadrant.

    Categories:
      - DECO wins on Q3 (correctly flags insufficient when baseline says sufficient)
      - DECO wins on Q2 (correctly flags sufficient when baseline abstains)
      - DECO failures (for honesty)
      - Diversity across question types
    """
    # Build a lookup from condition_id -> condition info
    cond_by_id = {c["condition_id"]: c for c in conditions}

    # Build per-example records
    records = []
    for i, m in enumerate(meta_test):
        cond_info = cond_by_id.get(m["id"], {})
        label = int(labels[i])
        deco_correct = int(deco_preds[i]) == label
        std_correct = int(std_preds[i]) == label

        records.append({
            "index": i,
            "condition_id": m["id"],
            "question": cond_info.get("question", ""),
            "gold_answer": cond_info.get("gold_answer", ""),
            "question_type": m["question_type"],
            "quadrant": m["quadrant"],
            "sufficient": bool(m["sufficient"]),
            "model_confident": bool(m["model_confident"]),
            "label": label,
            "deco_score": float(deco_probs[i]),
            "standard_probe_score": float(std_probs[i]),
            "deco_pred": int(deco_preds[i]),
            "standard_probe_pred": int(std_preds[i]),
            "deco_correct": deco_correct,
            "standard_correct": std_correct,
        })

    selected = []
    used_qtypes = set()

    # ── Category 1: DECO wins on Q3 (most important -- dangerous quadrant)
    q3 = [r for r in records if r["quadrant"] == "Q3"]
    # DECO correct, standard wrong
    q3_wins = [r for r in q3 if r["deco_correct"] and not r["standard_correct"]]
    # Try to pick diverse question types
    q3_wins.sort(key=lambda r: abs(r["deco_score"] - 0.5))  # most decisive first
    for r in q3_wins:
        if len([s for s in selected if s["category"] == "deco_wins_Q3"]) >= 2:
            break
        if r["question_type"] not in used_qtypes or len(q3_wins) <= 2:
            r["category"] = "deco_wins_Q3"
            selected.append(r)
            used_qtypes.add(r["question_type"])
    # Fallback: if fewer than 2, add any Q3 where DECO is correct
    if len([s for s in selected if s["category"] == "deco_wins_Q3"]) < 2:
        q3_correct = [r for r in q3 if r["deco_correct"] and r not in selected]
        for r in q3_correct[:2 - len([s for s in selected if s["category"] == "deco_wins_Q3"])]:
            r["category"] = "deco_wins_Q3"
            selected.append(r)

    # ── Category 2: DECO wins on Q2 (correctly identifies sufficiency)
    q2 = [r for r in records if r["quadrant"] == "Q2"]
    q2_wins = [r for r in q2 if r["deco_correct"] and not r["standard_correct"]]
    q2_wins.sort(key=lambda r: abs(r["deco_score"] - 0.5))
    for r in q2_wins:
        if len([s for s in selected if s["category"] == "deco_wins_Q2"]) >= 2:
            break
        if r["question_type"] not in used_qtypes or len(q2_wins) <= 2:
            r["category"] = "deco_wins_Q2"
            selected.append(r)
            used_qtypes.add(r["question_type"])
    if len([s for s in selected if s["category"] == "deco_wins_Q2"]) < 2:
        q2_correct = [r for r in q2 if r["deco_correct"] and r not in selected]
        for r in q2_correct[:2 - len([s for s in selected if s["category"] == "deco_wins_Q2"])]:
            r["category"] = "deco_wins_Q2"
            selected.append(r)

    # ── Category 3: DECO failures (for honesty)
    deco_wrong = [r for r in records if not r["deco_correct"] and r not in selected]
    # Prefer cases in the "hard" quadrants Q2/Q3
    deco_wrong_hard = [r for r in deco_wrong if r["quadrant"] in ("Q2", "Q3")]
    deco_wrong_hard.sort(key=lambda r: abs(r["deco_score"] - 0.5), reverse=True)
    # Pick 2, preferring different quadrants and qtypes
    for r in deco_wrong_hard:
        if len([s for s in selected if s["category"] == "deco_failure"]) >= 2:
            break
        r["category"] = "deco_failure"
        selected.append(r)
    # If still need more, take from easy quadrants
    if len([s for s in selected if s["category"] == "deco_failure"]) < 2:
        deco_wrong_easy = [r for r in deco_wrong if r not in selected]
        for r in deco_wrong_easy[:2 - len([s for s in selected if s["category"] == "deco_failure"])]:
            r["category"] = "deco_failure"
            selected.append(r)

    # ── Category 4: Both correct on Q1 (easy agreement) and Q4
    for quad in ["Q1", "Q4"]:
        q_recs = [r for r in records
                  if r["quadrant"] == quad and r["deco_correct"]
                  and r["standard_correct"] and r not in selected]
        # Try to pick a question type not yet represented
        for r in q_recs:
            if r["question_type"] not in used_qtypes:
                r["category"] = f"both_correct_{quad}"
                selected.append(r)
                used_qtypes.add(r["question_type"])
                break
        else:
            if q_recs:
                q_recs[0]["category"] = f"both_correct_{quad}"
                selected.append(q_recs[0])

    # Sort by quadrant for the table
    quad_order = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    selected.sort(key=lambda r: (quad_order.get(r["quadrant"], 9), r["question_type"]))

    return selected


# ── LaTeX table generation ───────────────────────────────────────────

def generate_latex_table(selected: list, model_key: str) -> str:
    """Generate a LaTeX table from selected examples."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Qualitative error analysis for \textsc{DECO} vs.\ standard probe "
        f"({model_key.capitalize()}). "
        r"``Score'' is P(sufficient). Checkmark = correct prediction.}"
    )
    lines.append(r"\label{tab:error_analysis}")
    lines.append(r"\begin{tabular}{p{4.5cm}lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Question (truncated) & Type & Quad & "
        r"DECO & Std & DECO & Std \\"
    )
    lines.append(
        r" & & & Score & Score & Correct & Correct \\"
    )
    lines.append(r"\midrule")

    current_quad = None
    for ex in selected:
        quad = ex["quadrant"]
        if current_quad is not None and quad != current_quad:
            lines.append(r"\midrule")
        current_quad = quad

        q_text = escape_latex(truncate(ex["question"], 50))
        qtype = ex["question_type"].replace("_", r"\_")
        deco_s = f"{ex['deco_score']:.2f}"
        std_s = f"{ex['standard_probe_score']:.2f}"
        deco_c = r"\cmark" if ex["deco_correct"] else r"\xmark"
        std_c = r"\cmark" if ex["standard_correct"] else r"\xmark"

        lines.append(
            f"{q_text} & {qtype} & {quad} & "
            f"{deco_s} & {std_s} & {deco_c} & {std_c} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ── Per-model analysis ───────────────────────────────────────────────

def run_error_analysis(model_key: str) -> dict:
    """Run error analysis for one model."""
    print(f"\n{'=' * 70}")
    print(f"ERROR ANALYSIS -- {model_key}")
    print(f"{'=' * 70}")

    # 1) Load benchmark conditions
    conditions = load_benchmark_test()
    print(f"  Loaded {len(conditions)} conditions from test benchmark")

    # 2) Load DECO results for best layer
    deco_results_path = RESULTS_DIR / model_key / "deco_results.json"
    with open(deco_results_path) as f:
        deco_results = json.load(f)
    best_deco_layer = deco_results["best_layers"]["DECO"]
    best_std_layer = deco_results["best_layers"]["standard"]
    print(f"  Best DECO layer: {best_deco_layer}")
    print(f"  Best Standard layer: {best_std_layer}")

    # 3) Train DECO probe and get per-example predictions
    deco_probs, deco_preds, labels, meta_test = train_deco_probe(
        model_key, best_deco_layer
    )
    print(f"  DECO probe: {np.mean(deco_preds == labels):.4f} accuracy")

    # 4) Train standard probe for comparison
    print(f"  Training standard probe (layer {best_std_layer}) ...")
    std_probs, std_preds = train_standard_probe(model_key, best_std_layer)
    print(f"  Standard probe: {np.mean(std_preds == labels):.4f} accuracy")

    # 5) Select interesting examples
    selected = select_examples(
        conditions, meta_test, deco_probs, deco_preds,
        std_probs, std_preds, labels,
    )
    print(f"  Selected {len(selected)} examples for error analysis")

    # Print summary
    from collections import Counter
    cat_counts = Counter(ex["category"] for ex in selected)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    # 6) Compute summary statistics
    summary = {
        "model": model_key,
        "best_deco_layer": best_deco_layer,
        "best_standard_layer": best_std_layer,
        "n_test": len(labels),
        "deco_accuracy": float(np.mean(deco_preds == labels)),
        "standard_accuracy": float(np.mean(std_preds == labels)),
        "n_selected": len(selected),
        "category_counts": dict(cat_counts),
        "selected_examples": selected,
    }

    # Per-quadrant disagreement stats
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        idx = [i for i, m in enumerate(meta_test) if m["quadrant"] == quad]
        if not idx:
            continue
        idx = np.array(idx)
        n = len(idx)
        deco_acc = float(np.mean(deco_preds[idx] == labels[idx]))
        std_acc = float(np.mean(std_preds[idx] == labels[idx]))
        # Count where DECO correct but standard wrong
        deco_wins = int(np.sum(
            (deco_preds[idx] == labels[idx]) & (std_preds[idx] != labels[idx])
        ))
        std_wins = int(np.sum(
            (std_preds[idx] != labels[idx]) | True  # placeholder
        ))
        std_wins = int(np.sum(
            (std_preds[idx] == labels[idx]) & (deco_preds[idx] != labels[idx])
        ))
        summary[f"{quad}_deco_acc"] = deco_acc
        summary[f"{quad}_std_acc"] = std_acc
        summary[f"{quad}_deco_wins"] = deco_wins
        summary[f"{quad}_std_wins"] = std_wins
        summary[f"{quad}_n"] = n
        print(f"  {quad} (n={n}): DECO={deco_acc:.3f}, Std={std_acc:.3f}, "
              f"DECO-wins={deco_wins}, Std-wins={std_wins}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative error analysis for paper"
    )
    parser.add_argument(
        "--model", default="all",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model to analyze (default: all)",
    )
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    all_summaries = {}
    primary_selected = None  # Will use llama (first model) for the LaTeX table

    for model_key in models:
        # Check prerequisites
        deco_path = RESULTS_DIR / model_key / "deco_results.json"
        hs_dir = HIDDEN_STATES_DIR / model_key / "test"
        if not deco_path.exists():
            print(f"  SKIP {model_key}: no deco_results.json")
            continue
        if not (hs_dir / "h_with_context.npy").exists():
            print(f"  SKIP {model_key}: no hidden states")
            continue

        try:
            summary = run_error_analysis(model_key)
            all_summaries[model_key] = summary
            if primary_selected is None:
                primary_selected = summary
        except Exception as e:
            print(f"  ERROR on {model_key}: {e}")
            import traceback
            traceback.print_exc()

    if not all_summaries:
        print("No models processed. Exiting.")
        return

    # ── Save JSON ────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model JSON
    for model_key, summary in all_summaries.items():
        model_out = out_dir / f"error_analysis_{model_key}.json"
        with open(model_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved {model_key} error analysis -> {model_out}")

    # Combined JSON
    combined_path = out_dir / "error_analysis.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"Saved combined error analysis -> {combined_path}")

    # ── Generate LaTeX table (primary model = llama) ─────────────────
    primary_key = "llama" if "llama" in all_summaries else list(all_summaries.keys())[0]
    primary = all_summaries[primary_key]
    selected = primary["selected_examples"]

    latex = generate_latex_table(selected, primary_key)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tables_dir / "table_error_analysis.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table -> {tex_path}")

    # ── Print the table for quick review ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("SELECTED EXAMPLES (primary model: {})".format(primary_key))
    print(f"{'=' * 70}")
    header = (
        f"{'Question':<52s} {'Type':<12s} {'Quad':<6s} "
        f"{'DECO':>6s} {'Std':>6s} {'D-ok':>5s} {'S-ok':>5s} {'Category':<20s}"
    )
    print(header)
    print("-" * len(header))
    for ex in selected:
        q = truncate(ex["question"], 50)
        print(
            f"{q:<52s} {ex['question_type']:<12s} {ex['quadrant']:<6s} "
            f"{ex['deco_score']:>6.2f} {ex['standard_probe_score']:>6.2f} "
            f"{'Y' if ex['deco_correct'] else 'N':>5s} "
            f"{'Y' if ex['standard_correct'] else 'N':>5s} "
            f"{ex['category']:<20s}"
        )

    # ── Cross-model comparison summary ───────────────────────────────
    if len(all_summaries) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-MODEL DISAGREEMENT SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Model':<10s} {'Quad':<6s} {'DECO':>8s} {'Std':>8s} {'D-wins':>8s} {'S-wins':>8s}")
        print("-" * 50)
        for mk, s in all_summaries.items():
            for quad in ["Q1", "Q2", "Q3", "Q4"]:
                d_acc = s.get(f"{quad}_deco_acc", 0)
                s_acc = s.get(f"{quad}_std_acc", 0)
                d_wins = s.get(f"{quad}_deco_wins", 0)
                s_wins = s.get(f"{quad}_std_wins", 0)
                print(f"{mk:<10s} {quad:<6s} {d_acc:>8.3f} {s_acc:>8.3f} {d_wins:>8d} {s_wins:>8d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
