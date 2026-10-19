"""
Per-quadrant analysis for SufficiencyBench paper.
Computes:
1. Per-quadrant-pair AUROC (Q1vsQ3, Q2vsQ4) with bootstrap CIs
2. Delta (PK contamination index)
3. McNemar's test for CSP vs Standard on Q2/Q3
4. Conditional false negative rates by quadrant
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats

np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import RESULTS_DIR, BENCH_DIR

DATA_DIR = BENCH_DIR

# ── Load test data with quadrant labels ──────────────────────────────────────

def load_test_data():
    """Load test.json and flatten into per-condition list."""
    with open(DATA_DIR / "test.json") as f:
        test = json.load(f)

    conditions = []
    for q in test:
        for c in q["conditions"]:
            conditions.append({
                "quadrant": c["quadrant"],
                "sufficient": c["sufficient"],
                "model_knows": c.get("model_confident", False),
                "question_type": q.get("question_type", "unknown"),
            })
    return conditions


# ── Load per-example probe scores from SCCD results ─────────────────────────

def load_probe_scores(model):
    """Load per-example probe scores from SCCD results."""
    path = RESULTS_DIR / model / "sccd" / "sccd_results.json"
    with open(path) as f:
        data = json.load(f)

    examples = data["examples"]
    return examples


# ── Load per-example probe scores from deco_rag_results ─────────────────────

def load_deco_rag_scores(model):
    """Load per-example CSP and standard probe scores."""
    path = RESULTS_DIR / model / "deco_rag_results.json"
    with open(path) as f:
        data = json.load(f)
    return data["examples"] if "examples" in data else data


# ── Bootstrap AUROC ─────────────────────────────────────────────────────────

def bootstrap_auroc(y_true, y_score, n_boot=1000):
    """Compute AUROC with bootstrap 95% CI."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if len(np.unique(y_true)) < 2:
        return None, None, None

    auroc = roc_auc_score(y_true, y_score)

    boot_aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))

    ci_lo = np.percentile(boot_aurocs, 2.5)
    ci_hi = np.percentile(boot_aurocs, 97.5)
    return auroc, ci_lo, ci_hi


# ── McNemar's test ──────────────────────────────────────────────────────────

def mcnemars_test(correct_a, correct_b):
    """
    McNemar's test for paired binary classifiers.
    correct_a[i] = True if method A got example i right
    correct_b[i] = True if method B got example i right
    """
    correct_a = np.array(correct_a, dtype=bool)
    correct_b = np.array(correct_b, dtype=bool)

    # b = A wrong, B right (B wins)
    b = np.sum(~correct_a & correct_b)
    # c = A right, B wrong (A wins)
    c = np.sum(correct_a & ~correct_b)

    # McNemar's test statistic (with continuity correction)
    if b + c == 0:
        return 0, 1.0, b, c

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value, int(b), int(c)


# ── Main analysis ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_conditions = load_test_data()

    print("=" * 80)
    print("PER-QUADRANT ANALYSIS FOR SUFFICIENCYBENCH")
    print("=" * 80)

    results = {}

    for model in ["llama", "mistral", "qwen"]:
        print(f"\n{'─' * 80}")
        print(f"MODEL: {model.upper()}")
        print(f"{'─' * 80}")

        # Load probe scores
        try:
            sccd_examples = load_probe_scores(model)
        except Exception as e:
            print(f"  Could not load SCCD data: {e}")
            continue

        # Extract quadrant, ground truth, and probe score
        quadrants = [ex["quadrant"] for ex in sccd_examples]
        sufficients = [ex["sufficient"] for ex in sccd_examples]
        probe_scores = [ex["probe_score"] for ex in sccd_examples]

        n_total = len(quadrants)
        print(f"  Total examples: {n_total}")
        print(f"  Quadrant distribution: " +
              ", ".join(f"{q}={quadrants.count(q)}" for q in ["Q1","Q2","Q3","Q4"]))

        # ── 1. Per-quadrant-pair AUROC ──
        print(f"\n  --- Per-Quadrant-Pair AUROC ---")

        # Q1 vs Q3 (model knows): can we tell suf from insuf when model knows?
        q1_q3_idx = [i for i, q in enumerate(quadrants) if q in ["Q1", "Q3"]]
        q1_q3_true = [1 if sufficients[i] else 0 for i in q1_q3_idx]
        q1_q3_scores = [probe_scores[i] for i in q1_q3_idx]
        auroc_knows, ci_lo_k, ci_hi_k = bootstrap_auroc(q1_q3_true, q1_q3_scores)

        # Q2 vs Q4 (model doesn't know): can we tell suf from insuf when model doesn't know?
        q2_q4_idx = [i for i, q in enumerate(quadrants) if q in ["Q2", "Q4"]]
        q2_q4_true = [1 if sufficients[i] else 0 for i in q2_q4_idx]
        q2_q4_scores = [probe_scores[i] for i in q2_q4_idx]
        auroc_no_knows, ci_lo_nk, ci_hi_nk = bootstrap_auroc(q2_q4_true, q2_q4_scores)

        # Overall AUROC
        overall_true = [1 if s else 0 for s in sufficients]
        auroc_overall, ci_lo_o, ci_hi_o = bootstrap_auroc(overall_true, probe_scores)

        print(f"  Overall:           AUROC = {auroc_overall:.3f} [{ci_lo_o:.3f}, {ci_hi_o:.3f}]")
        print(f"  Q1 vs Q3 (knows):  AUROC = {auroc_knows:.3f} [{ci_lo_k:.3f}, {ci_hi_k:.3f}]")
        print(f"  Q2 vs Q4 (no PK):  AUROC = {auroc_no_knows:.3f} [{ci_lo_nk:.3f}, {ci_hi_nk:.3f}]")

        # ── 2. Delta (PK contamination index) ──
        delta = abs(auroc_knows - auroc_no_knows)
        print(f"\n  --- PK Contamination Index ---")
        print(f"  Delta = |AUROC(knows) - AUROC(no_PK)| = {delta:.3f}")
        print(f"  Interpretation: {'Low contamination (good)' if delta < 0.05 else 'Moderate contamination' if delta < 0.1 else 'High contamination (PK-dependent)'}")

        # ── 3. Conditional false negative/positive rates ──
        print(f"\n  --- Conditional Error Rates ---")
        threshold = 0.5
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_idx = [i for i, qd in enumerate(quadrants) if qd == q]
            q_true = [sufficients[i] for i in q_idx]
            q_pred = [probe_scores[i] > threshold for i in q_idx]

            n_q = len(q_idx)
            if q in ["Q1", "Q2"]:  # Sufficient contexts
                # False negative = predicts insufficient when actually sufficient
                fn = sum(1 for t, p in zip(q_true, q_pred) if t and not p)
                fn_rate = fn / n_q if n_q > 0 else 0
                print(f"  {q} (sufficient, n={n_q}): FN rate = {fn_rate:.3f} ({fn}/{n_q})")
            else:  # Insufficient contexts
                # False positive = predicts sufficient when actually insufficient
                fp = sum(1 for t, p in zip(q_true, q_pred) if not t and p)
                fp_rate = fp / n_q if n_q > 0 else 0
                print(f"  {q} (insufficient, n={n_q}): FP rate = {fp_rate:.3f} ({fp}/{n_q})")

        # ── 4. McNemar's test (CSP vs Standard) ──
        print(f"\n  --- McNemar's Test: CSP vs Standard ---")
        try:
            deco_rag = load_deco_rag_scores(model)

            # Get predictions for Q2/Q3 only
            csp_correct_q23 = []
            std_correct_q23 = []
            csp_correct_all = []
            std_correct_all = []

            for ex in deco_rag:
                q = ex["quadrant"]
                suf = ex["sufficient"]
                csp_pred = ex.get("deco_pred_sufficient", ex.get("deco_prob", 0.5) > 0.5)
                std_pred = ex.get("std_pred_sufficient", ex.get("std_prob", 0.5) > 0.5)

                csp_right = (csp_pred == suf)
                std_right = (std_pred == suf)

                csp_correct_all.append(csp_right)
                std_correct_all.append(std_right)

                if q in ["Q2", "Q3"]:
                    csp_correct_q23.append(csp_right)
                    std_correct_q23.append(std_right)

            # McNemar on Q2/Q3
            if csp_correct_q23:
                chi2, p, csp_wins, std_wins = mcnemars_test(std_correct_q23, csp_correct_q23)
                print(f"  Q2/Q3 only (n={len(csp_correct_q23)}):")
                print(f"    CSP wins (std wrong, CSP right): {csp_wins}")
                print(f"    Std wins (CSP wrong, std right): {std_wins}")
                print(f"    McNemar chi2 = {chi2:.3f}, p = {p:.4f}")
                print(f"    {'Significant (p<0.05)' if p < 0.05 else 'Not significant'}")

            # McNemar on ALL examples (to check if CSP introduces errors elsewhere)
            chi2_all, p_all, csp_wins_all, std_wins_all = mcnemars_test(std_correct_all, csp_correct_all)
            print(f"  All quadrants (n={len(csp_correct_all)}):")
            print(f"    CSP wins: {csp_wins_all}")
            print(f"    Std wins: {std_wins_all}")
            print(f"    McNemar chi2 = {chi2_all:.3f}, p = {p_all:.4f}")

            # Check Q1/Q4 specifically (does CSP hurt on diagonal?)
            csp_correct_diag = []
            std_correct_diag = []
            for ex in deco_rag:
                if ex["quadrant"] in ["Q1", "Q4"]:
                    suf = ex["sufficient"]
                    csp_pred = ex.get("deco_pred_sufficient", ex.get("deco_prob", 0.5) > 0.5)
                    std_pred = ex.get("std_pred_sufficient", ex.get("std_prob", 0.5) > 0.5)
                    csp_correct_diag.append(csp_pred == suf)
                    std_correct_diag.append(std_pred == suf)

            if csp_correct_diag:
                chi2_d, p_d, csp_w_d, std_w_d = mcnemars_test(std_correct_diag, csp_correct_diag)
                print(f"  Q1/Q4 only - diagonal (n={len(csp_correct_diag)}):")
                print(f"    CSP wins: {csp_w_d}, Std wins: {std_w_d}")
                print(f"    Does CSP hurt on diagonal? {'Yes' if std_w_d > csp_w_d and p_d < 0.05 else 'No significant harm'}")

        except Exception as e:
            print(f"  Could not run McNemar's test: {e}")

        results[model] = {
            "auroc_overall": auroc_overall,
            "auroc_knows": auroc_knows,
            "auroc_no_pk": auroc_no_knows,
            "delta": delta,
        }

    # ── Summary table ──
    print(f"\n{'=' * 80}")
    print("SUMMARY: PK CONTAMINATION INDEX")
    print(f"{'=' * 80}")
    print(f"{'Model':<10} {'Overall':>10} {'Q1vsQ3':>10} {'Q2vsQ4':>10} {'Delta':>10}")
    print("-" * 50)
    for model in ["llama", "mistral", "qwen"]:
        if model in results:
            r = results[model]
            print(f"{model:<10} {r['auroc_overall']:>10.3f} {r['auroc_knows']:>10.3f} {r['auroc_no_pk']:>10.3f} {r['delta']:>10.3f}")

    # Save results
    output_path = RESULTS_DIR / "quadrant_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
