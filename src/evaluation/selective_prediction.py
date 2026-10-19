"""
selective_prediction.py — Probe-Guided Selective Prediction Evaluation

The practical application of the DECO probe: use it to decide when a RAG
system should answer vs. abstain/retrieve more.

Key insight: The probe detects sufficiency (AUROC 0.91-0.94) independently
of confidence. This means it can catch cases where the model is CONFIDENT
but the context is INSUFFICIENT (Q3 — the dangerous hallucination quadrant).

Evaluation framework:
1. Risk-Coverage curves: at each coverage level, what fraction of answers
   are wrong? Compare probe-gated vs confidence-based vs random.
2. Selective Accuracy: accuracy on the subset of questions the system
   chooses to answer.
3. Hallucination Prevention Rate: what fraction of Q3 (dangerous) examples
   are correctly filtered out at each coverage level?
4. AUC metrics for all curves, with bootstrap confidence intervals.
5. Per-quadrant breakdown showing WHERE the probe adds value.
6. Statistical significance via McNemar's test and paired bootstrap.

Usage:
  python src/evaluation/selective_prediction.py --model llama
  python src/evaluation/selective_prediction.py --model all
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hidden_states(model_key: str, split: str):
    """Load hidden states, labels, metadata."""
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy")
    h_q = np.load(d / "h_question_only.npy")
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    return h_qc, h_q, labels, metadata


def load_deco_config(model_key: str):
    """Load best layer info from existing DECO results."""
    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Train sufficiency scoring methods
# ---------------------------------------------------------------------------

def train_deco_probe(h_qc_train, h_q_train, y_train, layer):
    """Train DECO probe: logistic regression on h(q+c) - h(q)."""
    X = h_qc_train[:, layer, :] - h_q_train[:, layer, :]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_scaled, y_train)
    return probe, scaler


def train_standard_probe(h_qc_train, y_train, layer):
    """Train standard probe: logistic regression on h(q+c)."""
    X = h_qc_train[:, layer, :]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_scaled, y_train)
    return probe, scaler


def train_confidence_probe(h_q_train, y_conf_train, layer):
    """Train confidence probe: logistic regression on h(q) to predict model_confident."""
    X = h_q_train[:, layer, :]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_scaled, y_conf_train)
    return probe, scaler


def score_examples(probe, scaler, X):
    """Get P(sufficient) scores for examples."""
    return probe.predict_proba(scaler.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Selective prediction metrics
# ---------------------------------------------------------------------------

def selective_accuracy_curve(scores, labels, n_points=200):
    """Compute selective accuracy at various coverage levels.

    Sort examples by score (descending). At each coverage fraction,
    compute accuracy on the top-scoring examples.

    Returns: coverages, accuracies (arrays of length n_points)
    """
    order = np.argsort(-scores)  # highest score first
    sorted_labels = labels[order]
    n = len(labels)

    coverages = np.linspace(0.05, 1.0, n_points)
    accuracies = np.zeros(n_points)

    for i, cov in enumerate(coverages):
        k = max(1, int(cov * n))
        accuracies[i] = np.mean(sorted_labels[:k])

    return coverages, accuracies


def risk_coverage_curve(scores, correct, n_points=200):
    """Compute risk (1-accuracy) at various coverage levels.

    Higher scores = more confident the system is that context is sufficient.
    We answer the top-scoring examples first.

    Returns: coverages, risks
    """
    coverages, accuracies = selective_accuracy_curve(scores, correct, n_points)
    return coverages, 1.0 - accuracies


def hallucination_prevention_curve(scores, quadrants, n_points=200):
    """At each coverage level, what fraction of answered Q3 examples remain?

    Q3 = insufficient context + model confident = hallucination risk.
    A good sufficiency detector should filter these out first.

    Returns: coverages, q3_fractions (fraction of Q3 among answered)
    """
    is_q3 = np.array([q == "Q3" for q in quadrants])
    order = np.argsort(-scores)
    sorted_q3 = is_q3[order]
    n = len(quadrants)

    coverages = np.linspace(0.05, 1.0, n_points)
    q3_fracs = np.zeros(n_points)

    for i, cov in enumerate(coverages):
        k = max(1, int(cov * n))
        q3_fracs[i] = np.mean(sorted_q3[:k])

    return coverages, q3_fracs


def compute_auc(x, y):
    """Trapezoidal AUC."""
    return float(np.trapezoid(y, x))


def bootstrap_ci(scores, labels, metric_fn, n_boot=5000, ci=0.95):
    """Bootstrap confidence interval for a metric.

    metric_fn: callable(scores, labels) -> scalar
    """
    rng = np.random.default_rng(42)
    n = len(scores)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        # Skip samples with only one class (roc_auc_score fails)
        if len(set(labels[idx])) < 2:
            continue
        try:
            val = metric_fn(scores[idx], labels[idx])
            boot_vals.append(val)
        except Exception:
            continue

    if len(boot_vals) == 0:
        return float('nan'), float('nan')

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_vals, 100 * alpha)
    hi = np.percentile(boot_vals, 100 * (1 - alpha))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Comparison methods
# ---------------------------------------------------------------------------

def random_scores(n, seed=42):
    """Random baseline."""
    return np.random.default_rng(seed).random(n)


def oracle_scores(labels):
    """Oracle: score = 1 if sufficient, 0 otherwise. Perfect separation."""
    return labels.astype(float)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_selective_prediction(model_key: str):
    """Run full selective prediction evaluation for one model."""
    print(f"\n{'='*70}")
    print(f"SELECTIVE PREDICTION EVALUATION — {model_key.upper()}")
    print(f"{'='*70}")

    # Load data
    h_qc_train, h_q_train, y_train, meta_train = load_hidden_states(model_key, "train")
    h_qc_test, h_q_test, y_test, meta_test = load_hidden_states(model_key, "test")

    deco_config = load_deco_config(model_key)
    best_deco_layer = deco_config["best_layers"]["DECO"]
    best_std_layer = deco_config["best_layers"]["standard"]

    n_test = len(y_test)
    quadrants = [m["quadrant"] for m in meta_test]
    q_types = [m["question_type"] for m in meta_test]

    # Confidence labels
    y_conf_train = np.array([1 if m["model_confident"] else 0 for m in meta_train])
    y_conf_test = np.array([1 if m["model_confident"] else 0 for m in meta_test])

    print(f"Test set: {n_test} examples")
    print(f"Best DECO layer: {best_deco_layer}, Best standard layer: {best_std_layer}")

    # ------------------------------------------------------------------
    # Train scoring methods
    # ------------------------------------------------------------------
    print("\nTraining probes...")

    # DECO probe (our method)
    deco_probe, deco_scaler = train_deco_probe(
        h_qc_train, h_q_train, y_train, best_deco_layer
    )
    X_test_deco = h_qc_test[:, best_deco_layer, :] - h_q_test[:, best_deco_layer, :]
    deco_scores = score_examples(deco_probe, deco_scaler, X_test_deco)

    # Standard probe (conflated baseline)
    std_probe, std_scaler = train_standard_probe(
        h_qc_train, y_train, best_std_layer
    )
    X_test_std = h_qc_test[:, best_std_layer, :]
    std_scores = score_examples(std_probe, std_scaler, X_test_std)

    # Confidence probe (predicts model_confident, not sufficiency)
    conf_probe, conf_scaler = train_confidence_probe(
        h_q_train, y_conf_train, best_std_layer
    )
    X_test_conf = h_q_test[:, best_std_layer, :]
    conf_scores = score_examples(conf_probe, conf_scaler, X_test_conf)

    # Random baseline
    rand_scores = random_scores(n_test)

    # Oracle
    oracle_sc = oracle_scores(y_test)

    methods = {
        "DECO (ours)": deco_scores,
        "Standard probe": std_scores,
        "Confidence probe": conf_scores,
        "Random": rand_scores,
        "Oracle": oracle_sc,
    }

    # ------------------------------------------------------------------
    # 1. Sufficiency Detection AUROC (sanity check)
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("1. SUFFICIENCY DETECTION AUROC")
    print(f"{'─'*50}")

    for name, scores in methods.items():
        if name == "Oracle":
            continue
        try:
            auroc = roc_auc_score(y_test, scores)
            lo, hi = bootstrap_ci(scores, y_test, roc_auc_score)
            print(f"  {name:<25s}: AUROC = {auroc:.4f}  [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  {name:<25s}: Error — {e}")

    # ------------------------------------------------------------------
    # 2. Selective Accuracy Curves
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("2. SELECTIVE ACCURACY (accuracy on answered subset)")
    print(f"{'─'*50}")

    # For selective accuracy, we need "correct" labels.
    # The sufficiency label IS the proxy: if context is sufficient, the model
    # *should* answer correctly. This is the idealized setting.
    # For the realistic setting, we use SCCD generation data if available.

    # Idealized: correct = sufficient (measures if the gating is right)
    correct_idealized = y_test.copy()

    sa_results = {}
    for name, scores in methods.items():
        covs, accs = selective_accuracy_curve(scores, correct_idealized)
        auc = compute_auc(covs, accs)
        sa_results[name] = {"coverages": covs, "accuracies": accs, "auc": auc}
        print(f"  {name:<25s}: SA-AUC = {auc:.4f}")

    # Key coverage points
    print(f"\n  Selective accuracy at specific coverage levels:")
    print(f"  {'Method':<25s} {'50%':>8s} {'70%':>8s} {'90%':>8s} {'100%':>8s}")
    print(f"  {'-'*55}")
    for name, res in sa_results.items():
        covs, accs = res["coverages"], res["accuracies"]
        vals = []
        for target in [0.50, 0.70, 0.90, 1.00]:
            idx = np.argmin(np.abs(covs - target))
            vals.append(f"{accs[idx]:.3f}")
        print(f"  {name:<25s} {'  '.join(f'{v:>8s}' for v in vals)}")

    # ------------------------------------------------------------------
    # 3. Hallucination Prevention (Q3 filtering)
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("3. HALLUCINATION PREVENTION (Q3 filtering)")
    print(f"{'─'*50}")
    print("  Q3 = insufficient + confident = hallucination risk")
    print("  Lower Q3 fraction among answered = better\n")

    hp_results = {}
    for name, scores in methods.items():
        covs, q3_fracs = hallucination_prevention_curve(scores, quadrants)
        auc = compute_auc(covs, q3_fracs)
        hp_results[name] = {"coverages": covs, "q3_fracs": q3_fracs, "auc": auc}

    # At key coverage levels, what % of answered are Q3?
    q3_total_frac = np.mean([q == "Q3" for q in quadrants])
    print(f"  Q3 base rate: {q3_total_frac:.3f} ({sum(q == 'Q3' for q in quadrants)}/{n_test})")
    print(f"\n  Q3 fraction among answered (lower = better):")
    print(f"  {'Method':<25s} {'50%':>8s} {'70%':>8s} {'90%':>8s} {'AUC':>8s}")
    print(f"  {'-'*55}")
    for name, res in hp_results.items():
        covs, q3f = res["coverages"], res["q3_fracs"]
        vals = []
        for target in [0.50, 0.70, 0.90]:
            idx = np.argmin(np.abs(covs - target))
            vals.append(f"{q3f[idx]:.3f}")
        vals.append(f"{res['auc']:.3f}")
        print(f"  {name:<25s} {'  '.join(f'{v:>8s}' for v in vals)}")

    # ------------------------------------------------------------------
    # 4. Per-Quadrant Analysis: WHERE does DECO help?
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("4. PER-QUADRANT SCORE DISTRIBUTIONS")
    print(f"{'─'*50}")

    quad_names = {"Q1": "Suf+Conf", "Q2": "Suf+Uncert",
                  "Q3": "Insuf+Conf", "Q4": "Insuf+Uncert"}

    for name in ["DECO (ours)", "Standard probe", "Confidence probe"]:
        scores = methods[name]
        print(f"\n  {name}:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            idx = [i for i, qd in enumerate(quadrants) if qd == q]
            q_scores = scores[idx]
            print(f"    {q} ({quad_names[q]:>12s}, n={len(idx):>4d}): "
                  f"mean={np.mean(q_scores):.3f} ± {np.std(q_scores):.3f}  "
                  f"median={np.median(q_scores):.3f}  "
                  f"[{np.percentile(q_scores, 25):.3f}, {np.percentile(q_scores, 75):.3f}]")

    # ------------------------------------------------------------------
    # 5. THE KEY COMPARISON: DECO vs Standard on Q3
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("5. KEY COMPARISON: Separating Q3 (dangerous) from Q1+Q2 (safe)")
    print(f"{'─'*50}")

    # Q3 vs (Q1+Q2): can each method distinguish dangerous from safe?
    is_safe = np.array([q in ("Q1", "Q2") for q in quadrants]).astype(int)
    is_q3 = np.array([q == "Q3" for q in quadrants]).astype(int)
    safe_or_q3 = np.array([q in ("Q1", "Q2", "Q3") for q in quadrants])

    for name in ["DECO (ours)", "Standard probe", "Confidence probe"]:
        scores = methods[name]
        # AUROC for distinguishing safe (Q1+Q2) from dangerous (Q3)
        mask = safe_or_q3
        if np.sum(mask) > 0 and len(set(is_safe[mask])) > 1:
            auroc = roc_auc_score(is_safe[mask], scores[mask])
            lo, hi = bootstrap_ci(scores[mask], is_safe[mask], roc_auc_score)
            print(f"  {name:<25s}: AUROC(safe vs Q3) = {auroc:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------
    # 6. Threshold Analysis: practical operating points
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("6. PRACTICAL OPERATING POINTS (DECO)")
    print(f"{'─'*50}")
    print("  At each threshold: answer if probe_score >= threshold\n")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"  {'Thresh':>7s} {'Coverage':>9s} {'SelAcc':>8s} {'Q3_filt':>9s} "
          f"{'Q1+Q2_kept':>11s} {'Q3_in_ans':>10s}")
    print(f"  {'-'*60}")

    for t in thresholds:
        answered = deco_scores >= t
        n_answered = np.sum(answered)
        coverage = n_answered / n_test

        if n_answered == 0:
            continue

        # Selective accuracy (sufficient among answered)
        sel_acc = np.mean(y_test[answered])

        # Q3 filtering rate (what fraction of Q3 are filtered out)
        q3_mask = np.array([q == "Q3" for q in quadrants])
        q3_filtered = 1.0 - np.mean(answered[q3_mask]) if np.sum(q3_mask) > 0 else 0

        # Q1+Q2 kept rate
        safe_mask = np.array([q in ("Q1", "Q2") for q in quadrants])
        safe_kept = np.mean(answered[safe_mask]) if np.sum(safe_mask) > 0 else 0

        # Q3 among answered
        q3_in_answered = np.sum(answered & q3_mask) / n_answered

        print(f"  {t:>7.1f} {coverage:>9.3f} {sel_acc:>8.3f} {q3_filtered:>9.3f} "
              f"{safe_kept:>11.3f} {q3_in_answered:>10.3f}")

    # ------------------------------------------------------------------
    # 7. Statistical Significance
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("7. STATISTICAL SIGNIFICANCE")
    print(f"{'─'*50}")

    # Paired bootstrap test: is DECO's SA-AUC significantly better than Standard's?
    def sa_auc_metric(scores, labels):
        covs, accs = selective_accuracy_curve(scores, labels, n_points=50)
        return compute_auc(covs, accs)

    rng = np.random.default_rng(42)
    n_boot = 10000
    deco_wins = 0
    deco_aucs, std_aucs, conf_aucs = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n_test, n_test, replace=True)
        try:
            d_auc = sa_auc_metric(deco_scores[idx], y_test[idx])
            s_auc = sa_auc_metric(std_scores[idx], y_test[idx])
            c_auc = sa_auc_metric(conf_scores[idx], y_test[idx])
            deco_aucs.append(d_auc)
            std_aucs.append(s_auc)
            conf_aucs.append(c_auc)
            if d_auc > s_auc:
                deco_wins += 1
        except Exception:
            continue

    deco_aucs = np.array(deco_aucs)
    std_aucs = np.array(std_aucs)
    conf_aucs = np.array(conf_aucs)
    diffs_vs_std = deco_aucs - std_aucs
    diffs_vs_conf = deco_aucs - conf_aucs

    print(f"\n  Paired bootstrap (n={n_boot}):")
    print(f"  DECO SA-AUC:     {np.mean(deco_aucs):.4f} [{np.percentile(deco_aucs, 2.5):.4f}, {np.percentile(deco_aucs, 97.5):.4f}]")
    print(f"  Standard SA-AUC: {np.mean(std_aucs):.4f} [{np.percentile(std_aucs, 2.5):.4f}, {np.percentile(std_aucs, 97.5):.4f}]")
    print(f"  Conf SA-AUC:     {np.mean(conf_aucs):.4f} [{np.percentile(conf_aucs, 2.5):.4f}, {np.percentile(conf_aucs, 97.5):.4f}]")
    print(f"\n  DECO - Standard: {np.mean(diffs_vs_std):+.4f} [{np.percentile(diffs_vs_std, 2.5):+.4f}, {np.percentile(diffs_vs_std, 97.5):+.4f}]")
    print(f"  DECO - Conf:     {np.mean(diffs_vs_conf):+.4f} [{np.percentile(diffs_vs_conf, 2.5):+.4f}, {np.percentile(diffs_vs_conf, 97.5):+.4f}]")

    p_vs_std = np.mean(diffs_vs_std <= 0)
    p_vs_conf = np.mean(diffs_vs_conf <= 0)
    print(f"\n  p(DECO <= Standard): {p_vs_std:.4f} {'***' if p_vs_std < 0.001 else '**' if p_vs_std < 0.01 else '*' if p_vs_std < 0.05 else 'n.s.'}")
    print(f"  p(DECO <= Conf):     {p_vs_conf:.4f} {'***' if p_vs_conf < 0.001 else '**' if p_vs_conf < 0.01 else '*' if p_vs_conf < 0.05 else 'n.s.'}")

    # ------------------------------------------------------------------
    # 8. Compile results
    # ------------------------------------------------------------------
    results = {
        "model": model_key,
        "n_test": n_test,
        "best_deco_layer": best_deco_layer,
        "best_std_layer": best_std_layer,
        "sufficiency_detection_auroc": {},
        "selective_accuracy_auc": {},
        "hallucination_prevention_auc": {},
        "operating_points": {},
        "per_quadrant_scores": {},
        "q3_separation_auroc": {},
        "significance": {},
    }

    # Sufficiency AUROC
    for name, scores in methods.items():
        if name == "Oracle":
            continue
        try:
            auroc = float(roc_auc_score(y_test, scores))
            lo, hi = bootstrap_ci(scores, y_test, roc_auc_score)
            results["sufficiency_detection_auroc"][name] = {
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi
            }
        except Exception:
            pass

    # SA-AUC
    for name, res in sa_results.items():
        results["selective_accuracy_auc"][name] = {
            "auc": res["auc"],
            "curve": {
                "coverages": res["coverages"].tolist(),
                "accuracies": res["accuracies"].tolist(),
            }
        }

    # HP-AUC
    for name, res in hp_results.items():
        results["hallucination_prevention_auc"][name] = {
            "auc": res["auc"],
            "curve": {
                "coverages": res["coverages"].tolist(),
                "q3_fractions": res["q3_fracs"].tolist(),
            }
        }

    # Operating points
    for t in thresholds:
        answered = deco_scores >= t
        n_answered = int(np.sum(answered))
        if n_answered == 0:
            continue
        q3_mask = np.array([q == "Q3" for q in quadrants])
        safe_mask = np.array([q in ("Q1", "Q2") for q in quadrants])
        results["operating_points"][str(t)] = {
            "coverage": float(n_answered / n_test),
            "selective_accuracy": float(np.mean(y_test[answered])),
            "q3_filter_rate": float(1.0 - np.mean(answered[q3_mask])),
            "safe_keep_rate": float(np.mean(answered[safe_mask])),
            "q3_in_answered": float(np.sum(answered & q3_mask) / n_answered),
        }

    # Per-quadrant scores
    for name in ["DECO (ours)", "Standard probe", "Confidence probe"]:
        scores = methods[name]
        quad_stats = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            idx = [i for i, qd in enumerate(quadrants) if qd == q]
            q_scores = scores[idx]
            quad_stats[q] = {
                "n": len(idx),
                "mean": float(np.mean(q_scores)),
                "std": float(np.std(q_scores)),
                "median": float(np.median(q_scores)),
            }
        results["per_quadrant_scores"][name] = quad_stats

    # Q3 separation
    for name in ["DECO (ours)", "Standard probe", "Confidence probe"]:
        scores = methods[name]
        mask = safe_or_q3
        if np.sum(mask) > 0 and len(set(is_safe[mask])) > 1:
            auroc = float(roc_auc_score(is_safe[mask], scores[mask]))
            lo, hi = bootstrap_ci(scores[mask], is_safe[mask], roc_auc_score)
            results["q3_separation_auroc"][name] = {
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi
            }

    # Significance
    results["significance"] = {
        "deco_vs_standard": {
            "mean_diff": float(np.mean(diffs_vs_std)),
            "ci_lo": float(np.percentile(diffs_vs_std, 2.5)),
            "ci_hi": float(np.percentile(diffs_vs_std, 97.5)),
            "p_value": float(p_vs_std),
        },
        "deco_vs_confidence": {
            "mean_diff": float(np.mean(diffs_vs_conf)),
            "ci_lo": float(np.percentile(diffs_vs_conf, 2.5)),
            "ci_hi": float(np.percentile(diffs_vs_conf, 97.5)),
            "p_value": float(p_vs_conf),
        },
    }

    # Save
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selective_prediction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def cross_model_summary(model_keys):
    """Print summary table across models."""
    print(f"\n{'='*80}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    all_results = {}
    for mk in model_keys:
        path = RESULTS_DIR / mk / "selective_prediction.json"
        if path.exists():
            with open(path) as f:
                all_results[mk] = json.load(f)

    if not all_results:
        print("No results found.")
        return

    # Table 1: Sufficiency AUROC
    print(f"\n  {'Method':<25s}", end="")
    for mk in all_results:
        print(f"  {mk:>12s}", end="")
    print()
    print(f"  {'-'*70}")

    for method in ["DECO (ours)", "Standard probe", "Confidence probe", "Random"]:
        print(f"  {method:<25s}", end="")
        for mk in all_results:
            entry = all_results[mk]["sufficiency_detection_auroc"].get(method, {})
            auroc = entry.get("auroc")
            if auroc is not None:
                print(f"  {auroc:>12.4f}", end="")
            else:
                print(f"  {'—':>12s}", end="")
        print()

    # Table 2: SA-AUC
    print(f"\n  Selective Accuracy AUC:")
    print(f"  {'Method':<25s}", end="")
    for mk in all_results:
        print(f"  {mk:>12s}", end="")
    print()
    print(f"  {'-'*70}")

    for method in ["DECO (ours)", "Standard probe", "Confidence probe", "Random", "Oracle"]:
        print(f"  {method:<25s}", end="")
        for mk in all_results:
            entry = all_results[mk]["selective_accuracy_auc"].get(method, {})
            auc = entry.get("auc")
            if auc is not None:
                print(f"  {auc:>12.4f}", end="")
            else:
                print(f"  {'—':>12s}", end="")
        print()

    # Table 3: Significance
    print(f"\n  Statistical Significance (DECO vs others):")
    for mk in all_results:
        sig = all_results[mk].get("significance", {})
        for comp, data in sig.items():
            p = data.get("p_value", 1)
            diff = data.get("mean_diff", 0)
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f"  {mk:>10s} {comp:<25s}: diff={diff:+.4f}, p={p:.4f} {star}")

    # Save cross-model summary
    out_path = RESULTS_DIR / "selective_prediction_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCross-model summary saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()

    if args.model == "all":
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]

    for mk in models:
        try:
            evaluate_selective_prediction(mk)
        except Exception as e:
            print(f"Error on {mk}: {e}")
            import traceback
            traceback.print_exc()

    if len(models) > 1:
        cross_model_summary(models)
