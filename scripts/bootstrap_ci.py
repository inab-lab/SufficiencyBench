#!/usr/bin/env python3
"""
bootstrap_ci.py — Bootstrap 95% confidence intervals for all AUROC values
reported in the SufficiencyBench paper.

Covers:
  - Table 1: Main comparison (probes + baselines) across 3 models
  - Table 2: Real retrieval evaluation
  - Table 5: Frontier/API model results
  - Orthogonality: cosine similarity context from permutation tests

For probe-based methods (Standard, CSP, Neural variants), we re-train on the
train split and compute per-example predicted probabilities on the test split,
enabling exact bootstrap resampling of AUROC.

For methods where only aggregate metrics are available (verbalized confidence,
token entropy, semantic entropy, LLM judge, ReDeEP, API models), we use the
Hanley-McNeil (1982) formula for AUROC standard error, which gives a
normal-approximation CI based on the AUROC value and positive/negative counts.

Usage:
    python scripts/bootstrap_ci.py [--n-boot 1000] [--seed 42]
"""

import numpy as np
import json
import argparse
import warnings
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "data" / "experiments" / "hidden_states"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"

# Best layers from paper (found via validation)
BEST_LAYERS = {"llama": 12, "mistral": 12, "qwen": 16}
BEST_LAYERS_DECO = {"llama": 8, "mistral": 12, "qwen": 16}

MODELS = ["llama", "mistral", "qwen"]
MODEL_DISPLAY = {
    "llama": "Llama 3.1 8B",
    "mistral": "Mistral 7B",
    "qwen": "Qwen 2.5 7B",
}

# Test set composition: 1112 sufficient, 1112 insufficient
N_POS_TEST = 1112
N_NEG_TEST = 1112


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(model, split, layer):
    """Load hidden states at a specific layer and labels."""
    base = BASE / model / split
    h_ctx = np.load(base / "h_with_context.npy")[:, layer, :]
    h_q = np.load(base / "h_question_only.npy")[:, layer, :]
    labels = np.load(base / "labels.npy")
    return h_ctx, h_q, labels


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bootstrap and CI computation
# ---------------------------------------------------------------------------

def bootstrap_auroc(y_true, y_score, n_bootstrap=1000, seed=42):
    """Compute bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, ys))
    aurocs = np.array(aurocs)
    point = roc_auc_score(y_true, y_score)
    return point, np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


def hanley_mcneil_ci(auroc, n_pos, n_neg, alpha=0.05):
    """
    Hanley-McNeil (1982) normal-approximation CI for AUROC.
    Returns (point, ci_lower, ci_upper).
    """
    A = auroc
    Q1 = A / (2 - A)
    Q2 = 2 * A * A / (1 + A)
    se = np.sqrt(
        (A * (1 - A) + (n_pos - 1) * (Q1 - A * A)
         + (n_neg - 1) * (Q2 - A * A))
        / (n_pos * n_neg)
    )
    z = norm.ppf(1 - alpha / 2)
    return A, max(0.0, A - z * se), min(1.0, A + z * se)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, y_train, X_test, y_test, method="logreg", seed=42):
    """Train a probe and return test predicted probabilities."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if method == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 512),
            activation="relu",
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_test_s)[:, 1]


# ---------------------------------------------------------------------------
# Table 1: Main comparison (probes with exact bootstrap, baselines with H-M)
# ---------------------------------------------------------------------------

def compute_table1(n_boot, seed):
    """Compute AUROC CIs for the main results table."""
    print("\n" + "=" * 80)
    print("TABLE 1: Sufficiency Detection Performance (AUROC with 95% CIs)")
    print("=" * 80)

    all_results = {}

    for model in MODELS:
        print(f"\n  {MODEL_DISPLAY[model]}")
        print(f"  {'-' * 60}")

        layer_std = BEST_LAYERS[model]
        layer_deco = BEST_LAYERS_DECO[model]

        # Load train and test data at standard layer
        h_ctx_tr, h_q_tr, y_train = load_data(model, "train", layer_std)
        h_ctx_te, h_q_te, y_test = load_data(model, "test", layer_std)

        # Also load at DECO layer if different
        if layer_deco != layer_std:
            h_ctx_tr_d, h_q_tr_d, _ = load_data(model, "train", layer_deco)
            h_ctx_te_d, h_q_te_d, _ = load_data(model, "test", layer_deco)
        else:
            h_ctx_tr_d, h_q_tr_d = h_ctx_tr, h_q_tr
            h_ctx_te_d, h_q_te_d = h_ctx_te, h_q_te

        model_results = {}

        # --- Probe-based methods: exact bootstrap on per-example scores ---
        probe_variants = {
            "Standard Probe": (h_ctx_tr, h_ctx_te, "logreg"),
            "CSP Probe": (h_ctx_tr_d - h_q_tr_d, h_ctx_te_d - h_q_te_d, "logreg"),
            "Standard (Neural)": (h_ctx_tr, h_ctx_te, "neural"),
            "CSP (Neural)": (h_ctx_tr_d - h_q_tr_d, h_ctx_te_d - h_q_te_d, "neural"),
        }

        for name, (X_tr, X_te, method) in probe_variants.items():
            print(f"    Training {name}...", end=" ", flush=True)
            y_score = train_and_evaluate(X_tr, y_train, X_te, y_test, method, seed)
            point, lo, hi = bootstrap_auroc(y_test, y_score, n_boot, seed)
            hw = (hi - lo) / 2
            model_results[name] = {
                "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_half": float(hw), "ci_method": "bootstrap",
            }
            print(f"AUROC = {point:.3f} +/- {hw:.3f} [{lo:.3f}, {hi:.3f}]")

        # --- Baselines: Hanley-McNeil approximation ---
        baselines_path = RESULTS_DIR / model / "baseline_results.json"
        if baselines_path.exists():
            baselines = load_json(baselines_path)
            baseline_map = {
                "verbalized_confidence": "Verbalized Confidence",
                "token_entropy": "Token Entropy",
                "generation_match": "Generation Match",
                "embedding_similarity": "Embedding Similarity",
            }
            for key, display in baseline_map.items():
                if key in baselines and "overall" in baselines[key]:
                    auroc = baselines[key]["overall"]["auroc"]
                    point, lo, hi = hanley_mcneil_ci(auroc, N_POS_TEST, N_NEG_TEST)
                    hw = (hi - lo) / 2
                    model_results[display] = {
                        "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                        "ci_half": float(hw), "ci_method": "hanley_mcneil",
                    }
                    print(f"    {display}: {point:.3f} +/- {hw:.3f} [{lo:.3f}, {hi:.3f}]")

        # --- Semantic Entropy ---
        se_path = RESULTS_DIR / model / "semantic_entropy_results.json"
        if se_path.exists():
            se = load_json(se_path)
            auroc = se["overall"]["auroc"]
            point, lo, hi = hanley_mcneil_ci(auroc, N_POS_TEST, N_NEG_TEST)
            hw = (hi - lo) / 2
            model_results["Semantic Entropy"] = {
                "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_half": float(hw), "ci_method": "hanley_mcneil",
            }
            print(f"    Semantic Entropy: {point:.3f} +/- {hw:.3f} [{lo:.3f}, {hi:.3f}]")

        # --- LLM Judge ---
        judge_path = RESULTS_DIR / model / "llm_judge_results.json"
        if judge_path.exists():
            judge = load_json(judge_path)
            auroc = judge["overall"]["auroc"]
            point, lo, hi = hanley_mcneil_ci(auroc, N_POS_TEST, N_NEG_TEST)
            hw = (hi - lo) / 2
            model_results["LLM Judge"] = {
                "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_half": float(hw), "ci_method": "hanley_mcneil",
            }
            print(f"    LLM Judge: {point:.3f} +/- {hw:.3f} [{lo:.3f}, {hi:.3f}]")

        # --- ReDeEP ---
        redeep_path = RESULTS_DIR / model / "redeep_results.json"
        if redeep_path.exists():
            redeep = load_json(redeep_path)
            for sub_key, sub_name in [("ecs", "ReDeEP-ECS"),
                                       ("pks", "ReDeEP-PKS"),
                                       ("combined", "ReDeEP-Combined")]:
                if sub_key in redeep and "overall" in redeep[sub_key]:
                    auroc = redeep[sub_key]["overall"]["auroc"]
                    point, lo, hi = hanley_mcneil_ci(auroc, N_POS_TEST, N_NEG_TEST)
                    hw = (hi - lo) / 2
                    model_results[sub_name] = {
                        "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                        "ci_half": float(hw), "ci_method": "hanley_mcneil",
                    }
                    print(f"    {sub_name}: {point:.3f} +/- {hw:.3f} [{lo:.3f}, {hi:.3f}]")

        all_results[model] = model_results

    return all_results


# ---------------------------------------------------------------------------
# Table 2: Real retrieval
# ---------------------------------------------------------------------------

def compute_table2():
    """Compute CIs for real retrieval AUROC (Table 2 in paper)."""
    print("\n" + "=" * 80)
    print("TABLE 2: Real Retrieval Evaluation (AUROC with 95% CIs)")
    print("=" * 80)

    results = {}
    for model in MODELS:
        rr_path = RESULTS_DIR / model / "real_retrieval_results.json"
        if not rr_path.exists():
            continue
        rr = load_json(rr_path)
        n_pairs = rr["n_pairs"]  # 300 pairs = 600 total examples

        model_results = {}
        for probe_key, display in [("standard", "Standard Probe"),
                                    ("csp", "CSP Probe")]:
            if probe_key in rr["probes"]:
                auroc = rr["probes"][probe_key]["auroc"]
                point, lo, hi = hanley_mcneil_ci(auroc, n_pairs, n_pairs)
                hw = (hi - lo) / 2
                model_results[display] = {
                    "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                    "ci_half": float(hw),
                }
                print(f"  {MODEL_DISPLAY[model]} / {display}: "
                      f"{point:.3f} +/- {hw:.3f} [{lo:.3f}, {hi:.3f}]")

        results[model] = model_results

    return results


# ---------------------------------------------------------------------------
# Table 5: Frontier / API models
# ---------------------------------------------------------------------------

def compute_table5():
    """Compute CIs for frontier model results (Table 5 in paper)."""
    print("\n" + "=" * 80)
    print("TABLE 5: Frontier Model Evaluation (AUROC with 95% CIs)")
    print("=" * 80)

    api_dir = RESULTS_DIR / "api_models"
    api_files = [
        ("GPT-4o", "openai_results.json"),
        ("GPT-4o-mini", "openai_mini_results.json"),
        ("GPT-4o + CoT", "openai_cot_results.json"),
        ("Claude Opus 4", "anthropic_results.json"),
        ("Claude Sonnet 4", "anthropic_sonnet_results.json"),
        ("Claude Opus 4 + CoT", "anthropic_cot_results.json"),
        ("Gemini 2.5 Flash", "gemini_flash_results.json"),
    ]

    results = {}
    for display_name, filename in api_files:
        path = api_dir / filename
        if not path.exists():
            continue
        data = load_json(path)
        model_results = {}

        # Standard API result files have llm_judge + verbalized_confidence
        if "llm_judge" in data:
            for mkey, mname in [("llm_judge", "LLM Judge"),
                                 ("verbalized_confidence", "Verb. Conf.")]:
                if mkey not in data:
                    continue
                mdata = data[mkey]
                auroc = mdata["overall"]["auroc"]
                quads = mdata.get("per_quadrant", {})
                n_suf = quads.get("Q1", {}).get("n", 50) + quads.get("Q2", {}).get("n", 200)
                n_insuf = quads.get("Q3", {}).get("n", 49) + quads.get("Q4", {}).get("n", 201)
                point, lo, hi = hanley_mcneil_ci(auroc, n_suf, n_insuf)
                hw = (hi - lo) / 2
                model_results[mname] = {
                    "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                    "ci_half": float(hw),
                }
        # CoT-only files
        elif "overall" in data:
            auroc = data["overall"]["auroc"]
            quads = data.get("per_quadrant", {})
            n_suf = quads.get("Q1", {}).get("n", 50) + quads.get("Q2", {}).get("n", 200)
            n_insuf = quads.get("Q3", {}).get("n", 49) + quads.get("Q4", {}).get("n", 201)
            point, lo, hi = hanley_mcneil_ci(auroc, n_suf, n_insuf)
            hw = (hi - lo) / 2
            model_results["CoT Judge"] = {
                "auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_half": float(hw),
            }

        results[display_name] = model_results
        for mname, mr in model_results.items():
            print(f"  {display_name} / {mname}: "
                  f"{mr['auroc']:.3f} +/- {mr['ci_half']:.3f} "
                  f"[{mr['ci_lo']:.3f}, {mr['ci_hi']:.3f}]")

    return results


# ---------------------------------------------------------------------------
# Orthogonality: cosine similarity with permutation context
# ---------------------------------------------------------------------------

def compute_orthogonality():
    """Report cosine similarity values with permutation null context."""
    print("\n" + "=" * 80)
    print("ORTHOGONALITY: Cosine Similarities (with permutation null context)")
    print("=" * 80)

    results = {}
    for model in MODELS:
        orth_path = RESULTS_DIR / model / "orthogonality_analysis.json"
        if not orth_path.exists():
            continue
        orth = load_json(orth_path)

        cos_suf_conf = orth["cos_suf_conf"]
        cos_deco_conf = orth.get("cos_deco_conf")
        cos_suf_deco = orth.get("cos_suf_deco")
        null_std = orth.get("null_std", 0.015)
        perm_p = orth.get("permutation_p_value")

        # Use permutation null_std as conservative SE for the observed cosine.
        # This yields wider CIs than the true sampling variability.
        z = norm.ppf(0.975)
        model_results = {}

        model_results["cos(suf, conf)"] = {
            "value": float(cos_suf_conf),
            "ci_lo": float(cos_suf_conf - z * null_std),
            "ci_hi": float(cos_suf_conf + z * null_std),
            "permutation_p": perm_p,
            "null_std": float(null_std),
        }
        if cos_deco_conf is not None:
            model_results["cos(CSP, conf)"] = {
                "value": float(cos_deco_conf),
                "ci_lo": float(cos_deco_conf - z * null_std),
                "ci_hi": float(cos_deco_conf + z * null_std),
            }
        if cos_suf_deco is not None:
            model_results["cos(suf, CSP)"] = {
                "value": float(cos_suf_deco),
            }

        results[model] = model_results

        for name, vals in model_results.items():
            v = vals["value"]
            if "ci_lo" in vals:
                print(f"  {MODEL_DISPLAY[model]} / {name}: "
                      f"{v:.4f} [{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]"
                      + (f"  (perm p={perm_p})" if "permutation_p" in vals else ""))
            else:
                print(f"  {MODEL_DISPLAY[model]} / {name}: {v:.4f}")

    return results


# ---------------------------------------------------------------------------
# Confidence projection deltas
# ---------------------------------------------------------------------------

def compute_projection():
    """Report confidence projection delta AUROC values."""
    print("\n" + "=" * 80)
    print("CONFIDENCE PROJECTION: Delta AUROC (removing confound direction)")
    print("=" * 80)

    proj_path = RESULTS_DIR / "confidence_projection_combined.json"
    if not proj_path.exists():
        print("  [SKIP] confidence_projection_combined.json not found")
        return {}

    proj = load_json(proj_path)
    results = {}

    for model in MODELS:
        if model not in proj:
            continue
        d = proj[model]
        summary = d.get("summary", {})
        results[model] = {
            "delta_suf_auroc": summary.get("delta_sufficiency"),
            "delta_conf_auroc": summary.get("delta_confidence"),
            "random_delta_mean": d.get("random_projection_delta_mean"),
            "random_delta_std": d.get("random_projection_delta_std"),
            "orthogonality_validated": summary.get("orthogonality_validated"),
        }
        print(f"  {MODEL_DISPLAY[model]}:")
        print(f"    Delta suf AUROC (after removing conf): "
              f"{summary.get('delta_sufficiency', 0):+.5f}")
        print(f"    Delta conf AUROC (after removing suf): "
              f"{summary.get('delta_confidence', 0):+.5f}")
        print(f"    Random projection delta: "
              f"{d.get('random_projection_delta_mean', 0):.5f} "
              f"+/- {d.get('random_projection_delta_std', 0):.5f}")

    return results


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def generate_latex_table1(results):
    """Generate LaTeX for Table 1 with CIs."""
    print("\n" + "=" * 80)
    print("LATEX: Table 1 -- Main Results with 95% CIs")
    print("=" * 80)

    # Methods in paper Table 1 order
    methods_paper = [
        "Standard Probe", "CSP Probe",
        "Verbalized Confidence", "Token Entropy",
        "Generation Match", "Embedding Similarity",
    ]

    # Extended methods for appendix
    methods_extended = methods_paper + [
        "Standard (Neural)", "CSP (Neural)",
        "Semantic Entropy", "LLM Judge",
        "ReDeEP-ECS", "ReDeEP-PKS",
    ]

    # Find best AUROC per model among paper methods
    best_per_model = {}
    for model in MODELS:
        aurocs = [(m, results[model][m]["auroc"])
                  for m in methods_paper if m in results.get(model, {})]
        if aurocs:
            best_per_model[model] = max(aurocs, key=lambda x: x[1])[0]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sufficiency detection AUROC on \bench{} test set "
                 r"with 95\% confidence intervals (bootstrap for probe methods, "
                 r"Hanley-McNeil for others). Best per model in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mistral 7B & Llama 3.1 8B & Qwen 2.5 7B \\")
    lines.append(r"\midrule")

    model_order = ["mistral", "llama", "qwen"]

    for method in methods_extended:
        cells = []
        any_found = False
        for model in model_order:
            if model in results and method in results[model]:
                r = results[model][method]
                is_best = (best_per_model.get(model) == method)
                if is_best:
                    cell = (f"\\textbf{{{r['auroc']:.3f}}}"
                            f"$\\pm${r['ci_half']:.3f}")
                else:
                    cell = f"{r['auroc']:.3f}$\\pm${r['ci_half']:.3f}"
                cells.append(cell)
                any_found = True
            else:
                cells.append("---")

        if any_found:
            lines.append(f"  {method} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

        # Add midrules to separate method families
        if method == "CSP Probe":
            lines.append(r"\midrule")
        elif method == "Embedding Similarity":
            lines.append(r"\midrule")
        elif method == "CSP (Neural)":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    print(latex)
    return latex


def generate_latex_table2(results):
    """Generate LaTeX for Table 2 (real retrieval) with CIs."""
    print("\n" + "=" * 80)
    print("LATEX: Table 2 -- Real Retrieval with 95% CIs")
    print("=" * 80)

    model_order = ["mistral", "llama", "qwen"]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Real retrieval: probe AUROC with 95\% CIs "
                 r"(Hanley-McNeil, $n$=600).}")
    lines.append(r"\label{tab:real_retrieval_ci}")
    lines.append(r"\begin{tabular}{@{}llc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & Probe & AUROC \\")
    lines.append(r"\midrule")

    for i, model in enumerate(model_order):
        if model not in results:
            continue
        if i > 0:
            lines.append(r"\midrule")
        first = True
        for method in ["Standard Probe", "CSP Probe"]:
            if method not in results[model]:
                continue
            r = results[model][method]
            mcol = MODEL_DISPLAY[model] if first else ""
            lines.append(f"  {mcol} & {method} & "
                         f"{r['auroc']:.3f}$\\pm${r['ci_half']:.3f} \\\\")
            first = False

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    print(latex)
    return latex


def generate_latex_table5(results):
    """Generate LaTeX for Table 5 (frontier models) with CIs."""
    print("\n" + "=" * 80)
    print("LATEX: Table 5 -- Frontier Models with 95% CIs")
    print("=" * 80)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Frontier model sufficiency detection. AUROC with "
                 r"95\% CIs (Hanley-McNeil, $n$=500).}")
    lines.append(r"\label{tab:frontier_ci}")
    lines.append(r"\begin{tabular}{@{}llc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & Method & AUROC \\")
    lines.append(r"\midrule")

    prev_provider = None
    for model_name, methods in results.items():
        # Group by provider for midrules
        provider = model_name.split()[0]  # GPT / Claude / Gemini
        if prev_provider is not None and provider != prev_provider:
            lines.append(r"\midrule")
        prev_provider = provider

        first = True
        for mname, mr in methods.items():
            mcol = model_name if first else ""
            lines.append(f"  {mcol} & {mname} & "
                         f"{mr['auroc']:.3f}$\\pm${mr['ci_half']:.3f} \\\\")
            first = False

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    print(latex)
    return latex


# ---------------------------------------------------------------------------
# Compact summary
# ---------------------------------------------------------------------------

def print_summary(t1):
    """Print a compact summary table."""
    print("\n" + "=" * 80)
    print("COMPACT SUMMARY: AUROC +/- half-width of 95% CI")
    print("=" * 80)

    all_methods = [
        "Standard Probe", "CSP Probe",
        "Standard (Neural)", "CSP (Neural)",
        "Verbalized Confidence", "Token Entropy",
        "Generation Match", "Embedding Similarity",
        "Semantic Entropy", "LLM Judge",
        "ReDeEP-ECS", "ReDeEP-PKS", "ReDeEP-Combined",
    ]

    header = f"{'Method':<25s}"
    for model in MODELS:
        header += f"  {MODEL_DISPLAY[model]:>20s}"
    print(header)
    print("-" * len(header))

    for method in all_methods:
        row = f"{method:<25s}"
        found = False
        for model in MODELS:
            if model in t1 and method in t1[model]:
                r = t1[model][method]
                row += f"  {r['auroc']:.3f}+/-{r['ci_half']:.3f}       "
                found = True
            else:
                row += f"  {'---':>20s}"
        if found:
            print(row)


# ---------------------------------------------------------------------------
# Save all results
# ---------------------------------------------------------------------------

def save_results(t1, t2, t5, orth, proj, out_path):
    """Save all CI results to JSON."""
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out = {
        "table1_main": to_native(t1),
        "table2_real_retrieval": to_native(t2),
        "table5_frontier": to_native(t5),
        "orthogonality": to_native(orth),
        "confidence_projection": to_native(proj),
        "config": {
            "n_bootstrap": 1000,
            "seed": 42,
            "note": "Bootstrap CIs for probe methods; Hanley-McNeil CIs for "
                    "methods with only aggregate AUROC values.",
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAll results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CIs for all SufficiencyBench AUROC values")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Computing 95% confidence intervals "
          f"(n_boot={args.n_boot}, seed={args.seed})")

    # Compute all CIs
    t1 = compute_table1(args.n_boot, args.seed)
    t2 = compute_table2()
    t5 = compute_table5()
    orth = compute_orthogonality()
    proj = compute_projection()

    # Summary
    print_summary(t1)

    # LaTeX output
    latex1 = generate_latex_table1(t1)
    latex2 = generate_latex_table2(t2)
    latex5 = generate_latex_table5(t5)

    # Save JSON
    out_path = RESULTS_DIR / "bootstrap_confidence_intervals.json"
    save_results(t1, t2, t5, orth, proj, out_path)

    # Also save the LaTeX snippets
    latex_path = TABLES_DIR / "table1_with_ci.tex"
    with open(latex_path, "w") as f:
        f.write(latex1)
    print(f"Table 1 LaTeX saved to: {latex_path}")

    latex_path2 = TABLES_DIR / "table2_real_retrieval_ci.tex"
    with open(latex_path2, "w") as f:
        f.write(latex2)
    print(f"Table 2 LaTeX saved to: {latex_path2}")

    latex_path5 = TABLES_DIR / "table5_frontier_ci.tex"
    with open(latex_path5, "w") as f:
        f.write(latex5)
    print(f"Table 5 LaTeX saved to: {latex_path5}")

    print("\nDone.")


if __name__ == "__main__":
    main()
