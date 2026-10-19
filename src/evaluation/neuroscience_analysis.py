#!/usr/bin/env python3
"""
neuroscience_analysis.py — Neuroscience-inspired representational analysis
of context sufficiency in LLM hidden states.

Applies techniques from computational neuroscience (RSA, CKA, geometric
analysis, CCGP) to provide probe-independent evidence about how sufficiency
and confidence are represented in LLM hidden states.

Usage:
  MALLOC_ARENA_MAX=2 python src/evaluation/neuroscience_analysis.py --model mistral
  MALLOC_ARENA_MAX=2 python src/evaluation/neuroscience_analysis.py --model all

Memory-safe: designed for overcommit_memory=2 systems. Opens at most one
mmap file at a time and processes layer-by-layer.
"""

import os
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import json
import argparse
import sys
import gc
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, FIGURES_DIR, MODEL_CONFIGS

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODEL_LABELS = {"mistral": "Mistral 7B", "qwen": "Qwen 2.5 7B", "llama": "Llama 3.1 8B"}


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading — Memory-safe helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_labels(model_key, split):
    """Load only labels and metadata (tiny memory footprint)."""
    d = HIDDEN_STATES_DIR / model_key / split
    labels_suf = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    labels_conf = np.array([1 if m["model_confident"] else 0 for m in metadata])
    return labels_suf, labels_conf, metadata


def load_layer_slice(model_key, split, filename, indices, layer):
    """Load a single layer slice for specific indices. Returns (N, D) float32 array.

    Opens mmap, extracts the slice, closes mmap immediately.
    This avoids keeping large mmap reservations open.
    """
    path = HIDDEN_STATES_DIR / model_key / split / filename
    h = np.load(str(path), mmap_mode="r")
    # Extract just what we need and copy to a real array
    data = np.array(h[np.ix_(indices, [layer])].squeeze(1), dtype=np.float32)
    del h  # Close mmap
    return data


def load_layer_all(model_key, split, filename, layer):
    """Load a single layer for ALL examples. Returns (N, D) float32."""
    path = HIDDEN_STATES_DIR / model_key / split / filename
    h = np.load(str(path), mmap_mode="r")
    data = np.array(h[:, layer, :], dtype=np.float32)
    del h
    return data


def get_n_examples(model_key, split):
    """Get number of examples without loading the full array."""
    path = HIDDEN_STATES_DIR / model_key / split / "h_with_context.npy"
    h = np.load(str(path), mmap_mode="r")
    n = h.shape[0]
    del h
    return n


def balanced_subsample(labels_suf, labels_conf, n_per_quadrant=100, seed=42):
    """Return indices balanced across 4 quadrants (suf x conf)."""
    rng = np.random.RandomState(seed)
    indices = []
    for s in [0, 1]:
        for c in [0, 1]:
            mask = (labels_suf == s) & (labels_conf == c)
            pool = np.where(mask)[0]
            n = min(n_per_quadrant, len(pool))
            if n > 0:
                chosen = rng.choice(pool, n, replace=False)
                indices.append(chosen)
    return np.concatenate(indices)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1: RSA (Representational Similarity Analysis)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rdm(X, metric="correlation"):
    """Compute representational dissimilarity matrix. X is (N, D)."""
    return squareform(pdist(X, metric=metric))


def make_model_rdms(labels_suf, labels_conf):
    """Create theoretical model RDMs from labels."""
    N = len(labels_suf)
    suf_rdm = np.zeros((N, N), dtype=np.float32)
    conf_rdm = np.zeros((N, N), dtype=np.float32)
    quad_rdm = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            suf_rdm[i, j] = suf_rdm[j, i] = 0.0 if labels_suf[i] == labels_suf[j] else 1.0
            conf_rdm[i, j] = conf_rdm[j, i] = 0.0 if labels_conf[i] == labels_conf[j] else 1.0
            same_quad = (labels_suf[i] == labels_suf[j]) and (labels_conf[i] == labels_conf[j])
            quad_rdm[i, j] = quad_rdm[j, i] = 0.0 if same_quad else 1.0

    return {"sufficiency": suf_rdm, "confidence": conf_rdm, "quadrant": quad_rdm}


def rdm_upper_triangle(rdm):
    """Extract upper triangle of RDM as a vector."""
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def rsa_correlation(neural_rdm, model_rdm):
    """Spearman correlation between upper triangles of two RDMs."""
    v1 = rdm_upper_triangle(neural_rdm)
    v2 = rdm_upper_triangle(model_rdm)
    rho, p = spearmanr(v1, v2)
    return float(rho), float(p)


def rsa_permutation_test(neural_rdm, model_rdm, n_perms=500, seed=42):
    """Permutation test for RSA significance."""
    rng = np.random.RandomState(seed)
    observed_rho, _ = rsa_correlation(neural_rdm, model_rdm)

    N = neural_rdm.shape[0]
    null_rhos = np.zeros(n_perms)
    neural_vec = rdm_upper_triangle(neural_rdm)

    for i in range(n_perms):
        perm = rng.permutation(N)
        perm_rdm = model_rdm[np.ix_(perm, perm)]
        perm_vec = rdm_upper_triangle(perm_rdm)
        null_rhos[i], _ = spearmanr(neural_vec, perm_vec)

    p_value = float(np.mean(null_rhos >= observed_rho))
    return observed_rho, p_value


def run_rsa(model_key, split="test", n_per_quadrant=75):
    """Run RSA analysis across layers. Loads data layer-by-layer."""
    print("\n--- Analysis 1: RSA ---")
    labels_suf, labels_conf, _ = load_labels(model_key, split)
    idx = balanced_subsample(labels_suf, labels_conf, n_per_quadrant=n_per_quadrant)
    N = len(idx)
    print(f"  RSA: {N} subsampled examples ({n_per_quadrant}/quadrant)")

    sub_suf = labels_suf[idx]
    sub_conf = labels_conf[idx]
    model_rdms = make_model_rdms(sub_suf, sub_conf)

    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    layers_to_eval = list(range(0, n_layers, 2)) + [n_layers - 1]
    layers_to_eval = sorted(set(layers_to_eval))

    results = {"layers": [], "sufficiency_rho": [], "confidence_rho": [],
               "quadrant_rho": [], "sufficiency_p": [], "confidence_p": [],
               "n_examples": N}

    for layer in layers_to_eval:
        X = load_layer_slice(model_key, split, "h_with_context.npy", idx, layer)
        neural_rdm = compute_rdm(X)
        del X

        suf_rho, suf_p = rsa_permutation_test(neural_rdm, model_rdms["sufficiency"], n_perms=500)
        conf_rho, conf_p = rsa_permutation_test(neural_rdm, model_rdms["confidence"], n_perms=500)
        quad_rho, _ = rsa_correlation(neural_rdm, model_rdms["quadrant"])

        del neural_rdm
        gc.collect()

        results["layers"].append(layer)
        results["sufficiency_rho"].append(suf_rho)
        results["confidence_rho"].append(conf_rho)
        results["quadrant_rho"].append(quad_rho)
        results["sufficiency_p"].append(suf_p)
        results["confidence_p"].append(conf_p)

        sig_suf = "*" if suf_p < 0.01 else ""
        sig_conf = "*" if conf_p < 0.01 else ""
        print(f"    L{layer:2d}  suf_rho={suf_rho:+.4f}{sig_suf}  "
              f"conf_rho={conf_rho:+.4f}{sig_conf}  quad_rho={quad_rho:+.4f}")

    del model_rdms, labels_suf, labels_conf
    gc.collect()
    return results


def plot_rsa(rsa_results, model_key):
    """Plot RSA profiles across layers."""
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = rsa_results["layers"]

    ax.plot(layers, rsa_results["sufficiency_rho"], "o-", color="#D32F2F",
            label="Sufficiency", linewidth=2, markersize=4)
    ax.plot(layers, rsa_results["confidence_rho"], "s-", color="#1976D2",
            label="Confidence", linewidth=2, markersize=4)
    ax.plot(layers, rsa_results["quadrant_rho"], "^-", color="#7B1FA2",
            label="Quadrant", linewidth=1.5, markersize=3, alpha=0.7)

    for i, layer in enumerate(layers):
        if rsa_results["sufficiency_p"][i] < 0.01:
            ax.annotate("*", (layer, rsa_results["sufficiency_rho"][i]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=12, color="#D32F2F")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman $\\rho$ (RSA)")
    ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)}: Representational Similarity Analysis",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_rsa_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_rsa_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_rsa_{model_key}")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2: CKA (Centered Kernel Alignment)
# ──────────────────────────────────────────────────────────────────────────────

def center_gram(K):
    """Center a Gram matrix."""
    N = K.shape[0]
    col_mean = K.mean(axis=0)
    tot_mean = col_mean.mean()
    return K - col_mean[None, :] - col_mean[:, None] + tot_mean


def linear_cka(X, Y):
    """Compute linear CKA between X (N,D1) and Y (N,D2)."""
    K = X @ X.T
    L = Y @ Y.T
    Kc = center_gram(K)
    Lc = center_gram(L)
    hsic = np.sum(Kc * Lc)
    norm = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    if norm < 1e-10:
        return 0.0
    return float(hsic / norm)


def run_cka_cross_layer(model_key, split="test", n_subsample=400, seed=42):
    """Compute CKA between all pairs of layers.

    Memory-safe: computes Gram matrices one at a time, stores only
    the HSIC norms (scalars) and cross products incrementally.
    """
    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    rng = np.random.RandomState(seed)
    N_total = get_n_examples(model_key, split)
    idx = rng.choice(N_total, min(n_subsample, N_total), replace=False)
    n = len(idx)

    # Evaluate every 2nd layer to reduce computation
    eval_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    eval_layers = sorted(set(eval_layers))

    print(f"  CKA cross-layer: {n} examples, {len(eval_layers)} layers")

    # Store centered Gram norms and the centered Gram matrices
    # Each Gram matrix is n×n float32 = 400*400*4 = 640KB — very small
    grams = {}
    gram_norms = {}

    for layer in eval_layers:
        X = load_layer_slice(model_key, split, "h_with_context.npy", idx, layer)
        K = X @ X.T
        Kc = center_gram(K)
        grams[layer] = Kc
        gram_norms[layer] = float(np.sum(Kc * Kc))
        del X, K
        if layer % 8 == 0:
            print(f"    Precomputed gram for layer {layer}")

    cka_matrix = np.zeros((len(eval_layers), len(eval_layers)))
    for i, li in enumerate(eval_layers):
        for j, lj in enumerate(eval_layers):
            if i <= j:
                hsic = float(np.sum(grams[li] * grams[lj]))
                norm = np.sqrt(gram_norms[li] * gram_norms[lj])
                cka_val = hsic / norm if norm > 1e-10 else 0.0
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val

    del grams, gram_norms
    gc.collect()
    return cka_matrix, eval_layers


def run_cka_cross_condition(model_key, split="test", n_subsample=300, seed=42):
    """CKA between h(q+c) and h(q) at each layer, split by sufficiency."""
    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    labels_suf, _, _ = load_labels(model_key, split)
    rng = np.random.RandomState(seed)

    suf_idx = np.where(labels_suf == 1)[0]
    insuf_idx = np.where(labels_suf == 0)[0]
    suf_sub = rng.choice(suf_idx, min(n_subsample, len(suf_idx)), replace=False)
    insuf_sub = rng.choice(insuf_idx, min(n_subsample, len(insuf_idx)), replace=False)

    layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    layers = sorted(set(layers))

    cka_suf = []
    cka_insuf = []

    for layer in layers:
        # Load one slice at a time
        X_suf_qc = load_layer_slice(model_key, split, "h_with_context.npy", suf_sub, layer)
        X_suf_q = load_layer_slice(model_key, split, "h_question_only.npy", suf_sub, layer)
        cka_suf.append(linear_cka(X_suf_qc, X_suf_q))
        del X_suf_qc, X_suf_q

        X_insuf_qc = load_layer_slice(model_key, split, "h_with_context.npy", insuf_sub, layer)
        X_insuf_q = load_layer_slice(model_key, split, "h_question_only.npy", insuf_sub, layer)
        cka_insuf.append(linear_cka(X_insuf_qc, X_insuf_q))
        del X_insuf_qc, X_insuf_q
        gc.collect()

    del labels_suf
    gc.collect()
    return {"layers": layers, "cka_sufficient": cka_suf, "cka_insufficient": cka_insuf}


def plot_cka_heatmap(cka_matrix, eval_layers, model_key):
    """Plot cross-layer CKA heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka_matrix, cmap="magma", vmin=0, vmax=1, aspect="equal")
    # Label with actual layer numbers
    tick_pos = list(range(len(eval_layers)))
    tick_labels = [str(l) for l in eval_layers]
    # Show every other tick to avoid crowding
    step = max(1, len(eval_layers) // 8)
    ax.set_xticks(tick_pos[::step])
    ax.set_xticklabels(tick_labels[::step])
    ax.set_yticks(tick_pos[::step])
    ax.set_yticklabels(tick_labels[::step])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)}: Cross-Layer CKA",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="Linear CKA")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_cka_cross_layer_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_cka_cross_layer_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_cka_cross_layer_{model_key}")


def plot_cka_cross_condition(cka_cross, model_key):
    """Plot CKA(h_qc, h_q) for sufficient vs insufficient."""
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = cka_cross["layers"]
    ax.plot(layers, cka_cross["cka_sufficient"], "o-", color="#4CAF50",
            label="Sufficient context", linewidth=2, markersize=4)
    ax.plot(layers, cka_cross["cka_insufficient"], "s-", color="#F44336",
            label="Insufficient context", linewidth=2, markersize=4)
    ax.fill_between(layers, cka_cross["cka_sufficient"], cka_cross["cka_insufficient"],
                    alpha=0.15, color="gray")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA(h(q+c), h(q))")
    ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)}: Context Impact on Representations",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_cka_cross_condition_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_cka_cross_condition_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_cka_cross_condition_{model_key}")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 3: Geometric Analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_dprime(X, labels, n_pca=50):
    """d-prime: centroid distance normalized by pooled std in PCA space."""
    from sklearn.decomposition import PCA
    n_components = min(n_pca, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    mask0 = labels == 0
    mask1 = labels == 1
    if mask0.sum() < 2 or mask1.sum() < 2:
        return 0.0

    mu0 = X_pca[mask0].mean(axis=0)
    mu1 = X_pca[mask1].mean(axis=0)
    var0 = X_pca[mask0].var(axis=0).mean()
    var1 = X_pca[mask1].var(axis=0).mean()
    pooled_std = np.sqrt((var0 + var1) / 2)

    if pooled_std < 1e-10:
        return 0.0
    return float(np.linalg.norm(mu1 - mu0) / pooled_std)


def effective_dimensionality(X):
    """Participation ratio: (sum(eig))^2 / sum(eig^2)."""
    X_centered = X - X.mean(axis=0)
    # Use a smaller number of singular values for memory efficiency
    n_components = min(100, X_centered.shape[0], X_centered.shape[1])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_centered)
    eigs = pca.explained_variance_
    eigs = eigs[eigs > 1e-10]
    if len(eigs) == 0:
        return 0.0
    return float((eigs.sum()) ** 2 / (eigs ** 2).sum())


def run_geometric_analysis(model_key, split="test", n_subsample=500, seed=42):
    """Geometric analysis: d-prime, effective dimensionality, PCA 2D."""
    print("\n--- Analysis 3: Geometric Analysis ---")
    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    labels_suf, labels_conf, _ = load_labels(model_key, split)
    rng = np.random.RandomState(seed)
    N = len(labels_suf)
    idx = rng.choice(N, min(n_subsample, N), replace=False)

    layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    layers = sorted(set(layers))

    if n_layers >= 28:
        pca_layers = [4, n_layers // 2, n_layers - 2]
    else:
        pca_layers = [2, n_layers // 2, n_layers - 2]

    results = {
        "layers": layers,
        "dprime_sufficiency": [],
        "dprime_confidence": [],
        "eff_dim_sufficient": [],
        "eff_dim_insufficient": [],
        "pca_2d": {},
    }

    sub_suf = labels_suf[idx]
    sub_conf = labels_conf[idx]

    for layer in layers:
        X = load_layer_slice(model_key, split, "h_with_context.npy", idx, layer)

        dp_suf = compute_dprime(X, sub_suf)
        dp_conf = compute_dprime(X, sub_conf)
        results["dprime_sufficiency"].append(dp_suf)
        results["dprime_confidence"].append(dp_conf)

        mask_suf = sub_suf == 1
        mask_insuf = sub_suf == 0
        ed_suf = effective_dimensionality(X[mask_suf])
        ed_insuf = effective_dimensionality(X[mask_insuf])
        results["eff_dim_sufficient"].append(ed_suf)
        results["eff_dim_insufficient"].append(ed_insuf)

        if layer in pca_layers:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            results["pca_2d"][str(layer)] = {
                "coords": coords.tolist(),
                "suf": sub_suf.tolist(),
                "conf": sub_conf.tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist(),
            }

        print(f"    L{layer:2d}  d'_suf={dp_suf:.3f}  d'_conf={dp_conf:.3f}  "
              f"dim_suf={ed_suf:.1f}  dim_insuf={ed_insuf:.1f}")
        del X
        gc.collect()

    del labels_suf, labels_conf
    gc.collect()
    return results


def plot_geometry(geo_results, model_key):
    """Plot geometric analysis: d-prime + PCA 2D."""
    layers = geo_results["layers"]
    pca_layers = sorted(geo_results["pca_2d"].keys(), key=int)

    n_pca = len(pca_layers)
    fig, axes = plt.subplots(2, max(n_pca, 2), figsize=(5 * max(n_pca, 2), 8))

    ax1 = axes[0, 0]
    ax1.plot(layers, geo_results["dprime_sufficiency"], "o-", color="#D32F2F",
             label="Sufficiency d'", linewidth=2, markersize=4)
    ax1.plot(layers, geo_results["dprime_confidence"], "s-", color="#1976D2",
             label="Confidence d'", linewidth=2, markersize=4)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("d-prime")
    ax1.set_title("Discriminability Across Layers")
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(layers, geo_results["eff_dim_sufficient"], "o-", color="#4CAF50",
             label="Sufficient", linewidth=2, markersize=4)
    ax2.plot(layers, geo_results["eff_dim_insufficient"], "s-", color="#F44336",
             label="Insufficient", linewidth=2, markersize=4)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effective Dimensionality")
    ax2.set_title("Manifold Dimensionality")
    ax2.legend()

    for extra in range(2, max(n_pca, 2)):
        if extra < axes.shape[1]:
            axes[0, extra].set_visible(False)

    colors_suf = {0: "#F44336", 1: "#4CAF50"}
    markers_conf = {0: "o", 1: "^"}

    for i, layer_str in enumerate(pca_layers):
        if i >= axes.shape[1]:
            break
        ax = axes[1, i]
        pca_data = geo_results["pca_2d"][layer_str]
        coords = np.array(pca_data["coords"])
        suf = np.array(pca_data["suf"])
        conf = np.array(pca_data["conf"])
        ev = pca_data["explained_variance"]

        for s in [0, 1]:
            for c in [0, 1]:
                mask = (suf == s) & (conf == c)
                label = f"{'Suf' if s else 'Insuf'}+{'Conf' if c else 'Unc'}"
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=colors_suf[s], marker=markers_conf[c],
                           alpha=0.3, s=10, label=label)

        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"Layer {layer_str}")
        if i == 0:
            ax.legend(fontsize=7, markerscale=2)

    for extra in range(len(pca_layers), axes.shape[1]):
        axes[1, extra].set_visible(False)

    fig.suptitle(f"{MODEL_LABELS.get(model_key, model_key)}: Representational Geometry",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_geometry_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_geometry_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_geometry_{model_key}")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 4: CCGP + Information Flow
# ──────────────────────────────────────────────────────────────────────────────

def probe_auroc(X_train, y_train, X_test, y_test):
    """Train logistic regression probe and return AUROC."""
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return 0.5
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    probe = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    probe.fit(X_tr, y_train)
    y_prob = probe.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def run_ccgp(model_key):
    """Cross-Condition Generalization Performance. Layer-by-layer loading."""
    print("\n--- Analysis 4: CCGP ---")
    suf_train, conf_train, _ = load_labels(model_key, "train")
    suf_test, conf_test, _ = load_labels(model_key, "test")

    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    layers = sorted(set(layers))

    conf_train_mask = conf_train == 1
    unc_train_mask = conf_train == 0
    conf_test_mask = conf_test == 1
    unc_test_mask = conf_test == 0
    suf_train_mask = suf_train == 1
    insuf_test_mask = suf_test == 0

    results = {
        "layers": layers,
        "ccgp_suf_trainconf_testunc": [],
        "ccgp_suf_trainunc_testconf": [],
        "ccgp_suf_mean": [],
        "baseline_suf_auroc": [],
        "ccgp_conf_trainsuf_testinsuf": [],
        "baseline_conf_auroc": [],
    }

    for layer in layers:
        X_tr = load_layer_all(model_key, "train", "h_with_context.npy", layer)
        X_te = load_layer_all(model_key, "test", "h_with_context.npy", layer)

        # Baseline sufficiency
        base_suf = probe_auroc(X_tr, suf_train, X_te, suf_test)
        results["baseline_suf_auroc"].append(base_suf)

        # CCGP: train on confident, test on uncertain
        try:
            ccgp1 = probe_auroc(
                X_tr[conf_train_mask], suf_train[conf_train_mask],
                X_te[unc_test_mask], suf_test[unc_test_mask])
        except Exception:
            ccgp1 = 0.5

        # CCGP: train on uncertain, test on confident
        try:
            ccgp2 = probe_auroc(
                X_tr[unc_train_mask], suf_train[unc_train_mask],
                X_te[conf_test_mask], suf_test[conf_test_mask])
        except Exception:
            ccgp2 = 0.5

        results["ccgp_suf_trainconf_testunc"].append(ccgp1)
        results["ccgp_suf_trainunc_testconf"].append(ccgp2)
        results["ccgp_suf_mean"].append((ccgp1 + ccgp2) / 2)

        # Confidence baseline + CCGP
        base_conf = probe_auroc(X_tr, conf_train, X_te, conf_test)
        results["baseline_conf_auroc"].append(base_conf)

        try:
            ccgp_conf = probe_auroc(
                X_tr[suf_train_mask], conf_train[suf_train_mask],
                X_te[insuf_test_mask], conf_test[insuf_test_mask])
        except Exception:
            ccgp_conf = 0.5
        results["ccgp_conf_trainsuf_testinsuf"].append(ccgp_conf)

        print(f"    L{layer:2d}  CCGP_suf={results['ccgp_suf_mean'][-1]:.4f}  "
              f"base_suf={base_suf:.4f}  base_conf={base_conf:.4f}")

        del X_tr, X_te
        gc.collect()

    del suf_train, conf_train, suf_test, conf_test
    gc.collect()
    return results


def run_information_flow(model_key):
    """Track sufficiency and confidence information across layers.

    Memory-safe: loads one representation type at a time per layer.
    """
    print("\n--- Analysis 5: Information Flow ---")
    suf_train, conf_train, _ = load_labels(model_key, "train")
    suf_test, conf_test, _ = load_labels(model_key, "test")

    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    layers = sorted(set(layers))

    results = {
        "layers": layers,
        "suf_from_hqc": [],
        "conf_from_hqc": [],
        "suf_from_hq": [],
        "conf_from_hq": [],
        "suf_from_diff": [],
    }

    for layer in layers:
        # h(q+c) probes
        X_qc_tr = load_layer_all(model_key, "train", "h_with_context.npy", layer)
        X_qc_te = load_layer_all(model_key, "test", "h_with_context.npy", layer)
        results["suf_from_hqc"].append(probe_auroc(X_qc_tr, suf_train, X_qc_te, suf_test))
        results["conf_from_hqc"].append(probe_auroc(X_qc_tr, conf_train, X_qc_te, conf_test))
        del X_qc_tr, X_qc_te
        gc.collect()

        # h(q) probes
        X_q_tr = load_layer_all(model_key, "train", "h_question_only.npy", layer)
        X_q_te = load_layer_all(model_key, "test", "h_question_only.npy", layer)
        results["suf_from_hq"].append(probe_auroc(X_q_tr, suf_train, X_q_te, suf_test))
        results["conf_from_hq"].append(probe_auroc(X_q_tr, conf_train, X_q_te, conf_test))

        # diff probes — need both h(q+c) and h(q) simultaneously, but only for train/test separately
        X_qc_tr = load_layer_all(model_key, "train", "h_with_context.npy", layer)
        X_diff_tr = X_qc_tr - X_q_tr
        del X_qc_tr, X_q_tr
        gc.collect()

        X_qc_te = load_layer_all(model_key, "test", "h_with_context.npy", layer)
        X_diff_te = X_qc_te - X_q_te
        del X_qc_te, X_q_te
        gc.collect()

        results["suf_from_diff"].append(probe_auroc(X_diff_tr, suf_train, X_diff_te, suf_test))
        del X_diff_tr, X_diff_te
        gc.collect()

        print(f"    L{layer:2d}  suf_hqc={results['suf_from_hqc'][-1]:.3f}  "
              f"conf_hqc={results['conf_from_hqc'][-1]:.3f}  "
              f"suf_hq={results['suf_from_hq'][-1]:.3f}  "
              f"conf_hq={results['conf_from_hq'][-1]:.3f}")

    del suf_train, conf_train, suf_test, conf_test
    gc.collect()
    return results


def plot_ccgp(ccgp_results, model_key):
    """Plot CCGP results."""
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = ccgp_results["layers"]

    ax.plot(layers, ccgp_results["baseline_suf_auroc"], "o-", color="#D32F2F",
            label="Sufficiency (all data)", linewidth=2, markersize=4)
    ax.plot(layers, ccgp_results["ccgp_suf_mean"], "s--", color="#D32F2F",
            label="CCGP: Sufficiency (cross-confidence)", linewidth=2, markersize=4, alpha=0.8)
    ax.plot(layers, ccgp_results["baseline_conf_auroc"], "o-", color="#1976D2",
            label="Confidence (all data)", linewidth=2, markersize=4)

    ax.fill_between(layers, ccgp_results["ccgp_suf_trainconf_testunc"],
                    ccgp_results["ccgp_suf_trainunc_testconf"],
                    alpha=0.15, color="#D32F2F")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)}: Cross-Condition Generalization",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_ccgp_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_ccgp_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_ccgp_{model_key}")


def plot_information_flow(info_results, model_key):
    """Plot information flow across layers."""
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = info_results["layers"]

    ax.plot(layers, info_results["suf_from_hqc"], "o-", color="#D32F2F",
            label="Sufficiency from h(q+c)", linewidth=2, markersize=4)
    ax.plot(layers, info_results["suf_from_diff"], "^-", color="#FF9800",
            label="Sufficiency from h(q+c)-h(q)", linewidth=2, markersize=4)
    ax.plot(layers, info_results["suf_from_hq"], "v-", color="#9E9E9E",
            label="Sufficiency from h(q)", linewidth=1.5, markersize=3, alpha=0.7)
    ax.plot(layers, info_results["conf_from_hqc"], "s-", color="#1976D2",
            label="Confidence from h(q+c)", linewidth=2, markersize=4)
    ax.plot(layers, info_results["conf_from_hq"], "D-", color="#90CAF9",
            label="Confidence from h(q)", linewidth=1.5, markersize=3, alpha=0.7)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe AUROC")
    ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)}: Information Flow Across Layers",
                 fontweight="bold")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_info_flow_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_info_flow_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_info_flow_{model_key}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def numpy_to_python(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    return obj


def run_all_analyses(model_key):
    """Run all neuroscience analyses for one model.

    Each analysis loads and frees its own data independently.
    No large arrays are kept across analyses.
    """
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg["n_layers"]

    print(f"\n{'='*70}")
    print(f"NEUROSCIENCE ANALYSIS — {MODEL_LABELS.get(model_key, model_key)}")
    print(f"  {n_layers} layers, {cfg['hidden_dim']} dim")
    print(f"{'='*70}")

    # Quick data check
    suf_train, conf_train, _ = load_labels(model_key, "train")
    suf_test, conf_test, _ = load_labels(model_key, "test")
    print(f"  Train: {len(suf_train)} examples, Test: {len(suf_test)} examples")
    print(f"  Train suf: {suf_train.sum()}/{len(suf_train)}, "
          f"conf: {conf_train.sum()}/{len(conf_train)}")
    del suf_train, conf_train, suf_test, conf_test

    results = {"model": model_key, "n_layers": n_layers}

    # 1. RSA — each analysis manages its own memory
    results["rsa"] = run_rsa(model_key, split="test", n_per_quadrant=75)
    plot_rsa(results["rsa"], model_key)
    gc.collect()

    # 2. CKA
    print("\n--- Analysis 2: CKA ---")
    cka_matrix, eval_layers = run_cka_cross_layer(model_key, split="test", n_subsample=400)
    results["cka_cross_layer"] = cka_matrix.tolist()
    results["cka_eval_layers"] = eval_layers
    plot_cka_heatmap(cka_matrix, eval_layers, model_key)
    del cka_matrix
    gc.collect()

    results["cka_cross_condition"] = run_cka_cross_condition(
        model_key, split="test", n_subsample=300)
    plot_cka_cross_condition(results["cka_cross_condition"], model_key)
    gc.collect()

    # 3. Geometric Analysis
    results["geometry"] = run_geometric_analysis(model_key, split="test", n_subsample=500)
    plot_geometry(results["geometry"], model_key)
    gc.collect()

    # 4. CCGP
    results["ccgp"] = run_ccgp(model_key)
    plot_ccgp(results["ccgp"], model_key)
    gc.collect()

    # 5. Information Flow
    results["info_flow"] = run_information_flow(model_key)
    plot_information_flow(results["info_flow"], model_key)
    gc.collect()

    # Save results (strip PCA coords to keep JSON small)
    save_results = numpy_to_python(results)
    if "geometry" in save_results and "pca_2d" in save_results["geometry"]:
        for layer_key in save_results["geometry"]["pca_2d"]:
            save_results["geometry"]["pca_2d"][layer_key].pop("coords", None)

    out_path = RESULTS_DIR / model_key / "neuro_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # Print key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    rsa = results["rsa"]
    best_rsa_idx = np.argmax(rsa["sufficiency_rho"])
    best_rsa_layer = rsa["layers"][best_rsa_idx]
    print(f"  RSA: Best sufficiency rho = {rsa['sufficiency_rho'][best_rsa_idx]:.4f} "
          f"at layer {best_rsa_layer} (p={rsa['sufficiency_p'][best_rsa_idx]:.4f})")
    print(f"       Best confidence rho = {max(rsa['confidence_rho']):.4f}")

    ccgp = results["ccgp"]
    best_ccgp_idx = np.argmax(ccgp["ccgp_suf_mean"])
    best_ccgp_layer = ccgp["layers"][best_ccgp_idx]
    print(f"  CCGP: Best = {ccgp['ccgp_suf_mean'][best_ccgp_idx]:.4f} "
          f"at layer {best_ccgp_layer} "
          f"(baseline={ccgp['baseline_suf_auroc'][best_ccgp_idx]:.4f})")

    geo = results["geometry"]
    best_dp_idx = np.argmax(geo["dprime_sufficiency"])
    print(f"  d': Best sufficiency = {geo['dprime_sufficiency'][best_dp_idx]:.3f}, "
          f"Best confidence = {max(geo['dprime_confidence']):.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    args = parser.parse_args()

    if args.model == "all":
        for key in MODEL_CONFIGS:
            try:
                run_all_analyses(key)
            except Exception as e:
                print(f"Error on {key}: {e}")
                import traceback
                traceback.print_exc()
    else:
        run_all_analyses(args.model)
