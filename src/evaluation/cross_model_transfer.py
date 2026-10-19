"""
cross_model_transfer.py — Cross-Model Probe Transfer Experiments

Tests whether the sufficiency direction is universal across model families
by training probes on one model and evaluating on another (zero-shot transfer).

Key experiments:
1. Direct transfer (Llama ↔ Mistral, same 4096 dim)
2. PCA-aligned transfer (involving Qwen, 3584 dim)
3. Procrustes alignment transfer
4. CKA representation similarity across models
5. Probe direction cosine similarity
6. Layer-wise transfer heatmaps

Usage:
  python src/evaluation/cross_model_transfer.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.linalg import orthogonal_procrustes
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import HIDDEN_STATES_DIR, RESULTS_DIR, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(model_key: str, split: str, mmap=True):
    """Load hidden states, labels, metadata for a model/split."""
    d = HIDDEN_STATES_DIR / model_key / split
    mode = "r" if mmap else None
    h_qc = np.load(d / "h_with_context.npy", mmap_mode=mode)
    h_q = np.load(d / "h_question_only.npy", mmap_mode=mode)
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    return h_qc, h_q, labels, metadata


def get_best_layer(model_key: str, method="DECO"):
    """Get best layer from existing DECO results."""
    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        res = json.load(f)
    return res["best_layers"][method]


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def train_probe(X_train, y_train):
    """Train logistic regression with scaler. Returns probe, scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    probe.fit(X_scaled, y_train)
    return probe, scaler


def evaluate_probe(probe, scaler, X_test, y_test, metadata=None):
    """Evaluate probe, return AUROC + per-quadrant breakdown."""
    X_scaled = scaler.transform(X_test)
    y_prob = probe.predict_proba(X_scaled)[:, 1]
    y_pred = probe.predict(X_scaled)

    results = {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    if metadata is not None:
        for quad in ["Q1", "Q2", "Q3", "Q4"]:
            idx = [i for i, m in enumerate(metadata) if m["quadrant"] == quad]
            if len(idx) >= 5 and len(set(y_test[idx])) > 1:
                results[f"{quad}_auroc"] = float(roc_auc_score(y_test[idx], y_prob[idx]))
                results[f"{quad}_acc"] = float(accuracy_score(y_test[idx], y_pred[idx]))

    return results


def get_features(h_qc, h_q, layer, method="deco"):
    """Extract features at a layer. method: 'deco' or 'standard'. Returns a copy (not mmap view)."""
    if method == "deco":
        return np.array(h_qc[:, layer, :], dtype=np.float32) - np.array(h_q[:, layer, :], dtype=np.float32)
    else:
        return np.array(h_qc[:, layer, :], dtype=np.float32)


# ---------------------------------------------------------------------------
# Experiment 1: Direct Transfer (same dimension)
# ---------------------------------------------------------------------------

def _load_layer_features(model_key, split, layer, method="deco"):
    """Load features for a single layer using mmap. Memory-safe."""
    import gc
    d = HIDDEN_STATES_DIR / model_key / split
    h_qc = np.load(d / "h_with_context.npy", mmap_mode="r")
    h_q = np.load(d / "h_question_only.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy")
    with open(d / "metadata.json") as f:
        meta = json.load(f)

    if method == "deco":
        X = np.empty((h_qc.shape[0], h_qc.shape[2]), dtype=np.float32)
        # Process in chunks to avoid memory spikes
        chunk = 500
        for i in range(0, h_qc.shape[0], chunk):
            end = min(i + chunk, h_qc.shape[0])
            X[i:end] = h_qc[i:end, layer, :] - h_q[i:end, layer, :]
    else:
        X = np.empty((h_qc.shape[0], h_qc.shape[2]), dtype=np.float32)
        chunk = 500
        for i in range(0, h_qc.shape[0], chunk):
            end = min(i + chunk, h_qc.shape[0])
            X[i:end] = h_qc[i:end, layer, :]

    del h_qc, h_q
    gc.collect()
    return X, labels, meta


def run_direct_transfer():
    """Transfer probes between Llama ↔ Mistral (both 4096 dim)."""
    import gc
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DIRECT TRANSFER (same dimension)")
    print("=" * 70)

    models = ["llama", "mistral", "qwen"]
    results = {}

    best_layers = {}
    for mk in models:
        best_layers[mk] = {
            "deco": get_best_layer(mk, "DECO"),
            "standard": get_best_layer(mk, "standard"),
        }

    for method in ["deco", "standard"]:
        print(f"\n--- Feature type: {method} ---")
        print(f"{'Source→Target':<25s} {'SrcLayer':>8s} {'TgtLayer':>8s} "
              f"{'AUROC':>8s} {'Acc':>8s} {'Within':>8s}")
        print("-" * 70)

        for src in models:
            src_layer = best_layers[src][method]

            # Train on source
            X_train, y_tr, _ = _load_layer_features(src, "train", src_layer, method)
            probe, scaler = train_probe(X_train, y_tr)
            del X_train; gc.collect()

            # Within-model baseline
            X_test_within, y_te_w, meta_w = _load_layer_features(src, "test", src_layer, method)
            res_within = evaluate_probe(probe, scaler, X_test_within, y_te_w, meta_w)
            del X_test_within; gc.collect()

            for tgt in models:
                if tgt == src:
                    continue

                src_dim = MODEL_CONFIGS[src]["hidden_dim"]
                tgt_dim = MODEL_CONFIGS[tgt]["hidden_dim"]

                if src_dim != tgt_dim:
                    print(f"  {src}→{tgt:<10s} {'(dim mismatch, skip direct)':>50s}")
                    continue

                tgt_n_layers = MODEL_CONFIGS[tgt]["n_layers"]
                tgt_layer = min(best_layers[tgt][method], tgt_n_layers - 1)

                X_test_tgt, y_te_t, meta_te_t = _load_layer_features(tgt, "test", tgt_layer, method)
                res = evaluate_probe(probe, scaler, X_test_tgt, y_te_t, meta_te_t)
                del X_test_tgt; gc.collect()

                key = f"{src}→{tgt}_{method}"
                results[key] = {
                    "source": src, "target": tgt, "method": method,
                    "src_layer": src_layer, "tgt_layer": tgt_layer,
                    "transfer": res, "within_model": res_within,
                }

                print(f"  {src}→{tgt:<10s} {src_layer:>8d} {tgt_layer:>8d} "
                      f"{res['auroc']:>8.4f} {res['accuracy']:>8.4f} "
                      f"{res_within['auroc']:>8.4f}")

                # Layer sweep on target
                best_tgt_auroc = 0
                best_tgt_layer = 0
                layers_to_try = list(range(0, tgt_n_layers, 2)) + [tgt_n_layers - 1]
                layer_aurocs = {}
                for tl in layers_to_try:
                    X_tl, y_tl, _ = _load_layer_features(tgt, "test", tl, method)
                    try:
                        res_tl = evaluate_probe(probe, scaler, X_tl, y_tl)
                        layer_aurocs[tl] = res_tl["auroc"]
                        if res_tl["auroc"] > best_tgt_auroc:
                            best_tgt_auroc = res_tl["auroc"]
                            best_tgt_layer = tl
                    except Exception:
                        layer_aurocs[tl] = 0.5
                    del X_tl; gc.collect()

                results[key]["best_tgt_layer"] = best_tgt_layer
                results[key]["best_tgt_auroc"] = best_tgt_auroc
                results[key]["layer_sweep"] = layer_aurocs

                print(f"  {'':25s} {'best@L' + str(best_tgt_layer):>17s} "
                      f"{best_tgt_auroc:>8.4f}")

            del probe, scaler; gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 2: PCA-Aligned Transfer (cross-dimension)
# ---------------------------------------------------------------------------

def run_pca_transfer():
    """Transfer across different dimensions using PCA alignment."""
    import gc
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PCA-ALIGNED TRANSFER (cross-dimension)")
    print("=" * 70)

    models = ["llama", "mistral", "qwen"]
    shared_dims = [256, 512, 1024]
    results = {}

    best_layers = {mk: get_best_layer(mk, "DECO") for mk in models}

    for shared_d in shared_dims:
        print(f"\n--- Shared PCA dim: {shared_d} ---")
        print(f"{'Source→Target':<25s} {'AUROC':>8s} {'Acc':>8s}")
        print("-" * 45)

        for src in models:
            src_layer = best_layers[src]
            X_train_src, y_tr, _ = _load_layer_features(src, "train", src_layer, "deco")

            pca_src = PCA(n_components=shared_d, random_state=42)
            X_train_pca = pca_src.fit_transform(X_train_src)
            del X_train_src; gc.collect()

            probe, scaler = train_probe(X_train_pca, y_tr)
            del X_train_pca; gc.collect()

            for tgt in models:
                if tgt == src:
                    continue

                tgt_layer = best_layers[tgt]

                # Fit PCA on target train
                X_train_tgt, _, _ = _load_layer_features(tgt, "train", tgt_layer, "deco")
                pca_tgt = PCA(n_components=shared_d, random_state=42)
                pca_tgt.fit(X_train_tgt)
                del X_train_tgt; gc.collect()

                # Transform target test
                X_test_tgt, y_te, meta_te = _load_layer_features(tgt, "test", tgt_layer, "deco")
                X_test_pca = pca_tgt.transform(X_test_tgt)
                del X_test_tgt; gc.collect()

                res = evaluate_probe(probe, scaler, X_test_pca, y_te, meta_te)
                del X_test_pca; gc.collect()

                key = f"{src}→{tgt}_pca{shared_d}"
                results[key] = {
                    "source": src, "target": tgt, "shared_dim": shared_d,
                    "transfer": res,
                }
                print(f"  {src}→{tgt:<10s} {res['auroc']:>8.4f} {res['accuracy']:>8.4f}")

            del probe, scaler; gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Procrustes-Aligned Transfer
# ---------------------------------------------------------------------------

def run_procrustes_transfer():
    """Transfer using Procrustes alignment on val split as anchor."""
    import gc
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: PROCRUSTES-ALIGNED TRANSFER")
    print("=" * 70)

    models = ["llama", "mistral", "qwen"]
    results = {}
    best_layers = {mk: get_best_layer(mk, "DECO") for mk in models}

    for src in models:
        src_layer = best_layers[src]
        src_dim = MODEL_CONFIGS[src]["hidden_dim"]

        X_train_src, y_tr, _ = _load_layer_features(src, "train", src_layer, "deco")
        X_val_src, _, _ = _load_layer_features(src, "val", src_layer, "deco")

        for tgt in models:
            if tgt == src:
                continue

            tgt_layer = best_layers[tgt]
            tgt_dim = MODEL_CONFIGS[tgt]["hidden_dim"]
            shared_d = min(src_dim, tgt_dim, 1024)

            # PCA on source
            pca_src = PCA(n_components=shared_d, random_state=42)
            pca_src.fit(X_train_src)
            X_val_src_pca = pca_src.transform(X_val_src)
            X_train_src_pca = pca_src.transform(X_train_src)

            # PCA on target (fit on target train)
            X_train_tgt, _, _ = _load_layer_features(tgt, "train", tgt_layer, "deco")
            pca_tgt = PCA(n_components=shared_d, random_state=42)
            pca_tgt.fit(X_train_tgt)
            del X_train_tgt; gc.collect()

            X_val_tgt, _, _ = _load_layer_features(tgt, "val", tgt_layer, "deco")
            X_val_tgt_pca = pca_tgt.transform(X_val_tgt)
            del X_val_tgt; gc.collect()

            # Procrustes alignment
            mu_src = X_val_src_pca.mean(axis=0)
            mu_tgt = X_val_tgt_pca.mean(axis=0)
            A = X_val_tgt_pca - mu_tgt
            B = X_val_src_pca - mu_src
            R, scale = orthogonal_procrustes(A, B)

            # Test
            X_test_tgt, y_te, meta_te = _load_layer_features(tgt, "test", tgt_layer, "deco")
            X_test_tgt_pca = pca_tgt.transform(X_test_tgt)
            del X_test_tgt; gc.collect()

            X_test_aligned = (X_test_tgt_pca - mu_tgt) @ R + mu_src
            del X_test_tgt_pca; gc.collect()

            # Train probe in source PCA space
            probe_pca, scaler_pca = train_probe(X_train_src_pca, y_tr)
            res = evaluate_probe(probe_pca, scaler_pca, X_test_aligned, y_te, meta_te)
            del X_test_aligned; gc.collect()

            key = f"{src}→{tgt}_procrustes"
            results[key] = {
                "source": src, "target": tgt, "shared_dim": shared_d,
                "procrustes_scale": float(scale),
                "transfer": res,
            }
            print(f"  {src}→{tgt}: AUROC={res['auroc']:.4f}  Acc={res['accuracy']:.4f}  "
                  f"(shared_d={shared_d}, scale={scale:.4f})")

        del X_train_src, X_val_src; gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Probe Direction Similarity
# ---------------------------------------------------------------------------

def run_direction_similarity():
    """Compare probe weight vectors across models."""
    import gc
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: PROBE DIRECTION SIMILARITY")
    print("=" * 70)

    models = ["llama", "mistral", "qwen"]
    results = {}

    directions = {}
    for mk in models:
        best_layer = get_best_layer(mk, "DECO")
        X_train, y_tr, _ = _load_layer_features(mk, "train", best_layer, "deco")
        probe, scaler = train_probe(X_train, y_tr)

        w_scaled = probe.coef_[0]
        direction = w_scaled / scaler.scale_
        direction = direction / np.linalg.norm(direction)
        directions[mk] = {"direction": direction, "dim": len(direction), "layer": best_layer}
        del X_train, probe, scaler; gc.collect()

    print("\n  Probe direction cosine similarity:")
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            d1, d2 = directions[m1]["direction"], directions[m2]["direction"]
            if len(d1) == len(d2):
                cos = float(np.dot(d1, d2))
                print(f"  cos({m1}, {m2}) = {cos:.4f}  (direct, dim={len(d1)})")
                results[f"cos_{m1}_{m2}_direct"] = cos
            else:
                print(f"  {m1} ({len(d1)}d) vs {m2} ({len(d2)}d) — dim mismatch")

    print("\n  After PCA to shared dim 512:")
    shared_d = 512
    pca_directions = {}

    for mk in models:
        best_layer = directions[mk]["layer"]
        X_train, y_tr, _ = _load_layer_features(mk, "train", best_layer, "deco")
        pca = PCA(n_components=shared_d, random_state=42)
        X_pca = pca.fit_transform(X_train)
        del X_train; gc.collect()

        probe_pca, scaler_pca = train_probe(X_pca, y_tr)
        w_pca = probe_pca.coef_[0] / scaler_pca.scale_
        w_pca = w_pca / np.linalg.norm(w_pca)
        pca_directions[mk] = w_pca
        del X_pca, probe_pca, scaler_pca; gc.collect()

    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            cos = float(np.dot(pca_directions[m1], pca_directions[m2]))
            print(f"  cos({m1}, {m2}) = {cos:.4f}  (PCA-{shared_d})")
            results[f"cos_{m1}_{m2}_pca{shared_d}"] = cos

    return results


# ---------------------------------------------------------------------------
# Experiment 5: CKA Representation Similarity
# ---------------------------------------------------------------------------

def linear_cka(X, Y):
    """Centered Kernel Alignment (linear kernel)."""
    n = X.shape[0]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.sum((X @ X.T) * (Y @ Y.T)) / (n - 1) ** 2
    hsic_xx = np.sum((X @ X.T) ** 2) / (n - 1) ** 2
    hsic_yy = np.sum((Y @ Y.T) ** 2) / (n - 1) ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def run_cka_analysis():
    """CKA between corresponding layers of different models."""
    import gc
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: CKA REPRESENTATION SIMILARITY")
    print("=" * 70)

    models = ["llama", "mistral", "qwen"]
    results = {}
    n_subsample = 500
    rng = np.random.default_rng(42)
    idx = rng.choice(2224, n_subsample, replace=False)

    # CKA between DECO features at best layers
    print("\n  CKA between DECO features at best layers (n=500):")
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            l1 = get_best_layer(m1, "DECO")
            l2 = get_best_layer(m2, "DECO")

            X1_full, _, _ = _load_layer_features(m1, "test", l1, "deco")
            X1 = X1_full[idx]; del X1_full; gc.collect()
            X2_full, _, _ = _load_layer_features(m2, "test", l2, "deco")
            X2 = X2_full[idx]; del X2_full; gc.collect()

            cka = linear_cka(X1, X2)
            print(f"  CKA({m1}@L{l1}, {m2}@L{l2}) = {cka:.4f}")
            results[f"cka_{m1}_{m2}_best"] = cka
            del X1, X2; gc.collect()

    # CKA layer sweep for llama vs mistral
    print("\n  CKA layer sweep (Llama vs Mistral, every 4 layers):")
    layers_lm = list(range(0, 32, 4)) + [31]
    cka_matrix = np.zeros((len(layers_lm), len(layers_lm)))

    # Pre-load subsample for each layer (one model at a time)
    llama_layers = {}
    for li, l in enumerate(layers_lm):
        X_full, _, _ = _load_layer_features("llama", "test", l, "deco")
        llama_layers[l] = X_full[idx].copy()
        del X_full; gc.collect()

    for j, l2 in enumerate(layers_lm):
        X2_full, _, _ = _load_layer_features("mistral", "test", l2, "deco")
        X2 = X2_full[idx]
        del X2_full; gc.collect()

        for i, l1 in enumerate(layers_lm):
            cka_matrix[i, j] = linear_cka(llama_layers[l1], X2)
        del X2; gc.collect()

    del llama_layers; gc.collect()

    results["cka_llama_mistral_matrix"] = {
        "layers": layers_lm, "matrix": cka_matrix.tolist(),
    }

    print(f"  {'Layer':>6s} {'CKA':>8s}")
    for i, l in enumerate(layers_lm):
        print(f"  L{l:>4d}  {cka_matrix[i, i]:>8.4f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Layer-wise Transfer Heatmap
# ---------------------------------------------------------------------------

def run_layer_transfer_heatmap():
    """For Llama→Mistral and Mistral→Llama: AUROC at every (src_layer, tgt_layer)."""
    import gc
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: LAYER-WISE TRANSFER HEATMAPS")
    print("=" * 70)

    results = {}

    for src, tgt in [("llama", "mistral"), ("mistral", "llama")]:
        src_dim = MODEL_CONFIGS[src]["hidden_dim"]
        tgt_dim = MODEL_CONFIGS[tgt]["hidden_dim"]
        if src_dim != tgt_dim:
            continue

        print(f"\n  {src} → {tgt}:")

        src_layers = list(range(0, MODEL_CONFIGS[src]["n_layers"], 4)) + [MODEL_CONFIGS[src]["n_layers"] - 1]
        tgt_layers = list(range(0, MODEL_CONFIGS[tgt]["n_layers"], 4)) + [MODEL_CONFIGS[tgt]["n_layers"] - 1]

        heatmap = np.zeros((len(src_layers), len(tgt_layers)))

        for i, sl in enumerate(src_layers):
            X_train, y_tr, _ = _load_layer_features(src, "train", sl, "deco")
            probe, scaler = train_probe(X_train, y_tr)
            del X_train; gc.collect()

            for j, tl in enumerate(tgt_layers):
                X_test, y_te, _ = _load_layer_features(tgt, "test", tl, "deco")
                try:
                    res = evaluate_probe(probe, scaler, X_test, y_te)
                    heatmap[i, j] = res["auroc"]
                except Exception:
                    heatmap[i, j] = 0.5
                del X_test; gc.collect()

            best_tl_idx = np.argmax(heatmap[i, :])
            print(f"    src=L{sl:2d}: best_tgt=L{tgt_layers[best_tl_idx]:2d} "
                  f"AUROC={heatmap[i, best_tl_idx]:.4f}")
            del probe, scaler; gc.collect()

        results[f"{src}→{tgt}"] = {
            "src_layers": src_layers, "tgt_layers": tgt_layers,
            "heatmap": heatmap.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Bootstrap significance for transfer
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main: run all experiments
# ---------------------------------------------------------------------------

def run_all():
    """Run all cross-model transfer experiments."""
    all_results = {}

    # Experiment 1: Direct transfer
    all_results["direct_transfer"] = run_direct_transfer()

    # Experiment 2: PCA transfer
    all_results["pca_transfer"] = run_pca_transfer()

    # Experiment 3: Procrustes transfer
    all_results["procrustes_transfer"] = run_procrustes_transfer()

    # Experiment 4: Direction similarity
    all_results["direction_similarity"] = run_direction_similarity()

    # Experiment 5: CKA
    all_results["cka"] = run_cka_analysis()

    # Experiment 6: Layer heatmaps
    all_results["layer_heatmaps"] = run_layer_transfer_heatmap()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: CROSS-MODEL TRANSFER AUROC")
    print("=" * 70)

    print(f"\n  {'Pair':<20s} {'Direct':>8s} {'PCA-256':>8s} {'PCA-512':>8s} "
          f"{'PCA-1024':>9s} {'Procrust':>9s}")
    print(f"  {'-'*65}")

    models = ["llama", "mistral", "qwen"]
    for src in models:
        for tgt in models:
            if src == tgt:
                continue
            row = f"  {src}→{tgt:<10s}"

            # Direct
            key = f"{src}→{tgt}_deco"
            if key in all_results["direct_transfer"]:
                row += f" {all_results['direct_transfer'][key]['transfer']['auroc']:>8.4f}"
            else:
                row += f" {'—':>8s}"

            # PCA
            for d in [256, 512, 1024]:
                key = f"{src}→{tgt}_pca{d}"
                if key in all_results["pca_transfer"]:
                    row += f" {all_results['pca_transfer'][key]['transfer']['auroc']:>8.4f}"
                else:
                    row += f" {'—':>8s}"

            # Procrustes
            key = f"{src}→{tgt}_procrustes"
            if key in all_results["procrustes_transfer"]:
                row += f" {all_results['procrustes_transfer'][key]['transfer']['auroc']:>9.4f}"
            else:
                row += f" {'—':>9s}"

            print(row)

    # Save
    out_dir = RESULTS_DIR / "cross_model_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_results.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nAll results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_all()
