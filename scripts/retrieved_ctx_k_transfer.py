#!/usr/bin/env python
"""k-sensitivity of transferred probes on BM25 top-k retrieved contexts (test split only, no retraining).
Reuses retrieved_context_eval.py functions; reproduces cmd_analyse's label='support' transfer rows."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import src.evaluation.retrieved_context_eval as R  # noqa: E402
from src.analysis.store_baseline_scores import load_best_layers  # noqa: E402

ROWS = ["Standard probe (transfer)", "CSP probe (transfer)", "Verbalized confidence",
        "Model answers (no abstention)", "e5 cosine(q, ctx)"]


def one_tag(tag, model, n_boot, e5_device, label="support"):
    sc, recs, hqc, hq, _ = R._load_split(tag, model, "test")
    deco_layer, std_layer = load_best_layers(model)
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    F = {"std": f32(hqc[:, std_layer, :]), "csp": f32(hqc[:, deco_layer, :]) - f32(hq[:, deco_layer, :])}
    bdir = R.HIDDEN_STATES_DIR / model / "train"
    b_hqc = np.load(bdir / "h_with_context.npy", mmap_mode="r")
    b_hq = np.load(bdir / "h_question_only.npy", mmap_mode="r")
    b_y = np.load(bdir / "labels.npy")
    S = {}
    for key, nm in (("std", "Standard probe"), ("csp", "CSP probe")):
        bf = f32(b_hqc[:, std_layer, :]) if key == "std" else f32(b_hqc[:, deco_layer, :]) - f32(b_hq[:, deco_layer, :])
        S[f"{nm} (transfer)"] = R._fit_lr(bf, b_y).predict_proba(F[key])[:, 1]
    S["Verbalized confidence"] = np.array([r["vc"] for r in sc], float)
    S["Model answers (no abstention)"] = 1.0 - np.array([r["abstained"] for r in sc], float)
    try:
        e5dir = R.RC_DIR / tag / "e5_cache"; e5dir.mkdir(exist_ok=True)
        S["e5 cosine(q, ctx)"] = R._e5_scores(recs, e5dir / "test.json", e5_device)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] e5 failed for {tag}: {e}")
    m, y = R._label_arrays(recs, label)
    keep = np.array([not r.get("truncated", 0) for r in sc])
    y = y[keep[m]]; m = m & keep
    S = {k: v[m] for k, v in S.items()}
    n = len(y)
    B = np.random.default_rng(0).integers(0, n, size=(n_boot, n))  # same resampling as _metrics_block
    out = {"n": int(n), "pos_rate": float(y.mean()), "methods": {}}
    for k in ROWS:
        if k not in S: continue
        s = S[k]
        bt = np.array([R.auroc(y[b], s[b]) for b in B])
        lo, hi = np.nanpercentile(bt, [2.5, 97.5])
        out["methods"][k] = {"auroc": R.auroc(y, s), "ci95": [float(lo), float(hi)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--tags", nargs="+", default=["bm25_k3", "bm25_k5", "bm25_k10"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--e5-device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "retrieved_ctx_k_sensitivity"))
    a = ap.parse_args()
    res = {}
    for tag in a.tags:
        hs, _ = R._paths(tag, a.model, "test")
        try:
            assert (hs / "h_with_context.npy").exists() and (hs / "h_question_only.npy").exists()
            res[tag] = one_tag(tag, a.model, a.n_boot, a.e5_device)
            print(f"[ok] {tag} n={res[tag]['n']}")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {tag}: states missing/unusable ({type(e).__name__}: {e})")
    f = lambda v: f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
    tags = list(res)
    L = [f"# Retrieved-context k-sensitivity ({a.model}, BM25 top-k, test split, label = support)", "",
         "Probes are transferred from the SufficiencyBench-trained probes (no retraining on retrieved data); "
         f"AUROC with 95% bootstrap CI (B={a.n_boot}, resampling questions; truncated prompts dropped).", ""]
    for t in tags:
        L.append(f"- {t}: n = {res[t]['n']}, positive rate = {res[t]['pos_rate']:.3f}")
    L += ["", "| Predictor | " + " | ".join(t.replace("bm25_", "") for t in tags) + " |",
          "|---|" + "---|" * len(tags)]
    for k in ROWS:
        L.append(f"| {k} | " + " | ".join(f(res[t]["methods"][k]) if k in res[t]["methods"] else "--" for t in tags) + " |")
    Path(a.out + ".json").write_text(json.dumps({"model": a.model, "n_boot": a.n_boot, "tags": res}, indent=2))
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
