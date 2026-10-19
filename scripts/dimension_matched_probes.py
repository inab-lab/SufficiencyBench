"""Dimension-matched control: probe AUROC vs input dimensionality (PCA / random projection).
Pipeline: StandardScaler(fit train) -> [PCA or Gaussian RP] -> StandardScaler -> LogisticRegression(max_iter=1000, random_state=42)
'full' = exactly deco.make_classifier('logreg'). Bootstrap CI clustered by question id (B=1000, seed 0)."""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
HS = ROOT / "data/experiments/hidden_states"
PCA_DIMS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
RP_DIMS = [4, 32, 1024]
PAPER = {"standard": {"llama": .907, "mistral": .910, "qwen": .904},
         "csp": {"llama": .904, "mistral": .902, "qwen": .914}}

def load(model, split, layer_std, layer_csp):
    d = HS / model / split
    hq = np.load(d / "h_with_context.npy", mmap_mode="r")
    h0 = np.load(d / "h_question_only.npy", mmap_mode="r")
    y = np.load(d / "labels.npy")
    meta = json.load(open(d / "metadata.json"))
    qid = np.array([m["id"].rsplit("_", 1)[0] for m in meta])
    std = np.array(hq[:, layer_std, :], dtype=np.float32)
    csp = np.array(hq[:, layer_csp, :], dtype=np.float32) - np.array(h0[:, layer_csp, :], dtype=np.float32)
    return std, csp, y, qid

def boot_ci(y, p, qid, B=1000, seed=0):
    rng = np.random.RandomState(seed)
    uq, inv = np.unique(qid, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uq))]
    vals = []
    for _ in range(B):
        s = rng.randint(0, len(uq), len(uq))
        ii = np.concatenate([idx_by[k] for k in s])
        if len(set(y[ii])) < 2: continue
        vals.append(roc_auc_score(y[ii], p[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def run(Xtr, ytr, Xte, yte, qte, d, kind):
    sc = StandardScaler().fit(Xtr)
    A, B_ = sc.transform(Xtr), sc.transform(Xte)
    if kind == "full":
        proj = None
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        clf.fit(Xtr, ytr); p = clf.predict_proba(Xte)[:, 1]
    else:
        if kind == "pca":
            proj = PCA(n_components=d, random_state=0, svd_solver="full" if d > 256 else "auto")
        else:
            proj = GaussianRandomProjection(n_components=d, random_state=0)
        Ztr = proj.fit_transform(A); Zte = proj.transform(B_)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        clf.fit(Ztr, ytr); p = clf.predict_proba(Zte)[:, 1]
    return float(roc_auc_score(yte, p)), boot_ci(yte, p, qte)

def main():
    out = {}
    for model in ["llama", "mistral", "qwen"]:
        bl = json.load(open(ROOT / f"results/{model}/all_results_logreg.json"))["best_layers"]
        ls, lc = bl["standard"], bl["deco"] if "deco" in bl else bl["DECO"]
        tr = load(model, "train", ls, lc); te = load(model, "test", ls, lc)
        res = {"layers": {"standard": ls, "csp": lc}, "pca": {}, "rp": {}}
        for pi, name in enumerate(["standard", "csp"]):
            Xtr, Xte = tr[pi], te[pi]
            for d in PCA_DIMS + ["full"]:
                t = time.time()
                a, ci = run(Xtr, tr[2], Xte, te[2], te[3], d, "full" if d == "full" else "pca")
                res["pca"].setdefault(str(d), {})[name] = {"auroc": a, "ci": ci}
                print(model, name, "pca", d, round(a, 4), f"{time.time()-t:.0f}s", flush=True)
            for d in RP_DIMS:
                a, ci = run(Xtr, tr[2], Xte, te[2], te[3], d, "rp")
                res["rp"].setdefault(str(d), {})[name] = {"auroc": a, "ci": ci}
                print(model, name, "rp", d, round(a, 4), flush=True)
            full = res["pca"]["full"][name]["auroc"]
            ref = PAPER[name][model]
            res.setdefault("sanity", {})[name] = {"full": full, "paper": ref, "ok": abs(full - ref) <= 0.005}
            if abs(full - ref) > 0.005:
                json.dump({**out, model: res}, open(ROOT / "results/dimension_matched_probes.json", "w"), indent=1)
                sys.exit(f"SANITY FAIL {model} {name}: {full:.4f} vs {ref}")
        out[model] = res
    json.dump(out, open(ROOT / "results/dimension_matched_probes.json", "w"), indent=1)
    f = lambda r: f"{r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
    md = []; thr = {}
    for model, r in out.items():
        md += [f"### {model} (layers: standard {r['layers']['standard']}, CSP {r['layers']['csp']})", "",
               "| d | Standard (PCA) | CSP (PCA) | Standard (random proj.) | CSP (random proj.) |", "|---|---|---|---|---|"]
        for d in [str(x) for x in PCA_DIMS] + ["full"]:
            rp = r["rp"].get(d)
            md.append(f"| {d if d!='full' else 'full (4096)'} | {f(r['pca'][d]['standard'])} | {f(r['pca'][d]['csp'])} | "
                      f"{f(rp['standard']) if rp else '-'} | {f(rp['csp']) if rp else '-'} |")
        md.append("")
        for name in ["standard", "csp"]:
            for t in (0.80, 0.85):
                ds = [x for x in PCA_DIMS if r["pca"][str(x)][name]["auroc"] > t]
                thr.setdefault(model, {})[f"{name}>{t}"] = min(ds) if ds else None
    md += ["### Smallest PCA d (from grid) with test AUROC above threshold", "", "| model | Standard >0.80 | Standard >0.85 | CSP >0.80 | CSP >0.85 |", "|---|---|---|---|---|"]
    for m, t in thr.items():
        md.append(f"| {m} | {t['standard>0.8']} | {t['standard>0.85']} | {t['csp>0.8']} | {t['csp>0.85']} |")
    out["thresholds"] = thr
    json.dump(out, open(ROOT / "results/dimension_matched_probes.json", "w"), indent=1)
    (ROOT / "results/dimension_matched_probes.md").write_text("\n".join(md) + "\n")
main()
