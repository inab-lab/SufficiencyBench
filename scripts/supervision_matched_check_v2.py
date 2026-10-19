#!/usr/bin/env python
"""Check A, v2: supervision-matched trained output readout extended with the zero-shot output readouts
(vc_template, pyes_raw, pyes_template from results/<model>/output_readouts.json). Same folds/seeds as
scripts/supervision_matched_check.py (GroupKFold(5) grouped by question, same example order).
Output: results/supervision_matched_check_v2.json and .md"""
import json, glob, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "mistral", "qwen")
LAYER = {m: int(json.load(open(ROOT / f"results/{m}/all_results_logreg.json"))["best_layers"]["standard"]) for m in MODELS}
N_TEST_COND = 2 * len(json.load(open(ROOT / "data/experiments/sufficiency_bench/test.json")))
HS = ROOT / "data/experiments/hidden_states"
ORACLE = "Token Prob Delta"
NEW = ["vc_template", "pyes_raw", "pyes_template"]
def qid(cid): return cid.rsplit("_", 1)[0]
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

def output_scores(model):  # copied from original
    found = {}
    for f in glob.glob(str(ROOT / f"results/{model}/scores/*.json")):
        d = json.load(open(f))
        for name, v in d.items():
            if isinstance(v, dict) and "scores" in v and "metadata" in v and len(v["scores"]) == N_TEST_COND:
                if "probe" in name.lower() or "LogReg" in name or "Neural" in name: continue
                found[name] = {m["condition_id"]: float(s) for m, s in zip(v["metadata"], v["scores"]) if s is not None and not (isinstance(s, float) and np.isnan(s))}
    return found

def load_split(model, split, layer):  # copied from original
    d = HS / model / split
    meta = json.load(open(d / "metadata.json"))
    ids = [(m.get("condition_id") or m.get("id")) if isinstance(m, dict) else m for m in (meta if isinstance(meta, list) else meta["conditions"] if "conditions" in meta else meta["condition_ids"])]
    hq = np.load(d / "h_question_only.npy", mmap_mode="r")[:, layer, :]
    hc = np.load(d / "h_with_context.npy", mmap_mode="r")[:, layer, :]
    y = np.load(d / "labels.npy")
    return {"ids": ids, "std": np.asarray(hc, dtype=np.float32), "csp": np.asarray(hc - hq, dtype=np.float32), "y": np.asarray(y).astype(int)}

def cv_auroc(X, y, groups, seed=0):
    gkf = GroupKFold(5); p = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = lr().fit(X[tr], y[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))

def matrix(out, feats, ids):
    X = np.array([[out[f].get(c, np.nan) for f in feats] for c in ids], dtype=float)
    return np.where(np.isnan(X), np.nanmean(X, axis=0), X)

ref = json.load(open(ROOT / "results/supervision_matched_check.json"))
res = {}; disc = []
for model in MODELS:
    test = load_split(model, "test", LAYER[model]); ids = test["ids"]; y = test["y"]
    g = np.array([qid(c) for c in ids])
    out = output_scores(model)
    orig = sorted(out.keys())
    ro = json.load(open(ROOT / f"results/{model}/output_readouts.json"))
    dropped = {}
    for k in NEW:
        out[k] = {m["condition_id"]: float(s) for m, s in zip(ro[k]["metadata"], ro[k]["scores"]) if s is not None and not np.isnan(s)}
        dropped[k] = len(set(ids) - set(out[k]))
        assert set(out[k]) <= set(ids) and len(ro[k]["scores"]) == len(ids), (model, k)
        assert dropped[k] == 0, f"{model} {k}: {dropped[k]} conditions missing"
    nonor = [f for f in orig if f != ORACLE]
    ext = nonor + NEW; ext_or = ext + [ORACLE]
    r = {"layer": LAYER[model], "n": int(len(y)), "dropped_conditions": dropped, "original_features": orig,
         "extended_features": ext}
    r["A_original_readout_cv"] = cv_auroc(matrix(out, orig, ids), y, g)
    r["A_original_nonoracle_cv"] = cv_auroc(matrix(out, nonor, ids), y, g)
    r["A_extended_nonoracle_cv"] = cv_auroc(matrix(out, ext, ids), y, g)
    r["A_extended_oracle_cv"] = cv_auroc(matrix(out, ext_or, ids), y, g)
    Xe = matrix(out, ext_or, ids)
    sing = {f: (lambda a: max(a, 1 - a))(float(roc_auc_score(y, Xe[:, i]))) for i, f in enumerate(ext_or)}
    bf = max(sing, key=sing.get)
    r["A_best_single_output_feature"] = sing[bf]; r["A_best_single_output_feature_name"] = bf; r["single_feature_auroc"] = sing
    r["A_standard_probe_cv"] = cv_auroc(test["std"], y, g)
    r["A_csp_probe_cv"] = cv_auroc(test["csp"], y, g)
    for k_new, k_old in [("A_original_readout_cv", ref[model]["A_trained_output_readout_cv"]),
                         ("A_original_nonoracle_cv", ref[model]["A2_output_readout_subsets"]["non_oracle"]["overall"]),
                         ("A_standard_probe_cv", ref[model]["A_standard_probe_cv"]), ("A_csp_probe_cv", ref[model]["A_csp_probe_cv"])]:
        if round(r[k_new], 3) != round(k_old, 3): disc.append((model, k_new, r[k_new], k_old))
    res[model] = r; print(model, json.dumps(r, indent=1))
res["_discrepancies_vs_original"] = disc
json.dump(res, open(ROOT / "results/supervision_matched_check_v2.json", "w"), indent=1)
cols = [("A_original_readout_cv", "Orig readout"), ("A_original_nonoracle_cv", "Orig non-oracle"), ("A_extended_nonoracle_cv", "Ext non-oracle"),
        ("A_extended_oracle_cv", "Ext oracle"), ("A_best_single_output_feature", "Best single"), ("A_standard_probe_cv", "Standard probe"), ("A_csp_probe_cv", "CSP probe")]
L = ["| Model | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
for m in MODELS:
    L.append(f"| {m} | " + " | ".join(f"{res[m][k]:.3f}" + (f" ({res[m]['A_best_single_output_feature_name']})" if k == "A_best_single_output_feature" else "") for k, _ in cols) + " |")
open(ROOT / "results/supervision_matched_check_v2.md", "w").write("\n".join(L) + "\n")
print("\n".join(L)); print("DISCREPANCIES:", disc)
