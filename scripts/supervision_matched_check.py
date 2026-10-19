#!/usr/bin/env python
"""Two checks on whether the probe advantage is an artefact of supervision or of in-distribution evaluation.

(A) Supervision-matched comparison on the test split, 5-fold CV grouped by question:
    - 'trained output readout': logistic regression on every per-example output-based score available
      for the model (verbalized confidence, response delta, token-probability delta, verbalized comparison, ...)
    - Standard and CSP probes at the paper's selected layer, same folds.
(B) Leave-one-source-dataset-out probe transfer: train the CSP/Standard probe on the TRAIN split restricted
    to two source datasets, test on the TEST split of the held-out source (squad / hotpot / eli5),
    compared with the in-source number from the same train split.
Output: results/supervision_matched_check.json
"""
import json, glob, re, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]
# zero-based block index selected on validation (Appendix D), read from the probe results of the current benchmark version
LAYER = {m: int(json.load(open(ROOT / f"results/{m}/all_results_logreg.json"))["best_layers"]["standard"]) for m in ("llama", "mistral", "qwen")}
N_TEST_COND = 2 * len(json.load(open(ROOT / "data/experiments/sufficiency_bench/test.json")))
HS = ROOT / "data/experiments/hidden_states"
def qid(cid): return cid.rsplit("_", 1)[0]
def source(cid): return cid.split("_")[0]
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

def output_scores(model):
    """Collect every per-example score dict of the form {metadata:[{condition_id,label}], scores:[...]}."""
    found = {}
    for f in glob.glob(str(ROOT / f"results/{model}/scores/*.json")):
        d = json.load(open(f))
        for name, v in d.items():
            if isinstance(v, dict) and "scores" in v and "metadata" in v and len(v["scores"]) == N_TEST_COND:
                if "probe" in name.lower() or "LogReg" in name or "Neural" in name: continue
                found[name] = {m["condition_id"]: float(s) for m, s in zip(v["metadata"], v["scores"]) if s is not None and not (isinstance(s, float) and np.isnan(s))}
    return found

def load_split(model, split, layer):
    d = HS / model / split
    if not d.exists(): return None
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

res = {}
for model in ["llama", "mistral", "qwen"]:
    test = load_split(model, "test", LAYER[model])
    if test is None: print(model, "no test hidden states"); continue
    ids = test["ids"]; y = test["y"]; g = np.array([qid(c) for c in ids])
    out = output_scores(model)
    feats = sorted(out.keys())
    X_out = np.array([[out[f].get(c, np.nan) for f in feats] for c in ids], dtype=float)
    X_out = np.where(np.isnan(X_out), np.nanmean(X_out, axis=0), X_out)
    r = {"layer": LAYER[model], "n": int(len(y)), "output_features": feats,
         "A_trained_output_readout_cv": cv_auroc(X_out, y, g),
         "A_best_single_output_feature": max(float(roc_auc_score(y, X_out[:, i])) if roc_auc_score(y, X_out[:, i]) >= 0.5 else 1 - float(roc_auc_score(y, X_out[:, i])) for i in range(len(feats))),
         "A_standard_probe_cv": cv_auroc(test["std"], y, g),
         "A_csp_probe_cv": cv_auroc(test["csp"], y, g)}
    train = load_split(model, "train", LAYER[model])
    if train is not None:
        tr_src = np.array([source(c) for c in train["ids"]]); te_src = np.array([source(c) for c in ids])
        loso = {}
        for s in sorted(set(te_src)):
            for kind in ("std", "csp"):
                clf = lr().fit(train[kind][tr_src != s], train["y"][tr_src != s])
                held = float(roc_auc_score(y[te_src == s], clf.predict_proba(test[kind][te_src == s])[:, 1]))
                clf_in = lr().fit(train[kind], train["y"])
                insrc = float(roc_auc_score(y[te_src == s], clf_in.predict_proba(test[kind][te_src == s])[:, 1]))
                loso[f"{kind}_heldout_{s}"] = held; loso[f"{kind}_insource_{s}"] = insrc
        r["B_leave_one_source_out"] = loso
    res[model] = r
    print(model, json.dumps(r, indent=1))
json.dump(res, open(ROOT / "results/supervision_matched_check.json", "w"), indent=1)
