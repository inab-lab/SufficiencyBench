"""
Open-weight PK-stratified AUROC table -> results/pk_stratified_auroc.json.

For each open-weight generator (llama, mistral, qwen) we split the
SufficiencyBench test conditions by the generator's parametric knowledge (PK)
of the underlying question, then report the AUROC of sufficiency prediction for:
  - the CSP sufficiency probe  (results/{model}/scores/probe_scores.json, "CSP (LogReg)")
  - verbalized confidence (VC)  (results/{model}/scores/baseline_scores.json)

PK label per question comes from data/experiments/sufficiency_bench/pk_cache_{model}.json
(knows=True -> PK=1, else PK=0). Each question contributes its suf + insuf
conditions (same PK). AUROC target label = condition "sufficient".

Usage:
  PYTHONPATH=$PWD:$PWD/src python src/evaluation/build_pk_stratified_auroc.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.paths import RESULTS_DIR, BENCH_DIR

MODELS = ["llama", "mistral", "qwen"]


def base_id(cid):
    """squad_133_suf -> squad_133 (strip the _suf/_insuf condition tag)."""
    for tag in ("_suf", "_insuf"):
        if cid.endswith(tag):
            return cid[: -len(tag)]
    return cid


def auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) < 10 or len(set(labels.tolist())) < 2:
        return None
    return round(float(roc_auc_score(labels, scores)), 4)


def load_scores(path, method_key):
    d = json.load(open(path))[method_key]
    out = {}
    for m, s in zip(d["metadata"], d["scores"]):
        out[m.get("condition_id", m.get("id"))] = {
            "score": float(s), "label": int(m["label"])}
    return out


def main():
    table = {}
    for model in MODELS:
        probe = load_scores(RESULTS_DIR / model / "scores" / "probe_scores.json",
                            "CSP (LogReg)")
        vc = load_scores(RESULTS_DIR / model / "scores" / "baseline_scores.json",
                         "verbalized_confidence")
        pk_cache = json.load(open(BENCH_DIR / f"pk_cache_{model}.json"))

        rows = []
        n_no_pk = 0
        for cid, pr in probe.items():
            if cid not in vc:
                continue
            bid = base_id(cid)
            if bid not in pk_cache:
                n_no_pk += 1
                continue
            rows.append({
                "cid": cid,
                "label": pr["label"],
                "probe": pr["score"],
                "vc": vc[cid]["score"],
                "pk": 1 if pk_cache[bid].get("knows") else 0,
            })

        def block(subset):
            return {
                "n": len(subset),
                "n_sufficient": int(sum(r["label"] for r in subset)),
                "probe_auroc": auroc([r["label"] for r in subset],
                                     [r["probe"] for r in subset]),
                "vc_auroc": auroc([r["label"] for r in subset],
                                  [r["vc"] for r in subset]),
            }

        pk0 = [r for r in rows if r["pk"] == 0]
        pk1 = [r for r in rows if r["pk"] == 1]
        entry = {
            "n_conditions": len(rows),
            "n_no_pk_label": n_no_pk,
            "pk0": block(pk0),
            "pk1": block(pk1),
            "all": block(rows),
        }
        for tag in ("pk0", "pk1", "all"):
            b = entry[tag]
            if b["probe_auroc"] is not None and b["vc_auroc"] is not None:
                b["probe_minus_vc_auroc"] = round(b["probe_auroc"] - b["vc_auroc"], 4)
        table[model] = entry
        print(f"{model}: PK=0 (n={entry['pk0']['n']}) probe={entry['pk0']['probe_auroc']} "
              f"vc={entry['pk0']['vc_auroc']} | PK=1 (n={entry['pk1']['n']}) "
              f"probe={entry['pk1']['probe_auroc']} vc={entry['pk1']['vc_auroc']}")

    payload = {
        "description": ("Open-weight PK-stratified sufficiency AUROC on the "
                        "SufficiencyBench test split. probe = CSP delta-h LogReg; "
                        "vc = verbalized confidence. PK from per-model pk_cache "
                        "(knows->PK=1). Each question contributes its suf+insuf "
                        "conditions."),
        "models": table,
    }
    out = RESULTS_DIR / "pk_stratified_auroc.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
