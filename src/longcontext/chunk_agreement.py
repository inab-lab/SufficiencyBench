#!/usr/bin/env python
"""Agreement between plain generate() VC scores and chunked-prefill VC scores at 32k.

Reads data/experiments/longcontext/<model>/32000/vc_scores.json (plain generate) and
vc_scores_chunked.json (forced chunked prefill, same prompts) and reports, on the
overlapping condition ids: exact-score agreement, mean |diff|, and AUROC of each on the
overlap. Writes results/longcontext/chunk_agreement.json.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from longcontext.common import length_dir, load_manifest, manifest_entries  # noqa: E402

def _scores(path):
    d = json.load(open(path))
    out = {}
    for cid, rec in d.items():
        s = rec.get("score") if isinstance(rec, dict) else rec
        if s is not None:
            out[cid] = float(s)
    return out

def main():
    manifest = load_manifest()
    label = {}
    for e in manifest_entries(manifest, lengths=["32000"]):
        label[e["condition_ids"]["sufficient"]] = 1
        label[e["condition_ids"]["insufficient"]] = 0
    report = {}
    for model in ["llama", "qwen"]:
        d = length_dir(model, "32000")
        p_plain, p_chunk = d / "vc_scores.json", d / "vc_scores_chunked.json"
        if not (p_plain.exists() and p_chunk.exists()):
            report[model] = {"status": "missing", "plain": p_plain.exists(), "chunked": p_chunk.exists()}
            continue
        a, b = _scores(p_plain), _scores(p_chunk)
        ids = sorted(set(a) & set(b) & set(label))
        if not ids:
            report[model] = {"status": "no overlap"}; continue
        va = np.array([a[i] for i in ids]); vb = np.array([b[i] for i in ids]); y = np.array([label[i] for i in ids])
        report[model] = {
            "n": len(ids),
            "exact_agreement": float(np.mean(va == vb)),
            "mean_abs_diff": float(np.mean(np.abs(va - vb))),
            "max_abs_diff": float(np.max(np.abs(va - vb))),
            "auroc_plain": float(roc_auc_score(y, va)) if len(set(y)) > 1 else None,
            "auroc_chunked": float(roc_auc_score(y, vb)) if len(set(y)) > 1 else None,
        }
    out = ROOT / "results/longcontext/chunk_agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
