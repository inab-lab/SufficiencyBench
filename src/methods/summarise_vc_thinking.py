#!/usr/bin/env python
"""Rebuild results/qwen3/vc_thinking_summary.json from every per-variant jsonl cache.

The runner writes the summary per invocation, so running the cheap variants and the
thinking variant in separate calls leaves only the last one in the file. This recomputes
all variants present in results/qwen3/vc_thinking/*.jsonl.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/qwen3/vc_thinking"
BENCH = ROOT / "data/experiments/sufficiency_bench/test.json"
OUT = ROOT / "results/qwen3/vc_thinking_summary.json"
N_BOOT, SEED = 1000, 0


def labels_and_groups():
    lab, grp, qtype = {}, {}, {}
    for ex in json.load(open(BENCH)):
        for c in ex["conditions"]:
            cid = c["condition_id"]
            lab[cid] = int(bool(c["sufficient"]))
            grp[cid] = ex["id"]
            qtype[cid] = ex["question_type"]
    return lab, grp, qtype


def boot_ci(y, s, groups, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    idx = {q: np.flatnonzero(groups == q) for q in uq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        sel = np.concatenate([idx[q] for q in pick])
        if len(set(y[sel])) < 2:
            continue
        out.append(roc_auc_score(y[sel], s[sel]))
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    lab, grp, qtype = labels_and_groups()
    summary = {}
    for path in sorted(CACHE.glob("*.jsonl")):
        variant = path.stem
        recs = {}
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[r["condition_id"]] = r
        cids = [c for c in recs if c in lab and recs[c].get("score") is not None]
        if not cids:
            continue
        y = np.array([lab[c] for c in cids])
        s = np.array([float(recs[c]["score"]) for c in cids])
        g = np.array([grp[c] for c in cids])
        lo, hi = boot_ci(y, s, g)
        entry = {
            "variant": variant,
            "n": len(cids),
            "auroc": float(roc_auc_score(y, s)),
            "ci_lo": lo,
            "ci_hi": hi,
        }
        trunc = [bool(recs[c].get("truncated")) for c in cids]
        if any(t is not None for t in trunc):
            entry["truncation_rate"] = float(np.mean(trunc))
        think = [recs[c].get("thinking_tokens") for c in cids
                 if recs[c].get("thinking_tokens") is not None]
        if think:
            entry["mean_thinking_tokens"] = float(np.mean(think))
        per = {}
        for t in ["factual", "multi_hop", "comparative", "subjective"]:
            m = np.array([qtype[c] == t for c in cids])
            if m.sum() and len(set(y[m])) > 1:
                per[t] = {"auroc": float(roc_auc_score(y[m], s[m])), "n": int(m.sum())}
        entry["per_question_type"] = per
        summary[variant] = entry
        print(f"{variant:14s} n={entry['n']:5d} AUROC {entry['auroc']:.4f} "
              f"[{lo:.3f}, {hi:.3f}]"
              + (f" trunc={entry.get('truncation_rate'):.3f}" if "truncation_rate" in entry else "")
              + (f" think_tok={entry.get('mean_thinking_tokens'):.0f}" if "mean_thinking_tokens" in entry else ""))
    json.dump(summary, open(OUT, "w"), indent=2)
    print("->", OUT)


if __name__ == "__main__":
    main()
