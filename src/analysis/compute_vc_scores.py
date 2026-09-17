"""
Compute per-example verbalized-confidence (VC) scores on the SufficiencyBench
test split, for one open-weight model, and cache them to
results/{model}/scores/baseline_scores.json under "verbalized_confidence".

Needed for the PK-stratified AUROC table (probe & VC AUROC by PK=0/PK=1):
store_baseline_scores.py only caches the (hidden-state) probe scores; the VC
baseline requires the model. Uses the batched, left-padded VC scorer from
src/methods/baselines.py (max_new_tokens=5) for speed / thermal headroom.

Usage:
  PYTHONPATH=$PWD:$PWD/src python src/analysis/compute_vc_scores.py --model mistral --device cuda:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS
from src.methods.baselines import verbalized_confidence_batch

BATCH = 16


def main(a):
    cfg = MODEL_CONFIGS[a.model]
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required by the batched VC scorer
    print(f"Loading {a.model} from {cfg['path']} on {a.device}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], device_map=a.device, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.eval()

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    conds = []
    for ex in test_data:
        for c in ex["conditions"]:
            conds.append({
                "condition_id": c["condition_id"],
                "id": c["condition_id"],
                "quadrant": c.get("quadrant"),
                "label": int(c["sufficient"]),
                "prompt": c["prompt"],
            })
    print(f"  {len(conds)} conditions", flush=True)

    scores = []
    t0 = time.time()
    for i in range(0, len(conds), BATCH):
        batch = conds[i:i + BATCH]
        scores.extend(verbalized_confidence_batch(
            model, tok, [b["prompt"] for b in batch], a.device))
        if i % (BATCH * 20) == 0 and i:
            el = time.time() - t0
            print(f"    {i}/{len(conds)} ({el/i:.3f}s/ex, "
                  f"eta {(len(conds)-i)*el/i/60:.1f}min)", flush=True)
        time.sleep(0.02)

    labels = [c["label"] for c in conds]
    meta = [{"condition_id": c["condition_id"], "id": c["id"],
             "quadrant": c["quadrant"], "label": c["label"]} for c in conds]
    auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    print(f"  overall VC AUROC = {auroc:.4f}", flush=True)

    out_dir = RESULTS_DIR / a.model / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_scores.json"
    payload = {}
    if out_path.exists():
        try:
            payload = json.load(open(out_path))
        except Exception:
            payload = {}
    payload["verbalized_confidence"] = {
        "metadata": meta, "scores": [float(s) for s in scores],
        "labels": labels, "overall_auroc": auroc,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
