"""
Evaluate the EXISTING ModernBERT SufficiencyProbe (v2, delta architecture) on
the SciQ OOD gate dataset, without retraining. This is the SciQ evaluation the
trainer skips when data/experiments/sufficiency_bench_sciq/test.json is absent;
now that the dataset is built (build_sufficiency_bench_sciq.py) we can score it.

Loads results/sufficiencyprobe_v2/best_checkpoint.pt, derives the decision
threshold on the SufficiencyBench val split (balanced-accuracy optimal, as the
trainer does), then reports SciQ OOD AUROC + gate accuracy.

Output: results/sciq_ood_eval.json

Usage:
  PYTHONPATH=$PWD:$PWD/src python src/evaluation/eval_sciq_ood.py --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from configs.paths import RESULTS_DIR, BENCH_DIR
from src.evaluation.train_sufficiencyprobe_v2 import (
    SufficiencyProbeV2, SBDataset, SciQDataset, collate,
    evaluate_auroc, find_best_threshold, MODEL_NAME, BATCH_SIZE,
)

CKPT = RESULTS_DIR / "sufficiencyprobe_v2" / "best_checkpoint.pt"
SCIQ_PATH = ROOT / "data" / "experiments" / "sufficiency_bench_sciq" / "test.json"


def gate_acc(scores, labels, t):
    preds = (scores >= t).astype(int)
    return float(np.mean([p if l == 1 else (1 - p)
                          for p, l in zip(preds, labels)]))


def main(a):
    device = a.device
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SufficiencyProbeV2(MODEL_NAME).to(device)
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {CKPT} (val_auroc={ckpt.get('val_auroc'):.4f}, epoch={ckpt.get('epoch')})",
          flush=True)

    kw = dict(batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)

    # Threshold from SB val (balanced-accuracy optimal), matching the trainer.
    val_ds = SBDataset(BENCH_DIR / "val.json", tok)
    val_auroc, val_scores, val_labels = evaluate_auroc(
        model, DataLoader(val_ds, shuffle=False, **kw), device)
    best_t, best_ba = find_best_threshold(val_scores, val_labels)
    print(f"SB val: AUROC={val_auroc:.4f}  best_t={best_t:.3f}  bal_acc={best_ba:.4f}", flush=True)

    # SciQ OOD
    sciq_ds = SciQDataset(SCIQ_PATH, tok)
    sciq_auroc, sciq_scores, sciq_labels = evaluate_auroc(
        model, DataLoader(sciq_ds, shuffle=False, **kw), device)
    g_best = gate_acc(sciq_scores, sciq_labels, best_t)
    g_05 = gate_acc(sciq_scores, sciq_labels, 0.5)
    print(f"SciQ OOD: AUROC={sciq_auroc:.4f}  gate_acc(t={best_t:.3f})={g_best:.4f}  "
          f"gate_acc(t=0.5)={g_05:.4f}  n={len(sciq_scores)}", flush=True)

    out = {
        "checkpoint": str(CKPT),
        "ckpt_val_auroc": float(ckpt.get("val_auroc")),
        "sb_val_auroc": val_auroc,
        "threshold_best_t": float(best_t),
        "sciq_ood": {
            "auroc": sciq_auroc,
            "gate_acc_best_t": g_best,
            "gate_acc_t0.5": g_05,
            "n_conditions": int(len(sciq_scores)),
            "n_sufficient": int(sciq_labels.sum()),
        },
        "sciq_dataset": str(SCIQ_PATH),
    }
    out_path = RESULTS_DIR / "sciq_ood_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
