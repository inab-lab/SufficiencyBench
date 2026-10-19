"""
eval_sufficiencyprobe.py  —  Evaluate the deployable SufficiencyProbe (ModernBERT)

Loads the checkpoint at results/sufficiencyprobe/best_checkpoint.pt and computes:
  1. AUROC on SufficiencyBench test set (main metric)
  2. Per question-type AUROC (factual / multi-hop / comparative / subjective)
  3. AUROC on SciQ out-of-domain (no retraining)
  4. Calibration — reliability diagram (10 bins)
  5. PK-stratified AUROC (PK=0 vs PK=1 for Mistral / Llama / Qwen)
  6. Agreement with Llama and Qwen CSP probes (cross-model label consistency)
  7. ONNX export + CPU latency benchmark

Saves everything to results/sufficiencyprobe/evaluation_results.json
Also saves the reliability diagram to results/sufficiencyprobe/calibration.png

Usage (from MSD_Project root, after train_sufficiencyprobe.py completes):
  python src/evaluation/eval_sufficiencyprobe.py
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Paths  (all relative to MSD_Project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH    = PROJECT_ROOT / "results" / "sufficiencyprobe" / "best_checkpoint.pt"
OUT_DIR      = PROJECT_ROOT / "results" / "sufficiencyprobe"
BENCH_TEST   = PROJECT_ROOT / "data" / "experiments" / "sufficiency_bench" / "test.json"
SCIQ_DATA    = PROJECT_ROOT / "datasets_for_github" / "sufficiencybench_sciq" / "data.json"
LAYER_FILE   = PROJECT_ROOT / "results" / "layer_analysis.json"
HS_DIR       = PROJECT_ROOT / "data" / "experiments" / "hidden_states"

MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_LEN    = 512
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Model (same architecture used in training)
# ---------------------------------------------------------------------------
class SufficiencyProbe(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def flatten_bench(path):
    """Return list of (question, context, label, question_type, condition_id, pk_llama, pk_qwen)."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for ex in data:
        q     = ex["question"]
        qtype = ex["question_type"]
        pk_ll = ex.get("pk_llama", None)
        pk_qw = ex.get("pk_qwen", None)
        for cond in ex["conditions"]:
            rows.append({
                "question":      q,
                "context":       cond["context"],
                "label":         int(cond["sufficient"]),
                "question_type": qtype,
                "condition_id":  cond["condition_id"],
                "quadrant":      cond.get("quadrant", "?"),
                "pk_llama":      pk_ll,
                "pk_qwen":       pk_qw,
            })
    return rows


def make_batches(rows, tokenizer, batch_size):
    all_ids, all_masks = [], []
    for row in rows:
        enc = tokenizer(
            row["question"], row["context"],
            max_length=MAX_LEN,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    # Batch
    n = len(all_ids)
    batches = []
    for i in range(0, n, batch_size):
        ids   = torch.cat(all_ids[i:i+batch_size],   dim=0)
        masks = torch.cat(all_masks[i:i+batch_size], dim=0)
        batches.append((ids, masks))
    return batches


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, batches, device):
    model.eval()
    scores = []
    for ids, masks in batches:
        logits = model(ids.to(device), masks.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        scores.extend(probs.tolist())
    return np.array(scores)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_stats(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(float(np.mean(labels[mask])))
        bin_confs.append(float(np.mean(scores[mask])))
        bin_ns.append(int(mask.sum()))
    ece = sum(
        abs(acc - conf) * n / len(scores)
        for acc, conf, n in zip(bin_accs, bin_confs, bin_ns)
    )
    return {"bin_accs": bin_accs, "bin_confs": bin_confs, "bin_ns": bin_ns, "ece": float(ece)}


# ---------------------------------------------------------------------------
# CSP probe agreement (cross-model label consistency)
# ---------------------------------------------------------------------------
def csp_agreement_auroc(sp_scores, test_rows):
    """
    For each open-weight model, load best-layer hidden-state predictions on the
    same test examples and compute AUROC(SufficiencyProbe scores, CSP probe labels).
    High AUROC means the SufficiencyProbe recovers the same signal as the CSP probe.
    """
    results = {}
    id_to_idx = {r["condition_id"]: i for i, r in enumerate(test_rows)}
    with open(LAYER_FILE) as f:
        la_all = json.load(f)
    for mk in ["llama", "mistral", "qwen"]:
        hs = HS_DIR / mk / "test"
        if not hs.exists():
            results[mk] = None
            continue
        best_layer = la_all[mk]["best_val_layer"] if mk in la_all else 9
        h_qc = np.load(hs / "h_with_context.npy",   mmap_mode="r")
        h_q  = np.load(hs / "h_question_only.npy",  mmap_mode="r")
        with open(hs / "metadata.json") as f:
            meta = json.load(f)
        X = h_qc[:, best_layer, :] - h_q[:, best_layer, :]
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        # Train probe on train split to get CSP scores for test
        hs_tr = HS_DIR / mk / "train"  # same base HS_DIR
        h_qc_tr = np.load(hs_tr / "h_with_context.npy", mmap_mode="r")
        h_q_tr  = np.load(hs_tr / "h_question_only.npy", mmap_mode="r")
        y_tr    = np.load(hs_tr / "labels.npy")
        X_tr    = h_qc_tr[:, best_layer, :] - h_q_tr[:, best_layer, :]
        scaler  = StandardScaler()
        lr      = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(scaler.fit_transform(X_tr.astype(np.float32)), y_tr)
        csp_scores_test = lr.predict_proba(scaler.transform(X.astype(np.float32)))[:, 1]
        # Align by condition_id
        cond_ids_hs = [m["id"] for m in meta]
        aligned_sp, aligned_csp = [], []
        for cid, csp_s in zip(cond_ids_hs, csp_scores_test):
            if cid in id_to_idx:
                idx = id_to_idx[cid]
                aligned_sp.append(sp_scores[idx])
                aligned_csp.append(csp_s)
        if len(aligned_sp) > 10:
            csp_binary = (np.array(aligned_csp) > 0.5).astype(int)
            auroc = float(roc_auc_score(csp_binary, np.array(aligned_sp)))
            results[mk] = {"auroc_vs_csp_labels": auroc, "n": len(aligned_sp)}
        else:
            results[mk] = None
    return results


# ---------------------------------------------------------------------------
# ONNX export + latency benchmark
# ---------------------------------------------------------------------------
def onnx_latency(model, tokenizer, device, n_warmup=20, n_bench=200):
    dummy_q = "What is the capital of France?"
    dummy_c = "France is a country in Western Europe. Its capital city is Paris."
    enc = tokenizer(dummy_q, dummy_c, max_length=MAX_LEN, truncation="only_second",
                    padding="max_length", return_tensors="pt")
    ids   = enc["input_ids"].to(device)
    masks = enc["attention_mask"].to(device)
    model.eval()
    # PyTorch CPU latency
    model_cpu = model.cpu()
    ids_cpu   = ids.cpu()
    masks_cpu = masks.cpu()
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model_cpu(ids_cpu, masks_cpu)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        with torch.no_grad():
            model_cpu(ids_cpu, masks_cpu)
    elapsed = (time.perf_counter() - t0) / n_bench * 1000  # ms per example

    # ONNX export
    onnx_path = OUT_DIR / "sufficiencyprobe.onnx"
    try:
        torch.onnx.export(
            model_cpu,
            (ids_cpu, masks_cpu),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logit"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
            opset_version=17,
        )
        # ONNX runtime latency
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp  = {"input_ids": ids_cpu.numpy(), "attention_mask": masks_cpu.numpy()}
        for _ in range(n_warmup):
            sess.run(None, inp)
        t0 = time.perf_counter()
        for _ in range(n_bench):
            sess.run(None, inp)
        onnx_ms = (time.perf_counter() - t0) / n_bench * 1000
        onnx_exported = True
    except Exception as e:
        onnx_ms = None
        onnx_exported = False
        print(f"  ONNX export/benchmark failed: {e}")

    return {
        "pytorch_cpu_ms":  float(elapsed),
        "onnx_cpu_ms":     float(onnx_ms) if onnx_ms is not None else None,
        "onnx_exported":   onnx_exported,
        "onnx_path":       str(onnx_path) if onnx_exported else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer + model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = SufficiencyProbe(MODEL_NAME)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # Handle different checkpoint formats
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {CKPT_PATH}")

    results = {}

    # ------------------------------------------------------------------
    # 1. SufficiencyBench test set
    # ------------------------------------------------------------------
    print("\n[1/6] SufficiencyBench test set...")
    test_rows = flatten_bench(BENCH_TEST)
    batches   = make_batches(test_rows, tokenizer, BATCH_SIZE)
    sp_scores = run_inference(model, batches, device)
    labels    = np.array([r["label"] for r in test_rows])

    auroc_overall = float(roc_auc_score(labels, sp_scores))
    print(f"  Overall AUROC: {auroc_overall:.4f}")

    # Per question-type
    per_type = {}
    for qtype in ["factual", "multi_hop", "comparative", "subjective"]:
        mask = np.array([r["question_type"] == qtype for r in test_rows])
        if mask.sum() > 10:
            a = float(roc_auc_score(labels[mask], sp_scores[mask]))
            per_type[qtype] = {"auroc": a, "n": int(mask.sum())}
            print(f"  {qtype}: AUROC={a:.4f} (n={int(mask.sum())})")

    # Per quadrant AUROC
    per_quad = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = np.array([r["quadrant"] == q for r in test_rows])
        if mask.sum() > 10:
            per_quad[q] = {"auroc": float(roc_auc_score(labels[mask], sp_scores[mask])),
                           "n": int(mask.sum())}

    results["sufficiencybench_test"] = {
        "overall_auroc": auroc_overall,
        "n": len(test_rows),
        "per_type": per_type,
        "per_quadrant": per_quad,
    }

    # ------------------------------------------------------------------
    # 2. PK-stratified AUROC (Mistral PK labels)
    # ------------------------------------------------------------------
    print("\n[2/6] PK-stratified AUROC...")
    pk_results = {}
    for pk_field, model_name in [("pk_llama", "llama"), ("pk_qwen", "qwen")]:
        pk0_mask = np.array([r[pk_field] == 0 for r in test_rows
                             if r[pk_field] is not None])
        pk1_mask = np.array([r[pk_field] == 1 for r in test_rows
                             if r[pk_field] is not None])
        valid    = np.array([r[pk_field] is not None for r in test_rows])
        if valid.sum() > 100:
            pk0  = np.array([r[pk_field] == 0 for r in test_rows]) & valid
            pk1  = np.array([r[pk_field] == 1 for r in test_rows]) & valid
            pk_results[model_name] = {}
            if pk0.sum() > 10:
                pk_results[model_name]["pk0_auroc"] = float(roc_auc_score(labels[pk0], sp_scores[pk0]))
            if pk1.sum() > 10:
                pk_results[model_name]["pk1_auroc"] = float(roc_auc_score(labels[pk1], sp_scores[pk1]))
            pk0_str = f"{pk_results[model_name]['pk0_auroc']:.4f}" if "pk0_auroc" in pk_results[model_name] else "N/A"
            pk1_str = f"{pk_results[model_name]['pk1_auroc']:.4f}" if "pk1_auroc" in pk_results[model_name] else "N/A"
            print(f"  {model_name}: PK=0 AUROC={pk0_str}, PK=1 AUROC={pk1_str}")
    results["pk_stratified"] = pk_results

    # ------------------------------------------------------------------
    # 3. SciQ out-of-domain
    # ------------------------------------------------------------------
    print("\n[3/6] SciQ out-of-domain...")
    sciq_rows    = flatten_bench(SCIQ_DATA)
    sciq_batches = make_batches(sciq_rows, tokenizer, BATCH_SIZE)
    sciq_scores  = run_inference(model, sciq_batches, device)
    sciq_labels  = np.array([r["label"] for r in sciq_rows])
    sciq_auroc   = float(roc_auc_score(sciq_labels, sciq_scores))
    print(f"  SciQ AUROC: {sciq_auroc:.4f}")
    results["sciq_ood"] = {"auroc": sciq_auroc, "n": len(sciq_rows)}

    # ------------------------------------------------------------------
    # 4. Calibration
    # ------------------------------------------------------------------
    print("\n[4/6] Calibration...")
    cal = calibration_stats(sp_scores, labels)
    print(f"  ECE: {cal['ece']:.4f}")
    results["calibration"] = cal

    # Try to plot calibration
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.scatter(cal["bin_confs"], cal["bin_accs"], s=[n/2 for n in cal["bin_ns"]],
                   color="steelblue", zorder=3)
        ax.plot(cal["bin_confs"], cal["bin_accs"], color="steelblue", label=f"SufficiencyProbe (ECE={cal['ece']:.3f})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("SufficiencyProbe Calibration")
        ax.legend()
        fig.tight_layout()
        cal_path = OUT_DIR / "calibration.png"
        fig.savefig(cal_path, dpi=150)
        print(f"  Calibration plot saved: {cal_path}")
    except Exception as e:
        print(f"  Could not save calibration plot: {e}")

    # ------------------------------------------------------------------
    # 5. Cross-model CSP probe agreement
    # ------------------------------------------------------------------
    print("\n[5/6] Cross-model CSP probe agreement...")
    try:
        agreement = csp_agreement_auroc(sp_scores, test_rows)
        for mk, res in agreement.items():
            if res:
                print(f"  AUROC vs {mk} CSP labels: {res['auroc_vs_csp_labels']:.4f} (n={res['n']})")
        results["csp_agreement"] = agreement
    except Exception as e:
        print(f"  Skipped: {e}")
        results["csp_agreement"] = None

    # ------------------------------------------------------------------
    # 6. ONNX + latency
    # ------------------------------------------------------------------
    print("\n[6/6] ONNX export and latency benchmark...")
    try:
        lat = onnx_latency(model, tokenizer, device)
        print(f"  PyTorch CPU: {lat['pytorch_cpu_ms']:.2f} ms/example")
        if lat["onnx_cpu_ms"] is not None:
            print(f"  ONNX CPU:    {lat['onnx_cpu_ms']:.2f} ms/example")
        results["latency"] = lat
    except Exception as e:
        print(f"  Latency benchmark failed: {e}")
        results["latency"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Downstream accuracy at t=0.5 (approximation via sufficiency labels)
    # ------------------------------------------------------------------
    # This replicates the simulation in clean_probe_sim_results.json:
    # Gate = use context if score >= 0.5. "Correct" = gate matches sufficiency label.
    preds = (sp_scores >= 0.5).astype(int)
    gate_accuracy = float(np.mean(preds == labels))
    # Simulated downstream accuracy matches the existing framework's convention:
    # No gate accuracy = 0.442 (established baseline from ground-truth Sonnet sim)
    # Correct gate decision rate gives an approximation of downstream accuracy.
    # This is an approximation — actual downstream accuracy requires running Sonnet
    # on the SufficiencyProbe-gated examples. The gate accuracy here is provided
    # for reference only.
    results["gate_accuracy_at_t05"] = gate_accuracy
    print(f"\n  Gate accuracy (score>=0.5 vs sufficiency label): {gate_accuracy:.4f}")
    print("  Note: Actual downstream accuracy requires running Sonnet on gated examples.")
    print("  Compare: clean CSP probe downstream acc = 72.6%, Sonnet VC = 69.2%")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SufficiencyBench AUROC:  {auroc_overall:.4f}  (CSP probe upper bound: 0.910)")
    print(f"SciQ OOD AUROC:         {sciq_auroc:.4f}  (CSP probe: 0.824)")
    print(f"Calibration ECE:        {cal['ece']:.4f}")
    if "latency" in results and "onnx_cpu_ms" in results["latency"]:
        lat_val = results["latency"]["onnx_cpu_ms"]
        if lat_val:
            print(f"ONNX CPU latency:       {lat_val:.2f} ms")

    # Save
    out_path = OUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved: {out_path}")


if __name__ == "__main__":
    main()
