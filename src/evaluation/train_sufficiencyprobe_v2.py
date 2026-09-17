"""
Train SufficiencyProbe v2 — delta construction on ModernBERT-large.

Architecture: h_ModernBERT(q+c) - h_ModernBERT(q)  →  linear head
Training data: COVERAGE-ONLY matched pairs. Each pair is topic-matched:
    sufficient = gold context, insufficient = same context with the answer removed
    (DivBench triviaqa/nq/hotpotqa) or SufficiencyBench's native answer-replacement
    conditions (SB TRAIN/VAL only; SB test is never used and the splits are
    question-disjoint, so no test leakage — this mirrors build_surgical_pairs.py).
    Training on the raw diverse_bench combined set instead makes "insufficient" a
    topic/retrieval mismatch, so the probe learns topic-relevance and collapses to
    ~0.65 AUROC on SufficiencyBench.
    Loss: pairwise contrastive -logsigmoid(score_suf - score_insuf) + BCE. Plain
    BCE cannot learn this subtle matched-context signal (it collapses to ~0.5 val).
    Rebuild data with: paper2/src/data/build_coverage_pairs.py
Threshold: selected on SufficiencyBench val split
Evaluation: SufficiencyBench test + SciQ

Usage (from MSD_Project root):
  python src/evaluation/train_sufficiencyprobe_v2.py
"""

import json, time, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import PROJECT_ROOT

# DiverseBench lives in the sibling knowledge_state_geometry project.
# Override with the DIVERSE_DIR env var if it is stored elsewhere.
DIVERSE_DIR  = Path(os.environ.get(
    "DIVERSE_DIR",
    PROJECT_ROOT.parent / "knowledge_state_geometry" / "data" / "diverse_bench" / "coverage",
))
SB_DIR       = PROJECT_ROOT / "data" / "experiments" / "sufficiency_bench"
SCIQ_PATH    = PROJECT_ROOT / "data" / "experiments" / "sufficiency_bench_sciq" / "test.json"
OUT_DIR      = PROJECT_ROOT / "results" / "sufficiencyprobe_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "answerdotai/ModernBERT-large"
TRAIN_SOURCES = {"triviaqa", "nq", "hotpotqa", "sufficiencybench"}
MAX_LEN      = 512
Q_MAX_LEN    = 128
BATCH_SIZE   = 16
LR           = 2e-5
EPOCHS       = 10
PATIENCE     = 3
WARMUP_RATIO = 0.10


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DiverseDataset(Dataset):
    """TriviaQA+NQ examples from DiverseBench — flat (one row per condition)."""
    def __init__(self, path, tokenizer, sources):
        with open(path) as f:
            raw = json.load(f)
        self.samples = []
        for ex in raw:
            if ex["source"] not in sources:
                continue
            qc = tokenizer(ex["question"], ex["context"],
                           max_length=MAX_LEN, truncation=True,
                           padding="max_length", return_tensors="pt")
            q  = tokenizer(ex["question"],
                           max_length=Q_MAX_LEN, truncation=True,
                           padding="max_length", return_tensors="pt")
            self.samples.append({
                "qc_ids":  qc["input_ids"].squeeze(0),
                "qc_mask": qc["attention_mask"].squeeze(0),
                "q_ids":   q["input_ids"].squeeze(0),
                "q_mask":  q["attention_mask"].squeeze(0),
                "label":   float(ex["sufficient"]),
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class CoveragePairDataset(Dataset):
    """Topic-matched coverage PAIRS for contrastive+BCE training.

    build_coverage_pairs.py writes each pair as two consecutive rows
    (sufficient, then insufficient) sharing the same question. We reassemble
    them into (q, c_suf, c_insuf) triples so the trainer can apply the original
    stage4 pairwise contrastive loss -logsigmoid(score_suf - score_insuf), which
    the plain-BCE reconstruction lacked (and which BCE-alone cannot learn on this
    subtle, matched-context signal — it collapses to ~0.5 val AUROC)."""
    def __init__(self, path, tokenizer, sources):
        with open(path) as f:
            raw = json.load(f)
        rows = [ex for ex in raw if ex["source"] in sources]
        self.samples = []
        for i in range(0, len(rows) - 1, 2):
            a, b = rows[i], rows[i + 1]
            suf, ins = (a, b) if a["sufficient"] else (b, a)
            if suf["sufficient"] == ins["sufficient"]:
                continue  # not a clean pair, skip
            q = tokenizer(suf["question"], max_length=Q_MAX_LEN, truncation=True,
                          padding="max_length", return_tensors="pt")
            qc_s = tokenizer(suf["question"], suf["context"], max_length=MAX_LEN,
                             truncation=True, padding="max_length", return_tensors="pt")
            qc_i = tokenizer(ins["question"], ins["context"], max_length=MAX_LEN,
                             truncation=True, padding="max_length", return_tensors="pt")
            self.samples.append({
                "q_ids":   q["input_ids"].squeeze(0),
                "q_mask":  q["attention_mask"].squeeze(0),
                "s_ids":   qc_s["input_ids"].squeeze(0),
                "s_mask":  qc_s["attention_mask"].squeeze(0),
                "i_ids":   qc_i["input_ids"].squeeze(0),
                "i_mask":  qc_i["attention_mask"].squeeze(0),
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class SBDataset(Dataset):
    """SufficiencyBench — nested format (question → conditions)."""
    def __init__(self, path, tokenizer):
        with open(path) as f:
            raw = json.load(f)
        self.samples = []
        for ex in raw:
            pk = ex.get("model_knows_answer", True)
            pk_bool = (pk == "True" or pk is True)
            conds = ex["conditions"]
            if isinstance(conds, str):
                import ast; conds = ast.literal_eval(conds)
            for c in conds:
                qc = tokenizer(ex["question"], c["context"],
                               max_length=MAX_LEN, truncation=True,
                               padding="max_length", return_tensors="pt")
                q  = tokenizer(ex["question"],
                               max_length=Q_MAX_LEN, truncation=True,
                               padding="max_length", return_tensors="pt")
                self.samples.append({
                    "qc_ids":  qc["input_ids"].squeeze(0),
                    "qc_mask": qc["attention_mask"].squeeze(0),
                    "q_ids":   q["input_ids"].squeeze(0),
                    "q_mask":  q["attention_mask"].squeeze(0),
                    "label":   float(c["sufficient"]),
                    "id":      c["condition_id"],
                    "pk":      pk_bool,
                    "quadrant": c.get("quadrant", ""),
                    "qtype":   ex.get("question_type", ""),
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class SciQDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path) as f:
            raw = json.load(f)
        self.samples = []
        for ex in raw:
            for c in ex["conditions"]:
                qc = tokenizer(ex["question"], c["context"],
                               max_length=MAX_LEN, truncation=True,
                               padding="max_length", return_tensors="pt")
                q  = tokenizer(ex["question"],
                               max_length=Q_MAX_LEN, truncation=True,
                               padding="max_length", return_tensors="pt")
                self.samples.append({
                    "qc_ids":  qc["input_ids"].squeeze(0),
                    "qc_mask": qc["attention_mask"].squeeze(0),
                    "q_ids":   q["input_ids"].squeeze(0),
                    "q_mask":  q["attention_mask"].squeeze(0),
                    "label":   float(c["sufficient"]),
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ---------------------------------------------------------------------------
# Model — delta construction
# ---------------------------------------------------------------------------
class SufficiencyProbeV2(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0, :]

    def forward(self, qc_ids, qc_mask, q_ids, q_mask):
        delta = self.encode(qc_ids, qc_mask) - self.encode(q_ids, q_mask)
        return self.classifier(delta).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def collate(batch):
    return {
        "qc_ids":  torch.stack([b["qc_ids"]  for b in batch]),
        "qc_mask": torch.stack([b["qc_mask"] for b in batch]),
        "q_ids":   torch.stack([b["q_ids"]   for b in batch]),
        "q_mask":  torch.stack([b["q_mask"]  for b in batch]),
        "labels":  torch.tensor([b["label"]  for b in batch], dtype=torch.float),
    }


def collate_pairs(batch):
    return {k: torch.stack([b[k] for b in batch])
            for k in ("q_ids", "q_mask", "s_ids", "s_mask", "i_ids", "i_mask")}


def evaluate_auroc(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["qc_ids"].to(device), batch["qc_mask"].to(device),
                           batch["q_ids"].to(device),  batch["q_mask"].to(device))
            logits_all.extend(logits.cpu().tolist())
            labels_all.extend(batch["labels"].tolist())
    scores = torch.sigmoid(torch.tensor(logits_all)).numpy()
    return float(roc_auc_score(np.array(labels_all), scores)), scores, np.array(labels_all)


def find_best_threshold(scores, labels):
    """Find threshold maximising balanced accuracy on val set."""
    best_t, best_ba = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tpr = tp / (tp + fn + 1e-9)
        tnr = tn / (tn + fp + 1e-9)
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    train_ds = CoveragePairDataset(DIVERSE_DIR / "train.json", tok, TRAIN_SOURCES)
    val_ds   = DiverseDataset(DIVERSE_DIR / "val.json",   tok, TRAIN_SOURCES)
    sb_val   = SBDataset(SB_DIR / "val.json",  tok)
    sb_test  = SBDataset(SB_DIR / "test.json", tok)
    # SciQ is an optional post-training OOD gate (not used for training or the main
    # SB AUROC). Skip gracefully if its dataset hasn't been regenerated yet.
    sciq_ds  = SciQDataset(SCIQ_PATH, tok) if SCIQ_PATH.exists() else None
    print(f"  Train (coverage pairs): {len(train_ds)}  Val (coverage flat): {len(val_ds)}")
    print(f"  SB val: {len(sb_val)}  SB test: {len(sb_test)}  "
          f"SciQ: {len(sciq_ds) if sciq_ds is not None else 'MISSING (skipped)'}")

    kw = dict(batch_size=BATCH_SIZE, collate_fn=collate, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE,
                              collate_fn=collate_pairs, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    sb_val_loader  = DataLoader(sb_val,  shuffle=False, **kw)
    sb_test_loader = DataLoader(sb_test, shuffle=False, **kw)
    sciq_loader    = DataLoader(sciq_ds, shuffle=False, **kw) if sciq_ds is not None else None

    print("Loading model...")
    model = SufficiencyProbeV2(MODEL_NAME).to(device)

    n_steps    = len(train_loader) * EPOCHS
    n_warmup   = int(n_steps * WARMUP_RATIO)
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler  = get_linear_schedule_with_warmup(optimizer, n_warmup, n_steps)

    best_val_auroc, patience_count, best_epoch = 0.0, 0, 0
    log = []

    print(f"\nTraining: {EPOCHS} epochs, {len(train_loader)} steps/epoch  "
          f"(contrastive pairwise + BCE, bf16 AMP)")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, pair_losses, bce_losses = time.time(), [], []
        for batch in train_loader:
            optimizer.zero_grad()
            q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
            s_ids, s_mask = batch["s_ids"].to(device), batch["s_mask"].to(device)
            i_ids, i_mask = batch["i_ids"].to(device), batch["i_mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                score_s = model(s_ids, s_mask, q_ids, q_mask)   # want high
                score_i = model(i_ids, i_mask, q_ids, q_mask)   # want low
                # Pairwise ranking loss (matched suf/insuf share the same question)
                L_pair = -nn.functional.logsigmoid(score_s - score_i).mean()
                # BCE anchors the absolute scale so sigmoid(score) is calibrated
                L_bce = (nn.functional.binary_cross_entropy_with_logits(
                            score_s, torch.ones_like(score_s))
                       + nn.functional.binary_cross_entropy_with_logits(
                            score_i, torch.zeros_like(score_i)))
                loss = L_pair + L_bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            pair_losses.append(L_pair.item()); bce_losses.append(L_bce.item())

        val_auroc, _, _ = evaluate_auroc(model, val_loader, device)
        elapsed = (time.time() - t0) / 60
        pl, bl = float(np.mean(pair_losses)), float(np.mean(bce_losses))
        print(f"Epoch {epoch:2d}  pair={pl:.4f}  bce={bl:.4f}  "
              f"val_auroc={val_auroc:.4f}  ({elapsed:.1f}min)", flush=True)
        log.append({"epoch": epoch, "pair_loss": pl, "bce_loss": bl,
                    "val_auroc": val_auroc, "elapsed_min": elapsed})

        if val_auroc > best_val_auroc:
            best_val_auroc, best_epoch, patience_count = val_auroc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_auroc": val_auroc, "epoch": epoch},
                       OUT_DIR / "best_checkpoint.pt")
            print(f"  → New best: {val_auroc:.4f}", flush=True)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})", flush=True)
                break

    # Load best
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val_auroc={best_val_auroc:.4f})...")
    ckpt = torch.load(OUT_DIR / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # --- Threshold selection on SufficiencyBench val ---
    print("\nSelecting threshold on SufficiencyBench val...")
    sb_val_auroc, sb_val_scores, sb_val_labels = evaluate_auroc(model, sb_val_loader, device)
    best_t, best_ba = find_best_threshold(sb_val_scores, sb_val_labels)
    print(f"  SB val AUROC: {sb_val_auroc:.4f}  best_threshold={best_t:.3f}  balanced_acc={best_ba:.4f}")

    # --- SufficiencyBench test ---
    print("\nEvaluating on SufficiencyBench test...")
    sb_auroc, sb_scores, sb_labels = evaluate_auroc(model, sb_test_loader, device)

    pk0 = [(s, l, ex) for s, l, ex in zip(sb_scores, sb_labels, sb_test.samples) if not ex["pk"]]
    pk1 = [(s, l, ex) for s, l, ex in zip(sb_scores, sb_labels, sb_test.samples) if ex["pk"]]
    auroc_pk0 = roc_auc_score([l for _, l, _ in pk0], [s for s, _, _ in pk0]) if pk0 else float("nan")
    auroc_pk1 = roc_auc_score([l for _, l, _ in pk1], [s for s, _, _ in pk1]) if pk1 else float("nan")

    # Per question type
    per_type = {}
    for qt in ["factual", "multi_hop", "comparative", "subjective"]:
        mask = [ex["qtype"] == qt for ex in sb_test.samples]
        if sum(mask) > 10:
            s_qt = [s for s, m in zip(sb_scores, mask) if m]
            l_qt = [l for l, m in zip(sb_labels, mask) if m]
            per_type[qt] = {"auroc": float(roc_auc_score(l_qt, s_qt)), "n": sum(mask)}

    # Downstream gating accuracy at best threshold and t=0.5
    def gate_acc(scores, labels, threshold):
        preds = (np.array(scores) >= threshold).astype(int)
        return float(np.mean(preds == np.array(labels)))

    gate_t_best = gate_acc(sb_scores, sb_labels, best_t)
    gate_t05    = gate_acc(sb_scores, sb_labels, 0.5)

    # PK-aware gate: if PK=1 always predict sufficient; else use probe
    pk_aware_preds = np.array([
        1.0 if ex["pk"] else float(sb_scores[i] >= best_t)
        for i, ex in enumerate(sb_test.samples)
    ])
    pk_aware_acc = float(np.mean(pk_aware_preds == sb_labels))

    print(f"  Overall AUROC: {sb_auroc:.4f}  (n={len(sb_scores)})")
    print(f"  PK=0 AUROC:   {auroc_pk0:.4f}  (n={len(pk0)})")
    print(f"  PK=1 AUROC:   {auroc_pk1:.4f}  (n={len(pk1)})")
    print(f"  Per type: {per_type}")
    print(f"  Gate acc (t=0.5): {gate_t05:.4f}  (t={best_t:.3f}): {gate_t_best:.4f}  PK-aware: {pk_aware_acc:.4f}")

    # --- SciQ (optional OOD gate) ---
    sciq_block = None
    if sciq_loader is not None:
        print("\nEvaluating on SciQ...")
        sciq_auroc, sciq_scores, sciq_labels = evaluate_auroc(model, sciq_loader, device)
        sciq_gate = gate_acc(sciq_scores, sciq_labels, best_t)
        print(f"  SciQ AUROC: {sciq_auroc:.4f}  gate acc (t={best_t:.3f}): {sciq_gate:.4f}")
        sciq_block = {"auroc": sciq_auroc, "gate_acc_best_t": sciq_gate, "n": len(sciq_scores)}
    else:
        print("\nSciQ eval skipped (sufficiency_bench_sciq/test.json not present).")

    results = {
        "training": {
            "sources": sorted(TRAIN_SOURCES),
            "n_train": len(train_ds), "n_val": len(val_ds),
            "best_epoch": best_epoch, "val_auroc_best": best_val_auroc,
            "log": log,
        },
        "threshold": {
            "selected_on": "sufficiencybench_val",
            "value": best_t,
            "balanced_accuracy": best_ba,
            "sb_val_auroc": sb_val_auroc,
        },
        "sufficiencybench_test": {
            "auroc_overall": sb_auroc,
            "auroc_pk0": auroc_pk0,
            "auroc_pk1": auroc_pk1,
            "per_type": per_type,
            "gate_acc_t05": gate_t05,
            "gate_acc_best_t": gate_t_best,
            "gate_acc_pk_aware": pk_aware_acc,
            "n": len(sb_scores),
        },
        "sciq": sciq_block,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUT_DIR / 'results.json'}")
    print(f"Checkpoint:    {OUT_DIR / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
