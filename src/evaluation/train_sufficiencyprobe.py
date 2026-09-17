"""
train_sufficiencyprobe.py  —  Train the deployable SufficiencyProbe on SufficiencyBench.

ModernBERT-large (395M) with BCE loss on the benchmark ground-truth labels.
Saves best checkpoint (by val AUROC) to results/sufficiencyprobe/best_checkpoint.pt.

Usage (from MSD_Project root):
  python src/evaluation/train_sufficiencyprobe.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR    = PROJECT_ROOT / "data" / "experiments" / "sufficiency_bench"
OUT_DIR      = PROJECT_ROOT / "results" / "sufficiencyprobe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "answerdotai/ModernBERT-large"
MAX_LEN     = 512
BATCH_SIZE  = 16
LR          = 2e-5
EPOCHS      = 10
PATIENCE    = 3
WARMUP_RATIO = 0.10

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SufficiencyDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path) as f:
            data = json.load(f)
        self.samples = []
        for ex in data:
            q = ex["question"]
            for cond in ex["conditions"]:
                enc = tokenizer(
                    q, cond["context"],
                    max_length=MAX_LEN,
                    truncation="only_second",
                    padding="max_length",
                    return_tensors="pt",
                )
                self.samples.append({
                    "input_ids":      enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label":          float(cond["sufficient"]),
                    "id":             cond["condition_id"],
                    "question_type":  ex.get("question_type", "unknown"),
                    "quadrant":       cond.get("quadrant", "?"),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SufficiencyProbe(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # CLS
        return self.classifier(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.tensor([b["label"] for b in batch], dtype=torch.float),
    }


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    scores = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    return float(roc_auc_score(labels, scores)), scores, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    train_ds = SufficiencyDataset(BENCH_DIR / "train.json", tokenizer)
    val_ds   = SufficiencyDataset(BENCH_DIR / "val.json",   tokenizer)
    test_ds  = SufficiencyDataset(BENCH_DIR / "test.json",  tokenizer)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=4, pin_memory=True)

    print("Loading model...")
    model = SufficiencyProbe(MODEL_NAME).to(device)

    n_steps        = len(train_loader) * EPOCHS
    n_warmup_steps = int(n_steps * WARMUP_RATIO)
    optimizer      = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler      = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, n_steps)
    criterion      = nn.BCEWithLogitsLoss()

    print(f"\nTraining: {EPOCHS} epochs, {len(train_loader)} steps/epoch")
    best_val_auroc  = 0.0
    patience_count  = 0
    best_epoch      = 0
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0     = time.time()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss   = criterion(logits, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        val_auroc, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        entry = {"epoch": epoch, "loss": float(np.mean(losses)),
                 "val_auroc": val_auroc, "elapsed_min": elapsed / 60}
        log.append(entry)
        print(f"Epoch {epoch:2d}  loss={np.mean(losses):.4f}  val_auroc={val_auroc:.4f}  ({elapsed/60:.1f}min)", flush=True)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch     = epoch
            patience_count = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_auroc": val_auroc, "epoch": epoch},
                       OUT_DIR / "best_checkpoint.pt")
            print(f"  → New best: {val_auroc:.4f}", flush=True)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})", flush=True)
                break

    # Load best and evaluate on test
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val_auroc={best_val_auroc:.4f})...")
    ckpt  = torch.load(OUT_DIR / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_auroc, scores, labels = evaluate(model, test_loader, device)
    print(f"Test AUROC (SufficiencyBench): {test_auroc:.4f}")

    # Per question-type
    qtypes = [s["question_type"] for s in test_ds.samples]
    per_type = {}
    for qt in ["factual", "multi_hop", "comparative", "subjective"]:
        mask = np.array([q == qt for q in qtypes])
        if mask.sum() > 10:
            a = float(roc_auc_score(labels[mask], scores[mask]))
            per_type[qt] = {"auroc": a, "n": int(mask.sum())}
            print(f"  {qt}: {a:.4f} (n={mask.sum()})")

    # Per quadrant
    quads = [s["quadrant"] for s in test_ds.samples]
    per_quad = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = np.array([qd == q for qd in quads])
        if mask.sum() > 10:
            per_quad[q] = {"auroc": float(roc_auc_score(labels[mask], scores[mask])),
                           "n": int(mask.sum())}

    # Gate accuracy (proxy for downstream)
    preds = (scores >= 0.5).astype(int)
    gate_acc = float(np.mean(preds == labels))
    print(f"Gate accuracy at t=0.5: {gate_acc:.4f}")

    # Save results + test scores
    results = {
        "model_name": MODEL_NAME,
        "training": {"lr": LR, "batch_size": BATCH_SIZE, "epochs_run": epoch,
                     "best_epoch": best_epoch, "warmup_ratio": WARMUP_RATIO,
                     "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)},
        "val_auroc_best": best_val_auroc,
        "test_auroc": test_auroc,
        "per_type": per_type,
        "per_quadrant": per_quad,
        "gate_accuracy_t05": gate_acc,
        "training_log": log,
    }
    with open(OUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUT_DIR / 'training_results.json'}")
    print(f"Checkpoint:    {OUT_DIR / 'best_checkpoint.pt'}")

    # Save per-example scores for downstream simulation
    test_scores = [
        {"id": s["id"], "score": float(sc), "label": int(lb),
         "question_type": s["question_type"], "quadrant": s["quadrant"]}
        for s, sc, lb in zip(test_ds.samples, scores, labels)
    ]
    with open(OUT_DIR / "test_scores.json", "w") as f:
        json.dump(test_scores, f, indent=2)
    print(f"Test scores:   {OUT_DIR / 'test_scores.json'}")


if __name__ == "__main__":
    main()
