"""
run_sprobe_v2_sim.py

Score the 500 downstream-simulation examples with SufficiencyProbe v2
and compute gating accuracy. Reuses existing Mistral answers from
results/downstream_simulation/sim_checkpoint.jsonl.
"""

import json, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

CKPT_PATH   = RESULTS_DIR / "sufficiencyprobe_v2" / "best_checkpoint.pt"
SIM_CKPT    = RESULTS_DIR / "downstream_simulation" / "sim_checkpoint.jsonl"
OUT_PATH    = RESULTS_DIR / "downstream_simulation" / "sprobe_v2_sim_results.json"
MODEL_NAME  = "answerdotai/ModernBERT-large"
MAX_LEN     = 512
Q_MAX_LEN   = 128
BATCH_SIZE  = 16
THRESHOLD   = 0.42   # selected on SufficiencyBench val


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


class SimDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.samples = []
        for ex in examples:
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
                "condition_id": ex["condition_id"],
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate(batch):
    return {
        "qc_ids":  torch.stack([b["qc_ids"]  for b in batch]),
        "qc_mask": torch.stack([b["qc_mask"] for b in batch]),
        "q_ids":   torch.stack([b["q_ids"]   for b in batch]),
        "q_mask":  torch.stack([b["q_mask"]  for b in batch]),
        "condition_ids": [b["condition_id"] for b in batch],
    }


def simulate_gate(sim_examples, scores, threshold):
    correct = []
    for ex in sim_examples:
        gate = scores[ex["condition_id"]] >= threshold
        if gate:
            correct.append(ex["correct_vanilla"])
        else:
            correct.append(not ex["sufficient"])
    return float(np.mean(correct))


def find_optimal_threshold(sim_examples, scores):
    best_t, best_acc = THRESHOLD, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        acc = simulate_gate(sim_examples, scores, t)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sim checkpoint (500 examples with Mistral answers)
    sim_lookup = {}
    with open(SIM_CKPT) as f:
        for line in f:
            rec = json.loads(line)
            sim_lookup[rec["condition_id"]] = rec
    print(f"Loaded {len(sim_lookup)} checkpointed examples")

    # Load test.json to get questions and contexts
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    examples = []
    for item in test_data:
        for cond in item["conditions"]:
            cid = cond["condition_id"]
            if cid in sim_lookup:
                examples.append({
                    "condition_id": cid,
                    "question":     item["question"],
                    "context":      cond["context"],
                    "sufficient":   cond["sufficient"],
                    "correct_vanilla": sim_lookup[cid]["correct_vanilla"],
                    "quadrant":     cond["quadrant"],
                })
    print(f"Matched {len(examples)} examples from test.json")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading SufficiencyProbe v2...")
    model = SufficiencyProbeV2(MODEL_NAME).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Score all examples
    ds = SimDataset(examples, tok)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate, num_workers=4, pin_memory=True)

    scores = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["qc_ids"].to(device), batch["qc_mask"].to(device),
                           batch["q_ids"].to(device),  batch["q_mask"].to(device))
            probs = torch.sigmoid(logits).cpu().tolist()
            for cid, p in zip(batch["condition_ids"], probs):
                scores[cid] = p

    print(f"Scored {len(scores)} examples")

    # Simulate gate
    acc_val_t  = simulate_gate(examples, scores, THRESHOLD)
    acc_half   = simulate_gate(examples, scores, 0.5)
    opt_t, opt_acc = find_optimal_threshold(examples, scores)

    # No-gate and oracle for reference
    no_gate = float(np.mean([ex["correct_vanilla"] for ex in examples]))
    oracle  = float(np.mean([
        ex["correct_vanilla"] if ex["sufficient"] else True
        for ex in examples
    ]))

    results = {
        "n": len(examples),
        "threshold_val_selected": THRESHOLD,
        "strategies": {
            "No gate":                          round(no_gate, 4),
            "SufficiencyProbe v2 (t=0.42)":    round(acc_val_t, 4),
            "SufficiencyProbe v2 (t=0.5)":     round(acc_half, 4),
            "SufficiencyProbe v2 (optimal)":   round(opt_acc, 4),
            "Oracle gate":                      round(oracle, 4),
        },
        "optimal_threshold": round(opt_t, 4),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Results ===")
    for k, v in results["strategies"].items():
        print(f"  {k:<40} {v:.4f}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
