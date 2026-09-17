# SufficiencyBench

Main benchmark for context sufficiency detection in RAG systems.

## Overview

**2,780 questions** across train / val / test splits (1,390 / 278 / 1,112). Each example contains one question paired with two contexts: one **sufficient** (contains the answer) and one **insufficient** (the answer sentence is removed and replaced with a topically similar non-answering sentence from the same document). Sufficient and insufficient contexts are matched for length and drawn from the same source document, removing surface confounds present in natural RAG datasets.

## Question types

| Type | Description |
|---|---|
| `factual` | Single-hop, direct answer in one sentence |
| `multi_hop` | Requires chaining two supporting facts |
| `comparative` | Compares two entities on a shared attribute |
| `subjective` | Open-ended (ELI5-style) |

## File structure

```
sufficiencybench/
├── train/data.json      # 1,390 questions × 2 conditions = 2,780 conditions
├── val/data.json        # 278 questions × 2 conditions = 556 conditions
├── test/data.json       # 1,112 questions × 2 conditions = 2,224 conditions
├── metadata.json        # Dataset-level statistics
├── pk_cache_llama.json  # Closed-book (parametric knowledge) labels — Llama 3.1 8B
├── pk_cache_mistral.json  # Closed-book labels — Mistral 7B
└── pk_cache_qwen.json   # Closed-book labels — Qwen 2.5 7B
```

## Example record

```json
{
  "id": "hotpot_2300",
  "question": "What took place first, The Korean War or The Western Allied invasion of Germany?",
  "gold_answer": "The Western Allied invasion of Germany",
  "question_type": "comparative",
  "model_knows_answer": true,
  "pk_llama": true,
  "pk_qwen": false,
  "conditions": [
    {
      "condition_id": "hotpot_2300_suf",
      "context": "...",
      "sufficient": true,
      "quadrant": "Q1"
    },
    {
      "condition_id": "hotpot_2300_insuf",
      "context": "...",
      "sufficient": false,
      "quadrant": "Q2"
    }
  ]
}
```

## Quadrant labels

Each condition is assigned a quadrant based on sufficiency × parametric knowledge:

| Quadrant | Sufficient | Model knows answer |
|---|---|---|
| Q1 | Yes | Yes |
| Q2 | No | Yes |
| Q3 | Yes | No |
| Q4 | No | No |

## Sources

Questions are drawn from SQuAD 2.0 (factual), HotpotQA (multi-hop, comparative), and ELI5 (subjective). Insufficient contexts are created by surgical sentence removal from the same source document.

## Citation

If you use this dataset, please cite:

```bibtex
@article{llopis2025sufficiencybench,
  title={SufficiencyBench: A Confound-Controlled Benchmark for Disentangling Context Sufficiency from Parametric Knowledge in RAG},
  author={Llopis-Garc\'{i}a, Guillermo and del Rosario-Gilabert, David},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```
