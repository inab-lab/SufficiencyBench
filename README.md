# SufficiencyBench

**A confound-controlled benchmark and probes for context sufficiency detection in retrieval-augmented generation (RAG).**

This repository accompanies the manuscript *"Hidden-state probes surpass an output-level ceiling in context sufficiency detection for retrieval-augmented generation"* (Llopis, Ruiz-Fernández, del Rosario Gilabert; under review, 2026).

## What is in the paper

RAG systems must detect when the retrieved context is insufficient to answer a query. Output-based detectors (verbalized confidence, semantic entropy, token entropy, attention decomposition, LLM self-judge, ...) report strong scores on existing benchmarks, but those benchmarks contain two structural confounds: sufficient contexts are longer and more entity-dense, and parametric knowledge co-varies with sufficiency.

SufficiencyBench removes both confounds by construction. On it:

- every output-based method plateaus at **0.66–0.81 AUROC** across five open-weight models (7B–72B) and two frontier API models;
- a linear probe on intermediate hidden states reaches **0.91–0.97 AUROC**;
- the signal peaks at 28–54% of network depth and is overwritten in later layers (an *access limit*, not a calibration failure);
- a standalone 395M-parameter **SufficiencyProbe** (ModernBERT-large) reaches 0.754 AUROC at 7.3 ms with no LLM in the loop.

## The benchmark

`datasets_for_github/sufficiencybench/`

| Split | Questions | Examples (sufficient + insufficient) |
|---|---|---|
| train | 1,390 | 2,780 |
| val | 278 | 556 |
| test | 1,112 | 2,224 |

Each question appears with two contexts: the gold passage (**sufficient**) and a copy in which the answer-bearing sentence(s) are surgically replaced by a length-matched, topically similar non-answering sentence (**insufficient**). Pairs are matched in length (ratio 0.990 ± 0.045), entity density and topic. Sources: SQuAD v2 (factual), HotpotQA (multi-hop, comparative), ELI5 (subjective). Both conditions of a question are always in the same split.

Parametric-knowledge (PK) labels from closed-book evaluation are provided per model in `pk_cache_{llama,mistral,qwen}.json`, enabling the 2×2 sufficiency × PK stratification used in the paper. See `datasets_for_github/sufficiencybench/README.md` for the record format.

## Code map

| Paper component | Code |
|---|---|
| Benchmark construction | `src/data/build_sufficiency_bench.py` |
| Hidden-state extraction | `src/extraction/extract_states.py` |
| Standard and CSP probes (CSP = Context Subtraction Probe; historical name in code: `deco`) | `src/methods/deco.py`, `src/methods/neural_probe.py` |
| Output-based baselines (verbalized confidence, token entropy, generation match, token-prob delta) | `src/methods/baselines.py`, `src/methods/contrastive_baselines.py` |
| Semantic entropy | `src/methods/semantic_entropy.py` |
| ReDeEP (attention-based) | `src/methods/redeep.py`, `src/methods/redeep_baseline.py` |
| LLM self-judge, embedding similarity | `src/methods/llm_judge_baseline.py`, `src/methods/baselines.py` |
| Frontier API models (verbalized confidence, CoT) | `src/experiments/eval_new_frontier.py`, `src/methods/api_cot_test.py` |
| PK-stratified AUROC, per-type and per-quadrant analysis | `src/evaluation/build_pk_stratified_auroc.py`, `scripts/quadrant_analysis.py` |
| Layer curve | `src/analysis/layer_curve_analysis.py` |
| SufficiencyProbe (ModernBERT) training and evaluation | `src/evaluation/train_sufficiencyprobe.py`, `src/evaluation/eval_sufficiencyprobe.py` |
| Downstream gating simulation | `scripts/run_downstream_sim.py` |
| Bootstrap CIs, Table 1 assembly | `scripts/bootstrap_ci.py`, `scripts/assemble_table1.py` |
| Figures | `src/figures/`, `src/evaluation/generate_figures.py` |

`REPLICATION.md` walks through the full pipeline in order (benchmark build → PK caches → hidden states → probes → baselines → tables). Hidden-state extraction needs a GPU with ~16 GB for the 7B–8B models in 8-bit; probes train on CPU.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_datasets.py      # SQuAD v2, HotpotQA, ELI5 (only needed to rebuild the benchmark)
python scripts/download_models.py        # open-weight models from the Hugging Face Hub
```

Frontier-model scripts read `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment.

## Quick start: train and evaluate the CSP probe

```bash
python src/extraction/extract_states.py --model mistral --device cuda:0
python src/methods/deco.py --model mistral            # Standard + CSP probes, layer sweep
python src/methods/baselines.py --model mistral       # output-based baselines
python scripts/assemble_table1.py                     # Table 1
```

## Citation

```bibtex
@unpublished{llopis2026sufficiency,
  title  = {Hidden-state probes surpass an output-level ceiling in context sufficiency detection for retrieval-augmented generation},
  author = {Llopis, Guillermo and Ruiz-Fern{\'a}ndez, Daniel and del Rosario Gilabert, David},
  note   = {Under review},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`). The benchmark is derived from SQuAD v2 (CC BY-SA 4.0), HotpotQA (CC BY-SA 4.0) and ELI5 (Reddit data, released under the terms of the original ELI5 dataset); those licenses apply to the corresponding text.
