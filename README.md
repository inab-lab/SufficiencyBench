# SufficiencyBench

**Will the Retrieved Context Suffice? Confound-Controlled Evaluation of Sufficiency Predictors for Retrieval-Augmented Generation**

Guillermo Llopis-García (INAB and University of Alicante), Daniel Ruiz-Fernández (University of Alicante), David del Rosario Gilabert (INAB, corresponding author).
Manuscript under review at ACM Transactions on Information Systems.

## Summary

A retrieval-augmented generation (RAG) system should answer only when the retrieved context suffices for the question, and abstain otherwise. Deciding this before generation is a performance-prediction problem at the generation stage, and the predictors in common use read the model's output (verbalized confidence, semantic entropy and their refinements). Evaluating them on natural retrieval data answers the wrong question, because two confounds inflate their scores: sufficient passages are systematically longer than insufficient ones, and a model that already knows the answer is confident whatever the context contains. SufficiencyBench is a 5,370-example benchmark in which each insufficient context is a minimally edited copy of its sufficient counterpart, matched in length, entity density and topic, with labels stratified by each model's own closed-book knowledge and validated by human annotation, a surface-feature classifier at chance and label-noise tests. On it we compare 16 predictors on three open-weight large language models, the probes and output-based predictors on three more, and the output-based predictors on two frontier API models, organised by where each reads from the model. No zero-shot output-based predictor exceeds 0.82 overall AUROC on any model, including two frontier API models, while a linear probe on intermediate hidden states reaches 0.90-0.95. We read the ceiling as an access limit: the output exposes less of the sufficiency signal than the residual stream carries. The probe's advantage is largest in distribution: it loses 0.03 to 0.10 AUROC on a held-out source and most of its margin when the passage is embedded in 4,000 to 100,000 tokens.

## Key results

Sufficiency prediction AUROC on the SufficiencyBench test split (Table 1 of the paper; see the paper for the 95% confidence intervals, the other 13 methods and the remaining models). The probes are the logistic-regression variants.

| Model | Standard probe | CSP probe | Verbalized confidence |
|---|---|---|---|
| Mistral 7B | 0.910 | 0.902 | 0.701 |
| Llama 3.1 8B | 0.907 | 0.904 | 0.719 |
| Qwen 2.5 7B | 0.904 | 0.914 | 0.782 |

On BM25-retrieved contexts (Section 5.3 of the paper; top-5 passages over the pooled HotpotQA paragraphs, 1,500 test questions, sufficiency = every gold-support paragraph retrieved), AUROC:

| Predictor | Llama 3.1 8B | Mistral 7B | Qwen 2.5 7B |
|---|---|---|---|
| Best retriever-score predictor (logistic regression over five BM25 features) | 0.671 | 0.671 | 0.671 |
| E5 cosine (question, context) | 0.712 | 0.712 | 0.712 |
| Verbalized confidence | 0.707 | 0.617 | 0.700 |
| Standard probe, transferred from SufficiencyBench without retraining | 0.807 | 0.773 | 0.811 |
| Standard probe, retrained on retrieved contexts | 0.936 | 0.916 | 0.936 |

Per-model details, label variants and the paired tests are in `results/<model>/retrieved_ctx_bm25_k5/analysis.md`.

The Standard probe reads the last-token hidden state of question plus context, h(q+c). The CSP (Context Subtraction Probe) reads h(q+c) - h(q). In the code, the CSP probe is called "DECO" (for example `src/methods/deco.py`); the names refer to the same method.

## Benchmark

SufficiencyBench version 1.1 has 2,685 questions, each with a sufficient and an insufficient context (5,370 examples), split over questions into 1,352 train, 264 validation and 1,069 test questions. The questions come from SQuAD v2 (705, factual), HotpotQA (1,480, multi-hop and comparative) and ELI5 (500, subjective). Each insufficient context is made by replacing the answer-bearing sentence of the gold passage with a topically similar, non-answering sentence of matched length. Parametric knowledge (PK) is tracked per model from a closed-book run, giving a 2x2 design: Q1 (sufficient, PK=1), Q2 (sufficient, PK=0), Q3 (insufficient, PK=1), Q4 (insufficient, PK=0).

Files, record format and notes are in [`datasets_for_github/sufficiencybench/`](datasets_for_github/sufficiencybench/). The superseded version 1.0, on which the frontier API runs in the paper were scored, is in [`datasets_for_github/sufficiencybench_v1.0/`](datasets_for_github/sufficiencybench_v1.0/).

## Repository layout

| Path | Content |
|---|---|
| `datasets_for_github/` | Released benchmark, version 1.1 (and the superseded version 1.0) |
| `src/data/` | Benchmark construction |
| `src/extraction/` | Hidden-state extraction |
| `src/methods/` | Probes (CSP/Standard), neural probe, baselines, ReDeEP, semantic entropy, LLM judge, activation steering |
| `src/analysis/`, `src/evaluation/` | Score storage, PK-stratified AUROC, retrieved-context experiment, probe training and evaluation |
| `src/longcontext/`, `src/figures/` | Long-context experiment and figure generation |
| `scripts/` | Pipeline drivers, statistics, figure and steering scripts |
| `results/` | Result files (small JSON/Markdown tables; per-model outputs under `results/<model>/`) |
| `paper/tois_v1_src/` | LaTeX source of the submitted manuscript |
| `configs/` | Paths and configuration |
| `REPLICATION.md` | Ordered runbook |

`data/` (hidden states and model-specific intermediate data) is not included in the repository; the released benchmark is under `datasets_for_github/`.

## Reproducing the paper

The ordered runbook is [`REPLICATION.md`](REPLICATION.md). Main entry points:

| Step | Entry point |
|---|---|
| Build the benchmark (v1.0 construction) | `src/data/build_sufficiency_bench.py` |
| Whitespace normalisation to v1.1 | `scripts/build_v11.py` |
| Extract hidden states | `src/extraction/extract_states.py` |
| Train and evaluate the probes (CSP and Standard) | `src/methods/deco.py` |
| Store per-example scores of the baselines | `src/analysis/store_baseline_scores.py` |
| Baseline predictors | `src/methods/baselines.py` |
| Question-clustered bootstrap and paired tests | `scripts/paired_stats_v11.py` |
| PK labels with the first-sentence rule | `scripts/recompute_pk_v2.py` |
| Retrieved-context experiment | `src/evaluation/retrieved_context_eval.py` with `scripts/run_retrieved_ctx.sh` |
| Activation steering | `scripts/run_steering_mistral.py`, `scripts/run_steering_random_control.py` |

The experiments need GPUs and the open-weight models listed in the paper.

## SufficiencyProbe checkpoints

The three SufficiencyProbe variants of the paper (ModernBERT-large encoder, linear head on the `[CLS]` difference between the question-plus-context and the question-only encodings; 1.58 GB each) are release assets of this repository: https://github.com/inab-lab/SufficiencyBench/releases/tag/v1.1

| File | Training pairs | Threshold (balanced accuracy on the SufficiencyBench validation split) | SufficiencyBench test AUROC | SHA-256 |
|---|---|---|---|---|
| `sufficiencyprobe_cross_source.pt` | TriviaQA + NQ coverage pairs (788) | 0.67 | 0.769 | `bb4b5eab61337a82a84f8b9fd0e73d110e4713ac4ee724af14eca82709b84cda` |
| `sufficiencyprobe_cross_dataset.pt` | + HotpotQA coverage pairs (1,715) | 0.12 | 0.870 | `174332fab15603828408bf81377c2f4340dd861ad43eb9a7fa1327db6814a547` |
| `sufficiencyprobe_in_distribution.pt` | + SufficiencyBench train/val (3,067) | 0.62 | 0.966 | `823c0f9022d7d4213492c0c92d204af72030774aef375a381f763b15a9efcc88` |

Training script: `src/evaluation/train_sufficiencyprobe_v2.py` (the cross-dataset variant is trained with `V2_EXCLUDE_SB=1`, the cross-source one with `V2_SOURCES=triviaqa,nq`). Loading and scoring one (question, context) pair:

```python
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SufficiencyProbeV2(nn.Module):               # same class as in train_sufficiencyprobe_v2.py
    def __init__(self, name="answerdotai/ModernBERT-large"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
    def encode(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
    def forward(self, qc_ids, qc_mask, q_ids, q_mask):
        return self.classifier(self.encode(qc_ids, qc_mask) - self.encode(q_ids, q_mask)).squeeze(-1)

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
model = SufficiencyProbeV2()
model.load_state_dict(torch.load("sufficiencyprobe_cross_source.pt", map_location="cpu")["model_state_dict"])
model.eval()

qc = tok(question, context, max_length=512, truncation=True, return_tensors="pt")
q  = tok(question, max_length=128, truncation=True, return_tensors="pt")
with torch.no_grad():
    p_sufficient = torch.sigmoid(model(qc["input_ids"], qc["attention_mask"], q["input_ids"], q["attention_mask"])).item()
answer = p_sufficient >= 0.67     # the variant's threshold from the table above
```

## Citation

```bibtex
@misc{llopis2026sufficiency,
  title  = {Will the Retrieved Context Suffice? Confound-Controlled Evaluation of Sufficiency Predictors for Retrieval-Augmented Generation},
  author = {Llopis-Garc{\'\i}a, Guillermo and Ruiz-Fern{\'a}ndez, Daniel and del Rosario Gilabert, David},
  year   = {2026},
  note   = {Manuscript under review at ACM Transactions on Information Systems},
  url    = {https://github.com/inab-lab/SufficiencyBench}
}
```

## License

MIT License, see [`LICENSE`](LICENSE).

## Acknowledgements

We thank the 11 annotators who took part in the benchmark validation study.
