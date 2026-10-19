# SufficiencyBench — Reproduction Notes

Status of the Day-0 reconstruction's Table 1 (14 detection methods × 3 models),
comparing regenerated results to the paper's reported AUROC. Generated from
`scripts/assemble_table1.py` against `results/<model>/*.json`.

## Reproduced (the paper's core contribution)

| Method | mistral | llama | qwen | verdict |
|---|---|---|---|---|
| Standard Probe (LogReg) | 0.932 / 0.922 | 0.930 / 0.923 | 0.934 / 0.946 | ✓ |
| CSP Probe (LogReg) | 0.925 / 0.910 | 0.928 / 0.918 | 0.932 / 0.949 | ✓ |
| Standard Probe (Neural) | 0.937 / 0.931 | 0.943 / 0.928 | 0.943 / 0.953 | ✓ |
| CSP Probe (Neural) | 0.940 / 0.943 | 0.939 / 0.933 | 0.941 / 0.956 | ✓ |

The central claim reproduces cleanly: **all probe variants 0.93–0.94 across models;
every behavioural/mechanistic method ≤0.77** (the probe-over-behaviour ceiling).

Simple behavioural baselines (Verbalized Confidence, Token Entropy, Generation
Match, LLM Self-Judge) reproduce within ~0.05 of the paper.

## Known deviations — faithful-but-approximate reimplementations

These methods were reconstructed after the deletion and deviate from the paper's
exact numbers. They are documented rather than matched to the last digit, because
(a) all remain far below the probe (so the paper's conclusion is unaffected), and
(b) matching them exactly requires procedures the paper under-specifies.

- **ReDeEP (ECS / PKS / Combined).** Our implementation averages over *all* heads
  and layers; the paper first performs a **per-model copying-head + knowledge-FFN
  selection** and **regresses the ECS/PKS combination weights on RAGTruth**. Both
  are omitted here, so absolute AUROCs shift (qwen worst: 0.41–0.48). Reproducing
  the paper's values would require reimplementing that selection/regression.
  - *Qwen-specific fix applied:* Qwen2.5-7B has "massive activations" that overflow
    fp16 in the final decoder layer → inf → all-NaN scores. Fixed by loading qwen
    in **bf16** (`_MODEL_DTYPES` in `redeep.py`), consistent with the rest of the
    codebase; llama/mistral remain fp16.

- **Semantic Entropy.** Uses the discrete cluster-probability estimator
  (`p_c = |c|/K`, K=10) with DeBERTa-MNLI bidirectional-entailment clustering.
  Lands in the paper's ballpark (0.46–0.65 vs 0.52–0.56) but is sensitive to K,
  sampling temperature, and the estimator choice.

- **Token Prob Delta.** Our variant scores ~0.10 AUROC **higher** than the paper
  (0.73–0.77 vs 0.66–0.67) — i.e. it discriminates *better*, not a sign error.
  A definitional difference (answer-token log-prob averaging vs the paper's exact
  readout). Left as-is; it does not change the ranking.

## Other caveats

- **ELI5 mirror.** The subjective quadrant was rebuilt from
  `sentence-transformers/eli5` (the canonical HF repo was removed), so it may not
  byte-match the original; this is the likely cause of qwen's probe AUROCs sitting
  ~0.012–0.017 below the paper (0.934 vs 0.946) while llama/mistral match tightly.

- **Batched inference.** All generation baselines were re-engineered to batched
  inference for tractability. Greedy methods are bit-identical to per-item; the
  contrastive forward pass is exact in fp16 (2.6e-3); token-entropy differs ~0.025
  from fp16 GEMM non-associativity (does not move AUROC). `contrastive_baselines.py`
  was switched from 8-bit to fp16 (8-bit matmul is batch-size-sensitive).

---

## Finishing artifacts (reconstruction closeout, 2026-07-18)

- **SciQ OOD gate dataset** — `data/experiments/sufficiency_bench_sciq/test.json`
  (`src/data/build_sufficiency_bench_sciq.py`): 824 questions / 1648 matched
  conditions (824 sufficient = own support passage; 824 insufficient = topic-matched
  BM25 distractor support with the answer absent), in SufficiencyBench condition format.
  Evaluating the **existing ModernBERT v2 probe** on it (no retraining,
  `src/evaluation/eval_sciq_ood.py` → `results/sciq_ood_eval.json`):
  **SciQ OOD AUROC = 0.9539** (n=1648, gate acc 0.899 at t=0.760; SB-val AUROC 0.985).
  Strong cross-domain generalization (Wikipedia-trained probe on science textbook;
  cf. the CSP-probe reference of 0.824). This is the SciQ eval the trainer previously
  skipped for want of the dataset.

- **Open-weight PK-stratified AUROC** — `results/pk_stratified_auroc.json`
  (`src/evaluation/build_pk_stratified_auroc.py`): sufficiency AUROC on SufficiencyBench
  test, split by each generator's parametric knowledge (PK from
  `pk_cache_{model}.json`; per-example probe scores refit from cached hidden states via
  `store_baseline_scores.py`, per-example VC recomputed via `compute_vc_scores.py` and
  validated against the stored overall VC AUROC within 0.002).

  | model | PK=0 probe / vc (n) | PK=1 probe / vc (n) |
  |---|---|---|
  | llama | 0.929 / 0.709 (2174) | 0.920 / 0.740 (50) |
  | mistral | 0.919 / 0.692 (1838) | 0.956 / 0.715 (386) |
  | qwen | 0.930 / 0.775 (2134) | 0.974 / 0.758 (90) |

  The probe holds 0.92–0.97 AUROC in **both** PK strata and beats VC by +15–24 AUROC
  points throughout — sufficiency detection is not a proxy for parametric knowledge.

- **Figures regenerated** — `src/figures/generate_paper_figures.py` +
  `generate_{framework,frontier,overview}_figure.py` re-emit
  `results/figures/{fig1_framework,fig2_framework,fig_overview,fig_frontier}.{png,pdf}`
  (previous files were 0-byte placeholders). The per-question-type figure is still
  skipped (`per_qtype_auroc.json` not present); legacy 0-byte `fig1_overview.png` /
  `fig_main_comparison.png` / `fig_qtype_breakdown.png` are stale names from a prior
  scheme and are not produced by the current generators.

## Reproduction bug fixes applied

- `store_baseline_scores.py::load_best_layers` now reads best layers from
  `all_results_logreg.json` (fallback chain) — the legacy `deco_results.json` is
  absent in the reconstructed layout.

All jobs ran under the durable kernel CPU cap (app.slice CPUQuota=200%); peak CPU
package temp during GPU inference stayed ≤ ~84 °C (below the 86 °C guard).
