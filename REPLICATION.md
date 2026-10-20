# Replication guide — SufficiencyBench

Consolidated, ordered runbook to regenerate every artifact of the SufficiencyBench
paper from the downloaded models and source datasets.

---

## 0. Prerequisites (once)

```bash
# Python env (Python 3.11)
PY=~/miniconda3/envs/csp/bin/python          # or: conda activate csp
# Always export before running anything:
cd <repo-root>
export PYTHONPATH=$PWD:$PWD/src
```

**Models** (downloaded once) live under
`data/models/{llama-3.1-8b-instruct, mistral-7b-instruct, qwen2.5-7b-instruct, minilm}`;
32B/72B variants stream from `~/.cache/huggingface/hub/`. All paths resolve through
`configs/paths.py` — never hardcode.

**Source datasets** (downloaded once): `data/datasets/{squad_v2, hotpotqa,
eli5, trivia_qa, sciq}`. ELI5 uses the mirror `sentence-transformers/eli5` (canonical
HF repo removed) — see the data-loss ceilings at the end.

**HF eval models pulled on demand:** `answerdotai/ModernBERT-large`,
`lytang/MiniCheck-Flan-T5-Large`, `microsoft/deberta-large-mnli` (semantic entropy).

### 🔴 Thermal safety — READ FIRST

This machine has marginal cooling and has thermally shut down before. The durable
fix is a **kernel CPU cap on the user app slice** — full procedure in
**`scripts/THERMAL_SAFETY.md`**. Summary:

- Default working cap: `systemctl --user set-property app.slice CPUQuota=200%`
  (light jobs ~79 °C). Verify:
  `cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/cpu.max`
  → `200000 100000`.
- **For 7B-model GENERATION steps** (steering, baselines, PK caches, application):
  a single core can spike to ~89 °C at 200%. First **lower the cap to 90%**
  (`systemctl --user set-property app.slice CPUQuota=90%`, ≈76 °C), run the step
  while watching `sensors` (keep CPU < 86 °C), then **restore 200%**.
- Launch long jobs detached so the shell/harness cannot reap them:
  `setsid nohup <cmd> > run_logs/<name>.log 2>&1 &`.
- `day0_reconstruct.sh` has a built-in pause-first thermal watchdog
  (pause 83 °C CPU / 76 °C GPU) and caps `OMP_NUM_THREADS=4`.

Two RTX 5090s are available (`cuda:0`, `cuda:1`); all steps below default to `cuda:0`.

---

## Pipeline

Run from the repository root with `PYTHONPATH` exported. **Steps 1–4** are wrapped by the
resumable, thermally-guarded orchestrator:

```bash
setsid nohup bash day0_reconstruct.sh > run_logs/day0.out 2>&1 </dev/null & disown
# watch: run_logs/day0_master.log (stages) and run_logs/day0_*.temps (thermal)
```

or run them individually as below.

### 1. Build the benchmark (GPU; closed-book PK on ~3.5k Qs — long)
```bash
$PY src/data/build_sufficiency_bench.py --model llama --device cuda:0
# -> data/experiments/sufficiency_bench/{train,val,test}.json, metadata.json, pk_cache_llama.json
#    2,780 questions / 5,560 conditions, 1390/278/1112 split (Gate 0a).
```

### Version 1.1 (whitespace normalisation)

The benchmark used in the ACM TOIS paper is **version 1.1**, produced from the steps-1 output (v1.0) by
`python scripts/build_v11.py`: all whitespace runs in every context are collapsed to one space in both conditions,
prompts are rebuilt, and 95 whitespace-only pairs are dropped (1,352 / 264 / 1,069 train / val / test questions,
2,685 in total). Run it before step 3 (hidden-state extraction). The released copy is in
`datasets_for_github/sufficiencybench/`; v1.0 is kept in `datasets_for_github/sufficiencybench_v1.0/`.
PK rates in the paper use the first-sentence rule: `python scripts/recompute_pk_v2.py` (writes `results/pk_v2/`).
Paired bootstrap statistics: `python scripts/paired_stats_v11.py` (writes `results/paired_stats_v11.{json,md}`).

### 2. Per-model PK caches (GPU generation — use 90% cap)
```bash
$PY scripts/build_pk_cache_safe.py --model mistral --device cuda:0
$PY scripts/build_pk_cache_safe.py --model qwen   --device cuda:0
# -> data/experiments/sufficiency_bench/pk_cache_{mistral,qwen}.json  (closed-book "knows")
```

### 3. Extract hidden states (GPU; ~5.5 GB/model)
```bash
for m in llama mistral qwen; do
  $PY src/extraction/extract_states.py --model $m --device cuda:0
done
# -> data/experiments/hidden_states/<model>/<split>/{h_with_context,h_question_only,labels}.npy
#    Stored layer index i = output of decoder block i (outputs.hidden_states[1:]).
```

### 4. CSP/DECO + Standard probes (CPU sklearn — hottest step; keep OMP_NUM_THREADS≤4)
```bash
for m in llama mistral qwen; do
  for clf in logreg neural; do
    $PY src/methods/deco.py --model $m --clf $clf
  done
done
# -> results/<model>/all_results_{logreg,neural}.json
#    (layer_sweep, best_layers, best_overall, best_per_quadrant, best_per_question_type)
#    Gate 0b: all probe variants ~0.93-0.94 AUROC across models.
```

### 5. The other 10 Table-1 methods (GPU generation — use 90% cap)
```bash
for m in llama mistral qwen; do
  $PY src/methods/baselines.py            --model $m --device cuda:0   # VC, token entropy, generation match
  $PY src/methods/contrastive_baselines.py --model $m --device cuda:0  # Token Prob Delta
  $PY src/methods/redeep.py               --model $m --device cuda:0   # ReDeEP ECS/PKS/Combined (qwen loads bf16)
  $PY src/methods/semantic_entropy.py     --model $m --device cuda:0 --k 10
  $PY src/methods/llm_judge_baseline.py   --model $m --device cuda:0   # LLM Self-Judge
done
# -> results/<model>/{baseline_results,contrastive_baselines,redeep_results,
#                     semantic_entropy_results,llm_judge_results}.json
```

### 6. Assemble Table 1 & compare to paper
```bash
$PY scripts/assemble_table1.py       # per-method mine-vs-paper with PASS/FLAG
# See results/REPRODUCTION_NOTES.md for the reproduced-vs-approximate verdicts.
```

### 7. Per-example scores, VC, and PK-stratified AUROC
```bash
for m in llama mistral qwen; do
  $PY src/analysis/store_baseline_scores.py --model $m --with-baselines --device cuda:0  # results/<model>/scores/*.json
  $PY src/analysis/compute_vc_scores.py     --model $m --device cuda:0
done
$PY src/evaluation/build_pk_stratified_auroc.py   # -> results/pk_stratified_auroc.json
```

### 8. Deployable ModernBERT SufficiencyProbe (GPU — use 90% cap)
```bash
$PY src/evaluation/train_sufficiencyprobe_v2.py    # -> results/sufficiencyprobe_v2/ (target AUROC ~0.942, thr 0.42)
```

### 9. SciQ out-of-distribution gate
```bash
$PY src/data/build_sufficiency_bench_sciq.py       # -> data/experiments/sufficiency_bench_sciq/test.json (824 Q / 1648 conditions)
$PY src/evaluation/eval_sciq_ood.py --device cuda:0 # -> results/sciq_ood_eval.json (SciQ OOD AUROC ~0.954)
```

### 10. Activation-steering causal validation (GPU generation — 🔴 REQUIRES 90% CAP)
```bash
systemctl --user set-property app.slice CPUQuota=90%
export PYTHONPATH=$PWD:$PWD/src
setsid nohup $PY scripts/run_steering_mistral.py --n-per-quadrant 50 --batch-size 20 --max-new 60 \
  > run_logs/steer.log 2>&1 &
# monitor: sensors (CPU<86C), nvidia-smi
# -> results/steering_mistral.json
systemctl --user set-property app.slice CPUQuota=200%   # RESTORE when done
```
Injects `alpha * w_hat` (unit CSP-probe coefficient at layer 12, oriented toward the
sufficient pole) into Mistral's residual stream over decoder blocks 10–14, for
`alpha in {-2,-1,0,+1,+2}`, on a quadrant-balanced 200-example held-out set (PK from
`pk_cache_mistral`). Magnitude is calibrated to the live L12 residual norm
(`inject = alpha * 0.15 * ||h|| * w_hat`; calibration sweep in
`scripts/_calibrate_steer.py`). Result reproduces the paper's graded, symmetric,
collapse-at-`|alpha|=2` pattern (see figure note below).

### 11. Figures
```bash
# 4 lost figures reconstructed here (main comparison, layer curve, qtype breakdown,
# construction & methods schematics, steering) — writes BOTH results/figures/ and
# results/figures/paper/ with the exact tex filenames:
$PY scripts/generate_missing_figures.py
#   fig_main_comparison, fig_layer_curve, fig_qtype_breakdown (+results/per_qtype_auroc.json),
#   fig_construction_example, fig_methods_overview, fig_steering_mistral

# framework / overview / frontier figures:
$PY src/figures/generate_paper_figures.py     # delegates to the three below + per-qtype
#   (or individually: generate_framework_figure.py, generate_overview_figure.py, generate_frontier_figure.py)
```

### 12. Retrieved-context experiment (BM25 over pooled HotpotQA paragraphs)
```bash
$PY src/evaluation/retrieved_context_eval.py build --tag bm25_k5          # builds the retrieved contexts
bash scripts/run_retrieved_ctx.sh llama 0 bm25_k5                         # MODEL GPUIDX [TAG]: extract train, extract test, analyse
```
Run the second command for each model (llama, mistral, qwen) on a free GPU; it is resumable.
Sufficiency is labelled by gold-support containment; retriever-score predictors are the baselines.

### 13. Long-context extension
```bash
bash scripts/run_longcontext.sh 0        # GPUIDX; builds the haystacks, extracts states and VC at 4k/16k/32k (100k for llama/qwen), evaluates
```
Code in `src/longcontext/`; outputs `results/longcontext/<model>/summary.json`, `results/longcontext/layer_curves.json` and `fig_longcontext.pdf`.

### 14. Additional models (Qwen3 8B; Qwen 2.5 14B and DeepSeek-R1-Distill 32B)
```bash
bash scripts/run_qwen3_pipeline.sh cuda:0                    # -> results/qwen3/ (needs transformers>=4.56; see the script header)
$PY src/methods/vc_thinking_qwen3.py                          # thinking-mode VC variants (see the file header)
$PY src/methods/summarise_vc_thinking.py                      # -> results/qwen3/vc_thinking_summary.json
bash scripts/run_v11_scale.sh 0 qwen14 deepseek32             # -> results/{qwen14,deepseek32}/
$PY src/experiments/eval_opus_thinking.py --key-file <path>   # frontier VC with extended thinking; the key file is never printed
```

### 15. Other internal-signal and LLM-free baselines
```bash
bash scripts/run_internal_baselines.sh 0                                  # INSIDE/EigenScore and Lookback Lens -> results/<model>/internal_baselines.json
$PY src/methods/answerability_baselines.py --device cuda:0 --batch-size 16   # -> results/answerability_baselines.json
```

### 16. Review-round controls (CPU unless noted)
```bash
$PY scripts/retrieved_ctx_paired_vs_baselines.py            # paired tests, probes vs E5 cosine and vs the retriever-score regression -> results/retrieved_ctx_paired_vs_baselines.{json,md}
$PY src/evaluation/qpp_predictors.py                          # WIG, NQC, SMV, clarity, sigma_max from the Lucene index -> results/retrieved_ctx_qpp_proper.{json,md}
$PY scripts/dimension_matched_probes.py                       # probes on PCA / random projections of the stored states -> results/dimension_matched_probes.{json,md}
$PY scripts/supervision_matched_check.py                      # leave-one-source-out transfer (Table 8) and the original trained output readout -> results/supervision_matched_check.json
$PY scripts/supervision_matched_check_v2.py                   # trained output readout incl. the chat-template and P(Yes) readouts (Table 6) -> results/supervision_matched_check_v2.{json,md}
bash scripts/run_sprobe_retrieved.sh                          # GPU: text-only SufficiencyProbe transferred to and retrained on the retrieved contexts -> results/sufficiencyprobe_retrieved/
bash scripts/run_output_readouts.sh                           # GPU: chat-template and logit-based (P(Yes)) output readouts -> results/<model>/output_readouts.json, results/output_readouts_summary.md
bash scripts/run_splice_controls.sh                           # GPU: context-only probe and spliced-but-sufficient control -> results/splice_controls.{json,md}
$PY src/evaluation/splice_by_source.py; $PY src/evaluation/splice_hotpot_strata.py   # per-source breakdown and HotpotQA supporting-fact strata -> results/splice_controls_by_source.{json,md}
$PY src/evaluation/splice_controls.py build --split train; ... extract --model llama --split train --target splice; $PY src/evaluation/splice_augmented_probe.py   # GPU: Llama probes retrained with edited sufficient contexts -> results/splice_augmented_probe.{json,md}
bash scripts/run_docswap_control.sh                           # GPU: document-swap counterpart of the test split -> results/docswap_control.{json,md}
$PY src/evaluation/retrieved_context_eval.py build --retriever e5 --k 5 --tag e5_k5   # dense-retriever contexts; then run_retrieved_ctx.sh <model> <gpu> e5_k5, retrieved_ctx_paired_vs_baselines.py --tag e5_k5, retrieved_ctx_e5_summary.py -> results/retrieved_ctx_e5_summary.md
$PY src/evaluation/retrieved_context_eval.py build --k 3 --tag bm25_k3; ... extract --model llama --split test --tag bm25_k3; $PY scripts/retrieved_ctx_k_transfer.py --model llama --tags bm25_k3 bm25_k5 bm25_k10   # retrieval-depth sensitivity, transferred probes -> results/retrieved_ctx_k_sensitivity.{json,md}
$PY scripts/retrieved_ctx_length_only.py   # length-only control on BM25 and E5 top-5 contexts (CPU) -> results/retrieved_ctx_length_only.{json,md}
```

---

## Documented data-loss ceilings (from `results/REPRODUCTION_NOTES.md`)

The reconstruction rebuilt everything from re-downloaded models/datasets after a
delete. One ceiling is known and expected — it does **not** change any paper
conclusion:

- **Qwen ELI5 mirror.** The subjective quadrant was rebuilt from
  `sentence-transformers/eli5` (canonical repo removed from HF), so it may not
  byte-match the original. This is the likely cause of Qwen's probe AUROC sitting
  ~0.012–0.017 below the paper (0.934 vs 0.946) while Llama/Mistral match tightly.
  Factual/multi-hop quadrants are exact.

Other faithful-but-approximate reimplementations (ReDeEP head/FFN selection +
RAGTruth regression omitted; Semantic Entropy estimator; Token-Prob-Delta readout;
batched inference) are catalogued in `results/REPRODUCTION_NOTES.md`. All remain far
below the probe ceiling, so the central claim — **hidden-state probes above every output-based and attention-based method on every model** — reproduces cleanly.
