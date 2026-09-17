# Replication guide — SufficiencyBench

Ordered runbook to regenerate every artifact of the paper from the source datasets and
open-weight models. Run everything from the repository root.

## 0. Prerequisites (once)

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/src                        # export before running anything
PY=python
```

**Models:** `python scripts/download_models.py` fetches Llama 3.1 8B Instruct, Mistral 7B
Instruct v0.3, Qwen 2.5 7B Instruct and MiniLM-L6 into `data/models/`; the 32B/72B variants
stream from the Hugging Face cache. All paths resolve through `configs/paths.py`.

**Source datasets:** `python scripts/download_datasets.py` fetches SQuAD v2, HotpotQA, ELI5
(mirror `sentence-transformers/eli5`; the canonical HF repo was removed), TriviaQA, NQ and SciQ
into `data/datasets/`. The released benchmark in `datasets_for_github/sufficiencybench/`
already contains the final splits, so step 1 is only needed to rebuild it from scratch.

**Evaluation models pulled on demand:** `answerdotai/ModernBERT-large`,
`lytang/MiniCheck-Flan-T5-Large`, `microsoft/deberta-large-mnli` (semantic entropy).

**Hardware:** hidden-state extraction and generation-based baselines need one GPU with
~16 GB for the 7B–8B models in 8-bit (~5.5 GB of activations per model on disk); probes
train on CPU with scikit-learn.

## SufficiencyBench pipeline
### 1. Build the benchmark (GPU; closed-book PK on ~3.5k Qs — long)
```bash
$PY src/data/build_sufficiency_bench.py --model llama --device cuda:0
# -> data/experiments/sufficiency_bench/{train,val,test}.json, metadata.json, pk_cache_llama.json
#    2,780 questions / 5,560 conditions, 1390/278/1112 split (Gate 0a).
```

### 2. Per-model PK caches (GPU generation)
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

### 5. The other 10 Table-1 methods (GPU generation)
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

### 8. Deployable ModernBERT SufficiencyProbe (GPU)
```bash
$PY src/evaluation/train_sufficiencyprobe_v2.py    # -> results/sufficiencyprobe_v2/ (target AUROC ~0.942, thr 0.42)
```

### 9. SciQ out-of-distribution gate
```bash
$PY src/data/build_sufficiency_bench_sciq.py       # -> data/experiments/sufficiency_bench_sciq/test.json (824 Q / 1648 conditions)
$PY src/evaluation/eval_sciq_ood.py --device cuda:0 # -> results/sciq_ood_eval.json (SciQ OOD AUROC ~0.954)
```

### 10. Activation-steering causal validation (GPU generation)
```bash
export PYTHONPATH=$PWD:$PWD/src
setsid nohup $PY scripts/run_steering_mistral.py --n-per-quadrant 50 --batch-size 20 --max-new 60 \
  > run_logs/steer.log 2>&1 &
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
# paper/sufficiencybench_overleaf_v10/figures/ with the exact tex filenames:
$PY scripts/generate_missing_figures.py
#   fig_main_comparison, fig_layer_curve, fig_qtype_breakdown (+results/per_qtype_auroc.json),
#   fig_construction_example, fig_methods_overview, fig_steering_mistral

# framework / overview / frontier figures:
$PY src/figures/generate_paper_figures.py     # delegates to the three below + per-qtype
#   (or individually: generate_framework_figure.py, generate_overview_figure.py, generate_frontier_figure.py)
```

---


