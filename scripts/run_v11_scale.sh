#!/bin/bash
# Scale tier for the v1.1 rerun: Qwen 2.5 14B (8-bit) and DeepSeek-R1-Distill-Qwen-32B (4-bit NF4),
# weights streamed from the Hugging Face cache. PK cache -> hidden states -> probes -> VC/entropy/match
# -> per-example scores. Guarded, resumable. Usage: scripts/run_v11_scale.sh [GPUIDX] [models...]
set -u
ROOT=/home/inab/Documents/MSD_Project/paper1; cd "$ROOT"
GPUIDX=${1:-0}; shift || true; MODELS=${*:-"qwen14 deepseek32"}
W="$ROOT/run_logs/v11"; mkdir -p "$W"
export PYTHONPATH="$ROOT:$ROOT/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/miniconda3/envs/csp/bin/python; G="$ROOT/scripts/run_methods_guarded.sh"
log(){ echo "$(date '+%F %T') [v11_scale] $*" | tee -a "$W/chain.log"; }
cpu(){ local name=$1 out=$2; shift 2; [ -s "$ROOT/$out" ] && { log "SKIP $name"; return 0; }; log "START $name"; taskset -c 0-5 nice -n 5 "$@" >>"$W/$name.log" 2>&1; log "END $name rc=$?"; }
for m in $MODELS; do
  "$G" v11_scale "$GPUIDX" \
    "pk_${m}|data/experiments/sufficiency_bench/pk_cache_${m}.json|scripts/build_pk_cache_safe.py --model $m" \
    "extract_${m}|data/experiments/hidden_states/${m}/test/h_with_context.npy|src/extraction/extract_states.py --model $m"
  cpu deco_${m}_logreg results/$m/all_results_logreg.json $PY src/methods/deco.py --model $m --clf logreg
  cpu deco_${m}_neural results/$m/all_results_neural.json $PY src/methods/deco.py --model $m --clf neural
  cpu scores_${m} results/$m/scores/probe_scores.json $PY src/analysis/store_baseline_scores.py --model $m
  "$G" v11_scale "$GPUIDX" \
    "bl_${m}_base|results/${m}/baseline_results.json|src/methods/baselines.py --model $m --batch-size 8" \
    "vcscores_${m}|results/${m}/scores/baseline_scores.json|src/analysis/compute_vc_scores.py --model $m"
  [ -s results/$m/all_results_logreg.json ] && [ -s results/$m/baseline_results.json ] && echo "- $(date '+%F %T')  Scale tier: $m done (probes + output baselines + PK)" >> "$W/PROGRESS.md"
done
