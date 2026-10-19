#!/bin/bash
# run_internal_baselines.sh — driver for the two INTERNAL-SIGNAL baselines
# (INSIDE/EigenScore, Lookback Lens) added in src/methods/internal_baselines.py.
#
# Sequential, single GPU (default cuda:1), one stage per model
# (llama, qwen, mistral -- qwen3 is not part of the default sweep; run it
# manually with `--model qwen3` if needed). Thermal policy is handled by
# scripts/run_methods_guarded.sh (pause at GPU>=76C / CPU>=83C, resume at
# <=64/68, abort at >=90/97, 30-min duty cycle) -- not reimplemented here.
#
# Resumability:
#   - Each model's stage is skipped by run_methods_guarded.sh if its output
#     results/<model>/internal_baselines.json already exists (non-empty).
#   - Within a stage, Lookback Lens feature extraction is itself resumable:
#     features are cached to data/experiments/lookback/<model>/<split>_
#     features.npz per-condition and flushed every 50 items (see
#     internal_baselines.py:extract_lookback_split), so a stage killed by the
#     thermal guard mid-extraction picks up where it left off on re-run.
#     INSIDE has no such cache (cheap relative to lookback) and restarts from
#     scratch if a stage is interrupted before completion.
#
# Usage:
#   scripts/run_internal_baselines.sh            # cuda:1
#   scripts/run_internal_baselines.sh cuda:0     # or just: scripts/run_internal_baselines.sh 0
#   MODELS="llama" scripts/run_internal_baselines.sh
set -u
ROOT=/home/inab/Documents/MSD_Project/paper1
cd "$ROOT"

DEV_ARG=${1:-cuda:1}
GPUIDX=${DEV_ARG#cuda:}            # run_methods_guarded.sh wants the bare index
MODELS=${MODELS:-"llama qwen mistral"}
BATCH_SIZE=${BATCH_SIZE:-4}
TRAIN_SUBSET=${TRAIN_SUBSET:-800}

export PYTHONPATH="$ROOT:$ROOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR="$ROOT/run_logs"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/internal_baselines_master.log"
log(){ echo "$(date '+%F %T') [internal_baselines] $*" | tee -a "$MASTER"; }

log "===== RUN START on cuda:$GPUIDX  models=[$MODELS]  batch_size=$BATCH_SIZE train_subset=$TRAIN_SUBSET ====="

SPECS=()
for m in $MODELS; do
  SPECS+=("internal_${m}|results/${m}/internal_baselines.json|src/methods/internal_baselines.py --model $m --methods inside lookback --splits train test --batch-size $BATCH_SIZE --train-subset $TRAIN_SUBSET")
done

log "handing ${#SPECS[@]} GPU stage(s) to run_methods_guarded.sh"
"$ROOT/scripts/run_methods_guarded.sh" internal_baselines "$GPUIDX" "${SPECS[@]}"
RC=$?
log "run_methods_guarded.sh rc=$RC"
log "===== RUN COMPLETE — see results/<model>/internal_baselines.json ====="
exit "$RC"
