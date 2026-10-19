#!/bin/bash
# run_longcontext.sh — driver for the SufficiencyBench long-context extension.
#
#   build manifest  ->  for model in llama qwen mistral: extract, then VC
#                   ->  evaluate (tables + layer curves + figure)
#
# Everything is sequential on a SINGLE GPU (default cuda:1) and every stage is
# internally resumable, so this script is safe to re-run after an abort: finished
# lengths carry a .done marker and are skipped, partially finished ones resume
# from their .npz / .json checkpoint.
#
# Thermal policy is NOT reimplemented here — the GPU stages are handed to
# scripts/run_methods_guarded.sh (pause at GPU>=76C / CPU>=83C, resume at
# <=64/68, abort at >=90/97, 30-min duty cycle), and the two CPU-only stages run
# under scripts/run_capped.sh (kernel-enforced CPUQuota).  The two mechanisms are
# mutually exclusive: run_capped.sh moves the process into its own systemd scope,
# which would break run_methods_guarded.sh's SIGSTOP/SIGCONT process-group
# control, so no stage uses both.
#
# Usage:
#   scripts/run_longcontext.sh            # cuda:1
#   scripts/run_longcontext.sh cuda:0     # or just: scripts/run_longcontext.sh 0
#   MODELS="llama" scripts/run_longcontext.sh
set -u
ROOT=/home/inab/Documents/MSD_Project/paper1
cd "$ROOT"

DEV_ARG=${1:-cuda:1}
GPUIDX=${DEV_ARG#cuda:}            # run_methods_guarded.sh wants the bare index
MODELS=${MODELS:-"llama qwen mistral"}

export PYTHONPATH="$ROOT:$ROOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=~/miniconda3/envs/csp/bin/python
LOGDIR="$ROOT/run_logs"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/longcontext_master.log"
MANIFEST="$ROOT/data/experiments/longcontext/manifest.json"

log(){ echo "$(date '+%F %T') [longcontext] $*" | tee -a "$MASTER"; }

log "===== RUN START on cuda:$GPUIDX  models=[$MODELS] ====="

# --- 1. manifest (CPU only, ~3 min, cheap to redo but skipped if present) ----
if [ -s "$MANIFEST" ]; then
  log "SKIP build (manifest exists: $MANIFEST)"
else
  log "START build"
  CAP_QUOTA=${CAP_QUOTA:-200} "$ROOT/scripts/run_capped.sh" \
    "$PY" src/longcontext/build_longcontext.py \
    >>"$LOGDIR/longcontext_build.log" 2>&1
  log "END build rc=$?"
fi

# --- 2. GPU stages, thermally guarded, sequential --------------------------
# outfile "-" = never skip at this level; each script does its own resume.
STAGES=()
# Time-boxed plan (single GPU, thermal duty-cycling): full 300 pairs up to 32k,
# a nested 150-pair subsample for extraction at 100k and 100 pairs for VC at 100k; 64k dropped.
for m in $MODELS; do
  STAGES+=("extract_${m}|-|src/longcontext/extract_longcontext.py --model $m --verify-hooks --lengths base 4000 16000 32000")
  STAGES+=("extract100k_${m}|-|src/longcontext/extract_longcontext.py --model $m --lengths 100000 --subsample 150")
  STAGES+=("vc_${m}|-|src/longcontext/vc_longcontext.py --model $m --validate-chunking --lengths base 4000 16000 32000")
  STAGES+=("vc100k_${m}|-|src/longcontext/vc_longcontext.py --model $m --lengths 100000 --subsample 100")
done

log "handing ${#STAGES[@]} GPU stages to run_methods_guarded.sh"
"$ROOT/scripts/run_methods_guarded.sh" longcontext "$GPUIDX" "${STAGES[@]}"
RC=$?
log "run_methods_guarded.sh rc=$RC"
if [ "$RC" -ne 0 ]; then
  log "ABORTED by thermal guard — not running evaluation; re-run this script to resume"
  exit "$RC"
fi

# --- 3. evaluation (CPU only: many logistic-regression fits) ----------------
log "START evaluate"
CAP_QUOTA=${CAP_QUOTA:-200} "$ROOT/scripts/run_capped.sh" \
  "$PY" src/longcontext/evaluate_longcontext.py \
  >>"$LOGDIR/longcontext_evaluate.log" 2>&1
log "END evaluate rc=$?"

log "===== RUN COMPLETE — see results/longcontext/ ====="
