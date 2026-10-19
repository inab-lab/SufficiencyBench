#!/bin/bash
# run_qwen3_pipeline.sh — resumable, thermal-guarded Qwen3-8B pipeline.
#
# Mirrors the pause/resume/abort thermal-guard pattern in
# scripts/run_methods_guarded.sh (same PAUSE/RESUME/ABORT thresholds and
# SIGSTOP/SIGCONT duty-cycling), but does NOT source that script: each stage
# below carries its own full command line instead of having "--device"
# force-appended, because deco.py is CPU-only sklearn and takes no --device
# flag. Runs with the csp_qwen3 conda env's python (transformers>=4.56) since
# the base csp env's transformers (4.48.3) predates the Qwen3 architecture.
#
# Order: PK cache -> extract_states (train/val/test, one model load)
#        -> deco.py (logreg, neural) -> baselines.py (VC + token-entropy +
#        generation-match, all scored in one call).
#
# Usage:
#   scripts/run_qwen3_pipeline.sh [GPU_DEVICE]     # default cuda:0
#
# Resumable: a stage whose declared output file already exists (non-empty) is
# skipped; "-" means the stage always runs (build_pk_cache_safe.py and
# extract_states.py do their own internal/whole-run resume bookkeeping).
set -u
ROOT=/home/inab/Documents/MSD_Project/paper1
cd "$ROOT"
DEV=${1:-cuda:0}
GPUIDX=${DEV#cuda:}

export PYTHONPATH="$ROOT:$ROOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/inab/miniconda3/envs/csp_qwen3/bin/python
LOGDIR="$ROOT/run_logs"; mkdir -p "$LOGDIR"
STREAM=qwen3
MASTER="$LOGDIR/${STREAM}_master.log"

PAUSE_GPU=76; PAUSE_CPU=83; RESUME_GPU=64; RESUME_CPU=68
ABORT_GPU=90; ABORT_CPU=97; POLL=6; WORK_SECS=1800; REST_MIN=60; ABORT_STREAK=2

log(){ echo "$(date '+%F %T') [$STREAM] $*" | tee -a "$MASTER"; }
gpu_temp(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null; }
cpu_temp(){ sensors -j 2>/dev/null | $PY -c "import json,sys;d=json.load(sys.stdin);print(int(d['k10temp-pci-00c3']['Tctl']['temp1_input']))" 2>/dev/null; }
isnum(){ [[ "${1:-}" =~ ^[0-9]+$ ]]; }

CHILD=""
trap '[ -n "$CHILD" ] && kill -TERM -- -"$CHILD" 2>/dev/null; log "terminated"; exit 130' TERM INT

run_guarded(){ # $1=name, $2...=full argv (already includes the interpreter)
  local name=$1; shift
  local rlog="$LOGDIR/${STREAM}_$name.log"
  log "START $name : $*"
  setsid "$@" >>"$rlog" 2>&1 & local pid=$!; CHILD=$pid
  local work=0 paused=0 pstart=0 bad=0 astreak=0
  local tlog="$LOGDIR/${STREAM}_$name.temps"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL"
    local gt ct; gt=$(gpu_temp "$GPUIDX"); ct=$(cpu_temp)
    if ! isnum "$gt" || ! isnum "$ct"; then bad=$((bad+1)); [ "$bad" -ge 3 ] && [ "$paused" -eq 0 ] && { kill -STOP -- -"$pid" 2>/dev/null; paused=1; pstart=$(date +%s); log "$name PAUSED (sensors unreadable)"; }; continue; fi
    bad=0
    echo "$(date +%T) gpu=${gt} cpu=${ct} paused=$paused" >> "$tlog"
    if [ "$paused" -eq 0 ]; then
      work=$((work+POLL))
      if [ "$gt" -ge "$PAUSE_GPU" ] || [ "$ct" -ge "$PAUSE_CPU" ] || [ "$work" -ge "$WORK_SECS" ]; then
        local why="duty-rest"; [ "$gt" -ge "$PAUSE_GPU" ] && why="gpu ${gt}C"; [ "$ct" -ge "$PAUSE_CPU" ] && why="cpu ${ct}C"
        kill -STOP -- -"$pid" 2>/dev/null; paused=1; pstart=$(date +%s); astreak=0; log "$name PAUSED ($why)"
      fi
    else
      if [ "$gt" -ge "$ABORT_GPU" ] || [ "$ct" -ge "$ABORT_CPU" ]; then
        astreak=$((astreak+1)); log "$name CRITICAL-while-paused gpu=${gt} cpu=${ct} (streak $astreak/$ABORT_STREAK)"
        if [ "$astreak" -ge "$ABORT_STREAK" ]; then log "$name ABORT — cooling failing under no load; killing"; kill -TERM -- -"$pid" 2>/dev/null; exit 1; fi
      else
        astreak=0
        local rested=$(( $(date +%s) - pstart ))
        if [ "$rested" -ge "$REST_MIN" ] && [ "$gt" -le "$RESUME_GPU" ] && [ "$ct" -le "$RESUME_CPU" ]; then
          kill -CONT -- -"$pid" 2>/dev/null; paused=0; work=0; log "$name RESUMED (${rested}s, gpu=${gt} cpu=${ct})"
        fi
      fi
    fi
  done
  wait "$pid"; local rc=$?; CHILD=""; log "END $name rc=$rc"; return $rc
}

log "===== QWEN3 PIPELINE START on $DEV ====="

# name|outfile(relative to ROOT, "-"=always run)|full command (interpreter omitted)
SPECS=(
  "pk_qwen3|-|scripts/build_pk_cache_safe.py --model qwen3 --device $DEV"
  "extract_qwen3|data/experiments/hidden_states/qwen3/test/h_with_context.npy|src/extraction/extract_states.py --model qwen3 --device $DEV"
  "deco_qwen3_logreg|results/qwen3/all_results.json|src/methods/deco.py --model qwen3 --clf logreg"
  "deco_qwen3_neural|results/qwen3/all_results_neural.json|src/methods/deco.py --model qwen3 --clf neural"
  "baselines_qwen3|results/qwen3/baseline_results.json|src/methods/baselines.py --model qwen3 --device $DEV --batch-size 16"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name outfile cmd <<< "$spec"
  if [ "$outfile" != "-" ] && [ -s "$ROOT/$outfile" ]; then
    log "SKIP $name ($outfile exists)"; continue
  fi
  # shellcheck disable=SC2086
  run_guarded "$name" $PY $cmd || log "$name FAILED rc=$? (continuing)"
done

log "===== QWEN3 PIPELINE COMPLETE ====="
