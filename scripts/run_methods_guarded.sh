#!/bin/bash
# Parametric thermal-guarded stage runner (reused across baselines / PK / probe phases).
# Usage:  run_methods_guarded.sh <STREAM_NAME> <GPUIDX> "name|outfile|cmd..." ["name|outfile|cmd..." ...]
#   - runs each stage sequentially on cuda:<GPUIDX>, under the same pause-first thermal
#     policy as day0_reconstruct.sh (monitors that GPU + the shared CPU).
#   - resumable: a stage whose <outfile> already exists (non-empty) is SKIPPED.
#   - outfile is relative to $ROOT; use "-" to always run (no skip).
set -u
ROOT=/home/inab/Documents/MSD_Project/MSD_Project
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/csp/bin/python
LOGDIR="$ROOT/run_logs"; mkdir -p "$LOGDIR"

STREAM=$1; GPUIDX=$2; shift 2
DEV="cuda:$GPUIDX"
MASTER="$LOGDIR/${STREAM}_master.log"

PAUSE_GPU=76; PAUSE_CPU=83; RESUME_GPU=64; RESUME_CPU=68
ABORT_GPU=90; ABORT_CPU=97; POLL=6; WORK_SECS=1800; REST_MIN=60; ABORT_STREAK=2

log(){ echo "$(date '+%F %T') [$STREAM] $*" | tee -a "$MASTER"; }
gpu_temp(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null; }
cpu_temp(){ sensors -j 2>/dev/null | $PY -c "import json,sys;d=json.load(sys.stdin);print(int(d['k10temp-pci-00c3']['Tctl']['temp1_input']))" 2>/dev/null; }
isnum(){ [[ "${1:-}" =~ ^[0-9]+$ ]]; }

CHILD=""
trap '[ -n "$CHILD" ] && kill -TERM -- -"$CHILD" 2>/dev/null; log "terminated"; exit 130' TERM INT

run_guarded(){ # $1=name $2...=cmd
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

log "===== STREAM START on $DEV ====="
for spec in "$@"; do
  IFS='|' read -r name outfile cmd <<< "$spec"
  if [ "$outfile" != "-" ] && [ -s "$ROOT/$outfile" ]; then
    log "SKIP $name ($outfile exists)"; continue
  fi
  # shellcheck disable=SC2086
  run_guarded "$name" $PY $cmd --device "$DEV" || log "$name FAILED rc=$? (continuing)"
done
log "===== STREAM COMPLETE on $DEV ====="
