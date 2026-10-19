#!/bin/bash
# Day-0 reconstruction pipeline for SufficiencyBench (Paper 1), thermally guarded.
# Regenerates dataset -> hidden states -> DECO probes, verifying paper gates.
# Sequential, single GPU (cuda:0). Temp-gated pause/resume + duty-cycle rests.
# Resumable: build skips via pk_cache, extract skips existing caches, deco is cheap.
set -u
ROOT=/home/inab/Documents/MSD_Project/MSD_Project
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src"
# Cap CPU parallelism to flatten the model-load thermal spikes (24-core weight
# loading hit 95C uncapped; capped peak ~74C). GPU work is unaffected.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/csp/bin/python
LOGDIR="$ROOT/run_logs"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/day0_master.log"
DEV=cuda:0; GPUIDX=0

# Pause-first thermal policy: pause+cool on high temp; ABORT only if temp stays
# critical WHILE PAUSED (no load) for ABORT_STREAK checks = true cooling failure.
PAUSE_GPU=76; PAUSE_CPU=83; RESUME_GPU=64; RESUME_CPU=68
ABORT_GPU=90; ABORT_CPU=97; POLL=6; WORK_SECS=1800; REST_MIN=60; ABORT_STREAK=2

log(){ echo "$(date '+%F %T') $*" | tee -a "$MASTER"; }
gpu_temp(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null; }
cpu_temp(){ sensors -j 2>/dev/null | $PY -c "import json,sys;d=json.load(sys.stdin);print(int(d['k10temp-pci-00c3']['Tctl']['temp1_input']))" 2>/dev/null; }
isnum(){ [[ "${1:-}" =~ ^[0-9]+$ ]]; }

CHILD=""
trap '[ -n "$CHILD" ] && kill -TERM -- -"$CHILD" 2>/dev/null; log "orchestrator terminated"; exit 130' TERM INT

run_guarded(){ # $1=name $2...=cmd
  local name=$1; shift
  local rlog="$LOGDIR/day0_$name.log"
  log "START $name : $*"
  setsid "$@" >>"$rlog" 2>&1 & local pid=$!; CHILD=$pid
  local work=0 paused=0 pstart=0 bad=0 astreak=0
  local tlog="$LOGDIR/day0_$name.temps"
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
      # Paused = no compute load. If STILL critical here, cooling itself is failing.
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

log "===== DAY-0 RECONSTRUCTION START ====="

# Stage 1: regenerate SufficiencyBench (long: closed-book PK on ~3500 Qs, llama)
if [ ! -f "$ROOT/data/experiments/sufficiency_bench/test.json" ]; then
  run_guarded build $PY src/data/build_sufficiency_bench.py --model llama --device $DEV || { log "build FAILED"; exit 1; }
else
  log "SKIP build (test.json exists)"
fi

# Stage 2: extract hidden states for all 3 models
for m in llama mistral qwen; do
  if [ ! -f "$ROOT/data/experiments/hidden_states/$m/test/h_with_context.npy" ]; then
    run_guarded extract_$m $PY src/extraction/extract_states.py --model $m --device $DEV || log "extract $m FAILED (continuing)"
  else log "SKIP extract $m"; fi
done

# Stage 3: DECO + Standard probes (logreg + neural) — cheap (sklearn)
for m in llama mistral qwen; do
  for clf in logreg neural; do
    run_guarded deco_${m}_${clf} $PY src/methods/deco.py --model $m --clf $clf || log "deco $m $clf FAILED (continuing)"
  done
done

log "===== DAY-0 CORE COMPLETE (dataset + states + probes) ====="
log "Next (not in this script): pk_caches for mistral/qwen, ModernBERT SufficiencyProbe, baselines, packaging."
