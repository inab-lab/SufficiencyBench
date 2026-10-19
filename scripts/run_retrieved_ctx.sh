#!/bin/bash
# run_retrieved_ctx.sh -- retrieved-context sufficiency evaluation for ONE model on ONE GPU.
# Runs: extract train -> extract test -> analyse  (src/evaluation/retrieved_context_eval.py).
# Resume-safe: extract checkpoints every 200 records and skips splits whose final outputs exist;
# analyse is re-run (CPU, cheap) every time.
# Prerequisite: `retrieved_context_eval.py build --tag <TAG>` has written data/experiments/retrieved_ctx/<TAG>/.
# Usage: scripts/run_retrieved_ctx.sh MODEL GPUIDX [TAG]     e.g. scripts/run_retrieved_ctx.sh llama 0 bm25_k5
set -euo pipefail
MODEL=${1:?usage: run_retrieved_ctx.sh MODEL GPUIDX [TAG]}
GPUIDX=${2:?usage: run_retrieved_ctx.sh MODEL GPUIDX [TAG]}
TAG=${3:-bm25_k5}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
PY=$HOME/miniconda3/envs/csp/bin/python
S=src/evaluation/retrieved_context_eval.py
mkdir -p run_logs; LOG=run_logs/retrieved_ctx_${MODEL}.log
log(){ echo "$(date '+%F %T') [retrieved_ctx:$MODEL:$TAG] $*" | tee -a "$LOG"; }
[ -s "data/experiments/retrieved_ctx/$TAG/test.json" ] || { log "missing data/experiments/retrieved_ctx/$TAG/test.json -- run build first"; exit 1; }
log "START on cuda:$GPUIDX"
for SPLIT in train test; do
  log "extract $SPLIT"
  "$PY" "$S" extract --model "$MODEL" --split "$SPLIT" --device "cuda:$GPUIDX" --tag "$TAG" >>"$LOG" 2>&1
done
log "analyse"
"$PY" "$S" analyse --model "$MODEL" --tag "$TAG" >>"$LOG" 2>&1
log "DONE -> results/$MODEL/retrieved_ctx_$TAG/analysis.{json,md}"
