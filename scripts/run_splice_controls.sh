#!/bin/bash
# Splice controls driver. Usage: scripts/run_splice_controls.sh [LIMIT]   (LIMIT = smoke test, e.g. 30)
# Build + eval are CPU; extraction stages go through the thermal guard on cuda:0 (never cuda:1).
set -u
ROOT=/home/inab/Documents/MSD_Project/paper1; cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/inab/miniconda3/envs/csp/bin/python
LIM=${1:-}; LA=""; SF=""; [ -n "$LIM" ] && { LA="--limit $LIM"; SF="_limit$LIM"; }
$PY src/evaluation/splice_controls.py build $LA || exit 1
for M in llama mistral qwen; do
  scripts/run_methods_guarded.sh splice$SF 0 \
    "ctxonly_${M}$SF|data/experiments/hidden_states_context_only/${M}$SF/test/h_context_only.npy|src/evaluation/splice_controls.py extract --model $M --target context_only $LA" \
    "splice_${M}$SF|data/experiments/hidden_states_splice_control/${M}$SF/test/h_with_context.npy|src/evaluation/splice_controls.py extract --model $M --target splice $LA"
done
$PY src/evaluation/splice_controls.py eval $LA
