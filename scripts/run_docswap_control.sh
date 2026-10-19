#!/bin/bash
# Document-swap control: build swap set (CPU), then thermal-guarded GPU stages (VC + hidden-state extraction) for
# llama/mistral/qwen, then CPU report. Usage: scripts/run_docswap_control.sh [LIMIT]   (LIMIT e.g. 30 for a smoke test)
set -u
ROOT=/home/inab/Documents/MSD_Project/paper1; cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/miniconda3/envs/csp/bin/python
LIMIT=${1:-}; LA=""; SF=""; [ -n "$LIMIT" ] && { LA="--limit $LIMIT"; SF="_limit$LIMIT"; }
S=src/evaluation/docswap_control.py
$PY $S build || exit 1
STAGES=()
for m in llama mistral qwen; do
  STAGES+=("vc_$m|results/docswap_control/vc_${m}${SF}.json|$S vc --model $m --batch-size 8 $LA")
  STAGES+=("ext_$m|data/experiments/hidden_states_docswap/$m/test${SF}/h_with_context.npy|$S extract --model $m $LA")
done
# GPU 0 is shared with other jobs: retry until every stage output exists, waiting for enough free memory first
# (8-bit hidden-state extraction needs ~11 GB, fp16 verbalized-confidence generation ~18 GB).
run_group() { # $1=min free MiB, rest = stages
  local need=$1; shift
  for try in $(seq 1 300); do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)" -lt "$need" ]; do sleep 30; done
    bash scripts/run_methods_guarded.sh "docswap${SF}" 0 "$@"
    local missing=0; for st in "$@"; do IFS='|' read -r _n o _c <<< "$st"; [ -s "$o" ] || missing=1; done
    [ "$missing" -eq 0 ] && return 0
  done
}
EXT=(); VCS=()
for st in "${STAGES[@]}"; do case "$st" in ext_*) EXT+=("$st");; *) VCS+=("$st");; esac; done
run_group 11000 "${EXT[@]}"
run_group 18000 "${VCS[@]}"
$PY $S report $LA
