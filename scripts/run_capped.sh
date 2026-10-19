#!/bin/bash
# run_capped.sh — run ANY command under a KERNEL-ENFORCED CPU cap (cgroup v2
# CPUQuota via systemd --user scope) so it physically cannot overheat this
# marginal-cooling machine. Unlike a watchdog process, the kernel cap cannot
# fail, die, or miss multiprocessing children — the whole scope is capped.
#
# Calibrated on this box: 200% -> ~79C, 300% -> ~82C (idle ~45-60C, danger ~95C).
# Default 200% = 2 cores of throughput with a wide safety margin.
#
# Usage:  run_capped.sh <cmd> [args...]
#         CAP_QUOTA=150 run_capped.sh <cmd> ...   # override quota (% of one core)
set -u
QUOTA=${CAP_QUOTA:-200}
NAME="capped-$(od -An -N2 -tu2 /dev/urandom 2>/dev/null | tr -d ' ' || echo job)"
exec systemd-run --user --scope --unit="$NAME" \
  -p CPUQuota=${QUOTA}% \
  -p MemoryMax=85G \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 --setenv=NUMEXPR_NUM_THREADS=4 \
  --setenv=PYTHONPATH="${PYTHONPATH:-}" \
  --setenv=HF_HUB_OFFLINE=1 --setenv=HF_DATASETS_OFFLINE=1 \
  --setenv=CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --setenv=TOKENIZERS_PARALLELISM=false \
  -- "$@"
