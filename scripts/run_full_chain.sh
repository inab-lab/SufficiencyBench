#!/bin/bash
# =============================================================================
# NEURIPS CHAIN v2: Sequential, robust execution
# Runs ReDeEP (E1) and cross-model validation (E4) one at a time.
# Safe to re-run — skips completed steps.
# =============================================================================

set -o pipefail
# Resolve project root as the parent of this script's directory (scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "NEURIPS CHAIN v2 started at $(date)"
echo "=============================================="

check_done() {
    if [ -f "$1" ] && [ -s "$1" ]; then
        echo "  SKIP: $1 exists ($(wc -c < "$1") bytes)"
        return 0
    fi
    return 1
}

# ── PHASE 1: ReDeEP baselines (sequential) ──
for MODEL in llama mistral qwen; do
    echo ""
    echo "[$(date '+%H:%M')] ReDeEP — $MODEL"
    if check_done "results/$MODEL/redeep_results.json"; then
        continue
    fi
    echo "  Running..."
    python src/methods/redeep_baseline.py --model $MODEL --device cuda:0 \
        2>&1 | tee results/$MODEL/redeep_chain.log
    echo "  Done at $(date '+%H:%M')"
done

# ── PHASE 2: Cross-model validation (sequential) ──
for MODEL in llama mistral qwen; do
    echo ""
    echo "[$(date '+%H:%M')] Cross-validation — $MODEL"
    if check_done "results/validation/cross_validation_$MODEL.json"; then
        continue
    fi
    echo "  Running..."
    python scripts/cross_model_validation.py --model $MODEL --device cuda:0 --n_examples 500 \
        2>&1 | tee results/validation/cross_val_${MODEL}_v2.log
    echo "  Done at $(date '+%H:%M')"
done

# ── Aggregate ──
echo ""
echo "[$(date '+%H:%M')] Aggregating..."
python scripts/cross_model_validation.py --aggregate 2>&1

echo ""
echo "=============================================="
echo "NEURIPS CHAIN v2 completed at $(date)"
echo "=============================================="
