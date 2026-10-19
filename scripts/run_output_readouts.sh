#!/bin/bash
# Output-based readouts (VC raw/template, P(Yes) raw/template, digit expectation) for 3 models,
# sequentially on cuda:0 through the thermal guard; then CPU-only bootstrap summary.
ROOT=/home/inab/Documents/MSD_Project/paper1
cd "$ROOT"
"$ROOT/scripts/run_methods_guarded.sh" ro 0 \
  "ro_llama|results/llama/output_readouts.json|src/methods/output_readouts.py --model llama --batch-size 8" \
  "ro_mistral|results/mistral/output_readouts.json|src/methods/output_readouts.py --model mistral --batch-size 8" \
  "ro_qwen|results/qwen/output_readouts.json|src/methods/output_readouts.py --model qwen --batch-size 8"
export PYTHONPATH="$ROOT:$ROOT/src"
$HOME/miniconda3/envs/csp/bin/python src/methods/output_readouts.py --summarize
