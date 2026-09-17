#!/usr/bin/env bash
# Creates the PUBLIC inab-lab/SufficiencyBench repo from this clean release folder.
# Run ONLY after David approves. Steps:
#   1. rename the current private mirror out of the way (keeps history, stays private)
#   2. create a fresh public repo with the paper's URL and push this folder
set -euo pipefail
cd "$(dirname "$0")"
gh repo rename SufficiencyBench-internal --repo inab-lab/SufficiencyBench --yes
git init -q && git add -A && git commit -q -m "SufficiencyBench public release (code, benchmark data, replication guide)"
gh repo create inab-lab/SufficiencyBench --public --source=. --remote=origin --push \
  --description "SufficiencyBench: confound-controlled benchmark and probes for context sufficiency detection in RAG"
echo "Public repo: https://github.com/inab-lab/SufficiencyBench"
