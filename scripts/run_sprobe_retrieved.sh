#!/bin/bash
# Thermal-guarded SufficiencyProbe baseline on retrieved context (GPU 1). See src/evaluation/train_sufficiencyprobe_retrieved.py
cd /home/inab/Documents/MSD_Project/paper1
S=src/evaluation/train_sufficiencyprobe_retrieved.py; R=results/sufficiencyprobe_retrieved
exec scripts/run_methods_guarded.sh SPROBE_RC 1 \
  "transfer|$R/transfer_scores.json|$S transfer" \
  "train512|$R/train_len512.json|$S train --max_len 512" \
  "train1024|$R/train_len1024.json|$S train --max_len 1024 --micro 8" \
  "final|$R/summary.md|$S final"
