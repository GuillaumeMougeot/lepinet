#!/bin/bash
# Improved global independent-head Muon run: short LR warmup + lighter (mini_trainer-style)
# augmentation + grad_clip 5.0. Aimed at closing the species macro-F1 gap to mini_trainer's
# own loop (~89.6%). Launched detached; logs to logs/031_muon_warmup.log.
# Then evaluates on the global test fold via dev/032 (uses the fixed known-species filter).
cd /home/au761367/codes/lepinet
PY=.venv/bin/python

echo "=== training started $(date) ==="
$PY dev/030_hierarchical_heads_benchmark.py --config configs/20260713_heads_global_independent_muon_warmup.yaml
echo "=== training done $(date) ==="
