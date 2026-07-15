#!/bin/bash
# Wait for the current dev/030 cosine-head series (run by chain_028_to_030.sh) to fully
# finish, then run the cosine + Muon + flat_cos variant to get Muon's *final* accuracy
# (the original Muon run NaN'd at epoch 1; the fp32-head fix makes it stable now).
# Launched detached; logs to logs/030_muon_after_series.log.
cd /home/au761367/codes/lepinet
PY=.venv/bin/python

echo "=== queue started $(date); waiting for current dev/030 series to finish ==="
while pgrep -f "bash bash/chain_028_to_030.sh" >/dev/null; do sleep 120; done
while pgrep -f "\.venv/bin/python dev/030_hierarchical" >/dev/null; do sleep 120; done
echo "=== series done $(date); starting cosine+Muon+flat_cos run ==="

$PY dev/030_hierarchical_heads_benchmark.py --config configs/20260712_heads_global_independent_muon.yaml
echo "=== MUON_RUN_DONE $(date) ==="
