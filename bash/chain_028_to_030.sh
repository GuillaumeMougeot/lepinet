#!/bin/bash
# Wait for the running dev/028 global training to finish, then run the dev/030 cosine-head
# series (independent -> hierarchical -> autoregressive) with the fp16 + fp32-head fix.
# Launched detached (nohup) so it survives independently. Logs to logs/030_after_028.log.
cd /home/au761367/codes/lepinet
PY=.venv/bin/python

echo "=== chain started $(date); waiting for dev/028 global to finish ==="
while pgrep -f "20260710_multihead_v4_global" >/dev/null; do sleep 120; done
echo "=== dev/028 finished $(date); starting dev/030 series ==="

for h in independent hierarchical autoregressive; do
  echo "===== GLOBAL HEAD: $h  $(date) ====="
  $PY dev/030_hierarchical_heads_benchmark.py --config configs/20260709_heads_global_$h.yaml
  echo "===== DONE HEAD: $h  $(date) ====="
done
echo "=== ALL_030_DONE $(date) ==="
