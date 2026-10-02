#!/usr/bin/env bash
# C2b: the one decisive extra arch. tf_efficientnetv2_b2 (8.7 M) is the matched-size effnet to
# compare against fastvit_sa12 (10.6 M, test 0.8920) -- if a *smaller* effnet matches the
# transformer, effnets win outright and the transformer line is dropped (user's call).
# Waits for the C2 sweep to finish so it doesn't contend for the GPU, then trains + tests.
#   setsid nohup dev/run_c2b_effnetv2b2.sh >data/global/models/c2_logs/c2b.log 2>&1 </dev/null &
set -u
REPO=/home/au761367/codes/lepinet
PY=$REPO/.venv/bin/python
LOG=$REPO/data/global/models/c2_logs
C2LOG=$LOG/sweep.log
cd "$REPO" || exit 1

echo "=== $(date -Is) waiting for C2 sweep to finish ==="
while ! grep -qa "C2 sweep done; ledger" "$C2LOG" 2>/dev/null; do sleep 300; done
echo "=== $(date -Is) C2 done; train effnetv2b2 ==="
"$PY" "$REPO/dev/030_hierarchical_heads_benchmark.py" \
    --config "$REPO/configs/20260722_backbone_effnetv2b2.yaml" > "$LOG/train_effnetv2b2.log" 2>&1
echo "    train exit=$? $(date -Is)"
"$PY" "$REPO/dev/032_hierarchical_heads_test.py" \
    --config "$REPO/configs/20260722_test_backbone_effnetv2b2.yaml" > "$LOG/test_effnetv2b2.log" 2>&1
echo "    test exit=$? $(date -Is)"
"$PY" "$REPO/dev/036_ledger.py" --snapshot
echo "=== $(date -Is) c2b done ==="
