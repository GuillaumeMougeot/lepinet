#!/usr/bin/env bash
# Phase C2 of the lepi-app compression plan: the backbone sweep, at the C1 bottleneck.
#
# Trains + tests each candidate backbone (hidden=256, otherwise the 0.9148 recipe), sequential
# on the one 5090. Detach from any session -- these runs outlive the conversation:
#   setsid nohup dev/run_c2_backbone_sweep.sh >data/global/models/c2_logs/sweep.log 2>&1 </dev/null &
#
# WAIT_FOR_C1=1 makes it block until the C1 sweep has written its ledger line, so C2 can be
# queued now and start itself the moment C1's GPU frees. Set 0 to run immediately.
#
# Before launching, confirm `hidden` in configs/20260721_backbone_*.yaml matches C1's chosen
# knee (currently 256). Edit ORDER to drop any already in the ledger.
set -u  # NOT -e: one arch failing must not cancel the rest

REPO=/home/au761367/codes/lepinet
PY=$REPO/.venv/bin/python
LOGDIR=$REPO/data/global/models/c2_logs
C1LOG=$REPO/data/global/models/c1_logs/sweep.log
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

WAIT_FOR_C1=${WAIT_FOR_C1:-1}
if [ "$WAIT_FOR_C1" = "1" ]; then
    echo "=== $(date -Is) waiting for C1 to finish (watching $C1LOG) ==="
    while ! grep -qa "sweep done; ledger" "$C1LOG" 2>/dev/null; do sleep 300; done
    echo "=== $(date -Is) C1 done; starting C2 ==="
fi

# effnetv2b0 first: modern CNN anchor with the clearest lineage from the 0.9148 baseline.
ORDER="effnetv2b0 fastvit_sa12 repvit_m1_1 mnv4_medium fastvit_t12"

for TAG in $ORDER; do
    echo "=== $(date -Is) train $TAG ==="
    "$PY" "$REPO/dev/030_hierarchical_heads_benchmark.py" \
        --config "$REPO/configs/20260721_backbone_${TAG}.yaml" \
        > "$LOGDIR/train_${TAG}.log" 2>&1
    echo "    train exit=$? $(date -Is)"

    echo "=== $(date -Is) test $TAG ==="
    "$PY" "$REPO/dev/032_hierarchical_heads_test.py" \
        --config "$REPO/configs/20260721_test_backbone_${TAG}.yaml" \
        > "$LOGDIR/test_${TAG}.log" 2>&1
    echo "    test exit=$? $(date -Is)"
done

echo "=== $(date -Is) C2 sweep done; ledger ==="
"$PY" "$REPO/dev/036_ledger.py" --snapshot
