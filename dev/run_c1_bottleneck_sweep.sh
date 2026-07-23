#!/usr/bin/env bash
# Phase C1 of the lepi-app compression plan: the classifier bottleneck sweep.
#
# Trains and then tests `hidden` against the 20260716-154156 recipe (test species macro-F1
# 0.9148), changing nothing else. Sequential on purpose -- one 5090, and interleaving would
# make the per-run wall-clock numbers meaningless. ~8 h train + ~0.5 h test per value.
#
# Absolute paths throughout and fully detached from any interactive session -- an earlier
# harness-tracked background run was reaped after ~8 h mid-sweep (hidden=256 completed, 512
# was lost). Launch so it survives the session:
#   setsid nohup dev/run_c1_bottleneck_sweep.sh >data/global/models/c1_logs/sweep.log 2>&1 </dev/null &
#
# ORDER holds the values still to do; delete ones already in the ledger before relaunching.
set -u  # NOT -e: one config failing must not cancel the rest of the sweep

REPO=/home/au761367/codes/lepinet
PY=$REPO/.venv/bin/python
LOGDIR=$REPO/data/global/models/c1_logs
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

ORDER="512 128"

for H in $ORDER; do
    echo "=== $(date -Is) train hidden=$H ==="
    "$PY" "$REPO/dev/030_hierarchical_heads_benchmark.py" \
        --config "$REPO/configs/20260720_bottleneck_${H}.yaml" \
        > "$LOGDIR/train_${H}.log" 2>&1
    echo "    train exit=$? $(date -Is)"

    echo "=== $(date -Is) test hidden=$H ==="
    "$PY" "$REPO/dev/032_hierarchical_heads_test.py" \
        --config "$REPO/configs/20260720_test_bottleneck_${H}.yaml" \
        > "$LOGDIR/test_${H}.log" 2>&1
    echo "    test exit=$? $(date -Is)"
done

echo "=== $(date -Is) sweep done; ledger ==="
"$PY" "$REPO/dev/036_ledger.py" --snapshot
