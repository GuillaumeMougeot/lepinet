# bash/

Launchers. Two kinds live here, and the distinction matters because they run in different
places:

**SLURM submission scripts** (`#SBATCH` headers, paths under `/home/george/...`) target the
**GPU24 cluster**, not this workstation. They will not run as-is on the local box — the paths
are someone else's.

**Local orchestration** (`cd /home/au761367/codes/lepinet`, `nohup`/`pgrep` loops) targets
**this machine**. These are one-off queue scripts written to chain a specific sequence of runs
and wait on a specific predecessor — they are records of a past launch, not general tools.
Re-read one before reusing it; the run it was waiting for finished long ago.

## Active

| script | kind | what it did |
|---|---|---|
| `028_train_multihead_v4.sh` | SLURM | dev/028 multihead-v4 training on GPU24. |
| `030_train_heads_benchmark.sh` | SLURM | dev/030, one job per head type (hierarchical / independent / autoregressive) on the small subset. |
| `chain_028_to_030.sh` | local | Waited for a running dev/028 job, then ran the dev/030 head series. Historical. |
| `031_run_muon_warmup.sh` | local | The Muon + warmup + light-aug + grad_clip 5.0 run, then its dev/032 eval. This is the step that bought **0.8297 → 0.8769**. |
| `queue_muon_after_series.sh` | local | Waited on `chain_028_to_030.sh`, then ran Muon + flat_cos. Written after the first Muon run NaN'd at epoch 1 (the fp32-head fix made it stable). Historical. |
| `rsync_ucloud.sh` | utility | Sync to/from UCloud. See `ucloud/`. |

## Archived

The **multilabel era** (`dev/005`/`011`/`012`/`014`/`017`/`022`/`024`, 2025-10 → 2026-06),
superseded by the hierarchical-heads line, now lives in
[`../archive/bash/`](../archive/bash/), with its configs in
[`../archive/configs/`](../archive/configs/).

## Note on how runs are actually launched now

Recent work does not use these scripts. Runs are launched directly in `tmux`, e.g.

```bash
tmux new-session -d -s longtail -c /home/au761367/codes/lepinet \
  "source .venv/bin/activate && cd /home/au761367/codes/lepinet && \
   python dev/030_hierarchical_heads_benchmark.py -c configs/<cfg>.yaml 2>&1 | tee logs/<log>.log"
```

Two gotchas that have bitten:

- **Always `cd` explicitly** (and pass `-c` to `tmux new-session`). A shell whose working
  directory drifted will fail with `can't open file '.../site-packages/dev/030_...py'`.
- **`uv run` is broken in this repo** and **`uv sync` will destroy the venv** — use
  `.venv/bin/python` or `source .venv/bin/activate`. See
  [`journal/2026-07-venv-uv-sync-incident.md`](../journal/2026-07-venv-uv-sync-incident.md).
