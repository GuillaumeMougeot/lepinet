# archive/configs — the multilabel era (2025-10 → 2026-06)

Superseded. Kept as the record of what was run before the hierarchical-heads rewrite; not
expected to run on this box.

Two things date them:

- **No `head:` key.** These drive the multilabel/`multilabel-global_lepi` line
  (`dev/011_lepi_large_prod_v2`, `dev/014_..._v3`, `dev/022_..._v3_multihead`) rather than
  mini_trainer's hierarchical heads (`dev/028`, `dev/030`). Different model, different metrics.
- **Foreign paths.** `/home/george/...` (another workstation) and `/work/...` (UCloud mounts).
  Nothing here resolves against this machine's `data/` tree.

`_ece` = the ECE/GPU24 cluster · `_ucloud` = UCloud VM · unsuffixed = local at the time.
Their launchers are in [`../bash/`](../bash/), which reference these by their *original*
`configs/<name>.yaml` paths — see [`../README.md`](../README.md).

Also here: `20260710_test_global_autoregressive.yaml` — heads-era but never filled in
(`model_path: TODO_...`). The autoregressive run it was written for died at epoch 1 and the
head was dropped in favour of `independent`. Note it is being revisited: the 2026-07-17 UCloud
benchmark runs all three heads properly.

Current work: [`../../configs/README.md`](../../configs/README.md).
