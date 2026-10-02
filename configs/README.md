# configs/ — one YAML per training or evaluation run

Named `<YYYYMMDD>_<ID or what it does>.yaml`, e.g. `20260804_B8_best_at_2pct_dinov3cnxl.yaml`. A run is
started with `lepinet train -c <config>` (or a `dev/` script for experiments that need one); the
config is copied into the run directory, so every result on disk carries the config that made it.
The experiment IDs in the names are resolved in [`../RESULTS.md`](../RESULTS.md) section 2.

**Start from** `20260729_ucloud_singlehead_species_effnetv2s.yaml` (the cheap reference, 0.9135),
or from the config of the model you are extending (B8, P5). Copy a config that has already run;
configs are copied wholesale, not inherited, so `desc` and `model_name` are what tell a reader what
changed. Keep them honest.

**Gotchas.** The `set` column is a 10-fold split: `'0'` is the held-out test fold (never train on
it, never filter it), `'1'` validates, `2`-`9` train. `model_name` sets the output directory
(`<timestamp>-<model_name>/`). Margin heads (`head: arcface`) need `precision: bf16`.

Configs from July name `dev/030`/`dev/032`, the trainer that preceded the package; it now lives in
[`../archive/dev/`](../archive/dev/). The old version of this README, with the July recipe ladder,
is [`../archive/configs/README-2026-07.md`](../archive/configs/README-2026-07.md).
