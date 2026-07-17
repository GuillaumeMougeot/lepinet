# Configs

One YAML per experiment, named `<YYYYMMDD>_<what it does>.yaml`. A config is consumed by
exactly one dev script (`train:` → `dev/030`, `test:` → `dev/032`), and dev/030 copies it into
the run directory, so **every result on disk carries the config that produced it**.
`python dev/036_ledger.py` reads those copies back and prints what each run tested and scored —
that is the fastest way to find "which config produced the good number".

## The winning recipe

`20260716_heads_global_independent_muon_5ep_oversample.yaml` — **test species macro-F1 0.9148**,
the project best. Independent heads, Muon, `one_cycle`, `warmup_epochs: 0.5`, `grad_clip: 5.0`,
light aug, **square-root class oversampling** (`oversample_power: 0.5`), only **5 epochs**.
Beats mini_trainer's own loop (0.896) by +5.5pt and the prior best 10-epoch run (0.8976, no
oversampling) by +1.7pt, at half the epoch budget. Start new experiments by copying this, not
the 10ep one below. Story: [`journal/2026-07-does-longtail-help.md`](../journal/2026-07-does-longtail-help.md).

Not yet tested: whether oversampling stacks further at 10 epochs
(`20260715_heads_global_independent_muon_10ep_oversample.yaml`, not launched). Logit adjustment
was tested and is **not recommended** — see the long-tail section below.

## Layout

| | |
|---|---|
| `*.yaml` | Current work — the **hierarchical-heads era** (2026-07-07 →), `dev/028`+`dev/030`. |
|  `../archive/` | The **multilabel era** (2025-10 → 2026-06), `dev/011`/`014`/`022`. Superseded architecture (no `head:` key), and paths pointing at machines this box is not (`/home/george/...`). Kept as the record of what was run then; not expected to run here. Their launchers are in [`../archive/bash/`](../archive/bash/). |

## Current configs by line of work

**Small / smoke** — `20260707_*_small.yaml`: resnet18 on the small subset, for shaking out
head wiring quickly.

**Head comparison** — `20260709_heads_global_{independent,hierarchical,autoregressive}.yaml`:
the three mini_trainer heads on the global set. `independent` won and is what everything since
uses.

**The optimiser/schedule ladder** (this is the interesting sequence — each is one step of
[the ladder](../journal/2026-07-why-was-fastai-behind-mini-trainer.md)):

| config | change | test F1 |
|---|---|---|
| `20260712_heads_global_independent_muon.yaml` | Muon + flat_cos | 0.8297 |
| `20260713_heads_global_independent_muon_warmup.yaml` | + warmup, light aug, grad_clip 5.0 | 0.8769 |
| `20260714_heads_global_independent_muon_onecycle.yaml` | flat_cos → one_cycle | 0.8887 |
| `20260715_heads_global_independent_muon_onecycle_10ep.yaml` | 10 epochs | 0.8976 |
| `20260716_heads_global_independent_muon_5ep_oversample.yaml` | back to 5ep + `oversample_power: 0.5` | **0.9148** |

Every rung up to and including 10 epochs was an **optimisation** fix (optimiser, schedule
shape, epoch count). Oversampling is the first **data-distribution** fix, and it outsized all
of them — see the long-tail entry for why (53% of species have under 200 training images).

**Tested and rejected** — `20260714_heads_global_independent_muon_onecycle_reg.yaml`:
class-distribution regularisation, measured as a wash (0.8860 vs 0.8880, inside noise).
Kept so the negative result stays discoverable.

**Never run** — `20260714_heads_global_independent_muon_frontloaded.yaml`: the `front_loaded`
schedule is implemented in dev/030 but was never queued. Written for 5 epochs; the work has
since moved to 10.

**Long-tail** — `20260716_heads_global_independent_muon_5ep_oversample.yaml`: **the winning
recipe**, see above. `20260716_..._5ep_logitadjust.yaml`: tested and **not recommended** —
test species F1 0.9031 (a modest win over baseline) but genus/family both regressed
permanently, traced to one shared `tau=1.0` being too aggressive for family's much more
extreme class imbalance than species'. `20260715_*_10ep_{oversample,logitadjust}.yaml`: the
10-epoch variants; the oversample one is worth running (untested whether the win stacks with
more epochs), the logitadjust one is not worth running without fixing the per-level tau issue
first. See [`journal/2026-07-does-longtail-help.md`](../journal/2026-07-does-longtail-help.md).

**Resume** — `20260716_heads_global_independent_muon_onecycle_10ep_resume.yaml`: a historical
record of recovering the 10ep run from the machine hang. Its `resume_checkpoint` points at a
per-epoch `.pth` that has since been reaped (see `models/` note below), so it will **not** re-run
as-is; the mechanism it exercises (`fit_resume`) is live and documented in
[`journal/2026-07-gpu-hang.md`](../journal/2026-07-gpu-hang.md).

**UCloud** — `20260716_ucloud_smoke_9717_onecycle.yaml`: the heads recipe against `/work/...`
mounts, family-filtered to Erebidae for a minutes-long correctness smoke. See `ucloud/`.

**Test configs** — `*_test_*.yaml` / `*_test.yaml`: consumed by `dev/032`. Each names the `.pt`
it evaluates in `model_path`; results land in `data/<ds>/preds/<model>/<ts>-<eval>/`.

## Conventions and gotchas

- **`fold: '1'`** is the *validation* fold. The global `set` column is a 10-fold split: **`'0'`
  is the held-out test set** (dev/030 removes it from training), `'1'` validates, `2`–`9` train.
  Test configs therefore use `test_set: '0'`.
- **Muon requires LR-only scheduling.** `MuonAuxAdamW` re-partitions param groups (breaking
  fastai's freeze bookkeeping — unfrozen schedules only) and takes tuple betas, so fastai's
  stock `fit_one_cycle` chokes on it. `warmup_epochs > 0` selects dev/030's hand-built LR-only
  curve; that is the path you want with `optimizer: muon`.
- **Configs are copied wholesale, not inherited.** Two configs differing in one line duplicate
  the other 39. There is no `base:` key (yet) — so when you copy one, the `desc` and
  `model_name` are the *only* things telling a reader what changed. Keep them honest.
- **`model_name` sets the output directory**, as `<timestamp>-<model_name>/`. Reusing a name
  across runs is fine (timestamps disambiguate), but the ledger joins evals to runs by
  checkpoint *path* for exactly this reason.
- **Per-epoch `.pth` checkpoints under `models/` are reaped** once a run exports its `.pt`
  bundle (they cost ~173 MB *each*). The `.pt` is the deliverable and what `dev/032` loads.
