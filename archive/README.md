# archive/

Superseded material, kept as the record of what was run before. Mirrors the live repo's
layout — `archive/configs/` holds what `configs/` held, `archive/bash/` holds what `bash/`
held — so an archived file's original home is always its subfolder name.

Everything here is the **multilabel era** (2025-10 → 2026-06): the
`dev/011_lepi_large_prod_v2` / `dev/014_..._v3` / `dev/022_..._v3_multihead` line, superseded
by the hierarchical-heads rewrite (`dev/028`, `dev/030`) from 2026-07-07 onward. Two things
date it: no `head:` key in the configs, and paths pointing at machines this box is not
(`/home/george/...`, `/work/...`).

**Nothing here is expected to run.** Paths inside these files are relative to the repo layout
*at the time they were written* — an archived script referencing `configs/foo.yaml` means the
file now at `archive/configs/foo.yaml`. They were not rewritten on archiving: these scripts
also carry `#SBATCH` headers with another machine's absolute paths, so fixing one broken
reference while leaving another would create a false impression of runnability. They are
records, not tools.

| | |
|---|---|
| [`configs/`](configs/) | 16 YAMLs — the multilabel train/test configs. |
| [`bash/`](bash/) | 10 launchers — SLURM submissions for the GPU24 cluster. |

Current work: [`../configs/`](../configs/), [`../bash/`](../bash/). The story of what replaced
this: [`../journal/2026-07-why-was-fastai-behind-mini-trainer.md`](../journal/2026-07-why-was-fastai-behind-mini-trainer.md).
