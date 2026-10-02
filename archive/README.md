# archive/

Superseded material, kept as the record of what was run before. Mirrors the live repo's
layout — `archive/configs/` holds what `configs/` held, `archive/bash/` holds what `bash/`
held — so an archived file's original home is always its subfolder name.

`configs/` and `bash/` are the **multilabel era** (2025-10 → 2026-06): the
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
| [`configs/`](configs/) | 16 YAMLs — the multilabel train/test configs; plus `README-2026-07.md`, the July `configs/README.md`. |
| [`bash/`](bash/) | SLURM launchers of the multilabel era, plus (archived 2026-10-02) the July shell launchers for the `dev/028`/`030` trainer; their README is `bash/README-2026-07.md`. |
| [`dev/`](dev/) | archived 2026-10-02: `dev/000`-`044`, the multilabel era, the `028`/`030`/`032` trainer and the app-compression tools that depended on it, all superseded by `src/lepinet`. The old `dev/README.md` is `dev/README-2026-07.md`. |
| [`ucloud/`](ucloud/) | archived 2026-10-02: the job specs of every finished experiment (~300). Paths inside (`local = ".."`, `script = ...`) are relative to `ucloud/`, so copy a spec back there to re-run it. The old `ucloud/README.md`, with its list of cluster gotchas, is `ucloud/README-2026-07.md`. |

Current work: [`../dev/`](../dev/), [`../ucloud/`](../ucloud/). The story of what replaced
the multilabel era: [`../journal/2026-07-16-why-was-fastai-behind-mini-trainer.md`](../journal/2026-07-16-why-was-fastai-behind-mini-trainer.md).
