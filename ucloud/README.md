# Running lepinet on UCloud

Job specs live here, in the repo, next to the code they launch — so a spec is versioned
with the config and script it refers to, and `[sync]` ships the whole tree anyway. Nothing
here is read by lepinet itself; it is only input to
[`ucloud-api`](https://github.com/GuillaumeMougeot/ucloud-api).

| File | What it is |
| --- | --- |
| `smoke-9717.toml` | One family, MIG slice — a few minutes, for checking the plumbing |
| `onecycle-1ep.toml` | Full dataset, one epoch, node-local staging (timing probe) |
| `onecycle-nocopy.toml` | Full dataset, one epoch, straight off the mount (timing probe) |
| `onecycle-5ep.toml` | **The real run**: reproduce the 5-epoch reference (f1_species 0.888) |
| `setup.sh` | Builds the venv inside the job (embedded into the batch script) |
| `setup-staged.sh`, `stage.py` | Optional: copy images to node-local NVMe first (rarely worth it) |
| `../configs/20260716_ucloud_*.yaml` | The training configs, with `/work/...` paths |

Training configs stay in `configs/` with everything else. The only thing that makes one
"a UCloud config" is that its paths point at the job's mounts.

## Submitting

```bash
ucloud q submit ucloud/smoke-9717.toml --name lepi-smoke
ucloud q daemon          # auto-extend + dependencies; tmux it on an always-on box
ucloud q logs lepi-smoke # setup + training output, readable while it runs
ucloud q ls
```

The daemon is optional. Batch jobs self-terminate when the run exits (UCloud enforces
that, not the daemon), so stopping it never leaks a running GPU — you only lose
auto-extend and the launching of queued jobs.

## Mounts

| In the job | Comes from | Notes |
| --- | --- | --- |
| `/work/lepinet` | `[sync]` pushes this repo to `/12347837/repos/lepinet` | rw — write results here and they persist |
| `/work/mini_trainer` | `ucloud sync push ../mini_trainer /12347837/repos/mini_trainer` | dev/030 imports it; not a declared lepinet dep |
| `/work/global_lepi` | `/12347837/datasets/global_lepi` | read-only; 481 GB, never synced |

`[sync]` is incremental and `.gitignore`-aware, so `data/` never travels and a re-submit
pushes only what changed. **Re-run the `mini_trainer` push by hand when you change it** —
only `[sync]`'s own folder is automatic.

Because `out_dir` is under `/work/lepinet`, checkpoints land on the drive as they are
written. Fetch them with `ucloud files download`.

## Things that bit us, so they don't bite again

- **`ssh_enabled = true` is rejected by `pytorch-te`** ("This application does not support
  SSH but it is required") — the job won't even submit. Use `q logs` to watch runs.
- **`tensorboard` is missing from `pyproject.toml`.** `dev/030` imports
  `fastai.callback.tensorboard`, which imports `tensorboard`, which fastai declares
  nowhere. The local `.venv` has it, so this only breaks in a fresh environment.
  `setup.sh` installs it as a workaround; the real fix is a line in `pyproject.toml`.
- **`setuptools-scm` needs `.git`,** which the sync deliberately excludes, so `setup.sh`
  sets `SETUPTOOLS_SCM_PRETEND_VERSION`.
- **Don't use ucloud-api's `python = "uv"` shortcut here.** It runs `uv sync`, which would
  build the venv on the network drive. `setup.sh` builds it on the job's local disk.
- **The `gen_df` cache ignores filter settings** — it is keyed on path alone. A cached
  `.lepinet.parquet` from a full-dataset run would silently override `family_filter`, so
  filtered runs need their own `out_dir`.

## num_workers is the whole ballgame

**Set `num_workers: 256` in any config that reads images from `/work`.** fastai defaults to
`min(16, cpus)` regardless of the machine, and that single default is the difference
between 8.7 h and 1.4 h per epoch.

The `/work` mount is *latency*-bound (~90 ms per file), not bandwidth-bound. Measured from
a job, reading random images:

| readers | 1 | 16 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- |
| img/s | 11 | **173** | 1038 | 1545 | 1768 |

At fastai's default 16 the loader gets ~173 img/s — exactly what training was doing while
the B200 sat idle. Loader workers block on the network ~90 ms per image and decode for
~10, so running 256 processes on 48 vCPU is not oversubscription in any real sense: most
are parked on I/O at any instant.

## Measured, full dataset, one epoch

| setup | img/s | epoch |
| --- | --- | --- |
| local RTX 5090 (24 cores, local disk) | 710 | 2:13 |
| B200, `/work` mount, 256 workers | 950 | **1:24** |
| B200, node-local NVMe, 48 workers | 1108 | 1:09 |

Both UCloud variants beat the 5090. The ceiling is the decode/augment pipeline (~1100
img/s), not storage — which is why reading straight off the mount lands within 15% of
local NVMe, and why **staging is not worth it**: `ucloud/stage.py` + `setup-staged.sh`
copy 594 GB in 37 min to save 16 min/epoch, so they only pay back past ~2.3 epochs, and a
third of that copy is images the filtered parquet never opens. They are kept for long
(10+ epoch) runs; the default is no copy.

To go faster than ~1100 img/s the target is not storage at all: it is `aug_img_size: 460`
forcing a per-image CPU PIL resize, or moving JPEG decode to the GPU (DALI/nvJPEG).

## Choosing a product

`gpu-nvidia-b200-1-gpu` (full GPU, 48 vCPU) for real runs. `gpu-nvidia-b200-1-mig.1g`
(6 vCPU / 36 GB, a 1g MIG slice) is the cheapest GPU and is what `smoke-9717.toml` uses —
fine for smoke tests, wrong for training: 6 vCPU cannot host enough loader workers.

On the smoke subset the first epoch ran 4:28 and the second 1:58, because a single family
fits in page cache. The full 594 GB tree does not, so expect first-epoch speed throughout.
