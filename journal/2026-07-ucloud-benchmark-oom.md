# Why the UCloud MT-head benchmark keeps OOM-ing

**Status:** DIAGNOSED, not fixed. The benchmark is blocked on a dataloader memory leak that
worker-count tuning cannot solve. The real fix is an open engineering task.

## The question

Three 10-epoch benchmark jobs (independent / hierarchical / autoregressive heads, otherwise
identical) were meant to run on UCloud B200 nodes. Every attempt died in epoch 1 with no Python
traceback -- the log just stops mid-batch. What is killing them, and what worker count is safe?

## What it is: fork + refcount breaks copy-on-write on the dataframe

dev/030 sets `multiprocessing.set_start_method("fork")`, so each dataloader worker inherits the
5.67M-row preprocessed DataFrame by copy-on-write. But CPython refcounting *writes* to an
object's header every time it reads it, so COW pages get copied as workers touch rows. Host RAM
therefore climbs with batches consumed until the cgroup OOM-killer SIGKILLs the process -- which,
unlike a CUDA OOM (a loud `torch.cuda.OutOfMemoryError`), leaves no traceback. That silence was
the whole diagnostic: no Python exception ⇒ host RAM, not GPU.

Measured on-node (via the `HostMemoryGuard` logging every 25 batches, at 192 workers):

| phase | batches | memory | mechanism |
|---|---|---|---|
| fast burst | 25 → 300 | 72 → 237 GB | contiguous pointer-arrays + index dirtied |
| slow creep | 300 → 4525 | 237 → **265 GB (92%, guard abort)** | scattered string-objects, one page at a time |

Per-worker steady state of the burst is ~1.16 GB (matches the local probe's 1.1 GB). The creep
is coupon-collector-shaped -- decelerating, but not fast enough to flatten under the 288 GB cap
within an epoch.

## Why worker-count tuning can't fix it

Because the creep never stops within an epoch, *any* worker count that starts under the cap
still creeps into it:

| workers | plateau | died |
|---|---|---|
| 512 | ~563 GB | batch 0–36k (node-dependent; see below) |
| 256 | ~282 GB | ~batch 315 |
| 192 | ~237 GB → 265 | ~batch 4525 (guard) |

**Time-to-death is non-deterministic near the cap.** Two 512-worker runs died at batch 0 and at
batch ~36,000 respectively. Re-reading the batch-36k run: it wasn't healthy -- it had collapsed
to 2.5 img/s (a 440x slowdown), which is the kernel frantically reclaiming page cache under
memory pressure, starving the image I/O, before the eventual kill. So "512 survived 36k batches"
was really "512 thrashed for 35 minutes then died." The apparent worker-count paradox
(512 outlasting 256) dissolves: both are doomed; the exact moment depends on node page-cache
dynamics.

## Disproven fix: pyarrow string columns

Hypothesis: the leak is the object-dtype string columns (millions of Python str objects). Fix:
back them with one contiguous Arrow buffer (`--arrow` in dev/037). Result: **no change** (+4.72
vs +4.77 GB over 125 batches at 16 workers). fastai's DataBlock walks the frame with
`df.iloc[i]`, materialising each row as a Python-object Series regardless of column storage, so
the refcount churn is in the *traversal*, not the dtype. Tested for free on the local box before
spending any GPU -- the reason to have dev/037.

## The real fix (open)

Stop forking a pandas DataFrame for the hot path: pre-extract `image_path` + the three label
keys into numpy arrays (or encode labels as int codes) and give the DataBlock custom getters
that index those arrays instead of the df. numpy arrays are a single object each, so worker
access doesn't incref per-row. This needs **label-correctness validation** (a misaligned
path/label array would train and score plausibly while being silently wrong), which a memory
probe cannot provide -- so it is a real task, not a quick patch.

Interim options if a result is needed before that fix lands:
- **Run locally** on the 5090 (125 GB, ~24 workers): the leak is bounded there and multi-hour
  runs have always completed. Slower (~2:13/epoch x 10 x 3 heads, sequential), but proven.
- **Low worker count on UCloud** (~96): plateau ~126 GB, *might* keep the epoch-1 creep under
  the guard -- but "might" is exactly what burned the 192 attempt. The guard makes a wrong guess
  cheap (a clean abort, ~1.5 GPU-hr) rather than silent.

## What got built along the way

- `dev/037_dl_memory_probe.py`: measures per-worker dataloader memory with **zero GPU**.
  Isolated subprocess per worker count; PSS not RSS (summing RSS across a fork tree double-counts
  shared pages -- it reported 307 GB on a 125 GB box before that fix).
- `HostMemoryGuard` in dev/030: logs the real cgroup limit (not the physical host psutil sees)
  and the real num_workers (fastai's DataLoader hardcodes `.num_workers = 1`; the value lives on
  `.fake_l`), and aborts with an explanation at 92% so the next OOM is one clear line, not nine
  silent GPU-hours.
- `ucloud/setup.sh` preflight: imports both pipeline halves and checks the GPU, and now actually
  `exit 1`s on failure (a bare `SystemExit` in a `set -e`-less script printed and continued --
  which had also made the pre-existing CUDA check toothless).

## Bugs the pipeline smoke caught before any of the above mattered

All in the benchmark's back half, which no job had ever reached (every prior UCloud job was
train-only), each of which would otherwise have cost ~42 GPU-hours (3 jobs x ~14h) to discover:

1. `mini_metrics` absent from the job image entirely (dev/032 imports it; no pyproject declares
   it; it wasn't even on the drive).
2. The memory guard reading the physical host instead of the cgroup, and fastai's fake
   `num_workers`.
3. dev/032 `KeyError` on a hierarchy *wider* than the model's vocab (family_filter case).
4. The stale `/work/global_lepi/hierarchy.csv` (11,939 species vs the dataset's 12,041) --
   dev/030 saved a file-sourced hierarchy that disagreed with its own df-sourced masks by 102
   classes. Fixed by saving `build_hierarchy(df)`. Invisible locally because the local
   hierarchy.csv happens to match.

Total GPU spent across the entire investigation: ~18 GPU-hours, most of it the diagnostic runs,
none of it a completed benchmark. The smoke-first discipline is what kept it that low.

Related: [[does-longtail-help]] (the recipe being benchmarked), the `dev/037` /
`HostMemoryGuard` docstrings for the measurement detail.
