# Why the UCloud MT-head benchmark keeps OOM-ing

**Status:** ROOT-CAUSED. The per-worker cost is the fastai **image-decode pipeline** (~1.2 GB),
not the dataframe -- an earlier diagnosis in this file (fork+COW on the DataFrame) was WRONG and
is corrected below. Fix: cap the worker count so `workers x 1.2 GB` fits the node (128 on a
288 GB B200), verified one-job-first.

## The question

Three 10-epoch benchmark jobs (independent / hierarchical / autoregressive heads, otherwise
identical) were meant to run on UCloud B200 nodes. Every attempt died in epoch 1 with no Python
traceback -- the log just stops mid-batch. What is killing them, and what worker count is safe?

## Silence ⇒ host RAM, not GPU

The log stops mid-batch with no exception. A CUDA OOM raises a loud `torch.cuda.OutOfMemoryError`;
the *absence* of one means the kernel's cgroup OOM-killer SIGKILLed the process -- host RAM. That
was the first fork in the diagnosis and it was correct.

## The wrong turn (recorded on purpose)

Initial diagnosis: dev/030 forks its dataloader workers, CPython refcounting breaks copy-on-write
on the 5.67M-row DataFrame's object (string) columns, so host RAM climbs ~1.1 GB/worker. It fit
the on-node curve (192 workers: 72 → 237 GB burst by batch 300, then a coupon-collector creep to
265 GB / 92% at batch 4525 where the guard aborted) and the arithmetic (256 workers ≈ 282 GB vs
the 288 GB cap; 512 ≈ 563 GB).

**It was wrong.** Two free local tests killed it:
- `--arrow` (pyarrow string columns): no change (+4.72 vs +4.77 GB).
- The `lowmem` rewrite (items = integer indices into fixed-width numpy arrays, no object columns
  touched by workers at all, verified byte-identical to the DataFrame path): per-worker cost
  **1163 MB vs 1096 MB -- essentially unchanged.**

If removing the DataFrame from the workers' hot path doesn't move the number, the DataFrame
isn't the cause.

## What it actually is: the fastai image pipeline

Isolation test (`scratchpad/isolate.py`): a raw DataLoader over a **dummy dataset** that still
indexes the numpy arrays but returns a pre-made tensor instead of decoding a JPEG costs
**~20 MB/worker, dead flat** over 150 batches. The real pipeline costs ~1.2 GB/worker. So the
entire cost is **JPEG decode + resize/augment + prefetch buffers**, per worker.

Corroboration:
- A raw torchvision decode+resize pipeline (no fastai): ~200 MB/worker -- so ~1 GB of the cost is
  fastai's item/transform machinery on top of the decode, not the decode alone.
- The "creep" is the allocator high-water mark rising as a worker happens on progressively larger
  source images (variable JPEG dimensions). Coupon-collector-shaped -- hence decelerating, and
  hence bounded once each worker has seen its largest images. Not a leak; a high-water mark.

Per-worker cost ~1.2 GB, roughly fixed regardless of dataframe representation.

## Why worker-count tuning can't fix it

## Why UCloud OOMs but the 5090 never did (the puzzle)

Same per-worker cost on both machines (~1.2 GB); different **worker counts**, driven by storage
speed:

| | workers | why | worker x 1.2 GB | RAM | outcome |
|---|---|---|---|---|---|
| 5090 (local) | 24 | fastai default `defaults.cpus`; local NVMe is fast, 24 saturate it | ~29 GB | 125 GB | fine, invisible |
| UCloud B200 | 256–512 | the `/work` mount is latency-bound (~90 ms/file); needs many workers to hit the ~1100 img/s decode ceiling | 280–563 GB | 288 GB | OOM |

The local configs set no `num_workers`, so fastai used 24. The UCloud configs set 256–512
precisely to overcome the network-mount latency. The image-pipeline cost per worker is identical;
the cloud just runs 10–20x more of them, straight into the cap. **The bug was never
machine-specific -- it was worker-count-specific, and worker count is set by how fast storage is.**

Near the cap, time-to-death is node-dependent: two 512-worker runs died at batch 0 and at batch
~36,000. The batch-36k run wasn't healthy -- it had collapsed to 2.5 img/s (a 440x slowdown =
the kernel reclaiming page cache under memory pressure, starving the image I/O) before the kill.
"512 survived 36k batches" was really "512 thrashed for 35 minutes then died."

## The fix

The per-worker image-pipeline cost is ~1.2 GB and essentially irreducible without leaving fastai
or moving decode to the GPU. So the lever is the worker count: keep `workers x 1.2 GB` well under
the node RAM while still enough workers to saturate the decode ceiling on the mount.

**128 workers** on the 288 GB B200:
- 128 x 11 files/s (at 90 ms/file) ≈ 1400 files/s raw > the ~1100 img/s decode ceiling -> full
  throughput, no staging needed.
- 128 x 1.2 GB ≈ 158 GB plateau (55%) + the bounded decode-size creep (~+20 GB) ≈ 178 GB (~62%)
  -- real headroom under the guard's 92%.
- Launched **one job first** and watched the guard's per-batch memory log plateau under ~65%
  before committing the other two (the 192 attempt taught this: don't launch three on a
  prediction).

Kept regardless: the **`lowmem` make_dls rewrite** (numpy-indexed items, byte-identical to the
DataFrame path). It didn't fix the OOM -- the OOM wasn't the dataframe -- but it is correct, it
slightly lowers the parent footprint, and it removes a real (if secondary) refcount-churn source.

If 128 ever proves marginal on a busier node, the proven fallback is **staging images to
node-local NVMe** (`ucloud/stage.py` + `setup-staged.sh`): fast storage means ~48 workers
saturate the decode ceiling, and 48 x 1.2 GB ≈ 58 GB is bulletproof. The README long dismissed
staging as "not worth it for throughput" -- but it is the memory-safety lever, not a speed one.

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
