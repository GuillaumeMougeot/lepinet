# Making the B200 fast: staging, and GPU decode

**Status:** staging built + validated (targeted `ucloud/stage.py`); GPU decode is a design here,
not yet built. Both attack the same root fact: **the pipeline is CPU-JPEG-decode-bound at
~1100 img/s**, so a B200 that could train several× faster sits mostly idle, and the CPU decode
buffers (~1.2 GB/worker) are what force the worker-count/memory bind
([[2026-07-ucloud-benchmark-oom]]).

## Where the time goes

Measured: B200 epoch ~1:16 (independent) vs 5090 2:13 -- only 1.7× faster despite a vastly
stronger GPU, because both are limited by the same ~1100 img/s CPU decode, not by the GPU. GPU
allocated during training is ~1.0 GB out of ~180 GB: the B200 is starved.

Two independent bottlenecks, and it helps to keep them separate:
- **latency** (opening files): the `/work` mount is ~90 ms/file, so you need many concurrent
  readers to hit any throughput at all.
- **decode** (JPEG -> pixels): ~1100 img/s on the CPU regardless of storage, and each fastai
  worker's decode+resize+collate holds ~1.2 GB.

## Staging (built, validated) -- fixes latency + memory, not the decode ceiling

`ucloud/stage.py` copies **only the images the filtered df references** (~5.67M / ~430 GB, vs
the full ~594 GB) to node-local NVMe, in parallel (threads block on `/work` latency). Verified
byte-exact locally (staged set == referenced set, 0 missing/extra) with a short-copy abort.

With images on fast local disk the latency bottleneck is gone, so **~32-48 workers** saturate
the ~1100 img/s decode ceiling instead of 128+. That drops real memory from ~168 GB (128×1.2)
to ~40-58 GB -- comfortable on any node, and the fix for a *small*-RAM box like the 5090 where
you genuinely can't afford many workers. But it does **not** raise the ceiling: decode is still
~1100 img/s on CPU (README: 1108 staged vs 950 on /work -- a ~16% I/O win, not a step change).

So staging = memory lever + modest throughput. Good, cheap, done. Not the way to actually use
the B200.

**Measured end-to-end on UCloud (2026-07-18, `staged-test`):**
- targeted copy: **5,669,317 files / 450 GB in 12.6 min** (parallel, ~7500 files/s), verified complete.
- 32 workers on staged NVMe: **48 GB anon (17%)**, epoch **1:22**.
- vs 128 workers on /work: 168 GB anon, epoch 1:18.
- 32 workers *under-saturates* the decode ceiling (slightly slower); ~48 workers staged is the
  throughput optimum (~1:09, ~72 GB) -- but even that barely beats /work, because both are the
  same ~1100 img/s CPU decode. **Confirmed: staging cuts memory ~3.5× (168 -> ~50-70 GB) and
  does not meaningfully change speed.** So it is the fix for a RAM-tight node (the 5090), and
  unnecessary on the 288 GB B200 where 128 workers already runs.

## GPU decode (design) -- raises the ceiling AND removes the memory bind

The idea: stop decoding JPEGs on the CPU. Read **raw JPEG bytes** on the CPU (cheap, pure I/O)
and decode on the GPU with nvJPEG. The B200 has dedicated JPEG-decode hardware; nvJPEG does
thousands of images/sec, so the ~1100 img/s ceiling lifts and the bottleneck moves to the
actual training compute -- which is the whole point of renting a B200.

**It also dissolves the OOM problem.** If CPU workers only read bytes, their per-item memory is
~one JPEG (~80 KB), not a ~1.2 GB decode+resize+collate pipeline. So:
- the ~1.2 GB/worker anon cost -> near zero; worker count stops being a memory ceiling;
- you can run many lightweight byte-readers to beat `/work` latency *without* the memory blowup;
- staging becomes optional (still a mild I/O win, no longer needed for memory).

One design point: augmentation is already on the GPU. fastai runs `aug_transforms` (flip/rotate/
zoom) and `Normalize` as **batch_tfms on the GPU**; only the JPEG decode and the item `Resize`
are on the CPU. So GPU decode only has to relocate *decode + resize*, and fastai's existing GPU
aug stays as-is.

### Implementation options, lightest first

1. **torchvision.io.decode_jpeg(device="cuda")** -- nvJPEG, already in our stack (torchvision
   0.27). **Measured 2026-07-18 on the local 5090: batched GPU decode of 256 JPEGs = 3794 img/s
   (decode only), vs the ~1100 img/s CPU ceiling -- 3.4× on a weaker GPU; the B200's dedicated
   decode hardware should beat that.** Plan: a Dataset that returns raw bytes (a uint8 tensor of the file); a custom collate
   that keeps them as a list (JPEGs are variable-length); a fastai `before_batch`/batch_tfm that
   calls `decode_jpeg(list, device="cuda")`, resizes each to `aug_img_size` on the GPU, and
   stacks -- then fastai's existing aug/normalize batch_tfms run unchanged. Lightest dependency,
   fits fastai's batch_tfm hook. Main work: variable-size handling + verifying byte-exact aug.
2. **NVIDIA DALI** -- a full GPU input pipeline (decode+resize+augment), fed to fastai/torch via
   `DALIGenericIterator`. Fastest and most battle-tested, but a heavy CUDA-version-specific
   dependency and it *replaces* the fastai DataLoader (so aug/normalize move into DALI, and the
   metrics/label plumbing must be re-verified). More power, more rewrite.
3. **nvimgcodec / kornia** -- middle options; not obviously better than (1) for our case.

Recommend prototyping **(1)** first, measured the same way everything else here was: a decode
throughput probe (imgs/s CPU-bytes+GPU-decode vs the ~1100 CPU baseline) and a memory probe
(dev/037-style: confirm per-worker anon collapses to ~bytes). Both are GPU-cheap and catch the
risks (variable-size batching, aug correctness) before a full run.

### Risks / unknowns
- Variable JPEG dimensions: nvJPEG decodes to native size; must resize-on-GPU before stacking.
  Some source images may be large -> transient GPU memory spikes (bounded by the largest image
  in a batch; B200 has room, but worth watching).
- A few images may be CMYK/corrupt/PNG-in-disguise; `decode_jpeg` will throw where PIL coped.
  Need a fallback (CPU-decode the stragglers) or a pre-filter.
- Correctness: the decode+resize path must match the CPU one closely enough not to shift the
  numbers. Validate against a fixed batch (same images -> same normalized tensors within tol).

### Measured: GPU decode done right (2026-07-18, dev/038 + dev/039)

Profiling (dev/039) found the earlier ~650 img/s was NOT decode -- decode+resize+aug is ~11% of
a batch; the model is ~77% (bwd 41% + fwd 29% + Muon 6%). The two real caps were dev/038's
per-batch `gc.collect` (halving throughput) and no overlap. Fixes: double-buffer overlap (decode
N+1 on a side CUDA stream while the model runs N) + `gc.collect(0)` every N batches (the
per-batch gc guards mini_trainer's cosine-head GPU cycle; every-8 keeps GPU mem flat).

With both, decode is fully hidden and training is **model-bound**:

| GPU | bs | img/s | GPU mem | note |
|---|---|---|---|---|
| 5090 | 64 | ~820 | 7.8 GB | vs 650 before the fixes |
| 5090 | 128 | ~880 | 14.5 GB | |
| 5090 | 192 | ~930 | 21.3 GB | bs256 OOMs the 5090's 32 GB |
| B200 | 256 | **~1090** | 28 GB | staged + overlap; barely above the CPU pipeline's ~1100 |

**The punchline: for effnetv2s there is NO throughput win from GPU decode.** The CPU pipeline's
"1100 img/s, decode-bound" was ~coincident with the model ceiling: effnetv2s is a small, efficient
model, launch/bandwidth-bound at ~1100 img/s, and even the B200 at bs=256 only reaches ~1090 while
using 28 of 180 GB. Decode was never the true bottleneck for this model; the model is.

What GPU decode DID deliver: **memory**. ~28 GB GPU / ~15 GB host at bs=256, vs the CPU pipeline's
168 GB host at 128 workers -- a ~10x cut that removes the worker/OOM bind and (unlike staging) needs
no copy. And it makes big batches free memory-wise.

Where the throughput win WOULD appear: a heavier model (effnetv2_m/l, a ViT) that is genuinely
compute-bound. There, CPU decode caps below what the GPU could train, and GPU decode + big batch
would stretch the B200. For effnetv2s specifically, the B200 is overkill regardless of decode.

### Verdict
- **Now:** staging is the pragmatic win and it's done -- use it for memory-tight nodes and a
  small speed bump.
- **Next real lever:** GPU decode via torchvision nvJPEG. It's the only thing that both breaks
  the decode ceiling (actually using the B200) and eliminates the worker-count memory bind. It's
  a real prototype, so it goes behind a measured probe before any full run -- same discipline
  that kept the OOM investigation to ~25 GPU-hours.

Related: [[2026-07-ucloud-benchmark-oom]] (why worker count was capped), the `ucloud/README.md`
storage-latency table, `dev/037_dl_memory_probe.py` (the probe to reuse for a GPU-decode memory
check).
