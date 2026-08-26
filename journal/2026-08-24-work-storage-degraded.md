# Incident: /work storage read latency collapsed; two training jobs starved, not hung

**Kind:** incident · **Status:** **RESOLVED (2026-08-26). UCloud fixed the storage.** Parallel
read throughput back to **992 files/s**; G3b now trains at **509 img/s**, better than the 358 it
managed before the outage. All runs restarted, fully serialised. Both jobs launched on
2026-08-24 stalled. They were not deadlocked and had not crashed: **per-file read latency on `/work`
has gone from milliseconds to seconds**, and the pipeline is IO-bound. Throughput is ~41 img/s
against a historical ~1100.

## Symptom

Both jobs showed `RUNNING` with a progress bar that advanced in bursts and then froze for minutes.
No traceback, no OOM, no CUDA error. `ucloud q ls` reported them healthy throughout — the invariant
"read the logs, not the status" applied again, and this time even the logs looked fine.

| time | event |
|---|---|
| 11:12 | G3b launched. Ran at **3.73 it/s x 96 = 358 img/s** for 2 h 20 m |
| 11:20 | P1a launched. Initially **~9.9 it/s x 64 = 634 img/s** |
| ~13:33 | G3b stopped advancing entirely |
| ~14:05 | P1a stopped advancing |

## Measurements

Four probes, each a separate job reading the same image tree.

| probe | conditions | metadata | data read |
|---|---|---|---|
| ioprobe1 | 3 jobs alive | listdir 0.28 s | **200 files in 339 s** (0.59 files/s) |
| ioprobe2 | after G3b terminated | listdir 0.25 s, glob 550 s | 200 files in 0.1 s *(glob-adjacent, warm)* |
| P1a relaunched | 64 workers, alone | — | **41 img/s steady state** |
| ioprobe3 | **nothing else running**, 200 **random** dirs | scandir 50,948 dirs 0.28 s, pick 0.3 s | **did not return in 18 min** (<0.18 files/s) |

**Metadata is fast; bulk reads are not.** `scandir` over 50,948 directories returns in 0.28 s while
reading 200 files does not complete in eighteen minutes with an otherwise idle cluster. That rules
out the interpretations that cost the most time to chase: it is not a dataloader deadlock, not worker
OOM, not GPU, and not a mount that is simply down.

ioprobe2's "0.1 s" is the one number not to trust: those 200 files came straight out of a `glob`, so
their metadata and likely their contents were already warm. ioprobe3 was written specifically to read
from **random** directories, which is what a shuffled dataloader actually does.

**Per-request latency, not bandwidth.** A single-threaded probe manages <0.2 files/s while P1a with
64 parallel workers manages 41 img/s. That ratio says each read costs on the order of a second and
concurrency is the only thing recovering any throughput at all.

## It is still degrading with the cluster idle (17:00 update)

Re-probed on the owner's instruction, two hours after everything was terminated:

| time | cluster state | cold random read |
|---|---|---|
| ~14:00 | 3 jobs running | **0.59 files/s** |
| ~14:53 | idle | **<0.18 files/s** (200 files unfinished in 18 min) |
| **17:00** | **idle 2 h** | **0.03 files/s** — 8 files in 258.9 s |
| **25 Aug 11:00** | **idle ~20 h** | **0.0028 files/s** — 2 files in 719 s |

**It is still getting worse after twenty hours of complete idleness**, and roughly by an order of
magnitude per observation: 0.59 -> 0.18 -> 0.03 -> 0.0028 files/s. That is about **six minutes per
72 KB file**; a single pass over the 3 M-image dataset would take **34 years**. Metadata is
unaffected throughout (`scandir` over 50,948 directories in 0.45 s on the 25 Aug probe).

**It is getting worse while nothing runs.** That removes the last version of the story in which our
load is the cause: three concurrent jobs made it worse, but the trend continued down through two
hours of complete idleness.

The 17:00 probe also shows the failure is not a uniform slowdown but **stalls on individual reads**:
the loop was capped at 120 s and ran 258.9 s, because one `open().read()` of a 72 KB file blocked for
over two minutes. Metadata stayed fast throughout (`scandir` of 50,948 directories in 0.43 s).

**VERDICT: NO-GO.** Jobs were not restarted.

## What I got wrong, and what was genuinely a mistake

**The mistake:** I launched **three concurrent jobs** with `num_workers` 128 + 256 + 32 against one
network filesystem. Worse, the package's own memory helper printed a warning at P1a's startup that I
did not read:

```
[mem] WARNING: 256 workers x ~1.2 GB anon is close to the 288 GB limit -- lower num_workers.
```

and the run then sat at **anon 87 GB + page cache 204 GB against a 288 GB cgroup** — permanent
reclaim, page cache evicted, every image re-read from the network. That is what turned a slow run
into a burst-then-freeze one, and it is a repeat of [[2026-07-17-ucloud-benchmark-oom]].

**But it is not the root cause.** Lowering `num_workers` to 64 removed the warning and the memory
pressure, and throughput was still **41 img/s** — 4 % of historical. And ioprobe3 measured
pathological latency with *nothing at all* running. The concurrency made a bad situation worse; it
did not create it.

## What this is not

- **Not the model.** Two different architectures (ConvNeXt-L 198 M, BioCLIP-2 ViT-L/14) collapsed the
  same way within an hour of each other.
- **Not compute.** GPU memory sat at 2–4 GB and fluctuated throughout — the signature of a starved
  job. A compute-bound ViT-L would hold steady, high memory.
- **Not the new BioCLIP-2 code.** G3b, which uses none of it, stalled first.

## Actions taken

All jobs terminated; the cluster is idle. `num_workers` lowered 256 → 64 in the P1a/P1b configs and
128 → 64 in G3b's, with the reason recorded in each config header. Those changes are correct
regardless of the storage issue and should stay.

## What is needed

This looks like a UCloud storage-side problem — a degraded or rehydrating backing tier for
`/work/global_lepi`. It is not fixable from the job side: at 41 img/s a single 3-epoch P1a run would
take **over 100 hours** and hold a GPU for all of it.

**Recommendation: raise it with UCloud.** The 17:00 re-probe rules out waiting as a
strategy -- it degraded further across two idle hours. Keep the cluster idle until it recovers, rather than paying for GPUs to
wait on IO. Re-run `ucloud/lepinet-ioprobe3.toml` to check — it is a one-minute job and its
`COLD RANDOM READ ... files/s` line is the single number that says whether it is safe to resume.
Historical throughput implies the pipeline needs on the order of **1000 files/s**; anything under
~100 means training will not finish in reasonable time.

## Rule to carry forward

**Run one image-heavy job at a time on this cluster, and read the `[mem]` startup lines.** The
project already knew the pipeline is IO/CPU-decode bound ([[2026-07-18-ucloud-throughput]]); what was
missing was a rule against stacking jobs, and the habit of treating a startup warning as a result
rather than as noise. Both are now in `journal/PLAN.md`'s queue discipline.


---

## Resolved: UCloud fixed it (2026-08-26)

They confirmed a storage-side fix. Verified before restarting anything:

| probe | 25 Aug (worst) | 26 Aug (after fix) |
|---|---|---|
| single-threaded random read | 0.0028 files/s | **13.2 files/s** |
| **64-thread random read** | not measured | **992.5 files/s** (3,000 files, 224 MB, 3.0 s) |
| G3b in-job throughput | — | **509 img/s** (was 358 before the outage) |

**My go/no-go threshold was mis-calibrated, and the single-threaded probe nearly cost a day.** It
reported 13.2 files/s against a 100 files/s bar and returned **NO-GO** — but that bar was borrowed
from the pipeline's ~1000 *img/s*, which is achieved by 64–256 dataloader workers reading
concurrently. Comparing a single-threaded probe against a parallel figure is the same
denominator error this project keeps making, in a new place. The parallel probe
(`ucloud/lepinet-ioprobe-par.toml`, 64 threads) is the one that gates a restart; ioprobe3 only
answers the cruder question "is the storage broken at all".

**What the outage cost:** two runs terminated mid-flight (G3b at 58 % of epoch 1, P1a at 86 % of
epoch 2) and roughly two days of wall-clock. No results were lost or corrupted — everything already
scored was on disk and committed.

**What it bought:** the `num_workers` 256 → 64 change, which was a real latent problem (the package's
own startup warning had been firing unread), and three queue-discipline rules that were missing.

## Restart

Fully serialised — one image-heavy job at a time, each `--after` the previous:

`G3b -> G3b-probe -> G3b-probeho -> G3b-eval -> P1a -> P1a-eval -> P1b -> P1b-probe -> P1b-probeho -> P1b-eval`

Compute is not the constraint here and the parallel-job experiment cost two days, so the chain is
serial by choice rather than by necessity.
