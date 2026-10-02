# The crawl that spent 58 % of the project's CPU budget

**Kind:** incident · **Status:** **RESOLVED (2026-10-01).** The first ToL-200M crawl ran on a
64-vCPU node for 63 hours and spent **4,032 core-hours -- 58 % of the project's 7,000 core-hour CPU
allocation -- in one job**, for work that needed one or two cores. Root causes, three guards that now
make a repeat impossible without the owner's sign-off, the crawl resumed on 1 vCPU, and the dataset
moved to the `datasets` drive. CPU core-hours are scarcer here than GPU hours; the project had not
written that down, and I had not asked.

## What happened

| | |
|---|---|
| job | `lepinet-tolfetch`, 12375473, submitted 2026-08-28 |
| product | `cpu-amd-zen5-64-vcpu` |
| ran | 24 h initial + **39** hourly auto-extensions = **63 h** |
| spent | **64 x 63 = 4,032 core-hours** (quota afterwards: 4,125 / 7,000 used) |
| worst case as configured | `max_time = 168h` -> **10,752 core-hours** |
| why it stopped | the UCloud token expired, ticks failed, nothing extended it |

The last line is the uncomfortable one: **the spend was stopped by an unrelated failure, not by any
control.** Had the token stayed valid, it would have run another 105 hours.

The owner reported it as "the full 256 cpu cores". The bill was for 64; but the job did *see* 256 --
see the third finding below -- so the description was accurate about what the machine looked like
from inside.

## Where the 4,032 core-hours went

| phase | duration | images attempted | rate |
|---|---|---|---|
| S3 and CDN hosts | 0-40 h | 55.8 M | ~385 img/s |
| institutional tail | 40-63 h | 6.9 M | 84 falling to **22 img/s** |

At ~1.1 cores per 240 img/s of decode -- measured on the smoke test, three days before launch --
the fast phase needed under 2 cores and the tail under one. **The node was idle for essentially all
63 hours.** The last 23 hours, about 1,470 core-hours, went at 22 img/s, throttled entirely by the
politeness caps on museum servers; no number of cores could have made that faster.

## Root causes

1. **The node was sized by intuition, not by measurement.** I picked 64 vCPU for "decode headroom"
   while holding two measurements that said otherwise: the smoke test's CPU profile, and the plan
   stage's observed CPU below 0.5 %. Neither was consulted when choosing the product.
2. **No spend ceiling.** `auto_extend` with `max_time = 168h` is an open cheque on a 64-vCPU node.
3. **Nothing computed the cost before submission.** Core-hours never appeared in the launch, the
   commit message, or the report to the owner.

## Three guards

| layer | catches it | file |
|---|---|---|
| **pre-submit hook** | the moment a spec is submitted, by any agent session in this repo | `.claude/settings.json` -> `ucloud/budget_check.py --hook` |
| **CI test** | any committed spec | `tests/test_ucloud_budget.py` |
| **runtime guard** | a spec that got through anyway: the crawler refuses to start on more CPUs than `--max-cpus` | `dev/082_tol_crawler.py`, `stage_fetch` |

The rule, for CPU products: worst case = vCPU x (`max_time` if `auto_extend` else `hours`). More
than **8 vCPU** or more than **300 worst-case core-hours** is refused unless the spec carries
`# budget-approved: <who/when/why>`. `auto_extend` without `max_time` is refused outright. GPU
products are reported, not gated.

Proved, not assumed: the hook blocked a live `ucloud q submit` of the old spec in this session, and
the runtime guard refused to run on a 24-core laptop. The checker flagged all three ToL specs --
plan and split each carried a 3,072 core-hour worst case as well -- and the resized ones are 24, 8
and 150.

## Three things found while fixing it, each worth more than the fix

**UCloud reports 256 cores to every standard API.** On a 2-vCPU job: `os.cpu_count() = 256`,
`len(os.sched_getaffinity(0)) = 256`, and only `/sys/fs/cgroup/cpu.max = 200000 100000` says 2.
So any library that sizes a thread pool from `os.cpu_count()` -- pyarrow does -- starts 256 threads
on whatever the job was given. The crawler now reads the cgroup first and pins pyarrow's pools and
the BLAS variables to it. **This applies to every job in the project, GPU ones included**, and is in
`CLAUDE.md` as an invariant.

**The blocked-host breaker sent 1,662,674 requests to institutions that had refused us.** It checked
only at part boundaries, so each of eight blocked hosts received its whole first part -- 50,000
requests -- after saying no. And it used a lifetime ratio: `medialib.naturalis.nl` served ~137 k
images, then banned us, and the ratio took **1.26 million** further forbidden requests to cross 90 %.
Now: a sliding 200-outcome window, checked after the host semaphore immediately before each request
leaves, and blocked hosts persisted to `blocked_hosts.json` so a resume never re-probes them. Measured
on the two hosts that block us: **40 and 42 requests** before stopping, down from 50,000. A first
fix that checked before the semaphore still let all 120 queued requests through; the test caught it.

**Full JPEG decodes were the remaining CPU cost.** The remaining hosts are herbaria serving 50-100 MP
specimen scans that get shrunk to 256 px. PIL's `draft()` makes libjpeg decode at 1/2, 1/4 or 1/8
scale in the DCT domain: on a 54 MP scan, **250 ms -> 22 ms (11x)** for a mean pixel difference of
0.29/255. That is what lets the tail run on one core.

## The resume

`remaining` (new stage, reads parquet footers) on the moved manifest:

| | rows |
|---|---|
| finished parts | 62,593,470 |
| **still to crawl** | **6,661,357 over 17 hosts** |
| on blocked hosts, skipped | 18,305,238 |

The 17 are almost all herbaria at the politeness cap of 4 -- `sweetgum.nybg.org` alone holds 2.13 M.
Their speed is fixed by politeness, so duration is fixed and **spend scales only with node size: 1
vCPU, not 2, not 64.** `max_time = 150h` -> **150 core-hours worst case.** If the tail outlasts it,
the crawl resumes from where it stopped -- that is now true inside a part too, because an image
already on disk and complete (JPEG ends `FFD9`) is recorded without a request, and image writes are
atomic so a kill cannot leave a truncated file that looks finished.

Also fixed on the way: `split` now streams, so it no longer needs a large node's RAM for the 50 M-row
iNaturalist part -- which is why it had been on 64 vCPU too.

## The move to the `datasets` drive

`datasets` is a separate project drive, **`/12383016`**, alongside `global_lepi/` and
`flemming_helsing/` -- not the `datasets/` folder inside the owner's Member Files. The corpus is now at
**`/12383016/treeoflife_200m/{manifest,images}`**; it was inside the git sync folder
`/12347837/repos/lepinet/data/tol/`.

Moved with UCloud's server-side `POST /api/files/move` -- **0.04 s, a rename, zero job core-hours**.
A `mv` inside a job would have looked equivalent and been a 58 M-file copy: two separate mounts are
two mount points, and `rename(2)` across mount points fails with `EXDEV`, so `mv` silently copies.

Verified: the manifest by a full paginated census (354 hosts, 2,050 parts, 1,538 finished, identical
before and after); the images by 45 randomly sampled files from three hosts' recorded paths, 45
present.

**A trap in the owner's `ucloud-api` tool, reported not fixed:** `Files.list_path` requests one page
of 250 and ignores the continuation token, so `ucloud files ls`, and anything built on `walk_files`,
silently shows only the first 250 entries of a directory. My first census of the manifest said
"250 hosts, 927 parts -- identical", which was true and meaningless; the paginated recount said 354
and 2,050. Anything that downloads a directory through `walk_files` would miss files the same way.

## What I would do differently

Compute the core-hours before submitting anything, put the number in the message to the owner, and
size the node from the CPU profile already measured. None of that needed a tool. The tools exist now
because "remember to" is not a control -- and because the same mistake was made three times in one
session (fetch, plan, split), which is a habit, not a slip.

## Open

- **Ask for research access** from the nine hosts in `blocked_hosts.json` -- observation.org and
  medialib.naturalis.nl hold the most. 18.3 M rows sit behind them.
- **The tail's real rate**: measured in the first hours of the resume; if `sweetgum.nybg.org` runs as
  slowly as the end of the first crawl suggests, it alone could outlast 150 h, and the decision to
  continue goes back to the owner with a number attached.
