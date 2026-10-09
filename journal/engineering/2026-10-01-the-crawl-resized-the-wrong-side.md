# The crawl resized the wrong side, and a host that drops packets froze it

**Kind:** incident · **Status:** **OPEN (2026-10-01) -- re-crawl running on the workstation; stall root-caused 2026-10-02 (deadlock).** The
TreeOfLife-200M crawl stored every image at **256 px on the long side**. Training takes a square
crop of the *short* side and resizes it to 256, so every crawled image reached the model upsampled
(1.33x for a 4:3 photo) -- softer than our own Lepidoptera, and short of the detail the input can
use. The 58.3 M images already fetched (1.05 TB) have to be fetched again at **256 px on the short
side**. Separately, the crawl's recurring "stall" was traced: a host that silently drops
connections was being treated as a host asking us to slow down, and was retried forever. Both are
fixed in `dev/082`. The re-crawl runs on the workstation: **zero UCloud core-hours**.

## What went wrong, and whose mistake it was

The owner asked for "low quality images" sized for the model, and the model input is 256 x 256. I
implemented "long side <= 256" without checking it against the training pipeline, which is
`Resize(460)` (short side to 460, square crop) followed by `aug_transforms(size=256)`. What sets
the detail at the input is the number of original pixels in the square crop -- the short side:

| | stored | short side | into the 256 input |
|---|---|---:|---|
| our Lepidoptera (`global_lepi`) | 512 x 384 | 384 | downsampled 1.5x |
| first crawl (long side 256) | 256 x 192 | 192 | **upsampled 1.33x** |
| re-crawl (short side 256) | 341 x 256 | 256 | 1:1 |

A genuine mistake, and an expensive one in wall-clock: the size choice should have been derived
from the training transform, not from the word "256". The owner caught it.

Re-crawling costs ~2.2x the storage (32.4 kB per image measured, vs 18.1 kB): ~2.3 TB for ~70 M
images. Extreme aspect ratios (panoramas, herbarium strips) are capped at 4x on the long side.
JPEG draft decoding still applies: libjpeg's draft never goes below the requested size on either
side, so the short side stays >= 256 before the LANCZOS resize.

## The stall: connection timeouts were counted as throttling

Two UCloud crawls froze for hours with every counter still. Without SSH there was nothing to inspect,
so a watchdog was added (`--stall-minutes`: dump every thread's stack and every task's await point,
exit 75, restarted by a bounded loop). It fired on its first local test, which settled one thing at
once: the workstation has no memory ceiling, so the "cgroup memory.max" theory was not the cause.

The dump: all worker threads idle, all memory slots free, all 100 remaining tasks waiting on one
host's concurrency budget -- `sweetgum.nybg.org` (NYBG herbarium, **2.23 M rows**). Probing it:

* `http://sweetgum.nybg.org/...` (the catalog's URLs): **connection timeout** -- port 80 drops packets;
* `https://...` with our user agent: **403** in 0.4 s; with a browser user agent: 200.

So sweetgum refuses crawlers. The crawler never learned that, because aiohttp's
`ConnectionTimeoutError` is a subclass of `asyncio.TimeoutError`, which the crawler's handler treats
as *throttling* -- halve the budget, cool down 5 s, retry -- and throttling never feeds a breaker.
Each row cost 4 x (15 s connect timeout + 5 s cooldown) at concurrency 1: one row a minute,
forever. It now counts as `dead:connect_timeout` toward the dead-host breaker, like DNS and TLS
failures, and on a 40-row test the host was stopped and recorded after the probe window.

We do **not** switch to https with a browser user agent: the 403 is the institution's answer, and
the crawler's rule is to honour refusals, not to disguise itself. Sweetgum's 2.23 M rows (herbarium
sheets) join the substitution pass, like the other herbaria.

**The real stall, found 2026-10-02: a lock-order deadlock.** Sweetgum explained one frozen host,
not the UCloud stalls in which *every* host froze. The workstation re-crawl then stalled six times
in 18 h (each caught by the watchdog, 20 min lost plus restart), and every dump showed the same
picture: **all 64 memory slots held, all worker threads idle, every task waiting inside
`fetch_one`, iNaturalist holding 239 of its 256 host slots.** The memory slot is taken after the
response headers arrive and held through the encode. A body that failed *mid-stream*
(`ClientPayloadError`, `ServerDisconnectedError`, a read timeout -- routine at this volume) went
back to the top of the retry loop and **waited for a host slot while still holding its memory
slot**, while the requests holding host slots waited for a memory slot. Once every memory slot
belonged to a retrier, nothing could move. With UCloud's 16 slots that took minutes -- which is why
both UCloud stalls hit within ~20 minutes of start, and why the "memory.max" coincidence looked like
a cause. Fix: release the memory slot at the top of every attempt (the failed body is discarded
anyway). `tests/test_tol_crawler.py` reproduces it with one slot of each kind and two
rows whose first body read fails: it times out without the fix and passes with it. After the
restart: **559 img/s**, CPU-bound on the 6 pinned cores.

## The re-crawl

* **Where:** the workstation, `/data/au761367/tol256s/` (manifest copied from the 2026-10-01
  snapshot without its metas; fresh `images/`), pinned to cores 0-5 at nice 10, `run.sh` with the
  restart loop, log in `crawl.log`. `--file-root` maps the `global_lepi` fill rows onto the local
  copy, so they are read from disk, not fetched.
* **Measured CPU:** 38.9 core-seconds for 4,450 images on a mixed-host sample (iNaturalist, fill,
  eight institutional hosts) = **8.7 ms per image**, 99.7 % success, 97 % of images exactly 256 on the
  short side (the rest are smaller originals, kept as they are). For ~70 M images: ~170
  core-hours -- which on UCloud would be ~6 % of what remains of the allocation, and here is zero.
* **Afterwards:** pack into shards, upload to `/12383016/treeoflife_200m/` (~2.3 TB; ~8 h at the
  78 MB/s measured for the audit upload), unpack with a 1 vCPU job (tens of core-hours at most).
* **Images that cannot be re-fetched:** 580,310 were downloaded from servers that now refuse us or
  are gone. Their long-side-256 copies in the old `images/` stay as fallbacks, flagged as such; the
  old directory is not deleted until the re-crawl is merged.
* **UCloud spec:** `ucloud/lepinet-tolfetch.toml` is kept as a fallback and now writes `images_s256/`,
  so it can never mix the two sizes in one directory. Job 12409366 (stalled, long side) should be
  terminated; it has been burning an idle core-hour per hour.

## The retry of tripped servers (2026-10-07 to 10-09)

By 2026-10-07, 75 servers had tripped the breaker, 55 of them since 1 October and several within
the same second, which looked like a blip on our side. Probed by hand: `data.huh.harvard.edu` and
`live.staticflickr.com` answered again, `ecdysis.org` still returned 403. A second crawl
(`tol256s/retry1/`: a manifest of symlinks to those three hosts' parts, the same `images/`, cores 6-7)
re-ran them.

* **ecdysis.org:** 40 requests, 40 x 403. Written off.
* **Harvard:** 56,652 images in 47 h, **0.39 img/s**: the files are 30+ MB herbarium scans at 4
  connections, so ~362 k remaining rows would have taken ~11 days. Stopped on the owner's call
  (2026-10-09) and added to `retry1/manifest/blocked_hosts.json` with reason `slow`; its remaining
  rows join the substitution pass.
* **Flickr:** 480,376 images (of ~782 k rows) at 2.7-4 img/s under constant 429s. After the restart
  that dropped Harvard, every request got 429, including a single `curl` from this machine
  (CloudFront `FunctionGeneratedResponse`), so the block is per IP, not per connection. The crawler
  kept sending ~3 requests/s into it, so it was stopped too. ~300 k rows remain. **Lesson:** the
  budget halves on 429 but never drops below one connection, and a 429 without `Retry-After` is
  retried at once; a host that answers *only* 429 for minutes should pause the host for an hour, not
  keep one slot hammering it. Retry Flickr after a cool-down, probing with one `curl` first.
* **Fixed 2026-10-09** (`HostBudget.acquire` / `on_throttle`, `tests/test_tol_crawler.py`): the
  pause is now checked after a request takes its slot, so queued requests wait it out too; it
  doubles from 5 s to an hour while the 429s continue (in-flight answers do not escalate it twice);
  a 429/503 no longer spends one of the row's attempts, which had been writing rows off as
  `exhausted_attempts` during a block; and a host throttled for 6 h with no success is blocked as
  `throttled`, leaving its parts resumable for `--retry-blocked`. The running main crawl keeps the
  old code until its next restart.
