"""Fetch TreeOfLife-200M images from their source servers, under our own data policy.

`imageomics/TreeOfLife-200M` on HuggingFace hosts **metadata only** -- `catalog.parquet` is 16.8 GB
of URLs and taxonomy across 233,055,986 rows, and no image bytes. The images live on the servers of
the institutions that published them, so acquiring the corpus is a crawl, not a download.

Applying our own policy (`min_img_per_spc = 50`, cap 2,000/species) leaves **88.1 M images across
203,878 species** -- see `journal/2026-08-27-tol-at-our-policy-and-the-head-scaling-problem.md`.

## The design constraint, measured rather than assumed

Sampling 1,015,497 catalog rows gives 239 distinct hosts with a brutally skewed distribution:

      57.9 %  inaturalist-open-data.s3.amazonaws.com     (S3 -- built to be hammered)
       9.3 %  observation.org
       3.6 %  mediaphoto.mnhn.fr
       ...    41 hosts reach 95 %; the rest is a tail of museum and herbarium servers

**This is why a single global concurrency limit is the wrong design**, and it is the main thing this
tool does differently from `gbifxdl`. One semaphore of 128 sends 128 simultaneous requests to
whichever host happens to be next in the file -- fine for an S3 bucket, fatal for a university
herbarium that serves a few requests per second and will either fall over or ban us. And because the
catalog is physically partitioned by server (`base_dataset_file_path` contains `server=...`),
iterating it in order does exactly that: it hammers one host at a time, at maximum rate.

So the unit of scheduling is the **host**, not the row:

* the manifest is written **partitioned by host**, so each worker owns one host's queue;
* every host has its **own concurrency budget**, from a policy table (S3/CDN: high; unknown: low);
* budgets are **adaptive** -- a 429/503 halves the budget immediately, sustained success grows it
  back slowly, so we discover each server's real tolerance instead of guessing it;
* a slow or dead host blocks **only its own worker**, never the pipeline.

## Politeness is not optional here

These are public-good servers run by museums on small budgets, and the entire dataset is available
to us only because they serve it. The crawler identifies itself with a contact address, honours
`Retry-After`, backs off hard on the first sign of stress, and caps even the friendliest institutional
host well below what it could probably take. The S3 and CloudFront origins absorb the bulk of the
volume precisely so the small hosts do not have to.

## Quality control

Every image is verified before it counts as acquired: HTTP status, declared content type, magic
bytes, a full PIL decode (not merely `verify()`, which does not catch truncation), and a minimum
dimension. Survivors are resized to fit `--size` and re-encoded to JPEG, which is what makes 88 M
images 2.2 TB rather than 6.8. A perceptual-ish content hash is recorded per image so the
post-pass can drop exact duplicates and the "image unavailable" placeholders that several servers
return with HTTP 200.

## Stages

    # 1. plan: stream the catalog, apply the policy, write a host-partitioned manifest
    python dev/082_tol_crawler.py plan --out data/tol/manifest --min-img 50 --cap 2000

    # 2. fetch: crawl. Resumable, restartable, safe to run many times
    python dev/082_tol_crawler.py fetch --manifest data/tol/manifest --images /work/tol/images

    # 3. report: what we have, what failed, and why
    python dev/082_tol_crawler.py report --manifest data/tol/manifest

`fetch` is idempotent. A manifest part is skipped once its metadata parquet exists; within a part,
rows already recorded are skipped. Killing the job at any point loses at most one part's in-flight
work, so it survives UCloud time limits without special handling.

No new dependencies: `aiohttp` only, with image work on a thread pool. `aiofiles` buys nothing when
the encode already has to leave the event loop, and `aiohttp_retry` cannot express per-host adaptive
budgets, which is the whole point.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import re
import time
import urllib.parse as urlparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CATALOG = "datasets/imageomics/TreeOfLife-200M/dataset/catalog.parquet"
PLAN_COLS = ["uuid", "source_url", "kingdom", "phylum", "class", "order", "family",
             "genus", "species", "data_source", "source_id"]

CONTACT = os.environ.get("TOL_CRAWLER_CONTACT", "guillaumemougeot1@gmail.com")
# The "Mozilla/5.0 (compatible; <name>; +<contact>)" form is what Googlebot, bingbot and every other
# well-behaved crawler sends. It is self-identifying -- the bot is named and reachable -- while still
# passing the naive `UA must start with Mozilla` filters that a surprising number of institutional
# servers use. Measured: it unblocks nothing that a bare token does not, but it costs nothing either.
#
# What it does NOT do is impersonate a browser. Two hosts (observation.org, 9.3 % of the corpus, and
# mediaphoto.mnhn.fr, 3.6 %) return 403 to everything except a literal browser UA string. Those are
# explicit bot blocks, and the correct response is `--on-blocked stop` plus an access request to the
# institution, not a better disguise. See `blocked_hosts.json` written by `report`.
USER_AGENT = f"Mozilla/5.0 (compatible; lepinet-tol-crawler/1.0; +mailto:{CONTACT})"
BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Per-host concurrency ceilings. The bulk of the corpus sits on object stores that are designed for
# exactly this and can be driven hard; everything else is somebody's institutional server and gets
# a deliberately small budget. Unknown hosts inherit DEFAULT_CAP, which is low on purpose -- the
# adaptive controller raises it only if the host demonstrably tolerates more.
HOST_CAPS: dict[str, int] = {
    "inaturalist-open-data.s3.amazonaws.com": 256,
    "d2seqvvyy3b8p2.cloudfront.net": 128,
    "live.staticflickr.com": 64,
    "jbrj-public-img.s3-sa-east-1.amazonaws.com": 64,
    "content.eol.org": 16,
    "observation.org": 12,
    "images.ala.org.au": 12,
    "api.idigbio.org": 12,
    "www.boldsystems.org": 8,
    "fm-digital-assets.fieldmuseum.org": 8,
}
DEFAULT_CAP = 4          # unknown institutional host
MIN_CAP = 1
START_FRACTION = 0.25    # begin at a quarter of the ceiling and earn the rest

VALID_CT = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp",
            "image/tiff", "image/bmp"}
MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"GIF8", "gif"),
         (b"RIFF", "webp"), (b"II*\x00", "tiff"), (b"MM\x00*", "tiff"), (b"BM", "bmp"))

STATUS_OK = "ok"


# iNaturalist's open-data bucket serves every photo at four sizes, and the catalog always points at
# `original`. Measured on a sample:
#
#     original  1109 KB   ~1600-2048 px
#     large      382 KB    1024 px
#     medium     108 KB     500 px      <- 10.2x less transfer than original
#     small       31 KB     240 px      <- below our 256 px target, too lossy
#
# We resize everything to 256 px anyway, so fetching `original` means pulling ~1 MB to keep ~25 KB.
# At 58 % of 88 M images that is the difference between ~56 TB and ~5.5 TB of transfer, and it is
# pure waste: the discarded resolution never reaches the model. `medium` at 500 px leaves comfortable
# headroom above 256 px, so a later decision to train at 384 px would not require re-crawling.
#
# Any variant can 404 for an individual photo, so a miss falls back to the catalog URL rather than
# failing the row.
VARIANT_HOSTS = {"inaturalist-open-data.s3.amazonaws.com", "static.inaturalist.org"}


def variant_url(url: str, variant: str) -> str | None:
    """Rewrite an iNaturalist photo URL to a smaller size, or None if not applicable."""
    if not variant or variant == "original":
        return None
    if host_of(url) not in VARIANT_HOSTS:
        return None
    base, dot, ext = url.rpartition(".")
    if not dot or "/original" not in base:
        return None
    return base.replace("/original", f"/{variant}") + dot + ext


def host_of(url: str) -> str:
    try:
        return urlparse.urlparse(url).netloc.lower()
    except Exception:
        return "invalid"


def slug(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", h)[:80] or "unknown"


# ---------------------------------------------------------------------------------------------
# Stage 1 -- plan
# ---------------------------------------------------------------------------------------------

def n_row_groups(path: str) -> int:
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(path, "rb") as f:
        return pq.ParquetFile(f).metadata.num_row_groups


def iter_row_groups(path: str, columns: list[str], first: int, last: int,
                    workers: int, prefetch: int):
    """Yield ``(index, table)`` in order, fetching ahead with a thread pool.

    Reading row-groups one at a time over HTTP leaves the link idle for the whole decode, and the
    decode idle for the whole fetch. Observed on the first attempt at this scan: **CPU below 0.5 %
    and network alternating between 0 and 15 MB/s** -- i.e. almost all of the wall clock was one
    stream waiting on latency. At ~11 row-groups/min the two passes over 1,838 groups would have
    taken most of a day.

    So fetch ``prefetch`` groups concurrently while the consumer works. Results are still yielded
    **in order**, which matters because pass 2 applies a per-species cap by taking the first N rows
    it sees: out-of-order reads would still respect the cap but would select a different, seed-free
    subset on every run, and a corpus you cannot rebuild identically is a corpus you cannot debug.

    Each worker keeps its own file handle in thread-local storage. A `ParquetFile` wraps a single
    seekable stream and is **not** thread-safe -- sharing one handle across threads produces
    interleaved seeks and silently corrupt batches rather than an exception.
    """
    import threading
    from collections import deque
    from huggingface_hub import HfFileSystem

    local = threading.local()

    def read(i: int):
        if not hasattr(local, "pf"):
            local.fs = HfFileSystem()
            local.fh = local.fs.open(path, "rb")
            local.pf = pq.ParquetFile(local.fh)
        return local.pf.read_row_group(i, columns=columns)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending: deque = deque()
        todo = iter(range(first, last))
        for _ in range(prefetch):
            try:
                i = next(todo)
            except StopIteration:
                break
            pending.append((i, ex.submit(read, i)))
        while pending:
            i, fut = pending.popleft()
            yield i, fut.result()
            try:
                j = next(todo)
            except StopIteration:
                continue
            pending.append((j, ex.submit(read, j)))

def stage_plan(a):
    """Stream the catalog, apply the data policy, write a host-partitioned manifest.

    Two passes over the taxonomy columns, because the cap needs per-species totals before it can
    decide what to keep and 233 M rows of URLs do not fit in memory. Pass 1 counts species; pass 2
    emits rows for species that clear the floor, stopping each species at the cap.

    The species key is `genus + " " + species`: a bare epithet is not a species, "alba" occurs in
    hundreds of genera, and keying on it would silently merge unrelated taxa.
    """
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    counts_path = out / "species_counts.json"
    if counts_path.exists() and not a.recount:
        counts = {k: v for k, v in json.loads(counts_path.read_text()).items()}
        print(f"reusing species counts for {len(counts):,} species ({counts_path})")
    else:
        nrg = n_row_groups(CATALOG)
        print(f"pass 1/2: counting images per species over {nrg:,} row-groups "
              f"({a.readers} readers) ...")
        counts = Counter()
        t0 = time.monotonic()
        for i, tb in iter_row_groups(CATALOG, ["genus", "species"], 0, nrg,
                                     a.readers, a.prefetch):
            g = tb["genus"].to_pylist(); s = tb["species"].to_pylist()
            counts.update(f"{(gg or '').strip()} {(ss or '').strip()}".strip()
                          for gg, ss in zip(g, s))
            if i % 100 == 0:
                el = time.monotonic() - t0
                print(f"  row-group {i}/{nrg}: {len(counts):,} species "
                      f"| {i/max(el,1e-9)*60:.0f} rg/min", flush=True)
        counts.pop("", None)
        counts_path.write_text(json.dumps(counts))
        print(f"wrote {counts_path}")

    keep = {k for k, v in counts.items() if v >= a.min_img}
    total = sum(min(counts[k], a.cap) for k in keep)
    print(f"policy: min {a.min_img} / cap {a.cap} -> {len(keep):,} species, ~{total:,} images")

    nrg = n_row_groups(CATALOG)
    print(f"pass 2/2: emitting host-partitioned manifest over {nrg:,} row-groups ...")
    writers: dict[str, tuple] = {}
    emitted: Counter = Counter()
    n_rows = 0
    t0 = time.monotonic()
    if True:
        for i, _tb in iter_row_groups(CATALOG, PLAN_COLS, 0, nrg, a.readers, a.prefetch):
            t = _tb.to_pydict()
            rows_by_host: dict[str, list] = defaultdict(list)
            for j in range(len(t["uuid"])):
                sp = f"{(t['genus'][j] or '').strip()} {(t['species'][j] or '').strip()}".strip()
                if sp not in keep or emitted[sp] >= a.cap:
                    continue
                url = t["source_url"][j]
                if not url:
                    continue
                emitted[sp] += 1
                n_rows += 1
                rows_by_host[host_of(url)].append(
                    {"uuid": t["uuid"][j], "url": url, "species": sp,
                     "genus": (t["genus"][j] or "").strip(),
                     "family": (t["family"][j] or "").strip(),
                     "order": (t["order"][j] or "").strip(),
                     "data_source": t["data_source"][j] or "",
                     "source_id": str(t["source_id"][j] or "")})
            for h, rows in rows_by_host.items():
                d = out / f"host={slug(h)}"
                d.mkdir(exist_ok=True)
                if h not in writers:
                    tbl = pa.Table.from_pylist(rows)
                    writers[h] = (pq.ParquetWriter(d / "part-00000.parquet", tbl.schema), tbl.schema)
                    writers[h][0].write_table(tbl)
                else:
                    w, schema = writers[h]
                    w.write_table(pa.Table.from_pylist(rows, schema=schema))
            if i % 100 == 0:
                el = time.monotonic() - t0
                print(f"  row-group {i}/{nrg}: {n_rows:,} kept "
                      f"| {i/max(el,1e-9)*60:.0f} rg/min", flush=True)
    for w, _ in writers.values():
        w.close()
    (out / "plan_summary.json").write_text(json.dumps(
        {"min_img": a.min_img, "cap": a.cap, "species": len(keep), "rows": n_rows,
         "hosts": len(writers)}, indent=2))
    print(f"\nmanifest: {n_rows:,} rows over {len(writers):,} hosts -> {out}")


# ---------------------------------------------------------------------------------------------
# Stage 2 -- fetch
# ---------------------------------------------------------------------------------------------

@dataclass
class HostBudget:
    """Adaptive concurrency for one host.

    Servers do not publish what they tolerate, so we discover it. Start at a quarter of the ceiling;
    every ``grow_after`` consecutive successes add one slot; any 429/503 (or a timeout, which on a
    small server usually means the same thing) halves the budget at once. Halving on the *first*
    sign of stress and recovering slowly is deliberately asymmetric -- being wrong in the greedy
    direction gets us banned, being wrong in the shy direction costs throughput on a job that runs
    for days anyway.
    """
    host: str
    cap: int
    cur: int = 0
    ok_streak: int = 0
    grow_after: int = 64
    cooldown_until: float = 0.0
    stats: Counter = field(default_factory=Counter)
    blocked: bool = False

    def __post_init__(self):
        self.cur = max(MIN_CAP, int(self.cap * START_FRACTION))
        self._sem = asyncio.Semaphore(self.cur)
        self._slack = 0            # extra permits owed back when we shrink

    async def acquire(self):
        if self.cooldown_until > time.monotonic():
            await asyncio.sleep(self.cooldown_until - time.monotonic())
        await self._sem.acquire()

    def release(self):
        # When we have shrunk, swallow permits instead of releasing them until the debt is paid.
        if self._slack > 0:
            self._slack -= 1
        else:
            self._sem.release()

    def on_success(self):
        self.stats["ok"] += 1
        self.ok_streak += 1
        if self.ok_streak >= self.grow_after and self.cur < self.cap:
            self.ok_streak = 0
            self.cur += 1
            self._sem.release()          # hand out one more permit

    def on_throttle(self, retry_after: float | None = None):
        self.stats["throttled"] += 1
        self.ok_streak = 0
        shrink = self.cur - max(MIN_CAP, self.cur // 2)
        self.cur -= shrink
        self._slack += shrink
        self.cooldown_until = time.monotonic() + (retry_after if retry_after else 5.0)

    def on_error(self):
        self.stats["error"] += 1
        self.ok_streak = 0

    def note_forbidden(self, probe: int, ratio: float):
        """Circuit breaker for a host that is refusing us outright.

        A handful of 403s is normal -- individual records get withdrawn or embargoed. A host that
        returns 403 to *almost everything* is not losing records, it is blocking the crawler, and
        continuing to ask is both futile and rude. After ``probe`` attempts, if the forbidden rate
        exceeds ``ratio``, we stop that host and record it for a human to follow up.
        """
        self.stats["forbidden"] += 1
        seen = self.stats["ok"] + self.stats["forbidden"] + self.stats["error"]
        if seen >= probe and self.stats["forbidden"] / seen >= ratio:
            self.blocked = True


def verify_and_encode(body: bytes, size: int, quality: int, min_dim: int):
    """Decode fully, reject junk, resize, re-encode as JPEG. Runs on a worker thread.

    A *full* decode rather than ``Image.verify()``: verify() checks the header and misses truncated
    payloads, which are the common failure when a server drops a connection mid-response and still
    returned HTTP 200. The decode is the expensive part of this whole pipeline and the reason image
    work belongs on a thread pool rather than the event loop.
    """
    from PIL import Image
    if not any(body.startswith(m) for m, _ in MAGIC):
        return None, "bad_magic", None
    try:
        img = Image.open(io.BytesIO(body))
        img.load()                                   # full decode; catches truncation
    except Exception as e:
        return None, f"decode_fail:{type(e).__name__}", None
    if min(img.size) < min_dim:
        return None, f"too_small:{img.size[0]}x{img.size[1]}", None
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > size:
        scale = size / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    out = buf.getvalue()
    # Hash the *decoded, resized* pixels, not the transport bytes: two servers delivering the same
    # photograph at different compression must collide, and that is what finds the placeholders.
    content_hash = hashlib.sha1(img.tobytes()).hexdigest()[:16]
    return out, STATUS_OK, (content_hash, img.size)


class Fetcher:
    def __init__(self, a):
        self.a = a
        self.images = Path(a.images)
        self.pool = ThreadPoolExecutor(max_workers=a.workers)
        self.budgets: dict[str, HostBudget] = {}
        self.totals = Counter()
        self.t0 = time.monotonic()

    def budget(self, host: str) -> HostBudget:
        """`host` must be the real hostname, never the directory slug.

        This was a bug worth keeping a note about: the manifest directory is `host=<slug>`, with dots
        replaced by underscores, so looking `HOST_CAPS` up by directory name misses *every* entry and
        silently drops the whole crawl to `DEFAULT_CAP = 4` -- including the S3 bucket that is 58 % of
        the corpus and could take 256. Nothing errors; the crawl is just 60x slower than it should
        be, which is exactly the kind of failure a smoke test only catches if you read the numbers.
        Hosts are now resolved from the URLs in the manifest, which cannot drift from reality.
        """
        if host not in self.budgets:
            self.budgets[host] = HostBudget(host, HOST_CAPS.get(host, DEFAULT_CAP))
        return self.budgets[host]

    async def fetch_one(self, session, row, b: HostBudget):
        import aiohttp
        # Try the cheap size variant first; fall back to the catalog URL if it is missing.
        alt = variant_url(row["url"], self.a.variant)
        urls = [alt, row["url"]] if alt else [row["url"]]
        url = urls[0]
        for attempt in range(self.a.attempts):
            await b.acquire()
            try:
                async with session.get(url, allow_redirects=True) as r:
                    if r.status in (429, 503):
                        ra = r.headers.get("Retry-After")
                        b.on_throttle(float(ra) if ra and ra.isdigit() else None)
                        continue
                    if r.status >= 400:
                        if r.status == 403:
                            b.note_forbidden(self.a.block_probe, self.a.block_ratio)
                        else:
                            b.on_error()
                        if r.status == 404 and len(urls) > 1 and url == urls[0]:
                            url = urls[1]          # variant missing for this photo; use the original
                            self.totals["variant_fallback"] += 1
                            continue
                        if r.status in (404, 410, 403):
                            return None, f"http_{r.status}", None   # permanent; do not retry
                        continue
                    ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                    if ct and ct not in VALID_CT and not ct.startswith("image/"):
                        b.on_error()
                        return None, f"bad_content_type:{ct}", None
                    cl = r.headers.get("content-length")
                    if cl and int(cl) > self.a.max_bytes:
                        return None, f"too_large:{cl}", None
                    body = await r.read()
                    b.on_success()
            except asyncio.TimeoutError:
                b.on_throttle()                # a timeout from a small host is usually overload
                continue
            except aiohttp.ClientError as e:
                b.on_error()
                if attempt == self.a.attempts - 1:
                    return None, f"client_error:{type(e).__name__}", None
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            finally:
                b.release()

            if len(body) > self.a.max_bytes:
                return None, f"too_large:{len(body)}", None
            jpg, status, meta = await asyncio.get_running_loop().run_in_executor(
                self.pool, verify_and_encode, body, self.a.size, self.a.quality, self.a.min_dim)
            if jpg is None:
                return None, status, None
            return jpg, STATUS_OK, meta
        return None, "exhausted_attempts", None

    async def run_part(self, session, part: Path, host: str):
        meta_path = part.with_suffix(".meta.parquet")
        tmp_path = part.with_suffix(".meta.parquet.tmp")
        if meta_path.exists() and not self.a.retry_failed:
            return
        done: set[str] = set()
        if meta_path.exists():
            d = pq.read_table(meta_path, columns=["uuid", "status"]).to_pydict()
            done = {u for u, s in zip(d["uuid"], d["status"])
                    if s == STATUS_OK or not self.a.retry_failed}
        rows = pq.read_table(part).to_pylist()
        rows = [r for r in rows if r["uuid"] not in done]
        if not rows:
            return
        b = self.budget(host)
        results: list[dict] = []

        async def one(row):
            jpg, status, meta = await self.fetch_one(session, row, b)
            rec = {"uuid": row["uuid"], "species": row["species"], "status": status,
                   "path": "", "content_hash": "", "width": 0, "height": 0, "bytes": 0}
            if jpg is not None:
                d = self.images / slug(row["species"])
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{row['uuid']}.jpg"
                p.write_bytes(jpg)
                ch, (w, h) = meta
                rec |= {"path": str(p.relative_to(self.images)), "content_hash": ch,
                        "width": w, "height": h, "bytes": len(jpg)}
            results.append(rec)
            self.totals[status if status == STATUS_OK else "fail"] += 1
            self.totals["total"] += 1

        await asyncio.gather(*(one(r) for r in rows))
        pq.write_table(pa.Table.from_pylist(results), tmp_path)
        os.replace(tmp_path, meta_path)

    async def main(self):
        import aiohttp
        parts: list[tuple[Path, str]] = []
        for d in sorted(Path(self.a.manifest).glob("host=*")):
            ps = [p for p in sorted(d.glob("part-*.parquet")) if not p.name.endswith(".meta.parquet")]
            if not ps:
                continue
            # The real hostname, read from the data. The directory slug is lossy (dots -> "_") and
            # must never be used as a lookup key -- see Fetcher.budget.
            first = pq.read_table(ps[0], columns=["url"]).column("url")[0].as_py()
            host = host_of(first)
            parts.extend((p, host) for p in ps)
        if self.a.limit_hosts:
            keep = {h for _, h in parts}
            keep = set(sorted(keep)[: self.a.limit_hosts])
            parts = [(p, h) for p, h in parts if h in keep]
        by_host: dict[str, list[Path]] = defaultdict(list)
        for p, h in parts:
            by_host[h].append(p)
        print(f"{len(parts)} parts over {len(by_host)} hosts")

        timeout = aiohttp.ClientTimeout(total=self.a.timeout, connect=15)
        conn = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=600)
        async with aiohttp.ClientSession(timeout=timeout, connector=conn,
                                         headers=BASE_HEADERS) as session:
            async def host_worker(host, plist):
                for p in plist:
                    b = self.budget(host)
                    if b.blocked:
                        self.totals["skipped_blocked"] += len(plist)
                        print(f"  [blocked] {host}: {b.stats['forbidden']} forbidden of "
                              f"{sum(b.stats.values())} -- stopping this host. Request research "
                              f"access from the institution rather than retrying.", flush=True)
                        break
                    await self.run_part(session, p, host)

            reporter = asyncio.create_task(self.report_loop())
            await asyncio.gather(*(host_worker(h, ps) for h, ps in by_host.items()))
            reporter.cancel()
        self.summary()

    async def report_loop(self):
        while True:
            await asyncio.sleep(self.a.report_every)
            self.summary()

    def summary(self):
        dt = time.monotonic() - self.t0
        n = self.totals["total"]
        rate = n / dt if dt else 0
        ok = self.totals[STATUS_OK]
        print(f"[{dt/60:6.1f} min] {n:,} attempted | {ok:,} ok "
              f"({100*ok/max(n,1):.1f} %) | {rate:.0f} img/s", flush=True)
        hot = sorted(self.budgets.values(), key=lambda b: -b.stats["ok"])[:5]
        for b in hot:
            print(f"    {b.host[:46]:46s} conc={b.cur:3d}/{b.cap:3d} "
                  f"ok={b.stats['ok']:,} throttled={b.stats['throttled']} err={b.stats['error']}")


def stage_fetch(a):
    Path(a.images).mkdir(parents=True, exist_ok=True)
    asyncio.run(Fetcher(a).main())


# ---------------------------------------------------------------------------------------------
# Stage 3 -- report
# ---------------------------------------------------------------------------------------------

def stage_report(a):
    status = Counter()
    per_host = defaultdict(Counter)
    hashes = Counter()
    nbytes = 0
    for d in sorted(Path(a.manifest).glob("host=*")):
        for m in d.glob("part-*.meta.parquet"):
            t = pq.read_table(m, columns=["status", "content_hash", "bytes"]).to_pydict()
            status.update(t["status"])
            per_host[d.name[5:]].update(t["status"])
            nbytes += sum(t["bytes"])
            hashes.update(h for h in t["content_hash"] if h)
    tot = sum(status.values())
    print(f"attempted {tot:,} | acquired {status[STATUS_OK]:,} "
          f"({100*status[STATUS_OK]/max(tot,1):.2f} %) | {nbytes/1e12:.3f} TB")
    print("\nfailure modes:")
    for s, c in status.most_common(15):
        if s != STATUS_OK:
            print(f"  {c:>10,}  {s}")
    dupes = [(h, c) for h, c in hashes.most_common(10) if c > 1]
    if dupes:
        print("\nmost repeated content hashes (placeholder candidates -- drop in post-pass):")
        for h, c in dupes:
            print(f"  {c:>8,}  {h}")
    worst = sorted(per_host.items(), key=lambda kv: -(sum(kv[1].values()) - kv[1][STATUS_OK]))[:8]
    print("\nhosts by failure count:")
    for h, c in worst:
        t = sum(c.values())
        print(f"  {h[:50]:50s} {t - c[STATUS_OK]:>8,} / {t:,}")


# ---------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("plan", help="stream the catalog and write a host-partitioned manifest")
    q.add_argument("--out", default="data/tol/manifest")
    q.add_argument("--min-img", type=int, default=50)
    q.add_argument("--cap", type=int, default=2000)
    q.add_argument("--recount", action="store_true")
    q.add_argument("--readers", type=int, default=16,
                   help="concurrent row-group readers. The scan is latency-bound, not CPU-bound.")
    q.add_argument("--prefetch", type=int, default=32,
                   help="row-groups fetched ahead of the consumer (bounds memory)")
    q.set_defaults(fn=stage_plan)

    q = sub.add_parser("fetch", help="crawl the manifest; resumable")
    q.add_argument("--manifest", default="data/tol/manifest")
    q.add_argument("--images", required=True)
    q.add_argument("--size", type=int, default=256, help="longest side, px")
    q.add_argument("--quality", type=int, default=90)
    q.add_argument("--variant", default="medium", choices=["original", "large", "medium", "small"],
                   help="iNaturalist size variant to request (58 %% of the corpus). 'medium' is "
                        "500 px and 10x cheaper than 'original'; we downsize to --size anyway.")
    q.add_argument("--min-dim", type=int, default=64, help="reject images smaller than this")
    q.add_argument("--max-bytes", type=int, default=40_000_000)
    q.add_argument("--attempts", type=int, default=4)
    q.add_argument("--timeout", type=float, default=60.0)
    q.add_argument("--workers", type=int, default=16, help="decode/encode threads")
    q.add_argument("--limit-hosts", type=int, default=0, help="smoke test: only the first N hosts")
    q.add_argument("--retry-failed", action="store_true")
    q.add_argument("--report-every", type=float, default=60.0)
    q.add_argument("--block-probe", type=int, default=40,
                   help="attempts before the blocked-host circuit breaker may trip")
    q.add_argument("--block-ratio", type=float, default=0.9,
                   help="forbidden fraction at which a host is declared blocking")
    q.set_defaults(fn=stage_fetch)

    q = sub.add_parser("report", help="what we have and what failed")
    q.add_argument("--manifest", default="data/tol/manifest")
    q.set_defaults(fn=stage_report)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
