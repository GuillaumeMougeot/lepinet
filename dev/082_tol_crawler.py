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

The corpus lives on the `datasets` drive at `/12383016/treeoflife_200m/`, which a job mounts as
`/work/treeoflife_200m`. Run every stage on a small CPU node -- see the CPU budget note in
`ucloud/lepinet-tolfetch.toml` before choosing one.

    # 1. plan: stream the catalog, apply the policy, write a host-partitioned manifest
    python dev/082_tol_crawler.py plan --out /work/treeoflife_200m/manifest --min-img 50 --cap 2000

    # 1b. split: reshard into bounded parts, so resume granularity is 50k images not 50M
    python dev/082_tol_crawler.py split --manifest /work/treeoflife_200m/manifest --rows-per-part 50000

    # 2. fetch: crawl. Resumable, restartable, safe to run many times
    python dev/082_tol_crawler.py fetch --manifest /work/treeoflife_200m/manifest \
        --images /work/treeoflife_200m/images --workers 1 --max-cpus 1

    # 2b. remaining: rows left per host -- budget a resume from this, not from a guess
    python dev/082_tol_crawler.py remaining --manifest /work/treeoflife_200m/manifest

    # 3. report: what we have, what failed, and why
    python dev/082_tol_crawler.py report --manifest /work/treeoflife_200m/manifest

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
SKIPPED_BLOCKED = "skipped_blocked"   # never recorded: the row stays unattempted


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


def _complete_size(p: Path) -> int:
    """Size in bytes if `p` is a fully written JPEG, else 0. Runs in the I/O pool."""
    try:
        with open(p, "rb") as f:
            f.seek(-2, os.SEEK_END)
            if f.read(2) == b"\xff\xd9":
                return f.tell()
    except OSError:
        pass
    return 0


def _save_atomic(p: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a kill can never leave a truncated file that looks done."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jpg.part")
    tmp.write_bytes(data)
    os.replace(tmp, p)


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
    pin_thread_pools(effective_cpus())
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
# Stage 1c -- substitute
# ---------------------------------------------------------------------------------------------

def stage_substitute(a):
    """Replace capped images that sit on blocked hosts with uncapped ones on accessible hosts.

    The cap takes each species' first `cap` images in catalog order, blind to where they are
    served from. So a species can have 2,000 selected images on a server that refuses us while
    thousands more of the same species, never selected, sit on servers that do not. This pass
    replays the plan's selection exactly, and for every selected row on a blocked host emits one
    substitute: the next unselected row of the same species on an accessible host.

    It also measures what cannot be substituted -- species with no accessible images at all, and
    species pushed below the image floor -- broken down by kingdom, class and basis of record, so
    the decision to chase the blocked institutions can be made on numbers.

    Streaming and deterministic: within a species every selected row precedes every unselected
    one in catalog order, so by the time a pool row arrives its species' blocked count is final.
    Correctness check: the replayed selection must reproduce the manifest's row count exactly.

    Runs anywhere; it needs only the catalog (HuggingFace) and the manifest's species_counts.json
    and blocked_hosts.json. Run it on a workstation, not a UCloud CPU node.
    """
    pin_thread_pools(effective_cpus())
    counts = json.loads(Path(a.species_counts).read_text())
    keep = {k for k, v in counts.items() if v >= a.min_img}
    blocked = set(json.loads(Path(a.blocked).read_text()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # A second pass (e.g. after the crawl declares more hosts dead) must not re-emit what an
    # earlier pass already queued. Prior substitutes still count toward each species' quota, so
    # the arithmetic stays right; ones that landed on a host now unavailable drop out of the
    # pool and are replaced.
    prior: set[str] = set()
    if a.prior:
        for fp in Path(a.prior).glob("host=*/part-*.parquet"):
            prior.update(pq.read_table(fp, columns=["uuid"]).column("uuid").to_pylist())
        print(f"{len(prior):,} substitutes from the prior pass will be counted, not re-emitted")
    print(f"{len(keep):,} species at min {a.min_img} / cap {a.cap}; {len(blocked)} blocked hosts")

    emitted, sel_blocked, pool_ok, subs = Counter(), Counter(), Counter(), Counter()
    tax: dict[str, tuple] = {}
    by = {k: Counter() for k in ("host", "basis", "img_type", "kingdom", "class", "source",
                                 "publisher")}
    # Composition of the whole selection (not just the blocked part), for the dataset README.
    comp = {k: Counter() for k in ("kingdom", "class", "order", "basis", "img_type", "source",
                                   "host")}
    # Buffered per host and flushed in blocks: one write per row would give a parquet file with
    # millions of single-row row-groups -- slow to write, slow to read, and pointlessly large.
    bufs: dict[str, list] = defaultdict(list)
    parts: Counter = Counter()               # host -> parts written
    rows_out: Counter = Counter()            # host -> substitute rows written
    cols = PLAN_COLS + ["basis_of_record", "img_type", "publisher"]

    def flush(host: str):
        rows = bufs.pop(host, None)
        if not rows:
            return
        d = out / f"host={slug(host)}"; d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), d / f"part-{a.prefix}{parts[host]:05d}.parquet")
        parts[host] += 1; rows_out[host] += len(rows)

    def emit(host: str, row: dict):
        bufs[host].append(row)
        if len(bufs[host]) >= a.rows_per_part:
            flush(host)

    nrg = n_row_groups(CATALOG); t0 = time.monotonic()
    if a.limit_rg:
        nrg = min(nrg, a.limit_rg)            # smoke tests only: the row-count check will not match
    for i, tb in iter_row_groups(CATALOG, cols, 0, nrg, a.readers, a.prefetch):
        d = tb.to_pydict()
        for j in range(len(d["uuid"])):
            sp = f"{(d['genus'][j] or '').strip()} {(d['species'][j] or '').strip()}".strip()
            if sp not in keep:
                continue
            url = d["source_url"][j]
            if not url:
                continue
            host = host_of(url)
            if sp not in tax:
                tax[sp] = (d["kingdom"][j] or "?", d["class"][j] or "?", d["order"][j] or "?")
            if emitted[sp] < a.cap:                       # replay of the plan's selection
                emitted[sp] += 1
                comp["kingdom"][tax[sp][0]] += 1; comp["class"][tax[sp][1]] += 1
                comp["order"][tax[sp][2]] += 1; comp["host"][host] += 1
                comp["basis"][d["basis_of_record"][j] or "?"] += 1
                comp["img_type"][d["img_type"][j] or "?"] += 1
                comp["source"][d["data_source"][j] or "?"] += 1
                if host in blocked:
                    sel_blocked[sp] += 1
                    by["host"][host] += 1
                    by["basis"][d["basis_of_record"][j] or "?"] += 1
                    by["img_type"][d["img_type"][j] or "?"] += 1
                    by["kingdom"][tax[sp][0]] += 1
                    by["class"][tax[sp][1]] += 1
                    by["source"][d["data_source"][j] or "?"] += 1
                    by["publisher"][f"{host} | {d['publisher'][j] or '?'}"] += 1
            elif host not in blocked:                     # unselected, accessible: the pool
                pool_ok[sp] += 1
                if subs[sp] < sel_blocked[sp]:
                    subs[sp] += 1
                    if d["uuid"][j] in prior:
                        continue                          # queued by an earlier pass
                    emit(host, {"uuid": d["uuid"][j], "url": url, "species": sp,
                                "genus": (d["genus"][j] or "").strip(),
                                "family": (d["family"][j] or "").strip(),
                                "order": (d["order"][j] or "").strip(),
                                "data_source": d["data_source"][j] or "",
                                "source_id": str(d["source_id"][j] or "")})
        if i % 100 == 0:
            el = time.monotonic() - t0
            print(f"  row-group {i}/{nrg} | selected {sum(emitted.values()):,} | on blocked "
                  f"{sum(sel_blocked.values()):,} | substitutes {sum(subs.values()):,} | "
                  f"{i/max(el,1e-9)*60:.0f} rg/min", flush=True)
    for h in list(bufs):
        flush(h)

    # --- what substitution recovers, and what is still lost ------------------------------------
    sel_total = sum(emitted.values())
    affected = [sp for sp in sel_blocked if sel_blocked[sp]]
    lost_imgs = sum(sel_blocked[sp] - subs[sp] for sp in affected)
    final = {sp: emitted[sp] - sel_blocked[sp] + subs[sp] for sp in affected}
    gone = [sp for sp in affected if final[sp] == 0]
    below = [sp for sp in affected if 0 < final[sp] < a.min_img]
    def tally(spp, idx):
        return Counter(tax[s][idx] for s in spp).most_common(12)
    lepi = [sp for sp in affected if tax[sp][2] == "Lepidoptera"]
    report = {
        "selected_rows": sel_total, "manifest_rows_expected": a.expect_rows,
        "selection_reproduced": sel_total == a.expect_rows if a.expect_rows else None,
        "selected_on_blocked_hosts": sum(sel_blocked.values()),
        "species_affected": len(affected),
        "substitutes_found": sum(subs.values()),
        "images_still_lost": lost_imgs,
        "species_lost_entirely": len(gone),
        "species_pushed_below_floor": len(below),
        "species_kept": len(keep) - len(gone) - len(below),
        "lost_species_by_kingdom": tally(gone, 0), "lost_species_by_class": tally(gone, 1),
        "lepidoptera": {"species_affected": len(lepi),
                        "selected_on_blocked": sum(sel_blocked[s] for s in lepi),
                        "substituted": sum(subs[s] for s in lepi),
                        "species_lost_entirely": sum(1 for s in lepi if final[s] == 0)},
        "blocked_selected_by": {k: v.most_common(25) for k, v in by.items()},
        "selection_composition": {k: v.most_common(30) for k, v in comp.items()},
        "species_by_kingdom": Counter(tax[s][0] for s in emitted).most_common(),
        "species_by_class": Counter(tax[s][1] for s in emitted).most_common(30),
        "lepidoptera_planned": {"species": sum(1 for s in emitted if tax[s][2] == "Lepidoptera"),
                                "images": comp["order"]["Lepidoptera"]},
        "substitute_rows_by_host": dict(rows_out.most_common()),
    }
    (out / "substitution_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}, indent=2))
    print(f"wrote {out / 'substitution_report.json'}")

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
    reason: str = ""                             # "forbidden" or "dead:<kind>" once tripped
    window: list = field(default_factory=list)   # recent outcomes: "S" ok, "F" 403, "D" dead, "O" other

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

    def _observe(self, code: str):
        self.window.append(code)
        if len(self.window) > 400:
            del self.window[0]

    def on_success(self):
        self._observe("S")
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

    def on_error(self, dead: str = "", probe: int = 200, ratio: float = 0.97):
        """An error. `dead` names a kind that means the resource or server is gone (http_404,
        dns, tls, connect); enough of those, with no success in the window, stops the host.

        Without this the crawler only stopped hosts that *refused* it (403). A server that is
        simply gone -- `files.plutof.ut.ee` no longer resolves, `sernecportal.org` returns 404 for
        every image, `scan-bugs.org` was retired -- got every one of its rows tried, each with
        retries and back-off, for nothing. Partial 404s are normal (records get withdrawn), hence
        a high ratio *and* zero successes in the window before a host is declared dead.
        """
        self._observe("D" if dead else "O")
        self.stats["error"] += 1
        if dead:
            self.stats[f"dead_{dead}"] += 1
            w = self.window[-probe:]
            if (len(w) >= probe and "S" not in w and w.count("D") / len(w) >= ratio
                    and not self.blocked):
                self.blocked, self.reason = True, f"dead:{dead}"
        self.ok_streak = 0

    def note_forbidden(self, probe: int, ratio: float):
        """Circuit breaker for a host that is refusing us.

        A handful of 403s is normal -- individual records get withdrawn or embargoed. A host that
        returns 403 to *almost everything* is blocking the crawler, and continuing is futile and
        rude to a server someone runs on a museum budget.

        The rate is taken over a **sliding window of the last 200 outcomes**, not over the host's
        lifetime. The first crawl used a lifetime ratio, and `medialib.naturalis.nl` showed why
        that is wrong: it served ~137 k images, then banned us, and the lifetime ratio took **1.26
        million further forbidden requests** to climb past 90 %. A window trips within ``probe``
        requests of the ban however much the host served before it.
        """
        self.stats["forbidden"] += 1
        self._observe("F")
        w = self.window[-probe:]
        if len(w) >= probe and w.count("F") / probe >= ratio and not self.blocked:
            self.blocked, self.reason = True, "forbidden"


def verify_and_encode(body: bytes, size: int, quality: int, min_dim: int):
    """Decode fully, reject junk, resize, re-encode as JPEG. Runs on a worker thread.

    A *full* decode rather than ``Image.verify()``: verify() checks the header and misses truncated
    payloads, which are the common failure when a server drops a connection mid-response and still
    returned HTTP 200. The decode is the expensive part of this whole pipeline and the reason image
    work belongs on a thread pool rather than the event loop.
    """
    import warnings
    from PIL import Image
    # PIL treats anything over ~89 MP as a possible decompression bomb (warning) and over ~179 MP
    # as one (error). Herbarium and museum scans legitimately reach 100-200 MP, and they arrive
    # from known institutional servers, not adversaries. Draft decoding keeps a JPEG's memory at
    # its reduced size however large the original, so 400 MP is safe here; the warning, which
    # fired on every herbarium scan, is noise.
    Image.MAX_IMAGE_PIXELS = 400_000_000
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
    if not any(body.startswith(m) for m, _ in MAGIC):
        return None, "bad_magic", None
    try:
        img = Image.open(io.BytesIO(body))
        orig = img.size                              # from the header, before any decoding
        # DCT-domain downscaling: libjpeg decodes JPEGs directly at 1/2, 1/4 or 1/8 scale, never
        # below `size` on the short side. The remaining hosts are herbaria serving 50-100 MP
        # specimen scans that we shrink to 256 px anyway; measured on a 54 MP scan, full decode
        # took 250 ms and draft decode 22 ms (11x) for a mean pixel difference of 0.29/255. On a
        # 1-vCPU job that is the difference between keeping up and becoming the bottleneck.
        # A no-op for non-JPEG formats.
        # Draft decoding bounds a JPEG's memory however large it is; PNG/TIFF/WebP get decoded at
        # full size, so the raised 400 MP ceiling must not apply to them -- one 400 MP PNG is
        # ~1.2 GB decoded and ~2.4 GB once converted, most of a 3 GB job.
        if img.format != "JPEG" and orig[0] * orig[1] > 60_000_000:
            return None, f"too_large_pixels:{orig[0]}x{orig[1]}:{img.format}", None
        img.draft("RGB", (size, size))
        img.load()                                   # still a full decode of the (reduced) stream:
                                                     # truncation is still caught
    except Exception as e:
        return None, f"decode_fail:{type(e).__name__}", None
    if min(orig) < min_dim:
        return None, f"too_small:{orig[0]}x{orig[1]}", None
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
        # File-system work runs here, never on the event loop. On the 2026-10-01 resume the disk
        # checks for in-flight parts ran synchronously inside the coroutines: on a network FS that
        # is milliseconds per file, ~850 k files, and the whole event loop -- every host's HTTP --
        # stalled behind them ("0 attempted" for the first reports). These threads only wait on
        # I/O, so a pool much larger than the CPU count is correct even on 1 vCPU.
        self.disk_pool = ThreadPoolExecutor(max_workers=a.disk_threads)
        self.budgets: dict[str, HostBudget] = {}
        self.totals = Counter()
        self.t0 = time.monotonic()
        self.blocked_path = Path(a.manifest) / "blocked_hosts.json"
        try:
            self.known_blocked: dict = json.loads(self.blocked_path.read_text())
        except (OSError, ValueError):
            self.known_blocked = {}

    def record_blocked(self, b: "HostBudget"):
        """Persist a tripped breaker, so a resumed crawl skips the host instead of re-probing it.

        The file is also the to-do list for a human: these are institutions to ask for research
        access, which is the correct response to an explicit block -- not a better disguise.
        """
        if b.host in self.known_blocked:
            return
        self.known_blocked[b.host] = {
            "reason": b.reason or "forbidden",
            "forbidden": b.stats["forbidden"], "ok": b.stats["ok"], "error": b.stats["error"],
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.blocked_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.known_blocked, indent=2, sort_keys=True))
        os.replace(tmp, self.blocked_path)
        why = ("ask the institution for access rather than retrying" if b.reason == "forbidden"
               else "the server or its images are gone")
        print(f"  [stopped] {b.host} ({b.reason}): {b.stats['forbidden']:,} forbidden, "
              f"{b.stats['error']:,} errors, {b.stats['ok']:,} ok -- recorded in "
              f"{self.blocked_path.name}; {why}.", flush=True)

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
        """Fetch, verify and encode one image, holding one memory slot from request to encode.

        The slot used to be released when the body finished downloading, *before* decoding.
        Decoding runs on one core, so whenever downloads outpace it -- 235 active hosts, the
        iNaturalist substitutes at 64+ in parallel -- finished bodies queued for the decoder with
        no bound, and the 3 GB job was killed within two minutes, three times, with no traceback.
        Holding the slot through the encode makes `--max-inflight` a true bound on the bodies in
        memory: at most max_inflight x max_bytes, whatever the download/decode speed ratio.
        """
        state = {"mem": False}
        try:
            return await self._fetch_one(session, row, b, state)
        finally:
            if state["mem"]:
                self.inflight.release()

    async def _fetch_one(self, session, row, b: HostBudget, state: dict):
        import aiohttp
        # Try the cheap size variant first; fall back to the catalog URL if it is missing.
        alt = variant_url(row["url"], self.a.variant)
        urls = [alt, row["url"]] if alt else [row["url"]]
        url = urls[0]
        for attempt in range(self.a.attempts):
            await b.acquire()
            # Re-check AFTER the semaphore, immediately before the request leaves. Every row of a
            # chunk is scheduled at once and passes any earlier check while the breaker is still
            # closed, then waits here; checking only before the wait let a blocked host receive
            # every queued request anyway (measured: 120 of 120, with the breaker tripped at ~40).
            if b.blocked:
                b.release()
                return None, SKIPPED_BLOCKED, None
            try:
                async with session.get(url, allow_redirects=True) as r:
                    if r.status in (429, 503):
                        ra = r.headers.get("Retry-After")
                        b.on_throttle(float(ra) if ra and ra.isdigit() else None)
                        continue
                    if r.status >= 400:
                        if r.status == 404 and len(urls) > 1 and url == urls[0]:
                            url = urls[1]          # variant missing for this photo; use the original
                            self.totals["variant_fallback"] += 1
                            continue               # not evidence about the host: don't count it
                        if r.status == 403:
                            b.note_forbidden(self.a.block_probe, self.a.block_ratio)
                        elif r.status in (404, 410):
                            b.on_error(dead=f"http_{r.status}", probe=self.a.dead_probe,
                                       ratio=self.a.dead_ratio)
                        else:
                            b.on_error()
                        if r.status in (404, 410, 403):
                            return None, f"http_{r.status}", None   # permanent; do not retry
                        await asyncio.sleep(min(30, 2 ** attempt + random.random()))  # 5xx: back off
                        continue
                    ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                    # Only an obvious text response is rejected on its header. Everything else is
                    # judged by its bytes: the magic-byte check and full decode below are
                    # authoritative. Rejecting any non-`image/*` type threw away valid JPEGs that
                    # CDNs serve as `application/octet-stream` -- 52 % of one host's images.
                    if ct.startswith("text/"):
                        b.on_error()
                        return None, f"not_an_image:{ct}", None
                    cl = r.headers.get("content-length")
                    if cl and int(cl) > self.a.max_bytes:
                        return None, f"too_large:{cl}", None
                    # The memory slot covers exactly the life of a body: taken here, once the
                    # response is known to be worth reading, and held through the encode
                    # (released in fetch_one's finally). Not before the host slot -- coroutines
                    # queued behind a slow host would sit on global slots and starve fast hosts --
                    # and not for failed responses, which never hold a body, so dead hosts'
                    # retry back-offs cannot hog slots either.
                    if not state["mem"]:
                        await self.inflight.acquire()
                        state["mem"] = True
                    # Streamed with a hard cap: with no Content-Length header, r.read() would
                    # pull a body of any size into memory before the size check could run.
                    chunks, size = [], 0
                    async for piece in r.content.iter_chunked(1 << 16):
                        size += len(piece)
                        if size > self.a.max_bytes:
                            b.on_error()
                            return None, f"too_large:>{self.a.max_bytes}", None
                        chunks.append(piece)
                    body = b"".join(chunks)
                    del chunks
                    b.on_success()
            except asyncio.TimeoutError:
                b.on_throttle()                # a timeout from a small host is usually overload
                continue
            except aiohttp.ClientError as e:
                # A name that does not resolve or a certificate that does not verify will not fix
                # itself between retries; those count toward the dead-host breaker.
                kind = ("tls" if isinstance(e, (aiohttp.ClientConnectorCertificateError,
                                                aiohttp.ClientSSLError))
                        else "connect" if isinstance(e, aiohttp.ClientConnectorError) else "")
                b.on_error(dead=kind, probe=self.a.dead_probe, ratio=self.a.dead_ratio)
                if b.blocked or attempt == self.a.attempts - 1:
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
        """Crawl one manifest part, streaming its rows rather than materialising them.

        Four properties, each the fix for something that went wrong on the first full crawl:

        * **Streaming.** Rows are read `--chunk` at a time from Arrow. Materialising a whole 50 k-row
          part as Python dicts costs ~50 MB per active host, and a few hundred active hosts is what
          forced a large node. Memory is now ~hosts x chunk, which is what lets this run on 2 vCPU.
        * **Resumable inside a part.** A part's metadata is only written when it completes, so a
          crawl killed mid-part used to refetch the whole part. Now a row whose image is already on
          disk *and complete* is recorded without a request. Completeness = the JPEG ends in FFD9;
          a file truncated by a kill does not, and is refetched.
        * **Atomic image writes** (tmp + rename), so a future kill cannot leave a truncated file
          that looks finished.
        * **The circuit breaker is checked on every row**, not at part boundaries. Checking only
          between parts meant every blocked host still received its full first part -- 50,000
          requests to a server that had already said no.
        """
        meta_path = part.with_suffix(".meta.parquet")
        tmp_path = part.with_suffix(".meta.parquet.tmp")
        retrying = self.a.retry_failed or bool(self.a.retry_status)
        if meta_path.exists() and not retrying:
            return
        b = self.budget(host)
        if b.blocked:
            return
        done: set[str] = set()
        old_meta = None
        if meta_path.exists():
            old_meta = pq.read_table(meta_path)
            d = old_meta.select(["uuid", "status"]).to_pydict()
            # Retry a row if it failed and, when --retry-status is given, only if its failure
            # starts with one of those prefixes -- so a targeted retry of (say) `bad_content_type`
            # does not re-request 404s that are gone for good.
            want = tuple(self.a.retry_status)
            done = {u for u, s in zip(d["uuid"], d["status"])
                    if s == STATUS_OK or (want and not s.startswith(want))}
            if len(done) == len(d["uuid"]):
                return

        tables: list[pa.Table] = []
        aborted = False
        for batch in pq.ParquetFile(part).iter_batches(batch_size=max(1, self.a.chunk),
                                                       columns=["uuid", "url", "species"]):
            if b.blocked:
                aborted = True
                break
            rows = [r for r in batch.to_pylist() if r["uuid"] not in done]
            recs: list[dict] = []
            loop = asyncio.get_running_loop()
            paths = [self.images / slug(r["species"]) / f"{r['uuid']}.jpg" for r in rows]
            sizes = await asyncio.gather(*(loop.run_in_executor(self.disk_pool, _complete_size, p)
                                           for p in paths))

            async def one(row, p, size):
                if b.blocked:
                    return                    # leave unattempted, so a later --retry-blocked can finish it
                rec = {"uuid": row["uuid"], "species": row["species"], "status": "",
                       "path": "", "content_hash": "", "width": 0, "height": 0, "bytes": 0}
                if size:
                    rec |= {"status": STATUS_OK, "path": str(p.relative_to(self.images)),
                            "bytes": size}
                    self.totals["already_on_disk"] += 1
                    recs.append(rec)
                    return
                jpg, status, meta = await self.fetch_one(session, row, b)
                if status == SKIPPED_BLOCKED:
                    return
                rec["status"] = status
                if jpg is not None:
                    await loop.run_in_executor(self.disk_pool, _save_atomic, p, jpg)
                    ch, (w, h) = meta
                    rec |= {"path": str(p.relative_to(self.images)), "content_hash": ch,
                            "width": w, "height": h, "bytes": len(jpg)}
                recs.append(rec)
                self.totals[status if status == STATUS_OK else "fail"] += 1
                self.totals["total"] += 1

            await asyncio.gather(*(one(r, p, s) for r, p, s in zip(rows, paths, sizes)))
            if recs:
                tables.append(pa.Table.from_pylist(recs))

        if aborted or b.blocked:
            self.record_blocked(b)
            return                            # part not finished: no meta, so it stays resumable
        if tables:
            new_meta = pa.concat_tables(tables)
            if old_meta is not None:
                # Merge, never replace: a retry processes only the failed rows, and writing just
                # those would drop every earlier success from the part's record (the images would
                # still be on disk, but nothing would say so).
                retried = set(new_meta.column("uuid").to_pylist())
                keep = [u not in retried for u in old_meta.column("uuid").to_pylist()]
                new_meta = pa.concat_tables([old_meta.filter(pa.array(keep)),
                                             new_meta.cast(old_meta.schema)])
            pq.write_table(new_meta, tmp_path)
            os.replace(tmp_path, meta_path)

    async def main(self):
        import aiohttp
        # Memory bounds that do not grow with the number of hosts. Substitution gave 235 hosts
        # work instead of 17, and the 1-vCPU (3 GB) job was killed: every active host held a
        # chunk of pending coroutines and its in-flight bodies -- herbarium originals run to tens
        # of MB -- so memory scaled with host count. Politeness caps already bound each host's
        # speed, so capping active hosts and total in-flight requests costs no throughput.
        self.inflight = asyncio.Semaphore(self.a.max_inflight)
        self.active_hosts = asyncio.Semaphore(self.a.max_active_hosts)
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
        print(f"{len(parts)} parts over {len(by_host)} hosts", flush=True)

        timeout = aiohttp.ClientTimeout(total=self.a.timeout, connect=15)
        conn = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=600)
        async with aiohttp.ClientSession(timeout=timeout, connector=conn,
                                         headers=BASE_HEADERS) as session:
            async def host_worker(host, plist):
                if host in self.known_blocked and not self.a.retry_blocked:
                    self.totals["skipped_blocked_parts"] += len(plist)
                    return
                async with self.active_hosts:
                    for p in plist:
                        if self.budget(host).blocked:
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
              f"({100*ok/max(n,1):.1f} %) | {rate:.0f} img/s | "
              f"{self.totals['already_on_disk']:,} recovered from disk | {cgroup_memory()}",
              flush=True)
        # Sorted by activity, not successes: a table sorted by `ok` shows arbitrary idle hosts when
        # nothing is succeeding, which is exactly when you need to see where the requests are going.
        act = lambda b: sum(b.stats.values())
        hot = sorted((b for b in self.budgets.values() if act(b)), key=act, reverse=True)[:8]
        for b in hot:
            print(f"    {b.host[:46]:46s} conc={b.cur:3d}/{b.cap:3d} ok={b.stats['ok']:,} "
                  f"forbidden={b.stats['forbidden']:,} err={b.stats['error']:,} "
                  f"throttled={b.stats['throttled']}", flush=True)


def effective_cpus() -> int:
    """CPUs this process may actually use: the cgroup quota if set, else the affinity mask.

    `os.cpu_count()` reports the *host's* cores inside a container, which is the wrong number to
    budget against -- a 4-vCPU job on a 128-core machine would look huge, and the reverse mistake
    is just as easy.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def cgroup_memory() -> str:
    """Container memory as the kernel sees it: what the OOM killer acts on, not the process RSS.

    Added after two crawls were killed with no traceback; the next one should say how close to
    the limit it was and whether the kernel had already killed something.
    """
    def read(name):
        try:
            return Path(f"/sys/fs/cgroup/{name}").read_text().strip()
        except OSError:
            return ""
    cur, mx = read("memory.current"), read("memory.max")
    ooms = next((ln.split()[1] for ln in read("memory.events").splitlines()
                 if ln.startswith("oom_kill ")), "?")
    if not cur:
        return "mem n/a"
    gb = lambda s: f"{int(s)/1e9:.2f}" if s.isdigit() else s
    return f"mem {gb(cur)}/{gb(mx)} GB, oom_kill {ooms}"


def pin_thread_pools(n: int) -> None:
    """Size every library thread pool to the job's real allocation, not the machine's.

    Measured on UCloud (2026-10-01): a 2-vCPU job runs on a 256-core host, and inside the
    container `os.cpu_count()` and the affinity mask both report **256**; only the cgroup quota
    says 2. Libraries that size pools from `os.cpu_count()` -- pyarrow's CPU pool among them --
    therefore start 256 threads on 2 CPUs. Pin them to what `effective_cpus()` measured.
    """
    pa.set_cpu_count(max(1, n))
    pa.set_io_thread_count(max(4, 2 * n))   # I/O threads mostly wait; a few more than CPUs is right
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(max(1, n)))


def stage_fetch(a):
    """Crawl. Refuses to start on an oversized node.

    This job is network-bound: measured at ~1.1 cores per 240 img/s of decode, so the fastest
    phase of the first crawl (~385 img/s) needed under 2 cores and the slow tail under one. It was
    nonetheless run on 64 vCPU for 63 hours -- **4,032 core-hours, 58 % of the project's CPU
    allocation** -- because nothing measured the need or capped the spend. The guard below makes a
    repeat cost a few minutes instead of days. `ucloud/budget_check.py` refuses the spec before
    submission; this refuses the process if the spec got through anyway.
    """
    n = effective_cpus()
    print(f"effective CPUs: {n} (limit {a.max_cpus}); os.cpu_count() says {os.cpu_count()}", flush=True)
    pin_thread_pools(n)
    if n > a.max_cpus and not a.allow_big_node:
        raise SystemExit(
            f"REFUSING to crawl on {n} CPUs (limit {a.max_cpus}). This job is network-bound and "
            f"wastes CPU budget on a large node -- see the stage_fetch docstring. Use a smaller "
            f"product, or pass --allow-big-node with the owner's approval.")
    Path(a.images).mkdir(parents=True, exist_ok=True)
    asyncio.run(Fetcher(a).main())


# ---------------------------------------------------------------------------------------------
# Stage 1b -- split
# ---------------------------------------------------------------------------------------------

def stage_split(a):
    """Reshard an existing manifest into bounded parts, without re-scanning the catalog.

    `plan` writes one part per host, which is wrong for the hosts that matter: the iNaturalist bucket
    is ~50 M rows in a single file, and a part is only marked done when it finishes **in full**. So a
    crawl killed after 40 M images would resume from zero, and 24-hour job slots guarantee it gets
    killed. Resume granularity is a property of the part size.

    Resharding is local parquet I/O over a manifest already on disk -- minutes -- against 45 minutes
    to re-run `plan`. Idempotent: hosts already split into `part-00001` or beyond are left alone.
    """
    pin_thread_pools(effective_cpus())
    root = Path(a.manifest)
    total_in = total_out = 0
    for d in sorted(root.glob("host=*")):
        parts = [p for p in sorted(d.glob("part-*.parquet")) if not p.name.endswith(".meta.parquet")]
        if not parts:
            continue
        if len(parts) > 1 or pq.ParquetFile(parts[0]).metadata.num_rows <= a.rows_per_part:
            total_in += sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
            total_out += len(parts)
            continue
        src = parts[0]
        n = pq.ParquetFile(src).metadata.num_rows
        tmp = d / "_resharding"
        tmp.mkdir(exist_ok=True)
        # Streamed, so peak memory is one output part regardless of input size. Reading the
        # 50 M-row iNaturalist part whole needs ~15 GB of Arrow, which is what pushed this onto a
        # 64-vCPU node the first time; a batch at a time it runs on 2.
        k = 0
        for batch in pq.ParquetFile(src).iter_batches(batch_size=a.rows_per_part):
            pq.write_table(pa.Table.from_batches([batch]), tmp / f"part-{k:05d}.parquet")
            k += 1
        src.unlink()
        for f in sorted(tmp.glob("part-*.parquet")):
            f.rename(d / f.name)
        tmp.rmdir()
        total_in += n
        total_out += k
        print(f"  {d.name[5:]:52s} {n:>12,} rows -> {k:,} parts", flush=True)
    print(f"\nmanifest: {total_in:,} rows in {total_out:,} parts "
          f"(<= {a.rows_per_part:,} rows each)")


# ---------------------------------------------------------------------------------------------
# Stage 2b -- remaining
# ---------------------------------------------------------------------------------------------

def stage_remaining(a):
    """What is left to crawl, per host -- the number a resume should be budgeted from.

    Reads only parquet footers for row counts, so it is cheap even over 2,050 parts. A part with a
    `.meta.parquet` is finished; a part without one counts in full (an over-estimate by at most the
    in-flight rows, which the resumed crawl skips from disk without a request).
    """
    root = Path(a.manifest)
    try:
        blocked = set(json.loads((root / "blocked_hosts.json").read_text()))
    except (OSError, ValueError):
        blocked = set()
    rows_done = rows_left = 0
    left_by_host: Counter = Counter()
    blocked_left = 0
    for d in sorted(root.glob("host=*")):
        parts = [p for p in sorted(d.glob("part-*.parquet")) if not p.name.endswith(".meta.parquet")]
        if not parts:
            continue
        host = host_of(pq.read_table(parts[0], columns=["url"]).column("url")[0].as_py())
        for p in parts:
            n = pq.ParquetFile(p).metadata.num_rows
            if p.with_suffix(".meta.parquet").exists():
                rows_done += n
            elif host in blocked:
                blocked_left += n
            else:
                rows_left += n
                left_by_host[host] += n
    print(f"finished parts: {rows_done:,} rows | still to crawl: {rows_left:,} rows over "
          f"{len(left_by_host)} hosts | on blocked hosts (skipped): {blocked_left:,}")
    print("\nlargest remaining hosts, with the politeness cap that bounds their speed:")
    for h, n in left_by_host.most_common(a.top):
        print(f"  {n:>11,}  cap {HOST_CAPS.get(h, DEFAULT_CAP):>3}  {h}")


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
# Stage 4 -- describe
# ---------------------------------------------------------------------------------------------

def stage_describe(a):
    """Render README.md for the dataset folder from the manifest's own JSON files.

    Generated, not hand-written, so it can be refreshed as the crawl progresses instead of
    drifting from the data it describes. Inputs: plan_summary.json, substitution_report.json,
    blocked_hosts.json, and optionally status.json ({"as_of", "acquired", "attempted", ...}) and
    a contacts JSON for the blocked hosts.
    """
    m = Path(a.manifest)
    load = lambda f, d=None: json.loads((m / f).read_text()) if (m / f).exists() else d
    plan, rep = load("plan_summary.json", {}), load("substitution_report.json", {})
    blocked = load("blocked_hosts.json", {})
    status = json.loads(Path(a.status).read_text()) if a.status else {}
    contacts = json.loads(Path(a.contacts).read_text()) if a.contacts else {}
    f = lambda n: f"{n:,}" if isinstance(n, (int, float)) else str(n)
    comp = rep.get("selection_composition", {})
    total = plan.get("rows") or 1

    def table(pairs, head, n=12, pct_of=None):
        rows = [f"| {head[0]} | {head[1]} |" + (" share |" if pct_of else ""),
                "|---|---:|" + ("---:|" if pct_of else "")]
        for k, v in pairs[:n]:
            rows.append(f"| {k} | {f(v)} |" + (f" {100*v/pct_of:.1f} % |" if pct_of else ""))
        return "\n".join(rows)

    L = []
    L += [f"# TreeOfLife-200M, image subset ({f(plan.get('rows'))} images, "
          f"{f(plan.get('species'))} species)", "",
          "Images from **TreeOfLife-200M** (Imageomics, Gu et al. 2025) fetched from their source servers "
          "for the `lepinet` project, under our own sampling policy. TreeOfLife-200M on HuggingFace "
          "publishes metadata and URLs only; the images below were crawled with "
          "`dev/082_tol_crawler.py` in the lepinet repository.", "",
          f"*This file is generated by `python dev/082_tol_crawler.py describe`. Status as of "
          f"**{status.get('as_of', '?')}**.*", ""]
    L += ["## Status", "",
          "| | images |", "|---|---:|",
          f"| planned (selection below) | {f(plan.get('rows'))} |",
          f"| acquired so far | {f(status.get('acquired', '?'))} |",
          f"| still to crawl (accessible hosts) | {f(status.get('remaining', '?'))} |",
          f"| on hosts that refuse automated access | {f(status.get('blocked_rows', '?'))} |",
          f"| substitutes queued for those (same species, accessible host) | "
          f"{f(rep.get('substitutes_found', '?'))} |", ""]
    L += ["## Where things are", "",
          "```",
          "treeoflife_200m/",
          "  README.md                          this file",
          "  images/<folder>/<uuid>.jpg         <uuid> is TreeOfLife's image uuid; <folder> is a LOSSY slug",
          "                                     of the species key -- see 'Labels' below",
          "  manifest/",
          "    host=<server>/part-NNNNN.parquet      what to fetch, partitioned by source server",
          "    host=<server>/part-sNNNNN.parquet     substitutes for images on blocked servers",
          "    host=<server>/part-*.meta.parquet     what happened: status, path, hash, size, per image",
          "    species_counts.json                   images per species in the whole catalog",
          "    plan_summary.json                     the selection policy and totals",
          "    blocked_hosts.json                    servers that refused us, and when",
          "    substitution_report.json              what substitution recovered, what is lost",
          "```", "",
          "**Manifest columns:** `uuid, url, species, genus, family, order, data_source, source_id`. "
          "`source_id` is the provider's record id -- for `data_source == gbif` it is the **GBIF "
          "occurrence id**, the key for attribution, licence lookup and joining to other GBIF data.", "",
          "**Meta columns:** `uuid, species, status, path, content_hash, width, height, bytes`. "
          "`status == ok` means the file exists and passed every check; anything else names the "
          "failure (`http_404`, `decode_fail:*`, `too_small:*`, ...). Read the metas, not the folder "
          "listing, to know what you have.", ""]
    counts = load("species_counts.json", {})
    if counts:
        kept = {k: v for k, v in counts.items() if v >= plan.get("min_img", 50)}
        capn = lambda d: sum(min(v, plan.get("cap", 2000)) for v in d.values())
        single = {k: v for k, v in kept.items() if " " not in k}
        higher = {k: v for k, v in single.items() if k[:1].isupper()}
        bare = {k: v for k, v in single.items() if not k[:1].isupper()}
        bino = {k: v for k, v in kept.items() if " " in k}
        tot = capn(kept) or 1
        L += ["## Labels -- read before training", "",
              "**Take every label from the meta's `species` column (or join `uuid` to the "
              "TreeOfLife catalog), never from the folder name.** Two reasons:", "",
              "1. **Folder names are a lossy slug.** Capital letters were replaced by `_`, so "
              "`Acacia dealbata` is stored under `_cacia_dealbata/`, and 158 folders are shared by "
              "two names that differ only in their initial (`Sedum` / `Ledum` -> `_edum/`). Image "
              "filenames are unique uuids, so nothing is overwritten; only the folder is ambiguous. "
              "Kept as-is for consistency across the crawl rather than renamed mid-way.",
              "2. **Not every key is a species.** The key is `genus + ' ' + epithet` and degrades "
              "when a field is empty in the source:", "",
              "| key kind | keys | images | use |", "|---|---:|---:|---|",
              f"| binomial species | {f(len(bino))} | {f(capn(bino))} ({100*capn(bino)/tot:.1f} %) | species level |",
              f"| genus or higher only (e.g. `Megaselia`, `Sciaridae`) | {f(len(higher))} | "
              f"{f(capn(higher))} ({100*capn(higher)/tot:.1f} %) | genus/family level only |",
              f"| epithet without a genus (e.g. `occidentalis`) | {f(len(bare))} | "
              f"{f(capn(bare))} ({100*capn(bare)/tot:.1f} %) | **unreliable** -- merges unrelated "
              "genera; exclude or re-label from the catalog |", "",
              f"So the honest species count is **{f(len(bino))}**, not {f(len(kept))}.", ""]
    L += ["## How the subset was chosen", "",
          f"* **Species** = `genus + ' ' + specific epithet`. A bare epithet is not a species.",
          f"* **Floor:** species with at least **{plan.get('min_img', 50)}** images in the whole "
          f"catalog. This keeps {f(plan.get('species'))} of 884,662 species "
          f"and drops the long tail of taxa with a handful of images (about 7 % of images).",
          f"* **Cap:** at most **{plan.get('cap', 2000)} images per species** -- the first "
          f"{plan.get('cap', 2000)} in catalog order. The cap removes head images: uncapped, the ten "
          f"most-photographed species would take a large share of every epoch.",
          f"* **Substitution:** where a species' capped images sit on a server that refuses us, "
          f"unselected images of the same species from accessible servers take their place.", ""]
    L += ["## Image processing and quality control", "",
          "* Longest side resized to **256 px**, re-encoded **JPEG quality 90** (~16-25 KB each).",
          "* iNaturalist (58 % of images) is fetched at its `medium` size (500 px), not `original`: "
          "identical output after the resize, ~10x less transfer.",
          "* Every image is fully decoded before it counts: magic bytes, declared type, truncation, "
          "minimum 64 px. Large JPEGs are decoded at reduced scale in the DCT domain.",
          "* `content_hash` (SHA-1 of the decoded, resized pixels) finds exact duplicates and the "
          "\"image unavailable\" placeholders some servers return with HTTP 200. **Deduplicate "
          "before training.**", ""]
    if comp:
        L += ["## Composition of the selection", "",
              table(comp.get("kingdom", []), ("kingdom", "images"), 8, total), "",
              table(rep.get("species_by_kingdom", []), ("kingdom", "species"), 8), "",
              table(comp.get("basis", []), ("basis of record", "images"), 6, total), "",
              "`HUMAN_OBSERVATION` is mostly citizen-science photographs of living organisms; "
              "`PRESERVED_SPECIMEN` is museum and herbarium material -- pinned insects, herbarium "
              "sheets. They look very different and are worth separating for some uses.", "",
              table(comp.get("class", []), ("class", "images"), 15, total), ""]
        lp = rep.get("lepidoptera_planned", {})
        L += [f"**Lepidoptera:** {f(lp.get('species'))} species, {f(lp.get('images'))} images.", "",
              table(comp.get("host", []), ("largest source servers", "images"), 12, total), ""]
    if blocked:
        why = lambda v: {"forbidden": "refuses (HTTP 403)"}.get(v.get("reason", "forbidden"),
                                                             v.get("reason", "?").replace("dead:", "dead: "))
        L += ["## Servers we cannot fetch from", "",
              f"{len(blocked)} servers are skipped. Some **refuse** automated access (HTTP 403; the "
              "crawler identifies itself and does not impersonate a browser); others are **dead** -- "
              "images removed (`http_404`), a hostname that no longer resolves (`connect`), an expired "
              "certificate (`tls`). Their images stay in the manifest but are not fetched; most of "
              "their species are covered by substitutes. Contacts are official addresses, from the "
              "GBIF registry or the operator's own contact page. Which ones are worth writing to is "
              "argued in `journal/2026-10-01-what-the-blocked-servers-cost.md` in the lepinet repo.", "",
              "| server | why | operator | contact | requests before stopping |",
              "|---|---|---|---|---:|"]
        for h, v in sorted(blocked.items(),
                           key=lambda kv: -(kv[1].get("forbidden", 0) + kv[1].get("error", 0))):
            c = contacts.get(h, {})
            L.append(f"| `{h}` | {c.get('why') or why(v)} | {c.get('operator', '?')} | "
                     f"{c.get('contact', '?')} | "
                     f"{f(v.get('forbidden', 0) + v.get('error', 0))} |")
        L += [""]
    if rep:
        L += ["## What the unreachable servers cost", "",
              f"* {f(rep.get('selected_on_blocked_hosts'))} selected images sit on them, across "
              f"{f(rep.get('species_affected'))} species.",
              f"* Substitution finds {f(rep.get('substitutes_found'))} replacements, leaving "
              f"**{f(rep.get('images_still_lost'))} images** unrecoverable without access.",
              f"* **{f(rep.get('species_lost_entirely'))} species** have no accessible image at all, "
              f"and {f(rep.get('species_pushed_below_floor'))} more fall below the "
              f"{plan.get('min_img', 50)}-image floor.", "",
              "Lost species by kingdom: " + ", ".join(f"{k} {f(v)}" for k, v in
                                                     rep.get("lost_species_by_kingdom", [])[:6]), ""]
    L += ["## Licences -- read before redistributing", "",
          "Every image keeps the licence its publisher gave it (mostly CC0, CC-BY and CC-BY-NC; "
          "some datasets are more restrictive). This folder is a **research working copy**: fine "
          "for training models, **not** something to republish wholesale. For attribution or "
          "licence checks, look up `source_id` (the GBIF occurrence id) at "
          "`https://api.gbif.org/v1/occurrence/<source_id>`.", ""]
    L += ["## Reproducing or extending it", "",
          "```",
          "python dev/082_tol_crawler.py plan --out <manifest> --min-img 50 --cap 2000",
          "python dev/082_tol_crawler.py split --manifest <manifest>",
          "python dev/082_tol_crawler.py substitute --species-counts <manifest>/species_counts.json \\",
          "       --blocked <manifest>/blocked_hosts.json --out <dir>",
          "python dev/082_tol_crawler.py fetch --manifest <manifest> --images <images> --max-cpus 1",
          "python dev/082_tol_crawler.py remaining --manifest <manifest>",
          "python dev/082_tol_crawler.py describe --manifest <manifest> --out README.md",
          "```", "",
          "The crawl is network-bound: run it on **1 vCPU**. A 64-vCPU run once spent 58 % of the "
          "project's CPU allocation for no speed-up -- see "
          "`journal/2026-10-01-the-crawl-that-spent-58-percent-of-the-cpu-budget.md`.", "",
          "**Cite** TreeOfLife-200M (doi:10.57967/hf/8980) and the source data providers. "
          "Maintainer: Guillaume Mougeot, lepinet project.", ""]
    Path(a.out).write_text("\n".join(L))
    print(f"wrote {a.out} ({len(L)} lines)")

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

    q = sub.add_parser("substitute", help="replace capped rows on blocked hosts with accessible ones")
    q.add_argument("--species-counts", required=True, help="manifest/species_counts.json")
    q.add_argument("--blocked", required=True, help="manifest/blocked_hosts.json")
    q.add_argument("--out", required=True, help="local dir for substitute parts + report")
    q.add_argument("--min-img", type=int, default=50)
    q.add_argument("--cap", type=int, default=2000)
    q.add_argument("--rows-per-part", type=int, default=50_000)
    q.add_argument("--limit-rg", type=int, default=0, help="smoke test: first N row-groups only")
    q.add_argument("--prior", default="", help="dir of an earlier pass's substitute parts")
    q.add_argument("--prefix", default="s", help="part file prefix: part-<prefix>NNNNN.parquet")
    q.add_argument("--expect-rows", type=int, default=87_560_065,
                   help="manifest row count the replayed selection must reproduce")
    q.add_argument("--readers", type=int, default=16)
    q.add_argument("--prefetch", type=int, default=32)
    q.set_defaults(fn=stage_substitute)

    q = sub.add_parser("fetch", help="crawl the manifest; resumable")
    q.add_argument("--manifest", default="data/tol/manifest")
    q.add_argument("--images", required=True)
    q.add_argument("--size", type=int, default=256, help="longest side, px")
    q.add_argument("--quality", type=int, default=90)
    q.add_argument("--variant", default="medium", choices=["original", "large", "medium", "small"],
                   help="iNaturalist size variant to request (58 %% of the corpus). 'medium' is "
                        "500 px and 10x cheaper than 'original'; we downsize to --size anyway.")
    q.add_argument("--min-dim", type=int, default=64, help="reject images smaller than this")
    q.add_argument("--max-bytes", type=int, default=25_000_000)
    q.add_argument("--max-inflight", type=int, default=48,
                   help="requests in flight across all hosts; bounds memory held in bodies")
    q.add_argument("--max-active-hosts", type=int, default=48,
                   help="hosts crawled at once; bounds pending coroutines and buffers")
    q.add_argument("--attempts", type=int, default=4)
    q.add_argument("--timeout", type=float, default=60.0)
    q.add_argument("--workers", type=int, default=16, help="decode/encode threads")
    q.add_argument("--limit-hosts", type=int, default=0, help="smoke test: only the first N hosts")
    q.add_argument("--retry-failed", action="store_true", help="re-attempt every non-ok row")
    q.add_argument("--retry-status", nargs="*", default=[],
                   help="re-attempt only rows whose status starts with one of these, e.g. "
                        "bad_content_type client_error")
    q.add_argument("--report-every", type=float, default=60.0)
    q.add_argument("--chunk", type=int, default=2_000,
                   help="rows streamed per batch within a part; bounds memory and live coroutines")
    q.add_argument("--disk-threads", type=int, default=16,
                   help="threads for file-system checks and writes (I/O-bound; not CPU)")
    q.add_argument("--max-cpus", type=int, default=8,
                   help="refuse to run on more CPUs than this (network-bound job; see stage_fetch)")
    q.add_argument("--allow-big-node", action="store_true",
                   help="override --max-cpus. Owner approval only")
    q.add_argument("--retry-blocked", action="store_true",
                   help="re-attempt hosts recorded in blocked_hosts.json (e.g. after access is granted)")
    q.add_argument("--dead-probe", type=int, default=200,
                   help="recent outcomes over which a host may be declared dead (404/DNS/TLS)")
    q.add_argument("--dead-ratio", type=float, default=0.97,
                   help="dead fraction, with zero successes in the window, that stops a host")
    q.add_argument("--block-probe", type=int, default=40,
                   help="attempts before the blocked-host circuit breaker may trip")
    q.add_argument("--block-ratio", type=float, default=0.9,
                   help="forbidden fraction at which a host is declared blocking")
    q.set_defaults(fn=stage_fetch)

    q = sub.add_parser("split", help="reshard a manifest into bounded parts (resume granularity)")
    q.add_argument("--manifest", default="data/tol/manifest")
    q.add_argument("--rows-per-part", type=int, default=50_000)
    q.set_defaults(fn=stage_split)

    q = sub.add_parser("remaining", help="rows left to crawl, per host (budget a resume from this)")
    q.add_argument("--manifest", default="data/tol/manifest")
    q.add_argument("--top", type=int, default=25)
    q.set_defaults(fn=stage_remaining)

    q = sub.add_parser("describe", help="render README.md for the dataset folder")
    q.add_argument("--manifest", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--status", default="", help="status.json: as_of, acquired, remaining, blocked_rows")
    q.add_argument("--contacts", default="", help="JSON: host -> {operator, contact}")
    q.set_defaults(fn=stage_describe)

    q = sub.add_parser("report", help="what we have and what failed")
    q.add_argument("--manifest", default="data/tol/manifest")
    q.set_defaults(fn=stage_report)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
