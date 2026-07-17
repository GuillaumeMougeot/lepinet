"""Stage ONLY the images a training run reads onto node-local disk, in parallel.

Why this exists
---------------
The /work mount is latency-bound, not bandwidth-bound. Measured on a job (ucloud/README.md):

    readers      1     16     128     256     512
    img/s     11.1  172.6  1038.3  1545.4  1768.3

Training over /work needs a high `num_workers` to hit the ~1100 img/s decode ceiling, and each
fastai worker's image pipeline costs ~1.2 GB of real (anon) memory -- so many workers means a
large memory footprint, and on a tight cgroup that OOMs (see
journal/2026-07-ucloud-benchmark-oom.md). Staging to local NVMe removes the latency, so ~24-48
workers saturate the decode ceiling instead of 128+, cutting the memory footprint several-fold.
It is the memory lever, not primarily a speed lever (decode stays CPU-bound at ~1100 img/s).

What changed vs the old full-tree copy
--------------------------------------
This stages ONLY the images the (filtered) dataframe references -- `remove set '0'` +
`min_img_per_spc` + `family_filter`, exactly as dev/028.gen_df does -- not the whole image
tree. For the global set that is ~5.67M files / ~430 GB instead of the full ~594 GB, and the
list matches what training actually opens, so nothing staged is wasted and nothing training
reads is missing. filter_df/prepare_df are pandas-only, so this stays a light dependency (no
fastai/torch import).

Only the images move; out_dir stays on /work so checkpoints survive the job. The TEST phase
(fold '0', one pass) is deliberately NOT staged -- point its img_dir at /work; the 10 epochs of
repeated reads are what staging pays off, a single test pass is not.

Correctness: the copy preserves the `<speciesKey>/<filename>` layout, so training's img_dir
just points at DST. Idempotent (skips files already present at the right size), and verifies
the staged count equals the referenced count before returning success -- a short copy would
otherwise let training read a truncated dataset and silently under-train.

Usage:
    python ucloud/stage.py --parquet /work/global_lepi/<...>.parquet \
        --src /work/global_lepi/images --dst /tmp/global_lepi/images --min-img-per-spc 50
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


def referenced_paths(parquet, min_img_per_spc, family_filter):
    """The exact set of `<speciesKey>/<filename>` a training run reads, as dev/028.gen_df builds
    it: drop the held-out test fold ('0'), apply family_filter and min_img_per_spc, then form
    image_path. Kept in lockstep with filter_df/prepare_df -- if those change, this must too."""
    df = pd.read_parquet(parquet)
    df = df[~df["set"].isin(["0"])]                       # remove_in=["0"] (held-out test)
    if family_filter:
        df = df[df["familyKey"].astype(str).isin(family_filter)]
    if min_img_per_spc > 0:
        df = df[df.groupby("speciesKey")["speciesKey"].transform("count") >= min_img_per_spc]
    rel = (df["speciesKey"].astype(str) + "/" + df["filename"]).unique().tolist()
    return rel


def copy_chunk(chunk, src_root, dst_root):
    """Copy one chunk of relative paths. Returns (copied, skipped, bytes, errors)."""
    copied = skipped = nbytes = errors = 0
    for rel in chunk:
        s, d = os.path.join(src_root, rel), os.path.join(dst_root, rel)
        try:
            size = os.path.getsize(s)
            if os.path.exists(d) and os.path.getsize(d) == size:
                skipped += 1
                continue
            shutil.copyfile(s, d)
            copied += 1
            nbytes += size
        except OSError as exc:
            errors += 1
            if errors <= 5:
                print(f"  ! {rel}: {exc}", flush=True)
    return copied, skipped, nbytes, errors


def main():
    p = argparse.ArgumentParser(description="Stage the training run's images to local disk.")
    p.add_argument("--parquet", required=True, help="Source metadata parquet (the unfiltered one).")
    p.add_argument("--src", default="/work/global_lepi/images", help="Image root on the /work mount.")
    p.add_argument("--dst", default="/tmp/global_lepi/images", help="Node-local destination.")
    p.add_argument("--min-img-per-spc", type=int, default=50)
    p.add_argument("--family-filter", default="", help="Comma-separated familyKeys, or empty for all.")
    p.add_argument("--readers", type=int, default=int(os.environ.get("STAGE_READERS", "512")),
                   help="Parallel copy threads; well past the knee since they block on I/O.")
    args = p.parse_args()

    t0 = time.time()
    fam = [f for f in args.family_filter.split(",") if f]
    rel = referenced_paths(args.parquet, args.min_img_per_spc, fam)
    print(f"{len(rel):,} referenced images (filter: min_img_per_spc={args.min_img_per_spc}, "
          f"family_filter={fam or 'none'}) listed in {time.time()-t0:.1f}s", flush=True)

    # Pre-create every species dir once, so the copy threads never race on makedirs.
    for sp in {r.split("/", 1)[0] for r in rel}:
        os.makedirs(os.path.join(args.dst, sp), exist_ok=True)

    # Chunk for load balance: equal-size chunks keep every thread busy to the end, unlike
    # grouping by species dir (which has a long tail of huge dirs copied by one thread).
    n_chunks = max(args.readers * 16, 1)
    chunks = [rel[i::n_chunks] for i in range(n_chunks)]

    copied = skipped = nbytes = errors = done = 0
    last = t0
    with ThreadPoolExecutor(max_workers=args.readers) as pool:
        futs = [pool.submit(copy_chunk, c, args.src, args.dst) for c in chunks]
        for fut in as_completed(futs):
            c, s, b, e = fut.result()
            copied += c; skipped += s; nbytes += b; errors += e; done += 1
            now = time.time()
            if now - last > 30 or done == len(futs):
                el = now - t0
                rate = (copied + skipped) / el if el else 0
                print(f"  {done}/{len(futs)} chunks | copied {copied:,} skipped {skipped:,} "
                      f"| {nbytes/1e9:.1f} GB | {rate:.0f} files/s | {el/60:.1f} min", flush=True)
                last = now

    el = time.time() - t0
    staged = copied + skipped
    print(f"staged {staged:,}/{len(rel):,} files ({copied:,} copied, {skipped:,} already present), "
          f"{nbytes/1e9:.1f} GB in {el/60:.1f} min, {errors} errors", flush=True)

    free = shutil.disk_usage(args.dst).free / 1e9
    print(f"free on {args.dst}: {free:.0f} GB", flush=True)

    # Fail loudly on a short copy: training reading a truncated dataset would silently
    # under-train and score low for a reason no metric would reveal.
    if staged < len(rel):
        sys.exit(f"STAGING INCOMPLETE: {len(rel)-staged:,} referenced files missing "
                 f"({errors} errors). Refusing to let training read a truncated dataset.")
    if free < 20:
        sys.exit("refusing to continue: node-local disk nearly full")
    print("staging OK: every referenced image is present on local disk.", flush=True)


if __name__ == "__main__":
    main()
