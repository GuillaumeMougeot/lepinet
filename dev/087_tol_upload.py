"""Pack the short-side-256 TreeOfLife images into tar shards and upload them to UCloud.

64 M small files cannot be uploaded one by one, so they travel as ~2 GB uncompressed tars (the
JPEGs do not compress). Disk stays bounded: a batch of shards is packed while the previous batch
uploads, and each batch is deleted once uploaded. A ledger makes the run resumable.

    python dev/087_tol_upload.py list     # walk images/ once -> files.tsv (path, bytes)
    python dev/087_tol_upload.py plan     # cut files.tsv into shard lists of --shard-gb
    python dev/087_tol_upload.py run      # pack + index + upload every shard not in the ledger
    python dev/087_tol_upload.py index    # (re)build the index of shards uploaded before it existed

Remote layout: <remote>/shards/b<NNN>/shard-<NNNNN>.tar, at most 100 shards per folder because
`ucloud files ls` shows only the first 250 entries of a folder; <remote>/lists/ holds each
shard's file list. Images added later (the retry crawls) go up as extra shards: `list --since
files.tsv` lists only files not already planned.

**The shards are the training storage, not just the transport.** <remote>/index/shard-N.parquet
gives every member's (path, offset, size), so a loader reads one image with a single positioned
read from a large file: random access, which the recipe's weighted (sqrt-oversampled) sampler
needs and a streaming WebDataset reader does not give; no 64 M-file unpack, no second copy. The
tars stay standard, so `tar -x` or WebDataset remain possible later. Offsets depend only on the
member names and sizes, so re-taring the same list reproduces them (`index`, verified against a
downloaded shard).

Runs on the workstation: tar is one core per shard and the upload is network-bound, so it needs
no UCloud CPU at all.
"""
from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

IMAGES = "/data/au761367/tol256s/images"
WORK = "/data/au761367/tol256s/upload"
REMOTE = "/12383016/treeoflife_200m/s256"


def cmd_list(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / ("files_extra.tsv" if a.since else "files.tsv")
    # find visits one directory at a time, so each species' files stay contiguous in the list.
    with open(out, "w") as f:
        subprocess.run(["find", a.images, "-type", "f", "-name", "*.jpg", "-printf", "%P\t%s\n"],
                       stdout=f, check=True)
    if a.since:
        seen = {line.split("\t", 1)[0] for line in open(a.since)}
        rows = [line for line in open(out) if line.split("\t", 1)[0] not in seen]
        out.write_text("".join(rows))
    n = sum(1 for _ in open(out))
    print(f"{n:,} files -> {out}")


def cmd_plan(a):
    work = Path(a.work)
    lists = work / "lists"
    lists.mkdir(exist_ok=True)
    start = len(list(lists.glob("shard-*.txt")))      # extra shards continue the numbering
    src = work / ("files_extra.tsv" if a.extra else "files.tsv")
    limit = int(a.shard_gb * 1e9)
    k, size, buf = start, 0, []

    def flush():
        nonlocal k, size, buf
        if buf:
            (lists / f"shard-{k:05d}.txt").write_text("\n".join(buf) + "\n")
            k, size, buf = k + 1, 0, []

    for line in open(src):
        path, nbytes = line.rstrip("\n").split("\t")
        buf.append(path)
        size += int(nbytes) + 1024                    # tar header + padding, roughly
        if size >= limit:
            flush()
    flush()
    print(f"shards {start:05d}-{k - 1:05d} planned in {lists}")


def index_tar(fileobj, out: Path):
    """(path, offset, size) of every member's data, from a tar read as a stream."""
    import tarfile
    paths, offs, sizes = [], [], []
    with tarfile.open(fileobj=fileobj, mode="r|") as t:
        for m in t:
            if m.isfile():
                paths.append(m.name)
                offs.append(m.offset_data)
                sizes.append(m.size)
    pq.write_table(pa.table({"path": paths, "offset": pa.array(offs, pa.int64()),
                             "size": pa.array(sizes, pa.int32())}), out)
    return len(paths)


def cmd_index(a):
    work = Path(a.work)
    (work / "index").mkdir(exist_ok=True)
    ledger = work / "uploaded.txt"
    done = ledger.read_text().split() if ledger.exists() else []
    subprocess.run(["ucloud", "files", "mkdir", f"{a.remote}/index"], capture_output=True)
    for stem in done:
        out = work / "index" / f"{stem}.parquet"
        if out.exists():
            continue
        p = subprocess.Popen(["tar", "-cf", "-", "-C", a.images, "--no-recursion", "-T",
                              str(work / "lists" / f"{stem}.txt")], stdout=subprocess.PIPE)
        n = index_tar(p.stdout, out)
        if p.wait():
            out.unlink()
            raise RuntimeError(f"tar failed for {stem}")
        r = subprocess.run(["ucloud", "files", "upload", str(out), f"{a.remote}/index"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-300:])
        print(f"{stem}: {n:,} members indexed", flush=True)


def cmd_run(a):
    work = Path(a.work)
    lists = sorted((work / "lists").glob("shard-*.txt"))
    ledger = work / "uploaded.txt"
    done = set(ledger.read_text().split()) if ledger.exists() else set()
    todo = [p for p in lists if p.stem not in done]
    if a.max_shards:
        # A ceiling, so a run cannot fill the project's shared quota: when UCloud storage is full,
        # every member's jobs fail to write, not just ours.
        todo = todo[:max(0, a.max_shards - len(done))]
    # A batch never straddles a remote folder, so it uploads in one parallel call:
    # `ucloud files upload <dir> <folder>` puts the dir's *contents* into <folder>.
    folder_of = {p: f"{a.remote}/shards/b{int(p.stem.split('-')[1]) // 100:03d}" for p in todo}
    batches = []
    for p in todo:
        if batches and len(batches[-1]) < a.batch and folder_of[batches[-1][0]] == folder_of[p]:
            batches[-1].append(p)
        else:
            batches.append([p])
    print(f"{len(done)} shards uploaded, {len(todo)} to go in {len(batches)} batches", flush=True)
    # The folder must exist: a *file* uploaded to a missing path becomes a file of that name, so
    # without this every list overwrote one file called `lists` (first run, 2026-10-12).
    for sub in ("lists", "index"):
        subprocess.run(["ucloud", "files", "mkdir", f"{a.remote}/{sub}"], capture_output=True)
    (work / "index").mkdir(exist_ok=True)
    q: queue.Queue = queue.Queue(maxsize=1)           # one batch packed ahead of the upload

    def pack():
        for i, batch in enumerate(batches):
            d = work / "staging" / f"batch{i:04d}"
            if d.exists():
                shutil.rmtree(d)                      # a half-packed batch from a killed run
            d.mkdir(parents=True)
            procs = [subprocess.Popen(["tar", "-cf", str(d / f"{p.stem}.tar"), "-C", a.images,
                                       "--no-recursion", "-T", str(p)]) for p in batch]
            if any(pr.wait() for pr in procs):
                q.put(RuntimeError(f"tar failed in {d}"))
                return
            for p in batch:
                with open(d / f"{p.stem}.tar", "rb") as f:
                    index_tar(f, work / "index" / f"{p.stem}.parquet")
            q.put((d, batch))
        q.put(None)

    threading.Thread(target=pack, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        d, batch = item
        folder = folder_of[batch[0]]
        for attempt in range(5):
            r = subprocess.run(["ucloud", "files", "upload", "-j", str(len(batch)), str(d), folder],
                               capture_output=True, text=True)
            if r.returncode == 0 and f"Uploaded {len(batch)} file(s)" in r.stdout:
                break
            print(f"  upload {batch[0].stem} failed (attempt {attempt + 1}): "
                  f"{(r.stdout + r.stderr)[-300:]}", flush=True)
        else:
            raise RuntimeError(f"giving up on {batch[0].stem}..{batch[-1].stem}")
        for p in batch:
            subprocess.run(["ucloud", "files", "upload", str(p), f"{a.remote}/lists"],
                           capture_output=True, check=True)
            subprocess.run(["ucloud", "files", "upload", str(work / "index" / f"{p.stem}.parquet"),
                            f"{a.remote}/index"], capture_output=True, check=True)
        with open(ledger, "a") as f:
            f.write("".join(p.stem + "\n" for p in batch))
        shutil.rmtree(d)
        print(f"uploaded {batch[0].stem}..{batch[-1].stem} -> {folder}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=IMAGES)
    ap.add_argument("--work", default=WORK)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("list")
    q.add_argument("--since", default="", help="an earlier files.tsv: list only new files")
    q.set_defaults(fn=cmd_list)
    q = sub.add_parser("plan")
    q.add_argument("--shard-gb", type=float, default=2.0)
    q.add_argument("--extra", action="store_true", help="plan files_extra.tsv as further shards")
    q.set_defaults(fn=cmd_plan)
    q = sub.add_parser("run")
    q.add_argument("--remote", default=REMOTE)
    q.add_argument("--batch", type=int, default=4, help="shards packed in parallel")
    q.add_argument("--max-shards", type=int, default=0,
                   help="stop once this many shards are uploaded in total (0: all); ~2 GB each")
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("index")
    q.add_argument("--remote", default=REMOTE)
    q.set_defaults(fn=cmd_index)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
