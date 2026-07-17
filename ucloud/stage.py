"""Stage the image tree from the UCloud drive onto node-local disk before training.

Why this exists
---------------
The /work mount is latency-bound, not bandwidth-bound. Measured on a job (see
ucloud/README.md):

    readers      1     16     128     256     512
    img/s     11.1  172.6  1038.3  1545.4  1768.3

fastai defaults num_workers to 16 no matter how many cores the box has, so training reads
at ~170 img/s and a full epoch takes ~8.7 h -- four times slower than the local 5090,
while the B200 sits idle. Reading with enough concurrency instead drains the drive at
~138 MB/s, so copying the whole tree costs ~1 h once, after which every epoch reads from
local NVMe (1.2 GB/s) and is limited by JPEG decode rather than by the network.

This is the node-local scratch pattern from HPC: stage in, compute, write results back.
Only the images move; out_dir stays on /work so checkpoints survive the job.
"""

import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC = sys.argv[1] if len(sys.argv) > 1 else "/work/global_lepi/images"
DST = sys.argv[2] if len(sys.argv) > 2 else "/tmp/global_lepi/images"
# Well past the knee of the curve above; these threads block on network I/O, so it is fine
# for them to outnumber the cores by an order of magnitude.
READERS = int(os.environ.get("STAGE_READERS", "512"))


def species_dirs(root):
    return [e.name for e in os.scandir(root) if e.is_dir()]


def copy_dir(name):
    """Copy one species directory. Returns (files, bytes) actually transferred."""
    src, dst = os.path.join(SRC, name), os.path.join(DST, name)
    os.makedirs(dst, exist_ok=True)
    files = nbytes = 0
    try:
        entries = list(os.scandir(src))
    except OSError:
        return 0, 0
    for e in entries:
        if not e.is_file():
            continue
        target = os.path.join(dst, e.name)
        try:
            size = e.stat().st_size
            # Idempotent: a re-run (or a retry after an extend) skips what is already here.
            if os.path.exists(target) and os.path.getsize(target) == size:
                continue
            shutil.copyfile(e.path, target)
            files += 1
            nbytes += size
        except OSError as exc:
            print(f"  ! {e.path}: {exc}", flush=True)
    return files, nbytes


def main():
    t0 = time.time()
    print(f"staging {SRC} -> {DST} with {READERS} readers", flush=True)
    os.makedirs(DST, exist_ok=True)
    dirs = species_dirs(SRC)
    print(f"{len(dirs)} species dirs listed in {time.time() - t0:.1f}s", flush=True)

    done = files = nbytes = 0
    last = t0
    with ThreadPoolExecutor(max_workers=READERS) as pool:
        futures = {pool.submit(copy_dir, d): d for d in dirs}
        for fut in as_completed(futures):
            f, b = fut.result()
            files += f
            nbytes += b
            done += 1
            now = time.time()
            if now - last > 30 or done == len(dirs):
                el = now - t0
                print(
                    f"  {done}/{len(dirs)} dirs | {files} files | "
                    f"{nbytes / 1e9:.1f} GB | {nbytes / el / 1e6:.0f} MB/s | {el / 60:.1f} min",
                    flush=True,
                )
                last = now

    el = time.time() - t0
    print(f"staged {files} files / {nbytes / 1e9:.1f} GB in {el / 60:.1f} min", flush=True)
    free = shutil.disk_usage(DST).free / 1e9
    print(f"free on {DST}: {free:.0f} GB", flush=True)
    if free < 20:
        sys.exit("refusing to continue: node-local disk nearly full")


if __name__ == "__main__":
    main()
