#!/bin/bash
# Where does the project's storage go? (2026-10-12: 5,849 of 6,144 GB used; the web UI shows no
# folder sizes.) Run inside a 1-vCPU job with every project drive mounted read-only under /work.
# Also measures random pread throughput on the uploaded ToL shards (the training read pattern).
#
# The shard read test runs first (minutes); the du walks after it can take hours.

echo "=== random pread on the ToL shards ==="
python - <<'PY'
import glob, os, random, time, concurrent.futures as cf
import pyarrow.parquet as pq
root = (glob.glob("/work/treeoflife_200m/s256") + glob.glob("/work/*/treeoflife_200m/s256"))[0]
rows = []
for f in sorted(glob.glob(f"{root}/index/shard-*.parquet")):
    stem = os.path.basename(f)[:-8]
    shard = f"{root}/shards/b{int(stem.split('-')[1]) // 100:03d}/{stem}.tar"
    t = pq.read_table(f, columns=["offset", "size"]).to_pydict()
    rows += [(shard, o, s) for o, s in zip(t["offset"], t["size"])]
print(root, len(rows), "indexed members")
random.seed(int(time.time()))
fds = {s: os.open(s, os.O_RDONLY) for s in {r[0] for r in rows}}
def read(r):
    shard, o, s = r
    b = os.pread(fds[shard], s, o)
    assert len(b) == s and b[:2] == b"\xff\xd8", shard
    return s
for threads in (1, 8, 32, 64):
    sample = random.sample(rows, 4000)
    t0, n, nbytes = time.time(), 0, 0
    with cf.ThreadPoolExecutor(threads) as ex:
        for s in ex.map(read, sample):
            n += 1; nbytes += s
            if time.time() - t0 > 60: break
    dt = time.time() - t0
    print(f"threads {threads:>2}: {n} reads in {dt:.1f} s = {n/dt:.0f} reads/s, {nbytes/dt/1e6:.1f} MB/s", flush=True)
print("need ~1,100 reads/s at 20 M parameters, ~480 at 198 M")
PY

echo "=== sizes: one du walk per drive, two levels deep, 4 drives at a time (GB, path) ==="
# The storage exposes no recursive-size attribute (directories report size 0), so du walks every
# tree. One walk per drive at --max-depth=2 gives both levels; drives run in parallel because the
# walk waits on metadata round-trips, not on the single CPU.
ls -d /work/*/ | grep -v '^/work/lepinet/$' | xargs -P 4 -I{} sh -c \
  'timeout 18000 du -b --max-depth=2 "{}" 2>/dev/null | awk -F"\t" "{printf \"%9.1f GB  %s\n\", \$1/1e9, \$2}" > "/tmp/du.$(basename "{}").txt"; echo "--- {}"; sort -rn "/tmp/du.$(basename "{}").txt" | head -25'
echo "=== largest overall ==="
cat /tmp/du.*.txt | sort -rn | head -60
