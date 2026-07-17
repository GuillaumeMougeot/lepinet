"""Measure DataLoader worker memory: what does each `num_workers` process actually cost?

Written after the 2026-07-17 UCloud benchmark died: three jobs ran at the full ~1100 img/s
decode ceiling for ~36k batches, then collapsed 440x (to ~2.5 img/s) and were SIGKILLed with no
traceback. Silence + no Python exception = the kernel's OOM killer, i.e. HOST ram, not GPU (a
CUDA OOM raises a loud `torch.cuda.OutOfMemoryError`). The node has 288 GB; the run used
`num_workers: 512`, i.e. a 576 MB budget per worker.

The mechanism is copy-on-write breakage: dev/030 sets `multiprocessing.set_start_method("fork")`,
so each worker inherits the 5.7M-row DataFrame by COW -- but CPython's refcounting *writes* to an
object's header whenever it reads it, so pages are copied as workers touch rows. Footprint grows
with batches consumed rather than jumping at startup -- exactly the observed ramp-then-cliff.

Measured 2026-07-17, two phases (confirmed on-node with the guard's per-batch logging):
  1. A fast burst as the contiguous pointer arrays and index get dirtied: to ~1.16 GB/worker
     (237 GB at 192 workers) within ~300 batches.
  2. A slow creep as workers touch ever-more distinct row string-objects (scattered, one page
     dirtied at a time). Coupon-collector-shaped, decelerating (~0.007 -> 0.006 GB/batch) but
     NOT flattening under the cap within an epoch: 192 workers crept from 237 GB (82%) at batch
     300 to 265 GB (92%) at batch 4525, where the guard aborted. So the "plateau" is a knee,
     not a ceiling.

Consequence: worker-count tuning cannot make this safe on a long run -- any count that starts
comfortably still creeps into the cap. 512 -> OOM in the first minutes; 256 -> ~batch 315;
192 -> ~batch 4500. The real fix is to stop forking a pandas DataFrame at all.

DISPROVEN fix (--arrow): converting the worker-read columns to pyarrow strings changed the leak
by nothing (+4.72 vs +4.77 GB over 125 batches at 16 workers). fastai's DataBlock walks the frame
with `df.iloc[i]`, which materialises each row as a Python-object Series regardless of column
dtype, so the refcount churn is in the traversal, not the storage. A real fix must bypass the
DataFrame for the hot path (pre-extract paths+labels to numpy arrays + custom getters), and needs
label-correctness validation, not just this memory curve.

This probe reproduces the dataloader half only -- no model, no GPU, no CUDA -- so it costs zero
GPU-hours and can run on any box. It measures total PSS (not RSS -- see tree_pss_mb) across the
process tree while iterating, at a few small `num_workers` values, and extrapolates the
per-worker slope to whatever the target is.

Usage:
    python dev/037_dl_memory_probe.py --workers 4,8,16 --batches 40 --target 512
    python dev/037_dl_memory_probe.py --workers 4,8,16 --batches 40 --oversample 0.5
"""

import argparse
import json
import subprocess
import importlib
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent))
v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
longtail = importlib.import_module("034_longtail")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS


def tree_pss_mb(proc):
    """Total PSS of this process and every child, in MB.

    PSS, not RSS. A forked worker's RSS *includes* every copy-on-write page it still shares
    with the parent, so summing RSS across a fork tree counts the same physical page once per
    process -- which reports more memory than the machine physically has (measured 307 GB on a
    125 GB box before this was fixed). PSS divides each shared page by the number of processes
    sharing it, so the tree sum is the true footprint and, usefully here, a page migrating from
    shared to private (COW breaking) shows up as PSS rising even though RSS never moves.

    Racing children (a worker exiting mid-walk) are skipped rather than fatal.
    """
    def pss(p):
        try:
            return p.memory_full_info().pss
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return 0
    total = pss(proc)
    for child in proc.children(recursive=True):
        total += pss(child)
    return total / 1e6


def probe(df, vocabs, img_dir, n_workers, n_batches, batch_size, aug_img_size, img_size,
          aug_kwargs, sample_wgts, trace_every=0):
    """PSS while iterating `n_batches` at one worker count.

    Must run in a *fresh* process per worker count: torch, page cache and the previous probe's
    dead workers all leave memory behind, so a second probe in the same interpreter starts from
    a polluted and drifting baseline (measured base wandering 2 -> 11 -> 7 GB, making the
    worker counts read non-monotonic). `main` re-invokes this file with `--single` to isolate.

    `trace_every` samples PSS periodically: the COW hypothesis predicts footprint *grows with
    batches consumed* (pages being copied as workers touch more rows), as opposed to jumping at
    startup and flattening. The shape of that curve is the actual evidence, not the endpoint.
    """
    proc = psutil.Process()
    base = tree_pss_mb(proc)

    dls = v4.make_dls(df, vocabs, img_dir, aug_img_size, img_size, batch_size,
                      num_workers=n_workers, aug_kwargs=aug_kwargs, sample_wgts=sample_wgts)
    built = tree_pss_mb(proc)

    t0 = time.time()
    it = iter(dls.train)
    peak, trace = built, []
    for i in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
        if trace_every and (i + 1) % trace_every == 0:
            now = tree_pss_mb(proc)
            trace.append((i + 1, now))
            peak = max(peak, now)
    elapsed = time.time() - t0
    after = tree_pss_mb(proc)
    peak = max(peak, after)

    del it, dls
    return {"workers": n_workers, "base": base, "built": built, "peak": peak, "after": after,
            "trace": trace, "img_s": (n_batches * batch_size / elapsed) if elapsed else 0}


def main():
    p = argparse.ArgumentParser(description="Measure DataLoader worker memory (no GPU needed).")
    p.add_argument("--parquet", default="data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
    p.add_argument("--img-dir", default="data/global/images")
    p.add_argument("--out-dir", default="data/global/models")
    p.add_argument("--workers", default="4,8,16", help="Comma-separated worker counts to probe.")
    p.add_argument("--batches", type=int, default=40, help="Batches to iterate per probe.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--aug-img-size", type=int, default=460)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--min-img-per-spc", type=int, default=50)
    p.add_argument("--oversample", type=float, default=0.5,
                   help="oversample_power, as in the training config (0 = plain loader).")
    p.add_argument("--target", type=int, default=512, help="Worker count to extrapolate to.")
    p.add_argument("--host-ram-gb", type=float, default=288.0,
                   help="RAM of the machine being extrapolated for (B200 1-gpu node = 288).")
    p.add_argument("--trace-every", type=int, default=0,
                   help="Sample PSS every N batches to see whether it grows (the COW signature).")
    p.add_argument("--arrow", action="store_true",
                   help="Convert the worker-read columns to pyarrow strings (the COW-leak fix under test).")
    p.add_argument("--single", type=int, default=None,
                   help="Internal: probe exactly this worker count and print one JSON line. "
                        "main() re-invokes itself with this so each probe gets a clean process.")
    args = p.parse_args()

    if args.single is None:
        run_isolated(args)
        return

    print(f"Loading dataframe (same path dev/030 takes)...")
    parquet_path = Path(args.parquet)
    hierarchy_path = parquet_path.parent / "hierarchy.csv"
    df, _hierarchy = v4.gen_df(parquet_path, Path(args.out_dir), args.min_img_per_spc,
                               "1", hierarchy_path, [])
    vocabs = {level: sorted(df[level].unique().tolist()) for level in HIERARCHY_LEVELS}
    print(f"  {len(df):,} rows, {len(vocabs['speciesKey']):,} species")

    if args.arrow:
        # The leak's root cause: the columns workers read (image_path + the 3 keys) are pandas
        # object dtype, i.e. millions of individual Python str objects. Every ColReader access
        # in a forked worker increfs one, dirtying its shared COW page -- so RSS climbs with
        # rows touched. pyarrow-backed strings store the whole column in one contiguous Arrow
        # buffer (one object), so per-row access materialises a transient str and touches no
        # shared refcount. Same on-disk bytes, drastically fewer objects to incref.
        cols = ["image_path", *HIERARCHY_LEVELS]
        for c in cols:
            df[c] = df[c].astype("string[pyarrow]")
        print("  ARROW: converted image_path + keys to pyarrow strings (leak-fix candidate)")
    print(f"  df memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB (deep) — this is what fork shares COW\n")

    aug_kwargs = {"max_warp": 0.0, "max_lighting": 0.0, "p_lighting": 0.0,
                  "flip_vert": True, "max_rotate": 15.0, "max_zoom": 1.1}
    sample_wgts = None
    if args.oversample:
        sample_wgts = longtail.sample_weights(df, level="speciesKey", power=args.oversample)
        print(f"Oversampling ON (power={args.oversample}): weights array "
              f"{sample_wgts.nbytes/1e6:.0f} MB\n")

    r = probe(df, vocabs, Path(args.img_dir), args.single, args.batches, args.batch_size,
              args.aug_img_size, args.img_size, aug_kwargs, sample_wgts, args.trace_every)
    print("RESULT " + json.dumps(r), flush=True)


def run_isolated(args):
    """Probe each worker count in its own subprocess, then fit the per-worker slope."""
    counts = [int(x) for x in args.workers.split(",")]
    results = []
    for n in counts:
        print(f"probing num_workers={n} ({args.batches} batches, fresh process)...", flush=True)
        cmd = [sys.executable, __file__, "--single", str(n), "--batches", str(args.batches),
               "--parquet", args.parquet, "--img-dir", args.img_dir, "--out-dir", args.out_dir,
               "--batch-size", str(args.batch_size), "--aug-img-size", str(args.aug_img_size),
               "--img-size", str(args.img_size), "--min-img-per-spc", str(args.min_img_per_spc),
               "--oversample", str(args.oversample), "--trace-every", str(args.trace_every)]
        if args.arrow:
            cmd.append("--arrow")
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"  ! probe failed:\n{out.stdout[-500:]}\n{out.stderr[-500:]}")
            continue
        r = json.loads(line[len("RESULT "):])
        results.append(r)
        print(f"  base {r['base']:7.0f} MB | built {r['built']:7.0f} | peak {r['peak']:7.0f} MB"
              f" | {r['img_s']:6.0f} img/s")
        if r["trace"]:
            pts = " ".join(f"{b}:{m/1000:.1f}G" for b, m in r["trace"])
            print(f"  PSS by batch: {pts}")
            growth = r["trace"][-1][1] - r["trace"][0][1]
            print(f"  growth across the run: {growth:+.0f} MB "
                  f"({'GROWING -> COW leak' if growth > 500 else 'flat -> no leak'})")

    if len(results) < 2:
        return
    print(f"\n{'workers':>8} {'peak PSS (MB)':>14} {'net of parent':>14}")
    for r in results:
        print(f"{r['workers']:>8} {r['peak']:>14.0f} {r['peak']-r['built']:>14.0f}")

    # Least-squares over all points, not just the endpoints: the marginal cost of one worker is
    # the slope; the intercept is the parent's own fixed footprint, which would otherwise
    # inflate a naive peak/N "per worker" figure at small N.
    n = len(results)
    xs = [r["workers"] for r in results]
    ys = [r["peak"] for r in results]
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0
    intercept = my - slope * mx
    est = intercept + slope * args.target
    print(f"\nmarginal cost: {slope:.0f} MB per worker (parent baseline {intercept:.0f} MB)")
    print(f"extrapolated at num_workers={args.target}: {est/1000:.1f} GB vs "
          f"{args.host_ram_gb:.0f} GB host RAM -> "
          f"{'OVER BUDGET' if est/1000 > args.host_ram_gb else 'fits'}")
    print(f"\nThe leak SATURATES, it is not unbounded: a worker converges on its own full copy "
          f"of the\ndataframe and then stops growing, so per-worker steady state is roughly "
          f"df_size + torch +\nprefetch buffers. Measured slope above should land near that "
          f"sum -- if it does, the number\nis a real steady state and not an artifact of a "
          f"short {args.batches}-batch probe.\n"
          f"Extrapolating a few-worker slope to {args.target} still assumes linearity, which "
          f"page-cache\npressure and scheduler contention will break in the pessimistic "
          f"direction.")


if __name__ == "__main__":
    main()
