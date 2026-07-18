"""Host-memory accounting and guard — the hard-won OOM lessons.

Two mistakes this module exists to prevent, both from journal/2026-07-ucloud-benchmark-oom.md:

1. Watching the wrong number. `psutil.virtual_memory()` reads /proc/meminfo (the physical host),
   so inside a UCloud job it reports 2434 GB while the cgroup caps the job at 288 GB and the
   kernel OOM-kills there. And even the cgroup's `memory.current` is wrong to threshold on: it
   INCLUDES the page cache, which reading millions of images fills at ~5 MB/batch but which is
   reclaimable -- the kernel frees it under pressure instead of killing. Thresholding on
   memory.current false-aborted a healthy run at "92%" whose real (anon) memory was 58%. The
   number that actually triggers the OOM killer is `anon` (+ kernel stacks / page tables), read
   from `memory.stat`.

2. Summing RSS across a fork tree (the probe). A forked worker's RSS counts every copy-on-write
   page it still shares with the parent, so the sum over-reports (read 307 GB on a 125 GB box).
   Use PSS, which divides shared pages by the number of sharers.
"""
from __future__ import annotations

from pathlib import Path


def _cgroup_dir():
    """The cgroup-v2 dir whose memory.max caps this process, or None if uncapped. A container
    namespaces the mount (limit at the root); a systemd host nests the process in a slice named
    by /proc/self/cgroup -- try the nested path first, root second."""
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            rel = line.split(":")[-1].lstrip("/")
            if rel:
                candidates.append(Path("/sys/fs/cgroup") / rel)
    except OSError:
        pass
    candidates.append(Path("/sys/fs/cgroup"))
    for base in candidates:
        try:
            limit = (base / "memory.max").read_text().strip()
        except OSError:
            continue
        if limit == "max" or int(limit) > (1 << 62):
            return None
        return base
    return None


def memory_status():
    """(used_gb, limit_gb, cache_gb, source) where `used` is the OOM-RELEVANT memory: cgroup
    `anon` + kernel stacks + page tables (unreclaimable), NOT `memory.current`. `cache` (page
    cache, reclaimable) is reported for visibility but never counted. Off-cgroup, falls back to
    `total - available`, which already excludes reclaimable cache."""
    base = _cgroup_dir()
    if base is not None:
        try:
            limit = int((base / "memory.max").read_text().strip())
            stat = dict(l.split() for l in (base / "memory.stat").read_text().splitlines())
            unreclaimable = (int(stat.get("anon", 0)) + int(stat.get("kernel_stack", 0))
                             + int(stat.get("pagetables", 0)) + int(stat.get("unevictable", 0)))
            return unreclaimable / 1e9, limit / 1e9, int(stat.get("file", 0)) / 1e9, "cgroup-anon"
        except (OSError, ValueError, KeyError):
            pass
    import psutil
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1e9, vm.total / 1e9, vm.cached / 1e9, "host"


def tree_pss_gb(proc=None):
    """PSS of a process tree in GB -- the true footprint of forked dataloader workers (RSS
    double-counts shared COW pages). Use for the no-GPU worker-memory probe."""
    import psutil
    proc = proc or psutil.Process()

    def pss(p):
        try:
            return p.memory_full_info().pss
        except (psutil.Error, AttributeError):
            return 0

    return (pss(proc) + sum(pss(c) for c in proc.children(recursive=True))) / 1e9


try:
    from fastai.callback.core import Callback
except Exception:  # fastai optional at import time
    Callback = object


class HostMemoryGuard(Callback):
    """Log real (anon) memory every `every` batches and abort with an explanation before the OOM
    killer strikes silently. Thresholds on anon, not memory.current (page cache is reclaimable).

    Per-worker cost is ~1.2 GB (the fastai image-decode pipeline), so num_workers has a hard anon
    ceiling: on a 288 GB node, 128 workers ~= 168 GB (safe), 256 ~= the whole node. The high-water
    mark rises gradually as workers hit larger source images, so an OOM is invisible for a while
    then fatal -- which is what a periodic guard is for.
    """

    order = -7  # before anything that allocates

    def __init__(self, every: int = 500, abort_at_frac: float = 0.92):
        self.every = every
        self.abort_at_frac = abort_at_frac

    def _workers(self):
        # fastai's DataLoader keeps self.num_workers = 1 and hands the real value to the torch
        # loader it wraps (fake_l), so reading dls.train.num_workers reports 1.
        dl = self.learn.dls.train
        return getattr(getattr(dl, "fake_l", None), "num_workers", None) or getattr(dl, "num_workers", "?")

    def before_fit(self):
        import torch  # noqa
        used, limit, cache, src = memory_status()
        n = self._workers()
        print(f"[mem] limit {limit:.0f} GB ({src}), real(anon) {used:.0f} GB, cache {cache:.0f} GB "
              f"(reclaimable) | num_workers={n} -> ~1.2 GB/worker anon. Page cache grows with "
              f"images read but is not counted.", flush=True)
        if isinstance(n, int) and n * 1.2 > limit * 0.9:
            print(f"[mem] WARNING: {n} workers x ~1.2 GB anon is close to the {limit:.0f} GB limit "
                  f"-- lower num_workers.", flush=True)

    def after_batch(self):
        if not self.training or self.learn.train_iter % self.every:
            return
        import torch
        used, limit, cache, src = memory_status()
        gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[mem] batch {self.learn.train_iter}: real(anon) {used:.0f}/{limit:.0f} GB "
              f"({used/limit*100:.0f}%), cache {cache:.0f} GB | GPU {gpu:.1f} GB", flush=True)
        if used / limit > self.abort_at_frac:
            raise RuntimeError(
                f"Unreclaimable memory at {used:.0f}/{limit:.0f} GB ({used/limit*100:.0f}% of the "
                f"{src} limit) -- excludes reclaimable cache, so a real OOM risk. The kernel OOM "
                f"killer takes this process next with no traceback. num_workers={self._workers()} "
                f"x ~1.2 GB/worker is the usual cause; lower it.")
