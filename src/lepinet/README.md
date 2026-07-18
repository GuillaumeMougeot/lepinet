# lepinet — the library

The cleaned-up, reusable version of the pipeline that grew up in [`../../dev/`](../../dev/).
`dev/030` is the reference implementation; this package is that logic, factored into importable
modules with every lesson from the dev/ investigations baked in, plus the nvJPEG GPU-decode
reader as a first-class option.

**Status: in progress.** This is a deliberate migration, not a rewrite — each module states
whether it's `DONE` (extracted + usable) or `PORTING` (still living in dev/030, to move here).
The point is that the *lessons* are captured here even before every line is moved, so nothing
learned the hard way gets re-learned.

## Why a package now

`dev/` is a numbered lab notebook: scripts import each other by filename via `importlib`, only
run from the repo root, and duplicate setup. That was right for exploration. But 028/030/032/034
stopped being experiments and became infrastructure (030 is 800+ lines), and the same recipe now
runs in three places (local, UCloud, tests). A package makes it `from lepinet import train`,
testable, and installable on a fresh node without the repo-root `sys.path` trick.

## Architecture

```
lepinet/
  data.py        # gen_df / filter_df / prepare_df (parquet -> filtered df, cached),
                 #   make_dls (fastai DataBlock; numpy-indexed items to avoid the fork/COW leak)   [PORTING from dev/028]
  gpu_decode.py  # the nvJPEG reader: CPU workers read bytes, GPU decodes -- 10x less host memory   [DONE]
  memory.py      # HostMemoryGuard + cgroup-anon accounting + the worker-memory probe               [DONE]
  schedules.py   # warmup_cos / one_cycle / front_loaded + fit_resume (crash recovery)              [DONE]
  heads.py       # MTHeadAdapter, build_head, build_class_spec (fp32 head under mixed precision)     [PORTING from dev/030]
  optim.py       # muon_opt_func (Muon on backbone 2D weights, AdamW on the head)                    [PORTING from dev/030]
  callbacks.py   # GCCallback, NaNGuard, SupervisionContextCallback                                  [PORTING from dev/030]
  losses.py      # MultiLevelLossWrapper                                                             [PORTING from dev/030]
  longtail.py    # sqrt oversampling + logit adjustment                                              [PORTING from dev/034]
  metrics.py     # LevelAccuracy, LevelMacroF1, StreamingF1MultiHead                                 [PORTING from dev/028]
  train.py       # train() orchestration -- config -> trained .pt                                    [PORTING from dev/030]
  test.py        # evaluate() -- .pt -> predictions + mini_metrics report                            [PORTING from dev/032]
  infer.py       # single-image / folder inference for deployment                                    [TODO]
```

## The lessons this package encodes (so they're not re-learned)

Each is a scar from a dev/ investigation; the linked journal entry has the full story.

1. **Default to bf16, never fp16.** fp16 overflows mini_trainer's cosine head and, worse, the
   autoregressive decoder's compounding generation → NaN. bf16 has fp32's exponent range and
   fixes it upstream (the head is already forced fp32; the *backbone* is what overflows in fp16).
   → `train.py` defaults bf16. [[2026-07-autoregressive-fp16-instability]]

2. **The dataloader's memory is the image-decode pipeline (~1.2 GB/worker), not the dataframe.**
   `num_workers × 1.2 GB` must fit the node's *anon* memory. The guard must threshold on cgroup
   `anon`, NOT `memory.current` (which counts reclaimable page cache and false-aborts).
   → `memory.py`. [[2026-07-ucloud-benchmark-oom]]

3. **GPU decode (nvJPEG) is a memory lever, not a speed lever — for a small model.** It collapses
   per-worker memory ~10x (workers hold bytes, not decoded pipelines), but effnetv2s is model-
   bound at ~1100 img/s either way. Worth it for a heavier compute-bound model, or memory-tight
   nodes. Must be pipelined (overlap decode with compute) + gc-every-N, or it's *slower*.
   → `gpu_decode.py`. [[2026-07-ucloud-throughput]]

4. **Save a hierarchy derived from the training df, not from a hierarchy.csv file** — a stale file
   silently truncates the checkpoint's class set and breaks eval. → `data.py`/`train.py`.

5. **Under-annealing was the biggest optimisation lever; oversampling the biggest data lever.**
   one_cycle (anneals ~90% of the run) >> flat_cos; sqrt oversampling → +1.7pt.
   → `schedules.py` / `longtail.py`. [[2026-07-why-was-fastai-behind-mini-trainer]],
   [[2026-07-does-longtail-help]]

6. **Muon needs LR-only scheduling and an unfrozen model** (it re-partitions param groups and
   takes tuple betas). → `optim.py` / `schedules.py`.

7. **The cosine head emits NaN grads at init** (a ~0-norm feature) — skip non-finite-grad steps.
   → `callbacks.py` (NaNGuard) / `train.py`.

## Not moved on purpose

The `dev/` scripts stay as the frozen experiment record; this package is the forward path. When a
module here reaches `DONE`, the matching `dev/` script keeps working (it imports its own copy) but
new work should build on `lepinet`.
