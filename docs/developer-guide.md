# lepinet — developer guide

This is the clean, `mini_trainer`-free reimplementation of the pipeline that grew up in `dev/`.
The design intent, decisions, and reasoning live in
[`journal/2026-07-src-lepinet-baseline-port.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-src-lepinet-baseline-port.md);
this guide is the practical map for working in the package.

## The guarantee

`import lepinet` pulls **zero** `mini_trainer` / `mini_metrics` modules — enforced by a test and a
one-liner:

```bash
python -c "import lepinet, sys; assert not [m for m in sys.modules if 'mini_trainer' in m or 'mini_metrics' in m]"
```

Everything that used to come from `mini_trainer` (the cosine head, the multi-level loss, the Muon
optimizer, `cosine_to_zscore`, the parent-index masks) is reimplemented here. Only the
**independent** head is kept — the hierarchical/conditional/autoregressive variants were dropped
(autoregressive never won; the marginalization path is in `export.py`).

## Module map

```
src/lepinet/
  heads.py       IndependentHead (N-level cosine), PooledHead, build_head, HEAD_REGISTRY,
                 cosine_to_zscore, sparse_masks_from_labels, build_class_spec
  loss.py        MultiLevelCELoss (+ FastaiLossWrapper)
  optim.py       Muon + MuonAuxAdamW (verbatim port) + muon_opt_func
  schedules.py   warmup_cos / front_loaded schedules + fit_scheduled / fit_resume  (Muon-safe, LR-only)
  data.py        gen_df / filter_df / prepare_df / build_hierarchy / make_dls / sample_weights
  metrics.py     LevelAccuracy, LevelMacroF1 (== mini_metrics), StreamingF1MultiHead, default_metrics
  callbacks.py   NaNGuard, GCCallback (dormant), HostMemoryGuard (re-export)
  memory.py      HostMemoryGuard + cgroup-anon accounting
  model.py       resolve_arch, arch_body_features, build_backbone_model, build_learner
  config.py      TrainConfig dataclass + YAML loading + run-dir stamping
  train.py       train() / train_from_config()
  test.py        evaluate() / load_model() + native metric report
  infer.py       predict() with TTA
  export.py      export_onnx() (dynamo=False) + taxonomy.json + marginalize()
  cli.py         `lepinet {train,test,predict,export}`
```

`dev/` stays as the frozen experiment record; new experiments `import lepinet` rather than the
numbered scripts.

## How the head works

`IndependentHead` is a clean reimplementation of the *math* of `mini_trainer`'s cosine head — not
a subclass of it. A shared bottleneck (`hidden` → LeakyReLU → L2-normalize) produces one unit
embedding; each level scores it against its own unit-norm prototypes (`weight_norm` with the row
norm frozen at 1) via `cosine_to_zscore(F.linear(emb, weight)) + bias`. That's the whole head.

The design chooses **clarity over checkpoint-loadability**: the original mini_trainer checkpoint
does **not** load here, and everything that existed only to serve the mini_trainer class hierarchy
is gone — the `_weight_bias` cache and `active_indices` branching (the per-batch GPU reference-cycle
leak that once forced `GCCallback`), the dead `BatchNorm`, the `linear`/`layers[0]` alias, the
`_extra_state` dict, and the parent-index `mask` buffers (an *independent* head does not use parent
relationships; those live in the checkpoint's hierarchy table and `taxonomy.json`, used only by
`export.marginalize`). Baseline parity is established by **retraining**, not by loading old weights.

The result: no data-dependent control flow on the forward path, so it traces to ONNX with
`dynamo=False` and **no warm-up hack**.

## Adding a new head (the experiment seam)

`PooledHead` is head-agnostic (it pools `[N,C,H,W]→[N,C]` and hands off). Register a new head and
reach it through `build_head` / a config `head:` value without touching the rest:

```python
from lepinet.heads import HEAD_REGISTRY
HEAD_REGISTRY["my_head"] = MyHead   # __init__(in_features, n_classes, hidden=...)
```

A head that needs the taxonomy (e.g. a hierarchical head that propagates probabilities up the
tree) can take it in its own constructor; `build_class_spec` / `sparse_masks_from_labels` in
`heads.py` build the parent-index masks from a dataframe.

Do this in a `dev/` script, not in the package, unless it becomes a permanent option.

## N-level hierarchies

Everything is generic in the number of levels: pass `levels` (fine→coarse) through `TrainConfig`.
The head builds one prototype layer per level and `N-1` parent masks; the loss, metrics, data, and
export all iterate `levels`. The Lepidoptera default is the 3 keys, but nothing is hard-coded to 3.

## Testing

```bash
# unit + self-contained end-to-end (train a dummy model on a generated dataset, then
# eval / predict / export) -- CPU, no data needed. This is what CI runs.
pytest tests/test_heads.py tests/test_loss_metrics_config.py tests/test_export.py tests/test_e2e_synthetic.py

# optional CPU e2e on a slice of the real data/small set
LEPINET_RUN_SLOW=1 CUDA_VISIBLE_DEVICES= pytest tests/test_e2e_cpu.py -v

ruff check src/lepinet tests
```

CI (`.github/workflows/ci.yml`) installs a CPU torch + the fastai stack and runs lint + the unit
tests + the synthetic end-to-end test.

## Reproducing the 0.9148 baseline

Parity is by **retraining** (the clean head cannot load the old mini_trainer checkpoint by design).
The local GPU is currently unusable (driver fault), so the run goes to UCloud (see `../ucloud-api`):
sync the repo, `uv sync`, then `lepinet train -c
configs/20260716_heads_global_independent_muon_5ep_oversample.yaml` on a B200, and
`lepinet test --test-set 0` on a B200-MIG. Expect species macro-F1 **0.9148 ± ~0.2pt**. The package
defaults to `bf16`; add `precision: fp16` to match the original run exactly (both should land within
noise of each other).

## The lessons encoded (don't re-learn them)

Each is a scar from a `dev/` investigation; the journal has the full story.

1. **bf16, not fp16, by default** — fp16 overflows the cosine head to NaN; the head is forced fp32
   regardless. `[[2026-07-autoregressive-fp16-instability]]`
2. **Under-annealing was the biggest optimisation lever** — `one_cycle` ≫ `flat_cos`.
   **Oversampling was the biggest data lever** — +1.7pt. `[[2026-07-why-was-fastai-behind-mini-trainer]]`,
   `[[2026-07-does-longtail-help]]`
3. **Muon needs LR-only scheduling and an unfrozen model** (it re-partitions param groups).
4. **The dataloader's memory is the image-decode pipeline (~1.2 GB/worker)**, guarded on cgroup
   *anon*, not `memory.current`. `[[2026-07-ucloud-benchmark-oom]]`
5. **Derive the hierarchy from the training df, not a hierarchy.csv** — a stale file silently
   truncates the class set.
6. **Class-distribution regularization and uniform-tau logit adjustment both lost** — rejected by
   `TrainConfig`. `[[2026-07-does-longtail-help]]`

## Environment

The venv is **reproducible from `pyproject.toml` + `uv.lock`** — `uv sync` now works (it used to
break the hand-managed venv because the old pyproject didn't declare the full dependency set or the
CUDA-13 torch index; `[[2026-07-venv-uv-sync-incident]]` is the historical context, now resolved):

```bash
uv sync --all-extras                              # library + export + timm + dev tooling
uv pip install -r dev/requirements-experiments.txt  # mini_trainer / mini_metrics, for dev/ scripts only
```

`torch==2.12.1+cu130` (the Blackwell / RTX 5090 / B200 build) is pinned to the PyTorch CUDA-13 wheel
index via `[[tool.uv.index]]` + `[tool.uv.sources]`. The dev/ experiment packages are **not** locked
(they require Python ≥3.12, which would over-constrain lepinet's ≥3.10 support) — hence the separate
requirements file.

Gotchas:

- **NVML / driver mismatch:** if `nvidia-smi` fails with a version mismatch, training crashes in the
  backward (the allocator calls NVML). Reload the driver modules (`rmmod`/`modprobe nvidia*`) or, if
  that's not possible, train on UCloud.
- **`fork` start method** is forced (`ensure_fork_start_method`) — Python 3.14 defaults Linux to
  `forkserver`, which can't pickle a CUDA tensor in the aug warm-up batch.
