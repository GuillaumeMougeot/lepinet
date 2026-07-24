# lepinet — the library

The clean, **fastai-only, `mini_trainer`-free** reimplementation of the pipeline developed in
[`../../dev/`](../../dev/). It reproduces the project-best independent-head baseline (test species
macro-F1 **0.9148**) and is generic in the number of hierarchy levels.

- **Users:** [`../../docs/user-guide.md`](../../docs/user-guide.md) — install, CLI, Python API, config.
- **Developers:** [`../../docs/developer-guide.md`](../../docs/developer-guide.md) — architecture, the
  head, adding heads, testing, the lessons encoded.
- **Why it is built this way:** [`../../journal/2026-07-src-lepinet-baseline-port.md`](../../journal/2026-07-src-lepinet-baseline-port.md).

## Status: baseline complete

Every module is implemented and validated. `import lepinet` pulls zero `mini_trainer` /
`mini_metrics` modules.

| module | role |
|---|---|
| `heads.py` | `IndependentHead` (N-level cosine), `PooledHead`, `build_head`, `HEAD_REGISTRY` |
| `loss.py` | `MultiLevelCELoss` |
| `optim.py` | `Muon` + `MuonAuxAdamW` (verbatim port) + `muon_opt_func` |
| `schedules.py` | Muon-safe LR schedules + crash-recovery resume |
| `data.py` | parquet → filtered/cached df → fastai `DataLoaders` (+ oversampling, COW-safe) |
| `metrics.py` | per-level accuracy + macro-F1 (== `mini_metrics`) |
| `callbacks.py` / `memory.py` | `NaNGuard`, dormant `GCCallback`, `HostMemoryGuard` |
| `model.py` | backbone resolution + `Learner` assembly |
| `config.py` | `TrainConfig` (YAML ↔ dataclass) |
| `train.py` / `test.py` / `infer.py` / `export.py` | train / evaluate / predict (TTA) / ONNX |
| `cli.py` | `lepinet {train,test,predict,export}` |

## What was validated (GPU-free)

- **Head load-parity:** the 0.9148 checkpoint loads into `IndependentHead` **bit-exactly**.
- **Full-model reconstruction:** loads with 0 missing / 0 unexpected keys.
- **Training path:** full recipe (Muon + one_cycle + oversampling + callbacks) runs end to end.
- **Evaluate / predict / export:** run against the 0.9148 checkpoint; ONNX matches PyTorch to
  ~1e-5 with `dynamo=False`.

Still open: the from-scratch **train-parity** run to 0.9148 (needs a healthy GPU — see the
developer guide's environment notes).

## Not moved on purpose

`dev/` stays as the frozen experiment record. New experiments `import lepinet` instead of the
numbered scripts; add experimental heads via `HEAD_REGISTRY` in a `dev/` script.
