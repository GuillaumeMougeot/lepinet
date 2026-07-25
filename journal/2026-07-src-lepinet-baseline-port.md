# Porting the 0.9148 baseline into `src/lepinet` — a fastai-only, mini_trainer-free clean repo

**Status:** IN PROGRESS — package **built and validated GPU-free** on 2026-07-24 (see the
Execution log at the end); the one open item is the from-scratch train-parity run to 0.9148,
blocked on a local GPU driver fault. Companion in spirit to [[2026-07-lepi-app-claude]]: this
document is the *how* — module layout, what to keep vs cut, the sequencing, and the decisions
that need to be right before code is written. Written 2026-07-24.

**Goal in one line:** reproduce the project-best **0.9148 test species macro-F1**
(`20260716-154156`, [[2026-07-does-longtail-help]]) from a clean, installable `src/lepinet`
package that has **zero `mini_trainer` / `mini_metrics` imports**, implements only the
**independent** head, is **fastai-only**, and exports to ONNX without `dynamo`. `dev/030` stays
frozen as the experiment record; new experiments import `lepinet` instead of `028`/`030`/`034`.

---

## 0. The facts this plan is built on (measured, not assumed)

| quantity | value | source |
|---|---|---|
| target run | `20260716-154156` | `RESULTS.md`, [[2026-07-does-longtail-help]] |
| target checkpoint | `data/global/models/20260716-154156-…-oversample-effnetv2s/…​.pt` (173 MB) | on disk |
| test species macro-F1 to match | **0.9148** (val 0.9096) | `RESULTS.md` |
| winning recipe | effnetv2_s, independent head, Muon, one_cycle, warmup 0.5ep, grad_clip 5.0, light aug, bs 64, 460→256, **5 ep**, `oversample_power: 0.5`, precision fp16 (bf16 also fine) | `configs/20260716_heads_global_independent_muon_5ep_oversample.yaml` |
| classes | 12,041 species / 4,333 genus / 102 family | checkpoint vocabs |
| head is | cosine / L2-normalized prototypes, `normalized=True`, `hidden=True` (=1280) | `mini_trainer/modeling/classifier.py` |
| `src/lepinet` DONE modules | `gpu_decode`, `memory`, `schedules` | `src/lepinet/README.md` |

Two load-bearing facts about the head, because everything in §2–§4 turns on them:

1. **The independent head is `IndependentClassifier(ConditionalClassifier(HierarchicalClassifier(Classifier)))`.**
   Only the leaves of that chain are used at inference: per-level cosine logits computed from
   one shared, L2-normalized embedding (`marginals()`), with `cosine_to_zscore` + a frozen-norm
   `weight_norm` layer per level. The `HierarchicalClassifier` machinery (sparse masks,
   `batched_scatter_logsumexp`, `hierarchy()`) is **not** on the independent forward path — it
   is used only to *size* the coarser layers at construction, and (separately) for the
   marginalization export in `dev/042`.
2. **The saved `state_dict` persistent keys are few and stable:** `hidden.{weight,bias}`,
   `batch_norm.*`, `linear.bias`, `linear.parametrizations.weight.{original0,original1}`,
   `layers.{0,1,2}.*` (layer 0 *is* `linear`, shared), `mask_0`, `mask_1`, and an `_extra_state`
   metadata dict. The cache buffers (`_linear_weight`, `_mask_i`, `_filter_i`, …) are all
   `persistent=False` — they are **not** in the checkpoint. This is what makes a clean
   reimplementation able to load the existing 0.9148 weights (§4, D1).

---

## 1. Definition of done

The port is done when **all** of these hold:

- `pip install -e .` gives `from lepinet import train, evaluate, predict, export_onnx` with **no
  `mini_trainer` or `mini_metrics` on the import graph** (`python -c "import lepinet; import sys;
  assert not [m for m in sys.modules if 'mini_trainer' in m or 'mini_metrics' in m]"`).
- **Load parity:** the existing `20260716-154156.pt` loads into the new `IndependentHead` and
  reproduces dev/032's test report to <1e-4 on species macro-F1 (D1). This is the cheap
  validation — no GPU-hours.
- **Train parity:** a fresh 5-epoch run from the ported `train()` on the same config lands at
  **0.9148 ± noise** (the run-to-run noise floor in this project is ~0.2pt, [[2026-07-does-longtail-help]]).
- `evaluate()` writes the same per-level macro-F1 / micro-acc table dev/032 produced, computed
  **natively** (no mini_metrics), and those numbers match mini_metrics on one cross-checked run.
- `predict()` does single-image and folder inference **with TTA**, returning per-level top-k
  labels + calibrated-or-raw confidence.
- `export_onnx()` produces a `.onnx` + `taxonomy.json` that matches PyTorch logits to <1e-3
  (the dev/040 bar), with `dynamo=False`.
- `dev/` still runs: a thin `dev/048_*` demonstrates the *same* recipe driven by
  `import lepinet` instead of `028`/`030`/`034`.

---

## 2. What to keep, what to cut (the strict minimum)

The whole reason the gap to mini_trainer closed was the *training loop*, not the head
([[2026-07-why-was-fastai-behind-mini-trainer]]). So the head can be reimplemented faithfully
and small; the value is in the loop lessons, which are already partly in `src/lepinet`.

**Reimplement (the strict minimum to be mini_trainer-free):**

| piece | source today | size | notes |
|---|---|---|---|
| `IndependentHead` (cosine, per-level) | `Classifier`+`ConditionalClassifier`+`IndependentClassifier` | ~120 lines | flatten the 4-class chain into one class; keep numerics + persistent buffer names identical (D1); **drop** the `_weight_bias` cache and `active_indices` branching (D3) |
| `cosine_to_zscore` | `mini_trainer/utils/_core/math.py` | 6 lines | copy verbatim |
| `sparse_masks_from_labels` | `hierarchical/integration.py` | ~30 lines | copy; needed to size coarse layers + for export marginalization |
| `MultiLevelCELoss` | `hierarchical/loss.py` `MultiLevelWeightedCrossEntropyLoss` | ~40 lines | per-level CE + the hierarchical label-smoothing adjustment `ls_L = 1-(1-ls)^(1/(L+1))` |
| `Muon` + `MuonAuxAdamW` | `training/muon.py` | 456 lines | **port verbatim** — self-contained (only `torch`), and it is load-bearing for the 0.9148 recipe |
| `muon_opt_func` | `dev/030` | ~20 lines | already understood; names last group `head_nomuon` |
| `batched_scatter_logsumexp` | `hierarchical/utils.py` | ~15 lines | **export-only** (marginalization), not on the train/infer path |

**Cut entirely (not on the independent path, or a proven wash):**

- `HierarchicalClassifier` / `ConditionalClassifier` forwards, `AutoregressiveClassifier`,
  `AutoregressiveMixin`, `XADecoder`, the whole `transformer.py` — autoregressive was never a
  win and died repeatedly (`RESULTS.md`).
- `SupervisionContext` / `SupervisionContextCallback` — only the autoregressive head read it.
- `ClassDistributionRegularizer` — a measured **wash** (0.8860 vs 0.8880,
  [[2026-07-why-was-fastai-behind-mini-trainer]]). Do not port.
- `LogitAdjustment` / `LogitAdjustCallback` — a **structurally-flawed** result (0.9031, wrecked
  family/genus, [[2026-07-does-longtail-help]]). Do not port into the baseline; leave it as a
  dev/ experiment if revisited (per-level tau).
- `mini_metrics` — its macro-F1 is *already* reimplemented and verified byte-matching in
  `LevelMacroF1` (`dev/028`, docstring). Reuse that for the native report (D2).
- `EmbeddingContext` — drop from the baseline forward; re-add a clean hook when distillation
  needs embeddings (§7).

**Keep as-is (already DONE in `src/lepinet`):** `schedules.py` (warmup_cos / one_cycle /
front_loaded / `fit_resume`), `memory.py` (`HostMemoryGuard`), `gpu_decode.py` (optional reader).

---

## 3. Target module layout

```
src/lepinet/
  __init__.py     # export train, evaluate, predict, export_onnx, build_learner, __version__
  config.py       # yaml -> validated dataclass; version check; out-dir stamping (from 030.cli)
  data.py         # gen_df/filter_df/prepare_df/build_hierarchy/make_dls (028) + sample_weights (034)
  heads.py        # IndependentHead (cosine), build_head, build_class_spec, sparse_masks_from_labels,
                  #   cosine_to_zscore, MTHeadAdapter->PooledHead   [reimpl, mini_trainer-free]
  loss.py         # MultiLevelCELoss (+ scalar-sum wrapper for fastai)
  optim.py        # muon.py (ported verbatim) + muon_opt_func
  callbacks.py    # NaNGuard (keep), HostMemoryGuard (re-export memory.py), GCCallback (see D3)
  metrics.py      # LevelAccuracy, LevelMacroF1, StreamingF1MultiHead (from 028, verified==mini_metrics)
  model.py        # resolve_arch, arch_body_features, build_backbone_model, build_learner
  train.py        # train(cfg) -> .pt  (independent-only; the full winning recipe)
  test.py         # evaluate(.pt) -> predictions.csv + native metric report
  infer.py        # predict(): single-image / folder, with TTA  [NEW]
  export.py       # export_onnx() + taxonomy.json + marginalize()  (from 040/042)  [NEW here]
  cli.py          # lepinet-train / -test / -predict / -export entry points -> pyproject [scripts]
  schedules.py memory.py gpu_decode.py   # DONE, unchanged
```

`dev/README` gets a row for the new `dev/048` demo; `src/lepinet/README.md`'s status table flips
each module DONE as it lands.

---

## 4. The decisions that must be right before writing code

### D1 — Load the existing 0.9148 checkpoint into the clean head. **Recommend: yes.**

The clean `IndependentHead` should register the *same persistent buffers* and keep the *same
numerics* as the mini_trainer head, so `20260716-154156.pt` loads directly. Payoff: an
immediate, GPU-free proof of correctness (load → run dev/032's fold → expect 0.9148), plus a
working ONNX/predict/test pipeline on day one, before any retraining. The forward can still be
**much simpler** than mini_trainer's (D3) as long as it is numerically identical. Risk: the
`layers[0] is linear` aliasing and the `weight_norm` `original0/original1` split must be
reproduced exactly — verified feasible from the key list in §0. If a key mismatch appears, a
tiny `_remap_legacy_state_dict()` closes it.

### D2 — Drop `mini_metrics`, compute the report natively. **Recommend: yes.**

`LevelMacroF1` already reproduces mini_metrics' macro-F1 (per-class P/R harmonic mean, 0 when
either is 0, unweighted mean over present classes) and was validated like-for-like
([[2026-07-why-was-fastai-behind-mini-trainer]], the "0.83 == 0.83" resolution). Reimplementing
the report also deletes the broken-`--optimal` two-pass workaround (`dev/032.compute_metrics`).
**Gate:** on the first `evaluate()`, diff native vs mini_metrics on one run and assert
agreement before trusting it, then never depend on mini_metrics again.

### D3 — Delete the `_weight_bias` cache + `GCCallback`, or keep the callback. **Recommend: delete the cache; keep GCCallback as a cheap safety net initially, then measure.**

The `GCCallback` (`gc.collect(0)` every batch) exists solely to break the reference cycle that
mini_trainer's `_weight_bias()` creates by stashing a graph-attached weight view into a
persistent buffer every forward ([[2026-07-why-was-fastai-behind-mini-trainer]], landmines).
A clean forward that uses `self.linear.weight` directly (no caching, no `active_indices`) **has
no such cycle**, so the leak should vanish and `GCCallback` becomes unnecessary. But that is a
claim, not yet a measurement — so: reimplement without the cache, keep `GCCallback` for the
first full run, watch GPU memory, and remove it only after confirming memory stays flat without
it. (This is the single largest simplification available and it also produces a cleaner ONNX
graph — no `training`/`active_indices` branches to trace.)

### D4 — ONNX exporter: `dynamo=False`. **Recommend: yes (default), dynamo opt-in.**

dev/040 already exports successfully with the legacy TorchScript exporter, and the independent
head has **no data-dependent control flow** (just `linear`, `normalize`, `acos`, affine), so it
traces cleanly. Keep `dynamo=False` as the default for reproducibility of the app pipeline;
expose `--dynamo` as an experiment flag in `dev/`. The user's instinct here is right: nothing in
the independent path needs dynamo, and avoiding it removes a class of exporter surprises.

### D5 — fp16 vs bf16 for the reproduction run. **Recommend: reproduce at fp16 first, then keep bf16 as the package default.**

The 0.9148 run used `precision: fp16` and survived because the head is forced fp32 inside the
adapter. To *reproduce the number* exactly, run fp16. But the package should **default to bf16**
(the `src/lepinet/README.md` lesson #1, [[2026-07-autoregressive-fp16-instability]]) since it is
strictly safer at scale; verify bf16 lands within noise of fp16 as a second data point.

---

## 5. Step-by-step sequencing

Numbered so each step is independently checkable. Phases P0–P2 are GPU-free.

### P0 — Skeleton + the head, proven by load parity (no GPU)

| # | action | done when |
|---|---|---|
| P0.1 | `heads.py`: `cosine_to_zscore`, `sparse_masks_from_labels`, `IndependentHead`, `build_class_spec`, `build_head`, `PooledHead` (the fp32 pool+adapter) | imports clean, no mini_trainer |
| P0.2 | `model.py`: `resolve_arch` / `arch_body_features` / `build_backbone_model` (from 030) | effnetv2_s body builds, nf=1280 |
| P0.3 | Load `20260716-154156.pt` into `build_backbone_model(effnetv2_s, IndependentHead(...))` | `load_state_dict` returns no missing/unexpected keys (add `_remap_legacy_state_dict` if needed) |
| P0.4 | `metrics.py` + `test.py` native report; run on fold `set=='0'` | **species macro-F1 == 0.9148** and per-level table matches dev/032 (D1 ✔, D2 gate ✔) |

*P0.4 is the keystone: if the loaded checkpoint reproduces 0.9148 through the new code, the head,
metrics, and eval path are all correct before a single training step.*

### P1 — Export + predict on the loaded checkpoint (no GPU)

| # | action | done when |
|---|---|---|
| P1.1 | `export.py`: `export_onnx()` (bake imagenet norm + resize spec, raw logits out, `dynamo=False`) + `taxonomy.json` (from 040) | logits match PyTorch <1e-3 (D4 ✔) |
| P1.2 | `infer.py`: `predict()` single-image + folder, **TTA over flips** (identity / hflip / vflip / hflip+vflip; average per-level softmax), top-k + confidence | runs on a handful of images; TTA ≥ no-TTA on a small labelled sample |
| P1.3 | `export.py`: `marginalize()` (genus/family from species via parent arrays + `batched_scatter_logsumexp`) | reproduces dev/042 result; **not** on the default infer path |

*TTA note:* fastai's `learn.tta()` averages a single `learn.pred`, which is a **list** of per-level
tensors here — it won't average cleanly, so `predict()` implements TTA explicitly (cheap: 4
forward passes, flips only, matching the training aug `flip_vert=True` + hflip). Multi-crop is a
later knob.

### P2 — Training path, proven by train parity (GPU)

| # | action | done when |
|---|---|---|
| P2.1 | `loss.py` `MultiLevelCELoss` + scalar wrapper; `optim.py` port `muon.py` + `muon_opt_func`; `callbacks.py` `NaNGuard`(+`GCCallback` per D3); `data.py` port `gen_df`/`make_dls`/`sample_weights` | unit: loss/optimizer/dls build on a tiny parquet |
| P2.2 | `train.py`: assemble the full winning recipe via `build_learner`; wire `schedules.py` (one_cycle + warmup 0.5ep) | 1-epoch smoke run on a small dataset completes, .pt written |
| P2.3 | Full 5-ep run on `configs/20260716_…_oversample.yaml` (fp16, D5), fold '1' | **test species macro-F1 = 0.9148 ± ~0.2pt** vs `20260716-154156` |
| P2.4 | Repeat P2.3 at bf16 | within noise of P2.3 (D5 ✔) |
| P2.5 | `dev/048_baseline_via_lepinet.py`: same recipe, `import lepinet` | matches P2.3; `dev/README` row added |

### P3 — Package hygiene

CLI entry points in `pyproject [project.scripts]`; `src/lepinet/README.md` status → DONE;
regenerate `RESULTS.md` if P2.3/P2.4 add rows; a short `src/lepinet` usage snippet in the repo
README. Confirm the mini_trainer-free assertion (§1) in CI-style one-liner.

---

## 6. Modular room for experimentation (explicitly designed in)

The package must not paint experiments into a corner. Seams to leave open:

- **New heads.** `build_head(name, …)` keeps a registry; the baseline registers only
  `"independent"`. A `dev/` script can register a tiny-ViT head or re-add
  hierarchical/autoregressive *without touching the package* (the C2 sweep already wants tiny
  ViTs, [[2026-07-lepi-app-claude]] §7). Keep `PooledHead` head-agnostic (it just pools to
  `[N, nf]` and hands off).
- **mini_trainer compatibility.** Because `IndependentHead` keeps the persistent buffer names
  (D1), a checkpoint can still be *cross-loaded* both ways during the transition — useful for
  spot-checking the port against mini_trainer output. This is a debugging affordance, not a
  dependency.
- **nvJPEG GPU decode.** `gpu_decode.py` is already DONE; `make_dls` should accept a
  `reader=` seam so a dev/ run can swap the CPU decode for the nvJPEG reader
  ([[2026-07-ucloud-throughput]]) — a memory lever, not a speed one for effnetv2s.
- **Distillation.** Re-add a minimal embedding hook (the dropped `EmbeddingContext`, cleanly)
  behind a flag so `dev/045_distill` can pull soft targets / embeddings from a teacher. Keep it
  off the baseline forward.
- **Backbone/bottleneck sweeps.** `resolve_arch` already routes timm names; `hidden:int`
  already bottlenecks. These stay first-class config knobs, so the C1/C2 sweeps
  ([[2026-07-lepi-app-claude]]) run through `lepinet` unchanged.

---

## 7. Long-run direction (ideating; not part of the step-by-step)

Kept deliberately separate so it does not leak scope into §5:

- **Bigger everything.** Higher resolution, larger batches, more epochs, fastai default aug +
  **cutmix/mixup** (the winning recipe deliberately used *light* aug for a 5-epoch budget;
  mixup pays at longer budgets). Ensemble over the CV folds if it earns its inference cost.
- **Knowledge distillation** from the large teacher into small students — the core of the app's
  size budget ([[2026-07-lepi-app-claude]] Phase C). The package should make "train teacher →
  distil student → export student" a straight line.
- **Generalization beyond Lepidoptera.** Nothing in the head or loop is moth-specific except the
  `speciesKey/genusKey/familyKey` column names and the 3-level assumption. A later rename could
  parameterize the level list and become a general hierarchical image classifier. **Not now** —
  but keeping `HIERARCHY_LEVELS` a single injectable constant (not hard-coded across modules) is
  the cheap thing to do today that keeps that door open.

---

## 8. What this proposal asserts that is not yet measured

Stated up front so a future reader can score predictions vs results (the
[[2026-07-does-longtail-help]] discipline):

- **D1 load parity holds:** the clean head loads `20260716-154156.pt` and reproduces 0.9148 to
  <1e-4. *(Reasoning: persistent keys are few and numerics are copied.)*
- **D3 the GC leak is gone** once `_weight_bias` caching is removed — `GCCallback` becomes a
  no-op and can be deleted. *(Reasoning: the reference cycle was the cache; no cache, no cycle.
  Untested — hence keep the callback for run 1.)*
- **Train parity holds:** a fresh fp16 5-ep run reproduces 0.9148 within ~0.2pt.
- **bf16 == fp16 within noise** for this recipe (D5).
- **Native metrics == mini_metrics** (D2 gate).

If D1 or the train parity misses by more than noise, suspect (in order): the `weight_norm`
`original0`-frozen-at-1 reconstruction, the `layers[0] is linear` aliasing, then the fp16 head
autocast boundary — those are the three places numerics can silently drift.

---

## 9. Open questions for the owner

1. **Reproduce-then-simplify, or simplify-first?** The plan reproduces 0.9148 *faithfully*
   (D1/D5) before removing anything visible in the numbers. That is the conservative order and
   what §5 assumes. Say if you'd rather I take structural liberties earlier (e.g. drop
   `weight_norm` and retrain) at the cost of the free load-parity check.
2. **Where should the package's `data/` defaults point** for the smoke tests — a specific small
   parquet under `data/`? (`DEVELOPER.md` §5 says don't use `data/global` for tests.)
3. **CLI vs config-only.** Keep the `--config yaml` entry (dev parity) *and* add
   `lepinet-train`? Or config-only to start. I lean: both, config is the source of truth.

---

## Execution log — 2026-07-24 (the build)

Built the whole package in one session, **dual approach** (§9 Q1): the clean head was written
from scratch *and* kept load-compatible with the 0.9148 checkpoint, because load-parity is what
lets every downstream module be validated GPU-free instead of waiting on the retrain.

### What was validated (all GPU-free, all passing)

| check | result |
|---|---|
| **Head load-parity (D1)** | 0.9148 checkpoint → clean `IndependentHead`, **max\|Δlogit\| = 0.000e+00**, 0 missing/unexpected keys |
| **Full-model reconstruction** | `nn.Sequential(body, PooledHead)` loads the checkpoint, 0 missing / 0 unexpected |
| **Training path** | full recipe (Muon+AdamW, one_cycle+warmup, oversampling, NaNGuard/HostMemoryGuard/GradientClip/CSVLogger/SaveModel, all per-level metrics) runs end to end on CPU; loss decreases; checkpoint saved |
| **evaluate()** | reconstruct → stream inference → native per-level macro-F1/micro-acc → `predictions.csv` in mini_metrics format. On 80 imgs: species F1 0.89, micro-acc 0.94 (consistent with the model's known 0.9148/0.9476) |
| **predict() + TTA** | single/folder, 4-flip TTA, top-k; two held-out images correctly classified at 0.998–0.999 |
| **export_onnx() (D4)** | `dynamo=False`, **no warm-up hack needed** — PyTorch vs ORT max\|Δlogit\| 2.6e-5, 100% top-1 agreement; taxonomy.json + MANIFEST.json emitted |
| **marginalize()** | per-level marginals sum to 1 (consistent by construction) |
| **import is mini_trainer-free** | `import lepinet` loads zero `mini_trainer`/`mini_metrics` modules |
| **unit tests + ruff** | 22 synthetic unit tests pass; `ruff check` clean; full CPU e2e (train→eval→predict→export) passes under `LEPINET_RUN_SLOW=1` |

### How the decisions actually resolved

- **D1 (load the old checkpoint): confirmed, bit-exact.** The persistent-key list was small and
  stable exactly as predicted; the only surprise was the metadata field `hidden=None` (stale), so
  geometry is inferred from the weights instead (`infer_hidden_from_state_dict`).
- **D2 (drop mini_metrics): done**, and per the owner's steer `evaluate()` still writes
  `predictions.csv` in mini_metrics' long input format for interop. Native-vs-mini_metrics
  cross-check on the full fold is deferred to when the GPU is back.
- **D3 (drop `_weight_bias` cache): done.** `GCCallback` is shipped **dormant** (in `callbacks.py`,
  documented, off by default) per the owner's steer — kept ready, to be deleted only after a full
  run confirms memory stays flat without it.
- **D4 (`dynamo=False`): confirmed and *better than expected*.** The clean head's lack of
  data-dependent control flow removed both the `dynamo=True` requirement *and* the lazy-cache
  warm-up hack that `dev/040` needed. (Note: the legacy TorchScript exporter is deprecated in
  torch ≥2.9 but works; `--dynamo` remains available.)
- **D5 (bf16 default): done.** Package defaults to bf16; fp16 still supported for exact
  reproduction of the original run.
- **§7 generalization: implemented now, not deferred.** The head, loss, masks, data, metrics,
  export and config are all generic in the number of levels (`levels` config field), defaulting to
  the 3 Lepidoptera keys.

### Simplifications realized (the "build to simplify" half of the dual approach)

Removed vs `mini_trainer`, all verified harmless to numerics: the `_weight_bias` cache,
`active_indices` masking/branching, `EmbeddingContext`, `get/set_extra_state`, the
hierarchical/conditional/autoregressive forwards, the transformer/decoder, `SupervisionContext`,
the class-distribution regularizer, and logit adjustment. `TrainConfig` actively **rejects** the
two interventions that were measured to lose.

### The one blocker: local GPU driver fault

The RTX 5090's userspace NVML is out of sync with the running driver (`nvidia-smi` → "Driver/
library version mismatch", NVML 595.84). `torch.cuda` basic ops pass, but **real training crashes
in the backward** at `nvmlInit_v2_()` inside the CUDA caching allocator, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` does **not** fully fix it. So the from-scratch
train-parity run (P2.3, ~6.5h) could not be launched. Fix is a driver reload / reboot (host-level,
likely owner action) or run it on UCloud. Everything else was validated on CPU / against the
existing checkpoint, so the retrain is the *only* remaining validation, not a dependency for the
rest of the package.

**Next when GPU is healthy:** `lepinet train -c configs/20260716_heads_global_independent_muon_5ep_oversample.yaml`
(add `precision: fp16` to match the original exactly), then `lepinet test --test-set 0` → expect
0.9148 ± ~0.2pt, and cross-check native metrics vs mini_metrics once (D2 gate).

---

## Execution log — 2026-07-24 (phase 2: clarity, packaging, UCloud)

Second working block, after the owner's review. Everything below was done and validated this
session.

### What was learned/read (phase 2)
- **`ucloud-api`** (`~/codes/ucloud-api`, authenticated): job = a TOML spec (`ucloud/*.toml`)
  mirroring UCloud's JobSpecification + `[sync]` (rsync the repo to a drive, respects .gitignore)
  + `[setup]` (script + `run` batch command) + `[schedule]` (auto_extend). Queue: `ucloud q
  submit <spec> --name X [--after Y]` (afterok deps), advanced by `ucloud q daemon --until-idle`.
  Data lives at `/12347837/datasets/global_lepi` → mounts `/work/global_lepi`; the synced repo is
  `/work/lepinet`. Train product `gpu-nvidia-b200-1-gpu`, test `gpu-nvidia-b200-1-mig.1g`.
- **The local GPU driver fault is terminal for local training**: NVML mismatch crashes the
  backward; the owner can't reboot (encrypted disk needs a physical keyboard). → all GPU work
  moved to UCloud. [[gpu-nvml-driver-mismatch]]
- **The "broken venv" was a pyproject problem, not a uv problem.** `uv sync` broke the venv only
  because pyproject declared `packages=[]` + an incomplete dep set + no lock + no torch index. Fixed
  → `uv sync` now works. [[venv-is-hand-managed-never-uv-sync]] (now RESOLVED).

### What was done (phase 2)
- **Maximal clarity over checkpoint-loadability** (owner's call): stripped the load-compat cruft
  from `IndependentHead` — dead BatchNorm, `linear`/`layers[0]` alias, `mask` buffers, cls2idx from
  checkpoints. The old 0.9148 checkpoint no longer loads; parity is now by **retraining**. Head is
  now just bottleneck + N cosine layers.
- **typer CLI** (replaced argparse); **MkDocs + Material** docs with a **GitHub Pages** deploy
  workflow; **CI runs a real end-to-end** (synthetic dataset → train/eval/predict/export), not just
  units.
- **Reproducible venv**: rebuilt from a proper `pyproject.toml` + committed `uv.lock` (cu130 torch
  via `[[tool.uv.index]]`); dev-script deps (`mini_trainer`/`mini_metrics`, which pin py≥3.12 and
  would over-constrain the lock) live in `dev/requirements-experiments.txt`. Verified by full
  teardown + rebuild + tests.
- **Launched training on UCloud B200**: smoke (family 9717) → **validated the whole path on a real
  B200** (`uv sync --frozen` from the lock works, preflight OK, `lepinet train` runs) → 5-epoch
  oversample run auto-launched by the daemon and is **converging healthily** (train_loss 18.9→4.2 in
  epoch 1, ~1:17/epoch, host anon 91/288 GB — safe). Eval on fold 0 queued on a B200-MIG `--after`.
  This is the pending **train-parity** check vs 0.9148.

## Train-parity check — the 5ep run scored 0.9455, and why that is NOT a win (2026-07-25)

The 5ep run finished; my `test.py` reported **species macro-F1 0.9455** on the fold-0 test. That is
*above* the 0.9148 target, which is the wrong direction for a port meant to *reproduce* — a red
flag, not a celebration. Two independent checks:

**1. Is the metric implementation right?** Yes — **bit-exact with `mini_metrics`.** Downloaded the
run's `predictions.csv` (the interop file, 132 MB, 1.45 M rows) off the UCloud drive and ran
`mini_metrics.MacroF1` on it directly. At no-abstain threshold the two agree to 4 dp on every level:

| level | my `test.py` (native) | `mini_metrics` @thr=0 | @thr=0.5 |
|---|---|---|---|
| species | 0.9455 | **0.9455** | 0.9414 |
| genus | 0.9690 | **0.9690** | 0.9700 |
| family | 0.9803 | **0.9803** | 0.9821 |

So `LevelMacroF1`/`macro_f1` reproduces `mini_metrics` exactly (both macro-average over classes
*present as ground-truth labels*, both set F1=0 when precision or recall is 0). The 0.9455 is a
faithful number — of a *different eval set*.

**2. Is the eval set the same as the 0.9148 baseline?** **No — this was the bug.** The 0.9148
baseline (`20260716-154156`, journal [[2026-07-does-longtail-help]]) was measured over the *whole*
test fold: **629,742 images, 12,041 species** (dev/032 test default `min_img_per_spc=0`). My UCloud
test job (`lepinet-test.toml`) passed **`--min-img-per-spc 50`**, copied thoughtlessly from the
*training* config. That filter drops every species with <50 images *in fold 0* — the entire hard
long tail — leaving **484,299 images, 3,696 species**. Macro-F1 weights every class equally, so
removing 8,345 rare (low-per-class-F1) species lifts the average from ~0.91 to 0.9455. The tell:
**micro-acc went the other way** (93.79% here vs 94.76% baseline) — a genuinely better model raises
both; an easier *macro* subset raises only macro. `min_img_per_spc` belongs on *training* (which
species the model learns), never re-applied to the *test* fold.

**Fix + apples-to-apples re-run:** `ucloud/lepinet-test-allspc.toml` (`--min-img-per-spc 0`,
out-dir `data/ucloud_preds_allspc`), submitted as `lepi-test-all` (job 12359845, B200-MIG). Expected
~0.9148 if the fastai-only port reproduces the pipeline. Result pending below.

**Lesson (also saved to memory):** an eval number that *beats* the reference on a reproduction is a
measurement bug until proven otherwise. Audit the eval-set construction (row count, class count,
filters) before the metric code. Here the metric was perfect and the *filter* was wrong.

### Re-test result (all species, min_img_per_spc=0)
_(pending — `lepi-test-all` running on MIG; fill in species/genus/family macro-F1 over ~12,041
species and compare to 0.9148.)_

### Still open
- The apples-to-apples re-test (`lepi-test-all`, min_img 0) vs 0.9148 — running, result above when in.
- The **"bigger everything" teacher run** → [[2026-07-bigger-everything]] (queued this session).
- The app artifact path (calibration/thresholds, quantization, the versioned bundle) — the whole
  [[2026-07-lepi-app-claude]] Phase B/C on the new package.
- CutMix for multi-target heads (MixUp done; CutMix's `before_batch` also indexes the target tuple).
- Distillation (teacher→small student) and the geographic prior remain future levers.
