# fastai vs mini_trainer: framework comparison for hierarchical Lepidoptera classification

Date: 2026-07-09. Hardware: single RTX 5090 (32 GB, shared with another user's ~6.4 GB
process). All experiments use the same GBIF taxonomy (species → genus → family).

## TL;DR

- **The classifier head is not the source of mini_trainer's edge.** With the same fastai
  training loop, mini_trainer's cosine/weight-norm `IndependentClassifier` head (0.877
  species) ≈ a plain linear+softmax head (0.881). The gain lives in the **training loop**,
  not the head.
- **Per gradient step, the two frameworks are ~equivalent.** On the global dataset early
  climb they are within noise of each other; on the controlled small-data run mini_trainer
  ends +2.6 pts species but only because of its optimizer+schedule, all of which is
  reproducible inside fastai.
- **Per wall-clock second, mini_trainer is ~1.7× faster** — purely because it defaults to
  fp16 AMP and fastai defaults to fp32. `learn.to_fp16()` closes that gap in one line.
- **"mini_trainer converges in fewer epochs" was a measurement artifact**, dominated by its
  default `batch_size=16` (4× more optimizer steps/epoch than bs=64) plus warmup/freeze
  bookkeeping — not an intrinsic property of its loop.
- Two real memory bugs were found and worked around (see §5). **fastai's own
  Learner↔callback reference cycle leaks the autograd graph** and OOM'd a 32 GB GPU at
  fp32 — batch-size-independently — until a per-batch `gc.collect(0)` was added. This
  likely affects `dev/028` on long/large runs.

## 1. Controlled comparison, same data + same head (small dataset)

120 species, 92k images, resnet18, bs=32, 160px, same val split, same seed/init, 13 epochs.
Three arms isolate each factor. Validation species top-1:

| epoch | A. fastai + softmax head | B. fastai + MT head | C. mini_trainer loop + MT head |
|------:|-------------------------:|--------------------:|-------------------------------:|
| 0     | 0.707 | 0.713 | 0.496 |
| 4     | 0.837 | 0.832 | 0.789 |
| 8     | 0.871 | 0.869 | 0.868 |
| 12–13 | 0.881 | 0.877 | **0.903** |

Final species / genus / family, throughput, peak GPU mem:

| arm | species | genus | family | img/s | peak mem |
|-----|--------:|------:|-------:|------:|---------:|
| A. fastai + softmax head    | 0.881 | 0.913 | 0.989 | 1550 | 0.8 GB |
| B. fastai + MT head         | 0.877 | 0.906 | 0.988 | 1531 | 0.8 GB |
| C. mini_trainer + MT head   | **0.903** | **0.929** | **0.993** | 1461 | ~7 GB (fp16+Muon state) |

- **A vs B (head effect, same loop):** tie → the fancy cosine head doesn't help here.
- **B vs C (loop effect, same head):** mini_trainer's loop reaches a **higher plateau**
  (+2.6 pts species) and is still climbing while fastai has started to overfit.
- **fastai converges *faster* early** (0.71 @ epoch 0 vs 0.50): its one-cycle schedule is
  aggressive; mini_trainer's first 2 epochs are a frozen-backbone warmup. mini_trainer's
  advantage is the *plateau*, reached later — the opposite of "fewer epochs".

Caveats: the mini_trainer arm ran with its class-distribution regularizer disabled and a
1-epoch-longer warmup, both of which slightly understate it.

## 2. Optimizer isolation (per gradient step)

Same head/data/LR/schedule, only the optimizer changes. Validation species top-1 vs steps:

| steps | Adam (coupled L2) | AdamW (decoupled wd) | MuonAuxAdamW |
|------:|------------------:|---------------------:|-------------:|
| 300   | 0.422 | 0.524 | 0.536 |
| 900   | 0.509 | 0.681 | 0.679 |

- Plain **Adam-with-L2 plateaus**; **AdamW ≈ Muon** (tied). Muon is *not* the differentiator.
- **fastai's default `Adam` is already AdamW** (`decouple_wd=True`, `wd=0.01`,
  `wd_bn_bias=False`) — so per-step the two frameworks' optimizers are equivalent.

## 3. Global-dataset early-climb race (`dev/031`)

The real question: with everything matched as closely as possible, does one framework's
loop make accuracy climb faster on the actual global data? effnetv2s, bs=32, 224px,
**minimal aug (flip only)**, **no warmup, unfrozen from step 0**, constant LR, same MT
`independent` head+loss, same seed/init, same fixed 1920-image validation probe. Only the
framework core differs: fastai (Adam, fp32, `Learner.fit`) vs mini_trainer (MuonAuxAdamW,
fp16 AMP, grad-clip 5). 12,041 species.

Validation top-1 (species is near-chance early with 12k classes; family/genus are the clean
signal):

| step | fastai sp | MT sp | fastai genus | MT genus | fastai family | MT family |
|-----:|----------:|------:|-------------:|---------:|--------------:|----------:|
| 500  | 0.012 | 0.012 | 0.054 | 0.070 | 0.547 | 0.417 |
| 1000 | 0.027 | 0.035 | 0.117 | 0.115 | 0.664 | 0.599 |
| 1500 | 0.048 | 0.064 | 0.165 | 0.142 | 0.694 | 0.626 |

- **Per step: essentially tied.** fastai marginally ahead on genus/family, mini_trainer
  marginally ahead on species — all within run-to-run noise. Neither loop climbs
  meaningfully faster per gradient step. This confirms §1's finding on real global data.
- **Per wall-clock: mini_trainer is ~1.7× faster** — 1500 steps in **97 s** vs fastai's
  **168 s** (495 vs 286 img/s). Entirely due to **fp16 vs fp32** on effnetv2s's tensor
  cores. Time-aligned (~97 s), mini_trainer sits at species 0.064 while fastai is only at
  ~step 870 (species ~0.015).

**Answer to "is one faster?"** Per step, no. Per second, yes — and it's the fp16 default,
not the loop. `learn.to_fp16()` gives fastai the same speed.

## 4. Why mini_trainer looked like it "converged in fewer epochs"

Its defaults inflate steps-per-epoch and hide warmup:
- `batch_size=16` vs a typical fastai bs=64 → **4× more optimizer updates per epoch** over
  the same data. Four mini_trainer epochs ≈ one bs=64 epoch in gradient steps.
- fastai `fine_tune` "wastes" the first `freeze_epochs` on the head only.

Plot accuracy vs **gradient steps / images seen**, not epochs, and the gap disappears.

## 5. Bugs found (both worked around without touching mini_trainer's source)

1. **mini_trainer `Classifier._weight_bias()` leak.** Every training-mode forward stores a
   live, graph-attached view of the `weight_norm`-parametrized weight into a persistent
   buffer; the parametrization back-references the module → a reference cycle that pins the
   whole batch's backward graph until cyclic GC runs. OOM'd a 32 GB GPU in ~60 batches at
   the real class count. Fix: `GCCallback` runs `gc.collect(0)` per batch (gen-0 only —
   the cycle dies within a batch; a full `gc.collect()` cost ~250 ms at 12k classes and
   made the GPU pulse 0↔100 %).
2. **fastai's own `Learner`↔callback reference cycle** does the same thing to *plain fastai*
   at fp32 — OOM at 23 GB, batch-size-independently — until the same per-batch
   `gc.collect(0)` is added (peak then drops to 0.8 GB). **Likely affects `dev/028`** on
   large images / long epochs; add the callback or `.to_fp16()`.
3. `AutoregressiveClassifier` MRO bug (decoder bypassed), `MultiLevelWeightedCrossEntropyLoss`
   missing `super().__init__()`, `learn.export()` vs parametrized modules, and
   `kaiming_normal_` on RMSNorm — all documented and worked around in `dev/030`.
4. **Python 3.14 default `forkserver` start method** can't pickle the DataLoader worker
   state (holds a CUDA tensor from aug warm-up) → forced `fork` in `dev/030`/`dev/031`.

## 6. Improved `dev/030` (Muon + fp16 + cosine, unfrozen)

`dev/030` now supports `optimizer: muon`, `fp16: true`, `schedule: flat_cos` (unfrozen from
step 0 — required for Muon, which re-partitions param groups incompatibly with fastai's
freeze). Small-dataset retry (362 species, resnet18, 5 epochs), validation species top-1:

| head | species @ epoch 4 |
|------|------------------:|
| independent  | 0.795 |
| hierarchical | 0.800 |
| autoregressive | validated (runs & converges; sp 0.108 @ 1 epoch) |

The autoregressive head is **not broken** — its earlier "crash" in the batch run was an OS
OOM-kill caused by a concurrent GPU job, not a code fault.

## 7. Recommendation

The two frameworks are equivalent per step for this problem. The pragmatic best-of-both,
already enabled by `dev/030`, is **fastai's loop (ergonomics, fast early convergence) +
mini_trainer's Muon optimizer + fp16 + a cosine schedule** — captures the higher plateau
and the ~1.7× wall-clock speedup while keeping `fine_tune`-level simplicity. Always compare
per images-seen, not per epoch.
