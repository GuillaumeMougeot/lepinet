# The teacher → small-model → app bridge: make shipping a model one command

**Status:** OPEN (started 2026-07-25). The long-run shape of lepinet: it produces **two kinds of
model** and the path from one to the other, and from there to a running phone app, should be as
close to one button as possible.

1. **Teachers** — big, accurate models (effnetv2_s 0.9148 baseline; ConvNeXtV2-L / DINOv3 next,
   [[2026-07-bigger-everything]]). Kept for accuracy and as distillation sources.
2. **Students** — small, fast models distilled from a teacher, exported to ONNX and quantized, that
   ship **ready to use**: the `.onnx` plus the sister files that make it work (taxonomy, scientific
   names, calibration, thresholds, a `config.json` descriptor), bundled and uploaded as a **GitHub
   release**. The companion PWA [[2026-07-lepi-app-compression]] consumes exactly this bundle.

The two must be joined by a **smooth bridge**: when a teacher finishes, turning it into a shipped
small model should be pressing a button / calling a script — export → (distill) → quantize →
calibrate → assemble bundle → release.

## What already exists (surveyed 2026-07-25)

- **The compression science is done** (old mini_trainer pipeline, `dev/040`–`044`,
  [[2026-07-lepi-app-compression]]): ONNX export works (parity 2e-5), browser resize is a non-issue,
  marginalization (species→genus→family) beats separate heads, **int8 is −0.59 pp for 3.9×**,
  temperature/threshold calibration fitted. The app v1 shipped (effnetv2b2, GitHub Pages) and a
  GitHub **release** format exists (`model-v1-effnetv2b2`: model.onnx + qdq + taxonomy + calibration
  + thresholds + manifest).
- **The app is already config-driven** (`model/config.json` declares model + sidecars + IO names;
  "a new model is a folder swap"). It marginalizes coarse levels from species and greys by threshold.

## The gaps this bridge must close

1. **The export/quantize/calibrate pipeline lives on the *old* mini_trainer stack (`dev/040`–`044`),
   not the clean `lepinet` package.** The package's `export.py` did ONNX + taxonomy only. → **Closed
   the interop half today** (below); quantize/calibrate/names still to port into the package.
2. **Level-name mismatch (the concrete interop bug).** The package names levels by parquet columns
   (`speciesKey`, `parents.speciesKey_to_genusKey`) and emitted no `config.json`; the app hardcodes
   `LEVELS=['species','genus','family']` and reads `taxonomy.vocabs.genus` / `parents.species_to_genus`.
   A package model could not load in the app. → **Closed today.**
3. **QDQ int8 doesn't run in ORT-Web** (documented in [[2026-07-lepi-app-compression]]): static-QDQ
   verifies in Python (no `ConvInteger`) but throws a raw WASM numeric error at session creation, so
   v1 ships **fp32 (54 MB)**. This is the "forced to use fp32" issue. Candidates to try, in order:
   per-tensor QDQ (current is per-channel), fp16-with-fp32-head, a custom minimal ORT build. **Needs
   in-browser validation** (no JS toolchain on the box → owner tests, they offered).
4. **Distillation is not implemented** (C3 deferred). Teacher→student is the model-quality lever for
   the small model; the gap teacher 0.9148 → effnetv2b2 0.8871 is 2.8 pp, part recoverable.
5. **One-command bridge + release** don't exist yet as a package entry point.

## What was built today (interop half of the bridge)

`lepinet.export.export_onnx` now emits an **app-ready bundle** with zero app-code changes:
- `friendly_level_names()` maps `speciesKey→species` (strip `Key`; identity otherwise), used for
  **graph output names** (`logits_species`) and **taxonomy.json keys** (`vocabs.species`,
  `parents.species_to_genus`) — exactly what the app reads.
- Emits **`config.json`** (the app's bundle descriptor: model/sidecars/imageSize/inputName/outputs
  map/gbifBase). Sidecars (`names/calibration/thresholds.json`) are referenced by convention; the
  app degrades gracefully when they're absent.
- Verified on the **milestone 5ep model**: exported a full bundle, **graph parity max |Δlogit|
  2.2e-5, top-1 100%** at all three levels; `config.json` keys and `taxonomy.json` vocab/parent keys
  are byte-identical in shape to the app's shipped bundle. A lepinet model now drops into the app.
- 33 CPU tests green (existing export/e2e unaffected — output names moved to friendly, taxonomy keys
  moved to friendly, both self-consistent).
- **`lepinet-app` worktree** `../lepinet-app-wt` on branch `lepinet-package-bundle` created so the
  deployed Pages app (`main`) is never touched while this is developed.

Bundle from the 5ep teacher lives at `data/local_bundles/5ep_effnetv2s/` (fp32, 172 MB — the teacher
is large; this validates the *bridge*, not the shipped student). It is **not** committed (data/ is
gitignored, machine-local).

## Distillation implemented (`lepinet distill`, 2026-07-25)

Built the teacher→student half of the bridge (planned `dev/045`, never written). Design:

- **`DistillLoss`** (`loss.py`): `total = (1-α)·CE(student, labels) + α·Σ_level T²·KL(softmax(student/T)
  ‖ softmax(teacher/T))`. KD per level (fine→coarse), matching the N-level head. Honors the framing
  from [[2026-07-lepi-app]]: distillation is the student's *training method*, not a post-hoc
  compressor — the teacher's soft posterior over 12 k species carries the hierarchy + tail structure
  the hard labels can't. `T²` keeps the KD gradient scale comparable to CE.
- **`DistillCallback`** (`callbacks.py`): runs the **frozen teacher in fp32, no-grad**, on each
  *training* batch's input and feeds its per-level logits to the loss (cleared on validation → val
  loss/metrics stay KD-free). The teacher never touches the student's graph.
- **Label-free of the teacher's head geometry, strict on its vocab.** Teacher and student may differ
  in backbone/bottleneck, but KD aligns logits *by index*, so `train()` asserts **identical class
  vocab + level order** (raises with a clear message otherwise — the usual cause is a different
  `min_img_per_spc`/fold/family_filter). This is the one correctness trap and it's guarded.
- **Config**: `distill_teacher` (path/glob), `distill_alpha` (0=hard only, 1=teacher only, default
  0.5), `distill_temperature` (default 4). Guarded incompatible with mixup/cutmix (both rewrite the
  batch/loss path). **CLI**: `lepinet distill -c <student.yaml> [--teacher ... --alpha ... --temperature ...]`.
- **Validated GPU-free**: 5 unit tests (KD=0 when teacher==student; α=0 ⇒ pure CE; KD differentiable;
  mixup guard) + a synthetic **e2e distill** (train teacher → distil student → checkpoint). 40 tests
  green, ruff clean. Unlabelled-data soft targets (train the student on more images than are
  labelled) remain a future lever, per [[2026-07-lepi-app]].

**First runs (mock teacher = the 5ep effnetv2_s milestone):** distil into a small
`tf_efficientnetv2_b0` + 256-bottleneck (the shippable size), vs a from-scratch b0 **control** — the
test is "does the distilled student beat its from-scratch equivalent?" Configs
`20260725_ucloud_distill_effnetv2b0.yaml` / `..._b0_fromscratch.yaml`; both min_img_per_spc=50 to
match the teacher's vocab. Both ran on the B200 alongside the ConvNeXtV2-L teacher.

### Result — default KD (α=0.5, T=4) HURT. Negative result, with a mechanism. (2026-07-25)

Full fold-0 test, all 12,041 species / 629,742 images (`min_img_per_spc=0`, native macro-F1):

| model (b0, hidden 256, 5 ep) | species macro-F1 | species micro-acc |
|---|---|---|
| **from-scratch (control)** | **0.8692** | 0.9025 |
| distilled (α=0.5, **T=4**) | 0.8546 | 0.8957 |

**Distillation lost −1.46 pp** vs from-scratch — the opposite of "student beats its from-scratch
equivalent." The wiring is correct (KD engaged, vocab aligned, no NaN); the **hyperparameters are
wrong for this head**, and the reason is specific and worth recording:

- The teacher is a **cosine z-score head** — its logits are `cosine_to_zscore(cos θ)`, i.e. already
  ~unit-scale (roughly standard-normal), *not* the large-range logits Hinton-style KD assumes. And
  [[2026-07-lepi-app-compression]] §5 measured the model **under-confident** (calibration T≈0.8).
- Dividing already-flat, under-confident logits by **T=4** pushes the 12,041-class softmax target
  toward **uniform** — the KD target carries almost no class information beyond noise. At α=0.5 that
  diffuse signal replaces half the (working) hard-label gradient → the student ends up *worse*.
- **Standard KD temperature (3–6) is miscalibrated for a z-score cosine head.** Hypothesis: the
  right T here is ≈1 (use the teacher's posterior near as-is), possibly <1 (sharpen it).

**Next experiment (launched):** `20260725_ucloud_distill_effnetv2b0_T1.yaml` — same run with
**T=1.0** (α=0.5). Falsifiable: if T=1 distilled ≥ 0.8692 (beats from-scratch), temperature was the
whole story; if still below, distillation from this teacher/at this size needs α tuning or the gap
teacher→student (0.911→~0.87) is just too small for KD to help and it waits for the stronger
ConvNeXtV2-L/DINOv3 teacher. Either way it's a data point, not a wall.

### T=1 result — CONFIRMED. Distillation works; temperature was the whole story. (2026-07-26)

| b0 student (hidden 256, 5 ep) | species macro-F1 | Δ vs from-scratch |
|---|---|---|
| from-scratch (control) | 0.8692 | — |
| distilled, T=4 | 0.8546 | **−1.46 pp** |
| distilled, **T=1** | **0.8786** | **+0.94 pp** |

**T=1 distilled beats from-scratch by +0.94 pp** (and beats T=4 by +2.40 pp) — "student beats its
from-scratch equivalent," achieved. The diagnosis held exactly: the KD temperature, not the method,
was the problem, and the fix is the single change T=4→T=1 for the cosine z-score head. **Locked in as
the package default** (`distill_temperature: 1.0`, `DistillLoss` default). Genus/family marginal
levels also improved (0.939 / 0.961). Distillation is now a working lever; the next gains are a
**stronger teacher** (ConvNeXtV2-L/DINOv3, so the 0.911→0.879 headroom widens) and α/level tuning.
This is the KD path the shipped small model will use.

**Lesson (also a memory):** KD temperature is not head-agnostic. For a metric-learning / cosine
z-score head whose logits are already ~unit-scale (and under-confident), start at **T≈1**, not the
textbook 3–4. When the real teacher lands, swap `distill_teacher` — but carry T≈1 forward.

## Plan (next, in order)

1. **Port quantize + calibrate + names into the package** (`lepinet quantize` / a `lepinet bundle`
   one-command that runs export→quantize→calibrate→assemble). From `dev/043` (static-QDQ),
   `dev/044` (temperature + thresholds), `dev/047` (names.json from the parquet scientificName).
2. **Solve the ORT-Web small-format problem** — try per-tensor QDQ, then fp16. Owner validates each
   candidate in a real browser (the only reliable test). Until then fp32 stays the shippable format.
3. **Distillation** (`lepinet distill`, teacher→student): soft-target KD on the species logits, the
   student a small conv net (effnetv2b0 / a DINOv3-distilled ConvNeXt). Then the student goes through
   the same bundle path. This is where the small *and* accurate model comes from.
4. **One-command release**: `lepinet release` (or a CI action on a tag) uploads the bundle to a
   GitHub release; decide whether the release lives on the `lepinet` repo (model) or `lepinet-app`.
   Then generalise the app loader to pull-from-release + fully generic level names.

## Two backbone/head questions (owner, 2026-07-25) — answered

**Would DINOv3 / ConvNeXt benefit from ArcFace?** Likely yes, modestly, and it's **orthogonal to the
backbone** — ArcFace lives in the head+loss, so any backbone's embedding feeds it. The gains
(tighter tail/open-set separation, better-calibrated angular confidence for the app's abstain
threshold) apply regardless of backbone. Two caveats: (a) the *marginal* benefit shrinks on a very
strong backbone like DINOv3 whose features are already well-clustered — the margin has less to fix;
(b) ArcFace is **incompatible with MixUp** (the ConvNeXtV2-L teacher run uses MixUp), so they compete
as regularisers — pick one per run. Clean test: fix the best backbone, compare cosine vs ArcFace head
(species-only margin) with everything else held.

**Try DINOv3-distilled ConvNeXts before DINOv3 ViT?** **Yes.** `convnext_*.dinov3_lvd1689m` are
ConvNeXts distilled from the DINOv3 ViT-7B — they carry most of DINOv3's representation quality but
(1) **emit 4-D maps → drop into the existing `PooledHead` path with zero new plumbing** (no
ViTBody/FlatHead), (2) give **clean ONNX + fast WASM/WebGPU, no attention in the hot path** — exactly
what the browser target wants ([[2026-07-lepi-app-compression]] §C2), (3) **scale to high resolution
cheaply** (no quadratic tokens), and (4) as a **conv teacher align with the conv student family**,
which the whole export/quantize pipeline is proven on. `convnext_large.dinov3` (~200 M) also compares
apples-to-apples with the ConvNeXtV2-L run already training. Reserve the ViT path for a focused "can a
ViT teacher beat the best conv teacher" experiment, since it costs more to integrate and deploy.

## Production progress (2026-07-29)

- **`lepinet bundle` shipped** (Phase 3): one command, checkpoint → deployable folder — `export`
  (fp32 `model.onnx` + `taxonomy.json` + `config.json` + `MANIFEST.json`) + dynamic-int8
  `model.int8.onnx`. Verified on the distilled-b0: fp32 parity 2.6e-5, int8 **41.7→11.0 MB (3.79×)**.
  int8 is native/CPU-ready but still **not ORT-Web-runnable** (browser stays fp32 — the QDQ blocker).
- **App plug-and-play — browser test live.** A lepinet bundle drops into the PWA (its `config.json`
  + friendly taxonomy already match the app contract; species vocab order is *identical* to the
  shipped model, so `names.json` reuses verbatim). Wired a **query-param model switch** in the app
  (`?model=<slug>` → `./model-<slug>/`) so the **default app is byte-identical/untouched** and a
  candidate bundle previews on the live deploy: **`/?model=lepinet-test`** serves the distilled-b0
  fp32 bundle. Validate there before promoting it to the default `model/`. (Kept a strict slug regex;
  bumped the SW cache.) Next: wire `names/calibration` into `bundle`, then the ORT-Web small-format
  fix (per-tensor QDQ / fp16) — the only step needing an in-browser test.
- **OOD benchmark prepared** (the new direction): `dev/052` max-logit AUROC(known vs novel) on
  **global_lepi's <50-image species** (removed from training, still on UCloud → clean novelty, no
  domain shift) for arcface vs independent — tomls ready (`ucloud/lepinet-ood-*.toml`, on a `mig.2g`
  slice per the eval-slowness lesson). Submission pending a **sustained UCloud 502 window** blocking
  all job submits; the 12-epoch DINOv3-cnx run is likewise queued behind it (background retry loop).

## Plug-and-play validated + the ORT-Web small-format reality (2026-07-29)

- **Plug-and-play CONFIRMED (owner tested the browser link).** A `lepinet bundle` (distilled-b0,
  fp32) loads and predicts in the PWA via `?model=lepinet-test`, default app untouched. **This is the
  milestone: a lepinet-produced release drops into the app with no app-code change.** fp32 b0 is
  **41.7 MB** — already shippable (app v1 shipped 54 MB); the ≤8 MB dream is optimisation, not a
  blocker to shipping a real lepinet model.
- **ORT-Web small-format is the remaining, genuinely-iterative blocker.** Tried post-hoc fp16
  (`onnxconverter_common.float16`): converting the graph yields an **invalid Cast-type graph** (ORT
  refuses to load — `Cast_1 output float16 vs expected float`), because the legacy-TorchScript
  exporter emits explicit `Cast` nodes the fp16 pass can't rewrite consistently; blocking `Cast`
  makes it valid but converts *nothing* (all initializers stay fp32). So **post-hoc fp16 is a dead
  end for this graph.** The clean fix is a **source-level fp16 export** — trace an fp16 backbone with
  the cosine head kept fp32 (the `PooledHead` fp32 guard already exists) — then browser-test. Plus
  the still-open per-tensor-QDQ path. Both need in-browser validation (owner), which is why they
  stay after plug-and-play, not before it.
- `export.to_fp16_onnx` is kept as a utility (works for cleanly-convertible graphs) but is **not**
  usable on our TorchScript export — noted here so it isn't mistaken for the fix.

## Why checkpoints are fp32 even though training used fp16/bf16 (2026-07-29)

Owner's question — and the answer is not a bug, it's how mixed precision works:

**Mixed-precision training keeps fp32 *master weights*.** `to_fp16()`/`to_bf16()` (torch AMP) do not
store the model in half precision. They cast **activations** to half for the matmuls/convs (that's
where the speed and memory win is) while the **parameters and the optimizer state stay fp32**,
because weight *updates* are tiny relative to weight magnitudes: adding a ~1e-7 update to a ~1e-1
weight in fp16 (≈1e-3 relative resolution) rounds to *no change*, and training stalls. So the
checkpoint we save is fp32 by construction — precision was a *compute* choice, never a storage one.

Consequences: (1) an fp32 export is the honest default (it matches the trained master weights);
(2) shrinking for deployment is therefore a **separate post-training step** (quantize / convert), and
(3) that conversion is where it got hard — see the fp16 finding above: post-hoc fp16 conversion
breaks on the TorchScript-exported graph's explicit `Cast` nodes, so a **source-level** fp16 export
(trace with a half-precision backbone, cosine head kept fp32) is the correct route.

## Evals were slow *and* failing — both root-caused and fixed (2026-07-29)

- **Failing:** `FileNotFoundError` deep inside the DataLoader. The parquet is a *catalogue*; the
  image mirror is **incomplete** (a handful of the 630 k global_lepi files are absent), so a
  multi-hour eval died at the first gap and lost the whole run. Fix: `evaluate(skip_missing=True)`
  pre-filters absent files and **reports the count** (so a *large* number of misses stays visible
  rather than silently shrinking the test set).
- **Slow:** every eval ran on `gpu-nvidia-b200-1-mig.1g` — the **smallest** MIG slice (6 vCPU) — so a
  198 M model streamed 630 k images on a fraction of a GPU, ×4 with TTA. Fix: all eval/OOD jobs moved
  to a **full `b200-1-gpu` with `--num-workers 32`**. (Two evals already running on the old code were
  terminated and resubmitted; they were both slow and doomed.)

## Next: models out of the app repo, into releases (owner's design, 2026-07-29)

Shipping model binaries inside `lepinet-app` doesn't scale — every version bloats the repo *and*
`.git` forever (git keeps all history; the current test bundle alone is 41.7 MB). Agreed plan:

1. **Publish models as GitHub releases on the `lepinet` repo** (`lepinet bundle` output: model.onnx +
   taxonomy + config + names/calibration/thresholds), one release per model version.
2. **The app fetches models at runtime**: a small `models.json` manifest (hand-updated, or pulled
   from the releases API) lists available models; the user picks one, the app downloads and caches it
   (the service worker already caches the model on first fetch, not at install). First run uses the
   **default latest**.
3. **Purge the binaries from the app repo *and* its history** — `git filter-repo` (or BFG) to drop
   `model/`, `model-*/` and `ort/` blobs, then force-push; otherwise the `.git` keeps them forever.
   Requires care (rewrites history) and a fresh clone for anyone who had it — do it once, deliberately.
   The vendored `ort/` runtime (59 MB) is the other big offender and should be pinned from a CDN or
   fetched at install instead.

Note this makes the *app* repo tiny and turns model releases into the distribution channel — which is
exactly the "plug-and-play releases" goal.

## The real reason evals were ~1 img/s: a fastai attribute trap (2026-07-29)

Owner stopped every eval: **~1 img/s** — 630 k images would have taken **>7 days**, and the GPU was
idle. It was not the GPU, not the MIG size, and not the model. It was one line in `predict_df`:

```python
test_dl = dls.test_dl(test_df, num_workers=getattr(dls.train, "num_workers", 0))   # BUG
```

**fastai hardcodes `self.num_workers = 1`** in `DataLoader.__init__`
(`self.rng, self.num_workers, self.offs = random.Random(...), 1, 0`) — the public attribute is a
*dummy*; the effective worker count lives on the internal `fake_l`. So that `getattr` **always
returned 1**, and every evaluation ran **single-process** no matter what `--num-workers` said. On a
~90 ms/file network mount one worker is ~11 img/s ceiling, and with large images ~1 img/s. Verified
locally: requested 4 → `dls.train.num_workers == 1`, while the real value is 4.

**Fix:** `dl_num_workers(dl)` reads through to `fake_l.num_workers`, and `evaluate` now passes its
own `--num-workers` straight into `predict_df` rather than round-tripping it through fastai's
attribute. The loader now prints `num_workers=N, batches=M` at startup so a regression is visible in
the first log line. Locked in with a **regression test** (`dl_num_workers` must return 12, not the
dummy 1).

**Second, self-inflicted, slowness — fixed in the same pass.** The `skip_missing` guard I added
earlier did `(root / p).is_file()` *per row*: one network round-trip per image, ≈15 h for 630 k
files *before inference started*. Replaced with `_paths_exist`, which lists each **parent directory
once** (one round-trip returns every name) across a 32-thread pool — ~12 k directories in well under
a minute.

**Lesson (worth carrying):** on a high-latency network mount, *per-file* operations are the enemy —
batch by directory, and never trust a framework's public attribute to report what you configured.
Both the MIG sizing and the "GPU is slow" theory were red herrings; the profile was one worker.

### Measured after the fix: 898 img/s (was ~1) — 2026-07-29

Throughput probe (`ucloud/lepinet-probe-throughput.toml`, 20 k random images, effnetv2_s, full B200,
`--num-workers 32`):

```
Inference dataloader: num_workers=32, batches=313
Inference: 20000 images in 22.3s = 898.1 img/s
```

**~900× faster than the broken single-worker path**, and it matches the ~950 img/s measured for
*training* on the same mount — i.e. evaluation is now at the mount's practical ceiling, exactly where
it should be. Consequences:

- A full 630 k-image fold-0 eval is **~12 minutes**, not >7 days. Every eval in this project was
  previously bottlenecked by one line.
- **The MIG sizing was a red herring** (owner said so): the mount/CPU decode is the limit, not the
  GPU. With workers actually working, a small slice is fine for small models; what matters is the
  **vCPU count backing the workers**. Size eval jobs by *workers needed*, not by GPU fraction.
- `--limit N` + the printed `img/s` line make this a 2-minute check before any long run. Standing
  rule: **probe throughput before launching a multi-hour eval.**

## Source-level fp16 works — the small-format path is open (2026-07-29)

Post-hoc conversion was the dead end, not fp16 itself. `Fp16ExportWrapper` traces a genuinely
half-precision module instead of rewriting a finished graph:

- **fp16 backbone** (the bulk of the weights) + **fp32 cosine head** — the head L2-normalizes and
  takes `acos` near ±1, exactly where fp16 loses the precision that matters (same reason `PooledHead`
  forces fp32 under autocast). Graph I/O stays fp32 so the app feeds/reads it unchanged.
- Measured on the distilled b0: **41.7 MB → 30.1 MB**, **100 % top-1 agreement** with fp32 at all
  three levels, `max|Δlogit| = 0.009`, and **no `ConvInteger`/`MatMulInteger`** — the ops ORT-Web has
  no kernels for, which is what killed int8 in the browser.
- Wired into `lepinet bundle` (`fp16=True` by default), so one command now emits fp32 + fp16 + int8.

**Browser validation deployed** (default app untouched, query-param switch):
`?model=lepinet-test` (fp32, 41.7 MB — already confirmed working), `?model=lepinet-fp16` (30.1 MB),
`?model=lepinet-int8` (11.0 MB). The int8 variant is included **expecting it to fail** — it is the
in-browser confirmation of the ConvInteger hypothesis rather than a candidate. If fp16 loads, the
shipped model drops ~28 % with zero accuracy change and the ORT-Web blocker is resolved for practical
purposes; the ≤8 MB target then needs a smaller *student*, not a smaller *format*.

## int8 in the browser: CLOSED as not-worth-it; fp16 is the shipping format (2026-07-29)

Owner's in-browser test of the deployed candidates:

- **fp16 (30.1 MB): WORKS.** Loads and predicts. → **this is the shipping format.**
- **int8 (11.0 MB): FAILS**, with exactly the predicted error:
  `Could not find an implementation for ConvInteger(10) node '/model/model.0/conv_stem/Conv_quant'`.

**Is there a fix?** In principle yes — *static QDQ* quantization emits
`QuantizeLinear`/`DequantizeLinear` around ordinary `Conv`/`Gemm` instead of `ConvInteger`, which
ORT-Web does implement. But we **already tried that** ([[2026-07-lepi-app-compression]]): the static
QDQ graph verified in Python and then died in-browser at session creation with a raw numeric WASM
error. So both int8 encodings have now failed in a real browser for *different* reasons, and the
remaining ideas (per-tensor instead of per-channel QDQ, a custom minimal ORT build) are speculative
and slow to iterate — each costs a build + a device test.

**Decision (owner concurs): stop chasing int8.** fp16 gives 28 % off with *zero* accuracy change and
is confirmed working. The path to a genuinely small artifact is therefore a **smaller/better student**
(the model), not a smaller *format*: b0 caps ~0.88, while `fastvit_sa12` (10.6 M) reached 0.892 and
`effnetv2b2` (8.7 M) 0.887 in the compression sweep. A fp16 student at that size lands near ~15–20 MB
with *better* accuracy than today's b0 — strictly better than a broken 11 MB int8.

## ArcFace next steps: tune it, and the z-score adaptation (owner's idea, 2026-07-29)

The 0.732-vs-0.601 OOD result came from **untuned** ArcFace (`s=30, m=0.3`, first guess). Two levers:

1. **Hyperparameter tuning.** `m` controls how hard the margin pushes (too large hurts the tail, which
   we already emphasise via oversampling); `s` sets the logit scale and interacts with label
   smoothing. A small sweep (`m ∈ {0.1, 0.2, 0.3, 0.5}`, `s ∈ {16, 30, 64}`) on the *fixed* backbone,
   scored on **both** in-distribution F1 and OOD AUROC, should find a better point on that trade-off —
   0.732 is very unlikely to be the ceiling.
2. **ArcFace × `cosine_to_zscore` (owner's proposal — mathematically grounded, and I agree).** The two
   are complementary and currently *mutually exclusive* in our code: the independent head maps
   `cos θ → z-score` (`√(d−2)·(acos(−cos) − π/2)`, which stretches the concentrated cosine
   distribution of high-dimensional unit vectors into ~N(0,1) so the logits behave like ordinary
   pre-softmax scores), while `ArcFaceHead` drops that and returns raw `s·cos θ`. Combining them —
   apply the additive angular margin **then** the z-score transform, i.e. `z(cos(θ+m))` — keeps the
   margin's open-set geometry *and* restores the calibrated, dimension-aware logit scale. That should
   help exactly where the current arcface is weakest: comparable thresholds across heads (§ the OOD
   note that only AUROC, not raw scores, is comparable today) and better-behaved softmax/CE. It also
   makes `s` partly redundant, since `√(d−2)` already sets a principled scale.
   **Implementation is small** (an `zscore=True` flag on `ArcFaceHead` + the margin applied before the
   transform in the loss), and it is the more interesting of the two levers scientifically.

## Model distribution: models moved out of the app repo (2026-07-29)

The bridge's last mile. Shipping model binaries inside `lepinet-app` does not scale — git keeps
every version forever, so each release permanently grows the checkout *and* `.git`.

**The finding that shaped the design: GitHub *release* assets cannot serve a browser.** They redirect
to `release-assets.githubusercontent.com`, which returns **no `Access-Control-Allow-Origin`**, so
`fetch()` from the app's origin is blocked by CORS. `curl` succeeds (it ignores CORS) — exactly the
trap that would have looked fine until someone opened the page. Verified alternatives, with an
explicit `Origin` header: **GitHub Pages** (`ACAO: *`) and **Hugging Face Hub** (echoes the origin).

**Chosen: Hugging Face Hub** (`gmougeot/lepinet-models`) — purpose-built for models: versioned,
CDN-backed, free, CORS-correct. Both bundles live there (`b0-fp16/`, `effnetv2b2-v1/`), and the
GitHub release (`model-v2-b0-fp16`) is kept as the human/script-facing **archive**.

**App side.** `models.json` maps id → base folder URL (relative *or* absolute). Resolution is
`?model=<id>` → remembered choice → manifest default, so a first-time user still chooses nothing.
The service worker previously **skipped all cross-origin requests**, which would have re-downloaded
30 MB on every load and killed offline support — it now caches an explicit model-host allowlist
(`*.hf.co`) and accepts `cors` responses, not just `basic`. `models.json` is network-first (the model
*list* must be updatable without a cache-version bump); everything else stays cache-first.
`model/` + the preview folders are deleted and `model*/`/`*.onnx` gitignored, so no model can be
committed there again. **`ort/` (59 MB) is deliberately kept**: it is the runtime the app executes,
it does not grow per model, and moving it to a CDN is a separate change needing its own browser test.

**Three bugs found by the owner testing it, all mine, all instructive:**
1. `(_slug(_param) && (byId(_param) || null)) ?? remembered ?? default` — **`??` only falls through
   on null/undefined, not `false`**, so with no `?model=` param the chain pinned to `false` and the
   app always reverted to the bundled model.
2. `location.search = ''` is a **no-op when there is no query string** — the choice was stored and
   then never loaded.
3. `models.json` was served **cache-first** by the SW, so an updated model list could never reach a
   user (they saw three stale entries).

The durable fix was not a third patch but removing the ambiguity: **the selection is carried in the
URL** (`?model=<id>`), the highest-priority source, so what loads always matches the address bar and
is shareable/debuggable. Lesson: when state can come from three places (URL, localStorage, default),
make one of them authoritative and *visible* rather than reconciling them.

**Remaining:** purge the old model blobs from the app repo's **history** (`git filter-repo` +
force-push — rewrites history, needs a deliberate go-ahead); optionally move `ort/` to a CDN.
