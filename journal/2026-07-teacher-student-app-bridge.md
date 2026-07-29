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
