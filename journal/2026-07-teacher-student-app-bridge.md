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
