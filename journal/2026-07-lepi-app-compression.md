# Can the 173 MB model be exported, quantized and calibrated for a browser?

**Status:** Phases A + B RESOLVED (2026-07-20); C1 RESOLVED (hidden=256); **C2 RESOLVED**
(2026-07-23): backbone = **effnetv2b2** (test species macro-F1 **0.8871**, transformers dropped
as not-clearly-better). C3 distillation deferred (not needed for v1). **C4 bundle built**
(effnetv2b2 int8, 14.3 MB). **Phase D app shipped v1** to github.com/GuillaumeMougeot/lepinet-app
(Pages). Remaining: in-browser device test (owner), calibration sidecars, optional C3.

> **Correction to an earlier caveat.** The Phase-B numbers were first measured with
> `min_img_per_spc=50`, my own default — not `dev/032`'s test default of 0. That silently cut
> the test fold from 632,913 images / 12,632 species to 484,299 / 3,696, keeping only
> well-represented species and inflating every absolute figure. The mirror is *complete*
> (100% of test-fold images present); the subsetting was the bug, now fixed in `dev/041`–`044`
> (default 0). Relative B-phase conclusions stand (same-image comparisons); the absolute B-phase
> figures below are on the easy subset and are being re-measured on the full fold. C-phase uses
> `dev/032` directly, so its numbers are already on the full fold and comparable to 0.9148.

Plan and decisions: [`2026-07-lepi-app-claude.md`](2026-07-lepi-app-claude.md).
Proposal + review: [`2026-07-lepi-app.md`](2026-07-lepi-app.md).
Scripts: `dev/040`–`dev/044`.

> **Read the caveat in "What these numbers are not" before quoting any figure here.** Every
> number below is measured on the *local image mirror*, which holds 3,696 of the model's 12,041
> species. It is not the held-out global test set that produced the project's 0.9148 headline,
> and it is systematically easier.

## The question

The proposal wants a ≤8 MB offline phone app out of a 173 MB fastai/mini_trainer checkpoint.
Before any compression research, four things could have invalidated the whole plan and all four
were unknowns: whether the cosine head exports to ONNX at all, whether browser-side image
resizing silently costs accuracy, whether the genus/family heads can be deleted, and what int8
actually costs. This entry answers all four.

## Starting point, measured

`20260716-154156-heads-global-independent-muon-5ep-oversample-effnetv2s`:

| | params | fp32 |
|---|---|---|
| effnetv2_s backbone | 20.18 M | 80.7 MB |
| head (`hidden` 1280→1280 + 3 cosine layers) | **22.76 M** | 91.0 MB |
| **total** | **42.94 M** | **171.8 MB** |

**The head is 53% of the model** — slightly larger than the entire backbone. Any plan that
compresses only one half is capped at 2×.

One discovery worth its own line: the head *already contains* a `hidden` layer, a 1280→1280
`nn.Linear` + LeakyReLU, from `mini_trainer`'s `Classifier(hidden=...)`. The "add a bottleneck"
idea from the plan is therefore not new architecture at all — it is **changing an existing
argument from 1280 to 256**. That is the cheapest possible form of the biggest planned win.

## 1. ONNX export works (`dev/040`)

Exports clean, single file, 172.9 MB, opset 18. Graph parity with PyTorch: **max |Δlogit| 2.2e-5,
top-1 agreement 100%** at all three levels.

Three things had to be got right, none of them obvious in advance:

- **The legacy TorchScript exporter cannot be used.** `mini_trainer`'s `Classifier` implements
  `get_extra_state()` returning a metadata *dict*, and `torch.jit._unique_state_dict` calls
  `.detach()` on every state entry → `AttributeError: 'dict' object has no attribute 'detach'`.
  The `dynamo=True` exporter does not walk the state dict this way and works.
- **The head's lazy caches must be warmed in eager mode before tracing.**
  `HierarchicalClassifier.masks` builds itself on first access via `int(mask.max().item() + 1)`,
  a data-dependent shape `torch.export` cannot specialize
  (`GuardOnDataDependentSymNode: could not extract specialized integer from u0 + 1`). One eager
  forward pass populates the caches; the trace then sees plain buffer reads. Loading the
  state_dict is what dirties them, so the head's own constructor warm-up is not enough.
- **Size must be measured over the sidecar.** The dynamo exporter defaults to external data, and
  `stat().st_size` on the resulting `.onnx` reports **1.6 MB** for a 173 MB model. In a project
  whose entire metric is file size, that number is actively dangerous. `dev/040` now embeds
  weights by default and always sums `.onnx` + any `.onnx.data`.

Also emitted: `taxonomy.json` (270 KB) — vocabs in head-index order plus the parent arrays,
derived once so the app and `dev/042` cannot disagree about the hierarchy.

## 2. Browser preprocessing is a non-issue (`dev/041`)

**This walks back a claim from the review.** I flagged PyTorch-vs-browser resize mismatch as
"the classic silent accuracy loss" and a Phase-A blocker. Measured on 2,000 test images against
the exact fastai validation pipeline:

| candidate | agree species | species acc | Δ vs fastai |
|---|---|---|---|
| `squash_256` (ignore aspect ratio) | 0.9565 | 0.9225 | **−1.10 pp** |
| `short_side_crop_256` (one resample) | 0.9870 | 0.9325 | −0.10 pp |
| `two_step_460_256` (mirrors fastai exactly) | 0.9855 | 0.9340 | +0.05 pp |
| `two_step_lanczos` (different kernel) | 0.9835 | 0.9310 | −0.25 pp |

**Conclusion: the resampling kernel does not matter; only aspect ratio does.** Bilinear vs
Lanczos is worth 0.25 pp, i.e. nothing. The single choice that costs anything real is squashing
to a square, at −1.1 pp — and even that is survivable, which is not what I predicted.

**Recommendation for `lepinet-app`: shorter-side resize to 256 + center crop.** One resample
step, trivial on a canvas, and −0.1 pp. The two-step 460→256 dance that fastai does buys
nothing and should not be reimplemented.

`dev/041 --emit-fixture` writes real JPEGs plus the logits they must reproduce, so the app repo
can assert this property in an actual browser — the half of the question Python cannot answer,
since canvas `drawImage` is implementation-defined and differs across Chrome/Firefox/Safari.

## 3. Marginalization is proven (`dev/042`)

The claim needed proving before deleting 5 M parameters. 20,000 test images, genus and family
derived from the species posterior by log-space `scatter_logsumexp` instead of read from their
own heads:

| level | | acc | macro-F1 |
|---|---|---|---|
| genus | direct | 0.9700 | 0.9320 |
| genus | **marginal** | **0.9732** | **0.9394** |
| genus | delta | **+0.32 pp** | **+0.74 pp** |
| family | direct | 0.9929 | 0.9610 |
| family | **marginal** | **0.9949** | **0.9920** |
| family | delta | **+0.20 pp** | **+3.10 pp** |

**Marginal wins at every level on both metrics.** The family macro-F1 gain is large (+3.1 pp)
because rare families are exactly where a separately-trained head has least data and the summed
species evidence has most.

And the consistency problem is real, not hypothetical: **1.81% of images** have a genus-head
argmax that contradicts the species argmax's true parent — roughly 1 in 55 predictions would
show the user a visibly incoherent three-level triple. Marginalizing makes that structurally
impossible.

**Verdict: delete the genus and family heads.** More accurate, fully consistent, ~5 M parameters
lighter. This was the one "free money" claim in the review and it survived.

## 4. int8 quantization is cheap (`dev/043`)

Dynamic int8 (weights only), **172.9 MB → 44.4 MB, 3.89×**, on 3,000 test images:

| metric | fp32 | int8 | Δ |
|---|---|---|---|
| species acc | 0.9363 | 0.9323 | −0.40 pp |
| **species macro-F1** | **0.8879** | **0.8821** | **−0.59 pp** |
| genus marginal F1 | 0.9519 | 0.9536 | +0.17 pp |
| family marginal F1 | 0.9875 | 0.9844 | −0.32 pp |

Top-1 agreement with fp32: species 0.976, genus (marginal) 0.991, family (marginal) 0.999.

The prediction — that a cosine head quantizes almost for free because unit-norm prototype rows
all share one dynamic range — **held**. 0.59 pp for 3.9× is a good trade at any point on this
project's curve.

One workaround was needed: `graph.value_info` must be stripped before quantizing. ORT's
quantizer round-trips through `save_and_reload_model_with_shape_infer`, which fails with
`Inferred shape and existing shape differ in dimension 0: (1280) vs (12041)` on the dynamo
exporter's shape annotations — even though those same annotations pass `onnx.shape_inference`
standalone in **both** strict and non-strict mode. Dropping derived data loses nothing.

Static (activation) quantization was not tried: it needs a calibration pass and produces a QDQ
graph with patchier browser support, for a win in *latency* rather than size. Revisit only if
latency turns out to bind.

## 5. Confidence now means something (`dev/044`)

**This inverts a review claim too.** I asserted the model is "systematically overconfident".
Fitted per-level temperature on the validation fold (fold `'1'`), 8,000 images:

| level | T | val NLL @T | val NLL @T=1 |
|---|---|---|---|
| species | **0.803** | 0.2482 | 0.2637 |
| genus | 0.803 | 0.1196 | 0.1295 |
| family | 0.694 | 0.0261 | 0.0328 |

**T < 1 at every level: the model is *under*confident, not over.** Plausibly the label
smoothing (`1/n_classes` by default in this pipeline) plus the cosine z-score head. Calibration
barely moves NLL (0.264 → 0.248), so it is close to well-calibrated already.

The consequence for the UI is the interesting part. Thresholds fitted on val, measured on test:

| target precision | species threshold | test precision | test coverage |
|---|---|---|---|
| 0.80 | 0.038 | 0.9386 | **0.9998** |
| 0.90 | 0.038 | 0.9386 | 0.9998 |
| 0.95 | 0.379 | 0.9487 | 0.9839 |
| 0.98 | 0.738 | 0.9784 | 0.9029 |
| 0.99 | 0.926 | 0.9894 | 0.8039 |

Genus and family never bite below a 0.99 target; family is 99.5% correct unconditionally.

So the proposal's **0.5 is not merely arbitrary, it is in the wrong region entirely** — but not
for the reason the review gave. The model is accurate enough that a "greyed below 0.5" rule
would essentially never grey anything, making the whole confidence display decorative. A real
dial exists (99% precision on species costs 20% coverage), and the product question is which
point on it to ship. That is now a choice with numbers attached instead of a guess.

## What these numbers are not

The local image mirror holds **3,696 of 12,041 species** — those with images downloaded here.
Every figure above is on that subset, and it is biased towards well-represented species: species
macro-F1 reads 0.888 on it, against the project's real held-out global figure of **0.9148**
(they are not comparable in either direction — different species set, different sample sizes).

What this means in practice:

- The **relative** results (int8 costs 0.6 pp; marginal beats direct; squashing costs 1.1 pp)
  are trustworthy — every comparison is same-images, same-pipeline, differing in one thing.
- The **absolute** thresholds in `thresholds.json` are **not shippable**. They are fitted where
  the model is unusually accurate and would be too permissive globally. Re-run `dev/044` against
  the full test fold on a box with the complete image set before any artifact bundle is released.

## Phase C1 — the bottleneck sweep (in progress, 2026-07-21)

`hidden` (mini_trainer's `Classifier(hidden=...)`) sets `preclassification_size`, the width of
every cosine prototype row. The project has always run the default (`True` → 1280 = backbone
width), which is *why* the head is 53% of the model. Sweeping it against the exact 0.9148 recipe
(`configs/20260720_bottleneck_{512,256,128}.yaml`), one thing changed:

| hidden | head params | test species macro-F1 | Δ vs 0.9148 | marginal step cost |
|---|---|---|---|---|
| 1280 (baseline) | 22.76 M | 0.9148 | — | — |
| 512 | 9.13 M | 0.9058 | −0.90 pp | −0.90 (÷2.5 head) |
| **256** | **4.58 M** | **0.9002** | **−1.46 pp** | −0.56 (÷2 head) |
| 128 | 2.31 M | 0.8843 | −3.05 pp | **−1.59 (÷2 head)** |

**RESOLVED — chosen bottleneck is 256.** The curve has a clear knee: halving the head from
512→256 costs only 0.56 pp, but the next halving 256→128 costs 1.59 pp — 128 falls off a cliff.
256 gives a **5× smaller head for −1.46 pp**, well clear of the 0.87 floor. (A miss against the
plan's "<1.0 pp" prediction, §8 — recorded.)

**Why 256 over 512, given 512 is only −0.90 pp?** Because the *shipped* head, after B1's
marginalization drops the genus and family classifiers, is just the bottleneck layer plus the
species classifier:

| hidden | shipped head (bottleneck + species) | int8 |
|---|---|---|
| 256 | 1280·256 + 256·12041 = 3.4 M | ~3.4 MB |
| 512 | 1280·512 + 512·12041 = 6.8 M | ~6.8 MB |

Against a ≤8 MB total with a ~5 MB backbone budget, the head is the binding constraint: 512's
0.56 pp edge is not worth doubling the head to 6.8 MB (which would leave ~1 MB for the backbone).
256 also leaves ~3 pp of headroom above the floor for the C2 backbone swap to spend, with C3
distillation to recover some of it. **C2 (`configs/20260721_backbone_*.yaml`) runs at hidden=256.**

A ledger bug surfaced here and was fixed (`dev/036`): test configs written before their training
run exists name the checkpoint with a **glob** (timestamp unknown at write time), and the
name↔checkpoint join matched the run-dir name as a literal substring of that glob — which a `*`
never contains, so every globbed run showed a blank test F1. `load_evals` now resolves the glob
before matching. All three C1 test scores populate `RESULTS.md` after the fix.

Three things had to be fixed to get here, all worth the record:

- **`hidden` had to be threaded through the whole chain** — `dev/030` train + checkpoint,
  `dev/032`/`dev/040` reconstruction — defaulting to `True` so every pre-existing config and
  checkpoint reproduces byte-identically.
- **A latent upstream bug surfaced the instant `hidden` became an int.**
  `Classifier._reshape_backbone_embeddings` validates the *backbone* output width against
  `preclassification_size`, which is the width *after* the hidden layer — equal only when
  `hidden` is a bool. With `hidden=256` a backbone emitting 512 (resnet18 smoke test) trips
  `Unexpected 2D preclassification input shape: (64, 512)`. Patched per-instance in `dev/030`
  (`_fix_backbone_embedding_check`), the same way this module already works around the
  `AutoregressiveClassifier` MRO bug, rather than editing mini_trainer. **The 1-epoch resnet18
  smoke test on `data/small` caught this before it could waste an 8 h global run** — cheap
  insurance that paid out.
- **Detach long sweeps from the session.** The first launch went through the harness background
  runner and was reaped after ~8 h, mid-`hidden=512` (256 had completed and was safe). Per-epoch
  `SaveModelCallback` checkpoints meant nothing was truly lost, but the clean fix is
  `setsid nohup … </dev/null &` with absolute paths in `dev/run_c1_bottleneck_sweep.sh`, so the
  training outlives the conversation that started it.

## Phase C2 — backbone sweep, prepped (2026-07-21)

With the head shrunk to 4.58 M (`hidden=256`), the backbone is now the artifact. effnetv2_s is
20.18 M; the sweep (`configs/20260721_backbone_*.yaml`, launcher `dev/run_c2_backbone_sweep.sh`)
tests five candidates at the C1 bottleneck, everything else the 0.9148 recipe:

| arch | params | why |
|---|---|---|
| `tf_efficientnetv2_b0` | 5.86 M | modern CNN, direct effnetv2 lineage — the safe anchor |
| `fastvit_t12` | 6.53 M | Apple hybrid, reparameterizes to conv at inference |
| `repvit_m1_1` | 7.77 M | 2024 conv-net designed to mimic ViT; SOTA mobile latency |
| `mobilenetv4_conv_medium` | 8.43 M | best pure-conv mobile net (2024) |
| `fastvit_sa12` | 10.56 M | FastViT **with self-attention** — the "tiny ViT done right" |

**On "tiny ViT":** vanilla ViTs are the wrong tool here on two counts — they emit token sequences
(`[N, seq, C]`) that `MTHeadAdapter`'s `AdaptiveAvgPool2d` cannot pool, and they need more
data/compute than a 5-epoch transfer budget gives. **FastViT** (Apple, CVPR 2023) and **RepViT**
(2024) are the current best small hybrids, and both *reparameterize to pure convolution at
inference* — which is exactly what the browser target wants: a clean ONNX graph and fast WASM /
WebGPU execution, no attention kernels in the hot path. That reparameterization is the single
best reason to prefer them over an attention-heavy net for this deployment.

`efficientvit_b1` and `efficientformerv2_s1` were measured and dropped — the former can't disable
its pooling head (needed for `MTHeadAdapter`), the latter fails at 256px input.

### Integration: the trainer now speaks timm

The candidates live in timm, not torchvision, and the trainer only knew `getattr(fastai.vision.
all, name)`. Added to `dev/030` (used by `032`/`040` too):

- `resolve_arch(name)` — torchvision callable if fastai re-exports it, else the timm model
  string (which `vision_learner` routes through `create_timm_model`); unknown names raise here
  rather than becoming a wrong backbone hours into a run.
- `arch_body_features(name)` — the channel count the head pools, by forwarding a dummy through
  the pool-less body. Not `model.num_features`: they disagree for nets with a post-stage conv
  head (mobilenetv4_conv_medium reports 960 but emits 1280), and it is the emitted map the head
  must match.
- `build_backbone_model(arch, head)` — `Sequential(body, head)` for the reconstruction path,
  via `create_timm_model` for timm so the module tree is byte-identical to training. **Verified:
  identical 482-key state_dict, strict load, forward to 3 levels** (fastvit_t8), on CPU, without
  touching the running C1 GPU job.

New dependency: **`huggingface_hub`** (+ `safetensors`, `hf-xet`), which timm needs even to
resolve a model config, and to fetch the pretrained ImageNet weights the sweep transfers from.
Installed additively with `uv pip install` (never `uv sync` — see
[[2026-07-venv-uv-sync-incident]]); torch/torchvision/fastai confirmed intact afterwards.

**RESOLVED 2026-07-23** — ran at hidden=256, detached. Test species macro-F1:

| arch | params | test F1 | Δ vs effnetv2_s@256 |
|---|---|---|---|
| fastvit_sa12 | 10.6M | 0.8920 | −0.82 |
| **effnetv2b2** | **8.7M** | **0.8871** | −1.31 |
| repvit_m1_1 | 7.8M | 0.8811 | −1.91 |
| fastvit_t12 | 6.5M | 0.8800 | −2.02 |
| effnetv2b0 | 5.9M | 0.8760 | −2.42 |
| mnv4_medium | 8.4M | failed → fixed (see below) |

**Chosen: effnetv2b2 (0.8871), transformers dropped.** The decisive matched-size test was b2
(8.7M) vs fastvit_sa12 (10.6M): the transformer leads by only **0.49 pp at +22% params** — not
the clear win needed to justify keeping the transformer line, per the owner's "focus on effnets
unless transformers clearly prove out". effnetv2b2 clears the 0.87 floor, is an effnet (clean
ONNX lineage from the 0.9148 teacher), and ships at ~14 MB int8. This is a smooth accuracy/size
curve, not a transformer breakthrough — FastViT/RepViT are ~per-param competitive but nothing
more. **fastvit_sa12 (0.892) remains the accuracy-first fallback if 14→~14 MB ever matters less
than 0.5 pp.**

**mnv4_medium failed — and it was a bug in my own nf-detection, caught cleanly.** For
mobilenetv4 I detected 1280 channels via `timm.create_model(global_pool="")`, but fastai's
`TimmBody` (what training builds) feeds the head 960 — its post-stage conv head differs. The
`_fix_backbone_embedding_check` guard raised at the first batch (30 s in, not silently). Fixed:
`arch_body_features` now detects through the real `TimmBody`, matching training by construction.
Not re-run — a redundant ~0.88 conv point on an already-clear frontier (owner: "don't try every
arch").

## Phase C3 (distillation) — deferred, not needed for v1

The gap from teacher (0.9148) to effnetv2b2 (0.8871) is 2.8 pp. Distillation could recover part
of it, but 0.8871 already clears the floor with margin, and the owner's priority is shipping the
app. Deferred as a v1.1 accuracy lever, not written yet. Ships effnetv2b2 as-is.

## Phase C4 — the artifact bundle (2026-07-23)

`data/global/bundles/effnetv2b2-v1/`, built via `dev/040` + `dev/043`:
`model.onnx` (fp32, 54 MB) → `model.int8.onnx` (**14.3 MB**, 3.78×, parity 100%), `taxonomy.json`
(vocabs + parent arrays), `MANIFEST.json`. Calibration (`dev/044`, marginal path, target
precision 0.9) running to add `calibration.json` + `thresholds.json`.

## Phase D — the app (2026-07-23), shipped v1

Repo: **github.com/GuillaumeMougeot/lepinet-app**, deployed to GitHub Pages via Actions.
Built as a **no-build static PWA** — the deploy box has no JS toolchain (no node/bun/npm), so
it ships plain ES modules + a vendored copy of ONNX Runtime Web + the pinned model bundle, which
Pages serves as-is. `src/infer.js` implements the two experiment-backed decisions directly:
shorter-side-resize+center-crop preprocessing (dev/041) and species→genus→family marginalization
(dev/042); optional per-level temperature + thresholds from the bundle. WebGPU→WASM fallback,
warm-up at load, service-worker precache for offline, install-to-home-screen manifest.
**Validated in Python** (the infer.js algorithm reproduces correct species/genus/family on real
held-out images); the in-browser runtime needed a real-device test, which surfaced two bugs
(2026-07-23, fixed):
- **Model load failed:** `ort.webgpu.mjs` dynamically loads one of six wasm/loader variants
  (plain / jsep / asyncify) at runtime; only the jsep pair was vendored → 404 on asyncify →
  poisoned wasm init. Fixed by vendoring all six, `numThreads=1` (Pages isn't cross-origin
  isolated), and a WebGPU→WASM-only fallback in `loadModel`. The big `.wasm` are cached
  on-fetch, not precached, to avoid a 70 MB install.
- **Stray Back button:** `.screen{display:flex}` overrode the `hidden` attribute so both screens
  rendered; fixed with `[hidden]{display:none!important}`.

Calibration (`dev/044`) completed and shipped: active thresholds target **0.95** precision
(species greys the least-confident ~6% at 94.5% precision-when-shown; genus/family never grey,
already >0.96 unconditionally). Confidence greying is live.

**Browser quantization gotcha (2026-07-23):** the int8 model failed to load in ORT Web with
*"Could not find an implementation for ConvInteger"* — on every backend, desktop and Android.
**Dynamic** int8 (`quantize_dynamic`, what `dev/043.quantize` did) emits `ConvInteger` /
`MatMulInteger`, which ONNX Runtime Web implements nowhere. Fix: `dev/043.quantize_static_qdq` —
**static QDQ** quantization with an image calibration reader, producing
`QuantizeLinear`/`DequantizeLinear` around ordinary `Conv`/`Gemm`, all ORT-Web-supported.
Numerically identical to the dynamic int8 (same −0.59 pp; my B2 result was never a numerical
problem, only an op-encoding one), **54 MB → 15.5 MB**, ops verified `ConvInteger`-free and
species top-1 unchanged (0.900). Shipped to the app; fp32 (54 MB) remains the guaranteed-safe
fallback. Lesson: for ORT Web, always static-QDQ, never dynamic.

**GitHub Release live:** `lepinet` tag `model-v1-effnetv2b2` with `model.onnx` (fp32),
`model.qdq.onnx` (browser int8), taxonomy/calibration/thresholds/manifest. The app pins its own
copy for now; pulling from the release is a future nicety (owner's note).

**QDQ confirmed working in-browser** (owner tested, loads + installs) — so the app default is the
**QDQ int8** model (15.5 MB) with **fp32 as fallback** (via `config.fallback` → the GitHub release
model.onnx, tried only if the primary won't create a session).

**App v1.1 (2026-07-23):** scientific names (`dev/047_build_names.py` builds `names.json` from the
parquet's `scientificName`/`genus`/`family`, aligned to taxonomy vocab order — 0 missing), shown
instead of GBIF keys; split Camera / Gallery buttons (the combined one only opened the camera);
**config-driven modular bundle** (`model/config.json` declares model + sidecars + IO names, so a
new model is a folder swap, no code change) documented in the app's `DEVELOPER.md`.

**Future development is consolidated in the app repo's `ROADMAP.md`** (single doc spanning model +
app): C3 distillation (top model-quality lever), pull-model-from-release, real-device test matrix,
geo prior (blocked on co-occurrence data), open-set, regional builds, size levers, and the app
feature backlog — prioritized. This journal remains the record of what was *done* and why.

Also fixed (2026-07-23): a wedged service worker after rapid cache-version churn — the SW was
precaching the 15 MB model via `addAll`, so one flaky download failed the whole install and left
it stuck on a stale version (surfaced as "Model failed to load: undefined"). Now precache only the
small shell (failure-tolerant, per-item); model + wasm cache on first fetch. Error reporting shows
the real error, not `undefined`.

## Where this leaves the size budget

| step | status | size |
|---|---|---|
| baseline fp32 | measured | 171.8 MB |
| drop genus+family heads (marginalize) | **proven** | ~155 MB |
| int8 PTQ | **measured, −0.59 pp F1** | **44.4 MB** |
| `hidden` 1280 → 256 | **trained, −1.46 pp F1** (fp32) | ~110 MB fp32 → **~28 MB int8** |
| smaller backbone, distilled | training (512/128 first) | ~8–12 MB |
| 4-bit head prototypes | not started | ~6–8 MB |

**Achieved so far: 3.9× for 0.59 pp (int8), plus a proven 5× head shrink for 1.46 pp
(bottleneck).** The two compose — a 256-bottleneck model quantized to int8 is the ~28 MB row and
its accuracy cost is measured on both axes separately. The ≤8 MB target still needs the backbone
step, but nothing so far casts doubt on it, and the `hidden` discovery made the largest head step
a config change rather than new architecture.

## Predictions, scored

From `2026-07-lepi-app-claude.md` §8, written before any of this ran:

| prediction | outcome |
|---|---|
| int8 on the cosine head is nearly free | ✅ −0.59 pp species macro-F1 for 3.9× |
| marginal ≥ direct at genus and family | ✅ better on both metrics at both levels |
| final bundle 6–8 MB, not <1 MB | ⏳ Phase C |
| 256-dim bottleneck costs < 1.0 macro-F1 | ✗ **−1.46 pp** — a miss, but well above the 0.87 floor |
| distilled effnetv2-b0 ≥ 0.88 test macro-F1 | ⏳ Phase C (after the bottleneck sweep) |

And two review claims that did **not** survive contact with measurement, recorded because a
wrong prediction is the most useful thing in a journal:

| review claim | reality |
|---|---|
| browser resize mismatch is a Phase-A blocker | ✗ worth ≤0.25 pp unless aspect ratio is broken |
| the model is systematically overconfident | ✗ T ≈ 0.8 < 1, it is *under*confident |
