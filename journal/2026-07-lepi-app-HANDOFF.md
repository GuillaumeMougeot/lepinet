# Lepinet App — Handoff Report

**Date:** 2026-07-23 · **Purpose:** everything a developer (or a fresh Claude session) needs to
resume this work with *only this document + the two codebases*. Self-contained on purpose; deeper
reasoning is cross-referenced but the essentials are restated here.

---

## 0. Sixty-second summary

**Goal:** a fast, offline phone/desktop app that identifies a moth/butterfly photo to
**species, genus, family**, running the model on-device (no photo leaves the device).

**Two repos:**
- **`lepinet`** (`~/codes/lepinet`, github.com/GuillaumeMougeot/lepinet) — the model: training,
  the compression pipeline, all the reasoning (journals). This is where the ML work lives.
- **`lepinet-app`** (`~/codes/lepinet-app`, github.com/GuillaumeMougeot/lepinet-app) — the PWA.
  A **no-build static site** (no JS toolchain on the box), deployed to GitHub Pages at
  **https://guillaumemougeot.github.io/lepinet-app/**.

**Current state:** the app is **live and functional** — loads the model, on-device inference,
consistent 3-level output with scientific names, calibrated confidence greying, camera + gallery
input, installable, offline-capable.

**The one big wart:** it ships the **fp32 model (54 MB)**. A 15 MB int8 version exists and is
numerically correct, but **fails to load in ONNX Runtime Web** (see §7.1). Shrinking the download
is the top open problem, and it's **blocked on in-browser testing** (no browser automation on the
box).

**Model:** EfficientNetV2-B2 + 256-d bottleneck head, 12,041 species, **test species macro-F1
0.887** (teacher was 0.9148).

**HEADs at handoff:** `lepinet` = `6a375bf`, `lepinet-app` = `a2d0f94`.

---

## 1. Environment — read this first, it will save you hours

The box has an unusual setup. Assume nothing.

### Python / the venv (`~/codes/lepinet/.venv`)
- **NEVER run `uv sync` or `uv run` in `lepinet`.** The venv is hand-assembled with `uv pip
  install`; there is no lockfile, and `uv sync` prunes everything not in `pyproject.toml`
  (it once removed torch/torchvision/mini_trainer and broke the ABI). Always invoke
  **`.venv/bin/python`** directly (or `source .venv/bin/activate`).
- Python **3.14**, `torch==2.12.1+cu130` (pairs with `torchvision==0.27.1` — mismatched versions
  throw `operator torchvision::nms does not exist`), `fastai==2.8.7` (needs `fastcore<2`,
  `fastprogress==1.0.3`), `timm==1.0.28`.
- Sibling editable installs: `mini_trainer`, `mini_metrics` at `~/codes/mini_trainer`,
  `~/codes/mini_metrics`. The hierarchical classification heads live in `mini_trainer`.
- Extra deps added for this work (all via `uv pip install`, never sync): `onnx onnxruntime
  onnxscript onnxslim onnx-ir ml-dtypes` (export/quant), `huggingface_hub safetensors hf-xet`
  (timm needs them for pretrained weights).

### No JS toolchain, no gh by default
- **No `node`/`bun`/`npm`/`npx`.** This is *why the app is a no-build static site*. Do not add a
  build step; add libraries as vendored ES modules (see `ort/`).
- **`gh` is installed at `~/micromamba/bin/gh`** (via micromamba, conda-forge) and **is
  authenticated** as GuillaumeMougeot (keyring). Use `export PATH=~/micromamba/bin:$PATH` then
  `gh …`. Git itself pushes over **SSH** (`git@github.com:…`).
- `micromamba` is available if you need another CLI tool conda-forge has.

### GPU
- One **RTX 5090**, 32 GB. **Reserved for this project through ~2026-08-10.**
- Long training runs must be **detached from the session** — the harness reaps background tasks
  after a while, but `setsid nohup <script> >log 2>&1 </dev/null &` survives. Per-epoch
  checkpoints (`SaveModelCallback`) mean a killed run is resumable.
- Force CPU for anything that shouldn't touch the GPU (e.g. smoke tests while training runs):
  `CUDA_VISIBLE_DEVICES="" .venv/bin/python …`.

### Data (gitignored, machine-local)
- `~/codes/lepinet/data/` is a symlink to local storage, **gitignored**. So `RESULTS.md` (tracked)
  is the only copy of the run table that leaves the box.
- Global dataset: `data/global/0032836-…_quality_filtered.parquet` (the **raw** one still has the
  `set` fold column and the name columns; the `.lepinet.parquet` caches do not). Images under
  `data/global/images/<speciesKey>/<filename>`. **All 12,632 test-fold species' images are present**
  (not a partial mirror — an earlier note claiming otherwise was wrong).
- Model checkpoints: `data/global/models/<timestamp>-<name>/<name>.pt`.
- The app's model bundle source: **`data/global/bundles/effnetv2b2-v1/`** (contains `model.onnx`
  fp32, `model.int8.onnx`, `model.qdq.onnx`, `taxonomy.json`, `names.json`, `calibration.json`,
  `thresholds.json`, `MANIFEST.json`).

---

## 2. The model

### What it is
EfficientNetV2-B2 backbone (from timm, pretrained ImageNet) + **mini_trainer's hierarchical
classification head**: a cosine (L2-normalized prototype) classifier with a **256-d bottleneck**
(`hidden=256`) before the class layers. Trained through **fastai's** `vision_learner` loop (not
mini_trainer's own loop). Predicts species/genus/family; the three are trained as **independent**
heads.

- Dataset: ~5.7 M GBIF images, **12,041 species / 4,333 genera / 102 families**. 10-fold split in
  the `set` column; fold `'1'` = validation, fold `'0'` = held-out test, rest = train.
- The model outputs raw logits per level; **ImageNet normalization is baked into the exported
  ONNX graph** (input = RGB `[1,3,256,256]` in `[0,1]`).

### The training recipe (the 0.887 model)
Config: `configs/20260721_backbone_effnetv2b2.yaml` (+ `20260722_backbone_effnetv2b2.yaml`).
Trainer: `dev/030_hierarchical_heads_benchmark.py`. Key knobs (all held from the 0.9148 baseline):
sqrt **oversampling** (`oversample_power: 0.5`, the single biggest lever), **Muon** optimizer,
**one-cycle** schedule, warmup 0.5 ep, grad-clip 5.0, bs 64, aug 460→256, light augmentation,
5 epochs, bf16, `hidden: 256`.

### How to train / test / read results
```bash
cd ~/codes/lepinet
.venv/bin/python dev/030_hierarchical_heads_benchmark.py --config configs/<train>.yaml   # → .pt
.venv/bin/python dev/032_hierarchical_heads_test.py      --config configs/<test>.yaml     # → metrics
.venv/bin/python dev/036_ledger.py                # print the run table (val + test species macro-F1)
.venv/bin/python dev/036_ledger.py --snapshot     # regenerate RESULTS.md (commit it)
```
`dev/030` supports **both torchvision and timm** backbones (`resolve_arch`/`arch_body_features`/
`build_backbone_model`); `model_arch_name` can be `efficientnet_v2_s` or a timm name like
`tf_efficientnetv2_b2`, `fastvit_sa12`.

### The compression journey — results
How the model got from the 165 MB research checkpoint to the shipped one. Full reasoning:
`journal/2026-07-lepi-app-compression.md`. The numbers (test species macro-F1, full held-out fold):

| step | what | result |
|---|---|---|
| baseline | effnetv2_s, `hidden=1280` (teacher) | **0.9148** (165 MB fp32) |
| C1 bottleneck | `hidden` 512 / **256** / 128 | 0.9058 / **0.9002** / 0.8843 — knee at 256 (5× smaller head, −1.46 pp) |
| C2 backbone | at `hidden=256`: effnetv2b0 / repvit_m1_1 / fastvit_t12 / fastvit_sa12 / **effnetv2b2** | 0.8760 / 0.8811 / 0.8800 / 0.8920 / **0.8871** |

**Decision: effnetv2b2 (0.8871).** fastvit_sa12 led by only 0.49 pp at +22 % params — not a clear
enough win to keep the transformer line (owner preferred effnets). `mnv4_medium` failed (a
now-fixed nf-detection bug) and wasn't re-run.

Earlier phases (all **resolved, measured**):
- **A** — ONNX export works (`dev/040`), PyTorch↔ORT parity 2e-5.
- **A** — browser preprocessing is a non-issue (`dev/041`): shorter-side resize + center crop
  matches training within 0.1 pp; only aspect-ratio squashing costs anything.
- **B1** — genus/family are better **marginalized** from the species posterior than read from
  their own heads (`dev/042`): +0.7 / +3.1 pp AND consistent by construction. The app does this.
- **B2** — int8 costs only −0.59 pp (numerically). (But see §7.1 — int8 doesn't run in ORT Web.)
- **B4** — the model is *under*confident (temperature ≈ 0.8); calibration + precision-targeted
  thresholds via `dev/044`.

### C3 (distillation) — NOT done
The one deliberately-skipped step. Teacher (0.9148) is 2.8 pp above the student (0.887).
Distilling could recover ~0.5–1.5 pp. Needs a new `dev/045` that runs the teacher per batch and
adds a KL term to `MultiLevelWeightedCrossEntropyLoss`. Highest-value remaining *model* work.

---

## 3. The compression / bundle pipeline (`dev/040`–`047`)

Each is standalone; run from `~/codes/lepinet` with `.venv/bin/python`. Reasoning in the journal.

| script | what it does |
|---|---|
| `040_onnx_export.py` | `.pt` → `model.onnx` (fp32) + `taxonomy.json` + `MANIFEST.json`. Bakes normalization, emits raw logits, checks PyTorch↔ORT parity. Uses the **dynamo** exporter (legacy tracer dies on the head's dict `get_extra_state`) + an eager warm-up pass (the head's mask cache is data-dependent). |
| `041_ort_parity.py` | Measures browser-style resize vs the fastai transform; `--emit-fixture` writes images + expected logits for a future in-browser test. |
| `042_marginalize.py` | Proves/executes species→genus→family marginalization (log-space scatter-logsumexp). |
| `043_quantize.py` | `quantize()` = dynamic int8 (**emits ConvInteger — unusable in ORT Web**, server-measurement only). `quantize_static_qdq()` = static QDQ int8 with an image calibration reader (browser-*intended*, but see §7.1). |
| `044_calibrate.py` | Per-level temperature (grid-fit) + precision-targeted thresholds on the marginal path → `calibration.json`, `thresholds.json`. |
| `047_build_names.py` | `names.json` (scientific binomials from the parquet's `scientificName`/`genus`/`family`), aligned to taxonomy vocab order. |

Sweep runners: `run_c1_bottleneck_sweep.sh`, `run_c2_backbone_sweep.sh`, `run_c2b_effnetv2b2.sh`
(all designed to run detached).

### Rebuilding the app bundle from a checkpoint (the exact recipe used)
```bash
cd ~/codes/lepinet
B=data/global/bundles/effnetv2b2-v1
.venv/bin/python dev/040_onnx_export.py -c "data/global/models/*backbone-effnetv2b2*/*.pt" -o $B
.venv/bin/python dev/044_calibrate.py --onnx $B/model.onnx --out-dir $B --target-precision 0.95
.venv/bin/python dev/047_build_names.py --parquet data/global/0032836-*_quality_filtered.parquet \
    --taxonomy $B/taxonomy.json --out $B/names.json
# (int8: dev/043.quantize_static_qdq — but it does NOT load in ORT Web; see §7.1)
# Then copy model.onnx + taxonomy/names/calibration/thresholds into lepinet-app/model/,
# write config.json, bump sw.js cache, commit+push.
```

---

## 4. The app (`lepinet-app`)

### Philosophy
**No build step** (no node on the box). Plain ES modules, a vendored copy of ONNX Runtime Web
under `ort/`, and a **config-driven model bundle** under `model/`. GitHub Pages serves the repo
root as-is; `.github/workflows/deploy.yml` just uploads it (no compile).

### File map
```
index.html              app shell: home screen (camera/gallery + install) and result screen
src/app.js              UI controller: capture → infer → render 3-level result
src/infer.js            THE CORE: loads config+bundle, preprocess, run, marginalize, calibrate
src/install.js          install affordance (beforeinstallprompt button + written instructions)
src/style.css           styles (light/dark, [hidden] fix)
sw.js                   service worker: precache small shell; cache model+wasm on first fetch
manifest.webmanifest    PWA manifest
ort/                    vendored ONNX Runtime Web — ALL SIX wasm variants + loaders (see §7.2)
model/                  the bundle: config.json, model.onnx, taxonomy.json, names.json,
                        calibration.json, thresholds.json
DEVELOPER.md            "bring your own model" — the bundle contract (read this to swap models)
ROADMAP.md              consolidated future work (model + app), prioritized
README.md               overview
.github/workflows/deploy.yml   Pages deploy (upload-pages-artifact, no build)
```

### How inference works (`src/infer.js`)
1. `loadModel()` reads `model/config.json` (declares model file, sidecars, IO tensor names, image
   size, GBIF base), loads taxonomy/names/calibration/thresholds, creates the ORT session
   (WebGPU→WASM fallback; `numThreads=1` because Pages isn't cross-origin isolated), and warms up.
2. `preprocess()` — shorter-side resize to 256 + center crop, → `[1,3,256,256]` `[0,1]` RGB.
3. `predict()` — run model, take **species** logits, `log_softmax`, **marginalize** genus then
   family via `taxonomy.parents.*`, apply temperature, look up name+key, compare to threshold.

### The bundle contract (so you can swap models — full version in `DEVELOPER.md`)
`model/config.json`:
```jsonc
{ "name": "...", "model": "model.onnx", "fallback": null,
  "taxonomy": "taxonomy.json", "names": "names.json",
  "calibration": "calibration.json", "thresholds": "thresholds.json",
  "imageSize": 256, "inputName": "image",
  "outputs": { "species": "logits_species", "genus": "logits_genus", "family": "logits_family" },
  "gbifBase": "https://www.gbif.org/species/" }
```
- `taxonomy.json`: `vocabs.{species,genus,family}` = arrays of **GBIF taxon keys in model-output
  order**; `parents.{species_to_genus,genus_to_family}` = child-index → parent-index arrays.
- `names.json`: `names.{species,genus,family}` = display names aligned to the same vocab order.
- The **model must**: take RGB `[0,1]` `[1,3,S,S]`, bake its own normalization, emit raw logits;
  the app only strictly needs the species output. **Preprocessing the app does = shorter-side
  resize + center crop** — train to match.

### Deploy / test locally
- Push to `main` → GitHub Actions deploys to Pages (~1–2 min).
- Local: `cd ~/codes/lepinet-app && ~/codes/lepinet/.venv/bin/python -m http.server 8000`, open
  `http://localhost:8000`. Single-thread WASM needs no special headers.
- **`sw.js` `CACHE` version must be bumped every deploy** or clients keep stale files. Currently
  `lepinet-v8`.

---

## 5. Making a GitHub release of a model
`gh` is authed. From `~/codes/lepinet`:
```bash
export PATH=~/micromamba/bin:$PATH
gh release create <tag> --repo GuillaumeMougeot/lepinet --title "..." --notes "..." <files…>
gh release upload <tag> <file> --repo GuillaumeMougeot/lepinet   # add assets
```
Current release: **`model-v1-effnetv2b2`** — has `model.onnx` (fp32), `model.qdq.onnx`,
`taxonomy/names/calibration/thresholds/MANIFEST`. Owner wants the app to eventually **pull the
model from the release** instead of bundling it (not done — see ROADMAP).

---

## 6. Key decisions & why (so you don't re-litigate them)
- **effnetv2b2, transformers dropped** — best effnet under the teacher; the best transformer
  (fastvit_sa12, 0.892) beat it by only 0.49 pp at +22 % params. Owner preference: effnets unless
  a transformer *clearly* wins.
- **`hidden=256`** — knee of the bottleneck curve; the *shipped* head after marginalization is
  ~3.4 MB int8 vs 6.8 MB at 512, and the head is the size-binding part.
- **Marginalize genus/family** — more accurate + consistent (the stacked UI would otherwise show
  contradictory levels ~1.8 % of the time).
- **fp32 shipped, not int8** — int8 fails in ORT Web (§7.1). Size relaxed by owner (">8 MB ok").
- **Names shown, not GBIF keys** — from the parquet; key still drives the GBIF link.
- **Two capture buttons** — the single one forced the camera; gallery testing needs the
  no-`capture` input.

---

## 7. OPEN PROBLEMS (what to work on next)

### 7.1 ⭐ THE big one: the model download is 54 MB (int8 won't run in ORT Web)
- **Symptom:** the static-QDQ int8 model (15.5 MB, `data/global/bundles/effnetv2b2-v1/model.qdq.onnx`)
  throws a **raw numeric WASM error (e.g. `9399128`)** at `InferenceSession.create` on a real
  device (desktop + Android). fp32 loads fine.
- **What's ruled out:** it's not `ConvInteger` (that was *dynamic* int8, replaced). The QDQ graph
  has only `Conv` + `QuantizeLinear`/`DequantizeLinear` + `Acos/ReduceL2/…`, all nominally
  supported, and it produces correct outputs in Python (onnxruntime CPU). **Lesson learned the
  hard way: Python op-verification does NOT prove ORT-Web runtime support.**
- **The real blocker: no way to test in a browser on this box** (no browser automation; Claude's
  Chrome extension isn't connected). Every quantization attempt costs a human round-trip.
- **What to try (in order), each needing in-browser validation:**
  1. **Per-tensor** static QDQ (current is `per_channel=True`) — ORT-Web may not support
     per-channel `DequantizeLinear` axis for some ops. Regenerate with `per_channel=False`.
  2. **fp16** with the **head kept fp32** (block-list `ReduceL2`/`Div`/`Acos`/the cosine `Gemm` in
     the float16 converter) — the cosine head is fp16-sensitive. ~27 MB.
  3. Add per-attempt `console.error` logging in `createSession` to see *which* provider/model
     throws and capture the real message.
  4. Custom minimal ORT-Web build (only the ops this graph uses) to shrink the ~23 MB wasm too.
- **Fastest unblock:** get **claude.ai/chrome** connected so the agent can drive a browser and
  test variants directly — this is called out in ROADMAP as the key enabler.

### 7.2 Service-worker caching is fragile to rapid redeploys
- The SW precaches only the small shell now (per-item, failure-tolerant) and caches model+wasm on
  first fetch — because precaching the big model via `addAll` let one flaky download wedge the SW
  on a stale version. **Always bump `CACHE` in `sw.js` on deploy.** If a client is wedged, the fix
  is *Clear site data* + reload.
- The vendored `ort/` has **all six** wasm variants (plain/jsep/asyncify × mjs/wasm) because
  `ort.webgpu.mjs` dynamically imports one at runtime and 404s otherwise.

### 7.3 Other known gaps
- **In-browser runtime never auto-tested** — only the *algorithm* is validated (Python reproduces
  correct species/genus/family on held-out images). Needs a real-device matrix, esp. **iOS
  Safari** (untested).
- The app shows **GBIF taxon keys as the link target**; names are display-only (fine).
- Owner's requested nicety: **pull model from the GitHub release** (decouple model/app) — not done.

---

## 8. Suggested next steps (priority order)
1. **Unblock size:** connect a browser (claude.ai/chrome) → get per-tensor QDQ *or* fp16 loading →
   ship ~15–27 MB instead of 54 MB. Biggest product win. (§7.1)
2. **C3 distillation** — recover accuracy toward 0.9148; GPU is reserved. (§2)
3. **Real-device test matrix**, especially iOS Safari. (§7.3)
4. Then the feature/size backlog in `ROADMAP.md` (pull-from-release, top-k, history, geo prior —
   the last is *blocked* on co-occurrence data the project doesn't have).

Full, prioritized future work: **`lepinet-app/ROADMAP.md`**.

---

## 9. Where the detailed record lives
- **`journal/2026-07-lepi-app.md`** — the original proposal + a review.
- **`journal/2026-07-lepi-app-claude.md`** — the detailed plan (phases A–E) + the resolved product
  decisions (§7 there).
- **`journal/2026-07-lepi-app-compression.md`** — the blow-by-blow of A→D with every measurement,
  predictions scored, and every bug/lesson. **The primary reasoning record.**
- **`RESULTS.md`** (tracked) — the run table. `dev/README.md` — catalogs every `dev/` script.
- **`lepinet-app/DEVELOPER.md`** — bundle contract / bring-your-own-model. **`ROADMAP.md`** — next.
- Memories (`~/.claude/projects/-home-au761367-codes-lepinet/memory/`) capture the venv rules and
  testing pipeline for future Claude sessions.

## 10. Lessons banked (don't relearn these)
- **Python op-check ≠ ORT-Web support.** Validate models in an actual browser. (§7.1)
- **`uv sync`/`uv run` will break the venv.** `.venv/bin/python` + `uv pip install` only.
- **Detach long GPU jobs** (`setsid nohup … </dev/null &`); harness background tasks get reaped.
- **Bump the SW cache** every app deploy; precache only small files.
- **nf detection for timm must go through fastai's `TimmBody`**, not a bare `timm.create_model`
  (mobilenetv4's conv-head makes them disagree, 1280 vs 960 — it crashed a run).
- Two upstream `mini_trainer` bugs are worked around per-instance in `dev/030` (the
  `AutoregressiveClassifier` MRO bug and the `Classifier._reshape_backbone_embeddings` width
  check) — don't remove those shims.
- **The dev/032 test default is `min_img_per_spc=0`** (the full 12,632-species fold). Using 50
  silently evaluates only common species and inflates every number.
