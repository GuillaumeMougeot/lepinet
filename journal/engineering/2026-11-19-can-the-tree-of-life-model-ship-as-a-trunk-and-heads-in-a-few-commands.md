# Can the Tree of Life model ship as a trunk and heads, in a few commands?

**Kind:** subproject · **Status:** OPEN (2026-11-19). `mt_release` in mini_trainer (branch
`feature/export-tol`) exports, verifies and packages a checkpoint in two commands; tol_full and TL4
pass verification on real images. Open: the licence, calibration and int8 (owner's decisions and
later work, listed at the end).

Owner's request (2026-11-19): read everything lepinet learned about deploying a model (quantisation,
ONNX, the app bundle, the Hugging Face releases), check that mini_trainer has it and is as easy to
use and as readable, run end-to-end tests on the current Tree of Life (ToL) model, so that when a
ToL-based model is the best one, exporting and publishing it takes a few commands. Mid-task the
owner added (relayed by the supervisor): publish the **trunk and the head separately** (BioCLIP-2
published its model and its text embeddings as separate files), with specialisation heads in the
same format and a combined file kept; assess formats; drop the web app. Then: **safetensors is the
main Hugging Face release, ONNX the deployment file**.

Work done by an export agent on CPU (cores 0-3, under a 12 GB cap), in its own mini_trainer worktree
(`.worktrees/export-tol`). Nothing was uploaded.

## Prediction (written before the release's parity numbers; `mt_export` had already failed, below)

- The ToL graph would exceed protobuf's 2 GB because of the head (190,199 x 1536 fp32 = 1.17 GB plus
  a 0.8 GB trunk); the trunk alone would not.
- A release that reuses the checkpoint's own transform inside the graph and mini_trainer's own image
  reader would reproduce mini_trainer to float rounding: top-1 agreement 100 % at every rank, max
  logit difference below 1e-3 for fp32; fp16 trunk 99+ %.

## Result

**Yes, in two commands, and the files reproduce mini_trainer exactly at the decision level.** On 300
real ToL test images every artifact (`trunk.onnx`, `trunk_fp16.onnx`, `model.onnx`,
`trunk.safetensors` through timm, each with the head scored from its safetensors file in numpy)
agrees with mini_trainer's own evaluation on the top-1 class at **all seven ranks (100 %)**, with
embedding cosine at least 0.999997 and pixel-identical preprocessing on 100 % of images. Found before
any prediction: mini_trainer's own `mt_export` could not run on this machine (it needed more than
14 GB; fixed). Not as predicted: the maximum logit difference is 0.05, not below 1e-3. That is the cosine clamp: asin is
vertical at -1, and 73 % of (image, class) pairs sit within 1e-4 of it (TL7's floor). Away from the
clamp the largest difference is 1.7e-3, and at each image's top class 5e-5.

## Audit: lepinet's export work against mini_trainer

mini_trainer as it was (`feature/lepinet-integration`, c361471) against `feature/export-tol`. lepinet
paths are in this repo; mini_trainer paths in `~/codes/mini_trainer`.

| capability or lesson | lepinet | mini_trainer before | now | left |
|---|---|---|---|---|
| checkpoint -> deployable folder in one command | `lepinet bundle` (`src/lepinet/cli.py:133`, `export.py:322`); the HF release is `dev/083_hf_release.py`, 8 subcommands | `mt_export` (`mini_trainer/export.py:16`): graph and manifest only; MAMBO's release scripts are specific to it (`dev/releases/mambo_v3/build_bundle.py`) | `mt_release export`, then `mt_release package-hf` | |
| export fits in memory | small models only | `mt_export` held five copies of the weights: tol_full failed at the 14 GB cap | `mt_export` 9.0 GB (d70e6e9); `mt_release export` + verify 10.5 GB, verify alone 6.8 GB | |
| preprocessing inside the graph | normalisation (`export.py:29`) and the last resample (`dev/083:165`, `--input-size`) | outside the graph, `requires_configuration: true` (`mini_trainer/modeling/onnx.py:275`) | the checkpoint's own `ResampleNormalize` in the graph; input is the reader's uint8 square | |
| a reader a user can copy | card snippet and `predict.py` (`dev/083_hf_release_files/predict.py:24`) | MAMBO's recipe only (`deployment/mambo_deploy/preprocessing.py:9`) | `standalone.read_square` = mini_trainer's evaluation reader; identical pixels on 100 % of 600 images | |
| aspect ratio matters, the kernel does not when down-sampling; small up-sampled trap crops flip 3-6 % | 2026-07-20 section 2; 2026-09-29 | | the release reads exactly as training does, so no flips by construction (TL4's set includes 100 trap crops) | |
| parity on real images, per rank | `dev/083 eval` of the published file; `verify_onnx` on random input (`export.py:234`) | the example input only; for `mt_export`, preprocessed zeros, i.e. one constant image | `mt_release verify`: per rank top-1, max and mean logit difference, embedding cosine, per file, with gates | |
| size over the external-data sidecar; protobuf's 2 GB | 2026-07-20 section 1 | always external data; hashes, no sizes | one file under 1.8 GB, external data above; sizes and sha256 in `config.json` | |
| warm lazy caches before tracing | 2026-07-20 section 1 | done | done | |
| embedding output | `embedding` (`dev/083:210`) | not exported (`EmbeddingContext` at predict time only) | trunk graph: `features` and `embedding`; combined graph: `embedding` | |
| coherent coarse ranks | probability sums in the graph (`dev/083:211`) | logsumexp inside the head | the same logsumexp in the head file's reference scoring | |
| labels and tree as files | `taxonomy.json` (`export.py:88`), `names.json` (`calibrate.py:185`) | inside a 124 MB `manifest.json` (ToL) | `heads/<name>.json` (20 MB) and `parent.<k>` tensors | display names for GBIF-key heads (TL4) |
| fp16 from source, head kept fp32 | `Fp16ExportWrapper` (`export.py:59`); post-hoc conversion failed | the whole model cast to the input dtype (fp16 input and head) | `trunk_fp16.onnx`: fp16 backbone, fp32 hidden layer and normalisation | fp16 head file (measured, not built) |
| fp16 is a GPU format; ORT's NCHWc pass crashed fp16 ViTs on CPU | 2026-09-29 | | ConvNeXt fp16 loads and runs on CPU, at fp32's speed | GPU latency not measured (GPU busy) |
| int8 per architecture; ORT's quantizer mishandles `Gemm transB` | `dev/083:509 gemm_to_matmul`, `dev/083:547 quantize` | a static-calibration benchmark (`dev/benchmarks/inference/onnx_calibration.py`); no release path | not built | port `quantize --arch vit` (B8 is the same ConvNeXt-L: -0.18/+0.33/+0.05 pt, 1.46x, 3.5x smaller) |
| temperature in the graph; 95 % back-off thresholds | `calibrate.py:43`; `dev/083:386`, `dev/083:484` | nothing generic | not built; tol_full's species softmax is not saturated (2.3 % of known images at 0.999 or more; B8 had 81 % at exactly 1.0) | `mt_release calibrate`: temperature into the head's metadata, a thresholds file |
| CUDA provider choice (TensorRT listed first; `preload_dlls`) | `dev/083_hf_release_files/predict.py:38` | MAMBO's session code | `standalone.onnx_session` | |
| HF folder: card, licence, `predict.py` run in a clean environment | `dev/083` and its README templates | MAMBO's card only | `package-hf`: card from the release and an owner-written JSON; `predict.py` run in an isolated interpreter (no torch, timm or mini_trainer imported); not ready without a licence | |
| upload | `publish_to_hf` (`export.py:460`), `dev/083 upload` | none | prints `hf upload ...`; never uploads | |
| transformers | `dev/083 transformers`, P5 only | none | `trunk.safetensors` loads into transformers' `DINOv3ConvNextModel` with a key rename; features identical (difference 0.0) | the rename and a head module for `from_pretrained` |
| safetensors needs `format: pt` for transformers | 2026-09-29 | | written | |
| species lists | K1 | class masks (MAMBO presets) | `standalone.subset`: rows of the head; equals mini_trainer's class mask | |
| one trunk, several heads | | | the trunk's hash in `config.json`; `add-head` refuses a different trunk | head-only ONNX graphs |
| app bundle; int8 in ONNX Runtime Web | `export.py:188`; 2026-07-20, 2026-07-25 | | **deferred** (owner, 2026-11-19) | |

## What was built: `mt_release`

`mini_trainer/release/` (about 700 lines in five files): `build.py` (export), `verify.py`, `card.py`
(Hugging Face folder), `cli.py`, and `standalone.py`, which is copied into each release as
`predict.py` and imports nothing from mini_trainer, so **the code that was verified is the code that
ships**. Guide: `docs/release.md` in mini_trainer.

**Where to cut.** The trunk ends at the classifier's input, not at the normalised embedding.
mini_trainer's `--freeze-backbone`, which TL8 uses, freezes everything except `head.fc` (the
`Classifier`), so a head trained on the frozen ToL trunk trains its **own hidden layer**. A trunk
ending at ToL's embedding could not serve it. The trunk graph returns both `features` (the cut) and
`embedding` (ToL's normalised embedding, the one TL1, TL7 and Z20 measured), so it is the embedding
model too. A head file holds the hidden layer, the unit prototypes, `g`, `b` and `parent.<k>`; its
scoring is about 20 lines of numpy and handles `magnitude_after_angle` (tested against mini_trainer's
classifier in both modes). `add-head` accepts a checkpoint only if its trunk weights hash to the
release's trunk and its preprocessing matches; TL4 is correctly refused (below).

**Formats.** As the owner decided: safetensors for release (`trunk.safetensors`,
`heads/<name>.safetensors`, `model.safetensors` = both under `trunk.` / `head.` prefixes, checked
byte for byte against the separate files), ONNX for deployment (`trunk.onnx`, `trunk_fp16.onnx`,
`model.onnx`). `--no-combined` drops both combined files.

## End to end: tol_full (190,199 species, read 256, model 224)

**mini_trainer's existing path first.** `mt_export --weights tol_full_last.pt` failed: ONNX Runtime
could not allocate its verification session under the 14 GB address cap (11.8 GB resident). The
fix (d70e6e9: reference outputs computed first, the private copy exported directly and released,
the checkpoint state dropped) exports it in 29 s at 9.0 GB. Its own check is on one constant image
(max |Δ| 7.6e-5) and it writes no labels file, preprocessing or embedding output; that is what
`mt_release` adds.

**`mt_release`, 300 known-species test images** (`emb_known/paths.json`, seeded sample), 4 CPU threads:

| file | precision | top-1 agreement, every rank | max / mean logit difference (species) | embedding cosine min | CPU, 1 image | CPU, batch 16 |
|---|---|---|---|---|---|---|
| `trunk.onnx` + head | fp32 | 100 % | 0.046 / 1.3e-5 | 1.000000 | 141 ms | 6.2 img/s |
| `trunk_fp16.onnx` + head | fp16 trunk | 100 % | 1.63 / 2.8e-3 | 0.999997 | 142 ms | 6.2 img/s |
| `model.onnx` | fp32 | 100 % | 0.055 / 1.9e-5 | 1.000000 | 167 ms | 6.1 img/s |
| `trunk.safetensors` (timm) + head | fp32 | 100 % | 0.012 / 1.5e-6 | 1.000000 | | |
| `model.safetensors` | | identical bytes to the separate files | | | | |

The head in numpy: 27 ms for one image, 149 img/s at batch 16. Sizes: `trunk.onnx` **797 MB, one
file** (well under 2 GB, as predicted), `trunk_fp16.onnx` 404 MB, `trunk.safetensors` 785 MB,
`heads/tol.safetensors` 1,181 MB, `heads/tol.json` 20 MB, `model.safetensors` 1,966 MB, `model.onnx`
1.7 MB + 1,966 MB external data; 6.7 GB in all, 3.2 GB without the combined files. Export about 1
min, verification 5 min. Peak memory: export and verification in one process 10.5 GB; verification
alone 6.8 GB; `package-hf` 3.5 GB.

**Further checks** (`dev/114`; outputs in `data/release_checks/`):

- *Against mini_trainer's GPU run of 2026-11-17* (`mt_hpredict`, the `emb_known` arrays):
  `predict.py` on CPU gives the same class at six ranks on 300 / 300 images and at order on 299 / 300;
  embedding cosine with the stored fp16 embeddings at least 0.999997.
- *Where the logit differences sit* (64 images, the numpy head against mini_trainer's classifier on
  the same features): the largest (0.014) is at g·cos = -0.9999993; 73 % of pairs are within 1e-4 of
  the clamp; elsewhere at most 1.7e-3; at each image's top class at most 5e-5.
- *A species list is a subset of rows:* the head restricted to its 17,862 Lepidoptera rows equals
  mini_trainer's `set_active_features` at every rank (top-1 100 %, max difference 2e-3; 5,707 genera
  down to one kingdom).
- *fp16 prototypes* (head file 1,181 -> 590 MB): top-1 unchanged at every rank on 300 images (max
  logit difference 0.24). Not built; a one-line option if wanted.
- *transformers:* the 342 trunk tensors, renamed in order, load into `DINOv3ConvNextModel`
  (transformers 5.23, no missing or unexpected keys) and its `pooler_output` equals timm's features
  exactly (difference 0.0).
- `package-hf` with the draft card (`dev/114_tol_card.json`): README and `predict.py` written,
  `predict.py` ran in an isolated interpreter without torch, timm or mini_trainer; reported **not
  ready: licence missing**, as designed. `predict.py` on a held-out gecko photo: *Phyllodactylus
  tuberculosus* 0.75, then genus 0.87 ... kingdom Animalia 0.99.

## End to end: TL4 (the ToL trunk fine-tuned on moths; 12,632 species, read 384, model 320)

`mt_release export ... --ranks species,genus,family --head-name moths` on 300 images: 200 test-fold
moth photos and 100 trap crops (`dev/114 tl4-images`; the crops are the small, up-sampled images
where pipelines disagreed in 2026-09-29). Pixels identical on 300 / 300; every file 100 % top-1 at
every rank; fp32 max logit difference 1.2e-4 (`trunk.onnx`), 1.5e-4 (`model.onnx`), 2.7e-5 (timm);
fp16 trunk 0.047 with embedding cosine at least 0.99998. CPU: 284 ms per image, 2.9 img/s at batch 16
(the trunk at 320 px; fp16 the same on CPU); the 12,632-class head 1.5 ms. Sizes: `trunk.onnx` 796 MB,
`trunk.safetensors` 785 MB, `heads/moths.safetensors` 87 MB, `model.onnx` 874 MB **in one file**,
`model.safetensors` 872 MB. Export and verification in one process: 10 min, 7.3 GB peak. TL4's
small logit differences (against ToL's 0.046) are consistent with the clamp explanation: the
differences come from the ToL head's clamp, not from the export.

**The trunk guard.** `mt_release add-head --weights tl4_last.pt --release release/tol --name moths`
is refused in 4 s ("different trunk ... export it as its own release"): TL4 fine-tuned the trunk
(0 of its 340 trunk tensors equal tol_full's), so its head cannot run on ToL's trunk. TL8's will.

## Predictions, scored

| prediction | outcome |
|---|---|
| the full graph passes 2 GB, the trunk does not | wrong by a hair: 1,966 MB of weights, just under protobuf's 2 GiB (2,147 MB); `mt_release` keeps external data above 1.8 GB as a margin, so a slightly larger head still exports. The trunk: 797 MB, one file |
| fp32 reproduces mini_trainer to below 1e-3 in logits, 100 % top-1 | top-1 right; the logit bound wrong (0.046), because of asin at the clamp; 5e-5 at the top class |
| fp16 trunk 99 % or more top-1 | right: 100 % at every rank |

## Other formats (one line each)

- **safetensors + a small PyTorch loader**: the release format (owner, 2026-11-19); the trunk loads
  into timm strictly (features identical) and into transformers' `DINOv3ConvNextModel` after a
  rename (identical).
- **ONNX**: the deployment format; ONNX Runtime on CPU and GPU, and the input to the TensorRT,
  OpenVINO and Core ML converters.
- **torch.export (`.pt2`)**: the dynamo ONNX export already traces through it, so it is nearly free;
  it serves only PyTorch runtimes. Not built.
- **ExecuTorch**: PyTorch's on-device runtime, built from `torch.export`; for phones, where an 800 MB
  trunk is not the model. Not built.
- **AOTInductor**: compiles a `torch.export` program into a library for C++ servers without
  Python. Not built.
- **TensorRT**: NVIDIA's fastest path, an engine built from the ONNX file on the target GPU
  (mini_trainer has `dev/benchmarks/inference/tensorrt_build.py`). Not built.
- **OpenVINO**: Intel's runtime, reads ONNX directly; worth a test if users run on Intel CPUs.
- **Core ML**: Apple devices, converted with coremltools; only for an iOS app.
- **LiteRT (TensorFlow Lite)**: Android, converted from PyTorch with ai-edge-torch; only for an
  Android app.

## Commands

Export, verify and package tol_full today (mini_trainer branch `feature/export-tol`; venv with the
`export` and `timm` extras plus safetensors):

```bash
mt_release export --weights data/population_probes/tol_e1/tol_full_last.pt --output release/tol \
    --ranks species,genus,family,order,class,phylum,kingdom --head-name tol \
    --images data/population_probes/tol_e1/emb_known/paths.json
mt_release package-hf --release release/tol --repo-id <user>/<name> --card dev/114_tol_card.json
# then, by the owner only:  hf upload <user>/<name> release/tol . --repo-type model
```

A head trained on the frozen trunk (TL8) joins the same release with
`mt_release add-head --weights tl8.pt --release release/tol --name moths --ranks species,genus,family --images ...`.

## What still blocks a one-command release

1. **The licence** (owner): the trunk derives from DINOv3 weights, whose licence requires
   derivatives to be distributed under it with a copy included (B8 was released that way); the
   training images are GBIF and TreeOfLife-200M sources with mixed licences, many non-commercial
   (77 % of the Lepidoptera set; not measured for tol256). `package-hf` stays "not ready" until
   the card JSON has `license` and the licence file to copy.
2. **The card's claims** are a draft citing RESULTS rows W3, Z19, Z20; the owner should choose them.
3. **Calibration** (temperature, back-off thresholds) is not ported; nothing in the release defends a
   confidence threshold yet.
4. **Display names** for GBIF-key heads (TL4) need a join with the dataset (lepinet's `build_names`).
5. **int8** and GPU measurements are not done (GPU busy; no int8 path in mini_trainer).

## Open questions for the owner

- Ship the combined files? They double the repository (6.7 GB against 3.2 GB) for convenience only.
- Store prototypes in fp16 (head 1.18 -> 0.59 GB; no top-1 change measured)?
- A transformers `from_pretrained` path for the trunk (a rename; the head stays numpy or a small
  module)?
- One repository per model, or one repository per trunk with heads as folders (the format allows both)?

## Commits (mini_trainer, branch `feature/export-tol`)

- `d70e6e9` Keep one copy of the weights resident during ONNX export (`mt_export` on tol_full: failed at 14 GB -> 9.0 GB, 29 s).
- `4cd0a29` Add `mt_release` (`mini_trainer/release/`: build, verify, card, cli, standalone) and `tests/release` (9 tests).
- `9eb4df2` Document `mt_release` (`docs/release.md`; links from `docs/onnx.md` and the README).

Pushed to `origin/feature/export-tol` (`9eb4df2`) on 2026-11-19, as the owner allowed; it sits on the local
`feature/lepinet-integration` head `c361471`, so the push also carries three integration commits not yet on
origin (`1554c9c`, `f3cfe3d`, `c361471`). Not merged: the supervisor arranges the merge with the mt session.
The release folders were built in the session's scratchpad; their `config.json`, `verification.json` and the
generated ToL card are kept in `data/release_checks/release_{tol,tl4}/`.

Checks: `dev/check.sh static` (ruff, format, import contracts) passes; the full suite (the integration
worktree's venv on this tree: 958 passed, 255 skipped, 1 expected failure, 0 failed); `tests/export` and
`tests/release` 33 passed, 2 skipped (optional backends); the native-INT8 export tests need torchao
and CUDA and were not run. `dev/check-wheel.sh` was not run because it calls `uv sync` (into a
disposable environment); `uv build` alone includes `mini_trainer/release/` and the `mt_release`
entry point. The scratch environment: `uv venv` (Python 3.14) under the session's scratchpad with
torch 2.12.0+cpu, timm, onnx 1.23, onnxruntime 1.32, onnxscript, safetensors, transformers 5.23.
