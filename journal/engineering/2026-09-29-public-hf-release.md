# Releasing the best models to the public on Hugging Face

**Kind:** subproject · **Status:** **RESOLVED (2026-09-29).** Both models are public, and the
published files reproduce the training pipeline to within 0.9 pt macro-F1. The prediction below was
**falsified at its 0.5 pt bound and held at its 1 pt line**; the cause is up-sampling small trap crops,
not the export. B3rep5x's softmax turned out 71 % saturated and ships with a temperature (T = 1.70),
which also lifted its marginalised family macro-F1 by 3.4 pt.

Until now the only published weights were the two phone-app bundles in `gmougeot/lepinet-models`
(distilled b0, EfficientNetV2-B2 v1): no model card, app-shaped, and neither is the model the paper
recommends. This entry releases the recommended models for *general* use and records what that took.

## What to release, and why these two

| release | run | why |
|---|---|---|
| `gmougeot/lepinet-bioclip2-vitl14` | **P5** | the paper's "ship this": ties B8 on accuracy, **+17.3 pt useful-answer rate** under a 95 %-precision back-off ([[2026-08-28-two-tied-models-differ-by-17-points-in-deployment]]) |
| `gmougeot/lepinet-effnetv2s` | **B3rep5x** | 37 M params (20 M backbone), probe 0.7706 / held-out 0.7704 -- within ~0.5 pt of P5 (321 M) at 1/9 of the size; runs on a laptop CPU |

**Not released:** B8, A2, F1 (DINOv3 trunks: its licence carries redistribution terms, and B8 is
dominated by P5 on deployability); P5b (P5's repeat; P5 is the draw every deployment number was
measured on). **Licence: CC-BY-NC-4.0** (owner decision): 77 % of the 6.33 M training-set images are CC-BY-NC
or CC-BY-NC-ND, which the weights should not launder into commercial use.

## Format: why ONNX with the post-processing inside

The app bundles are shaped for one consumer. A public release has to work for someone with
`onnxruntime` + `Pillow` and nothing of ours, so the graph (`dev/083_hf_release.py`) carries
everything that is easy to get wrong -- normalisation (including BioCLIP-2's ImageNet->CLIP
re-normalisation), **marginalisation** (`prob_genus`/`prob_family` are sums of species
probabilities, dev/042), and the L2-normalised **embedding** for retrieval / open-set use. Raw
`logits_<level>` keep their app names, so each release folder is also a valid lepinet-app bundle.
`model.safetensors` carries the full-precision weights for fine-tuning with lepinet.

One repo per model (HF convention: own card, licence, download count), grouped in a collection owned
by `insectai-cost` -- that org is a model zoo that *points at* models, which a collection does across
namespaces. The app's `lepinet-models` repo is untouched apart from a README, so no app URL moves.

## Prediction (written before the evaluation ran)

The released ONNX, fed by the model card's own 12-line preprocessing (shorter side -> S, centre
crop, bicubic), reproduces lepinet's own probe macro-F1 **within 0.5 pt** for both models (P5 0.7810,
B3rep5x 0.7706). dev/041 measured the resize kernel as irrelevant and only aspect squashing as
costly, and the card crops rather than squashes. Falsified if either misses by more than 1 pt.

## Published

| | repo | files |
|---|---|---|
| P5 | [`gmougeot/lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14) | `model.onnx` (fp32, 1.29 GB), `model.safetensors`, `taxonomy.json`, `names.json`, `thresholds.json`, `config.json`, `predict.py`, card |
| B3rep5x | [`gmougeot/lepinet-effnetv2s`](https://huggingface.co/gmougeot/lepinet-effnetv2s) | the same, plus `model_fp16.onnx` (109 MB) |
| app store | [`gmougeot/lepinet-models`](https://huggingface.co/gmougeot/lepinet-models) | README only added; bundle paths unchanged |

Grouped in [a collection](https://huggingface.co/collections/gmougeot/lepinet-lepidoptera-identification-6abbc33d250430f8c67db428).
The token in use could not create one under `insectai-cost` (403), so the owner adds the org copy.
The two releases share byte-identical `taxonomy.json` / `names.json`. That needed B3rep5x's coarse
vocab rebuilt from the dataset (a species-only checkpoint carries none) in P5's string-sorted order,
because the parent indices are baked into each graph. The card's quick-start and `predict.py` were run
from a fresh venv holding only `onnxruntime pillow numpy huggingface_hub`, downloading from the Hub.

## Result: does the published file reproduce the pipeline?

Same metric function on both sides (it reproduces lepinet's own numbers exactly from its
`predictions.csv`), species macro-F1:

| | lepinet pipeline | published ONNX + card preprocessing | delta | per-image agreement |
|---|---|---|---|---|
| B3rep5x probe | 0.7706 | 0.7649 | -0.57 | 97.3 % |
| B3rep5x held-out species | 0.7704 | 0.7719 | +0.15 | 94.7 % |
| P5 probe | 0.7810 | 0.7723 | -0.87 | 96.4 % |
| P5 held-out species | 0.7806 | 0.7897 | +0.92 | 94.1 % |

**The graph is not the cause:** PyTorch vs ONNX on identical tensors agree to 3e-5. **The resize
kernel mostly is not either.** On B3rep5x probe: bicubic 0.7649, bilinear 0.7647, and a
re-implementation of fastai's two-stage validation resize 0.7677, which recovers only 0.28 of the
0.57. What is left is that **trap crops are tiny** (median shorter side 157 px; 5th percentile 93),
so every pipeline *up*-samples them, and 3-6 % of individual predictions flip with details of that
resampling. The deltas scatter in both directions and sit inside the probe noise floors (0.0107
P5, 0.0119 frozen 20 M), so there is no bias to correct. The model cards give both columns and warn
that small crops should be resized consistently. dev/041 found the kernel irrelevant because it only
tested *down*-sampling of GBIF photos; up-sampling is the regime it did not cover.

**In-distribution check** of the published files, on the same random 10,000 images of the GBIF test
fold (a sanity check that the card's preprocessing is right for ordinary photos, not a benchmark:
macro-F1 over a sample is not the full-fold number, and P5's sample is partly inside BioCLIP-2's
pre-training):

| | species macro-F1 | top-1 | genus macro-F1 | family macro-F1 |
|---|---|---|---|---|
| P5 | 0.9247 | 94.2 % | 0.9655 | 0.9864 |
| B3rep5x | 0.9096 | 92.3 % | 0.9527 | 0.9742 |

B3rep5x never had an in-distribution number in the project. On photos the card preprocessing is
downsampling, where dev/041 already showed the kernel to be irrelevant. The planned full-fold run of
B3rep5x (~3 h on CPU) was dropped as not worth it for a release check.

## Result: calibration, and why it changed a coarse-rank number

`calibrate` fits one species temperature by NLL on 10,000 (B3rep5x) / 3,000 (P5) validation-fold
images.

| | T | NLL at T=1 -> T | share with p = 1.0 exactly (fp32) |
|---|---|---|---|
| B3rep5x | **1.70** | 0.626 -> 0.484 | **71 % -> 4.7 %** |
| P5 | 1.08 | 0.2513 -> 0.2485 | 0 % -> 0 % |

P5 ships at T = 1 (a 1 % NLL change is not worth a new artefact), which is O1's "P5 is
well calibrated" re-measured on in-distribution data. B3rep5x's ArcFace head saturates float32:
before the temperature its 95 %-precision species threshold landed at exactly 1.00, the same artefact
O1 flagged for B8. The temperature divides the published `logits_species` inside the graph, so
`softmax(logits_species) == prob_species` holds for every consumer, including the app.

**Unplanned finding:** the temperature moved B3rep5x's *marginalised* scores on probe, with species
unchanged: **family macro-F1 0.7635 -> 0.7977 (+3.42)**, genus 0.8135 -> 0.8163. Summing a posterior
depends on calibration, and finding 12 says the margin damages exactly that. So **one scalar fitted on
in-distribution data undoes a large part of the margin's coarse-rank cost.** n = 1 model, so this is a
lead, not a finding. The obvious test is to re-score the A1-vs-baseline marginal comparison (paper
section 4.3) with each model temperature-fitted first.

## Result: thresholds (95 % target, fitted on half the trap nights, verified on the other half)

| | species t | genus t | family t | answered | precision answered | useful |
|---|---|---|---|---|---|---|
| P5 | 0.992 | 0.992 | 0.975 | 86.6 % | 96.2 % | **83.3 %** |
| B3rep5x (T = 1.70) | 0.94 | never | 0.53 | 90.0 % | 95.3 % | **85.8 %** |

Group-split fitting is stricter than O1's in-sample fit (P5 useful 88.4 % there), as it should be.
P5 overshoots the target on the held-out nights (96.2 %). I first blamed a threshold grid too coarse
near 1; refining it moved the threshold only 0.995 -> 0.992, so that explanation was wrong. The
fitting nights are simply harder than the held-out ones: **between-halves variance of about a point
of precision**, which is the honest error bar on any threshold fitted to 7,400 trap images. Fitting
in-sample would hide it. B3rep5x edging P5 on useful rate is inside that variance and is **not** a
ranking; O1's P5-over-B8 claim stands on its own evidence. Shipped thresholds are refitted on all
nights: P5 0.99 / 0.99 / 0.965, B3rep5x 0.93 / never / 0.505.

## Follow-up (same day): int8 for CPUs, fp16 for GPUs, and two onnxruntime bugs

The owner asked whether other precisions would help users. P5 was the case that mattered: on CPU it
ran at 7.8 img/s (127 ms per image), and it is a 1.3 GB download.

**Measured on 1,000 probe images, then on the full sets below** (ORT 1.30; Core Ultra 9 285K with
AVX-VNNI; RTX 5090):

| P5 variant | size | CPU img/s | GPU img/s | agreement with fp32 |
|---|---|---|---|---|
| fp32 | 1285 MB | 7.8 | 371 | - |
| **dynamic int8**, per-channel, MatMul only | **327 MB** | **14.9** | 20 (falls back to CPU) | 97.3 % |
| 8-bit weight-only (MatMulNBits, int8 compute) | 336 MB | 8.8 | 367 | 99.6 % |
| 4-bit weight-only | 266 MB | 7.5 | - | 96.4 % |
| **fp16** (PyTorch half export) | **680 MB** | crashes | **680** | 100.0 % |
| fp16 (converted fp32 graph) | 644 MB | crashes | 559 | 99.9 %, floods buffer-reuse warnings |

(The table is the 1,000-image screen, and at first it read "ship plain dynamic int8". The full sets
said otherwise, as recorded next.)

**Full sets, species macro-F1 delta vs fp32** (probe / held-out species / GBIF 10k):

| file | delta | agreement | verdict |
|---|---|---|---|
| P5 fp16, GPU | 0.00 / 0.00 / -0.01 | 99.8 % | **shipped** (`model_fp16.onnx`) |
| P5 plain dynamic int8 | **-1.60** / -1.18 / -0.37 | 97.3 % | rejected (bar: 0.5 pt) |
| P5 hybrid int8: dynamic int8 + 8-bit weight-only `c_proj` | -0.07 / -0.67 / -0.09 | 98.6 % | **shipped** (`model_int8.onnx`) |
| B3rep5x static QDQ int8, GBIF calibration | -1.01 / +0.09 / -0.02 | 95.5 % | rejected |
| B3rep5x static QDQ, 3:1 trap:GBIF calibration | -0.58 on probe | 95.7 % | rejected |
| B3rep5x static QDQ, depthwise convs fp32 | -0.78 / -1.15 on probe (two op sets) | 95.0-96.4 % | rejected -- **B3rep5x ships no int8** |

**Final speeds** (ORT 1.30, shared server with background load, so ratios are the robust part):

| file | CPU 24 thr, batch 32 | CPU 24 thr, 1 image | CPU 4 thr, 1 image | GPU batch 32 | GPU 1 image |
|---|---|---|---|---|---|
| P5 fp32 | 7.7 img/s | 125 ms | 307 ms | 371 img/s | 4.5 ms |
| P5 int8 | 12.3 img/s (1.6x) | 69 ms (1.8x) | 167 ms (1.8x) | - | - |
| P5 fp16 | - | - | - | 667 img/s (1.8x) | 5.3 ms |
| B3rep5x fp32 | 174 img/s | 10 ms | 18 ms | 2,780 img/s | 3.3 ms |
| B3rep5x fp16 | 107 img/s (slower) | 18 ms | 27 ms | 3,432 img/s (1.23x) | 3.8 ms |

So the shipped matrix is fp32 + fp16 for both models and int8 only for P5. The rule behind it:
**ship a precision where it is faster on the hardware it targets and costs under 0.5 pt**, measured
per architecture. fp16 is a GPU format for both (on CPU it is slower, or crashes for the ViT). int8
passed only for the ViT: its compute is matmuls, whose int8 kernels are fast and whose error could be
confined to one layer type. The CNN's int8 error is spread across its depthwise and
squeeze-excite structure. `dev/083 quantize --arch vit` reproduces the shipped P5 file byte for byte.

**Where the int8 error lives differs by architecture, and that is why the two models got different
recipes.** In P5 it is concentrated in the 24 MLP output projections: they read post-GELU
activations with outliers that per-tensor int8 cannot represent. Keeping only those fp32 raises
agreement from 97.3 % to 98.9 %. 8-bit weight-only quantization keeps them float at run time and
the file at 331 MB (1.44x faster at batch 32, 1.65x on one image). So the outlier hypothesis I
first used to explain the 0 % result was *also* right, just for the wrong bug and at a
thousandth of the size. For B3rep5x, *dynamic* int8 is simply the wrong tool: ConvInteger has no
fast kernel and ran 4x slower. *Static* QDQ runs 2.2x faster (374 vs 173 img/s). Its error does not
come from calibration data: trap-aware calibration moved agreement 95.5 -> 95.7 %. It sits in the
architecture (depthwise convs, squeeze-excite gates).

**Minimum onnxruntime:** every file loads in ORT 1.20 except P5's int8, which needs **1.22**
(8-bit MatMulNBits; 1.20-1.21 support only 4-bit). `predict.py` says so instead of crashing.

**Bug 1, the quantizer.** Every dynamic-int8 variant first scored **0 % agreement**, on both models.
I read that as CLIP's activation outliers crushing per-tensor quantization. **This was a genuine
mistake**, and a test that took the words literally exposed it. A quantization that touched *only*
B3rep5x's final classifier MatMul still destroyed the *embedding*, which is computed upstream of it
(cosine 0.04 with fp32), and so did a no-op quantization. Cause: ORT's quantizer pre-processing
rewrites `Gemm` as `MatMul + Add` and ignores `transB=1`. All our Gemms are square (1024^2 attention
out-projections, 1280^2 bottlenecks), so the untransposed weight type-checks and computes garbage
silently. `dev/083 quantize` does the rewrite itself first (`gemm_to_matmul`), and int8 then agrees
97 %. Same lesson as CLAUDE.md section 4: suspect the harness before the model.

**Bug 2, fp16 on CPU.** Any fp16 P5 graph segfaults at *default* CPU session creation in ORT 1.27
and 1.30. Bisected to `NchwcTransformer`, the x86 blocked-layout pass that only runs at
`ORT_ENABLE_ALL`. It crashes even after the patch-embedding conv is rewritten as reshape + Linear
(`PatchLinear`, kept in the export anyway), so the trigger is not the Conv itself. Not bisected
further: fp16 is not faster on CPU, and on CUDA the pass never runs. The file ships as GPU-only,
with the workaround (`ORT_ENABLE_EXTENDED`) in the card and applied automatically by `predict.py` and
`dev/083`.

**Found by session lepinet-19 (GPU):** `predict.py` passed `ort.get_available_providers()`, which
with onnxruntime-gpu puts TensorRT first. Without TensorRT, ORT falls back to *CPU* silently because
the pip CUDA libraries are only loaded by `ort.preload_dlls()`. Fixed in `predict.py` and in both
cards. That session also established that ORT's CUDA provider works on this box although
`nvidia-smi` reports an NVML driver mismatch, which is what made the GPU measurements possible.

## Follow-up 2 (owner): transformers support for P5, and B8 as a third release

**transformers.** P5 now loads with `pipeline("image-classification", ..., trust_remote_code=True)`
and `AutoModelForImageClassification`. The design point is **one weights file**:
`modeling_lepinet.py` (torch only, no open_clip, no lepinet) mirrors the lepinet parameter names,
with an open_clip ViT under `0.visual` and the cosine head under `1.head`. So the same
`model.safetensors` serves lepinet, transformers and the ONNX export, and the repo does not carry a
second 1.3 GB copy. `config.json` is shared too: transformers keeps unknown keys, so the ONNX / app
keys survive next to `model_type` and `id2label` (species names). Measured: 306/306 weights load, no
missing or unexpected keys; identical inputs give 100 % top-1 agreement with `model.onnx` (max logit
difference 3e-4). Through `CLIPImageProcessor`'s own preprocessing, 99.6 % of 1,000 probe images
agree with the ONNX quick start (tiny crops again). Verified from the Hub in a fresh
`transformers torch torchvision pillow` venv. Two traps, both worked around:
- transformers 5 builds models on the meta device and leaves **non-persistent buffers
  uninitialised**. The genus/family index buffers came out as garbage (an index error, luckily
  rather than silent), so they are built from the config at first use instead.
- transformers refuses a safetensors file without `format: pt` metadata, which lepinet's
  `save_file` did not write. Re-saved with the same tensors (`dev/083 transformers`).

Only P5 has this. B3rep5x and B8 would need their own modeling code (torchvision EfficientNet, timm
ConvNeXt), and the ONNX route already covers them.

**B8 (DINOv3 ConvNeXt-L, 217 M) -> `gmougeot/lepinet-dinov3-convnextl`.** The owner asked for a
mid-size model. Published under the **DINOv3 License**, because its section 1.b.i allows
derivatives to be distributed only under that agreement, with a copy included. The card adds a
separate, clearly labelled request for non-commercial use, which concerns the training data (owner's
decision). The export is species-only: `marginal_arcface` computes genus/family as species marginals
in the checkpoint's own label order, so the graph's shared-taxonomy sums carry the same
information and keep all three releases interchangeable.

| | probe | held-out species | GBIF 10k | GPU fp32 / fp16 img/s |
|---|---|---|---|---|
| B8 ONNX (fp16 identical to 1e-4) | **0.7766** | 0.7708 | **0.9261** | 378 / 715 |
| P5 ONNX | 0.7723 | **0.7897** | 0.9247 | 371 / 667 |

**O1 revisited.** B8 is badly over-confident: 81 % of validation images saturate at p = 1.0 in fp32,
and the fitted T = 1.91 cuts NLL 0.567 -> 0.387. With calibrated probabilities and the split-fitted
95 % policy, B8 answers correctly on **77.2 %** of trap images against P5's **83.3 %** and B3rep5x's
85.8 %. O1 reported 71.2 % vs 88.4 %, but with uncalibrated probabilities *and* a different fitting
procedure (dev/058), under which P5 itself scores 5 points higher. So the two gaps are not
comparable, and I do not attribute a share of O1's 17 points to calibration. What survives:
**calibrated and measured one way, P5 still leads B8 by about 6 points on useful answers**, and
O1's recommendation stands.

**B8 precisions.** fp16 is identical to fp32 (to 1e-4 on all three sets; GPU 715 vs 378 img/s) and,
unlike the ViT, loads in a default CPU session. The hybrid int8 recipe transfers once `mlp/fc2`
stands in for `c_proj`: -0.18 / +0.33 / +0.05 pt, 1.46x faster per image, 245 MB. `dev/083
quantize --arch vit` reproduces it byte for byte.

## Follow-up 3 (owner): using the open-set behaviour, measured

The owner asked whether the cards show how to get a genus / family / "unknown" answer. They showed
the rule as pseudo-code only. Now every card has a runnable back-off + entropy snippet, run verbatim
from the published card, and P5's transformers `predict()` returns the same `answer` and
`novelty` (thresholds stored in `config.json`). The claims are measured on the published fp32 files:
3,171 GBIF test-fold photos of **591 species outside the label set** (the species below the training
floor; O1's novel set) against the 10,000-photo known sample.

| | claims a species on unseen species (always wrong) | genus (correct) | family (correct) | unknown | known photos: species answered (precision) | AUROC, entropy |
|---|---|---|---|---|---|---|
| **P5** | **13.7 %** | 8.6 % (87 %) | 26.2 % (98 %) | 51.5 % | 80.6 % (99.1 %) | 0.915 |
| B8 | 20.9 % | 0.9 % (79 %) | 2.0 % (100 %) | 76.3 % | 91.9 % (97.6 %) | 0.917 |
| B3rep5x | 25.2 % | 0 % | 30.4 % (93 %) | 44.4 % | 88.6 % (97.7 %) | 0.908 |

Three readings. (1) The entropy AUROC barely separates the models (0.908-0.917), yet the *policy*
outcome does: P5 commits to a wrong species half as often as B3rep5x. As in O1, the ranking metric and
the deployment metric disagree, and the deployment one is the one a user meets. (2) The genus rung is
almost unused by B8 and B3rep5x, for different reasons. B3rep5x's genus threshold never reaches
95 % conditional precision on trap data, so it is disabled (> 1). B8's is 0.945, but when its species
confidence fails, its genus confidence rarely clears that bar either. P5 is the only model that often
says "*Colias*, species unsure". (3) These
thresholds were fitted on trap images and are conservative on photos (97.6-99.1 % precision against a
95 % target). A photo-domain fit would answer more, which is a cheap improvement for the app.
