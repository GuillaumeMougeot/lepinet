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
