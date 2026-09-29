---
license: other
license_name: dinov3-license
license_link: LICENSE.md
library_name: onnx
pipeline_tag: image-classification
base_model: timm/convnext_large.dinov3_lvd1689m
tags:
  - biology
  - lepidoptera
  - moths
  - butterflies
  - insects
  - species-classification
  - hierarchical-classification
  - taxonomy
  - gbif
  - camera-trap
  - onnx
  - convnext
  - dinov3
---

# lepinet · DINOv3 ConvNeXt-L: moth and butterfly identification

Identifies **Lepidoptera** (moths and butterflies) from a photo at three ranks at once: **species**
(12,041), **genus** (4,333) and **family** (102). The three answers are coherent, because genus and
family are computed from the species probabilities.

This is the **mid-size** model of the [lepinet](https://github.com/GuillaumeMougeot/lepinet)
project: a DINOv3 ConvNeXt-L (217 M parameters) that is as accurate as the recommended
[`gmougeot/lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14)
(321 M), and the most accurate of the three on ordinary GBIF photos. Its weakness is confidence: at
a 95 % precision target it gives a correct answer on {{B8_USEFUL}} of light-trap images, against
{{P5_USEFUL}} for the BioCLIP-2 model. **If you rely on the confidence thresholds, prefer the BioCLIP-2
model**; if you use the top prediction, or want the smaller download, this one is as good.

| | |
|---|---|
| **Architecture** | [DINOv3](https://github.com/facebookresearch/dinov3) ConvNeXt-L backbone (timm `convnext_large.dinov3_lvd1689m`), fine-tuned end to end, plus a 1536-d cosine classifier; 217 M parameters |
| **Input** | one RGB image of **one** insect, 320×320 |
| **Output** | species, genus and family probabilities, calibrated species logits, a 1536-d embedding |
| **Format** | ONNX: fp32 (869 MB), fp16 for GPUs (476 MB){{INT8_FORMAT}}. Runs with `onnxruntime` alone |
| **Licence** | [DINOv3 License](https://huggingface.co/gmougeot/lepinet-dinov3-convnextl/blob/main/LICENSE.md), plus a non-commercial request for the training data (see [Licence](#licence)) |
| **Other sizes** | [`lepinet-effnetv2s`](https://huggingface.co/gmougeot/lepinet-effnetv2s) (37 M, fast on CPU) · [`lepinet-dinov3-convnextl`](https://huggingface.co/gmougeot/lepinet-dinov3-convnextl) (217 M) · [`lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14) (321 M, recommended); all in [one collection](https://huggingface.co/collections/gmougeot/lepinet-lepidoptera-identification-6abbc33d250430f8c67db428) |

## Quick start

```bash
pip install onnxruntime pillow numpy huggingface_hub
```

```python
import json, numpy as np, onnxruntime as ort
from huggingface_hub import snapshot_download
from PIL import Image

repo = snapshot_download("gmougeot/lepinet-dinov3-convnextl", allow_patterns=["model.onnx", "*.json"])
session = ort.InferenceSession(f"{repo}/model.onnx")
names = json.load(open(f"{repo}/names.json"))["names"]

def load(path, size=320):  # shorter side -> size, centre crop, RGB in [0, 1]
    img = Image.open(path).convert("RGB")
    s = size / min(img.size)
    img = img.resize((max(size, round(img.width * s)), max(size, round(img.height * s))), Image.BICUBIC)
    l, t = (img.width - size) // 2, (img.height - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)), np.float32).transpose(2, 0, 1)[None] / 255

out = dict(zip([o.name for o in session.get_outputs()], session.run(None, {"image": load("moth.jpg")})))
for rank in ("species", "genus", "family"):
    p = out[f"prob_{rank}"][0]
    print(f"{rank:8s} {names[rank][p.argmax()]}  {p.max():.2f}")
```

Or use the bundled command-line script, which applies the confidence thresholds for you:

```bash
hf download gmougeot/lepinet-dinov3-convnextl model.onnx predict.py config.json taxonomy.json names.json thresholds.json --local-dir lepinet
python lepinet/predict.py moth1.jpg moth2.jpg --top 3
```

The three lepinet releases share **identical** `taxonomy.json` and `names.json` files, so they are
drop-in interchangeable. Batching works: the input's first axis is dynamic.

## Which file to use

| file | use it on | size | batch of 32 | one image |
|---|---|---|---|---|
| `model.onnx` (fp32) | anywhere | 869 MB | CPU {{CPU32_B32}} · GPU 378 img/s | CPU {{CPU32_B1}} ms · GPU 3.8 ms |
| `model_fp16.onnx` | **NVIDIA GPU** | 476 MB | GPU **715** img/s | GPU 2.9 ms |
{{INT8_ROW}}
Measured with onnxruntime 1.30 on an RTX 5090 and a 24-core Intel Core Ultra 9 285K (a shared server
under some background load, so read the ratios rather than the absolute CPU numbers). fp16 matches
fp32: the same species macro-F1 to 0.0001 on all three evaluation sets. {{INT8_NOTE}}

### On a GPU

```bash
pip install "onnxruntime-gpu[cuda,cudnn]"   # instead of onnxruntime; do not install both
```

```python
ort.preload_dlls()   # loads the CUDA/cuDNN libraries that pip installed
session = ort.InferenceSession(f"{repo}/model_fp16.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

Name the providers explicitly. `ort.get_available_providers()` puts TensorRT first, and without
TensorRT installed onnxruntime then falls back to the CPU without saying so. `predict.py` picks CUDA
automatically.

## Files

| file | what it is |
|---|---|
| `model.onnx`, `model_fp16.onnx`{{INT8_FILE}} | the network. Input `image`: float32 `[N, 3, 320, 320]`, RGB, values in [0, 1]. **Normalisation is inside the graph** (do not normalise yourself) |
| `taxonomy.json` | `vocabs.<rank>[i]` is the [GBIF](https://www.gbif.org) taxon key of output index `i`; `parents` maps species to genus and genus to family |
| `names.json` | scientific names, aligned index for index with `taxonomy.json` |
| `thresholds.json` | per-rank confidence thresholds for a 95 %-precision back-off policy, with the precision and coverage they achieve on held-out data |
| `config.json` | preprocessing and output description (it doubles as a [lepinet-app](https://github.com/GuillaumeMougeot/lepinet-app) bundle manifest) |
| `predict.py` | a standalone CLI and `Lepinet` class (onnxruntime + Pillow only) |
| `model.safetensors` | full-precision PyTorch weights, for fine-tuning with lepinet |
| `LICENSE.md` | the DINOv3 License, which these weights are distributed under |

### Outputs

| name | shape | meaning |
|---|---|---|
| `prob_species` | `[N, 12041]` | softmax over species (temperature-calibrated) |
| `prob_genus` | `[N, 4333]` | sum of each genus's species probabilities |
| `prob_family` | `[N, 102]` | sum of each family's genus probabilities |
| `logits_species` | `[N, 12041]` | species scores; `softmax(logits_species) == prob_species` |
| `embedding` | `[N, 1536]` | L2-normalised image embedding (cosine similarity = dot product) |

**Calibration.** The raw model is strongly over-confident: 81 % of validation images get a species
probability of exactly 1.0 in float32. The graph divides the species logits by a temperature
**T = 1.91**, fitted by likelihood on 10,000 in-distribution validation images (negative
log-likelihood 0.567 → 0.387, saturated predictions 81 % → 0 %). Rankings, and so accuracy, are
unchanged.

## Using the confidence

Every image gets a species prediction, including images of species outside the label set and
images that are not moths at all. `thresholds.json` implements the back-off policy the project
evaluates:

```
answer species  if prob_species.max() >= t_species
else genus      if prob_genus.max()   >= t_genus
else family     if prob_family.max()  >= t_family
else "unknown"
```

In code, continuing the quick start above (`out`, `names` and `repo` come from there):

```python
thresholds = {k: v["threshold"] for k, v in json.load(open(f"{repo}/thresholds.json"))["levels"].items()}

def answer(out, i=0):
    """Deepest rank the model is confident about for image i, or None ("unknown")."""
    for rank in ("species", "genus", "family"):
        p = out[f"prob_{rank}"][i]
        if p.max() >= thresholds[rank]:          # a threshold above 1 means "never answer at this rank"
            return rank, names[rank][p.argmax()], float(p.max())
    return None

print(answer(out))    # ('species', 'Macaria notata', 0.99), ('family', 'Erebidae', 0.97) or None

p = out["prob_species"][0]
novelty = float(-(p * np.log(p + 1e-12)).sum())   # entropy: higher = less familiar image
```

`predict.py` does exactly this.

The thresholds were fitted on half of the light-trap capture nights and verified on the other half:

{{THRESHOLDS_TABLE}}

### Species the model has never seen

Any closed-set classifier gives a species to a species it does not know. What the back-off buys is
that it usually does not *commit* to one. Measured with the file above on **3,171 GBIF photos of 591
species outside the label set** (the test-fold species below the training floor):

{{NOVEL_BLOCK}}

"Unknown" is not proof of a novel species, and a species answer is not proof that the species is in
the label set: the same rule catches hard photos of known species. Use `novelty` for a graded
score, for instance to send the most unfamiliar images to an expert.

With the same protocol the BioCLIP-2 model answers correctly on {{P5_USEFUL}} and the small
EfficientNetV2-S model on {{B3_USEFUL}}. An earlier study used this model's *uncalibrated*
probabilities and a different fitting protocol, and found a 17-point gap to the BioCLIP-2 model.
Calibrated and measured as above, the gap is about 6 points: smaller, but real.

## Evaluation

Species **macro-F1** averages F1 over species, so the rare species count as much as the common ones.

- **GBIF test fold (in-distribution):** held-out images from the same GBIF sources as training. The
  published file was checked on a random 10,000-image sample (macro-F1 over a sample is not the
  full-fold number). Whether DINOv3's LVD-1689M pre-training images overlap this fold has not been
  checked.
- **Probe:** 15,200 light-trap images of 368 Danish species, from (trap, night) groups never used in
  training. This is a real domain shift: night-time camera crops, not curated photos.
- **Probe, held-out species:** 2,455 images of 58 species for which no trap images were used in any
  form during training.

| evaluation | training pipeline | **`model.onnx`** + quick-start preprocessing | `model_fp16.onnx` |{{INT8_EVAL_HEAD}}
|---|---|---|---|{{INT8_EVAL_SEP}}
| GBIF test fold, full (629,742 images) | 0.9060 | not re-run | |{{INT8_EVAL_FULL}}
| GBIF test fold, random 10,000 images | not measured | 0.9261 | 0.9262 |{{INT8_EVAL_GBIF}}
| Probe (light traps) | 0.7798 | 0.7766 | 0.7765 |{{INT8_EVAL_PROBE}}
| Probe, held-out species | 0.7816 | 0.7708 | 0.7707 |{{INT8_EVAL_HO}}

All numbers are species macro-F1. The training-pipeline and ONNX columns differ by up to 1.1 points
on the smallest set. The network is numerically identical to the PyTorch model; the difference is
image resizing of small, up-sampled trap crops (median shorter side 157 px).

## Training

1. **Data:** 6.3 M GBIF occurrence images of Lepidoptera; species with at least 50 images; at most
   about 2,000 images per species; a held-out test fold. No class-balanced oversampling.
2. **Model:** the DINOv3 ConvNeXt-L backbone (self-supervised on LVD-1689M, distilled from DINOv3
   ViT-7B) with a single species cosine classifier. It is trained with an ArcFace margin (0.3) and
   with **marginal supervision**: genus and family losses on the summed species probabilities, so
   there are no genus/family parameters. 6 epochs, 320 px, Muon optimiser, one-cycle schedule.
3. **Trap robustness:** domain augmentation that mimics light-trap imagery, plus self-training:
   about 2 % of each epoch is unlabelled light-trap crops, pseudo-labelled by an earlier model. The
   trap nights used for evaluation are excluded.

Full recipe and every number: [the lepinet repository](https://github.com/GuillaumeMougeot/lepinet),
starting at `START-HERE.md`. The configuration is `configs/20260804_B8_best_at_2pct_dinov3cnxl.yaml`.

## Limitations

- **One insect per image.** The model classifies; it does not detect. Run a detector first on
  scenes, trap screens or multi-insect photos (for example [flat-bug](https://github.com/darsa-group/flat-bug))
  and classify the crops.
- **Confidence is its weak point**, as described above. Use the thresholds shipped here, not a
  fixed 0.5, and re-fit them on your own data if precision matters.
- **Closed label set, uneven geography.** It knows 12,041 species. Training images are 36 % North
  American, 33 % European and 15 % Asian, but only 6 % South American or African. A species outside
  the label set still receives one of these labels.
- **Domain shift is large.** Compare the in-distribution and probe rows. The thresholds come from
  Danish light traps and may not transfer to your camera or region.
- **Adults only, in practice.** Larvae are about 1 % of the training images.
- **Not for decisions** about conservation status, pest control or toxicity without expert
  verification.

## Licence

**The weights are distributed under the [DINOv3 License](https://huggingface.co/gmougeot/lepinet-dinov3-convnextl/blob/main/LICENSE.md)**, because they are a
derivative of Meta's DINOv3 weights, and that licence requires derivatives to be distributed under
its own terms, with a copy included. Among other things it requires you to acknowledge DINOv3 in
publications, to comply with trade controls, and not to use the model for military, weapons or
other end uses it prohibits.

**Request about the training data.** About 77 % of the 6.3 M training images are licensed CC-BY-NC
by their GBIF contributors. The two other lepinet releases are CC-BY-NC-4.0 for that reason. We ask
that you use this model for non-commercial purposes too. This request is ours, about the data; it
is not a term of the DINOv3 License.

## Citation

No paper yet. Please cite the repository, and DINOv3:

```bibtex
@software{mougeot_lepinet_2026,
  author = {Mougeot, Guillaume},
  title  = {lepinet: hierarchical Lepidoptera identification that knows what it does not know},
  year   = {2026},
  url    = {https://github.com/GuillaumeMougeot/lepinet}
}
@article{simeoni2025dinov3,
  title   = {DINOv3},
  author  = {Sim{\'e}oni, Oriane and others},
  journal = {arXiv preprint arXiv:2508.10104},
  year    = {2025}
}
```

Training data: GBIF.org (15 May 2025) GBIF Occurrence Download
[https://doi.org/10.15468/dl.hg37y9](https://doi.org/10.15468/dl.hg37y9).
