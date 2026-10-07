---
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: image-classification
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
  - efficientnet
---

# lepinet · EfficientNetV2-S: moth and butterfly identification, small and fast

Identifies **Lepidoptera** (moths and butterflies) from a photo at three ranks at once: **species**
(12,041), **genus** (4,333) and **family** (102). The three answers are coherent, because genus and
family are computed from the species probabilities.

This is the **small** model of the [lepinet](https://github.com/GuillaumeMougeot/lepinet) project:
37 M parameters (a 20 M backbone plus the 12,041-class classifier), fast enough for a laptop CPU or an
edge device. It is about 2 points of species macro-F1 behind the two larger released models (0.765 vs 0.777 and
0.783 on light-trap images; 0.899 vs 0.906 and 0.921 on ordinary photos), and close to the best of
them under a back-off policy; see [Which lepinet model?](#which-lepinet-model). Use a large one when
accuracy matters more than speed.

| | |
|---|---|
| **Architecture** | EfficientNetV2-S (torchvision, ImageNet-initialised) plus a 1280-d cosine classifier trained with an ArcFace margin |
| **Input** | one RGB image of **one** insect, 256×256 |
| **Output** | species, genus and family probabilities, calibrated species logits, a 1280-d embedding |
| **Format** | ONNX fp32 (149 MB) or fp16 (109 MB, same predictions). Runs with `onnxruntime` alone |
| **Licence** | CC-BY-NC-4.0 (non-commercial; see [Licence](#licence)) |
| **Other sizes** | [`lepinet-effnetv2s`](https://huggingface.co/gmougeot/lepinet-effnetv2s) (37 M, fast on CPU) · [`lepinet-dinov3-convnextl`](https://huggingface.co/gmougeot/lepinet-dinov3-convnextl) (217 M) · [`lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14) (321 M); all in [one collection](https://huggingface.co/collections/gmougeot/lepinet-lepidoptera-identification-6abbc33d250430f8c67db428) |

## Which lepinet model?

The three released models are about equally accurate on light-trap images (species macro-F1 0.783,
0.777 and 0.765) but answer differently when they are allowed to back off to genus or family. Under
the same 95 %-precision back-off policy, with thresholds fitted on half of the held-out light-trap
nights and measured on the other half:

| model | size | correct at species | correct at any rank | no answer |
|---|---|---|---|---|
| [`lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14) | 321 M | 67.4 % | **86.4 %** | 10.1 % |
| [`lepinet-dinov3-convnextl`](https://huggingface.co/gmougeot/lepinet-dinov3-convnextl) | 217 M | **74.7 %** | 77.2 % | 19.0 % |
| [`lepinet-effnetv2s`](https://huggingface.co/gmougeot/lepinet-effnetv2s) | 37 M | 72.9 % | 85.8 % | 10.0 % |

- To get the **species name** as often as possible, the DINOv3 model names it most often at 95 %
  precision, and answers "unknown" more often.
- If a **genus or family** answer is useful, the BioCLIP-2 model gives some correct answer most
  often, because it backs off more.
- The **small** model is close to the best on both and runs on a laptop CPU.
- On **ordinary photos** the BioCLIP-2 model is the most accurate (species macro-F1 0.921, against
  0.906 and 0.899 over the full GBIF test fold).

Differences of about a point are within the variation between the two halves of the nights. An
earlier comparison reported a 17-point advantage for the BioCLIP-2 model; it counted a family answer
like a species answer, fitted thresholds in-sample and used the DINOv3 model's uncalibrated
probabilities ([analysis](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-10-02-is-the-deployment-gap-a-readout-artefact.md)).

## Quick start

```bash
pip install onnxruntime pillow numpy huggingface_hub
```

```python
import json, numpy as np, onnxruntime as ort
from huggingface_hub import snapshot_download
from PIL import Image

repo = snapshot_download("gmougeot/lepinet-effnetv2s", allow_patterns=["model.onnx", "*.json"])
session = ort.InferenceSession(f"{repo}/model.onnx")
names = json.load(open(f"{repo}/names.json"))["names"]

def load(path, size=256):  # shorter side -> size, centre crop, RGB in [0, 1]
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
hf download gmougeot/lepinet-effnetv2s model.onnx predict.py config.json taxonomy.json names.json thresholds.json --local-dir lepinet
python lepinet/predict.py moth1.jpg moth2.jpg --top 3
```

### Which file to use

| file | use it on | size | batch of 32 | one image |
|---|---|---|---|---|
| `model.onnx` (fp32) | **CPU**, and anywhere | 149 MB | CPU 174 img/s (4 threads: 61) · GPU 2,780 img/s | CPU 10 ms (4 threads: 18) · GPU 3.3 ms |
| `model_fp16.onnx` | **NVIDIA GPU** | 109 MB | GPU **3,430 img/s** · CPU 107 img/s (slower than fp32) | GPU 3.8 ms |

Measured with onnxruntime 1.30 on an RTX 5090 and a 24-core Intel Core Ultra 9 285K (a shared server
under some background load, so read the ratios rather than the absolute CPU numbers). fp16 keeps the
classifier in fp32 and agrees with fp32 on {{FP16_AGREE}} of species predictions on the probe set; on
the GPU, predictions match the CPU on every image tested.

**Why there is no int8 file.** Plain dynamic int8 made this network 4× *slower*, because
onnxruntime has no fast kernel for its int8 convolutions. Calibrated static int8 does run 2.2×
faster (374 img/s), but it lost 0.6–1.2 points of species macro-F1 on light-trap images in every
variant tried: different calibration data, keeping the depthwise convolutions in fp32. At 10 ms per
image on a CPU, that trade is not worth it. The large model, for which it is, has an int8 file.

### Running on a GPU

```bash
pip install "onnxruntime-gpu[cuda,cudnn]"   # instead of onnxruntime; do not install both
```

```python
ort.preload_dlls()   # loads the CUDA/cuDNN libraries that pip installed
session = ort.InferenceSession(f"{repo}/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

Name the providers explicitly. `ort.get_available_providers()` puts TensorRT first, and without
TensorRT installed onnxruntime then falls back to the CPU without saying so. `predict.py` picks CUDA
automatically.

The three lepinet releases share **identical** `taxonomy.json` and `names.json` files, so they are
drop-in interchangeable.

## Files

| file | what it is |
|---|---|
| `model.onnx`, `model_fp16.onnx` | the network. Input `image`: float32 `[N, 3, 256, 256]`, RGB, values in [0, 1]. **Normalisation is inside the graph** (do not normalise yourself) |
| `taxonomy.json` | `vocabs.<rank>[i]` is the [GBIF](https://www.gbif.org) taxon key of output index `i`; `parents` maps species to genus and genus to family |
| `names.json` | scientific names, aligned index for index with `taxonomy.json` |
| `thresholds.json` | per-rank confidence thresholds for a 95 %-precision back-off policy, with the precision and coverage they achieve on held-out data |
| `config.json` | preprocessing and output description (it doubles as a [lepinet-app](https://github.com/GuillaumeMougeot/lepinet-app) bundle manifest) |
| `predict.py` | a standalone CLI and `Lepinet` class (onnxruntime + Pillow only) |
| `model.safetensors` | full-precision PyTorch weights, for fine-tuning with lepinet |

### Outputs

| name | shape | meaning |
|---|---|---|
| `prob_species` | `[N, 12041]` | softmax over species (temperature-calibrated, see below) |
| `prob_genus` | `[N, 4333]` | sum of each genus's species probabilities |
| `prob_family` | `[N, 102]` | sum of each family's genus probabilities |
| `logits_species` | `[N, 12041]` | species scores; `softmax(logits_species) == prob_species` |
| `embedding` | `[N, 1280]` | L2-normalised image embedding (cosine similarity = dot product), for retrieval, clustering or novelty detection |

The model has a single species head; genus and family exist only as sums over species.

**Calibration.** The ArcFace margin makes the raw softmax over-confident: 71 % of validation images
would get a species probability of exactly 1.0 in float32. The graph therefore divides the species
logits by a temperature **T = {{T}}**, fitted by likelihood on 10,000 in-distribution validation
images. This lowers the negative log-likelihood from {{NLL1}} to {{NLLT}}, and the share of
saturated predictions to {{SATT}}. Rankings, and so accuracy, are unchanged.

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

## Evaluation

Species **macro-F1** averages F1 over species, so the rare species count as much as the common ones.

- **GBIF test fold (in-distribution):** held-out images from the same GBIF sources as training.
  The full fold is 629,742 images of all 12,041 species; the fp16 file was checked on a random
  10,000-image sample.
- **Probe:** 15,200 light-trap images of 368 Danish species, from (trap, night) groups never used in
  training. This is a real domain shift: night-time camera crops, not curated photos.
- **Probe, held-out species:** 2,455 images of 58 species for which no trap images were used in any
  form during training.

| evaluation | species macro-F1, training pipeline | species macro-F1, **this ONNX file** + the quick-start preprocessing |
|---|---|---|
| GBIF test fold, full (629,742 images) | not measured | **0.8985** (top-1 92.5 %; genus 0.950, family 0.968) |
| GBIF test fold, random 10,000 images | not measured | 0.9096 |
| Probe (light traps) | 0.7706 | 0.7649 (top-1 82.9 %) |
| Probe, held-out species | 0.7704 | 0.7719 |

Genus and family macro-F1 on probe, from the summed probabilities: **0.816** and **0.798**.

The two columns differ by under 1 point, which is within the run-to-run noise of these evaluation
sets. The network is numerically identical to the PyTorch model; the difference is image resizing.
Trap crops are small (median shorter side 157 px), so they are *up*-sampled, and the resampling
details change 3–5 % of individual predictions. **For small crops, resize consistently.**

For comparison, the project's EfficientNetV2-S baseline, trained without the trap-domain steps, scores 0.6270 on probe.

## Training

1. **Data:** 6.3 M GBIF occurrence images of Lepidoptera; species with at least 50 images; at most
   about 2,000 images per species; a held-out test fold. Square-root class-balanced oversampling.
2. **Model:** EfficientNetV2-S with a cosine classifier trained under an ArcFace angular margin (0.3,
   z-score logits); species head only. 5 epochs, end to end, Muon optimiser, one-cycle schedule.
3. **Trap robustness, both at training time:**
   - *domain augmentation* that mimics light-trap imagery (blur, low light, JPEG);
   - *self-training*: about 2 % of each epoch is unlabelled light-trap crops, pseudo-labelled by an
     earlier model. This is the single largest robustness gain in the project. The trap nights used
     for evaluation are excluded.

Full recipe, ablations and every number: [the lepinet repository](https://github.com/GuillaumeMougeot/lepinet),
starting at its README. The configuration is `configs/20260804_B3rep5x_effnetv2s.yaml`.

## Limitations

- **One insect per image.** The model classifies; it does not detect. Run a detector first on
  scenes, trap screens or multi-insect photos (for example [flat-bug](https://github.com/darsa-group/flat-bug))
  and classify the crops.
- **Closed label set, uneven geography.** It knows 12,041 species. Training images are 36 % North
  American, 33 % European and 15 % Asian, but only 6 % South American or African, as GBIF coverage
  is. A species outside the label set still receives one of these labels: use the thresholds, and
  expect the tropics to be served worse.
- **Domain shift is large:** compare the in-distribution and probe rows above. The thresholds come
  from Danish light traps and may not transfer to your camera or region. Re-fit them on a few
  hundred labelled images of your own if precision matters.
- **Adults only, in practice.** Larvae are about 1 % of the training images; caterpillars, eggs and
  pupae will be poorly identified.
- **Look-alike species** that need genitalia dissection or DNA cannot be separated from a photo.
  Treat the genus answer as the honest one there.
- **Not for decisions** about conservation status, pest control or toxicity without expert
  verification.

## Licence

**CC-BY-NC-4.0.** About 77 % of the training images are licensed CC-BY-NC by their GBIF
contributors, so the weights are released for non-commercial use. The lepinet training code is
GPL-3.0.

## Citation

No paper yet. Please cite the repository:

```bibtex
@software{mougeot_lepinet_2026,
  author = {Mougeot, Guillaume},
  title  = {lepinet: hierarchical Lepidoptera identification that knows what it does not know},
  year   = {2026},
  url    = {https://github.com/GuillaumeMougeot/lepinet}
}
```

Please also acknowledge the GBIF contributors whose images made this possible: GBIF.org (15 May
2025) GBIF Occurrence Download [https://doi.org/10.15468/dl.hg37y9](https://doi.org/10.15468/dl.hg37y9).
