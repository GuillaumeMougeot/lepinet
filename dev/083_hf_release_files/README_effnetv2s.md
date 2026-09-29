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
edge device. It is about half a point behind the
project's best model ([`gmougeot/lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14),
9× larger) on images from automated light traps. Use the large one when accuracy matters more than
speed.

| | |
|---|---|
| **Architecture** | EfficientNetV2-S (torchvision, ImageNet-initialised) plus a 1280-d cosine classifier trained with an ArcFace margin |
| **Input** | one RGB image of **one** insect, 256×256 |
| **Output** | species, genus and family probabilities, calibrated species logits, a 1280-d embedding |
| **Format** | ONNX fp32 (149 MB) or fp16 (109 MB, same predictions). Runs with `onnxruntime` alone |
| **Licence** | CC-BY-NC-4.0 (non-commercial; see [Licence](#licence)) |

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

The two lepinet releases share **identical** `taxonomy.json` and `names.json` files, so they are
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

The thresholds were fitted on half of the light-trap capture nights and verified on the other half:

{{THRESHOLDS_TABLE}}

## Evaluation

Species **macro-F1** averages F1 over species, so the rare species count as much as the common ones.

- **GBIF test fold (in-distribution):** held-out images from the same GBIF sources as training.
  The published file was scored on a random 10,000-image sample of the 629,742-image fold. Macro-F1
  over a sample is not the same number as over the whole fold, so treat it as a check that the file
  works on ordinary photos, not as a benchmark.
- **Probe:** 15,200 light-trap images of 368 Danish species, from (trap, night) groups never used in
  training. This is a real domain shift: night-time camera crops, not curated photos.
- **Probe, held-out species:** 2,455 images of 58 species for which no trap images were used in any
  form during training.

| evaluation | species macro-F1, training pipeline | species macro-F1, **this ONNX file** + the quick-start preprocessing |
|---|---|---|
| GBIF test fold, random 10,000 images | not measured | 0.9096 (top-1 92.3 %; genus 0.953, family 0.974) |
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
starting at `START-HERE.md`. The configuration is `configs/20260804_B3rep5x_effnetv2s.yaml`.

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
