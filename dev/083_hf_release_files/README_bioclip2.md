---
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: image-classification
base_model: imageomics/bioclip-2
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
---

# lepinet · BioCLIP-2 ViT-L/14: moth and butterfly identification

Identifies **Lepidoptera** (moths and butterflies) from a photo at three ranks at once: **species**
(12,041), **genus** (4,333) and **family** (102). The three answers are coherent, because genus and
family are computed from the species probabilities. The model also tells you when it should not
answer at species level.

It is the model recommended by the [lepinet](https://github.com/GuillaumeMougeot/lepinet) project:
tied for the most accurate of the project's models on images from a *different* source than its training data
(automated light-trap cameras), and the best calibrated. In the project's deployment study, at a
95 % precision target, it answered 93 % of trap images against 73 % for an equally accurate
alternative. That study fitted thresholds in-sample; the stricter fit shipped here answers 87 %
(see [Using the confidence](#using-the-confidence)).

| | |
|---|---|
| **Architecture** | [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) ViT-L/14 image tower (303 M params), fine-tuned end to end, plus a 1024-d cosine classifier; 321 M in total |
| **Input** | one RGB image of **one** insect, 224×224 |
| **Output** | species, genus and family probabilities, raw logits, a 1024-d embedding |
| **Format** | ONNX (fp32, 1.3 GB). Runs with `onnxruntime` alone: no PyTorch, no lepinet |
| **Licence** | CC-BY-NC-4.0 (non-commercial; see [Licence](#licence)) |

## Quick start

```bash
pip install onnxruntime pillow numpy huggingface_hub
```

```python
import json, numpy as np, onnxruntime as ort
from huggingface_hub import snapshot_download
from PIL import Image

repo = snapshot_download("gmougeot/lepinet-bioclip2-vitl14", allow_patterns=["model.onnx", "*.json"])
session = ort.InferenceSession(f"{repo}/model.onnx")
names = json.load(open(f"{repo}/names.json"))["names"]

def load(path, size=224):  # shorter side -> size, centre crop, RGB in [0, 1]
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
hf download gmougeot/lepinet-bioclip2-vitl14 model.onnx predict.py config.json taxonomy.json names.json thresholds.json --local-dir lepinet
python lepinet/predict.py moth1.jpg moth2.jpg --top 3
#   -> species: Macaria notata (0.99)  https://www.gbif.org/species/5880550
```

Batching works: the input's first axis is dynamic. On a GPU, install `onnxruntime-gpu`; the session
picks CUDA automatically.

## Files

| file | what it is |
|---|---|
| `model.onnx` | the network. Input `image`: float32 `[N, 3, 224, 224]`, RGB, values in [0, 1]. **Normalisation is inside the graph** (do not normalise yourself) |
| `taxonomy.json` | `vocabs.<rank>[i]` is the [GBIF](https://www.gbif.org) taxon key of output index `i`; `parents` maps species to genus and genus to family |
| `names.json` | scientific names, aligned index for index with `taxonomy.json` |
| `thresholds.json` | per-rank confidence thresholds for a 95 %-precision back-off policy, with the precision and coverage they achieve on held-out data |
| `config.json` | preprocessing and output description (it doubles as a [lepinet-app](https://github.com/GuillaumeMougeot/lepinet-app) bundle manifest) |
| `predict.py` | a standalone CLI and `Lepinet` class (onnxruntime + Pillow only) |
| `model.safetensors` | full-precision PyTorch weights, for fine-tuning with lepinet |

### Outputs

| name | shape | meaning |
|---|---|---|
| `prob_species` | `[N, 12041]` | softmax over species |
| `prob_genus` | `[N, 4333]` | sum of each genus's species probabilities (**use this**, not the genus head) |
| `prob_family` | `[N, 102]` | sum of each family's genus probabilities |
| `logits_species`, `logits_genus`, `logits_family` | | raw scores of the three trained heads (the genus and family heads are kept for completeness; the summed probabilities are more accurate) |
| `embedding` | `[N, 1024]` | L2-normalised image embedding (cosine similarity = dot product), for retrieval, clustering or novelty detection |

A species' GBIF page is `https://www.gbif.org/species/<key>`.

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

A low maximum species probability, or high entropy of `prob_species`, is also the best novelty
signal this model has. Ranking known against unseen species by entropy gives **AUROC 0.916** on the
GBIF test fold.

## Evaluation

Species **macro-F1** averages F1 over species, so the rare species count as much as the common ones.
Three evaluation sets:

- **GBIF test fold (in-distribution):** held-out images from the same GBIF sources as training.
  BioCLIP-2's pre-training data contains about two thirds of these exact images, so the score is
  reported on the 219,048-image subset BioCLIP-2 never saw ("clean fold"). The published file was
  also checked on a random 10,000-image sample of the full fold. That sample is partly
  contaminated, and macro-F1 over a sample is not the full-fold number, so read it as "the file
  works on ordinary photos", not as a benchmark.
- **Probe:** 15,200 light-trap images of 368 Danish species, from (trap, night) groups never used in
  training. This is a real domain shift: night-time camera crops, not curated photos.
- **Probe, held-out species:** 2,455 images of 58 species for which no trap images were used in any
  form during training.

| evaluation | species macro-F1, training pipeline | species macro-F1, **this ONNX file** + the quick-start preprocessing |
|---|---|---|
| GBIF test fold, clean subset | 0.9113 | not re-run |
| GBIF test fold, random 10,000 images | not measured | {{GBIF_ONNX}} |
| Probe (light traps) | 0.7810 (a repeat training run: 0.7703) | 0.7723 |
| Probe, held-out species | 0.7806 | 0.7897 |

Genus and family macro-F1 on probe, from the summed probabilities: **0.826** and **0.842**. At family
level the sum beats the model's own family head (0.818), which is why `prob_family` is the
recommended output.

The two columns differ by under 1 point, which is within the run-to-run noise of these evaluation
sets. The network is numerically identical to the PyTorch model (max |Δ| ≈ 3e-5); the difference is
image resizing. Trap crops are small (median shorter side 157 px), so they are *up*-sampled, and
up-sampling with a different kernel changes 3–6 % of individual predictions. **For small crops,
resize consistently**, and prefer larger crops where you can.

For comparison, BioCLIP-2 fine-tuned *without* the trap-domain stage scores 0.6630 on probe, and
the project's in-house EfficientNetV2-S baseline scores 0.6270.

## Training

1. **Data:** 6.3 M GBIF occurrence images of Lepidoptera; species with at least 50 images; at most
   about 2,000 images per species; a held-out test fold.
2. **Stage 1 (P3c):** BioCLIP-2's image tower (the pooled 1024-d feature, before the text
   projection) with a cosine classifier with one head per rank, fine-tuned end to end at lr 1e-5.
3. **Stage 2 (P5):** 2 more epochs, unfrozen, on the same data plus about 2 % of unlabelled
   light-trap images, pseudo-labelled by an earlier model (self-training). This is the step that buys
   robustness to trap imagery. The trap nights used for evaluation are excluded.

Full recipe, ablations and every number: [the lepinet repository](https://github.com/GuillaumeMougeot/lepinet),
starting at `START-HERE.md`. The configuration is `configs/20260828_P5_bioclip2_adapted_unfrozen.yaml`.

## Limitations

- **One insect per image.** The model classifies; it does not detect. Run a detector first on
  scenes, trap screens or multi-insect photos (for example [flat-bug](https://github.com/darsa-group/flat-bug))
  and classify the crops.
- **Closed label set, uneven geography.** It knows 12,041 species. Training images are 36 % North
  American, 33 % European and 15 % Asian, but only 6 % South American or African, as GBIF coverage
  is. A species outside the label set still receives one of these labels: use the thresholds or
  the entropy, and expect the tropics to be served worse.
- **Domain shift is large.** In-distribution macro-F1 is 0.91, while on trap images it is 0.78. The
  thresholds come from Danish light traps and may not transfer to your camera or region. Re-fit
  them on a few hundred labelled images of your own if precision matters.
- **Adults only, in practice.** Larvae are about 1 % of the training images; caterpillars, eggs and
  pupae will be poorly identified.
- **Look-alike species** that need genitalia dissection or DNA cannot be separated from a photo, by
  this model or by an expert. Treat the genus answer as the honest one there.
- **Not for decisions** about conservation status, pest control or toxicity without expert
  verification.

## Licence

**CC-BY-NC-4.0.** About 77 % of the training images are licensed CC-BY-NC by their GBIF
contributors, so the weights are released for non-commercial use. BioCLIP-2, the base model, is MIT.
The lepinet training code is GPL-3.0.

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

Also cite [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) (Gu et al., 2025) and acknowledge
the GBIF contributors whose images made this possible: GBIF.org (15 May 2025) GBIF Occurrence
Download [https://doi.org/10.15468/dl.hg37y9](https://doi.org/10.15468/dl.hg37y9).
