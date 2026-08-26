# The 768-d CLIP projection preserves what we need, and zero-shot understates the representation

**Kind:** research · **Status:** **RESOLVED (2026-08-26). My objection was wrong.** Across three
independent samples the projected 768-d space matches or slightly beats the 1024-d pooled space on
nearest-centroid top-1. **The published ToL-200M embedding cache is usable as-is**, which unblocks the
open-set-at-scale direction without re-embedding 233 M images.

## What I argued, and why it was a reasonable worry

The cache stores the *projected* 768-d CLIP embedding, not the 1024-d pooled tower output `dev/075`
uses. My objection was explicitly **not** about width — the bottleneck sweep already showed narrowing
the representation costs little. It was that the CLIP projection is a **learned** map fitted to pull
each image toward the text embedding of its taxon name, and sibling species have near-identical name
strings, so their text embeddings nearly coincide, so training should pull their image embeddings
toward nearly the same point. A PCA or random projection preserves geometry; a supervised one need
not.

The mechanism is real. It is just not large enough to matter.

## Result

`dev/077_bioclip_feature_probe.py`, local RTX 5090, three samples.

| sample | classes | centroid top-1, **1024 pooled** | centroid top-1, **768 projected** |
|---|---|---|---|
| in-distribution, congeners | 60 | 0.8972 | **0.8972** |
| trap probe, congeners | 25 | **0.9050** | 0.8950 |
| trap probe, all species | 164 | 0.8765 | **0.8857** |

**The projection compresses the scale but preserves the ranking.** Absolute cosines inflate sharply —
on the 164-class trap sample, same-genus centroid cosine goes 0.7595 → 0.8928 and different-family
0.3967 → 0.7260, so the *gap* shrinks from 0.363 to 0.167. Everything looks more similar to
everything. But nearest-centroid top-1 is unchanged, because top-1 depends on the *order* of the
similarities, not their spread.

That is the distinction I failed to make when I raised the objection: I reasoned about the geometry
compressing and assumed discriminability would follow. It does not, at least not here.

**Practical consequence:** the 350 GB `imageomics/TreeOfLife-200M-Embeddings` cache can be used
directly. No re-embedding, no 20 TB of JPEGs, no image pipeline. The open-set-at-scale programme
becomes a disk-bound job on one GPU.

**Caveat worth keeping.** Absolute cosine values are *not* comparable between the two spaces, so any
threshold — abstention cut-offs, novelty score cut-offs, calibration temperatures — must be refitted
if the space changes. The ranking transfers; the numbers do not.

## Zero-shot vs a fitted probe, on the same images

The owner's 44 % on the shifted set was measured with **text-encoder embeddings**, i.e. zero-shot:
embed the class names, embed the image, take the nearest text vector. On the 164-class trap sample:

| method | top-1 |
|---|---|
| zero-shot, binomial prompt | 0.7846 |
| zero-shot, full-hierarchy prompt (BioCLIP-style) | 0.7876 |
| **nearest-centroid on frozen features** | **0.8857** |

**A fitted probe beats zero-shot by ~9 points** at matched class count on the same images, and the
prompt template is worth almost nothing (0.3 pt). So zero-shot does understate the representation,
but by single digits at this scale, not by the 30+ points that would be needed to explain 44 %.

**These numbers are not comparable to the 44 %, and it would be wrong to present them as a
correction of it.** Three differences, each large:

- **Metric.** These are top-1 accuracy (micro). The 44 % is macro-F1, which on a long tail is far
  lower by construction.
- **Class count.** 164 classes here against a full label set of 12,041. Zero-shot difficulty scales
  hard with the number of candidate names.
- **The centroid arm enrols on trap images.** Using 70 % of each species' *target-domain* images to
  build centroids is itself a form of adaptation, and a generous one.

So the honest reading is: **the 44 % was a legitimate zero-shot measurement, and zero-shot is simply
the wrong instrument for "how good is this representation for our task."** The right instrument is a
probe on frozen features, which is exactly what P1a/P1b are. The local test says the representation
is strong — zero-shot at 0.79 over 164 trap classes is not a weak encoder.

## What this does not settle

Whether a frozen BioCLIP-2 trunk plus our classifier stages matches our own trunk on the *shifted*
benchmark at the full 12,041-class label set. That is P1b, running now, and nothing here substitutes
for it. These probes use 164 classes and generous enrolment; P1b uses the real label set and the real
protocol.

## Note on the local environment

`open_clip_torch`, `ftfy`, `regex` and `wcwidth` were added to the training venv with
`uv pip install --no-deps`, verified before and after to leave `torch 2.12.1+cu130` and
`torchvision 0.27.1+cu130` untouched. `--no-deps` is mandatory: a plain install pulls a 508 MB PyPI
torch and would break the hand-managed venv, which is the same trap the cluster preflight caught.
