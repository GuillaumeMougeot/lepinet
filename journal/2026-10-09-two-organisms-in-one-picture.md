# When a picture holds two organisms, what should a whole-tree model say? (S1)

**Kind:** research · **Status:** **OPEN (2026-10-09) -- design; hypotheses committed, nothing run
except the duplicate count below.**

## The question (the owner's framing)

A Lepidoptera-only model has no name for the leaf the moth sits on, so the leaf is background. A
model trained on the whole tree of life (W3) has a name for it. On a photo of an insect on a plant,
which is a large share of insect photos, it may respond to both, although every training label
names one organism. Some of its "false positives" are then not false: the organism is in the
picture, just not in the label. How should we read, train and evaluate for that? This is about
inference as much as training.

## Why it matters for this project specifically

Four places where it changes a number we report or plan to report:

1. **Top-1 errors that are not errors.** If the top-1 answer is the host plant, the accuracy and
   macro-F1 counts are wrong in a way no metric we use can see. ImageNet went through the same
   thing: re-labelling found that a large part of the "errors" of strong models were other objects
   that really are in the image (ReaL labels, Beyer et al. 2020 [VERIFY]).
2. **Open-set detection, the paper's subject.** Every rule we use (max-softmax, entropy; O1 found
   entropy best for B8 and P5) reads a spread-out distribution as "unknown". Two organisms that are
   really there also spread the distribution. **A two-organism image can look exactly like a novel
   one.** A whole-tree model makes this worse, because it can name the second organism.
3. **W3's evaluation protocol.** Scoring a whole-tree model on Lepidoptera test images requires a
   choice: argmax over the whole tree (the plant can win) or over the Lepidoptera subtree only
   (conditional on "it is a moth"). The two give different numbers and answer different questions.
   This has to be fixed before W3 runs, not after.
4. **Self-training (B3, our largest lever).** Pseudo-labels are hard argmaxes over trap images. A
   whole-tree teacher can pseudo-label a moth crop by whatever else is in the crop.

## What a softmax does with two organisms

* **Softmax assumes exactly one class per image.** Cross-entropy on "moth" pushes every other logit
  down, the plant's included. So the model does not learn "the plant is absent". It learns **which
  organism the photographer meant**: centred, in focus, largest. That signal is real and partly
  what we want: users usually ask about the subject. The risk is that it is taught implicitly and
  inconsistently: on an observation of the plant, the same moth would be the background.
* **The hierarchy makes it worse at the top.** Marginals are sums of leaf probabilities. A moth on
  a leaf must split one unit of mass between Animalia and Plantae, while in the picture both are
  present with probability 1. At species level the two compete for mass, and at kingdom level the
  split is plain wrong.
* **The geometry.** B8 is a ConvNeXt: average-pooled features feed a linear map. Pooling and a
  linear map commute, so the image's logits are exactly the average of per-location logits (the
  identity behind class activation maps, Zhou et al. 2016 [VERIFY]). The cosine head normalises
  after pooling, which breaks exactness but not the idea. Consequences:
  - **The global embedding of a two-organism image is roughly an area-weighted mix of the two.** It
    sits between the two clusters and is close to neither, so a centroid-distance novelty score
    also flags it.
  - **The per-location logits** still hold each organism separately. Reading them costs one
    matrix product, with no retraining.

## What the data says (measured today)

The question was: do identical images appear in TreeOfLife under two species? The new crawl's
metas, 64,077,776 downloaded images, `content_hash` of the 256 px pixels:

| | count |
|---|---|
| hashes carrying two or more species | 1,318 (14 shared by more than five species: placeholders) |
| exactly two species | 1,208 |
| ... same genus | 371 |
| ... same order, different genus | 310 |
| ... different orders | 527 |
| ... of which an insect with a plant | ~0 (the only insect pairs: 6 Diptera / Lepidoptera) |

* **Half the two-species images come from FathomNet** (1,457 of 2,727 rows), and they dominate the
  cross-order pairs: sponges, corals, sea stars, crabs, octopus. These are deep-sea video frames:
  one frame, several organisms, each annotation exported as its own record. FathomNet publishes
  bounding boxes for every organism in a frame [VERIFY for these exact images], and **23,228
  FathomNet images are in our download**. That is a real multi-organism evaluation set, inside the
  tree, labelled by experts, at no annotation cost. It is marine rather than insects, but it tests
  the mechanism.
* **The iNaturalist / GBIF duplicates are not co-occurrence.** In a random sample of 25 iNaturalist
  pairs, every pair was same genus or same family (*Oxytropis ambigua* / *O. strobilacea*,
  *Pisolithus arhizus* / *P. tinctorius*): one photo filed under two names, i.e. label conflict.
* **Exact hashing cannot find insect-on-plant photos.** When a user files the same photo as both
  the insect and its host plant, the copies are re-encoded and the pixels differ. In ToL such a
  photo is simply single-labelled. Finding them needs a perceptual hash, the source's interaction
  fields (GBIF `associatedTaxa`, iNaturalist interaction annotations; whether they are populated
  for our rows is unchecked), or a model.
* **The audit currently drops the second label.** `dev/085` keeps one copy of each pixel-duplicate
  group, which is right for training under softmax but discards exactly these pairs. The pair list
  is cheap to regenerate (the script above, ~5 min on 6 cores) and should be kept as an artefact.

## What the literature offers (from memory; every reference [VERIFY])

* **Single-positive multi-label learning** is this exact setting: each training image has one
  positive and the other labels are *unknown*, not negative. Cole et al., CVPR 2021, show that
  treating the unknowns as negatives (what softmax CE does implicitly) is the main damage, and
  propose regularised alternatives. Kim et al., CVPR 2022 ("large loss matters") reject or flip
  the negatives the model insists are positive, late in training. Zhou et al., ECCV 2022, maximise
  entropy on the unknowns and pseudo-label them asymmetrically.
* **ReLabel** (Yun et al., CVPR 2021) uses a strong teacher to produce *per-location* multi-label
  maps for ImageNet, then trains on the label of the crop actually seen. That combines
  "teacher soft labels" with "where in the image", and it is close to what our self-training
  already has the parts for.
* **Copy-paste augmentation** (Ghiasi et al., CVPR 2021) pastes segmented objects onto other
  images; with both labels kept positive it manufactures multi-organism training data whose
  ground truth is known.
* **Multi-instance learning**: the image label says "at least one region is this class", which
  leaves other regions free to be something else.

None of these works deal with a 185 k-class hierarchy. That is where we might add something.

## Strategies

### Evaluation first (no strategy can be scored without it)

| | what | cost |
|---|---|---|
| V1 | **Synthetic composites with known truth**: paste a segmented moth onto (a) another moth image, (b) a plant photo, (c) a trap background, at area ratios 0.1-0.9 | workstation GPU, hours; needs a segmenter or trap crops with clean backgrounds |
| V2 | **FathomNet frames**: 23 k images, every organism boxed upstream | download the box annotations; free |
| V3 | **300-500 hand-checked photos of insects on plants**: mark every visible organism to family | ~1 day of human time |
| V4 | GBIF / iNaturalist interaction fields joined on `source_id` | check whether they are populated |

Metrics: subject accuracy (the label), **secondary recall at k** (is the other organism in the top
k?), **share of top-1 errors where the predicted taxon is present**, and the open-set false-alarm
rate on two-organism images at the threshold fitted on single-organism ones.

### Inference only (cheapest; try these first)

* **I1 -- per-location logits.** Apply the head to the feature map before pooling (ConvNeXt) or to
  patch tokens (ViT, P5), group locations by their top class, and report "subject + other organisms
  present". Exact for a linear head, approximate for the cosine head. No retraining.
* **I2 -- subtree-conditional readout.** Renormalise within the clade the deployment cares about:
  "which moth, given it is a moth", separately from "what else is here". For a moth trap this is the
  natural question, and it decides W3's protocol (point 3 above). Equivalent to K1's checklist
  restriction, with a clade in place of a region.
* **I3 -- an open-set rule that does not punish two organisms.** Compute the score inside the
  subtree (I2), or on the dominant region's logits (I1), instead of over the whole distribution.
  E2 showed the scoring rule can be worth more than the model; this is the same lesson again.
* **I4 -- detect, then classify.** A generic "organism" detector, then one classification per box.
  Trap pipelines already work this way. Heavy for phone photos.

### Training (only if evaluation shows harm)

* **T1 -- nothing.** Softmax CE; the model learns the subject. The baseline every option must beat.
* **T2 -- tree-aware partial negatives.** Make classes exclusive *within* a clade (siblings compete:
  two congeneric species are rarely both the subject) and *unknown across* distant clades (a plant
  logit is neither pushed up nor down by a moth label). Concretely: conditional softmax among
  siblings below some level, and above it, per-clade presence heads where unlabelled clades are
  ignored rather than treated as negatives. Unlike flat single-positive methods it uses the tree,
  which nobody seems to have done at this scale. It slots in as a `HEAD_REGISTRY` entry.
* **T3 -- soft targets from a teacher** (distillation / ReLabel-style). The teacher's secondary
  mass on co-present organisms becomes part of the target instead of being trained away. Our
  self-training pipeline already produces teacher outputs, so this is cheap to try. Risk: the
  teacher's own subject bias is inherited.
* **T4 -- reject large-loss negatives** (Kim et al.). Needs a sigmoid/BCE head, which we do not
  have, so it is a large change.
* **T5 -- composites as training data** (copy-paste / CutMix with *both* labels present, not
  area-weighted). Teaches presence instead of mass-splitting. Risk: paste artefacts become a
  shortcut.
* **T6 -- co-occurrence-aware label rebalancing (the owner's idea, made concrete).** Label
  smoothing spreads ε uniformly over all classes. Instead, give that mass to classes likely to be
  present: the host plants of a moth from an interaction database (GloBI [VERIFY]), or classes the
  model predicts confidently on images labelled otherwise (mined, as in T4, but under softmax).
  In practice this down-weights only the negatives that are probably wrong.

## Predictions (committed before any run)

* **P1.** On single-labelled photos of insects on plants, a whole-tree softmax model names the
  plant at top-1 rarely (**< 3 %** of images), because CE trains subject selection; the plant is in
  the top 5 often (**> 25 %**).
* **P2 (the one that matters for the paper).** On V1 composites, max-softmax and entropy degrade
  with the pasted area ratio. At the threshold fitted on single organisms, **two-organism images
  are flagged as novel at least twice as often as single-organism images of the same species**.
  Our open-set scores confuse "two known things" with "one unknown thing".
* **P3.** I1 recovers the second organism in the top 5 of its own region on **most (> 70 %)**
  composites with area ratio ≥ 0.2, without retraining.
* **P4.** The composite's global embedding stays close to the chord between the two class means:
  its cosine to the subject's centroid falls roughly monotonically as the subject's area shrinks.

If P2 is false, i.e. the open-set scores are robust to a second organism, most of this direction
becomes a readout feature (I1, I2) rather than a research question.

## Recommended order

1. **S1a, now, no download needed**: V1 composites (moth on moth, moth on trap background) with
   the published B8 and P5; measure P2-P4 and try I1/I3. Lepidoptera-only models cannot name a
   plant, so moth-on-plant composites test distraction, not secondary recall.
2. **S1b, with the W3 model**: the same plus moth-on-plant, V2 (FathomNet), P1 on V3. Fix W3's
   readout (I2) before W3 is scored.
3. **Training (T3 or T6 first, then T2)** only if S1 shows harm that the readout cannot fix. T3
   and T6 come first because they change only the classifier's targets, which is where this
   project's interventions have worked (§4.15), and T3 reuses the self-training machinery. T2 is
   the most interesting scientifically and the most work.
