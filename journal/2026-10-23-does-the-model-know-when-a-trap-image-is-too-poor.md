# On trap images, does the model back off when the image is poor, and notice a species it never saw?

**Kind:** research · **Status:** **RESOLVED** (2026-10-29). The back-off cascade does back off on blurry
crops (species answers 91 -> 65 % from sharpest to blurriest fifth) and keeps species precision within 2
points of target; but on trap images an unseen species of a known genus is barely detected (C3b AUROC
0.566 against 0.8717 on museum images) and 38 % of its crops get a confident wrong name.
Out-of-vocabulary species lose 6-8 points (B8 0.857, P5 0.835).

## The question (the owner's)

Many AMI crops are blurry or tiny. A species-level answer on them is a guess; the model should
answer at genus or family, or abstain. Separately, open-set detection is measured in the paper only
on museum-style images (§4.4) and, under shift, only for old models read with the wrong rule (§4.5).
Both must be measured on trap images, even if they end up in another section of the paper.

## What exists

* The release's back-off policy (`dev/083` cascade: per-rank thresholds at 95 % precision, fitted on
  half the trap nights) was applied to B8, P5 and B3rep5x on the probe set
  ([[2026-10-02-is-the-deployment-gap-a-readout-artefact]]): 10-19 % abstention, and a composition of
  species, genus and family answers. Never broken down by image quality.
* Per-image outputs at three ranks for the probe set are local: `data/hf_release/{b8,p5v2,b3}/eval_probe.parquet`.
* Unseen species on trap images: the C3b model (231 common taxa withheld from training,
  [[2026-08-08-is-novelty-monotone-or-just-rare]]) never saw 9 species that occur at the traps:
  8 in genera it knows (742 crops, 202 of them in the probe set) and 1 in a genus it does not
  (5 crops). No withheld family occurs at the traps. Reconstructed with `dev/072`'s original
  arguments (2 families, 40 genera, 120 species, matching the entry). Every model also lacks the 6
  out-of-vocabulary trap species (234 crops).

## Design

**Part A, image quality (local, no GPU).** For every probe crop: sharpness as the variance of the
Laplacian of the greyscale crop resized to 128 pixels on its short side (so it does not just measure
size), and the crop's pixel area. Split each into quintiles. Per quintile and model: the share
answered at species, genus, family and abstain under the release cascade (thresholds fitted on the
other half of the nights, as in `dev/086`), and the precision of each returned rank.

**Part B, unseen species under shift (one UCloud evaluation job).** C3b scores all 47,905 trap crops,
keeping its logits so that the five open-set rules of §4.9 can be computed. Novel = the 747 crops of
withheld species plus the 234 out-of-vocabulary crops; known = the rest. AUROC per rule, by stratum
(near, mid), and what the cascade answers on novel crops (a correct genus is a good answer for a
near-novel species).

## Hypotheses (2026-10-23, before any measurement)

* **H1. The cascade backs off on poor images.** The share of species answers falls monotonically
  over sharpness quintiles, and the genus, family and abstain shares rise. Falsified if the blurriest
  quintile gets species answers as often as the sharpest (within 3 points).
* **H2. But its precision does not hold on poor images.** The thresholds are fitted over all images,
  so on the blurriest quintile species precision falls more than 5 points below the 95 % target.
  This is §4.6's lesson (calibrate on the images a threshold will serve) applied to image quality.
* **H3. Size behaves like sharpness**, with a weaker gradient: small crops are often small species,
  not poor images.
* **H4. Open-set detection is much weaker on trap images than on the held-out fold.** C3b's near-novel
  AUROC drops from 0.8717 to 0.60-0.75 under shift: shift makes known species look unfamiliar
  (§4.5's logits), and near-novel is the hardest stratum.
* **H5. The best rule may change under shift.** Exploratory; no direction predicted.

## Caveats known in advance

Nine withheld species is few; per-species results will be noisy, and a few common species dominate
the crop counts. The out-of-vocabulary crops come from six species. Sharpness on trap crops also
reflects the moth's texture, not only focus.

## Result, part A: image quality (2026-10-23, `dev/095`)

Held-out half of the probe nights (7,759 crops), release cascade at 95 % precision. Sharpness
quintiles, Q1 the blurriest:

| model | species answers Q5 → Q1 | abstain Q5 → Q1 | species precision Q5 → Q1 | family precision Q1 | correct answer Q5 → Q1 |
|---|---|---|---|---|---|
| B8 (T = 1.91) | 90.9 → 64.5 % | 6.3 → 30.9 % | 0.960 → 0.935 | 0.887 | 90.0 → 64.4 % |
| P5 | 82.7 → 50.6 % | 2.3 → 23.2 % | 0.988 → 0.931 | 0.919 | 96.8 → 71.6 % |
| B3rep5x (T = 1.70) | 88.3 → 62.8 % | 3.0 → 21.1 % | 0.955 → 0.935 | 0.912 | 93.0 → 73.5 % |

* **H1 holds.** Every model answers at species less often as crops get blurrier, monotonically in
  all five quintiles, and abstains or answers at family more often.
* **H2 is falsified, narrowly.** Species precision on the blurriest fifth is 0.931-0.935, 1.5-2
  points under the target, not more than 5. Family answers on blurry crops fall further (0.887 for
  B8). The thresholds were fitted on all images, so precision on poor images drifts down, but the
  policy mostly protects itself by backing off.
* **H3 holds.** Crop size gives a weaker and non-monotone gradient (B8: 71 % species answers on
  the smallest fifth, 74 % on the middle one, 86 % on the largest).
* **Surprise: B8 and B3rep5x never answer at genus.** No genus threshold reaches 95 % on the images
  that failed the species bar (fitted threshold 1.01), so they go straight from species to family.
  P5 uses genus for 5-8 % of crops. The genus rank of a margin-trained head adds nothing on trap images
  under this policy.

Part B (open-set on trap images) runs as three evaluation jobs: N2 (C3b, 747 crops of 9 withheld
species plus 234 out-of-vocabulary crops), N3 (B8 and P5, the 234 out-of-vocabulary crops).

## Result, part B: species the model never saw, on trap images (2026-10-23, `dev/096`)

AUROC of each open-set rule, novel crops against all 46,924 known trap crops (higher score = more novel):

| model | novel crops | best rule | AUROC | same model on museum-style images |
|---|---|---|---|---|
| C3b, near (8 withheld species, genus known) | 742 | MSP | **0.566** | 0.8717 (near stratum, entropy) |
| C3b, mid (1 withheld genus) | 5 | entropy | 0.955 | 0.9463 |
| C3b, out-of-vocabulary (6 species) | 234 | MSP | 0.781 | — |
| C3b, all 981 novel crops | 981 | MSP | 0.619 | — |
| B8, out-of-vocabulary (6 species) | 234 | top-2 margin | 0.857 | 0.9153 (entropy) |
| P5, out-of-vocabulary (6 species) | 234 | max-logit | 0.835 | 0.9161 |

* **H4 holds, and worse than predicted.** On trap images C3b barely detects an unseen species from a
  genus it knows: 0.566, against 0.8717 for the same stratum on museum-style images and a predicted
  0.60-0.75. The unseen genus (5 crops) and the out-of-vocabulary species stay detectable.
* **One species dominates.** *Pleuroptya ruralis* is 682 of the 742 near-novel crops (AUROC 0.563).
  The other seven near species average 0.61 over 60 crops. Read the near result as "one common
  trap species and seven rare ones", not as a population estimate.
* **Out-of-vocabulary species lose 6-8 points of AUROC on trap images** for both release models (B8
  0.9153 to 0.857, P5 0.9161 to 0.835), much less than C3b's near stratum loses. P5's rules sit within
  0.016 of each other; energy stays usable for P5 (0.835) and collapses for B8 (0.47).
* **H5: the best rule changes under shift.** C3b's best rule is MSP on trap images (entropy on museum
  images); B8's is the top-2 margin (entropy on museum images). Max-logit and energy fall below
  chance for C3b on trap images (0.42, 0.39): shift lowers every logit, known species included.
* **What the back-off policy says to an unseen species** (C3b, thresholds fitted on known crops of
  half the nights, 95 % target):

  | crops | species answer | genus | family | abstain | genus correct | family correct |
  |---|---|---|---|---|---|---|
  | known (other half of the nights) | 46.2 % | 10.3 % | 27.1 % | 16.4 % | 0.947 | 0.951 |
  | novel, near | **37.6 %** (all wrong) | 9.2 % | 45.6 % | 7.7 % | 0.426 | 0.988 |
  | novel, out-of-vocabulary | 3.4 % (all wrong) | 19.2 % | 34.2 % | 43.2 % | 1.000 | 0.712 |

  **38 % of the crops of an unseen species in a known genus receive a confident, wrong species
  name.** The family answers are right (0.988), so backing off would have helped; the policy simply
  does not see that the species is new. Out-of-vocabulary crops are mostly caught (43 % abstain).

**What it means.** On trap images, novelty detection is the weak link, not the back-off. A species
close to a known one passes as known, with a confident name. This confirms §4.5's direction with a
current model and the right rules: novelty detection is not domain-robust. It also means a deployed
model's species answers on trap images should be read as "this species or a close relative it
was never shown".
