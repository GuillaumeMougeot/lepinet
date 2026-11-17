# Beyond scaling what works: what would make a better biodiversity classifier and backbone?

**Kind:** research · **Status:** OPEN (2026-11-13; first steps run 2026-11-15, section 7, which
corrects this paragraph: against the regional list the pipeline already ships, the estimated prior
adds +4.6 accuracy to Nemo and nothing to Otto; self-training's gain over the list is +6.5 to +8.0
accuracy, not +15 to +17; the observer split re-ranks no model; marginal supervision makes
congeners neighbours on every backbone. Section 8, same day, on whether self-training repairs the
images or the prior: mostly the prior. A regional list and an estimated prior, with no training,
score level with self-training alone on the calibrated macro-F1, and 76 to 78 % of the crops
self-training repairs are the same crops; only the small backbone also learned to read trap
crops. Section 9, 2026-11-17, on the first Tree of Life model: frozen, it is a better encoder than
BioCLIP-2 on Rare Species, on taxa only BioCLIP-2 has seen and on moths; its Lepidoptera head trails
the specialist mainly through the label space, resolution and confidence, not sight). A review of the project for new scientific
routes, with six CPU checks run the same day on saved predictions and logits. Short version: under
shift the models mostly fail by **leaving the local fauna** (87 % of trap errors name a species that
occurs at no trap). That is a prior the model carries in its weights, and it can be estimated from
unlabelled crops instead: **+14.4 points of probe accuracy for Nemo, with no training and no
labels**. Also: 68 % of the test fold comes from an observer who photographed the same species in
training; N2's unseen trap species was a duplicated class; and the one shipped model trained with
marginal supervision has a taxonomic prototype geometry (nearest prototype a congener for 84 % of
species against 17-23 % for the two without; confounded with backbone). Six routes follow.

## Why this entry

The owner asked (2026-11-13) for routes to further scientific improvement, beyond the engineering
routes already in `PLAN.md` (ME, W3, U, TAX, PB), toward the best model for biodiversity images as a
classifier and, more importantly, as an embedding backbone. Theory first, implementation second.

What I read: `PLAN.md`, `RESULTS.md`, the docs, the head and loss code, and 16 journal entries in
full (the reframe, scaling the head, H4, N, Q, X, U, S1, W3, ME, PB, AG, Otto, TAX). Not every entry.

## Predictions for the CPU checks (written 2026-11-13, before each ran)

All use files already on the workstation (`dev/107_population_probes.py`), no GPU.

**Z1, the observer split of the test fold.** Each test image is put in one of three groups by its
`recordedBy`: **S**, the observer has training images of the same species; **O**, the observer has
training images, none of this species; **N**, the observer is absent from training. Species recall
is compared between groups on the species that occur in both (recall decomposes over images,
macro-F1 does not). Models: L0 (uniform sampling) and singlehead (√-oversampling), the same recipe
otherwise.

- Z1a. S holds more than half of the test images.
- Z1b. Recall is 4 to 10 points higher on S than on N, with O between. Falsified if S - N is under
  2 points.
- Z1c. Oversampling's recall gain on rare species (under 200 training images) is larger on S than
  on N, by a factor 1.2 to 2. This is the test of the mechanism the paper states for finding 5
  ("what a model learns from 43 photographs is mostly about those photographs ... the test fold
  shares those artefacts"). If the factor is under 1.2, the test fold does not show that mechanism,
  and a shift of the label prior toward the tail explains the trade better.

**Z2, novelty read from a population of crops, not from one crop.** C3b's saved scores on the
47,905 trap crops, grouped by predicted species (10 crops or more). A group is *present* if most
of its crops are that species, a *ghost* if most are a species C3b never saw (withheld or out of
vocabulary), *confusion* otherwise.

- Z2a. The 90th percentile of the max-softmax within a group (its confident core) separates ghost
  groups from present groups with AUROC at or above 0.85, against 0.566 to 0.62 per crop. Falsified
  below 0.70. Ghost groups will be few (5 to 20), so the interval is wide.
- Z2b. Bags of n crops drawn from the near-novel crops against bags from known crops, scored by the
  mean max-softmax: AUROC at or above 0.80 at n = 30.

**Two follow-ups, added after Z1 and Z2a were read and before either of these ran.**

- Z1d. If the cost of oversampling under shift is a label prior moved toward the tail, then on the
  47,905 trap crops the oversampled single head names rare species that occur at no trap more often
  than L0 does: the share of crops predicted as an absent species with under 200 training images is
  at least 1.5 times L0's. Falsified under 1.2. L1 and L2 (balanced softmax) should be higher still.
- Z2c. A presence filter without labels: a species counts as present at the traps if at least one
  adapt-night crop is predicted as it with max-softmax 0.9 or more; on the probe nights, predictions
  of other species are dropped. Macro-F1 over labelled and predicted species (mini_metrics'
  convention at zero threshold) rises by at least 10 points for models that never saw trap images
  (C3b, B1, L0), and macro recall over the labelled species falls by under 2 points.

**Z4, an estimated site prior, added after Z2c was read and before it ran.** Z2c's hard filter
cost 20 points of macro recall, so the soft version: the class prior of the traps estimated by
expectation-maximisation on the unlabelled adapt-night crops (Saerens et al. 2002 [VERIFY]), from
the saved single-view logits of the TTA study, then `argmax p(y|x) * prior(y)` on the probe nights.
No labels, no training. Models: Nemo (never saw a trap image) and P5.

- Z4a. Nemo's top-1 accuracy on the probe nights rises by at least 5 points (the Danish checklist
  gave +5, K1), and macro-F1 over labelled and predicted species by at least 15.
- Z4b. Macro recall over the labelled species falls by under 3 points. Falsified above 8.
- Z4c. A prior estimated on the two other sites keeps at least half of the accuracy gain at the
  third.

**Z3, how taxonomic the prototype geometry already is.** Species prototypes of the local models
(unit directions).

- Z3a. In every model the nearest other prototype is a congener for at least 30 % of the species
  that have one (chance is under 1 %), and the mean cosine is ordered congener > same family >
  other.
- Z3b. No direction predicted between models (margin against plain, marginal supervision against
  none): the local models differ in too many factors.

## 1. What the checks found

Outputs: `data/population_probes/*.csv`. Each subcommand of `dev/107` prints its table.

### Z1: two thirds of the test fold is "same observer, same species"

Test fold, 629,742 images with predictions; 232,579 distinct observers in training.

| group | images | share | accuracy, L0 | accuracy, singlehead |
|---|---|---|---|---|
| S: observer has this species in training | 430,085 | 68.3 % | 0.9441 | 0.9451 |
| O: observer in training, not with this species | 186,917 | 29.7 % | 0.9094 | 0.9129 |
| N: observer absent from training | 12,671 | 2.0 % | 0.8899 | 0.8912 |

Paired over the species present in both groups (macro recall, singlehead; 95 % interval by
bootstrap over species):

| contrast | species | recall | difference |
|---|---|---|---|
| S against N | 3,278 | 0.9329 / 0.8592 | **+7.4** [+6.4, +8.4] |
| S against O | 11,514 | 0.9299 / 0.8900 | +4.0 [+3.7, +4.3] |
| O against N | 3,282 | 0.9051 / 0.8591 | +4.6 [+3.6, +5.6] |

- **Z1a held** (68.3 %). **Z1b held** (+7.4, O between).
- Within S, an image whose observer has a training image of the species on the *same day* is no
  easier (0.9443 against 0.9453). So it is not near-duplicate sessions.
- **Z1c is falsified.** Oversampling's gain on rare species (singlehead minus L0) is +8.3 on S
  against +9.6 on N (524 species, difference -1.3 [-4.5, +1.7]), and +6.4 on S against +6.9 on O
  (5,502 species, -0.5 [-1.0, +0.1]). On common species it is zero in every group. What oversampling
  teaches about a rare species works as well for an observer the model never saw.
- **Limit.** The groups differ in more than the observer. An N observer has no training image
  because all of their few records fell in the test fold: a one-off contributor, with whatever that
  means for photo quality. So 7.4 points is an upper bound on what observer familiarity is worth.
  The clean number needs a fold split by observer (route 3).

### Z1d: the cost of tail weighting under shift is species that are not there

All 47,905 trap crops; 486 species occur at the traps. Share of crops:

| model | accuracy | names an absent species | absent, rare | absent, common | wrong, a trap species |
|---|---|---|---|---|---|
| L0, uniform sampling | 0.659 | 29.6 % | 3.4 % | 26.1 % | 4.5 % |
| singlehead, √-oversampling | 0.616 | 34.7 % | 10.1 % | 24.6 % | 3.8 % |
| L1, balanced softmax | 0.521 | 45.2 % | 23.8 % | 21.4 % | 2.7 % |
| L2, both | 0.469 | 50.6 % | 34.4 % | 16.2 % | 2.5 % |

- **Z1d held** (2.9 times L0's share; L1 6.9 times, L2 10 times). The whole accuracy loss of
  oversampling (-4.3 points) is predictions of rare, absent species (+6.7); every other error type
  shrinks.
- With Z1c, this replaces the explanation given for finding 5. Tail weighting does not teach
  photograph-specific features that fail to transfer; it moves the model's label prior toward 11,500
  species that a Danish trap never sees. The trade is **label shift**, and label shift has a
  correction that needs no retraining (route 1).
- **The larger fact is in the first row.** For the model with no tail weighting, 87 % of trap errors
  (29.6 of 34.1 points) name a species that occurs at no trap. Only 4.5 % of crops are confusions
  between two species of the local fauna.

### Z2: a present species has a confident core; and the main "unseen" species was not unseen

Groups of crops with the same predicted species, C3b, all traps pooled, 10 crops or more: 250
present, 265 confusion, 11 ghost.

| statistic of the group | ghost against present | confusion against present |
|---|---|---|
| 90th percentile of max-softmax | 0.922 [0.81, 0.99] | 0.972 |
| mean max-softmax | 0.911 [0.82, 0.99] | 0.970 |

Per (predicted species, trap): 600 present, 233 confusion, 15 ghost; not-present against present
0.942 (90th percentile). Median 90th percentile: present 0.976, confusion 0.462.

- **Z2a held on the letter (0.922), and the reading changed.** The largest ghost group, 426 crops,
  looks exactly like a present species (90th percentile 0.982). It is one. C3b withheld *Pleuroptya
  ruralis* (key 1879252), and its training set kept *Patania ruralis* (key 9674732, 1,907 images):
  **one species in COL XR, two classes in our labels** (one of the 281 duplicated classes,
  [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]]). C3b sends 423 of the 682
  *Pleuroptya* trap crops to *Patania ruralis*, with mean max-softmax 0.716 (correct known crops:
  0.743). The population statistic was right and the label was wrong.
- **This corrects N2** ([[2026-10-23-does-the-model-know-when-a-trap-image-is-too-poor]]). Its
  headline, near-novel AUROC 0.566 on trap crops and "38 % of the crops of an unseen species get a
  confident wrong name", rests on 742 crops of which 682 are this species. Recomputed (max-softmax):

  | novel crops | n | AUROC |
  |---|---|---|
  | near, as published | 742 | 0.566 |
  | near, *Pleuroptya ruralis* only | 682 | 0.563 |
  | near, without it | 60 | 0.595 (top-2 margin 0.677) |
  | out of vocabulary | 239 | 0.784 |
  | truly unseen (the last two) | 299 | 0.746 |

  So open-set detection on trap crops is weaker than on museum-style images (0.87 to 0.97), but
  "barely detected" described a species the model knew. The museum-style C3b result is not affected:
  2 of its 120 withheld near species have a duplicate class left in training, 188 of 10,391 test
  images.
- The four groups that are truly unseen species sit at the 0th, 2nd, 6th and 17th percentile of the
  present groups' confident core (*Tethea or* read as *T. ocularis* and as *Euthyatira*, *Acronicta
  cuspis* as *A. psi*, *Timandra comae* as *T. griseata*). Four groups is an anecdote.
- **Z2b held at the limit:** bags of 1 / 10 / 30 / 100 crops give 0.565 / 0.683 / 0.796 / 0.934,
  on the curve expected for independent crops (0.566 / 0.700 / 0.818 / 0.951). The populations are
  those of the published near stratum, so this shows the square-root law, not a detection rate.
- A side finding: ***Tethea or*, a common European moth, has no row in the unfiltered GBIF table**
  (7.49 M rows; *T. ocularis* has 1,753). Its epithet is the word "or". A query or a name parser
  upstream probably dropped it; not checked. Its 169 trap crops are out of vocabulary for that reason.

### Z2c: a hard presence filter works for the metric and kills the tail

Presence from the adapt nights (no labels), scored on the probe nights (368 labelled species).

| model | rule | macro-F1, labelled + predicted | macro recall, labelled | species predicted |
|---|---|---|---|---|
| C3b | none | 0.160 | 0.621 | 1,457 |
| C3b | presence filter | **0.404** | 0.422 | 243 |
| C3b | confidence threshold fitted with adapt labels | 0.366 | 0.370 | 332 |
| L0 | none / filter / threshold | 0.164 / **0.401** / 0.367 | 0.620 / 0.414 / 0.345 | 1,405 / 236 / 305 |
| singlehead | none / filter / threshold | 0.145 / **0.363** / 0.310 | 0.576 / 0.386 / 0.334 | 1,523 / 258 / 371 |
| B1 (margin) | none / filter / threshold | 0.154 / 0.294 / **0.410** | 0.655 / 0.517 / 0.517 | 1,604 / 569 / 453 |

- **The first half of Z2c held** (+24 points for the plain heads, +14 for B1). A rule that uses no
  label beats the confidence threshold fitted with labels on the three plain heads, at a higher
  recall. For the margin head the fixed 0.9 bar is too low (769 species pass), as O3 found for B8.
- **The second half is falsified by an order of magnitude:** macro recall falls 20 points, not 2. My
  mistake in reasoning: I counted only wrong predictions as dropped. A rare species that is at the
  traps has no confident crop on the adapt nights either, so the filter removes it: the same damage
  as the checklist (K1) and the global confidence gate (R2). A hard yes/no on presence is the wrong
  form; the prior has to be soft. That is Z4.

### Z4: an estimated site prior is worth 14 points of accuracy to a model that never saw a trap

Probe nights (15,200 crops), single view, full vocabulary. The prior is estimated on every crop of
the adapt nights (32,705, no labels), 50 EM iterations from a uniform start.

| model | prior | accuracy | macro-F1, labelled + predicted | macro recall, labelled | species predicted |
|---|---|---|---|---|---|
| Nemo | none | 0.633 | 0.137 | 0.597 | 1,691 |
| Nemo | EM | **0.777** | 0.396 | 0.573 | 452 |
| Nemo | EM, smoothed (0.01 pseudo-crop per class) | **0.778** | 0.412 | 0.599 | 471 |
| Nemo | oracle: the adapt nights' true label counts | 0.887 | 0.655 | 0.707 | 335 |
| P5 | none | 0.836 | 0.394 | 0.792 | 699 |
| P5 | EM | 0.860 | 0.542 | 0.739 | 433 |
| P5 | EM, smoothed | 0.864 | 0.575 | 0.774 | 447 |
| P5 | oracle | 0.915 | 0.688 | 0.747 | 331 |

Accuracy with one site left out of the estimate (no prior / prior from the two other sites / prior
from the site's own adapt nights):

| | LV | OH | SS |
|---|---|---|---|
| Nemo | 0.628 / 0.734 / 0.778 | 0.626 / 0.708 / 0.714 | 0.644 / 0.698 / 0.777 |
| P5 | 0.823 / 0.835 / 0.851 | 0.842 / 0.832 / 0.821 | 0.852 / 0.783 / 0.878 |

- **Z4a held:** +14.4 accuracy (predicted at least 5) and +25.9 macro-F1 in mini_metrics' convention
  (at least 15), from 1,691 predicted species to 452 where 368 are labelled. For scale, the Danish
  checklist gave +5.1 accuracy to another model (K1), so this is not like for like.
- **Z4b held:** macro recall -2.4 with the plain estimate, +0.2 with the smoothed one. The soft
  prior keeps the rare species that the hard filter of Z2c removed.
- **Z4c held at two sites of three.** A prior from the other two sites keeps 70 % (LV) and 94 % (OH)
  of the own-site gain, and 40 % at SS.
- **Self-training had already learned most of it.** P5, self-trained on these traps, gains +2.4 to
  +2.8 accuracy where Nemo gains +14.4. The two are substitutes, which is the first direct evidence
  for the reading of section 2 (two models that differ in much else, so not a decomposition yet).
  P5 still gains 15 to 18 points in mini_metrics' convention, because it stops naming 250 species
  that are not there. On P5 a prior from other sites hurts at SS (-6.9): its weights already hold
  its own sites' prior, and a foreign one overrides it.
- **With the true prior, the unadapted EfficientNetV2-S (0.887) is above the self-trained BioCLIP-2
  ViT-L without one (0.836).** An upper bound that uses labels, but it says where the headroom is:
  the gap between the estimated and the true prior (0.778 to 0.887) is larger than anything a
  backbone bought us on these crops.
- **The failure mode.** A species absent from the estimation pool is suppressed. With the held-out
  species' crops removed from the pool, their accuracy falls from 0.589 to 0.192 (Nemo) and from
  0.759 to 0.405 (P5). So the prior must be updated as crops arrive, needs a floor from a regional
  and seasonal prior (route 1), and novelty must be read **before** the prior is applied: a new
  arrival is exactly what a site prior explains away.
- Not done: a temperature before the EM, TTA views, margin heads, other countries.
- A mistake of mine in the code: `em_prior` allocates a 414 MB temporary per chunk, and 45 of the
  run's 54 minutes were kernel time. The numbers are right; the function should work in place
  before it is reused.

### Z3: the one model with marginal supervision has a taxonomic geometry; the two without do not

Species prototypes as unit directions; 9,658 of 12,041 species have a congener in the vocabulary;
chance for "nearest is a congener" is 0.1 %.

| model | what differs | nearest prototype is a congener | in the same family | mean cosine: congener / same family / other |
|---|---|---|---|---|
| B3rep5x | V2-S, margin, species loss only | 23.4 % | 39.2 % | 0.142 / 0.104 / 0.104 |
| P5 | BioCLIP-2, three independent heads | 16.9 % | 65.3 % | 0.067 / 0.004 / 0.000 |
| B8 | DINOv3 ConvNeXt-L, margin, **marginal loss** | **84.1 %** | 88.0 % | 0.344 / 0.272 / 0.261 |

- **Z3a is falsified** for two of three models (23 % and 17 %, under the 30 % I predicted), and
  B3rep5x has no family structure at all (0.1039 against 0.1041).
- B8 differs from B3rep5x in backbone, marginal supervision and sampling, so this is not an
  attribution. But a gap of 60 points on a statistic with a chance level of 0.1 % is worth the
  one-factor check, which is CPU-only: the campaign has F, BU, I and TD on one backbone with three
  seeds each (route 4).

## 2. One reading of the results: three priors live in the weights

Bayes' rule for a classifier is `p(y|x) ∝ p(x|y) · π(y)`. Shift makes the evidence `p(x|y)` weaker
(soft, small, oddly framed crops), so the answer falls back on `π(y)`. Our `π` is the training
distribution over 12,041 species worldwide; the traps hold 486. Read this way, results that were
journalled separately say one thing:

| result | as a statement about the prior |
|---|---|
| 87 % of trap errors name an absent species (Z1d) | the fallback goes to the wrong fauna |
| tail weighting costs robustness, monotonically (finding 5; Z1c, Z1d) | it moves `π` further from the site's |
| a Danish checklist gives +5 accuracy and hurts the tail (K1) | a prior of zeros and ones |
| two thirds of the pseudo-labels' probe gain is reachable with the trunk truly frozen (VK: +8.6 of +13.1) | a prior is a classifier-side quantity |
| a new site keeps 23 to 74 % of self-training's gain, and the rare tail none (X1, X2) | the learned prior is local |
| real labels at 11 copies lose the unseen species; 91 % of those errors land in the labelled fauna (T1rep) | a prior made too sharp |
| mini_metrics' convention turns 0.69 into 0.15 (1,271 predicted-only species) | it counts the wrong-fauna answers |
| TTA adds 5 to 14 points on trap crops and costs 1 to 4 on photos (VN, VY, PB) | it sharpens `p(x|y)` where it is weak, so the prior weighs less |

Two more priors sit in the weights the same way: **framing** (a GBIF photo against a tight crop,
which is what padding at test time repairs) and **observer** (Z1). None of the three is a property
of the representation. So the design principle this project already has (prefer mechanisms that do
not need one constant to be right at every level) extends to deployment: **what is specific to a
site should be a separate quantity, estimated from the site's unlabelled images, and not a
retraining.** Self-training then has a smaller, better-defined job: teach the conditions, not the
species list. How much of its +8 to +22 probe is the list is measurable (route 1).

For the backbone the same reading gives the complementary principle: **the embedding should keep
more than the labels ask for; the classifier should read an invariant part of it.** Today's recipe
does the opposite to a foundation model (P5 keeps 44.6 of BioCLIP-2's 75.4 on Rare Species, PB).

## 3. Routes

Ranked by what they could change per unit of work. Each has a claim that can fail.

### Route 1. The site prior as a first-class, estimated component

**Claim.** Most of the gap between the test fold and the traps for an unadapted model is a wrong
label prior, and a prior estimated by EM from unlabelled crops recovers it without training, without
a curated list, and without killing the rare present species.

**Theory.** Under label shift the maximum-likelihood target prior is the fixed point of
`π(y) ← mean_i [ p_i(y) π(y) / Σ_k p_i(k) π(k) ]` (Saerens et al. 2002 [VERIFY]). It is exact only if
`p(x|y)` is unchanged, which is false here, so two things decide whether it works: calibration on
target crops (a margin head needs its temperature first, O3), and how sharp the posteriors still are.
The literature's advice is to calibrate on source validation data before the EM (Alexandari et al.
2020 [VERIFY]).

**Z4 held** (+14.4 accuracy, -2.4 to +0.2 macro recall), with no temperature and one view, so
this is the uncalibrated version. **What follows.**
- *Train for the likelihood, deploy with the prior.* Balanced softmax (L1) makes the network output
  something close to `p(x|y)`: it was the worst model under shift (0.573) because it was deployed
  with a uniform prior (Z1d: a quarter of its trap answers are rare absent species). With the site
  prior it may be the best, and the 2026-11-09 recipe decision (uniform sampling, because
  oversampling costs 6 to 7 probe points on F) would then be deciding the wrong thing. Test, eval
  only: L0, singlehead, L1, L2 logits on trap crops with the EM prior. Prediction: the ordering
  L0 > singlehead > L1 > L2 under shift flattens to within 2 points or reverses.
- *A cold-start prior from GBIF.* 5.9 M of our records carry coordinates and dates. A density
  `π(y | place, week)` (kernel, or a learned range model such as SINR, Cole et al. 2023 [VERIFY])
  gives a soft regional checklist with phenology, used as the Dirichlet prior of the EM. That covers
  the new site (X2) and the first week of a deployment, and keeps a rare species alive until the
  trap's own crops speak.
- *Decompose self-training.* Z4 says the gains substitute across two different models (Nemo +14.4,
  P5 +2.4). The controlled version applies the EM prior to one model before and after
  self-training (v2s_bu_s42 and its st26, DINOv3 and its st26: logits on UCloud). What remains after
  the prior is the condition adaptation that X1 showed transfers to unseen species (81 %). Then
  train it without the list: at each step, subtract the log of the current prior estimate from the
  logits of the pseudo-labelled crops (the logit-adjusted loss with a target prior), so the weights
  cannot store it. Prediction: such a model transfers to a held-out site as well as to a held-out
  night, which X2 says today's self-training does not.
- *Close the gap to the oracle* (0.778 against 0.887 for Nemo): a temperature fitted on source
  validation data before the EM, TTA views averaged first, and the prior per site and per week
  instead of one for all traps and both summers.
- *Order of operations.* The prior hides a new arrival (Z4, the pool without the held-out species),
  so the population test of route 2 runs on the unweighted posteriors, and the prior is applied to
  what passes it.
- The estimated prior is itself a product: a species list per site and season with uncertainty.

**The falsification line** (under 2 points of accuracy, or more than 8 of macro recall) was not
crossed. What can still fail: the EM on margin heads, whose posteriors saturate (O3), and on
sites with few crops.

### Route 2. Open-set as a population problem

**Claim.** A species the model cannot flag in one crop can be flagged in a month of crops, and the
right statistic is not the mean score but the absence of a confident core and the shape of the
cluster.

**Why.** Per-crop rules are bounded by one soft crop. Monitoring sees a new species repeatedly (an
invasive species arrives as a population: the OneStop use case). Z2: per-crop 0.57 to 0.75; per
group 0.92 to 0.97 with a one-line statistic; bags follow the square-root law.

**Theory to test: the shared-shift hypothesis.** Write a class mean in embedding space at a trap as
`μ_c(trap) = A μ_c(GBIF) + b + r_c`. If the domain shift is mostly a class-independent map `(A, b)`
of low rank, then (i) the map can be fitted on common species and applied to all, which transfers to
species never seen at the trap by construction, and (ii) novelty is the residual `r_c`: a group of
crops whose mean sits where no known class is predicted to sit. Cheap test: embed the labelled trap
crops with one model, fit `(A, b)` on half the species, report the variance of the other half's
displacement it explains. Above 70 % the map is the adaptation; under 30 % shift is class-specific
and this route narrows to score statistics.

**Then:** split a not-present group into "new species" (one tight cluster) and "confusion" (scatter
drawn from several present species): generalised category discovery (Vaze et al. 2022 [VERIFY]) with
the taxonomy as a prior on where a new cluster may sit.

**It needs a clean benchmark first.** The trap near stratum has 60 valid crops. One 20 M run that
withholds about 30 species common at the traps (100 crops or more each, checked against COL XR for
duplicates) gives several thousand truly unseen trap crops; the museum-style side has tol256-v1's
1,810 held-out species at three depths.

### Route 3. A second robustness axis with almost no noise: the observer

**Claim.** "In-distribution" is two populations. The new-observer part of the test fold is a shift
benchmark on the same species with hundreds of thousands of images, where the trap probe has 15,200
crops and a seed spread of up to 3.5 points (VS2).

**Steps.**
1. CPU: re-score every saved `predictions.csv` (about 200 folders) on S, O and N. If the ranking of
   heads and recipes on O/N follows the trap probe's, we have a well-powered proxy for shift; if it
   follows the test fold's, observer and trap shift are different things. Either is worth a line in
   the paper, and paper A's head comparison under shift currently has no powered benchmark.
2. One 20 M run on a fold split by observer (all records of a random 10 % of observers held out),
   for the number Z1 only bounds.
3. Tree of Life: tol256-v1 splits by observation. An evaluation list of held-out
   `copyright_owner`s, excluded from training like `heldout_taxa`, would have to be drawn before the
   full run starts (owner's decision; the freeze is otherwise untouched).
4. The same idea across imaging domains: `basis_of_record` and `img_type` separate museum specimens
   from field photographs for the same species. Species trained from one domain only and tested in
   the other is the trap problem with labels on both sides and five orders of magnitude more data.

### Route 4. The taxonomy as geometry: who is whose neighbour

**Theory.** With far more classes than dimensions (12,041 in 1,280; 190,199 in 768 to 1,536 for the
Tree of Life run) prototypes cannot all be far apart: every class has near neighbours, and species
cross-entropy does not care which. A loss on the marginals does. Probability leaked to a congener
costs nothing at genus rank and above; leaked to another family it is charged at every rank. The
cheapest packing puts congeners side by side. In-distribution almost nothing leaks, so the packing
is invisible in accuracy (finding 1). Under shift everything leaks to the neighbours, and the
neighbours decide whether genus and family stay right and whether the species error is a near miss.
The margin works against this: it pushes the nearest classes apart hardest, and the nearest should
be the congeners.

**Evidence so far.** Z3 (84 % against 17 to 23 %, confounded). Q1: margin heads' coarse confidences
equal the species confidence on uncertain trap crops, so a back-off rule has nothing to trade.

**Predictions, CPU-only on the campaign's checkpoints** (F, BU, I, TD; V2-S, B0, DINOv3; seeds
42 to 44):
- BU's congener-neighbour rate is above 60 % on every backbone; F and I are under 35 %.
- Across models, at equal species accuracy, a higher rate goes with higher genus and family scores
  on trap crops.
- The effect of BU against F grows with crowding (classes per dimension): larger on B0 (width 1,280
  but weaker features) than DINOv3, and largest in the Tree of Life pilot. Suggested pilot metric:
  for tol256-v1's held-out species, is the predicted genus (family) right? That is taxonomic
  placement of an unseen species, the task a backbone for biodiversity is for.

**Two constructions.**
- *A margin on the marginals.* mini_trainer's current option puts the margin on the species logit
  only; lepinet's `marginal_arcface` put it before marginalising and damaged calibration (finding
  11). The untested third form applies it to the true parent's marginal logit at genus and family,
  and none at species: space between genera, congeners left close. One run against BU.
- *Train the novelty instead of scoring it ("taxon dropout").* Finding 7 says the margin relocates
  open-set signal and does not create it. Supervision can create it, and C3b showed that withholding
  taxa costs 0.04 points. Each step, mask the logit columns of a random 5 to 10 % of species; their
  images get the target "a species of genus g that is not listed", one extra logit per genus inside
  the genus marginal (hierarchical novelty detection, Lee et al. 2018 [VERIFY], never tried at this
  scale or with a marginalising head). The output is then a posterior over tree nodes, leaves and
  "new child here" alike, which is open-set detection and rank back-off in **one** quantity instead
  of a score plus a threshold cascade. Predictions: near-novel AUROC on C3b's split above 0.92
  (from 0.8717); and on the 3,171 museum-style images of unseen species, the share given a species
  name under the release policy falls from 86 % (B8 and P5, Q1) to under 40 %. Risk: the extra logit learns "poor image", not "new
  species"; the truly withheld taxa are the test.

### Route 5. A backbone that stays general while it specialises

**The problem, measured.** Fine-tuned and adapted on moths, BioCLIP-2 loses 31 of 75 points of
5-shot accuracy on Rare Species at the same layer (PB). Meghan, the same trunk frozen, keeps them
and is 9 points worse on AMI-Traps. The recipe converts a foundation model into a specialist, and
the Tree of Life model will meet the same fate when it is specialised for traps.

**Theory.** Cross-entropy at its optimum keeps nothing in the embedding except class identity
(neural collapse, Papyan et al. 2020 [VERIFY]); a margin pushes further that way by design (intra
0.667). What a later task needs, life stage, sex, health, a species outside the label set, is
within-class or off-label variation, the part the objective removes. This is also the honest prior
for ME: the margin arm should win species retrieval and lose the non-species tasks.

**Steps, cheapest first.**
1. *The interpolation path, no training.* `θ(α) = (1-α) BioCLIP-2 + α P3c` (weight-space ensembles,
   Wortsman et al. 2022 [VERIFY]), read out with class centroids (H1: -0.29 points), so every α can
   be scored on the test fold, the probe and Rare Species without fitting anything. Prediction: the
   trade is concave; some α in 0.5 to 0.8 keeps the test fold within 1 point and recovers at least
   half of the 31. Falsified if it is linear.
2. *A soup of Otto's runs.* Otto, OT1 to OT5 and the three site folds share one initialisation and
   one recipe; their weight average is an ensemble at single-model cost (model soups [VERIFY]).
   Probability ensembles gave +1.2 to +3.2 probe. Eval only.
3. *Fine-tune with a retention term:* a distillation loss to the frozen trunk's embedding on a
   10 % stream of non-moth Tree of Life images, or low-rank updates. One factor on P3c.
4. *An embedding in two parts, supervised by metadata we already hold.* `e = [e_id | e_ctx]`. The
   species loss reads only `e_id`. Heads on `e_ctx` predict what GBIF and ToL give for free:
   `lifeStage`, `sex`, `basisOfRecord`, `img_type`, source, month, latitude. A decorrelation penalty
   keeps the two apart. Users get both parts; the classifier gets the part that was told to ignore
   the imaging domain. Predictions: better than a species-only margin model on the non-species tasks
   of BioCLIP-2's suite, and a higher probe before adaptation. This is the third arm ME lacks: not
   margin against contrastive, but margin plus a reason to keep what the margin discards. A
   self-distillation term (DINO, iBOT) is the generic version of the same reason.
5. Track generality during training as val loss is tracked: Rare Species 5-shot and few-shot on
   tol256-v1's held-out taxa, every epoch.

### Route 6. The individual as the unit: tracks

Traps photograph every 10 minutes; a resting moth is in consecutive frames. The project removes
these as duplicates. They are the cheapest supervision in the target domain:
- *Inference:* average the logits along a track. TTA with synthetic views gives +5 to +11; these
  are real views (pose, blur, light).
- *Self-training:* label the track from its pooled prediction and train on **all** its crops. The
  confidence gate keeps easy crops (species answers fall from 91 % on sharp crops to 65 % on blurry
  ones, N1), so today's pseudo-labels never teach the hard views. A track carries the label from the
  sharp frame to the blurry one.
- *Representation:* same-individual pairs are positives without labels, for 50 M European crops.
- The ecological quantity is individuals per species per night, so the unit is the right one anyway.

Prediction: on tracks of three or more crops, pooled prediction adds at least 3 points of probe
accuracy; pseudo-labels reach the gate's 98.6 % accuracy at 85 % coverage instead of 70 %. Not
testable on the labelled crops (their filenames carry trap, snapshot and crop number, no box
position). It needs the pipeline's boxes or its tracker output: the open question of U (what metadata
the European crops carry) decides it.

### Smaller, adjacent

- *Match the framing at training time.* Run the trap pipeline's detector on GBIF photos and train on
  its crops. One padded view at test time is worth +5.3 for Nemo, and AG's geometric augmentation
  failed because training never saw a tight crop; this removes the mismatch at its source.
- *Synthetic trap crops for all 12 k species:* segmented GBIF moths pasted on trap sheets (S1a builds
  the composites for evaluation). Self-training covers the 300 species that are at the traps; this
  covers the rest with true labels.

## 4. Order of work I would propose

| step | what | cost |
|---|---|---|
| 1 | the EM prior beyond Z4: with a temperature and TTA views, per site and week, on OT3 and the campaign's before/after self-training pairs, on AMI-Traps | CPU (saved logits), minutes once `em_prior` works in place |
| 2 | observer split of every `predictions.csv`; congener-neighbour rate of the campaign checkpoints | CPU, head weights copied from UCloud |
| 3 | eval only: L0 / singlehead / L1 / L2 with the site prior; interpolation path with centroid readout; Otto soup; shared-shift fit | about 1 GPU-day, workstation |
| 4 | 20 M runs, one factor each: taxon dropout on C3b's split; margin on the marginals against BU; observer-disjoint fold; trap-common species withheld | 4 runs of 6.4 h |
| 5 | Tree of Life pilot (16-20 Nov): add placement of held-out taxa, congener rate and few-shot on held-out taxa to what is logged | no extra run |
| 6 | after the paper: the two-part embedding as ME's third arm; tracks once the European crops' metadata is known | with ME and U |

Steps 1 to 3 change no plan and no GPU queue. Nothing here is started without the owner.

## 5. What this touches in existing claims (for the owner)

- **Finding 5's explanation** (paper §4.13: tail weighting "up-weights the least transferable part
  of the training signal"; "the test fold shares those artefacts"). The trade stands; Z1c and Z1d say
  the cause is the label prior. The sentence about cRT removing the trade should be re-read in that
  light.
- **N2** (RESULTS, the N entry): the near-novel trap numbers are a duplicated class. Annotated in
  RESULTS; the N entry is frozen and left as written. `paper/merge` does not quote 0.566.
- **The recipe decision of 2026-11-09** (uniform sampling for shift) is worth revisiting once step 3
  has prior-corrected numbers.
- ***Tethea or*** is missing from the GBIF download.

## 6. Limits and mistakes

- Z1's groups are not matched for photo quality or observer experience; +7.4 is an upper bound.
- Z2a's ghost groups were 11, seven of them one species, and that species turned out to be a
  duplicate; the valid sample is four groups.
- Z2c: my recall prediction was wrong by an order of magnitude (above).
- Z3a: wrong for two of three models; I assumed visual similarity alone would make congeners
  neighbours.
- Z3 and the B8 contrast are confounded three ways. No claim rests on them yet.
- Z4 is one country, 12 traps, one view, no temperature, and its "substitutes" reading compares two
  models that differ in backbone and training. The prior was estimated on the adapt nights of the
  same traps and summers as the probe nights; only the leave-one-site-out rows say anything about a
  new place.
- All citations are from memory and marked [VERIFY].
- Section 7: everything is the 12 Danish traps, accuracy and zero-threshold scores of my own, not
  the calibrated (trap, night) macro-F1 the papers quote. Z8 is one seed per backbone, single view,
  and the unadapted DINOv3 is another seed than its self-trained partner. The list was chosen on
  these crops (VN). Nemo's training sampler is unknown to me, so Z5d cannot say whether the
  division fails because of shift or because Nemo's training prior was already flat.
- Mistakes of mine on 2026-11-15: the first commit placed PLAN's Z row under the wrong heading
  (fixed in 3f857e9); a wait loop matched its own command line and never ended (stopped by hand);
  and the first report's headline compared the EM with the full vocabulary instead of the list.
- Section 8: one seed per backbone and the unadapted DINOv3 is seed 43 (the calibrated probe score of
  three unadapted seeds spans 3 points on V2-S and 1 on DINOv3). No unadapted BioCLIP-2 BU exists,
  so nothing in it is a like-for-like statement about Otto. The EM prior is estimated on the same
  traps and summers. Z9d and the EM on self-trained models count their own trap prior twice. Z10b
  cannot see absent species. Probe-HO moves with the threshold convention.
- A mistake of mine on 2026-11-15, section 8: Z10 was computed on logits stored minus their
  per-crop maximum for six models and reported for an hour in this entry as -61 % and -39 %. I had
  not read the dump tool's `fp16s` note. Caught when a second view was dumped; Z10b replaces it,
  and DINOv3's result changed from a clear gain to none.
- Routes 5.4, 4 (taxon dropout) and 6 are designs with no measurement behind them.

## 7. First steps (opened 2026-11-15, at the owner's request)

The owner asked to proceed with steps 1 and 2 of section 4, and noted that Nemo is not our best
model. That is right: Otto (OT3) and P5 lead on trap crops, B8 on the test fold. Z4 used Nemo
because it never saw a trap image, so the prior's effect is not mixed with self-training's. Whether
the prior still pays on the models we would ship, in the setting we would ship them (TTA, a regional
list), is the first thing to measure. CPU only, `dev/107` (`siteprior2`, `observers`).

### A correction to Z4 before anything else

Z4 computed `argmax p(y|x) · w(y)`, with `w` the point of the simplex that maximises the likelihood
of the unlabelled crops. The label-shift rule is `p(y|x) · π_site(y) / π_train(y)`: the training
prior has to be divided out. Z4's `w` equals `π_site / π_train` only for a model whose training
prior is uniform, and its "oracle" row multiplied by the true trap frequencies without dividing
either. The numbers of Z4 are what that procedure gives; the name "site prior" was loose. Z5d tests
the rule as written.

### Predictions (written before any of these ran)

**Z5, the site prior on Otto and P5, in the shipped setting.** Saved logits of Nemo, P5 and OT3;
one view, and tta3bm (mean logits of three views). Probe nights, same three scores as Z4. "EM" is
Z4's smoothed estimate unless said otherwise.

- Z5a. OT3 with one view gains +1 to +4 accuracy from the EM prior (P5 gained +2.4 to +2.8), and at
  least +10 macro-F1 in mini_metrics' convention.
- Z5b. The prior's accuracy gain is smaller with tta3bm than with one view, for every model; for
  Nemo by at least 3 points. Reason: TTA sharpens `p(x|y)`, so the prior weighs less (section 2).
  Falsified if the two gains are within 1 point.
- Z5c. On Nemo the EM prior beats the north_europe list (1,977 species, a prior of zeros and ones)
  by at least 3 points of accuracy; the list keeps the higher macro recall; list and EM together
  are within 0.5 point of EM alone.
- Z5d. Dividing by the training prior (class frequencies to the power 1, and 0.5 for a
  square-root sampler) changes Nemo's accuracy by less than 2 points either way and raises macro
  recall over the labelled species by at least 2 points, because a trap species that is rare on
  GBIF is no longer handicapped. Same direction for the oracle row.
- Z5e. With a temperature T on the logits before the EM, accuracy peaks at T between 1.4 and 2 and
  the peak is worth at most 3 points over T = 1; macro recall falls as T rises. So the temperature
  is not where the gap to the oracle is.
- Z5f. A prior for each probe night from the same site's crops within 15 nights, shrunk toward
  the pooled estimate with a weight of 100 crops, gives Nemo at least +2 accuracy over the pooled
  prior (moths are seasonal). Falsified under +0.5. And a prior estimated on the probe crops
  themselves, with no separate adaptation set, is within 1 point of the adapt-night estimate.

**Z6, the observer split on every saved test-fold prediction file** (lepinet's runs and the
campaign's seed-42 heads).

- Z6a. The paired S - N recall gap is 5 to 10 points for every EfficientNetV2-S run, and at least
  1.5 points smaller for the large pretrained backbones (DINOv3 ConvNeXt-L, BioCLIP-2).
- Z6b. Across runs, recall on N follows recall on S (Spearman at least 0.95), and the S - N gap does
  not predict probe accuracy among runs that never saw a trap image (absolute Spearman under 0.3).
  That would mean observer shift and trap shift are different things, which Z1c already suggests.
  If it is 0.5 or more, the split is a powered proxy for the trap probe.
- Z6c. Trap augmentation and self-training leave the gap unchanged (within 1 point).

**Z7, congener neighbours on the campaign's heads**: the predictions of route 4 stand (BU above
60 % on every backbone, F and I under 35 %). Needs the checkpoints from UCloud.

**Added after the first rows of Z5 were read (Nemo and OT3, one view), before these ran.** Those
rows showed the list and the EM estimate together far above either alone on Nemo (0.867 against
0.795 and 0.778), and no accuracy gain from the EM on OT3.

- Z5g. The EM run inside the list (only the list's species can receive mass) is at least as good as
  masking the unrestricted estimate: Nemo at or above 0.867, because mass the EM gave to a foreign
  look-alike returns to the local species.
- Z7b, added after Z7's table was read (flat 54 %, marginal 69 to 74 %) and before this ran: if the
  geometry matters it shows in the errors. Among wrong species answers, the share that names a
  congener is at least 3 points higher for the marginal head than for the flat one, on the test
  fold and on the probe crops, for both backbones with per-image predictions (V2-S and B0, seed 42).
  And the statistic survives centring the prototypes (they share a large common direction).
- Z8a, the controlled decomposition: one recipe before and after self-training (V2-S BU seed 42 and
  its st26; DINOv3 BU and its st26), single-view logits dumped on the workstation GPU. With the
  list and the EM, the unadapted model gains at least +12 accuracy and the self-trained one at most
  +3, so the prior closes at least 60 % of the gap between them. What is left (3 to 8 points) is
  what self-training teaches that is not the species list.
- Z8b, a new site: the site folds (self-trained without one site; V2-S and Otto's BioCLIP-2) scored
  on the probe crops of the site they never saw, with the prior estimated on that site's own
  unlabelled crops. The prior adds at least +3 accuracy and brings the fold within 2 points of the
  model self-trained on all three sites, at that site. Falsified under +1. This is the claim of
  route 1 that matters for deployment: a new trap needs its crops, not a retraining.

### Results (2026-11-15)

Outputs: `data/population_probes/z5_siteprior.csv`, `z5g_inlist.csv`, `z6_observers.csv`,
`z7_congeners.csv`, `z7b_severity.csv`, `z8_v2s.csv`, `z8_large.csv`. Two checks that the harness
is right: the faster EM reproduces Z4 to four decimals (Nemo 0.7768 and 0.7784, P5 0.8599 and
0.8639; 200 iterations change nothing), and OT3 with the list scores macro-F1 0.675 (one view) and
0.674 (tta3bm) at zero threshold, the campaign scorer's numbers (`ot_scores.md`).

#### Z5: the regional list is the prior that matters; the EM adds accuracy only where nothing else has

Probe nights, accuracy / macro-F1 over labelled and predicted species / macro recall over labelled
species. "List" is north_europe (1,977 species, 480 of the 486 trap species). "True frequencies"
uses the adapt nights' labels.

| model | views | nothing | list | EM | list and EM | true frequencies |
|---|---|---|---|---|---|---|
| Nemo | one | 0.633 / 0.137 / 0.597 | 0.795 / 0.295 / 0.752 | 0.778 / 0.412 / 0.599 | **0.867** / 0.608 / 0.678 | 0.887 / 0.655 / 0.707 |
| Nemo | tta3bm | 0.781 / 0.205 / 0.731 | 0.853 / 0.380 / 0.802 | 0.888 / 0.618 / 0.740 | **0.898** / 0.688 / 0.754 | 0.915 / 0.712 / 0.753 |
| P5 | one | 0.836 / 0.394 / 0.792 | 0.879 / 0.504 / 0.812 | 0.864 / 0.575 / 0.774 | **0.901** / 0.649 / 0.783 | 0.915 / 0.688 / 0.747 |
| P5 | tta3bm | 0.856 / 0.406 / 0.803 | 0.894 / 0.536 / 0.822 | 0.889 / 0.667 / 0.785 | **0.912** / 0.708 / 0.789 | 0.929 / 0.723 / 0.766 |
| OT3 | one | 0.903 / 0.630 / 0.808 | **0.910** / 0.675 / 0.815 | 0.896 / 0.705 / 0.756 | 0.901 / 0.711 / 0.759 | 0.925 / 0.714 / 0.753 |
| OT3 | tta3bm | 0.918 / 0.599 / 0.823 | **0.922** / 0.674 / 0.834 | 0.916 / 0.735 / 0.781 | 0.918 / 0.737 / 0.782 | 0.938 / 0.741 / 0.775 |

- **My first report overstated the EM prior.** Its +14.4 for Nemo was measured against the full
  vocabulary. The pipeline already ships a regional list, and the list alone gives Nemo +16.3 accuracy
  **and** +15.5 macro recall. Against the shipped setting (list, tta3bm) the EM adds +4.6 to Nemo,
  +1.8 to P5 and -0.5 to OT3, and it costs 3 to 5 points of macro recall on every model.
- **Z5a is falsified.** On OT3 the EM gives no accuracy (-0.7 with one view, -0.2 with tta3bm). It
  raises macro-F1 in mini_metrics' zero-threshold convention by 4 to 14 points depending on the
  setting (list and tta3bm: 0.674 to 0.737) because it removes species that are predicted and not there. Not scored
  with the calibrated (trap, night) protocol, where thresholds already do part of that.
- **The three models converge.** With nothing: 0.633 / 0.836 / 0.903. With list, EM and tta3bm:
  0.898 / 0.912 / 0.918. An EfficientNetV2-S that never saw a trap image ends 2 points of accuracy
  under a BioCLIP-2 ViT-L self-trained on these traps (and 8 points under it in macro recall).
- **Z5c: wrong in two parts of three.** The list alone beats the EM alone with one view, and the two
  together are far above either. They do different jobs: the EM cannot tell a local species from a
  foreign look-alike the model prefers, the list can (crops named as a species absent from the
  traps: 33.1 % with nothing, 15.3 % with the EM, 1.3 % with both); the list cannot tell which of its
  1,977 species are at the trap. The list does keep the higher macro recall, as predicted.
- **Z5d is falsified, and it is the instructive one.** Dividing by the training prior, which is the
  rule as the theory writes it, costs Nemo 7.0 points with the square-root exponent and 19.2 with
  the full one (0.708 and 0.586 against 0.778). With the true trap frequencies the division changes
  nothing (0.883 to 0.887), and on P5 and OT3 it changes nothing either (-0.8 to +0.1). Measured
  cause: the EM then puts 34 % of its mass on rare species absent from the traps (7.5 % without the
  division), and 35 % of crops are named as one. It is finding 5 again: whatever divides the GBIF
  prior out hands the answer to the tail, because under shift the posteriors of rare classes are
  noise and the division multiplies them by thousands. So the correction above was right as a
  description and wrong as a recommendation.
- **Z5b held for Nemo only** (gain of the EM +14.6 with one view, +10.7 with tta3bm). On P5 and
  OT3 the two gains are within a point.
- **Z5e held.** T = 1.4 is best for Nemo (+1.3 over T = 1) and macro recall falls with T (0.625 at
  0.5, 0.406 at 2). On OT3 any T above 1 hurts. Both models are close to calibrated on the adapt
  nights (lowest NLL between T = 1 and 1.4).
- **Z5f: not met.** A prior per site and season adds at most +1.3 to Nemo (all sites, nights within
  15), +0.8 per site, +0.4 with only the 30 nights before, and nothing to P5 and OT3. The pooled
  estimate is nearly as good. A prior from the probe crops themselves: -1.4 (Nemo), +0.1 (P5), +0.3
  (OT3), so an adapted model needs no separate adaptation set.
- **Z5g falsified by a little.** The EM inside the list gives Nemo 0.860 against 0.867 for masking
  the unrestricted estimate (macro recall 0.696 against 0.678); P5 and OT3 are unchanged.
- **Headroom.** The true frequencies are worth +1.4 to +2.0 accuracy over the best label-free rule on
  every model, OT3 included (0.922 to 0.938 with tta3bm). That bounds what any better estimate of
  the prior can buy.

#### Z8: self-training's accuracy gain is mostly the fauna, which a list gives for nothing

One recipe before and after self-training (campaign checkpoints: V2-S BU seed 42 and its st26;
DINOv3 BU and its st26, where the unadapted model is seed 43 because the seed-42 checkpoint is
gone), single view, logits dumped on the workstation GPU with the campaign's
`eval/tools/logits.py`. Probe nights, accuracy / macro recall:

| | nothing | list | list and EM |
|---|---|---|---|
| V2-S BU, never saw a trap | 0.657 / 0.624 | 0.790 / 0.744 | **0.835** / 0.654 |
| V2-S BU self-trained (st26) | 0.824 / 0.741 | **0.855** / 0.777 | 0.828 / 0.603 |
| DINOv3 BU, never saw a trap | 0.699 / 0.689 | 0.797 / 0.774 | **0.847** / 0.658 |
| DINOv3 BU self-trained (st26) | 0.846 / 0.757 | **0.877** / 0.784 | 0.859 / 0.658 |

- **Z8a held** (+17.8 for the unadapted model, +0.4 for the self-trained one). Self-training is worth
  +16.7 accuracy with the full vocabulary. The list alone gives the unadapted model +13.3 of them and
  all of the macro recall (0.744 against 0.741). **On top of the list, self-training adds +6.5
  accuracy and +3.3 macro recall**; the EM adds +4.5 accuracy and takes 9.0 of macro recall. Each
  model with its best rule: 0.855 against 0.835, a gap of 2.0 where I predicted 3 to 8.
- **DINOv3 says the same:** self-training is worth +14.7 with the full vocabulary, the list gives
  +9.8 of it, and on top of the list self-training adds +8.0 accuracy and +1.0 macro recall (the
  EM: +5.0 accuracy, -11.6 macro recall). Best rule each: 0.877 against 0.847.
- So the two are not the same thing. The list and the EM buy crops of the common local species;
  self-training also improves the rare ones. One seed per backbone.
- **Z8b is falsified on V2-S.** The site folds (self-trained without one site), at the site they never saw,
  accuracy with nothing / list / list and EM from that site's own crops: LV 0.820 / 0.850 / 0.829,
  OH 0.762 / 0.803 / 0.750, SS 0.807 / 0.832 / 0.793. The site's prior adds nothing to a model that
  was self-trained elsewhere (-1.4 to +0.9), and is 2 to 5 points under the list. The same holds
  for the model self-trained on all sites.
- What does hold at a new site: the model that never saw any trap, with the list and the site's own
  EM prior, scores 0.850 / 0.805 / 0.841 (LV / OH / SS). That equals the folds with the list (0.850 /
  0.803 / 0.832) and is 1 to 2 points under the model self-trained on the site itself (0.864 / 0.813
  / 0.863), with 9 to 11 points less macro recall.
- The folds lose little at the site they never saw (-0.8, -2.1, -3.4 accuracy against the all-site
  model, full vocabulary). With the list on both, a fold keeps 81 / 77 / 51 % (LV / OH / SS) of what
  self-training on the site itself adds over the unadapted model.

- **On Otto's own folds the prediction mostly holds, and the contrast says why.** Otto (BioCLIP-2,
  the first candidate, labels from the per-species top 35) and its three folds, one view:

  | | nothing | list | list and EM |
  |---|---|---|---|
  | Otto, all probe crops | 0.773 / 0.772 | 0.827 / 0.796 | **0.862** / 0.769 |
  | OT3, all probe crops (Z5) | 0.903 / 0.808 | **0.910** / 0.815 | 0.901 / 0.759 |

  Folds at the site they never saw, accuracy with nothing / list / list and that site's EM (Otto
  trained on all sites, same rules, in brackets): LV 0.788 / 0.835 / 0.874 (0.763 / 0.825 / 0.875),
  OH 0.780 / 0.819 / 0.827 (0.793 / 0.824 / 0.837), SS 0.772 / 0.813 / 0.839 (0.778 / 0.831 /
  0.863). The site's prior adds +3.9 / +0.8 / +2.6 over the list and leaves the fold 0.1 / 1.0 / 2.4
  points under the all-site model.
- **A reading, not yet a test:** whether self-training stores the fauna depends on how its labels
  were drawn. Otto's were the top 35 crops per species, 8,305 crops over 913 species where the adapt
  nights hold 451, so they taught no site frequencies and many absent species: the EM still helps
  (+3.5 over the list). OT3's and st26's are the most confident crops at their natural frequencies
  (19,061 crops, 314 species): the EM is redundant or harmful. Each with its best label-free rule,
  the gap between Otto and OT3 falls from 13.0 points to 4.8. So part of what the gated labels bought over
  the top-35 labels (attributed to label accuracy, 71 % against 98.6 %) may be the site's
  frequencies. OT1's logits (gated labels alone) would separate the two; not on this machine.

#### Z6: an unseen observer doubles the error rate, for every model alike

58 test-fold prediction files (39 EfficientNetV2-S, 12 DINOv3 ConvNeXt-L, 6 B0, 1 FastViT; no
BioCLIP-2 run has per-image test predictions on this machine). Paired macro recall as in Z1.

| backbone | runs | S - N gap, points (range) | S - O gap | error rate N / S | error rate O / S |
|---|---|---|---|---|---|
| B0 | 6 | 9.3 (8.4 to 10.7) | 4.6 | 1.86 | 1.35 |
| EfficientNetV2-S | 39 | 7.8 (6.7 to 9.1) | 4.0 | 1.96 | 1.46 |
| DINOv3 ConvNeXt-L | 12 | 6.7 (6.3 to 7.3) | 3.7 | 2.08 | 1.54 |

- **Z6a: first half held, second not.** The gap is 1.1 points smaller on DINOv3 (I predicted 1.5),
  and only because it makes fewer errors. As a ratio the penalty is the same or larger: an unseen
  observer multiplies the species error rate by 1.98 (sd 0.12 over 58 runs), a known observer with a
  new species by 1.47 (sd 0.10). A large pretrained backbone is not less tied to the photographer.
- **Z6b held.** Recall on N follows recall on S (Spearman 0.974), and among the 31 runs that never
  saw a trap image the gap does not predict probe accuracy (Spearman -0.13, p 0.50; S - O: -0.09).
  Observer shift and trap shift are different things, and the observer split re-ranks nothing.
- **Z6c held.** Trap augmentation and self-training leave the gap where it was (B1 8.4 against A1
  8.4; B3 8.2; B4 6.8 against A2 6.9; the campaign's gated and real-label runs 7.5 and 7.0 against
  7.4). Repeats of one recipe differ by 0.03 (singlehead and A5) and 0.2 (the two cap-1000 runs).
- **What it is good for:** an honest in-distribution number. On images from observers absent from
  training the error rate is twice the published one, for every model. It is not a tool for
  choosing between models. Route 3 shrinks to a caveat and one supplementary table.

#### Z7: marginal supervision makes congeners neighbours, on every backbone

The campaign's heads, three seeds each (DINOv3 BU has two: the seed-42 checkpoint is no longer on
UCloud). These are cosine heads, so the direction of a prototype is the geometry. Share of species
whose nearest other prototype is a congener (chance 0.1 %); in brackets, with the prototypes'
common direction removed.

| backbone | flat (F) | marginal (BU) | independent (I) | conditional (TD) |
|---|---|---|---|---|
| B0 | 54.0 (71.4) | **68.4 (81.5)** | 59.7 (74.0) | 68.1 (71.0) |
| EfficientNetV2-S | 53.5 (70.5) | **69.2 (81.0)** | 56.9 (73.8) | 64.6 (68.7) |
| DINOv3 ConvNeXt-L | 56.1 (70.0) | **74.4 (81.5)** | 57.1 (70.6) | 52.5 (61.8) |

- **One factor, three backbones:** the marginal loss raises the rate by 14 to 18 points (10 to 12
  centred). The seed range is at most 2.7 points, and at most 1.0 centred. Nearest in the same
  family, centred: 92 % flat, 95 % marginal.
- **Scored against route 4:** "marginal above 60 % on every backbone" held. "Flat and independent
  under 35 %" is falsified: a flat head already puts a congener nearest for more than half of the
  species. "The effect grows with crowding" is falsified: it is the same on the three backbones.
- Z3's low rates (17 and 23 %) were margin heads in lepinet, the campaign's are plain cosine heads,
  so the margin may be what lowers it; package and recipe differ too. Untested.
- **Z7b: it shows in the errors on the test fold and not on the traps.** Among wrong species answers,
  the share naming a congener (seed 42):

  | | test fold, flat | test fold, marginal | probe, flat | probe, marginal |
  |---|---|---|---|---|
  | EfficientNetV2-S | 53.0 % | **57.2 %** | 19.1 % | 18.5 % |
  | B0 | 48.5 % | **51.9 %** | 19.4 % | 19.2 % |

  +4.2 and +3.4 on the test fold at equal accuracy (0.9316 and 0.9307 on V2-S), nothing on the probe
  crops. So route 4's claim that the neighbourhood decides what happens under shift is not supported:
  on trap crops four errors in five leave the genus whatever the head, because they go to species
  that are not there (Z1d), not to neighbours.

### What the first steps change

| route | before | after |
|---|---|---|
| 1, site prior | "a first-class component, +14 for Nemo" | the fauna is the lever and the regional list already pulls most of it. The EM is a label-free baseline that brings a model with no trap exposure to within 2 points of accuracy of a self-trained one, at a cost in rare species. It adds nothing to a self-trained model, here or at a new site. Never divide by the training prior. |
| 3, observer | "a second, well-powered robustness axis" | a constant factor of two on the error rate that no recipe of ours moves. A caveat for both papers. |
| 4, geometry | confounded observation | confirmed one factor at a time; visible in test-fold errors, absent under trap shift. A piece for paper A. |
| 2, 5, 6 | untouched | untouched |

**Claims this touches.** Self-training's headline gain is measured against the full vocabulary. On
V2-S BU it is +16.7 accuracy there and +6.5 once both models use the regional list (DINOv3 BU:
+14.7 and +8.0). Paper B should
report both, or the reader who deploys with a list will find a third of the promised gain. The same
recomputation on the calibrated macro-F1, with the campaign scorer, is the next thing to do and I
have not done it.

**For paper A (owner's question, 2026-11-15).** In order of what I would add:
1. Z7 with Z7b, into the piece on heads, parameters and supervision. The paper's result is that
   heads do not differ in accuracy; this says what the marginal loss does change, with one factor,
   three backbones and three seeds: which classes are neighbours, and which errors are made on the
   test fold. It is the mechanism behind the marginal head's higher genus and family scores at equal
   species score. Stated with its limit: no effect on trap crops.
2. Z1c and Z1d, into the long-tail piece: the cost of tail weighting under shift is predictions of
   rare species that are absent (3.4 / 10.1 / 23.8 / 34.4 % of crops), not photograph-specific
   features; Z5d shows the same mechanism from the other side.
3. Z6, as one sentence and a supplementary table: the head ranking is the same for unseen
   observers, and the test fold is 68 % "same observer, same species".
4. Z1d and Z5 for the metric-conventions piece: at zero threshold the convention that counts
   predicted-only species moves from 0.14 to 0.69 for one model when a list and a prior are applied
   and nothing is retrained. It measures the fauna assumed as much as the classifier.

**One-step self-training (owner's question, 2026-11-15).** Today's recipe trains a teacher, labels
the trap crops, and trains a student on GBIF plus those labels. Folding the two into one run is an
existing family of methods: the model labels the unlabelled crops itself during training, through
a slow average of its own weights, and a crop joins the loss once its prediction is confident
(FixMatch and Mean Teacher; AdaMatch for a shifted target; FreeMatch for a threshold that follows
the model's confidence, which is the "as entropy falls" idea) [VERIFY, from memory]. What this
project already knows about how it would go:
- Entropy alone is the wrong gate here. R2 lost 3.8 points with cleaner labels because the confident
  crops covered half the species; coverage of species matters more than label accuracy, and a
  confidence gate keeps the sharp crops of common species (N1). The inclusion rule has to be per
  species (a threshold per class, or a cap per class as in VE), and confidence on trap crops should
  be read relative to confidence on GBIF, not on an absolute scale.
- A teacher now exists. Otto, OT3 and any model trained from here on take their labels from an
  ensemble of earlier models, so they are already one training run. One-step matters where there
  is no teacher: a new taxon, a new camera.
- Label rounds have saturated (OT5 against OT3), so a teacher that improves during training should
  not be expected to beat a good fixed one.
- Z8a: most of what self-training buys in accuracy, the regional list buys for nothing. What is
  specific to it (+6.5 accuracy, +3.3 macro recall over the list on V2-S BU) is the part worth
  making cheaper or better.
If wanted, the test is one 20 M run against st26: trap crops in every batch from the first epoch,
labels from the weight average, a per-species confidence rule, the loss weight ramped up, and
pseudo-label accuracy and species coverage logged per epoch (the adapt crops have labels, so both
are measurable). My expectation, not a measurement: within 1.5 probe points of the two-step run
with its own teacher, and under the ensemble-teacher version.

## 8. Does self-training repair the images or the label prior? (opened 2026-11-15, owner's question)

The owner read section 7 as "self-training is tackling a label shift, not a domain shift" and asked
how that can be affirmed. Section 7 does not show it. This section states the two shifts, what
section 7 does show, and three checks that can separate them on saved logits.

**The two shifts.** A classifier's answer is `p(y|x) ∝ p(x|y) · π(y)`: what the image says about
each species, times how common each species is assumed to be.
- **Label shift**: `π(y)` changes, `p(x|y)` does not. Training holds 12,041 species worldwide,
  sampled to be nearly even; a Danish trap holds 486, a few of them most nights. It can be repaired
  at prediction time with no labels and no training: multiply the posteriors by the new prior.
- **Appearance shift** (what "domain shift" usually means): `p(x|y)` changes. The same species
  looks different on a trap crop (resting posture, white sheet, flash, low resolution) from a
  photograph. No prior repairs it: the model has to be trained on such images, or shown better views.
- They interact. Weak evidence makes the prior decide, so a wrong prior costs little on photographs
  and a lot on trap crops. Either side can therefore repair part of the other's damage, and an
  accuracy gain does not say which side moved.

Self-training moves both at once: the pseudo-labelled crops are trap images (appearance), and they
come in the trap's species and frequencies (prior). Z8 shows that a prior alone, with no training,
brings an unadapted model to the self-trained model's accuracy (0.835 against 0.824 on V2-S, 0.847
against 0.846 on DINOv3). That bounds the prior's share from below. It does not show that
self-training learned nothing about the images.

### Predictions (written before any of these ran)

**Z9, Z8 on the papers' metric.** The same pairs scored with the campaign's scorer
(`flemming.py score`, species macro-F1 calibrated by (trap, night), 20 splits), the prior added to
the species logits as a log-prior, genus and family by logsumexp as `score.py` does. Harness check
first: OT3 with the list must reproduce 0.742 / 0.675 (one view) and 0.749 / 0.674 (tta3bm).
- Z9a. With the list on both models, self-training's gain in calibrated macro-F1 on probe is 40 to
  55 % of its gain with the full vocabulary (in accuracy: 39 % on V2-S, 54 % on DINOv3).
- Z9b. The EM prior does **not** close the gap on this metric. Calibration already drops the
  low-confidence predictions of absent species, and the EM takes rare species away. List and EM on
  the unadapted model lands within 2 points of the list alone; on the self-trained model it costs 2
  to 6. Best label-free rule against best: at least 8 points of calibrated macro-F1 stay with
  self-training, where accuracy left 2 to 3.
- Z9c. On probe-HO the EM costs every model more than on probe (its species are rare in the pool).
- Z9d, a diagnostic and not a method: with the probe's own species frequencies as the prior on both
  models, the accuracy gap falls under 3 points (from 16.7 and 14.7) and the calibrated macro-F1 gap
  under 6.

**Z10, discrimination that no prior can move.** The one-vs-rest AUROC of a species' raw logit over
the probe crops is unchanged by any constant added to that species, so by any label prior, a list
included. What self-training changes there, it learned about trap images. Prediction: it lowers
the mean ranking error (1 - AUROC, species with at least 5 probe crops) by 20 to 40 % on both
backbones. Not zero, because a third of the pseudo-labels' gain needed the trunk (VK); less than
the 49 % by which it cuts the error rate on both. Against congeners only (negatives restricted to
the same genus) the cut is smaller, under 20 %. On the held-out species it is under 10 %.

**Z11, is there appearance shift at all?** Under label shift alone a species keeps its row of the
confusion matrix, so its recall with a fixed model is the same on photographs and on trap crops.
For the unadapted V2-S BU, over the probe's species: recall above 0.90 on the test fold and 0.62 on
the probe (Z8), a gap above 25 points. Z7b already points there (53 % of errors name a congener on
the test fold, 19 % on the probe).

**Z12, do the labels' frequencies decide whether a prior still helps?** (added after Z10 and Z11
were read, before OT1's and OT2's logits were downloaded.) Otto, OT1, OT2 and OT3 are a two-by-two
on one backbone and seed: labels drawn as the top 35 crops per species (Otto, OT2) or as the 70 %
most confident crops at their natural frequencies (OT1, OT3), without or with TrivialAugmentWide.
Section 7's reading says the gated labels carry the site's frequencies and the top-35 labels do
not. Prediction, one view, accuracy of list and EM over the list alone: +2 to +4 on Otto and OT2,
within 1 point of zero on OT1 and OT3. If OT1 gains as much as Otto, the reading is wrong.

**Z13, are they the same errors?** (added after Z10, Z11 and the first rows of Z9 were read.) Z10
says self-training learned about the images; Z8 and Z9 say a prior alone reaches about the same
score. Both can be true if the two remove the same wrong answers by different means. Crop by crop
on the probe, V2-S and DINOv3, one view: of the crops the unadapted model gets wrong and the
self-trained one gets right (full vocabulary on both), 70 to 85 % are also right for the unadapted
model with the list and the EM prior, and 55 to 70 % with the list alone. At least 80 % of the
errors that self-training repairs named a species that occurs at no trap. Of the errors that
remain after self-training, under half do.

**Z14, the shipped setting and the seed** (added after Z9's one-view rows were read). The papers
quote tta3bm with the list, and Z9 is one view and one seed. More views sharpen the evidence, so
a prior should weigh less (Nemo: the EM adds 7.1 calibrated points over the list with one view,
3.4 with tta3bm). Predictions for the two pairs with tta3bm, calibrated probe macro-F1: with the
list on both, self-training's gain is 45 to 60 % of its full-vocabulary gain; the unadapted model
with the list and the EM (probe nights included) lands within 3 points of the self-trained model
with the full vocabulary; best rule against best, 4 to 8 points stay with self-training. Seeds 43
and 44 of the unadapted V2-S BU, one view, list and EM with the probe nights: within 2 points of
seed 42's 0.618.

**Z10b, the same question with a measure that is actually free of the prior** (added after Z10
was found to be measured wrongly, see the results). For two species k and j of the probe, the
choice between them on a crop rests on `z_k(x) - z_j(x)`. The AUROC of that difference, crops of
k against crops of j, is unchanged by a constant per species (any prior, any list) and by a
constant per crop (which the softmax ignores). Mean of 1 - AUROC over all pairs of the 243 species.
I have seen two contaminated versions of the number, so this is an informed guess: self-training
lowers it by 20 to 50 % on V2-S and by 0 to 30 % on DINOv3; on pairs of congeners by under 20 % on
both.

**Z15, a new site, on the papers' metric** (the owner asked whether a prior travels better than
self-training; Z8's answer was in accuracy only). For each site in turn, calibrated macro-F1 on
that site's probe crops, V2-S BU, one view: the model that never saw a trap with the list and an EM
prior from that site's own unlabelled crops; the fold self-trained on the two other sites, with the
list; the model self-trained on all three, with the list. Prediction: the first is within 3
points of the second at every site, and 3 to 8 points under the third.

If Z10 and Z11 both hold, the answer is: the traps have a large appearance shift, self-training
repairs a part of it that is measurable, and most of its accuracy gain is nonetheless the prior.

### Results (2026-11-15)

Outputs in `data/population_probes/`: `z9_pairs.csv`, `z9_shipped.csv`, `z10_discrimination.csv`
(and `_per_species.csv`), `z11_invariance.csv`, `z12_labels.csv`, `z13_overlap.csv`,
`z14_v2s_tta.csv`, `z14_dinov3_tta.csv`, `z14_v2s_seeds.csv`, `z15_newsite.csv`; the scorer's own
tables under `z9/` and `z15/`.

#### Z11: the appearance shift is real and large

Same model, same 243 species (those with at least 5 probe crops), full vocabulary, no prior, one
view. Under label shift alone the two recalls would be equal.

| model (seed 42) | recall, test fold | recall, probe crops | gap (95 % interval) | species lower on the probe |
|---|---|---|---|---|
| V2-S BU, never saw a trap | 0.947 | 0.662 | 28.6 (25.4 to 31.7) | 89 % |
| V2-S F | 0.948 | 0.640 | 30.8 | 89 % |
| B0 BU | 0.941 | 0.636 | 30.6 | 92 % |
| V2-S BU, real labels of the gated crops | 0.946 | 0.830 | 11.6 (8.8 to 14.5) | 63 % |
| V2-S BU, real labels at 11 copies (T1rep) | 0.948 | 0.863 | 8.5 (5.8 to 11.4) | 52 % |

- **Held** (above 0.90 and a gap above 25 points). A species that is recognised 95 times in 100 on
  photographs is recognised 66 times in 100 on trap crops by the same model. That cannot be a
  change of frequencies; the crops are harder to read. The destination of the errors moves too: 50 %
  name a congener on the test fold, 18 % on the probe.
- This number alone does not say what closes the gap: a model trained on trap labels raises probe
  recall to 0.83 to 0.86, and its implicit prior moved together with its features.

#### Z10: measured wrongly, replaced by Z10b

A mistake of mine, caught before it left this entry. Six of the logit sets (the campaign pairs and
Otto) are stored with each crop's maximum subtracted (`fp16s`), so for them the "raw logit" was the
gap to the top class, which a prior does move. With the maxima added back the number changed sign
on DINOv3 (ranking error +80 % after self-training instead of -39 %), because a raw logit compared
across crops also carries a per-crop offset that the softmax ignores. Neither version measures what
Z10 claimed. Its first table (-61 % and -39 %) is withdrawn; Z10b replaces it.

#### Z10b: self-training taught the small backbone to read trap crops, and the large one almost nothing

For two species of the probe, the AUROC of `z_k - z_j`, crops of k against crops of j. No prior, no
list and no per-crop offset can move it. Mean of 1 - AUROC over the 29,403 pairs of the 243 species
with at least 5 probe crops; one view.

| model | all pairs | pairs of congeners (122) | same family, other genus | other family |
|---|---|---|---|---|
| V2-S BU, never saw a trap | 0.00132 | 0.0260 | 0.00137 | 0.00118 |
| V2-S BU self-trained (st26) | 0.00063 | 0.0249 | 0.00076 | 0.00048 |
| DINOv3 BU, never saw a trap (seed 43) | 0.00045 | 0.0225 | 0.00054 | 0.00032 |
| DINOv3 BU self-trained (st26) | 0.00040 | 0.0266 | 0.00059 | 0.00022 |
| Nemo | 0.00046 | 0.0196 | 0.00065 | 0.00032 |
| P5 | 0.00030 | 0.0166 | 0.00032 | 0.00021 |
| Otto (top 35 per species) | 0.00044 | 0.0196 | 0.00064 | 0.00029 |
| OT2 (Otto + TrivialAugmentWide) | 0.00024 | 0.0177 | 0.00033 | 0.00013 |
| OT1 (gated labels) | 0.00017 | 0.0150 | 0.00028 | 0.00008 |
| OT3 (gated labels + TrivialAugmentWide) | 0.00014 | 0.0145 | 0.00021 | 0.00006 |

- **V2-S: -52 %** (paired over species, interval of the difference -0.00090 to -0.00050; 66 % of
  species improve). This is knowledge of trap images that no prior gives. It is all between
  unrelated species (-59 % across families); among congeners nothing (-4 % over pairs).
- **DINOv3: -13 %, interval -0.00014 to +0.00002.** Not distinguishable from no change, and worse
  among congeners (+18 % over pairs; per species +28 %, interval excluding zero). The unadapted model
  is seed 43 and the self-trained one seed 42. For this backbone, self-training's probe gain is
  close to all prior.
- **Scored:** V2-S fell 2 points outside my range (20 to 50 %); DINOv3 inside (0 to 30 %);
  congeners under 20 % held on both.
- **The Otto two-by-two is not only the prior.** Gated labels lower the pairwise error by 60 %
  against the top-35 labels (interval excluding zero), 22 % among congeners; TrivialAugmentWide by
  45 % and 13 %. So Z12's "two thirds is recovered by a prior" is about accuracy; in what the
  model can tell apart, the better labels and the augmentation both bought something real.
- **What the measure cannot see:** an absent species has no crops, so no pair. An answer from the
  wrong fauna is, by this construction, an error that some prior can always remove.
- **The prior self-training learns is tied to the domain.** After self-training the model names a
  species that occurs at no trap on 8.0 % of probe crops (V2-S; DINOv3 7.4 %), against 29.6 % and
  25.1 % before. Its test-fold macro-F1 over all species does not move (V2-S: 0.8870 against
  0.8866 to 0.8877 for three unadapted seeds; DINOv3: 0.9181 against 0.9174 to 0.9179). So it did not learn "these species are rare"; it
  learned "on an image that looks like a trap crop, answer from the trap fauna". A list and an
  estimated prior at prediction time are the same thing, kept outside the weights.

#### Z13: the two repair the same crops

Probe crops (15,200), one view, full vocabulary unless a rule is named.

| | V2-S BU | DINOv3 BU |
|---|---|---|
| crops the unadapted model gets wrong | 5,209 | 4,576 |
| of those, right after self-training | 2,780 | 2,565 |
| of those, right for the unadapted model with the list | 2,018 | 1,487 |
| of those, right for the unadapted model with list and EM | 2,855 | 2,456 |
| crops self-training repairs that list and EM also repair | **78 %** | **76 %** |
| crops self-training repairs that the list alone also repairs | 56 % | 47 % |
| crops list and EM repair that self-training also repairs | 76 % | 79 % |
| errors self-training repairs: named a species at no trap | 92 % | 89 % |
| errors self-training leaves: name a species at no trap | 45 % | 48 % |
| errors that name a congener: unadapted / left by self-training | 18 % / 23 % | 20 % / 26 % |
| wrong under both best rules (unadapted with list and EM; self-trained with list) | 1,682 | 1,451 |

- **Held:** 70 to 85 % for list and EM (78 and 76 %), at least 80 % absent species among the repaired
  errors (92 and 89 %), under half among those left (45 and 48 %, close to the bound). The list
  alone is at 56 % (held) and 47 % (under my range) for DINOv3.
- So Z10 and Z8 do not contradict each other. The dominant trap error is an answer from the wrong
  fauna (86 and 83 % of the unadapted model's errors name a species that occurs at no trap). It has
  two cures. A prior removes the candidates. Training on trap crops stops the model preferring
  them. Three quarters of the crops are cured by either.
- What neither cures is the same set too: 1,682 and 1,451 crops are wrong under both best rules,
  62 to 77 % of what each model still gets wrong (V2-S: 2,507 and 2,210 crops).

#### Z9: on the papers' metric, a prior with no training equals self-training alone; the two together are best

Harness: OT3 with the list reproduces `ot_scores.md` to four digits (probe 0.7423 / 0.6753
calibrated / zero, probe-HO 0.4941 / 0.3910); Nemo, P5 and Otto reproduce their journalled rows.
Species macro-F1 calibrated by (trap, night), probe / probe-HO, one view. "With probe nights" is
the EM run on every unlabelled crop, the scored nights included (it uses no label). The last
column gives each model the probe's true species frequencies; it is a ceiling for any label
prior, not a method.

| model | nothing | list | list and EM | list and EM, with probe nights | probe's own frequencies |
|---|---|---|---|---|---|
| V2-S BU, never saw a trap | 0.372 / 0.303 | 0.553 / 0.458 | 0.585 / 0.461 | 0.618 / 0.479 | 0.727 / 0.569 |
| V2-S BU self-trained (st26) | 0.592 / 0.365 | **0.669** / 0.379 | 0.574 / 0.343 | 0.606 / 0.361 | 0.685 / 0.434 |
| DINOv3 BU, never saw a trap | 0.464 / 0.390 | 0.619 / 0.478 | 0.626 / 0.482 | 0.649 / 0.513 | 0.742 / 0.571 |
| DINOv3 BU self-trained (st26) | 0.627 / 0.430 | **0.691** / 0.463 | 0.634 / 0.406 | 0.673 / 0.431 | 0.736 / 0.494 |

- **Z9a held.** Self-training is worth +22.1 (V2-S) and +16.3 (DINOv3) with the full vocabulary,
  and +11.6 and +7.1 once both models use the list: 53 % and 44 %.
- **The whole gain, two ways.** From the unadapted model with nothing to the self-trained model
  with the list is +29.8 (V2-S) and +22.7 (DINOv3). With no training, the list gives 61 and 69 % of
  it, the list and the EM 72 and 72 %, with the probe nights 83 and 82 %. Self-training with no
  prior gives 74 and 72 %. So a regional list and an estimated prior, on a model that never saw a
  trap, score level with self-training alone (0.585 to 0.618 against 0.592; 0.626 to 0.649 against
  0.627), and the two overlap for the most part.
- **Z9b: wrong on the unadapted model, and in my favour on the gap.** The EM adds +3.2 (V2-S) and
  +0.7 (DINOv3) over the list, +6.5 and +3.0 with the probe nights; I predicted within 2. On the
  self-trained models it costs 9.5 and 5.7 (predicted 2 to 6). Best label-free rule against best,
  self-training keeps 5.2 to 8.5 points (V2-S) and 4.2 to 6.5 (DINOv3); I predicted at least 8.
  The calibration spread on probe is 0.5 to 1 point; the seed range of the unadapted V2-S BU is 3
  points (0.349 to 0.378), and each row is one seed.
- **Z9c held for the unadapted models only.** The EM gains less on probe-HO than on probe (+0.3
  against +3.2 on V2-S). On the self-trained models it costs as much or less there.
- **Z9d held on the letter and says less than I hoped.** With the probe's own frequencies the
  self-trained models are not ahead: -0.8 and +0.5 in accuracy, -4.2 and -0.6 in calibrated
  macro-F1. But a self-trained model already carries a trap prior, so this diagnostic counts it
  twice for that model; the EM hurting it (above) is the same double counting. It cannot rank the
  two. Z10 and Z13 are the tests that can.
- **Held-out species, mixed.** With the list on both, the self-trained V2-S is 7.9 calibrated points
  under the unadapted one on probe-HO (0.379 against 0.458, calibration spread 1.6 and 2.7) and 3.7
  above it at zero threshold (0.224 against 0.187). DINOv3: -1.5 and +3.6. Not a finding yet.

The models the papers quote, calibrated probe / probe-HO:

| model, views | nothing | list | list and EM | list and EM, with probe nights |
|---|---|---|---|---|
| Nemo, tta3bm | 0.489 / 0.371 | 0.660 / 0.519 | 0.694 / 0.492 | 0.713 / 0.523 |
| P5, tta3bm | 0.661 / 0.519 | 0.711 / 0.546 | 0.732 / 0.556 | **0.741** / 0.557 |
| OT3, tta3bm | 0.739 / 0.564 | 0.749 / 0.562 | 0.713 / 0.528 | **0.752** / 0.543 |
| Nemo, one view | 0.338 / 0.293 | 0.546 / 0.448 | 0.618 / 0.475 | 0.631 / 0.485 |
| P5, one view | 0.622 / 0.488 | 0.679 / 0.500 | 0.687 / 0.526 | 0.703 / 0.511 |
| OT3, one view | 0.729 / 0.479 | **0.742** / 0.494 | 0.689 / 0.473 | 0.733 / 0.481 |
| Otto, one view | 0.588 / 0.450 | 0.661 / 0.500 | 0.654 / 0.472 | **0.679** / 0.484 |

- The shipped model is untouched: on OT3 the EM from other nights costs 3.6 points and the EM with
  the probe nights gives the list's score back.
- **P5 with the list and the EM scores 0.741, against OT3's 0.749 with the list.** P5's labels were
  balanced over species (as Otto's), so the prior was still missing; adding it closes most of what
  OT3 gained over P5 on probe (3.8 points to 0.8). Not on probe-HO (0.557 against 0.562 is level).
- Nemo, which never saw a trap, moves from 9 points under OT3 (list on both) to 3.6 with the EM.

#### Z14: in the shipped setting the two are within 1.5 points, and the prior is steadier across seeds

Calibrated probe / probe-HO with tta3bm (the two rotated views dumped today for the four models).

| model | nothing | list | list and EM | list and EM, with probe nights |
|---|---|---|---|---|
| V2-S BU, never saw a trap (seed 42) | 0.499 / 0.409 | 0.624 / 0.475 | 0.667 / 0.533 | **0.691** / 0.544 |
| V2-S BU self-trained (st26) | 0.632 / 0.450 | 0.682 / 0.479 | 0.658 / 0.484 | **0.706** / 0.515 |
| DINOv3 BU, never saw a trap (seed 43) | 0.591 / 0.481 | 0.694 / 0.540 | 0.683 / 0.568 | **0.718** / 0.587 |
| DINOv3 BU self-trained (st26) | 0.662 / 0.480 | 0.716 / 0.512 | 0.688 / 0.512 | **0.724** / 0.530 |

Three seeds of the unadapted V2-S BU (42 / 43 / 44), calibrated probe:

| views | nothing | list | list and EM | list and EM, with probe nights |
|---|---|---|---|---|
| one | 0.372 / 0.378 / 0.349 | 0.553 / 0.557 / 0.544 | 0.585 / 0.593 / 0.594 | 0.618 / 0.631 / 0.621 |
| tta3bm | 0.499 / 0.536 / 0.487 | 0.624 / 0.627 / 0.633 | 0.667 / 0.657 / 0.654 | 0.691 / 0.694 / 0.689 |

- **All three predictions falsified, each in the prior's favour.** With the list and tta3bm on both
  models, self-training is worth +5.8 (V2-S) and +2.2 (DINOv3), against +13.3 and +7.2 with the full
  vocabulary: 44 % and 30 % (predicted 45 to 60 %). The unadapted model with the list and the EM is
  5.9 and 5.6 above its self-trained twin at full vocabulary (predicted within 3), and above its
  twin with the list (0.691 against 0.682; 0.718 against 0.716). Best rule against best, 1.5 and 0.6
  points stay with self-training (predicted 4 to 8).
- On the held-out species the unadapted model with the prior is ahead: 0.544 against 0.515 and
  0.587 against 0.530 (calibration spread on this set 2 to 5 points).
- **Seeds held, and say more than asked.** With nothing the three seeds span 2.9 points (one view)
  and 4.9 (tta3bm); with the list and the EM they span 1.3 and 0.5. The seed noise of a trap score
  at full vocabulary is mostly answers from the wrong fauna. The self-trained models are still one
  seed each.
- With three views the EM that includes the scored nights helps the self-trained models too
  (+2.4 and +0.8 over the list); from other nights only it still costs.
- Everything scored today with tta3bm, best label-free rule, probe / probe-HO: OT3 0.752 / 0.543
  (list alone 0.749 / 0.562), P5 0.741 / 0.557, DINOv3 st26 0.724 / 0.530, DINOv3 unadapted 0.718 /
  0.587, Nemo 0.713 / 0.523, V2-S st26 0.706 / 0.515, V2-S unadapted 0.691 / 0.544.

#### Z12: a prior still helps when the pseudo-labels were evened out per species, and not otherwise

BioCLIP-2, seed 42, one view, probe accuracy / macro recall:

| labels | augmentation | nothing | list | list and EM | EM over the list |
|---|---|---|---|---|---|
| top 35 per species (Otto) | none | 0.773 / 0.772 | 0.827 / 0.796 | 0.862 / 0.769 | **+3.5** |
| top 35 per species (OT2) | TrivialAugmentWide | 0.775 / 0.788 | 0.827 / 0.811 | 0.864 / 0.777 | **+3.6** |
| most confident 70 %, natural frequencies (OT1) | none | 0.892 / 0.797 | 0.901 / 0.813 | 0.898 / 0.764 | -0.3 |
| most confident 70 %, natural frequencies (OT3) | TrivialAugmentWide | 0.903 / 0.808 | 0.910 / 0.815 | 0.901 / 0.759 | -0.9 |

- **Held** (+2 to +4 on Otto and OT2; within a point of zero on OT1 and OT3). One factor, twice.
  Pseudo-labels at their natural frequencies teach the site's frequencies; the same crops evened
  out per species do not, and the prior can then be added afterwards.
- It re-reads OT1 against Otto. The gated labels were worth 11.9 points of accuracy with the full
  vocabulary and 3.9 once each model has its best label-free rule (0.901 against 0.862). The gain
  was credited to label accuracy (98.6 % against 71 %); two thirds of it is recovered by a prior with
  no training, so it was the fauna and the frequencies the labels carried.

#### Z15: at a new site, the site's own prior equals self-training done elsewhere

V2-S BU, one view, calibrated macro-F1 on each site's probe crops.

| | LV | OH | SS |
|---|---|---|---|
| never saw a trap, list | 0.553 | 0.566 | 0.570 |
| never saw a trap, list and EM from the site's other nights | 0.593 | 0.571 | 0.606 |
| never saw a trap, list and EM from all of the site's crops | 0.629 | 0.615 | 0.648 |
| self-trained on the two other sites, list | 0.639 | 0.605 | 0.632 |
| self-trained on all three sites, list | 0.668 | 0.653 | 0.653 |

- **Held with the prior from all of the site's crops** (-1.0, +1.0, +1.6 against the fold; 3.9, 3.9
  and 0.5 under the model self-trained on the site). **Not with the prior from other nights** (2.6
  to 4.6 under the fold, 4.7 to 8.2 under the all-site model). The estimate needs the nights it is
  applied to, which costs nothing: it uses no label.
- So a prior travels as well as self-training does. A model that never saw any trap, given the
  new site's unlabelled crops and no training, scores what a model self-trained at two other
  Danish sites scores. Self-training on the site itself adds 0.5 to 3.9 points more. One seed, one
  view, three sites of one country.

### What section 8 says

**To the owner's question.** It is not "label shift instead of domain shift". The traps have both
(Z11). What the measurements say about self-training is narrower and, I think, more useful:
1. Most trap errors are answers from the wrong fauna (83 to 86 %). They exist because the crops
   are hard to read, and they can be removed without reading the crops better.
2. Self-training removes them by learning, inside the weights, a prior that it applies to images
   that look like trap crops (Z13: absent species named on 8 % of crops instead of 25 to 30 %, test
   fold unchanged). A regional list and an estimated prior do the same outside the weights, and
   three quarters of the repaired crops are the same crops.
3. On top of that, the small backbone learned to tell trap species apart (-52 % on a measure no
   prior can move), the large one did not (-13 %, not significant), and neither improved on
   congeners (Z10b).
4. On the papers' metric, in the shipped setting, the difference between a model that never saw a
   trap and its self-trained twin is 1.5 (V2-S) and 0.6 (DINOv3) calibrated points once both have a
   list, three views and an estimated prior; on held-out species the unadapted model is ahead.

**What I cannot affirm.** Anything like-for-like about Otto: no BioCLIP-2 BU was trained without
pseudo-labels, and OT3 stays the best model here, 3 to 6 points above every model that never saw a
trap (0.749 against 0.718 for the unadapted DINOv3, 0.713 for Nemo, 0.691 for the unadapted V2-S). The self-trained models are one
seed. It is one country. And OT3's better labels did buy real discrimination (Z10b).

**What would confirm it,** in order of value per run: (a) Otto's recipe with no pseudo-labels, one
run, the missing control for the production model; (b) two more seeds of `v2s_bu_st26`; (c) a
labelled trap set from another country, for the list and the EM as much as for self-training.

**For paper B.** The headline for self-training should be quoted against the baseline a pipeline
ships: +5.8 and +2.2 calibrated points (list, tta3bm), or +11.6 and +7.1 with one view, not +13 to
+22. The stronger story is the decomposition: how much of the photograph-to-trap gap is the fauna
(a list and an unsupervised prior: 72 to 83 % of the whole gain with one view, no training), how
much is reading the crops (what self-training adds beyond), and what stays (species that are
present, congeners first; 1,451 to 1,682 crops wrong under both best rules). It also gives a practical
order: list always, estimated prior where no adapted model exists, self-training where it can be
afforded, with gated labels at natural frequencies if no prior will be applied afterwards.

### Z16, the control for Otto (opened 2026-11-16, before OT6 has trained)

The owner approved the first run of "what would confirm it": OT6, OT3's recipe with the
pseudo-labels removed and nothing else (mini_trainer session, UCloud 12463508, about 24 h). The run,
its plan row and its headline predictions live in the Otto entry
([[2026-11-07-otto-a-production-model-for-ami-traps]], 2026-11-16): OT6 under OT3 by 2 to 6
calibrated points with list and tta3bm, within 2 with an estimated prior, level or above on
probe-HO. I would have said within 3; both get scored.

What this entry adds is the part that needs OT6's per-view logits (`calibrated`, `overlap`,
`discrimination` of `dev/107`, as for OT1), with predictions those lines do not cover:
- **The headline shrinks as on the other pairs.** One view, full vocabulary, calibrated probe: OT6
  at 0.50 to 0.58 against OT3's 0.729, a lead of 15 to 23 points that falls to the 2 to 6 above
  once both have the list and three views.
- **The same crops.** Of the probe crops OT6 gets wrong and OT3 gets right (one view, full
  vocabulary), 65 to 80 % are right for OT6 with the list and the EM prior (V2-S 78 %, DINOv3 76 %).
- **Discrimination, the prediction I am least sure of.** Z10b's pairwise error for OT6 between
  0.00018 and 0.00035, with OT3 (0.00014) 20 to 60 % lower. DINOv3 says a large backbone gains
  little there; OT1 against Otto says accurate labels gain a lot, but that was against bad labels,
  not against none.
- **The test fold does not move** (within 0.3 points of OT3's), as for st26.

What would make me narrow finding 4b: OT6 with list, estimated prior and tta3bm more than 5 points
under OT3 with the list on probe. Then self-training with gated labels on this backbone buys what
no prior gives, and 4b holds for the campaign's st26 recipe only. Both models are one seed; three
seeds of an unadapted V2-S span 0.5 points under that rule, so a gap above 2 would be read as real.

## 9. The first Tree of Life model: what to compare it with (opened 2026-11-17, owner's question)

tol-full is trained (W3: DINOv3 ConvNeXt-L, hierarchical head over seven ranks, 190,199 species, 224
px, five passes). TL3 scored its Lepidoptera head without fine-tuning against VD, the same trunk
trained on Lepidoptera alone: test fold 0.875 against 0.912, probe 0.590 against 0.689 (list,
tta3bm, calibrated). The owner doubts that this is the right comparison and asks which is, how
good the embedding is, and whether more passes are the next step. The run, TL3 and their record are
the `lepinet` session's ([[2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know]]);
this section holds the checks I ran and my reading. TL1 (the frozen model against BioCLIP-2 and raw
DINOv3) was planned there and had not run; I ran it at that session's request. (The W3 steps E1 to E6
were renamed TL1 to TL6 on 2026-11-17, since E1 and E2 are taken in RESULTS.)

### Predictions (written before any of these ran)

**Z17, where the test-fold gap of 3.7 points sits.** Macro-F1 is a mean over species, so it splits
exactly over groups of species. (a) The 162 vocabulary species ToL never trained on score zero for
ToL and about 0.85 for VD: 1.0 to 1.3 of the 3.7 points. (b) The keys that share a ToL class: the
losing key of each class scores zero in the protocol line: 0.5 to 1.0 point. (c) On species both
models can name one to one the gap is 1.5 to 2.2 points, and it grows as ToL's training images for
the species fall below lepinet's; where ToL has at least as many, under 1.5.

**Z18, the trap gap under my rules** (`dev/107`, ToL's per-view logits against VD's). Read from
TL3's sheet before any run: on the calibrated score the gap is 9.9 points with one view and 9.9 with
tta3bm, so TTA does not favour VD on the papers' metric (the sheet's -2.9 is at zero threshold).
(a) With the list and the prior estimated with the probe nights the gap narrows to 5 to 8 points,
since a weaker model gains more from the prior (Nemo did). (b) Prior-free pairwise error (Z10b):
ToL at 1.5 to 3 times VD's, so most of the gap is in telling trap species apart, not in the fauna.
(c) 80 to 90 % of ToL's errors at its full Lepidoptera vocabulary name a species seen at no trap.

**Z19, TL1 on the full model** (`dev/109`; BioCLIP-2's SimpleShot, five folds). Rare Species 1- and
5-shot: ToL 50 to 56 and 68 to 73 (pilot 42.8 and 60.2; BioCLIP-2 60.4 and 77.0); raw DINOv3
ConvNeXt-L 25 to 38 and 45 to 58. The 1,810 held-out species: ToL 54 to 62 and 75 to 82 (pilot
46.1 and 68.7); BioCLIP-2, which trained on these species, 3 to 10 points above ToL at 5 shots,
most on whole held-out families; raw DINOv3 more than 20 below ToL. Placement of held-out taxa by
the coarse marginals: genus 62 to 70 %, family (whole genus held out) 58 to 66, order (whole family)
50 to 60 (pilot 46.9, 47.3, 41.1). Nearest prototype a congener 93 to 96 % (pilot 90.2). Kingdom
right on at least 98.5 % of held-out images (the last epochs' drift in val kingdom loss is noise).

**Z20, the three encoders frozen, on Lepidoptera and on trap crops** (the comparison I would put
beside TL3; run only if the shared GPU allows). Nearest class mean from the same training images per
species, 224 px, each model's own preprocessing; scored on the part of the test fold outside
ToL-200M and on the probe with the regional list. Test fold: ToL above BioCLIP-2 by 2 to 8 points
of macro-F1 (BioCLIP-2 with a fitted classifier was 0.844, P1a), raw DINOv3 far below both (under
0.55). Probe: BioCLIP-2 level with ToL or above it (0 to 8 points of accuracy), the three-axes
pattern of finding 2. Least sure of the probe.

### Results (2026-11-17)

**Z17: of the 3.7 test-fold points, 1.6 are the label space and 2.1 are vision** (`dev/107 tolgap`,
TL3's line-1 prediction files, per-species F1 at zero threshold; macro-F1 recomputed 0.8751 and
0.9124 against the sheet's 0.8751 and 0.912; `data/population_probes/z17_tolgap.csv`).

| species group | species | test images | F1 ToL | F1 VD | share of the 3.73-point gap |
|---|---|---|---|---|---|
| no ToL class (138 absent from tol256, 23 below its floor, 1 test fold only) | 162 | 3,624 | 0 | 0.831 | -1.11 |
| share a ToL class, the key ToL never answers | 124 | 4,276 | 0 | 0.523 | -0.53 |
| share a ToL class, the key ToL answers | 128 | 13,552 | 0.772 | 0.850 | -0.08 |
| one to one | 11,627 | 608,290 | 0.906 | 0.927 | -2.01 |

On the one-to-one species the gap barely depends on how many images ToL had: -1.3 points where
ToL had more than twice lepinet's count, -2.0 at 1.1 to 2 times, -2.1 at parity, -3.5 at half (33
species with under half: -16.5). By lepinet's count it is -2.9 under 100 images and -1.6 to -1.7
above 200. So the in-distribution gap is not data per species; resolution (224 against 320 px),
a head shared with 178 k other species and five passes over everything are the candidates, and TL4
removes all three. The 162 species are a hole in the crawl (several with over 1,000 lepinet
images, *Aedia acronyctoides*, *Lithosia quadra*, *Plusia festucae*), not a model property.
Predictions: (a) 1.0 to 1.3 held (1.11); (b) 0.5 to 1.0 held (0.61); (c) 1.5 to 2.2 held (2.1),
"under 1.5 where ToL has at least as many images" falsified (2.0 at 1.1 to 2 times).

**Z18: on the traps ToL tells the species apart as well as VD; it is less sure of itself.**
`dev/107 calibrated` on ToL's per-view logits (12,268 classes, fp32, UCloud `otto_eval/tol_full`)
and VD's (TL3's local copy), same crops, same scorer (`z18_tol_trap.csv`):

| probe, calibrated macro-F1 (ToL / VD / gap) | one view | tta3bm |
|---|---|---|
| nothing | 0.310 / 0.456 / -14.6 | 0.466 / 0.591 / -12.5 |
| list | 0.518 / 0.617 / -9.9 | 0.590 / 0.689 / -9.8 |
| list and EM | 0.534 / 0.633 / -9.8 | 0.601 / 0.682 / -8.1 |
| list and EM, probe nights included | 0.561 / 0.662 / -10.1 | 0.642 / 0.714 / -7.2 |
| accuracy, same four rules (gap) | -12.8, -4.8, -3.6, -3.4 | -8.0, -4.2, -1.9, -1.7 |

By accuracy the prior closes most of the gap (ToL 87.0 % against 88.7 % with list, EM and tta3bm);
by calibrated macro-F1 it does not. The reason is in two measurements:
- *Discrimination is level* (Z10b on the 234 probe species with five or more crops that both
  models can name, `--common`): pairwise error ToL 0.00060 against VD 0.00056 with one view,
  0.00018 against 0.00014 with tta3bm; on congener pairs ToL is better, 0.0229 against 0.0243 and
  0.0150 against 0.0174. Prediction (b), ToL at 1.5 to 3 times VD's error, falsified. At 224 px
  the generalist separates trap species as well as the specialist at 320 px.
- *Confidence is worse.* AUROC of the top-1 confidence for right against wrong answers on the probe:
  ToL 0.867 against 0.905 (list, one view), 0.856 against 0.901 (list, tta3bm); share of crops
  answerable at 95 % precision 52 against 69 % and 66 against 80 %. ToL is less confident overall
  (median 0.89 against 0.95). The calibrated score abstains by confidence per (trap, night), so it
  charges this to ToL; and the estimated prior, which multiplies the posteriors, pushes a model with
  flatter posteriors harder toward the common species, so ToL's errors land on the rarer trap species
  and cost macro-F1 far more than accuracy.

The 12 probe species ToL has no class for cost 0.6 to 1.2 points of the macro-F1 gap; the four
that share a class, nothing. Errors naming a species seen at no trap (full vocabulary): ToL 91.6 %
(one view) and 81.6 % (tta3bm), VD 83.9 and 75.0; prediction (c) held with tta3bm, 1.6 above its
range with one view. Prediction (a), the gap narrowing to 5 to 8 with list and estimated prior:
held with tta3bm (7.2), falsified with one view (10.1).

**Z19 (TL1): frozen, the ToL model is above BioCLIP-2 on its own benchmark, and on taxa only
BioCLIP-2 has seen.** `dev/109` on tol_full's `last.pt` (head input, 1,536-d) and on the two frozen
baselines, same images and protocol (outputs in `data/population_probes/tol_e1/`). `embed_frozen`
crashes under Python 3.14 (forkserver workers cannot pickle its local Dataset class); run through a
wrapper that sets the fork start method, `dev/109` left as it is. 1-shot / 5-shot accuracy:

| | Rare Species (400 species) | held-out taxa (1,810 species) | near / mid / far, 5-shot |
|---|---|---|---|
| ToL (5 passes) | **62.9 / 79.1** | **68.9 / 86.3** | 91.5 / 82.5 / 61.0 |
| ToL pilot (1 pass of 8 M, at most 50 per species) | 42.8 / 60.2 | 46.1 / 68.7 | 74.0 / 64.0 / 44.6 |
| BioCLIP-2, frozen | 60.4 / 77.0 | 66.2 / 85.0 | 89.9 / 81.5 / 61.5 |
| DINOv3 ConvNeXt-L, raw (ToL's start) | 44.2 / 54.0 | 26.3 / 39.5 | 43.1 / 36.9 / 21.9 |

BioCLIP-2 reproduces our port to the digit. Paired bootstrap over species, ToL minus BioCLIP-2,
1-shot / 5-shot: Rare Species +2.6 [1.6, 3.5] / +2.1 [1.4, 2.8]. On the 356 Rare Species that are ToL
classes +3.2 / +2.8; on the 44 that are not, -3.0 [-6.2, +0.2] / -3.4 [-5.8, -0.8]. Held-out taxa
(which BioCLIP-2 trained on and ToL did not): near +3.9 / +1.6, mid +0.5 / +1.0, far -0.2 / -0.1
(far: confidence intervals include zero). **The lead falls with distance from the ToL model's own
classes**: it is largest on species it trained on, smaller on unseen species of trained genera, and
gone at unseen families. That gradient is what a classification objective predicts (route 5: it
keeps what separates its labels). BioCLIP-2 is flatter across distance, and it has seen every one of
these taxa.

Placement of the held-out taxa by the coarse marginals (macro over species): genus 67.1 % (near),
family 59.9 % (mid), order 45.6 % (far). Kingdom right on 98.8, 97.6 and 95.7 % of near, mid and far
images (pilot 98.2, 97.9, 97.1). The errors are what is in the picture: gall wasps, gall mites and
parasitoids photographed on their plants, cochineal on cactus (Dactylopiidae, 95 images) and a
puffball-like fungus family (Calostomataceae, 127) go to Plantae. Whole held-out fungus families are
right on only 58 % of images. The drift in val kingdom loss that mt saw is not a broken rank. In the
run's summary the val loss rises over the last two or three half-pass epochs at kingdom
(0.0295 to 0.0316), phylum (0.042 to 0.044) and class (0.064 to 0.066), with accuracy flat: the
coarse ranks become slightly overconfident as the rate anneals. Species val loss still falls (0.94 to
0.92), but the train species loss has been under val since epoch 7 (0.77 against 0.92 at the end).
The prototypes became less taxonomic as the embedding improved: nearest prototype a congener
78.2 % (centred 86.7), against 90.2 (92.2) at the pilot.

Predictions scored. Rare Species 50 to 56 and 68 to 73, and held-out 54 to 62 and 75 to 82: both
falsified, the model is better than I said. BioCLIP-2 3 to 10 points above ToL at 5 shots on the
held-out taxa: falsified (ToL +1.3); "most on whole families": the only depth where BioCLIP-2 is not
behind. Raw DINOv3 25 to 38 and 45 to 58 on Rare Species: 1-shot above (44.2), 5-shot in (54.0);
"more than 20 below ToL on held-out": held (47). Placement: genus 62 to 70 held, family 58 to 66
held, order 50 to 60 falsified (45.6). Congeners 93 to 96 falsified (78.2). Kingdom at least
98.5 %: falsified (98.2 % of all held-out images). The owner's view, the ToL model level with BioCLIP-2
on species tasks unless training length decides it: held, and better than level on the taxa near
its classes.

**Z20: frozen, on moths, ToL is the better encoder too, and by a wide margin on the traps once the
list is applied** (`dev/107 frozen_index` and `frozen`; `z20_frozen.csv`). Support: 10 training
images of each of lepinet's 12,041 vocabulary species. Queries: up to 8 test-fold images per species
whose GBIF occurrence is outside ToL-200M (68,764 images of 11,537 species, of 217,856 such images;
BioCLIP-2 never saw them), and the probe crops. SimpleShot's nearest class mean, each encoder at its
own preprocessing (224 px for all three). Beside them, the trained heads on the same images.

| readout | clean test fold, acc / macro-F1 | probe acc, full vocabulary / list | probe-HO acc, list |
|---|---|---|---|
| ToL, frozen, class means | 84.6 / 0.824 | 47.2 / **65.4** | 49.5 |
| BioCLIP-2, frozen, class means | 82.8 / 0.798 | 46.4 / 57.0 | 45.8 |
| DINOv3 raw, frozen, class means | 5.2 / 0.042 | 0.2 / 1.5 | 0.5 |
| ToL's own head (TL3) | 89.3 / 0.860 | 57.5 / 75.4 | 68.5 |
| VD's head (Lepidoptera only, 320 px) | 92.5 / 0.906 | 70.3 / 80.2 | 73.3 |
| OT3's head | 92.9 / 0.910 | | |

ToL minus BioCLIP-2, paired bootstrap: test fold accuracy macro over species +1.9 [1.6, 2.3]; probe
with the list +8.5 [7.0, 10.0] over 588 (trap, night) groups. The harness reproduces TL3's
sheet for ToL's head (probe 0.5747 and 0.7543, zero-threshold macro-F1 0.1421). Raw DINOv3 is not
a bug: its features fail the same checks the other two pass (support images classified by class
means that include them: 21 % against 94 and 94 %; 100 random species: 37 % against 99 and 99 %).
Its pooled feature does not separate moth species at all, so everything in ToL's column was learnt
in the five passes. Predictions: ToL above BioCLIP-2 by 2 to 8 macro-F1 on the test fold, held
(+2.6); raw DINOv3 under 0.55, held; BioCLIP-2 level with ToL or above on the probe, held at full
vocabulary (+0.8 for ToL), falsified with the list (ToL +8.5).

One check of my own reading of Z18 failed. I suspected that restricting ToL's 190 k-species head to
its Lepidoptera classes throws away an "is this a moth at all" signal. It does not: the full head
calls 99.2 % of probe crops Lepidoptera (the rest are 4 % right), its order confidence ranks right
against wrong far worse than the restricted species confidence (AUROC 0.73 against 0.87), and the
product does not help (0.86). The confidence deficit is in the species posterior itself.

### What section 9 says

**On the comparison.** TL3 compares answers, not representations. It is the right comparison for
"can the ToL model replace the Lepidoptera model as it is", and the answer is no (test fold -3.7,
calibrated probe -7 to -10). As a measure of the backbone it is unfair three ways. 1.6 of its 3.7
points are the label space: 162 species missing from the crawl, 124 keys merged by COL XR. The
specialist reads 320 px, ToL 224. And the specialist spent its whole head on 12 k moths. Three
comparisons answer three different questions, and each has its own fair setting:
1. *Is it a good general biodiversity embedding?* Frozen, against BioCLIP-2 and its own starting
   trunk, on BioCLIP-2's benchmarks (Z19). Answer: yes. It is above BioCLIP-2 on Rare Species and
   on 1,810 taxa that only BioCLIP-2 has seen, and 25 to 47 points above raw DINOv3. BioCLIP-2's
   other suite tasks are still to build.
2. *Is it a good moth embedding?* Frozen, same support images, on the part of our test fold
   outside ToL-200M (the owner's "subset within global_lepi", required for any BioCLIP-2 comparison
   since BioCLIP-2 trained on 65 % of our test fold), and on the traps (Z20). Answer: yes. It is
   ahead of BioCLIP-2 on photos (+1.9) and far ahead on trap crops with the list (+8.5).
3. *Is it a better start for our task?* Fine-tuned with the same recipe at the same resolution
   against the same trunk from DINOv3 (TL4 against VD) and against BioCLIP-2's start (P3c). This
   is the like-for-like that paper B needs, and it is running.

**On the traps, the gap is confidence, not sight.** ToL at 224 px tells trap species apart as well as
VD at 320 px (prior-free pairwise error level, congeners better), and with list, estimated prior and
tta3bm it is 1.7 points of accuracy behind. What it lacks is a confidence that ranks its own errors
(AUROC 0.86 against 0.90), which the calibrated macro-F1 and the estimated prior both amplify. Two
candidate causes, both testable: the resolution (blurrier inputs give flatter posteriors), which the
288/320 px check tests; and a head whose softmax is spread over 190 k classes, which fine-tuning
(TL4) replaces. If TL4 recovers VD's confidence AUROC, the generalist trunk is not the problem.

**On generality.** The lead over BioCLIP-2 is largest on taxa near ToL's own classes and gone at
whole held-out families; on the 44 Rare Species that are not ToL classes BioCLIP-2 is ahead (-3.4 at
5 shots). The prototypes became less taxonomic as training went on (congeners 90 to 78 %), and
far-family kingdom placement slipped (97.1 to 95.7 %). A classification objective keeps what
separates its labels, as route 5 says. BioCLIP-2's text objective degrades less with distance.
ME's margin arm should be read against this.

**On more passes.** I would not buy more ToL passes now. (i) Nothing in these results needs them:
on species tasks the model is already above BioCLIP-2, so training length is not what it lacks
there. (ii) The moth gap is not lack of moth training: the in-distribution gap is the same
whatever ToL's image count per species (Z17), and on the traps discrimination is level (Z18).
More ToL passes address neither resolution, nor the label space, nor confidence. (iii) The run's own
signals are mixed. Val species F1 was still rising, by about 3 points per half-pass before the anneal,
so the 190 k-way task is under-trained. But train loss has been under val since epoch 7, coarse-rank
val losses rose in the last epochs, and the prototypes and far placement got less taxonomic, so
more passes may trade generality at distance for in-distribution accuracy. (iv) Five more passes cost about 110 B200-hours. If length is to be tested
later, judge it on Rare Species, held-out few-shot by depth and Z20 rather than on val loss. A
cheap bound first: TL1 on `checkpoint_5.pth` (two and a half passes; confounded by the rate, which
had not annealed).

**For the next ToL version, not now:** fill the 162 vocabulary species and 12 trap species missing
from tol256 from global_lepi (the original design's "global_lepi fill"); kingdom errors are galls,
hosts and fungi, the "two organisms in one picture" problem (S1a), not a broken rank.

## Log

- 2026-11-17: section 9 opened (the first ToL model); predictions for Z17 to Z20 written before the
  checks ran, then run the same morning at the owner's request and the `lepinet` session's (TL1 on
  the workstation GPU, 4 loader workers; tol_full's `last.pt`, its training summary and its per-view
  trap logits downloaded from UCloud, no job submitted). Scoreboard: held Z17a, Z17b, Z17c's gap,
  Z18c (tta3bm), Z18a (tta3bm), Z19's raw-DINOv3 gap and placement at genus and family, Z20 on the
  test fold and raw DINOv3; falsified Z17c's image-count dependence, Z18a (one view), Z18b (ToL
  discriminates as well as VD), Z19's ToL ranges (too low), BioCLIP-2 ahead on held-out taxa,
  placement at order, congeners and kingdom, Z20 on the probe with the list. Not committed.
- 2026-11-16: section 8 committed and pushed the evening before as abd3ccf and reported to the
  `lepinet` session. The owner approved the control for Otto (OT6, submitted by the mini_trainer
  session); Z16's predictions written before it trained. Not committed.
- 2026-11-15, evening: second batch committed and pushed as b63eeb6 (this session's hunks). The
  owner asked whether self-training is tackling label shift and not domain shift: section 8, Z9 to
  Z15. Of its predictions: held Z9a, Z9c for unadapted models, Z9d on the letter, Z11, Z12, Z13
  (except the list alone on DINOv3), Z10b for DINOv3 and for congeners, Z14's seeds, Z15 with the
  prior from all of the site's crops; falsified Z9b in most parts, Z10b for V2-S by 2 points, Z14's
  three predictions on tta3bm (each in the prior's favour), Z15 with the prior from other nights. Z10
  was measured wrongly and is withdrawn. Six checkpoints and OT1's and OT2's logits downloaded from
  UCloud (files only, no job); checkpoints deleted, 15 logit sets kept (33 GB). The
  congener finding was handed to the `lepinet` session for the Merge Desk of paper A.
- 2026-11-15: committed and pushed as 7efcf65 (this session's hunks only), reported to the
  `lepinet` session. Section 7 opened with its predictions.
- 2026-11-15: Z5 to Z8 run and scored. Of section 7's predictions: held Z5e, Z6b, Z6c, Z8a, Z7's
  "marginal above 60 %", Z7b on the test fold and under centring, Z5b for Nemo, Z6a's range;
  falsified Z5a, Z5c (two parts of three), Z5d, Z5g, Z7's "flat under 35 %" and its crowding effect,
  Z7b on the probe, Z6a's backbone effect, Z8b on V2-S (it holds on Otto's folds at two sites of
  three); not met Z5f. 44 checkpoints were downloaded from UCloud (no job submitted) and deleted
  after use; 11 single-view logit sets stay in `data/population_probes/logits/` (13 GB) for the
  calibrated rescoring. Not committed.
- 2026-11-13: opened at the owner's request. Predictions written before each check; Z1, Z1d, Z2,
  Z2c, Z3 and Z4 run and scored the same day. Scoreboard of my 13 predictions: held Z1a, Z1b, Z1d,
  Z2a (on the letter), Z2b, Z2c's first half, Z4a, Z4b; held at two sites of three Z4c; falsified
  Z1c (which was the point of the test), Z2c's second half, Z3a. Nothing committed, nothing
  submitted; `PLAN.md` carries the routes as one proposed row.
