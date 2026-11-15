# Beyond scaling what works: what would make a better biodiversity classifier and backbone?

**Kind:** research · **Status:** OPEN (2026-11-13). A review of the project for new scientific
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
- Routes 5.4, 4 (taxon dropout) and 6 are designs with no measurement behind them.

## Log

- 2026-11-13: opened at the owner's request. Predictions written before each check; Z1, Z1d, Z2,
  Z2c, Z3 and Z4 run and scored the same day. Scoreboard of my 13 predictions: held Z1a, Z1b, Z1d,
  Z2a (on the letter), Z2b, Z2c's first half, Z4a, Z4b; held at two sites of three Z4c; falsified
  Z1c (which was the point of the test), Z2c's second half, Z3a. Nothing committed, nothing
  submitted; `PLAN.md` carries the routes as one proposed row.
