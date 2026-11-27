# Should the cosine head's per-class scale and bias be learned or frozen?

**Kind:** research · **Status:** RESOLVED (2026-11-26, by J4: [[2026-11-26-can-one-model-have-a-clean-embedding-and-the-learned-heads-f1]]): learned, without the leaky ReLU before the normalisation. The geometry costs below were the learned scale together with the ReLU: without it the learned head (J4b) gains 0.7 points of species F1, its rows become prototypes (+0.70) and most of its embedding gap to the frozen head closes. History: J1 scored (seed 42). Frozen loses 2.7 points of test-fold species F1 (6.5 in species under 100 images); bias only recovers nothing. The learned head wrecks the embedding for open set (maximum cosine to a head row 0.54 against 0.78, 1-NN -3.4; class means of training images 0.86 against 0.90, the best score on the frozen head), while the best logit score is equal (about 0.90). The learned scale causes both effects; the bias is inert. J2 scored (2026-11-26): with sqrt oversampling the frozen head is still 1.6 points below the learned head (1.5 calibrated; tail 2.1), and one learned global scale (s = 8.9) recovers only part of that (-1.0 zero, -0.7 calibrated) and points away from its images too (cosine -0.25), so the geometry follows the size of the scale, not its being per class; the global scale has the worst open-set scores and the best calibration. The decision rule falls on its 1-point line. J3 scored (2026-11-26): on the frozen head's trunk a learned scale and bias recover 2 % of the gap, a whole retrained head 28 % (its rows drift toward anti-prototypes), so the learned head's lead is in the trunk it trains, not in its decision rule; supervisor recommended the learned head (tol-full's), then, after J4, the learned head without the ReLU; owner decides the bundle's head.

## The question, in plain words

Every head in this project scores an image against each species with a cosine, stretched into a logit by a
fixed constant: `z = sqrt(D - 2) * asin(cos)`, where D is the embedding width. Two versions exist.

| | logit for species k | learned per species |
|---|---|---|
| frozen (mini_trainer's default; paper A's main recipe) | `sqrt(D-2) * asin(cos)` | nothing |
| learned (lepinet; paper A's replication recipe; every recent run) | `sqrt(D-2) * asin(clip(g_k * cos)) + b_k` | a scale `g_k` and a bias `b_k` |

The learned version was never chosen. lepinet meant to freeze `g` and `b`, but fastai's optimiser setup trained both
in every lepinet run ([[2026-08-06-the-cosine-head-is-not-unit-norm]]). The mini_trainer port reproduced that for
parity, through the `lepinet-*` presets (`trainable_scale`, `trainable_bias`). The parity audit marked the factor
"unknown" ([[2026-11-03-moving-the-recipe-into-mini-trainer]]), and the no-grid decision of 2026-11-04 kept it. So
tol-full, TL10, the 1 % fit runs, OT6, SC1 and the TL8 heads all learn a scale and a bias per class.

With `g > 1`, distant species fall below -1 inside the arcsine and are clipped to a floor: tol-full has median `g`
1.41 and 78 % of image-species pairs on the floor. Its logits are therefore not the standard scores the frozen
version's argument relies on.

The owner prefers no trainable parameters in the head (2026-11-24). For the frozen version:
- the logits keep a fixed scale and a known null distribution, which is what open-set scores and calibration need;
- there is no floor;
- the head rows stay plain prototypes.

The one related test is CF ([[2026-11-19-does-the-cosine-heads-clamp-floor-cost-accuracy]]). It removed the floor by
moving `g` after the arcsine and lost 1.0 point of test-fold macro-F1 on V2-S and 1.1 top-1 at 190 k classes. So the
floor acts as a learned gate. Freezing `g` at 1 also removes the floor, but no run has tested that.

## The test (J1)

The SC1 and CF testbed: the shard control v2s_bu_ctl_sh_s42 (EfficientNetV2-S BU, Global Lepidoptera shards, seed 42,
5 epochs at 256 px, no oversampling, one B200 per arm). The whole network is trained, not a head on a frozen trunk.
The three arms run in one job and differ from the control only in these keys:
- **J1-ctl**, the fresh control: scale and bias learned. This is the unfrozen run the owner asked for, and it checks
  today's code against the earlier shard controls (0.8880 / 0.8793) and the seeds (0.8866-0.8877).
- **J1-frz**: scale and bias frozen; no per-class parameter.
- **J1-b**: scale frozen, bias learned. It separates the floor (scale) from the per-class prior (bias), so a loss
  by J1-frz does not need a second round.

Scored on:
- the test fold, by rank and by frequency bin;
- top-1;
- the traps (probe and probe-HO);
- open-set detection on the cut tail (prediction 7), with logit scores and embedding scores. Also, mt's VO pipeline needs models trained with 231 taxa withheld (c3b_train), and J1 trains on all
  12,041 species. So the J1 report uses a proxy from its own trap logits: crops of species outside the vocabulary
  against crops inside it (entropy, MSP and max-logit AUROC). The proxy is domain-shifted and not comparable to VO.
  VO-style runs (about 6-7 B200-h per head, possibly after a shard copy of c3b) follow only if a head passes the
  accuracy rule, or loses by a margin open-set could outweigh; the owner decides;
- calibration error without a temperature;
- the floor counts and the median `g`.

## Predictions (before any number)

1. J1-ctl equals the earlier shard control within the seed range: no code drift.
2. J1-frz scores 0.5-1.5 points below J1-ctl on test-fold species macro-F1, mostly in the <100 and 100-200 bins, and
   0.2-0.8 lower on top-1. The reason: CF lost 1.0 when the floor went, and a frozen head cannot sharpen per class.
3. J1-b is within 0.3 of J1-frz: the scale, through the floor, is the lever, not the bias.
4. Without a temperature, J1-frz has a lower calibration error than J1-ctl on the test fold (moderate confidence).
5. Open-set AUROC (trap proxy): no predicted direction, within 0.02 either way.
7. Open-set on the cut tail (added 2026-11-24, before any number; owner: "if trainable scale and bias are downgrading
   [open-set detection] we might wanna not include them. But I would like to be very sure"). Unknowns are the global_lepi
   species under the 50-image floor: 37,597 species and 372,263 images on UCloud, the same photo domain, scored near / mid /
   far by what is trained. On the learned heads, VO found max logit and energy at 0.5-0.72 against 0.88 for entropy.
   - Max logit: J1-frz 15-30 points above J1-ctl. A frozen head's max logit is a monotone function of the maximum raw
     cosine, while a learned scale and bias shift every class by its own offset.
   - The best rule per arm (entropy or MSP): within 1.5 points.
   - Embedding scores (maximum raw cosine to a prototype, k-NN): within 1.5 points. The trunk sets them, and the scale and
     bias touch it only through training.
   - If this holds, the learned head costs nothing for the best rule, but makes the raw logit unusable as a score.
6. Calibrated trap probe: within 1 point either way.

## An observation before J1 (mt, 2026-11-24, on the cfsh control's weights, learned head)

While checking the open-set pipeline, mt found that the learned head's embedding sits on the far side of the prototypes:
- the predicted class's raw cosine is about -0.3 to -0.45;
- the mean cosine over all classes is -0.49;
- 54 % of classes are on the floor (`g * cos <= -1`).

A class is chosen by being the least anti-aligned, not by being close. With `g > 1` the floor makes this a cheap
solution: every other class is pushed to the floor (a logit of about `-sqrt(D-2) * pi/2 + b`), and the target only has to
stay above it. So the learned head's rows are not prototypes of their images in the usual sense. That matters for the
owner's use of the normalised space for open-set detection: a distance to a head row means something else here. The
image-to-image geometry (nearest class means, k-NN) is a separate matter: Z19 and Z20 used it and it was strong. A
frozen head has no floor, so cross-entropy can only raise the target logit by raising its cosine. I expect positive,
prototype-like cosines there. J1 measures it (raw cosine of the predicted and true class, and the floor share, per arm).

## Decision rule

- If J1-frz is within 0.33 points (three times the seed range) of J1-ctl on test-fold species macro-F1, and is not
  worse beyond noise on the traps or on open-set AUROC, future runs use the frozen head.
- If J1-frz loses by more and J1-b recovers to within 0.33, the head keeps only the bias.
- Otherwise the learned head stays, and the owner decides whether its open-set and calibration properties are worth
  the accuracy.

J1 lands before the ToL bundle launches, so the owner can add the head choice to it. That would make it a fifth change
the bundle cannot attribute.

## Result (mt, 2026-11-25; seed 42; `plans/j1/README.md` in the campaign, `/data/au761367/lepinet/j1/`)

J1 trained in 4.7 h (job 12477531). The evaluations e-j1-ctl / frz / b and the open-set job e-j1-os2 all exited 0.
e-j1-os failed at setup with no compute spent: its inputs sat in `.agents/local`, which the campaign sync leaves out.
mt uploaded them and resubmitted the job unchanged.

**Accuracy (test fold line 1).**

| arm | species F1 zero / calibrated | genus | family | species top-1 | val macro-F1 e1 ... e5 |
|---|---|---|---|---|---|
| learned (ctl) | **0.8876** / 0.8799 | 0.9559 | 0.9754 | 93.18 | .711 .781 .822 .867 .887 |
| frozen (frz) | 0.8602 / 0.8493 | 0.9442 | 0.9713 | 92.28 | .645 .704 .762 .824 .849 |
| bias only (b) | 0.8590 / 0.8481 | 0.9432 | 0.9692 | 92.29 | .643 .705 .762 .826 .849 |

- The frozen head loses 2.7 points of species F1 (3.1 calibrated), 27 times the seed range of the control. It loses
  them in the tail. Per bin (<100, 100-200, 200-500, 500-1000, >=1000), ctl scores .844 .891 .914 .935 .944 and frz
  .779 .867 .901 .926 .939: -6.5 points under 100 images, -0.5 at 1,000 or more.
- On validation the gap narrows during training, from 7.7 points at epoch 2 to 3.8 at epoch 5. Whether a longer run
  closes it is untested.
- A learned bias recovers nothing. The scale carries the gain.
- Note: `mini_metric.csv`'s `correct` column is coded +1 right, -1 wrong, 0 no prediction (the mini_metrics convention,
  mini_trainer `logging/collector.py`), so its mean is 2 x top-1 - 1 (0.864 for 0.932). mt checked that no earlier top-1
  in the project came from it (CF, cftol, SC1, the TL8 and OT tables use label == prediction or F1).

**Traps (ne-tta3bm, line 1).**

| arm | probe calibrated / zero | probe-HO calibrated / zero | with list + EM prior (rule c): probe / probe-HO |
|---|---|---|---|
| ctl | .627 / .419 | .465 / .239 | .664 / .536 |
| frz | .635 / .420 | .494 / .235 | .643 / .543 |
| b | .626 / .423 | .507 / .241 | .646 / .509 |

Without the list, the frozen arms are level or better (probe-HO +2.9 and +4.2). With the list and the EM prior, the
deployed rule, the learned head leads on the probe by about 2 points. No clear winner on traps.

**Calibration.** Debiased L2 error, temperature 1:
- test fold: ctl .030, frz .029, b .030;
- traps (in-vocabulary crops): ctl .019, frz .084, b .086. The frozen heads are underconfident on traps (mean
  confidence .729 against accuracy .784).

**Open set, in domain.** Unknowns are the cut tail: 61,160 images, near / mid / far 32,159 / 27,851 / 1,150, from
species under the 50-image floor. Two sets of knowns from the test fold:
- known_bal (main): at most 5 per species, which weights rare species up as a macro average does;
- known_uniform: a uniform sample, closer to a real stream.
k-NN uses up to 10 training images per species as reference. AUROC, all strata. The paired 95 % intervals of frz - ctl
(a bootstrap over species, mt's method) are +/-0.002 to +/-0.008:

| score | ctl (bal) | frz (bal) | frz - ctl (bal) | frz - ctl (uniform) |
|---|---|---|---|---|
| entropy | **.897** | .894 | -.003 | +.011 |
| MSP family | .897 | **.898** | +.001 | +.015 |
| max logit | .648 | .783 | +.135 | +.176 |
| maximum cosine to a head row | **.537** | .783 | **+.246** | +.331 |
| 1-NN (image to image) | .857 | **.890** | +.034 | +.036 |
| 10-NN mean | .795 | .873 | +.077 | +.075 |

On known_bal's near stratum the learned head leads on the logit scores by about 0.01 (entropy -.010 [-.013, -.007]).
The frozen head leads on the embedding scores there too (1-NN +.028). On mid and far the frozen head is better or
equal on every score. The bias-only arm equals frz throughout. The trap open-set proxy had 234 out-of-vocabulary crops
from 6 species, with intervals of +/-0.1-0.2: uninformative.

**Geometry (known_bal, median and 5-95 %).**

| | ctl | frz | b |
|---|---|---|---|
| cosine to the predicted class | -0.44 (-0.52, +0.04) | **+0.29** (+0.12, +0.42) | +0.29 |
| share of classes on the floor | 0.84 | 0 | 0 |
| median g | 1.81 | 1 | 1 |

The true class sits within 0.01 of the predicted class's cosine. The learned head's rows are anti-prototypes: an image
is assigned to the class it points away from least. The frozen head's rows are prototypes.

## Predictions scored

1. ctl equals the shard control: **right** (0.8876 / 0.8799 against 0.8880 / 0.8793).
2. frz 0.5-1.5 below ctl, in the <100 and 100-200 bins; top-1 0.2-0.8 lower: **direction and place right, size wrong**.
   The loss is -2.7 (-6.5 in the rarest bin), and top-1 -0.9.
3. b within 0.3 of frz: **right** (-0.1).
4. frz better calibrated without a temperature: **wrong**. It is equal on the test fold and worse on traps.
5. Trap-proxy open set within 0.02: **not scoreable** (234 crops).
6. Calibrated trap probe within 1 point: **right** (+0.8).
7. Cut tail:
   - max logit, frz 15-30 points above ctl: **about right** (+13.5 on known_bal, +17.6 on known_uniform);
   - best rule within 1.5 points: **right** (within 0.2);
   - embedding scores within 1.5 points: **wrong**. Max cosine +24.6, 1-NN +3.4, 10-NN mean +7.7.
   I reasoned that the trunk sets these scores and the head touches them only through training. Training is exactly
   where the head reshapes the space. The learned head damages even image-to-image neighbourhoods, which never touch a
   head row.

## Answer to the owner's question

Does a learned scale and bias downgrade open-set detection in the normalised embedding space? **Yes, and decisively.**
- Maximum cosine to a prototype falls to chance (0.54; 0.48 on uniform knowns).
- k-NN loses 3.4 to 7.7 points.
- The best logit score (entropy, or family MSP) is unaffected (about 0.90 in both).
This is one seed, but the confidence does not rest on the intervals alone. The embedding differences are 10 to 40 times
the intervals' half-width, and they follow from a change of geometry, from anti-prototypes at -0.44 to prototypes at +0.29. A second seed
would not flip that. Both observations hold on two different sets of knowns.

## Decision rule applied

- frz is 2.7 below ctl, beyond 0.33.
- b does not recover.
- So the rule's third branch applies: the learned head stays unless the owner trades accuracy for the embedding's
  open-set use. The trade is 2.7 points of species F1, 6.5 of them in species under 100 images. At ToL scale, 103 k of
  190 k species sit in that bin.

**A third option (supervisor, 2026-11-25).** The frozen head is the s = 1 member of a family with one fixed constant:
`z = s * sqrt(D-2) * asin(cos)`. That is a temperature, not a per-class parameter, and it has no floor.
- In CF, the version with a learned per-class scale after the arcsine (no floor) learned median g 2.42 and lost 1.0
  point, not 2.7. So most of the scale's value may be a global temperature.
- In CF, that version also kept wrong species near orthogonal (median cosine +0.02) instead of anti-aligned.
- The frozen arms' underconfidence on traps also fits a temperature that is too low.
Proposed J2, for the owner's go: the J1 testbed with a fixed s of 2.4 and 1.8, two 1-B200 arms, about 10 B200-h with evaluations.
Decision rule, written before any number:
- adopt the fixed-s head if it is within 0.33 of ctl on test-fold species F1 and within 0.01 of frz on 1-NN and
  max-cosine AUROC;
- between 0.33 and 1.0 below ctl, the owner decides;
- beyond 1.0 below ctl, the learned head goes into the bundle, and open-set uses entropy or family MSP.
Feasible without mini_trainer code (mt, 2026-11-25):
- `magnitude_after_angle` with the scale and bias frozen gives exactly this head, but mini_trainer always initialises g
  to 1.
- So a campaign-side helper builds the J2 model at seed 42, sets g to s, saves it, and trains with `--weights`. The
  load keeps the frozen flag and skips the spherical initialisation.
- A local check that the loaded g equals s comes first.
If a fixed scale is adopted, an opt-in mini_trainer option for it is the clean form, so the recipe's config states s.
Prediction: s = 2.4 lands 0.5-1.0 below ctl, with a positive predicted-class cosine (+0.10 to +0.25) and embedding
AUROC within 0.02 of frz.

Seeds 43 and 44 of ctl and frz are not needed for this decision. The accuracy gap is 27 times the seed range, and the
open-set effect is structural.

## The scale as a hyperparameter (owner's questions, 2026-11-25)

The owner asked for the heads written out, whether a fixed scale is a hyperparameter, whether a formula can set it, and
whether it should depend on the class. `e` is the image embedding and `w_k` row k of the head, both unit length;
`c_k = w_k . e = cos(theta_k)`; `S = sqrt(D - 2)` (35.7 for V2-S, D = 1280).

```
learned (every recent run)   z_k = S * asin(clip(g_k * c_k, -1, 1)) + b_k    g_k, b_k trained per species
frozen (J1-frz)              z_k = S * asin(c_k)                              nothing trained
bias only (J1-b)             z_k = S * asin(c_k) + b_k
CF's fix                     z_k = g_k * S * asin(c_k) + b_k                  g_k trained per species
J2 (fixed scale)             z_k = s * S * asin(c_k)                          s one constant for all species
```

Since `asin(cos theta) = pi/2 - theta`, the frozen logit is `S * (pi/2 - theta_k)`, linear in the angle. J2 multiplies
it by `s`, so `s * S` is an inverse temperature on angles, and s = 1 is the frozen head.

The learned head is steeper by `g_k` near orthogonal, and it is cut at both ends:
- a ceiling (`+S*pi/2 + b_k`) once `c_k >= 1/g_k`;
- a floor (`-S*pi/2 + b_k`, no gradient) once `c_k <= -1/g_k`. At the median `g` of 1.81, a species further than
  about 123 degrees from the image is switched off.

Why `S = sqrt(D - 2)`: for a random direction, `pi/2 - theta` has standard deviation `1/sqrt(D - 2)`, so the frozen
logit is a z-score. It counts the standard deviations by which the image is closer to species k than chance.
- A global `s` keeps a known null, the same for every species: a standard deviation of `s`.
- A per-species `g_k` or `b_k` gives each species its own null. That is why the learned head's maximum logit and
  maximum cosine fail as open-set scores.

**Is s a hyperparameter? Yes: one number, like CLIP's or SimCLR's temperature.** The z-score fixes the unit, but
nothing in it says how sharp the softmax must be to train well.

**Formulas exist, and none explains J1.**
- NormFace (Wang et al. 2017) and L2-constrained softmax (Ranjan et al. 2017) bound the scale from below. Cross-entropy
  over C classes needs a logit range of order `ln C`. The frozen head has `+/-S*pi/2 = +/-56` against `ln 12,631 = 9.4`.
- AdaCos (Zhang et al. 2019) sets the scale so that a target at the batch's median angle gets probability 0.5. Its
  adaptive version recomputes it from each batch, with no trained parameter. Applied roughly to J1-frz (median target
  cosine +0.29, logit 10.5 against `ln(C-1)`), it gives `s` of about 0.9.
- So the formulas say the frozen head is sharp enough, yet it lost 2.7 points. They were built for face verification
  (balanced classes, embedding quality), not for accuracy on a long tail.

A mechanism that fits J1 (a hypothesis):
- A soft softmax spreads each image's push-away gradient over many species. A rare species collects many pushes from
  other species' images and few pulls from its own, so its row is suppressed. This is the long-tail "negative gradient"
  problem (Equalization Loss, Tan et al. 2020).
- A sharper softmax (larger `s`) concentrates the push on the nearest competitors. The learned floor goes further and
  switches far species off (CF: learned hard-negative mining).
- It predicts what J1 shows: the frozen head's loss sits in the tail (-6.5 under 100 images, -0.5 at 1,000 or more).
- It also predicts that sqrt oversampling, which gives rare species more pulls, shrinks the deficit. J1's testbed has
  no oversampling and the ToL bundle has sqrt oversampling, so J1 does not yet answer the bundle's question.

**Per species or global? Global (agreeing with the owner).**
- `b_k` is a learned log-prior: it bakes in the training frequencies, which label shift makes wrong. The project
  applies priors explicitly (list, EM), and J1 shows the bias buys nothing.
- `g_k` sets each species' confidence from its training history and breaks the common null.
- One counterpoint: on traps with the list and the EM prior, the learned head led the probe by 2 points. It also led
  the test fold by 2.7, so this does not single out the per-species parameters.

**J2, revised (supervisor, 2026-11-25; replaces the proposal above; needs the owner's go and free storage).**
- The J1 testbed with sqrt oversampling (`class_sampling_power` 0.5) in every arm, as in the bundle. Three arms, one
  factor apart: learned, frozen (s = 1), fixed s = 2.4.
- One 3-B200 job of about 4.7 h, about 16 B200-h with evaluations.
- The open-set job scores J1's and J2's models together, adding one embedding score: cosine to the nearest class mean
  of the reference images. That is a prototype computed from data, so it exists for tol-full too.

Predictions (before any number):
1. learned + sqrt beats J1-ctl by 0.5-2 points (TL11 (C): +1.6 to +1.9 on bank specialists).
2. frozen + sqrt is 1.0-2.0 below learned + sqrt, down from 2.7, with the gain in the <100 bin.
3. s = 2.4 is within 0.5 of learned + sqrt, with a predicted-class cosine of +0.10 to +0.25 and embedding AUROC within
   0.02 of frozen.
4. Class-mean distance on J1-ctl scores far above its maximum cosine to head rows (0.54), and below frozen's 1-NN.

Decision rule:
- if frozen + sqrt is within 0.33 of learned + sqrt and no worse on open set, the bundle uses the frozen z-score head
  (no hyperparameter);
- otherwise, if s = 2.4 is within 0.33, the fixed scale;
- between 0.33 and 1.0, the owner decides;
- beyond 1.0, the learned head, with open set by entropy.

A fixed `s` chosen at 12.6 k classes may not carry to 190 k (the formulas scale it with `ln C`). The 1 % ToL subset
could check that before the bundle (about 20 B200-h); not proposed yet.

Consequence for existing models: tol-full and every recent run have the learned head (tol-full: median g 1.41, 78 % of
pairs on the floor). Embedding-based open-set scores on them (maximum cosine to a head row) are probably degraded the
same way. Paper B's open-set numbers should use entropy or MSP, as TL2 did, or k-NN, until a model without the floor exists.

## Class means (supervisor, `dev/119`, 2026-11-25, CPU)

A fourth embedding score: the cosine to the nearest class mean, the renormalised average of each species' 10 reference
images, from e-j1-os2's saved embeddings. It is a prototype computed from data, so it exists for any model.

Checks passed first:
- 1-NN recomputed from the embeddings matches mt's stored values on 3,000 queries (largest gap 0.0000);
- the AUROC code reproduces mt's 1-NN and maximum-cosine AUROC.

AUROC on known_bal (known_uniform agrees within 0.004), with 95 % intervals from a bootstrap over species:

| | near | mid | far | all |
|---|---|---|---|---|
| class mean, ctl | 0.820 | 0.912 | 0.913 | **0.864** [0.859, 0.868] |
| class mean, frz | 0.856 | 0.955 | 0.950 | **0.903** [0.899, 0.906] |
| head row (maximum cosine), ctl / frz | 0.518 / 0.741 | 0.554 / 0.826 | 0.660 / 0.935 | 0.537 / 0.783 |
| 1-NN, ctl / frz | 0.811 / 0.839 | 0.909 / 0.949 | 0.870 / 0.910 | 0.857 / 0.890 |
| entropy, ctl / frz | 0.857 / 0.847 | 0.941 / 0.945 | 0.959 / 0.969 | 0.897 / 0.894 |
| family MSP, ctl / frz | 0.848 / 0.849 | 0.949 / 0.951 | 0.982 / 0.980 | 0.897 / 0.898 |

- **Class means rescue the prototype route on the learned head:** 0.864 against 0.537 for its head rows (+0.33
  [+0.32, +0.34]). They stay 3.3 points below that head's own entropy.
- **On the frozen head, the class mean is the best single score overall** (0.903, above family MSP 0.898 and entropy
  0.894). It is best on near and mid; family MSP stays best on far (0.980).
- Freezing helps the class mean by +0.039 [+0.036, +0.043], every stratum alike.
- The bias-only arm equals frz.

Prediction 4 of the revised J2 (ctl's class mean far above its head rows, and below frz's 1-NN): **right** on both
counts.

For tol-full and the other existing models: prototype-based open set does not need a new model; class means of their
training images replace the head rows. The measured cost of the learned head here is about 4 points on that score.


## The owner's reply, and what J1 already says about it (2026-11-25)

The owner: a fixed s looks like a strange middle ground that defeats the purpose of S. The preference for the frozen
head came from wanting a "natural" embedding whose head rows are species prototypes. If an output score (entropy) does
open-set detection as well on the learned head, the learned scale and bias can stay and only the score changes; "can we
then do the same things, head rows as prototypes included?" Does ArcFace learn its scale and bias? Would oversampling
remove the tail effect? How much do the scale and the bias each matter? All of it only matters if a better embedding
is wanted; otherwise no more experiments.

What is already measured, and the answers:

- **Scale against bias: J1 separates them.** The bias-only arm equals the frozen head on every measure (species F1
  0.8590 against 0.8602, cosine to the predicted class +0.29 for both, class-mean AUROC 0.903 for both). The learned
  bias does nothing measurable, good or bad. The per-species scale alone buys the 2.7 points, and the per-species
  scale alone turns rows into anti-prototypes.
- **ArcFace and its family use one scale for all classes.** ArcFace and CosFace fix s = 64 on the raw cosine and drop the
  bias; NormFace learns s as one scalar; AdaCos adapts one scalar from the batch. None has a per-class scale or bias.
  The fixed-s arm is therefore the ArcFace setting, not a middle ground; the departure is our learned per-class g.
  Their well-structured embeddings are the geometry the frozen and bias-only arms have here (rows at +0.29 to their
  images). S = sqrt(D - 2) sets the unit (a z-score under random directions); s, ArcFace's 64, is the temperature on
  top. If a fixed s is ever tested, a single learned scalar (NormFace) removes the hand-set value without giving back
  per-class freedom; it needs a small mini_trainer change (one g shared by all rows).
- **Untearing the space with the inverse of g_k and b_k.** Inverting them on a logit returns c_k, the cosine to the
  row: exactly what maximum cosine uses (0.537). The transform acts on scores; the rows themselves moved during
  training, so no per-class map of the scores turns them back into prototypes. The logit, which already includes g_k
  and b_k, is the comparison that works on this head, and entropy and family MSP use it. The data prototypes (class
  means) are the repair: 0.864.
- **What still works on the learned head.** Output-score open set: yes (entropy 0.897 against the frozen head's best
  score, class means, 0.903). Head rows as prototypes: no; class means instead, 3.9 points below the frozen head's and
  as large as the head to ship. Image-to-image neighbours: 1-NN 0.857 against 0.890.
- **Oversampling.** That is the mechanism the revised J2 tests (every arm with sqrt oversampling); untested. The bundle
  has sqrt oversampling and tol-full's learned head, so it gives no comparison.

Supervisor's recommendation: keep tol-full's head in the bundle unchanged, as agreed for the bundle; entropy or family
MSP as the open-set score; class means computed and shipped only when a use needs prototypes (adding a species from a
few images, retrieval, paper B's prototype claims). J2 is not run now. If such a use appears, the arm to run is one
learned global scale with sqrt oversampling. Owner decides.

## J2, as decided (owner, 2026-11-25): one learned global scale, before the bundle

The owner wants the best head before the bundle spends about 500 GPU-hours, and asked for the learned global scale to be
tested against an oversampled frozen head and an oversampled learned head. The supervisor's "not now" had assumed the
bundle came first and that 0.6 points on the best open-set score did not justify a run; at about 16 B200-h, 3 % of the
bundle, testing first is the better bet. The bundle waits for J2.

Arms: J1's testbed (EfficientNetV2-S BU, Global Lepidoptera shards, seed 42, J1's schedule), every arm with sqrt
oversampling (`class_sampling_power` 0.5) as in the bundle:
- **ctl-os**, the learned head: z_k = S asin(clip(g_k c_k)) + b_k;
- **frz-os**, the frozen head: z_k = S asin(c_k);
- **gs-os**, one learned global scale: z_k = s S asin(c_k), s a single trained scalar (initialised at 1, no weight
  decay on it), no bias. NormFace's learned scale on our z-score unit; it keeps one null for every class. Needs a
  small opt-in change in mini_trainer (the magnitude-after-angle form with g tied across rows, bias frozen at 0).

Same evaluations as J1 (test fold line 1, traps ne-tta3bm, calibration, open set on the cut tail with
`/work/global_lepi/cut_tail`), plus the class-mean score (dev/119) and the final s.

Predictions (before any number):
1. Oversampling shrinks the frozen deficit: frz-os - ctl-os between -0.5 and -1.5 species F1 (J1 without
   oversampling: -2.7), still mostly under 100 images.
2. s rises above 1 and ends between 1.5 and 3; gs-os within 0.5 of ctl-os.
3. gs-os keeps prototype geometry: median cosine to the predicted class above +0.2; maximum-cosine AUROC within 0.03
   of frz-os; class-mean AUROC at least 0.89.
4. ctl-os keeps anti-prototypes (median cosine to the predicted class below 0).
5. The best open-set score of each arm within 0.01 of the others.

Decision rule (written now): gs-os goes into the bundle if its species F1 is within 0.3 of ctl-os (J1's seed range
was about 0.1); otherwise frz-os if within 0.3; a gap of 0.3 to 1 point goes to the owner; beyond 1 point, the learned
head with entropy for open set. Transfer risk: 12,632 classes here against 190,000 in ToL. A learned s adapts to the
class count by itself, which a hand-set s would not.

Submitted 2026-11-25 as J1 ran, three arms on one 3-GPU node (job 12479511). It waited 4.5 h for a node, while every
1-GPU job of the past week had started within 0.7 h (J1's 3-GPU job waited 5 h; 8-GPU jobs 0.1 to 14 h). On the owner's
question it was cancelled unstarted and resubmitted at 19:13 as three 1-GPU jobs (j2-ctl 12479905, j2-frz 12479906,
j2-gs 12479907). A 1-GPU B200 node is a third of the 3-GPU one (48 vCPU, 288 GB), so each arm keeps its GPU, its 44
data workers (and with them the shard order) and its configuration, byte for byte; only the node is no longer shared.

## J2 result (scored 2026-11-26)

All seven jobs exited 0 (training 19:50 to 01:08, about 4.3 h per arm; about 16 B200-h in all). Every weights folder
ends with best.pt and last.pt only, so checkpoint retention works at this scale. Numbers from mt's scoring
(mini_trainer `.agents/local/campaign/plans/j2/README.md`, "J2 result"); class means from dev/119 run on e-j2-os's
embeddings (`--dir /data/au761367/lepinet/j2/openset --arms ctl,frz,gs`), which first reproduced mt's 1-NN and maximum
cosine AUROC to three decimals.

| | learned (ctl-os) | frozen (frz-os) | global scale (gs-os) |
|---|---|---|---|
| test-fold species F1, zero / calibrated | **0.8961 / 0.8998** | 0.8798 / 0.8850 | 0.8861 / 0.8926 |
| gap to the learned head, zero / calibrated | | -1.63 / -1.49 | -1.00 / -0.72 |
| species F1 under 100 images / 1,000 and over | **0.885** / 0.942 | 0.864 / 0.935 | 0.876 / 0.935 |
| genus / family F1 (zero) | **0.956** / 0.968 | 0.948 / **0.975** | 0.949 / 0.974 |
| trap probe, calibrated: line 1 / list + EM | **0.630** / 0.630 | 0.608 / 0.624 | 0.590 / **0.641** |
| trap probe held out, calibrated: line 1 / list + EM | 0.472 / 0.464 | 0.468 / 0.496 | **0.490 / 0.511** |
| calibration error (debiased L2, T = 1): test fold / trap vocabulary / trap list | 0.057 / 0.062 / **0.034** | 0.037 / 0.089 / 0.082 | **0.014 / 0.046** / 0.040 |
| open set, best logit score (AUROC, balanced knowns) | 0.913 (max logit; entropy 0.903) | **0.924** (family MSP) | 0.908 (family MSP) |
| open set: class mean / 1-NN / maximum cosine to a row | 0.893 / 0.876 / 0.542 | **0.912 / 0.902 / 0.812** | 0.862 / 0.846 / 0.726 |
| median cosine to the predicted class's row | -0.58 | **+0.33** | -0.25 |
| learned scale s (epochs 1 to 5) | | | 3.65, 5.09, 6.45, 8.12, 8.92 |

Paired bootstrap over species: class mean frozen - learned +0.020 [+0.018, +0.022], global scale - learned -0.031
[-0.033, -0.028]; uniform knowns +0.030 and -0.028. The trap rows have no intervals yet and disagree with each other
(the learned head leads on line 1, the global scale on list + EM); they decide nothing here.

**Oversampling, one factor against J1** (same testbed and seed): the learned head +0.85 species F1 zero, +2.0
calibrated, +4.1 under 100 images (0.844 to 0.885); the frozen head +2.0 and +3.6, +8.5 under 100 images. The frozen
deficit shrinks from 2.7 to 1.6 points (tail 6.5 to 2.1) but does not close. On this testbed, sqrt oversampling is
confirmed for the bundle.

**Predictions scored.**
1. Frozen - learned between -0.5 and -1.5: -1.63 zero (just outside), -1.49 calibrated (just inside); mostly the
   tail (-2.1 under 100 images, -0.7 over 1,000): **half right**.
2. s ends between 1.5 and 3, global scale within 0.5 of the learned head: s = 8.92 and still rising (+0.8 in epoch 5);
   -1.00 / -0.72: **wrong on both**.
3. The global scale keeps prototype geometry (cosine above +0.2, maximum cosine within 0.03 of frozen, class mean at
   least 0.89): -0.25, 0.726 against 0.812, 0.862: **wrong on all three**.
4. The learned head keeps anti-prototypes: -0.58 (J1 -0.44): **right**.
5. Best open-set scores within 0.01: frozen leads by 0.011 (balanced) and 0.019 (uniform): **wrong**, narrowly.

**What it changes.** J1 concluded that the per-species scale causes both the F1 gain and the anti-prototypes. J2
corrects the second half: one scale shared by every class, which cannot favour any species, grew to 8.9 and also
turned the rows away from their images (-0.25). The geometry follows the size of the logit scale, not its being per
class. A reading consistent with all three arms: at a large scale, small cosine differences already make the softmax
confident, so training only orders the cosines and leaves their level free to drift; at the frozen head's fixed
scale, the only way to a confident softmax is a large cosine to the true row (+0.33). The per-species freedom adds
the rest of the F1 gain (the global scale recovers 39 % of the frozen deficit at zero threshold, 52 % calibrated) and
deeper anti-prototypes. So in this family of heads accuracy and prototype geometry trade against each other. Nothing
measured so far gets both. The global scale's one clear win is calibration at T = 1 (0.014 on the test fold): a
learned temperature. The learned head's calibration can probably be brought there with one temperature fitted after
training; not yet checked on J2.

**The decision rule.** Global scale - learned is -1.003 points zero and -0.721 calibrated. The rule does not say
which F1; J1 reported zero first. Zero puts it 0.003 past the 1-point line, which goes to the learned head with a
logit open-set score; calibrated puts it in the 0.3 to 1 band, which goes to the owner. Under neither reading does
the rule pick the global scale by itself. The frozen head is beyond 1 point on both. The supervisor's recommendation
is the learned head:
- the global scale was tested for one reason, prototype geometry at the learned head's accuracy, and fails on both
  halves: 0.7 to 1.0 points below in F1, and the worst open-set scores of the three (class mean 0.862);
- the frozen head is 1.5 to 1.6 points below in F1 (2.1 in the tail) for +0.011 open-set AUROC on the best score
  and +0.020 on class means;
- the learned head is tol-full's, so choosing it leaves the bundle with one changed factor fewer.

Transfer caveat, as before: 12,632 classes here against about 190,000 in ToL. Owner decides.

## Which model has the best embedding space? (owner, 2026-11-26)

The owner cares about the embedding itself and read the open-set numbers as favouring the frozen head. dev/122 measures
the embedding with the head set aside, on e-j2-os's saved embeddings: class means of 10 training images per species
as prototypes, test-fold images of known species (`known_bal`, 12,630 species) and the cut tail's species the models
never saw (`near`: new species of known genera, 4,000 species; `mid`: new genera of known families, 4,000 species).
Macro over species; brackets are paired bootstrap intervals over species.

| embedding only | learned | frozen | global scale | frozen - learned |
|---|---|---|---|---|
| known species, nearest class mean: species | 0.883 | **0.888** | 0.840 | +0.5 [+0.3, +0.7] |
| known species, nearest training image (1-NN) | 0.852 | **0.865** | 0.804 | +1.3 [+1.1, +1.5] |
| known species, nearest class mean: genus / family | 0.954 / 0.990 | **0.958 / 0.991** | 0.927 / 0.982 | +0.4 / +0.1 |
| new species of known genera: genus right | 0.513 | **0.526** | 0.479 | +1.3 [+0.6, +2.0] |
| new species of known genera: family right | 0.858 | **0.878** | 0.843 | +2.0 [+1.6, +2.5] |
| new genera of known families: family right | 0.639 | **0.672** | 0.627 | +3.3 [+2.5, +4.1] |
| open set, class mean / 1-NN AUROC (dev/119) | 0.893 / 0.876 | **0.912 / 0.902** | 0.862 / 0.846 | +2.0 / +2.6 |
| nearest other class mean is a congener (10,190 species) | 0.736 | **0.764** | 0.692 | |
| cosine between class means: same genus / same family / other family | 0.81 / 0.69 / 0.67 | 0.41 / 0.06 / 0.02 | 0.61 / 0.28 / 0.17 | |

| the head, same images | learned | frozen | global scale |
|---|---|---|---|
| known species: species / genus / family | **0.914 / 0.967 / 0.992** | 0.900 / 0.962 / 0.991 | 0.906 / 0.960 / 0.989 |
| new species of known genera: genus / family | **0.540** / 0.863 | 0.537 / **0.882** | 0.515 / 0.856 |
| new genera of known families: family | 0.638 | **0.687** | 0.624 |

The frozen head's model has the best embedding on every measure taken, and the global scale's the worst. The margins
over the learned head are small for known species (+0.5 by class mean) and grow with distance from the training
labels (+1.3 genus for new species, +3.3 family for new genera). The learned head's embedding sits in a narrow cone:
class means of different families have cosine 0.67, so the taxonomy is a small modulation on top of one shared
direction; the frozen head's means are near-orthogonal across families (0.02) and the taxonomy is the large structure
(0.41 within genus). Cosines are not comparable across the three spaces; the accuracies are.

The useful split: on known species the learned head beats its own embedding's prototypes by 3.1 points (0.914
against 0.883), the frozen head only by 1.3 (0.900 against 0.888), and the frozen embedding's prototypes are the
better of the two. The learned head's F1 advantage is therefore in the head's decision rule, not in a better trunk.
That is the case for the owner's proposal below.

**J3, a learned head on the frozen head's trunk** (owner's idea; owner GO 2026-11-26, with J3c added). Take frz-os,
freeze the whole trunk (and check its tensors are unchanged afterwards: 2026-11-05), and train the learned head's form,
z_k = S asin(clip(g_k c_k)) + b_k, with sqrt oversampling:
- J3a, scale and bias only: rows kept as frz-os's, g_k and b_k learned (25,264 numbers). Kang et al. (ICLR 2020) call
  the scale part learnable weight scaling. The embedding and the rows are unchanged, so every embedding and maximum
  cosine score stays frz-os's exactly, and the frozen logits remain available (g = 1, b = 0) for open set.
- J3b, the whole head retrained from frz-os's rows (classifier retraining in Kang et al.).
Predictions (before any number): J3a recovers at least half of the 1.6-point gap (species F1 zero at least 0.888),
most of it under 100 images; J3b at least as much as J3a; neither changes any embedding score (by construction). If
J3a reaches within 0.3 of ctl-os, a model trained with the frozen head plus a learned head afterwards has the
frozen embedding and the learned F1, and the one-step version to explore is a second, detached head (no gradient
into the trunk or the rows) trained alongside the frozen one.

J3 as launched (owner, 2026-11-26): three 1-GPU jobs in parallel, so each starts on its own.
- J3a as above, one epoch (the owner: so few parameters should need one); J3b three epochs, since it retrains about
  16 M row parameters (Kang et al. retrain the classifier for 10 epochs from a fresh start; J3b starts from frz-os's
  rows). Validation species F1 after every epoch and the training loss with a summary of g and b through each epoch
  are logged, to see whether they plateau. The hidden layer that produces the embedding sits inside mini_trainer's
  classifier module, so it is frozen with the trunk, dropout off (mt).
- **J3c, the one-step version, trained from scratch** with frz-os's configuration for 5 epochs: the frozen head as in
  frz-os, plus a second head that shares the frozen head's rows but sees them and the embedding detached, with its own
  learned g and b. The trunk and the rows learn only from the frozen head; the second head's gradient reaches only its
  g and b. Both heads are evaluated.
- In every arm g and b are treated as ctl-os's (same weight decay and parameter group), so the comparison with ctl-os
  is of the trunk and rows only. J3a and J3b: AdamW peak 1e-3, warmup 0.05 epoch (0.5 in the testbed), cosine to 0.
- J3c's second head is gradient-clipped on its own. mini_trainer clips the joint gradient norm at 5, and a shared
  clip would let the second head's gradient shrink the trunk's update; a test shows the trunk and the main head train
  identically with and without the second head (mt).
- Submitted 2026-11-26 10:23: j3a 12480677, j3b 12480678, j3c 12480680; e-j3a/b/c, e-j3clh (J3c's second head as
  an ordinary learned head) and e-j3-os after them; about 11-12 B200-h expected, 41 at the caps. Each training job
  compares every tensor of its last.pt with frz-os's and fails if anything but the trained tensors differs.

More predictions, before any number:
6. J3a's and J3b's trunk tensors, and J3a's rows, are bit-identical to frz-os's; so are their embeddings.
7. J3a: species F1 zero at least 0.888; the gain over frz-os is largest under 100 images; one epoch is enough (the
   loss flat over the last quarter of the epoch).
8. J3b within 0.3 of J3a or above it; its rows move away from the class means (median predicted-class cosine falls
   from +0.33), while the embedding scores stay frz-os's by construction.
9. J3c's frozen head lands within 0.3 of frz-os (0.880; seed and order are the same, the extra head adds no gradient
   to the trunk); its second head within 0.3 of J3a; its embedding scores within 0.01 of frz-os's.
10. The best of J3 reaches within 0.3 of ctl-os (0.896): the frozen embedding and the learned F1 together. If it
    falls short, the remaining gap is the trunk's.

**J3a interim (2026-11-26, training job only; the test fold comes from e-j3a).** One epoch took 22 min (forward-only
trunk). All 790 frozen tensors equal frz-os's (333 normalisation buffers, the hidden layer, the rows), so prediction 6
holds for J3a. Validation species macro-F1 (mini_trainer's validation split): J3a 0.8947 against frz-os 0.8939 and
ctl-os 0.9076, so **+0.08 of the 1.37-point gap**; genus and family move by under 0.001. One epoch was enough: the
training loss is flat from the first summary (0.456 at batch 2,000, 0.453 at the end), g settles by batch 26,000
(median 1.065, 5-95 % 0.96 to 1.26). The median b keeps drifting (-0.02 to -0.42) until the rate runs out, but a
shift common to every b is invisible to the softmax, and the flat loss says the fit is done. Prediction 7 is heading
for failure: on a frozen model with prototype geometry, a per-species scale and bias are a recalibration with almost
nothing to recalibrate. The indirect argument above ("the lead is in the head's decision rule, since the learned
head beats its own prototypes by 3.1 points") was weaker than it looked: that comparison is of jointly trained rows
against class means, so it speaks for the rows, not for g and b. J3b (rows retrained) and J3c test the rest.

**Why the learned head's embedding is a cone (owner's question, 2026-11-26).** mini_trainer's embedding is
`normalize(leaky_relu(hidden(dropout(pooled trunk feature))))`: one 1,280 to 1,280 linear layer, then a leaky ReLU
(slope 0.01), then unit length. The coordinates are therefore almost all non-negative, and the embedding lives in the
positive orthant, where dense vectors are far from orthogonal (unit vectors with half-normal coordinates have mean
cosine 0.64). Measured on the 126,310 reference images:

| | learned | frozen | global scale |
|---|---|---|---|
| coordinates above 0.001 per image (of 1,280) | 867 | 84 | 281 |
| coordinates below 0 (all tiny: minimum -0.018) | 31 % | 93 % | 78 % |
| mean pairwise cosine | 0.58 | 0.02 | 0.12 |
| participation ratio of the centred covariance (effective dimensions) | 142 | 496 | 168 |

The learned head's model writes a dense code and gets the cone for free; the frozen head's model writes a sparse code
(84 active coordinates), which is how non-negative vectors become near-orthogonal. The frozen head needs that: its
only route to a confident softmax is a high cosine to the true row and a low one to the rest. The learned head does
not: only logit differences matter and per-class g with the clamp does the rest.

dev/122 `--center` (subtract the reference images' mean, renormalise) removes the cone: the learned head's centred
class means have the frozen head's structure (same genus 0.41, same family 0.04, other family -0.004). But no accuracy
moves (known species by class mean 0.882 against 0.883 uncentred; genus of new species 0.512 against 0.513). The cone
is a shared offset, harmless to any ranking. What separates the spaces is the number of directions used, about 140
against about 500, and that is what costs the learned head's embedding 0.5 to 3.3 points. *(Wrong, tested the same
day: within-class whitening gives both models about 1,000 effective dimensions and the gap stays; see "Did the leaky
ReLU handicap the embedding?" below.)* Calibration, for the same
owner message: the debiased L2 error is computed over 15 equal-count bins of top-1 confidence (mt's
`j2_trap_cal.py`); on the test fold the learned head is underconfident (mean confidence 0.895, accuracy 0.931), the
frozen head too (0.901 against 0.922), the global scale slightly overconfident (0.934 against 0.923).

**J3b interim (2026-11-26, validation only; the test fold comes from e-j3b).** Validation species macro-F1 after each
epoch: 0.8698, 0.8858, **0.8984** (frz-os 0.8939, J3a 0.8947, ctl-os 0.9076); genus 0.9531, family 0.9759 (frz-os
0.9518 / 0.9753, ctl-os 0.9592 / 0.9795). The peak rate (1e-3) first knocked the converged rows off (-2.4 after one
epoch), and the cosine decay brought them back and past; the last epoch alone added 1.3 points. Retrained rows on the
frozen trunk recover about a third of the 1.37-point gap (+0.45), the scale and bias about nothing (+0.08). So
prediction 8 holds (J3b above J3a), and prediction 10 (within 0.3 of ctl-os) fails on validation by about 0.9. The
curve does not say J3b has converged: it is still rising when the rate runs out, so a lower peak rate or more epochs
may recover more. All 789 frozen tensors equal frz-os's (prediction 6 holds for J3b). With the rows free, g grows as in
the learned head: median 1.39, 1.66, 1.75 at the end of epochs 1 to 3 (95th percentile 2.08), against 1.065 in J3a;
the training loss is flat after the first epoch (0.39 to 0.43), so the validation gain of the last epoch is the
annealing of the rate, not more fitting. Whether the rows turned away from the images (as ctl-os's did) comes with
e-j3-os.

**J3c cancelled (owner, 2026-11-26; terminated 11:37 in epoch 1, before any checkpoint).** Its second head learns only g and b, the part J3a
showed recovers nothing; the part that helped in J3b, the rows, is not in its design. Prediction 9 is left untested.

## J3 result (scored 2026-11-26)

e-j3a 12480766, e-j3b 12480909 and e-j3-os 12480910, scored by mt as J2 was (mini_trainer
`.agents/local/campaign/plans/j3/README.md`, "J3 result"). J3a's and J3b's embeddings equal frz-os's bitwise (303,886
x 1280, maximum difference 0), so every embedding score (1-NN 0.902, the class means, dev/122) is frz-os's.

| test fold | ctl-os (learned) | frz-os (frozen) | J3a (g, b on frz-os) | J3b (rows, g, b on frz-os) |
|---|---|---|---|---|
| species F1, zero / calibrated | **0.8961 / 0.8998** | 0.8798 / 0.8850 | 0.8801 / 0.8855 | 0.8843 / 0.8894 |
| genus / family F1, zero | **0.9559** / 0.9678 | 0.9483 / **0.9746** | 0.9478 / 0.9745 | 0.9490 / 0.9741 |
| species top-1 | **0.9314** | 0.9223 | 0.9229 | 0.9248 |
| species F1 under 100 images / 1,000 or more | **0.885 / 0.942** | 0.864 / 0.935 | 0.865 / 0.936 | 0.872 / 0.938 |
| calibration error (debiased L2): test fold / trap vocabulary / trap list | 0.057 / **0.062 / 0.034** | 0.037 / 0.089 / 0.082 | 0.030 / 0.065 / 0.070 | **0.010** / 0.077 / 0.088 |
| trap probe, calibrated: tta3bm / held out | **0.630** / 0.472 | 0.608 / 0.468 | 0.603 / **0.472** | 0.601 / 0.451 |
| open set, best logit score: known_bal / known_unif | 0.913 / 0.925 | **0.924 / 0.945** | 0.920 / 0.937 | 0.916 / 0.928 |
| open set, maximum cosine to a row (known_bal) | 0.542 | **0.812** | **0.812** | 0.712 |
| median cosine to the predicted class's row | -0.58 | +0.33 | +0.33 | +0.155 |

**Predictions scored.**
6. Trunk, hidden layer and (J3a) rows bit-identical to frz-os's, embeddings too: **right**.
7. J3a at least 0.888: **wrong** (0.880, +0.03 of 1.63); no frequency bin gains more than 0.1. One epoch enough:
   right (the loss was flat from batch 2,000).
8. J3b within 0.3 of J3a or above it: **right** (+0.42); its rows move away from the class means (+0.33 to +0.155):
   **right**; embedding scores unchanged: right, by construction.
9. Not tested (J3c cancelled).
10. The best of J3 within 0.3 of ctl-os: **wrong** (J3b -1.18 zero, -1.04 calibrated).

**What it changes.** Of the 1.63-point gap between the learned and frozen heads, a learned scale and bias on the
frozen model recover 2 %, and a fully retrained head 28 % (three epochs; still rising on validation when the rate ran
out, so a longer run would take a little more). The rest, about 70 %, is in the representation the learned head
trained: the backbone and the 1,280 to 1,280 hidden layer, both frozen here. The J2 conclusion that "the learned
head's F1 lead lies in its head" is therefore wrong. It rested on the learned head beating its own class means by 3.1
points, but that compares jointly trained rows against class means, and says nothing about whether those rows could be
learned on another trunk. The two-step model does not give the frozen embedding with the learned F1. As J3b's rows
move toward the learned head's optimum they also move toward anti-prototypes (+0.33 to +0.155), and maximum cosine
falls (0.812 to 0.712), as does the best logit score (0.924 to 0.916). So accuracy and prototype geometry trade
against each other inside the trunk, as J1 and J2 found across heads. The two-step model's one clear gain is
calibration on the test fold (J3b 0.010, the best of all four), but it is worse than the learned head on the traps.
Getting both needs a training-time change that alters the trunk itself (the owner's "explore the head geometries to
avoid this two step process"); the first candidate examined is the leaky ReLU in front of the normalisation, below.

## Did the leaky ReLU handicap the embedding? (owner's question, 2026-11-26)

The owner asked whether the leaky ReLU in front of the normalisation had "sandbagged" the models. Notation: f is the
pooled trunk feature (EfficientNetV2-S averages SiLU outputs; *corrected 2026-11-26: "so f is itself almost
non-negative" was assumed, not measured, and is wrong. SiLU goes down to -0.28, and J4f measured 59-76 % of f's
coordinates negative, carrying 46-53 % of its squared length; see the J4 entry*), h = W f + c the 1,280 to 1,280 hidden layer, a = leaky_relu(h) with slope 0.01, x = a / |a| the
embedding. A coordinate is "on" when x_i > 0.001 (x_i > 0 gives 883, 85 and 287 instead of 867, 84 and 281). Effective
dimensions are the participation ratio (sum l)^2 / sum l^2 of the eigenvalues l of the centred covariance: d when
the variance is spread evenly over d axes, 1 when it lies along one. The mean cosine of an image to the mean direction is |mean x| (0.76 learned, 0.14 frozen,
0.35 global scale), so the learned head's images sit about 40 degrees from one axis, the frozen head's about 82.

dev/122 `--transform` asks two questions of the saved embeddings, each fitted on the reference images only:
- `preact` undoes the ReLU. It is positively homogeneous and invertible, so dividing the negative coordinates by 0.01
  gives h / |a|, whose direction is h's: the signed output the hidden layer computed. (22 % of the learned head's
  negatives are fp16 subnormals and lose precision; they are its smallest coordinates.)
- `whiten` is within-class whitening, x -> Sw^(-1/2) (x - mean), Sw the covariance of the reference images around their own
  species' mean: nearest class mean becomes LDA's rule. It asks whether one model's lead is geometry a linear map can
  undo, or a difference in what the representation encodes.

| macro accuracy | known species, class mean | known species, 1-NN | new species: genus | new species: family | new genera: family |
|---|---|---|---|---|---|
| learned, as saved | 0.8825 | 0.8524 | 0.5132 | 0.8576 | 0.6391 |
| learned, ReLU undone | 0.8794 | 0.8513 | 0.5056 | 0.8472 | 0.6252 |
| learned, whitened | 0.8801 | 0.8488 | 0.5143 | 0.8546 | 0.6395 |
| frozen, as saved | 0.8876 | 0.8650 | 0.5260 | **0.8777** | 0.6718 |
| frozen, ReLU undone | 0.8799 | 0.8611 | 0.5169 | 0.8750 | 0.6695 |
| frozen, whitened | **0.8895** | **0.8671** | **0.5279** | 0.8761 | 0.6703 |
| global scale, as saved | 0.8402 | 0.8043 | 0.4785 | 0.8427 | 0.6271 |
| global scale, ReLU undone | 0.8430 | 0.8069 | 0.4797 | 0.8316 | 0.6050 |
| global scale, whitened | 0.8493 | 0.8077 | 0.4937 | 0.8451 | 0.6231 |

Both together (`preact_whiten`) are mixed: new genera +0.7 learned, +1.7 frozen (0.6888 [+1.1, +2.4]), known species
-1.4 and -0.7. Output `embedding_quality_none_preact_whiten_preact_whiten.json`, 200 paired bootstrap draws.

1. **Undoing the ReLU makes the trained models worse** (frozen, class mean -0.8 [-1.0, -0.6]; learned, new genera -1.4
   [-1.8, -1.0]); only the global scale gains, +0.3 on known species, and it loses 2.2 on new genera. The squashed half
   holds no better information. In the frozen model it holds 98.6 % of the pre-activation's energy: the hidden layer
   drives 93 % of the units well below zero and lets about 84 through, so the ReLU is the gate of its sparse code.
   Before the gate the frozen model's images form a cone too (mean pairwise cosine 0.72).
2. **No linear map closes the gap.** Whitening spreads both models over about 1,000 effective dimensions (learned
   1,018, frozen 930) and moves nothing by more than 1.5 points; the frozen model's lead stays (class mean +0.95,
   new species genus +1.35, family +2.15, new genera +3.07, all intervals clear of 0). The sentence above that the
   number of directions used "is what costs the learned head's embedding" was wrong: equalise it and the cost stays.
   The spread is a symptom. The difference is in which images the representation puts together, set during training:
   the same place J3 found the learned head's F1 lead.
3. **No coordinate is dead.** In every model each of the 1,280 coordinates is on for at least 1 % of reference images
   (median 66 % learned, 6 % frozen, 22 % global scale): the frozen code is a sparse distributed code, not a small
   subspace.

**Theory and literature.** With 12,632 species in 1,280 dimensions, the best class-mean arrangement for non-negative
features is an orthogonal frame (cosine 0) and for signed ones a simplex (cosine -1/(C-1), here -0.00008); the two
differ only by subtracting the global mean (Kini et al., ICLR 2024; arXiv 2510.21078). The frozen model's class means
are such a frame (other families 0.02), and centring, which turns one into the other, moved nothing. So the orthant
does not shrink the room for separating classes; it decides how the code is written (sparse, or dense plus a shared
offset). Face recognition removes the final ReLU (SphereFace's appendix: it confines features to the non-negative
range; ArcFace ends BN-Dropout-FC-BN), while Kini et al. add one on purpose, because under class imbalance it keeps the
class means symmetric. The literature records it as a design choice with a known geometric effect, not a defect.

**Verdict.** No sign of a handicap in the trained models. Whether a model trained without the ReLU would learn a better
representation, only a training run can say. Prediction for one (learned head, J2's testbed): species F1 within 0.3 of
ctl-os, embedding measures within 1 point, so it is not the lever. The evidence points elsewhere:
- Where the embedding is read. It is the hidden layer's output, the one nearest the head. Representations before a
  projection head transfer better: SimCLR; Wang et al. (CVPR 2022), with an MLP projector in supervised training;
  Kornblith et al. (NeurIPS 2021), where losses that raise accuracy make the penultimate layer transfer worse, and the
  differences appear only in the last layers. Re-embedding the three J2 models at the pooled trunk feature f costs about
  1 B200-h. If the learned head's f is as good as the frozen head's, one model gives both: its head for species,
  f for prototypes and open set.
- The head's scale (temperature), which J1 to J3 keep pointing at: fixed S (frozen) gives the best embedding, 8.9 S
  (global scale) the worst.
