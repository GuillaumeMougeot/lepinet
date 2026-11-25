# Should the cosine head's per-class scale and bias be learned or frozen?

**Kind:** research · **Status:** OPEN (2026-11-25): J1 scored (seed 42). Frozen loses 2.7 points of test-fold species F1 (6.5 in species under 100 images); bias only recovers nothing. The learned head wrecks the embedding for open set (maximum cosine to a head row 0.54 against 0.78, 1-NN -3.4; class means of training images 0.86 against 0.90, the best score on the frozen head), while the best logit score is equal (about 0.90). The per-species scale alone causes both effects; the bias is inert. Owner's decision (2026-11-25): J2 first, three sqrt-oversampled arms (learned, frozen, one learned global scale); the bundle waits for it

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
