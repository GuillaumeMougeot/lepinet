# O1: the two models the paper recommends are tied on macro-F1 and differ by 17 points in deployment

**Kind:** research · **Status:** **RESOLVED for abstention (2026-08-28); open-set rules running.**
B8 and P5 tie on probe macro-F1 (0.7798 vs 0.7810, inside any floor). Under a 95 %-precision
back-off policy on the same fold they are not close: **B8 answers 73.3 % of images and P5 answers
92.8 %**, for a useful-answer rate of **71.17 % against 88.44 % -- a 17.3 point gap** between models
the headline metric calls equivalent.

This is the gap the paper's own Limitations section flagged: every abstention number in section 4.6
comes from the *original single-head model* on the *in-distribution* fold, and neither recommended
model had a curve.

## Result: rank abstention on the probe fold, 95 % precision target

| | B8 (198 M, ours) | P5 (BioCLIP-2 fine-tuned) |
|---|---|---|
| species: coverage / precision | 69.72 % / 0.9737 | 71.48 % / 0.9530 |
| genus | 1.43 % / **0.8073** | **0.00 %** |
| family | 2.14 % / 0.9939 | 21.31 % / 0.9537 |
| **abstain** | **26.70 %** | **7.21 %** |
| answered | 73.30 % | **92.79 %** |
| precision among answered | **97.10 %** | 95.31 % |
| **useful (answered AND correct)** | **71.17 %** | **88.44 %** |

For comparison, section 4.6's in-distribution figure on the old single-head model: 99.18 % answered
at 95.04 % precision, with 0.82 % abstention.

## Three things worth separating

**1. Domain shift is what makes abstention expensive, and the paper understates it by 30x.** In
distribution, reaching 95 % precision costs **0.82 %** abstention. On shifted data the same target
costs **26.70 %** for B8. Section 4.6's coverage numbers are not wrong, but they describe the easy
case, and the paper presents abstention as a deployment mechanism. The honest headline is that under
source shift a quarter of images may get no answer at all at a 95 % bar.

**2. The back-off ladder partially breaks under shift, and it breaks exactly where 4.6 predicted.**
Section 4.6's lesson was that coarse thresholds must be calibrated *conditionally on reaching that
rank*, because the coarse posterior inherits the uncertainty that triggered the back-off. Under shift
that effect is sharper: on the 4,603 images where B8's species confidence is below threshold, genus
precision is **0.5570** against 0.8511 overall, and **never reaches 95 % at any threshold** (max
0.8073). So the genus rung is not usable for B8 at this target -- the policy above hands it 1.43 %
of images at 0.8073, below the promise.

P5 skips the rung entirely: **genus coverage is 0.00 %** and family absorbs 21.31 %. A model whose
species confidence is well-calibrated falls straight past genus to a rank where the evidence is
actually sufficient. That is the ladder working as designed rather than failing.

**3. Tied accuracy, different systems.** B8 and P5 are indistinguishable on probe macro-F1 and differ
by 17.3 points on useful-answer rate. The difference is **not** discriminative power, it is
**confidence calibration**: B8 buys its higher precision-among-answered (97.10 % vs 95.31 %) by
refusing to answer, which is what a model does when its confidence does not separate its right
answers from its wrong ones cleanly enough to sit near the threshold.

This is the strongest instance yet of the project's recurring theme that **the selection metric and
the deployment metric are different metrics**. Section 4.10 already showed the three axes disagree
mildly; here two models agree exactly on the headline axis and disagree enormously on the one a user
experiences. Selecting on probe macro-F1 would have called this a coin flip.

## What it changes

**For the recommendation.** P5 is the model to ship, and the reason is not accuracy. Until now the
argument for P5 over B8 was a tie plus "theirs is a better representation"; now there is a concrete
deployment margin. **A user of P5 gets a usable answer on 93 % of photographs; a user of B8 gets one
on 73 %.**

**For the paper.** Section 4.6 needs a shifted column, and the Limitations entry saying the
recommended models have no abstention curve can be removed. The framing changes too: abstention is
currently presented as a cheap safety mechanism, and on shifted data it is a **coverage cost of 27 %**
for one of our two best models. That is a finding, not a footnote.

**A caveat to carry.** B8's species threshold lands at **t = 1.00**, i.e. against the top of the
confidence range. That is the saturation the clamp analysis warned about, and it means B8's coverage
figure is a floor set by threshold resolution rather than a smooth operating point. Worth a look
before the number is quoted, though it does not change the direction: even at t = 1.00 B8 covers only
69.72 %.

## Open-set: both models beat everything in the paper, and entropy wins twice

632,913 images, 3,171 novel-species images, five rules from one forward pass:

| rule | B8 | P5 |
|---|---|---|
| **entropy** | **0.9153** | **0.9161** |
| margin | 0.9108 | 0.8890 |
| max-logit | 0.8889 | 0.9160 |
| msp | *degenerate* | 0.9087 |
| energy | *degenerate* | 0.9156 |
| **best** | **0.9153** | **0.9161** |

**Three results here, and the first contradicts the paper.**

**1. Section 4.10's claim that the smallest model leads open-set detection is wrong.** Its table has
`efficientnet_v2_s` best at 0.9068, and both recommended models beat it: **B8 0.9153, P5 0.9161.**
The "three axes disagree" story weakens accordingly -- P5 now leads or ties on *all three*
(in-distribution 0.9113, probe 0.7810, open-set 0.9161), and the mild disagreement 4.10 reports does
not survive scoring the models we actually recommend. That section needs rewriting, not patching.

**2. Entropy wins for both, which is a fourth distinct answer and finally a pattern.** Section 4.9
recorded max-logit best at 20 M and MSP best at 198 M and concluded rules do not transfer across
capacity. With two more models the finer statement is available: **the best rule tracks the head's
output convention and the representation's calibration, not the parameter count.** Both of these
models are well-calibrated and shape-based rules win on both. The recommendation in 4.9 -- name the
rule, re-select it per model -- is unaffected and if anything better supported.

**3. Rule-insensitivity is not a property of the angular margin.** Section 4.3 attributes it to the
margin: the ArcFace head's five rules span 1.2 pt against the plain head's 28.4. **P5 has no margin**
-- it is the plain `independent` cosine head -- and its three best rules sit within **0.0005** of each
other, with a full span of 2.71 pt. So a strong, well-calibrated representation delivers the same
insensitivity the margin was credited with. The margin's remaining distinct contribution is smaller
than 4.3 implies.

**B8's `energy` and `msp` are correctly flagged as meaningless**, not measured as bad: its
`marginal_arcface` head emits log-probabilities, so `energy = logsumexp(log p) = 0` for every image
and `msp` is a monotone function of `max` (identical to 4 decimals, as predicted). `dev/061` detects
this from the data. Section 4.9's DINOv3 row is *not* affected -- its max and msp differ (0.8298 vs
0.8904), so that head was not emitting log-probabilities.

## Open

- **Restriction plus abstention.** D1 showed a checklist concentrates false positives onto scored
  classes on the probe fold; a restricted head that may also abstain should send that mass to
  abstention instead. With P5 abstaining on only 7.21 % there is room to spend.
- **Open-set under shift** for both models. Everything above is the no-domain-shift benchmark;
  section 4.5 says novelty detection is not domain-robust, and neither recommended model has that
  number either.

## Method note

`dev/061` could not load P5 at all until this session: `model_arch_name: bioclip2` resolves only
inside `dev/075`, so the backbone was invisible the same way `dev/050`'s heads are. `--bioclip2` now
installs that seam. **That single missing import is why the project's best model had no open-set
number** -- not cost, not difficulty, just an unregistered extension point. Worth remembering as the
same failure mode as the head registry, which has now cost five scripts.
