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

## Open

- **Open-set AUROC for both models under all five rules** is running (`dev/061`), and is the other
  half of this gap. Section 4.9 says rules do not transfer across capacity; P5 is a 303 M ViT-L and
  B8 a 198 M ConvNeXt, so neither inherits the other's best rule and neither inherits A1's.
- **Restriction plus abstention.** D1 showed a checklist concentrates false positives onto scored
  classes on the probe fold; a restricted head that may also abstain should send that mass to
  abstention instead. With P5 abstaining on only 7.21 % there is room to spend.

## Method note

`dev/061` could not load P5 at all until this session: `model_arch_name: bioclip2` resolves only
inside `dev/075`, so the backbone was invisible the same way `dev/050`'s heads are. `--bioclip2` now
installs that seam. **That single missing import is why the project's best model had no open-set
number** -- not cost, not difficulty, just an unregistered extension point. Worth remembering as the
same failure mode as the head registry, which has now cost five scripts.
