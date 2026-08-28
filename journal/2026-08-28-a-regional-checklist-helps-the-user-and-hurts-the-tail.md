# D1: a regional checklist helps the average observation and hurts the rare species

**Kind:** research · **Status:** **RESOLVED (2026-08-28), both folds. Two of three predictions
falsified, and the result is better than the one predicted.** Restricting B8's label space to the
464-species Danish trap checklist is worth **+5.06 pt micro-accuracy on probe** and **+6.31 pt
macro-F1 / +4.39 micro on held-out species**, while *costing* 1.70 pt of macro-F1 on probe. The sign
of the macro-F1 effect is set by **what fraction of the allowed labels the average is taken over**,
not by the model -- and a real deployment sits in the regime where it wins on both.

## Result

B8, probe fold, 15,200 images / 368 species. The eval set is byte-identical across arms -- only the
set of labels the model may emit changes, and the script asserts the scored frame does not move.

| labels allowed | macro-F1 | micro-accuracy |
|---|---|---|
| **464** (the actual checklist) | 0.7628 | **0.8901** |
| 1,000 | 0.7782 | 0.8855 |
| 2,000 | 0.7883 | 0.8736 |
| **4,000** | **0.7908** | 0.8655 |
| 12,041 (unrestricted) | 0.7798 | 0.8395 |

**Micro-accuracy is monotone decreasing in checklist size.** The tighter the list, the better --
exactly the owner's intuition, and worth **+5.06 pt** at the real checklist size. For a Danish user
that is 89.0 % of observations correct instead of 84.0 %, for free, with no retraining.

**Macro-F1 does the opposite, and is not even monotone.** It peaks at 4,000 labels and the true
checklist is the *worst* arm, 1.70 pt below doing nothing at all.

## And on held-out species it helps *both* metrics, by a lot

The probe-held-out-species fold -- 2,455 images, **58** species the adaptation never saw:

| labels allowed | macro-F1 | micro-accuracy |
|---|---|---|
| 464 (the checklist) | **0.8447** (+6.31) | **0.7637** (+4.39) |
| 1,000 | **0.8462** | 0.7589 |
| 2,000 | 0.8418 | 0.7564 |
| 4,000 | 0.8283 | 0.7483 |
| 12,041 (unrestricted) | 0.7816 | 0.7198 |

**I predicted this fold would pay for the restriction. It gains the most of anything measured**:
+6.31 pt of macro-F1 on exactly the taxa the paper calls hardest and most deployment-relevant.
Falsified, and in the opposite direction.

## Why they disagree, which is the actual finding

Restriction redistributes the probability mass that used to land on out-of-checklist species. Where
it lands decides which metric benefits.

Take an image of species X, which is in the checklist. Unrestricted, the model's second choice might
be a foreign congener Y that does not occur in Denmark.

- **Unrestricted:** the error goes to Y. X loses recall; Y gains a false positive -- but **Y is not
  in the evaluated species set, so Y's precision is never scored.** The error costs one class's
  recall and nothing else.
- **Restricted:** Y is masked out, so the mass moves to the nearest *checklist* species Z. X still
  loses recall, and now **Z's precision drops too, because Z is scored.**

So restriction **concentrates false positives onto the classes being averaged over.** Errors that
used to leak harmlessly out of the evaluated label set now land inside it and are counted twice.

For the *typical* image the redistribution mostly lands on the right answer, because the true species
is in the checklist and was often the runner-up -- hence micro-accuracy rises sharply. For *rare*
species the redistributed mass lands on common confusables instead, and since macro-F1 weights every
species equally, that tail damage dominates the average.

**The held-out fold then supplies the general rule, and it is a ratio.** The pollution only costs you
when it lands on a class you are scoring. So what matters is **how much of the checklist is inside
the evaluated species set**:

| fold | scored species | checklist | scored / allowed | macro-F1 effect |
|---|---|---|---|---|
| probe | 368 | 464 | **79 %** | **-1.70** |
| probe held-out species | 58 | 464 | **12 %** | **+6.31** |

On the probe fold the checklist and the scored set are nearly the same 464 species, so almost every
redirected false positive lands on something being averaged, and the pollution outweighs the recall
recovered. On the held-out fold only 58 of the 464 allowed labels are scored, so ~88 % of the
redirected errors land on classes nobody is measuring -- while the recall gain lands entirely on the
58 that are. Same intervention, same model, opposite sign, and the sign is set by the benchmark's
construction rather than by the model.

**So the honest general statement is:**

> Restricting the label space always improves how often the model is right (micro-accuracy rises on
> both folds). Its effect on a **per-class average** depends on what fraction of the allowed labels
> that average is taken over -- it is a large gain when the scored set is a small part of the
> checklist, and a small loss when the two nearly coincide.

**A real deployment looks like the held-out fold, not the probe fold.** A Danish user's checklist is
~950 published macro-moths and any given night's trap yields a few dozen species, so the scored set
is a small fraction of the allowed set -- the regime where restriction wins on both metrics. The
probe fold's −1.70 is the pessimistic corner, and it exists because the probe was *built* to cover
most of the trap's species.

This is the fourth instance of the macro-F1 subtlety of finding 4a, and the sharpest: *what a
per-class average scores depends on which classes are in the denominator, and an intervention that
changes where errors land changes the average without changing how often the model is right.*

## Scoring the prediction: mis-specified, and I should say so plainly

I predicted "checklist (486) on probe: 0.72-0.78, against the unrestricted 0.6270". The number landed
at **0.7628, inside the range** -- and that is a coincidence, not a hit.

**The prediction was written against the wrong baseline.** 0.6270 is the *baseline model's* probe
score; D1 ran on **B8**, whose unrestricted probe is 0.7798. So I predicted a gain of roughly +13 pt
and the truth is a loss of 1.70. Recording it as "inside the range" would be scoring the digits and
ignoring the claim. **The prediction is falsified.**

The two secondary predictions fare better:

- *"the curve is strongly concave; padding to 4,000 keeps under half the gain"* -- **falsified for
  macro-F1**, which has its maximum at 4,000, and **correct in shape for micro-accuracy**, which
  decays gently (89.0 -> 86.6) rather than collapsing.
- *"held-out species should pay for it"* -- **falsified, and it is the largest gain measured**:
  +6.31 macro-F1 and +4.39 micro. I had the mechanism backwards. I reasoned that held-out taxa are
  the most likely to fall outside a checklist, but they are *in* this checklist by construction; what
  they actually are is the taxa with the least reliable evidence, so they benefit most from having
  12,041 wrong answers removed.

I explicitly declined to predict whether restriction helps more or less on stronger models. That
hedge was right for the wrong reason: the interesting variable turned out not to be model strength
but **which metric you read**.

## What this changes

**For the paper.** This belongs in the deployment section, and its headline is the disagreement, not
the gain. The honest summary is: *a regional checklist is worth +5 points of top-1 accuracy to a
real user and costs 1.7 points of macro-F1, because it concentrates the remaining errors onto the
species you are averaging over.* Reporting only one of those numbers would be a misrepresentation in
either direction.

**For the recommendation.** If the deployment goal is "the user gets the right answer", restrict
hard. If it is "rare species are not systematically absorbed into common ones" -- which is the whole
argument for a macro metric in biodiversity monitoring -- restrict loosely or not at all. **The right
answer depends on whether the tool is for identification or for monitoring**, and those are different
products.

**Open.** Whether abstention closes the probe fold's small loss. A restricted head that may also say
"not in my list" should recover the pollution, because the mass currently forced onto checklist
confusables would go to abstention instead. That composes directly with the rank abstention of paper
section 4.6, and it is the natural next experiment.

**Also open, and cheap:** the same sweep on P5, and a checklist built from a *published* Danish
species list rather than from the trap corpus. The latter matters because a published list is
over-inclusive relative to what one trap sees, which is precisely the regime where this result says
restriction is most valuable.

## Method notes worth keeping

**The harness validated itself.** The unrestricted arm reproduced B8's published probe of **0.7798**
exactly, before any restricted arm ran, which is what makes the deltas trustworthy.

**Three failures before it ran**, all mine, and all the same species of error -- a default that was
never checked against reality: the `dev/050` head registry (fourth script to hit it, now handled
inside the script), a vocabulary loaded lazily inside a branch the baseline arm did not take, and
`--level` defaulting to `species` when this checkpoint calls it `speciesKey`. The last two each cost
a full job to discover. The lesson is the cheap one: **resolve names from the artefact, do not
default them.**

**6 of the 470 regional species are not in the model's vocabulary** and are silently dropped from
the checklist -- it could not predict them either way. Worth remembering when quoting "464".
