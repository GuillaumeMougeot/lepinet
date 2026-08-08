# Round 2 of self-training loses 3.8 points, because a better labeller keeps a narrower set

**Kind:** research · **Status:** **RESOLVED (2026-08-08). Badly falsified, and the mechanism is not
the one everybody warns about.** Re-labelling with F2 (probe 0.7541) instead of B4 (0.7101) produced
**more accurate labels covering less than half the species**, and cost **3.80 points**. The failure is
not confirmation bias — it is that a fixed confidence *quantile* selects a narrower set as the model
improves.

## Result

| | round 1 (B4 labels) | round 2 (F2 labels) |
|---|---|---|
| images kept | 12,230 | 8,169 (−33 %) |
| **species covered** | **346** | **156 (−55 %)** |
| pseudo-label accuracy | 98.15 % | **99.84 %** (+1.69) |
| **probe** | **0.7541** | **0.7161 (−3.80)** |

**Predicted probe 0.758–0.772, falsified below 0.7582.** Landed at 0.7161 — falsified by 4.2 points,
the largest prediction miss of the project. The reasoning was that a 4.4-point-better labeller would
produce cleaner labels and eat into the gap to end-to-end training. **The labels did get cleaner. It
made things worse.**

## The mechanism

The gate keeps the top 30 % by confidence — a *quantile*, chosen because absolute softmax values are
not comparable across models ([[2026-08-03-b3-self-training]]).

Round 1's labeller was **saturated**: its confidence distribution had a median of 0.9999996 and ties
at exactly 1.0, so the 0.70 quantile landed *on* 1.0 and every tied image was kept — 44.9 % rather
than 30 %, spanning 346 species.

Round 2's labeller is better, and therefore **better separated**. Its cut falls at 0.99896, no ties,
exactly 30 % kept — and the top 30 % of a good model is a *narrower, easier, more homogeneous* slice
of the domain. It concentrated on 156 species.

So the improvement in the labeller **caused** the loss of coverage. The gate's behaviour depends on
the shape of the confidence distribution, and improving the model changes that shape in the direction
that shrinks the selection.

**This is not the failure mode the design was guarding against.** The stated risk was circularity —
the model reinforcing its own errors. Label accuracy went *up*, so that risk did not materialise. The
one that did was invisible in the design: **selection collapse.**

## Coverage beats label accuracy, and by a lot

+1.69 points of label accuracy did not pay for −55 % of species. Put beside the replication sweep
([[2026-08-04-replication-sweep]]), where 0.39 % of training bought 97 % of the gain, the two results
say the same thing from different directions:

> **What unlabelled target data supplies is *coverage of the input space*, not label information or
> gradient volume.** Anything that trades coverage for label quality is trading the wrong way.

That also explains why round 1 worked as well as it did *by accident*: its labeller was
overconfident, the gate saturated, and the saturation kept a broader set than the design intended.
A better-calibrated round-1 labeller would have scored worse.

## The fix, and the prediction it makes

The gate should target **coverage**, not a confidence quantile. Two candidates, both cheap:

1. **Keep a fixed number per species** — say the top *k* images of every species the model predicts,
   so coverage is guaranteed by construction and confidence only orders within a class.
2. **Raise `keep_frac` for later rounds** to hold the species count constant.

**Prediction (committed):** round 2 with per-species selection at matched coverage (346 species)
lands at probe **0.755–0.775**, i.e. at least matching round 1 and plausibly beating it, because the
labels really are 1.69 points cleaner. If it still loses, iteration genuinely does not work here and
the cause is circularity after all.

## What it changes now

**The recommended recipe stays at one round.** Round 1 at 2 % of training, and no iteration until
the gate is fixed. That is a real limitation of the headline result and belongs in the paper as one:
the method's largest lever works once, and the obvious way to compound it fails for a reason that
took a diagnostic to find.

**And it is a warning worth generalising.** A confidence-quantile gate is standard in self-training,
and its interaction with the labeller's calibration is the kind of thing that silently degrades an
iterated pipeline while every intermediate metric — label accuracy most of all — looks better.
