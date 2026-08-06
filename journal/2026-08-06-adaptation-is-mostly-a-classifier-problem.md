# T2: 2 epochs on a frozen trunk captures 83 % of self-training's gain

**Kind:** research · **Status:** **RESOLVED (2026-08-06). Prediction wrong, and it corrects a
mechanism I asserted three days ago.** Freezing the representation and adapting only the classifier
recovers **83 % of the probe gain and 89 % of the held-out-species gain** from full self-training, in
2 epochs. So domain adaptation here is mostly a **re-fitting** problem, not a representation-learning
one — which is both a better deployment story and a correction to [[2026-08-03-b3-self-training]].

## Result

| | probe | held-out species |
|---|---|---|
| B1 — no target data | 0.6912 | 0.6974 |
| **T2 — classifier only, trunk frozen, 2 epochs** | **0.7572** | **0.7621** |
| B3rep5x — full training, 5 epochs | 0.7706 | 0.7704 |
| **T2's share of the full gain** | **83 %** | **89 %** |

Floors: probe 0.0041, held-out 0.0052. T2 − B1 is 16× and 12× its floor; B3rep5x − T2 is 3.3× and
1.6×, so full training is genuinely better, but by a small margin over a much cheaper procedure.

**Committed prediction: 0.720–0.745 ("about half the gain"), falsified above 0.760 or below 0.6981.**
Landed at **0.7572** — above the range, and 0.3 pt short of the line I set for "the pixels story is
wrong". So the prediction is wrong without quite tripping my own falsification criterion, which is an
uncomfortable place to be and worth stating plainly rather than rounding either way.

## The correction

[[2026-08-03-b3-self-training]] explained self-training's +7.94 pt like this:

> **Target-domain pixels.** The model had never seen a trap image during training — only GBIF
> specimens plus three hand-authored corruptions meant to imitate one. Real trap frames carry the
> true nuisance distribution: actual sensor noise, actual backgrounds, actual poses.

That reads as a claim about the *representation* learning something new from real pixels. If it were
the main mechanism, a frozen representation could not capture 83 % of the gain. Most of what
self-training buys is therefore the **classifier re-fitting to a shifted feature distribution** —
the features were already adequate, and the decision boundaries were in the wrong place.

**The honest scope of that correction.** T2 starts from B1's representation, which was trained *with*
`domain_aug: trap` — three hand-authored corruptions imitating a trap frame. So the trunk is not
naive about the target domain; it has an approximation of it. The correct statement is:

> Given a representation trained with domain-mimicking augmentation, **classifier-only adaptation
> recovers 83–89 % of what full self-training gives.** Whether a trunk with no domain augmentation at
> all would behave the same way is untested, and is the obvious control.

That control is cheap and it matters, because it separates "features were already good enough" from
"hand-authored augmentation had already done the representational work, and self-training only
re-fits". The two have different implications for a new deployment domain.

## Why this is the better result of the two

**It is a deployment story rather than a research one.** Adapting to a new trap now means: collect
unlabelled images, pseudo-label them with the shipped model, retrain the classifier for 2 epochs on a
frozen trunk. That is minutes of GPU rather than hours, needs no labels, and leaves the backbone —
the expensive, validated part — untouched. It also composes with the earlier finding that the
prototype matrix can be replaced by centroids at inference: both say the classifier is the cheap,
swappable end of this model.

**And it echoes L4 exactly.** cRT showed that oversampling's *damage* lives in the representation and
that rebalancing the classifier alone avoids it. T2 shows that domain adaptation's *benefit* is
mostly available at the classifier. Two different questions, same structural answer: **on this
problem, the representation is more robust and more inert than the classifier**, and interventions
that act on the classifier are both cheaper and safer than those that act on the data the backbone
sees.

## Next

1. **The control named above**: classifier-only adaptation from a trunk trained *without*
   `domain_aug`. Separates "features were already good" from "augmentation did the representational
   work". One 2-epoch run.
2. **Iterate on the frozen trunk.** If adaptation is classifier re-fitting, a second pseudo-labelling
   round using T2 rather than B4 should be nearly free and might close the remaining 17 %.
3. **The 198 M confirmation, once**, when the recipe stops moving — per the scale discipline, not now.
