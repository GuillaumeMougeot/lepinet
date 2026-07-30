# Do single-head marginalisation and ArcFace × z-score compose? (A1)

**Kind:** research · **Status:** OPEN — in-distribution result landed 2026-07-30 and **falsified the
prediction**; the shifted and open-set numbers that decide whether A1 is still the right architecture
are queued (`lepi-A1-shift`, `lepi-A1-ood`).

## Why this run exists

Every ArcFace result in this project was measured on the **old multi-head** baseline, and every
single-head result was measured **without** ArcFace. The recommended architecture — one species head,
coarse levels by marginalisation, ArcFace × z-score geometry — had therefore **never been trained**,
and every claim about it was an inference across two separate experiments. A1 is that training run.

**Committed prediction** ([`PLAN.md`](PLAN.md)): *"A1 ≈ 0.906–0.915 F1 with AUROC ≥ 0.90 — the two
effects are mechanistically independent (one changes which heads exist, the other the logit
geometry), so they should compose."*

## Result (`20260730-*`, test fold, 629,742 images, 12,041 species)

| effnetv2_s, 5 ep, identical recipe | species | genus | family |
|---|---|---|---|
| multi-head, plain cosine | 0.9110 | 0.9587 | 0.9708 |
| single head + marginals | **0.9135** | **0.9606** | **0.9739** |
| multi-head + ArcFace × z-score | 0.9069 | 0.9572 | 0.9699 |
| **A1: single head + ArcFace × z-score** | **0.9035** | **0.9491** | **0.9628** |

**The prediction failed.** 0.9035 is below the predicted floor of 0.906. Not by much — but the
prediction was not merely a number, it was the *claim that the two effects are independent*, and that
claim is what broke.

## The two effects do not compose — and the interference is concentrated in the coarse levels

Taking each intervention's measured delta from the shared multi-head baseline and adding them:

| level | single-head Δ | ArcFace Δ | additive forecast | **actual** | **interference** |
|---|---|---|---|---|---|
| species | +0.25 | −0.41 | 0.9094 | 0.9035 | **−0.59** |
| genus | +0.19 | −0.15 | 0.9591 | 0.9491 | **−1.00** |
| family | +0.31 | −0.09 | 0.9730 | 0.9628 | **−1.02** |

That shape is the finding. If the margin simply cost accuracy, species would absorb the damage and
the coarse levels would follow proportionally. Instead **the coarse levels lose roughly twice what
species does** — so what is being damaged is not the species decision, it is the *marginalisation*.

### The mechanism, and why we already had the evidence for it

Marginalisation sums the species posterior. That makes it a **calibration-dependent** operation: it
does not care only about which species ranks first, it cares how mass is distributed across all the
children of a genus. An additive angular margin deliberately trains against a *harder* target than
the true label, which sharpens the decision boundary at the cost of the posterior's calibration.
Sharper boundaries, worse sums.

The clinching evidence is a result from the same day, pointing the opposite way.
[[2026-07-30-marginal-supervision]] found that *supervising* the marginals during training left
species **exactly unchanged** (0.9135 → 0.9135) while lifting genus **+0.27** and family **+0.39** —
a pure calibration effect, with no discrimination component at all.

So the same lever appears twice in one day, in both directions:

| intervention | species | genus / family | acts via |
|---|---|---|---|
| marginal supervision | 0.00 | **+0.27 / +0.39** | calibration of the sum |
| ArcFace margin (under marginalisation) | −1.00 | **−1.15 / −1.11** | calibration of the sum |

Two independent experiments identifying the same mechanism from opposite sides is much stronger
evidence than either alone, and neither was designed to test it.

## What follows

**The obvious next run is A1 + marginal supervision**, and it is well-motivated rather than a guess:
one intervention degrades the calibration of the summed posterior, the other directly optimises it.
If the mechanism above is right, marginal supervision should recover most of the coarse-level loss
while leaving species where the margin puts it. If it does not, the mechanism is wrong and the
interference is something else — which is equally worth knowing.

**A1 is not condemned yet.** ArcFace has never been justified by closed-set accuracy; it costs
accuracy *by construction* and earns its place on open-set AUROC (0.601 → 0.9115 on the multi-head).
A 1.0 pt accuracy cost for a 30 pt AUROC gain was the trade the project already accepted. The
question A1 actually has to answer is whether that AUROC **survives the move to a single head** —
queued as `lepi-A1-ood` — and how it behaves under shift (`lepi-A1-shift`). Until those land, "A1 is
worse" is only true on the axis ArcFace was never chosen for.

## Caveat, stated because it now bounds several conclusions

The single-head and ArcFace deltas being composed here are 0.09–0.41 pt, and **this project has
never measured its seed-to-seed spread**. The species interference (−0.59 pt) could plausibly be
inflated by run-to-run noise; the coarse interference (−1.0 pt) is large enough, consistent across
two levels, and corroborated by an independent mechanism, so it is the part worth believing.

Measuring that spread — the same config run twice — is now overdue. It costs 1.5 h and it retroactively
sets the interpretation threshold for every sub-half-point row in [`../RESULTS.md`](../RESULTS.md).
