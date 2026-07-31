# Do single-head marginalisation and ArcFace × z-score compose? (A1)

**Kind:** research · **Status:** **RESOLVED (2026-07-31).** They do **not** compose — the prediction
was falsified — but **A1 stands**, because the cost lands on closed-set accuracy while the benefit
(open-set AUROC **0.9068**, vs ~0.601 for a plain head) survives the single head intact. The
interference then **replicated at 10x scale** in A2, unplanned, which promotes the calibration
mechanism from a plausible story to a measured one and makes A4 worth building.

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

---

## The verdict: open-set survives, and the cost replicates at 10× scale (2026-07-31)

### A1 earns its place on the axis it exists for

| effnetv2_s | species F1 | open-set AUROC |
|---|---|---|
| single head, plain cosine | 0.9135 | ~0.601 (plain-head baseline) |
| multi-head + ArcFace × z-score | 0.9069 | 0.9115 |
| **A1: single head + ArcFace × z-score** | 0.9035 | **0.9068** |

**C2 answered: open-set detection survives the move to a single head** — 0.9068 vs 0.9115, a loss of
0.47 pt against a 30-point gain over the plain head. So the trade the project provisionally accepted
on the multi-head baseline holds on the architecture that will actually ship: **about 1 pt of
closed-set accuracy for roughly 0.30 of AUROC.**

That is the number that matters, because a deployed classifier that cannot flag an unseen species is
wrong in a way macro-F1 does not measure. A1 stands.

### A2 replicates the interference pattern at 10× the parameters

A2 (DINOv3-ConvNeXt-L, 198 M) was never designed to test the compose question — it is the candidate
final model. It tested it anyway:

| | species | genus | family |
|---|---|---|---|
| **A1** vs its plain single-head reference | −1.00 | −1.15 | −1.11 |
| **A2** vs its plain multi-head reference | −0.95 | −1.21 | −1.13 |

Two backbones, an order of magnitude apart in capacity, and the ArcFace × z-score cost is the same
to within ~0.06 pt at every level — **including the signature asymmetry**, coarse levels losing more
than species in both.

This matters more than either run alone. The calibration mechanism was proposed from a single
observation and could have been noise or an effnetv2_s quirk; an unplanned replication at 10× scale
is much harder to explain away. It also says the cost is **not capacity-bound**: a bigger model does
not absorb the margin's calibration damage, so this will not quietly go away by scaling, and A4 (the
marginal-supervision fix) is worth building rather than waiting out.

### A2 in-distribution, and its own falsified prediction

| DINOv3-ConvNeXt-L @320, 6 ep | species | genus | family |
|---|---|---|---|
| multi-head, plain cosine | 0.9311 | 0.9731 | 0.9867 |
| **A2: single head + ArcFace × z-score** | **0.9216** | 0.9610 | 0.9754 |

Predicted 0.93; landed at 0.9216. **Falsified, by almost exactly the amount A1 was** — and for the
same reason, which is the point. The A2 prediction was written before A1 reported, so it inherited
the same wrong assumption of independence. Two falsifications from one bad premise, not two.

A2 is nonetheless the **best open-set-capable model in the project**: +1.8 pt species over A1 with
the same head. Its shifted and AUROC numbers are queued (`lepi-A2-shift`, `lepi-A2-ood`) and decide
whether it ships.
