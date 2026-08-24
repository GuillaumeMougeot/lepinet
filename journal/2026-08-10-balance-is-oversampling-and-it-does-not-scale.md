# G3 + B10: the pseudo-label class distribution is a small-model lever

**Kind:** research · **Status:** **RESOLVED (2026-08-24, all four cells measured).** At 20 M,
balancing swings results by up to 3.6 pt in either direction depending on whether the trunk is
frozen; at 198 M every one of the four effects attenuates to under 1 pt. **End-to-end keeps a real
lead at 198 M that it does not have at 20 M.** *(The 2026-08-10 framing — "it inherits oversampling's
scale profile" — was generalised from one cell and is corrected in the B10 addendum.)*

**Original status (2026-08-10). Prediction narrowly missed, and it
reverses the previous day's conclusion.** At 198 M, balancing the pseudo-label set buys **+0.92 probe** and
costs **−0.82 held-out species** — a trade, where at 20 M it was free (+1.51 / +1.87). And with both
arms at their best configuration, **end-to-end training keeps a real lead at 198 M** that it does not
have at 20 M.

## Result

| | in-distribution | probe | held-out species |
|---|---|---|---|
| G2 — staged, natural pseudo-labels | **0.9150** | 0.7648 | **0.7600** |
| **G3 — staged, balanced** | 0.9138 | **0.7740** | 0.7518 |
| Δ | −0.12 | **+0.92** (2.2× floor) | **−0.82** (1.6× floor) |

**Predicted probe 0.775–0.795, falsified below 0.7689. Landed 0.7740** — 0.10 pt below the range,
clearing falsification by 0.51. **Held-out predicted 0.770–0.790, landed 0.7518** — 1.8 pt below.
In-distribution predicted within 0.3 pt of G2 and landed 0.12 below, inside. So one near-miss, one
clear miss, no falsification.

Both deltas are above the noise floor and they point in opposite directions, so this is a genuine
trade rather than a wash.

## The mechanism was already in the ledger, under a different name

At 20 M, balancing was free. At 198 M it costs held-out species. That is not a new phenomenon in this
project — it is the *oversampling* phenomenon:

| intervention | 20 M | 198 M |
|---|---|---|
| √-oversampling's shifted cost (L5) | −1.52 | **−2.88** |
| **balanced replication's held-out effect** | **+1.87** | **−0.82** |

**Balanced replication *is* oversampling, applied to the pseudo-label subset.** Both flatten a class
distribution by repeating examples from under-represented classes; the only difference is which rows
they act on. So it should have been predicted to worsen with capacity, and by this project's own
stated heuristic it *was* predictable:

> ask whether an intervention adds information or merely constrains the optimisation. Constraints
> weaken with capacity; changes to what the model sees do not. — [[2026-08-06-f2-capstone]] (G1)

with the standing exception recorded in the same place: *"√-oversampling, which reshapes the data,
gets **worse** with scale."* Balanced replication reshapes the data. It belongs in the exception, and
I put it in the wrong bucket when I predicted 0.770–0.790 for held-out. **The heuristic was right and
I misapplied it** — worth recording, because the heuristic's value is entirely in whether it is
actually consulted before the run rather than after.

## The correction to yesterday's conclusion

Yesterday, on the strength of B9, I wrote into `paper/DRAFT.md`:

> The 1.65-point external deficit reported in earlier drafts was an artefact of scoring both arms in
> the configuration that suits end-to-end training; **it is not a property of staging.**

**That is true at 20 M and false at 198 M.** Each regime at its own best configuration:

| | in-distribution | probe | held-out species |
|---|---|---|---|
| **20 M** — R5 (staged) vs B3rep5x (end-to-end) | **+0.71** | −0.14 (0.3× floor) | **+0.77** (1.5×) |
| **198 M** — G3 (staged) vs B8 (end-to-end) | **+0.78** | **−0.58** (1.4×) | **−2.98** (5.7×) |

At 20 M the probe gap is a third of a noise floor and the staged recipe wins the other two axes. At
198 M end-to-end leads *both* shifted axes by margins well outside noise, while staged keeps its
in-distribution advantage. The trade does not dissolve at scale; it dissolves at **small** scale.

**This conclusion is robust to the run still in flight.** B10 (balanced end-to-end at 198 M) can only
raise end-to-end's best or leave it at B8's 0.7798 — it cannot help the staged arm. So the 198 M
verdict stands whichever way B10 lands. *(Confirmed 2026-08-24: it landed at 0.7800, a tie. See the
addendum.)*

**And the in-distribution advantage is the stable part.** +0.71 at 20 M, +0.78 at 198 M, +0.90 for
G2 — the staged recipe reliably buys closed-set accuracy at a fraction of the training cost. What is
not stable is whether it also matches end-to-end under shift.

## What the paper should now say

Not "staging is free", and not "staging costs 1.65 pt". The defensible statement is:

> The staged recipe buys **+0.7 to +0.9 pt in-distribution** at a fraction of the training cost, and
> is **re-runnable per deployment without labels**. Whether it also matches end-to-end training under
> domain shift depends on capacity: at 20 M it does (−0.14 pt probe, +0.77 held-out); at 198 M it does
> not (−0.58, −2.98). The pseudo-label class distribution must be tuned per regime, and the setting
> that is free at 20 M becomes a trade at 198 M.

That is a weaker headline than yesterday's and a stronger one than F2's, and unlike either it is
consistent with every number measured.

## Process note

This is the second consecutive day a conclusion has moved, in both directions: B9 dissolved the trade,
G3 restored it at scale. The common cause is that **both were measured at one scale and asserted
generally**. The rule this project already has — *change one factor per run* — needs a companion:
**a claim about a method is not established until it has been measured at more than one capacity**,
because five interventions here have now changed magnitude across 10× and two have changed sign.
Added to `docs/design-decisions.md`.


---

## B10: the verdict holds, and the scale story needs one correction (2026-08-24)

Balanced end-to-end at 198 M, read 14 days late because the UCloud token expired.

| | in-distribution | probe | held-out |
|---|---|---|---|
| B8 — end-to-end, natural | 0.9060 | 0.7798 | **0.7816** |
| **B10 — end-to-end, balanced** | 0.9058 | **0.7800** | 0.7741 |
| Δ | −0.02 (0.4× floor) | **+0.02 (0.05× floor)** | **−0.75** (1.4×) |

**Predicted probe 0.770–0.785 and expected to lose. Landed 0.7800** — above the range by 0.15 pt, and
0.0002 above B8, which nominally trips the falsification line I set at "exceeds 0.7798". Tripping a
line by **5 % of a noise floor** is a tie, not a refutation, and I am recording it as one rather than
claiming either a hit or a clean falsification.

**The 198 M verdict is unchanged and now rests on a fair comparison.** Best-vs-best:

| | staged (G2/G3) | end-to-end (B8/B10) | Δ |
|---|---|---|---|
| in-distribution | **0.9150** | 0.9060 | **+0.90** |
| probe | 0.7740 | **0.7800** | −0.60 (1.5× floor) |
| held-out species | 0.7600 | **0.7816** | −2.16 (4.2× floor) |

B10 did its job: it confirmed B8 is representative of end-to-end's best at 198 M, so the staged arm's
deficit is not an artefact of comparing a tuned arm against an untuned one.

## The correction: attenuation, not amplification

On 10 August I wrote that balanced replication "inherits oversampling's scale profile", meaning it
gets *worse* with capacity. With all four cells measured, that is too simple:

| | 20 M | 198 M |
|---|---|---|
| balance on **staged**, probe | +1.51 | +0.92 |
| balance on **staged**, held-out | +1.87 | **−0.82** |
| balance on **end-to-end**, probe | −0.71 | **+0.02** |
| balance on **end-to-end**, held-out | −3.62 | **−0.75** |

**Every one of the four attenuates with capacity.** The largest effect at 20 M is 3.62 pt; at 198 M it
is 0.92. One of them (staged held-out) crosses zero on the way down, which is what I generalised from
— and generalising a sign flip from a single cell was the error.

The defensible statement is: **the pseudo-label class distribution is a small-model lever.** At 20 M it
swings results by up to 3.6 pt in either direction depending on whether the trunk is frozen; at 198 M
it barely matters at all. That is the ordinary "constraints weaken with capacity" pattern this project
has now seen six times, not the oversampling exception I filed it under.

The oversampling comparison still holds for the *staged held-out* cell specifically, and the mechanism
argument (balancing concentrates replication on the classes with fewest unique images) still explains
why the frozen and trainable trunks differ in **sign**. What it does not explain, and what I asserted
without evidence, is a claim about magnitude across scale.

**Practical consequence:** at 198 M, do not bother balancing. The setting is worth tuning at 20 M and
is within noise at deployment scale — which also means the recommended recipe can drop a
hyperparameter rather than carry one that must be set per capacity.
