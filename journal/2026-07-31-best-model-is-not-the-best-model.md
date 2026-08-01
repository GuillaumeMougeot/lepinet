# The best in-distribution model is the worst deployable one

**Kind:** research · **Status:** **RESOLVED (2026-07-31), SUBSTANTIALLY CORRECTED (2026-08-01).**
The open-set column below used `max-logit`, and that rule turns out to be badly suited to the large
model: rescoring with `msp` recovers +6.05 pt on A2, cutting the capacity penalty from 8.8 pt to
**1.64 pt**. **The strong form of this entry's claim — that the best in-distribution model is the
*worst* deployable one — is not supported.** The axes do still disagree in ordering, and
in-distribution macro-F1 still should not be the sole criterion, but the trade-off was largely
manufactured by the readout. See [[2026-08-01-the-scoring-rule-was-the-bug]] before quoting anything
below.

Original status: **RESOLVED (2026-07-31)** — the three evaluation axes rank our
models in **opposite orders**, and the model that wins the headline metric loses on both axes that
describe deployment. A 20 M model with augmentation beats a 198 M model without it under shift, and
the 198 M model is the *worst* of the three at detecting unseen species.

## The full triple, finally measured on one architecture family

Every model below uses a single species head with ArcFace × z-score and marginalisation. Only the
backbone and the augmentation differ.

| model | params | in-distribution | shifted (flemming) | open-set AUROC |
|---|---|---|---|---|
| **A1** effnetv2_s | 20 M | 0.9035 | 0.6437 | **0.9068** |
| **B1** = A1 + `domain_aug: trap` | 20 M | 0.8999 | **0.6836** | 0.9010 |
| **A2** DINOv3-ConvNeXt-L | 198 M | **0.9216** | 0.6616 | 0.8298 |
| *(reference)* DINOv3-ConvNeXt-L, multi-head, no margin | 198 M | 0.9311 | 0.7098 | — |

**Read the bold entries.** Three axes, three different winners, and the ordering is close to
reversed: A2 is first in-distribution and last at novelty detection; B1 is last in-distribution and
first under shift.

## Three findings, in order of how much they change what we do

### 1. Augmentation beats capacity for robustness — at a tenth the size

B1 (20 M, with three hand-named nuisance transforms) scores **0.6836** under shift. A2 (198 M, no
domain augmentation) scores **0.6616**. The small model wins by 2.2 pt while being **10× smaller**,
trained at lower resolution, and cheaper on every axis that matters for a phone.

This does not contradict H4 ("backbone choice matters 3× more under shift"). H4 compared backbones
*to each other* with augmentation held fixed, and that finding stands. What is new is the exchange
rate **between** the two levers: a few percent of training throughput spent on augmentation buys more
robustness than a 10× parameter increase. If the goal is a model that works on someone else's
camera, augmentation is the better purchase, and it is not close.

### 2. Scale *hurts* open-set detection — *(corrected 2026-08-01: mostly a scoring-rule artifact; the real penalty is 1.64 pt, not 7.7. See [[2026-08-01-the-scoring-rule-was-the-bug]].)*

Same head, same margin, same open-set benchmark: **0.9068 at 20 M, 0.8298 at 198 M.** A 7.7 pt loss
from making the model an order of magnitude larger and 1.8 pt better in-distribution.

The mechanism is the familiar one in a new place: a higher-capacity model fits the training species
more tightly, which makes it *more* confident everywhere — including on inputs it has never seen. It
projects a novel moth onto a known prototype with conviction. Closed-set fit and open-set caution
are in tension, and this is the first time this project has measured that tension directly.

**Caveat, stated plainly:** A1 and A2 differ in backbone *and* resolution (256 vs 320) *and* epochs
(5 vs 6), so "scale" is shorthand for the whole bundle. The headline — the best in-distribution model
is the worst at novelty — does not depend on attributing it correctly, but the mechanism claim does.
Isolating it needs a resolution-matched run and is not yet done.

### 3. The margin's cost amplifies under shift

Adding ArcFace × z-score to DINOv3-ConvNeXt-L costs **0.95 pt in-distribution** but **4.82 pt under
shift** (0.7098 → 0.6616) — five times larger where deployment actually lives.

This extends [[2026-07-30-does-arcface-compose-with-marginalisation]] and fits its calibration story.
Shift already corrupts the geometry the score is read from: [[2026-07-28-flemming-generalization]]
measured the *known* max-logit mean collapsing 32.6 → 20.6 under shift. A margin that sharpens
boundaries makes the model less tolerant of exactly that displacement. The two degradations compound
rather than add.

**So yesterday's verdict on A1 was right but under-priced.** "About 1 pt of closed-set accuracy for
0.30 of AUROC" was measured in-distribution. On the shifted benchmark the accuracy side of that trade
is several times worse. The margin still earns its place — 0.601 → 0.9068 is not available any other
way we know — but the invoice is bigger than it looked.

## What this changes

**The recommendation flips.** Before today the candidate final model was A2, on the strength of
0.9216 in-distribution. On the axes that describe deployment — does it work on someone else's images,
does it know what it doesn't know — **A2 is the weakest of the three**. If something had to ship
tonight it would be B1: 20 M parameters, shifted 0.6836, AUROC 0.9010, and small enough for the
browser bundle without distillation.

**Evaluation policy, sharpened.** The triple rule adopted after H4 said to *report* all three
numbers. Today's result says something stronger: **they must not be aggregated, and in-distribution
macro-F1 must stop being called the headline.** It is now demonstrably anti-correlated with the other
two across our own model set. A paper that ranked these models by 4.2's metric would recommend the
worst one.

**The obvious next run is the combination**, queued as B4: DINOv3-ConvNeXt-L + ArcFace × z-score +
`domain_aug: trap`. The two levers act on different failure modes — capacity on discrimination,
augmentation on nuisance invariance — so they *should* be complementary. But "should compose" is
exactly the reasoning that failed for A1, so the prediction is stated and will be scored.

**Prediction (committed):** shifted **0.70–0.73** (B1's +4.0 augmentation gain applied to A2's 0.6616,
minus some overlap), in-distribution **0.915–0.920**, AUROC **0.83–0.86** (augmentation slightly
helped nothing on this axis for B1, so expect A2's weak AUROC to persist — **if AUROC comes back
above 0.88 my "scale hurts open-set" mechanism is wrong** and resolution or epochs deserve the
blame instead).

## What is still missing

**A5 (the repeat run) has never mattered more.** Several deltas argued above are 2 pt or larger and
safe, but the −0.95 in-distribution margin cost and B1's −0.36 are not, and this project still has no
measured run-to-run spread. It is running now.
