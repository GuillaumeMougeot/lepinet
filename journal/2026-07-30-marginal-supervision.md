# Does marginal supervision during *training* help? (the hierarchical head, done right)

**Kind:** research · **Status:** **RESOLVED (2026-07-30), amended 2026-08-01.** H2 confirmed at
species level — marginal supervision changes species macro-F1 by **exactly nothing**
(0.9135 → 0.9135). It was *not* inert at the coarse levels: genus **+0.27 pp** (real — 5× the
measured noise floor) and family **+0.39 pp** (**suggestive only** — the repeat run
[[2026-08-01-how-noisy-are-our-numbers]] puts family's run-to-run spread at 0.24 pp, so this clears
it by just 1.6×). The architecture claim survives intact and gains a footnote.

## The correction that prompted this

The owner's original "hierarchical head" idea was: **one species head, coarser levels derived by
marginalisation, and the loss applied at every level *during training*** — so the difficulty of the
species decision is informed by coarse supervision.

What I actually implemented and benchmarked ([[2026-07-16-why-was-fastai-behind-mini-trainer]],
`dev/050`) was mini_trainer's `ConditionalClassifier`: **separate per-level prototype layers** plus
top-down *conditioning* (`C[i] = M[i] + gather_parent(C[i+1] − logΣ_siblings M[i])`). That is a
different model — it has its own genus/family parameters and pushes information *down*, whereas the
owner's idea has no coarse parameters at all and pushes information *up*. **So the 0.8845 result
does not test the owner's hypothesis.** Recorded plainly because the mislabel could have buried a
good idea under an unrelated negative result.

## What the evidence so far actually says

| model (effnetv2_s, identical recipe) | species | genus | family |
|---|---|---|---|
| multi-head independent | 0.9110 | 0.9587 | 0.9708 |
| **single species head + marginals (inference only)** | **0.9135** | **0.9606** | **0.9739** |
| hierarchical (conditional, separate coarse layers) | 0.8845 | 0.9471 | 0.9683 |

**The single head wins at all three levels.** Not merely "redundant" — the trained coarse heads are
slightly *worse* than deriving the coarse levels from the species posterior. Combined with `dev/042`
(marginals beat trained coarse heads by +0.7 pp genus / +3.1 pp family, and are
consistency-guaranteed), the multi-head is now dominated on every axis: accuracy, parameter count,
and coherence. What remains untested is whether the marginal *loss* adds anything on top.

## The proposal: `MarginalHead`

One species prototype layer. Coarse logits are `log Σ_children exp(species log-prob)` — computed in
the forward pass — and the existing `MultiLevelCELoss` supervises all three. No coarse parameters
exist, so genus/family gradients flow **into the species head**.

## Hypotheses, before running

**H1 (why it might help — the tail).** A rare species is sampled seldom, so its prototype gets few
gradients. Under marginal supervision, *every* image of a sibling species also produces gradient
through the shared genus term, so the model is pushed to concentrate mass in the right genus even
when it cannot resolve the species. On a long-tailed set (53 % of species < 200 images) that could
improve exactly the classes macro-F1 rewards. This is the strongest argument for the idea.

**H2 (why it might do nothing — redundancy).** The marginal is a *deterministic function* of the
species posterior. CE on the marginal is minimised precisely when the correct genus holds all the
mass — which is already implied by minimising species CE. So the extra terms add **no information**;
they only re-weight the same gradient. Expected effect: ≈0, possibly a small optimisation benefit
early in training (a smoother, lower-entropy objective when the species posterior is still diffuse).

**H3 (why it might hurt — a cheaper way to win).** The coarse losses can be reduced *without*
getting the species right: put mass on any sibling within the correct genus and two of the three
terms are already happy. That is a genuine local optimum that trades species precision for coarse
confidence — the same mechanism that makes the *conditional* head worse. With equal level weights
this risk is real; it can be tuned down with `level_weights`.

**Prediction (committed):** **≈0 to +0.3 pp species macro-F1, with the tail (macro) helped more than
micro-accuracy.** I put ~60 % on "indistinguishable from the single-head 0.9135", ~25 % on a small
tail win, ~15 % on a small loss via H3. The honest expected value is low **but the experiment is
cheap (~1.5 h) and the question is central to the paper's architecture claim**, so it is worth one
run. If it wins, it stacks on ArcFace×z-score (the two are orthogonal: one shapes the loss over the
taxonomy, the other the geometry of the embedding).

**Falsification:** species macro-F1 within ±0.1 pp of 0.9135 ⇒ marginal supervision adds nothing and
the architecture claim becomes "one head, marginalise at inference, nothing else". Below 0.90 ⇒ H3,
and `level_weights` (e.g. `[1, 0.3, 0.1]`) is the follow-up before abandoning it.

## Result (`20260730-074913`, test fold, 629,742 images, 12,041 species)

| effnetv2_s, identical recipe | species | (micro) | genus | family |
|---|---|---|---|---|
| single head + marginals, **inference only** (`20260729-182718`) | 0.9135 | 0.9344 | 0.9606 | 0.9739 |
| single head + marginals, **supervised during training** | 0.9135 | 0.9339 | **0.9633** | **0.9778** |
| Δ | **0.0000** | −0.0005 | **+0.0027** | **+0.0039** |

### Scoring the prediction

The committed prediction was "**≈0 to +0.3 pp species macro-F1**, tail helped more than micro",
60 % on indistinguishable, 25 % on a small tail win, 15 % on a loss via H3.

**Species landed on the falsification criterion exactly** — not within ±0.1 pp of 0.9135, but
*identical to four decimals*. So the 60 % branch was right and **H2 is confirmed**: the marginal
loss adds no information to the species decision, because it is a deterministic function of a
posterior that species CE is already optimising. H3 (the "cheap win" local optimum) did not
materialise — micro-accuracy moved −0.05 pp, which is noise, not a trade.

**What the prediction missed: the coarse levels improved.** Nothing in H1–H3 anticipated genus and
family moving while species stood still, and the prediction only reasoned about species. That is the
part worth understanding rather than filing away.

### Why coarse can improve when species does not

The marginal is a deterministic function of the species posterior, but **genus macro-F1 is not a
deterministic function of the species *argmax***. It depends on the argmax of the *summed* posterior.
So the coarse losses can only redistribute mass **among siblings within a genus** — which leaves the
species prediction untouched for the overwhelming majority of images, while flipping the genus call
on borderline cases where mass was previously split across two genera.

That is exactly a **calibration** effect, not a discrimination one: the model was already ranking the
right species first, but its posterior was diffuse enough that summing it sometimes elected the wrong
parent. Supervising the sum fixes the sum. It is consistent with the existing finding that this model
is *under*confident ([[2026-07-20-lepi-app-compression]]).

### What this changes

**Not the architecture claim.** "One species head, marginalise for coarse levels" stands, and its
justification is unchanged: the coarse *parameters* are what hurt, not the coarse *supervision*.

**A footnote worth having**, because the two knobs are now separable and their effects are disjoint:

| | species | genus / family |
|---|---|---|
| marginalise at inference | the win vs multi-head | the win vs multi-head |
| additionally supervise the marginals | nothing | a further +0.27 / +0.39 pp |

So marginal supervision is **free coarse accuracy** — no parameters, no species cost, ~0 compute.
*(Amended 2026-08-01: read this as a genus result. Family's +0.39 pp is only 1.6× its measured
noise floor of 0.24 pp and should not be quoted as established — see
[[2026-08-01-how-noisy-are-our-numbers]].)*
Worth enabling by default *if* the coarse ranks matter, which for this project they do: rank
abstention (C1) backs off to genus and family precisely when species is uncertain, and its weakest
link was the conditional genus precision of 0.487 on the hard subset. Whether this improvement
survives *there* — on the hard subset rather than in aggregate — is the follow-up that matters, and
it is a re-run of `dev/058` on these predictions, not a new training run.

**Do not stack it with `level_weights` tuning yet.** The effect is small enough (+0.3 pp) that a
hyperparameter search over level weights would be fitting noise without a repeat run to establish the
seed-to-seed spread — a spread this project has never measured, and which now bounds the
interpretation of every sub-half-point result in `RESULTS.md`.
