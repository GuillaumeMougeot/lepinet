# Does marginal supervision during *training* help? (the hierarchical head, done right)

**Kind:** research · **Status:** OPEN — hypotheses written before the run, per the journal convention. 2026-07-30.

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

## Result

_(pending)_
