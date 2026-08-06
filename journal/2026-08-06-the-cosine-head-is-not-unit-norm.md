# The cosine head's prototypes are not unit-norm, and the docs say they are

**Kind:** incident · **Status:** **OPEN (2026-08-06).** Observation confirmed on two independently
trained checkpoints; **mechanism not yet identified.** Written now rather than after diagnosis
because the observation contradicts a documented invariant and several claims rest on it.

## The observation

`dev/068` prints the prototype row norms. On both trained checkpoints they are not 1:

| checkpoint | min | mean | max | std |
|---|---|---|---|---|
| ArcFace × z-score (A1) | 0.582 | **1.081** | 1.711 | 0.103 |
| plain cosine (single-head baseline) | 1.550 | **1.767** | 2.374 | 0.113 |

The design says otherwise. `IndependentHead._normalize_layer` applies `weight_norm(dim=0)`, sets the
magnitude `original0` to 1 and calls `requires_grad_(False)`, with the docstring:

> This is what makes the layer a *cosine* classifier: the effective ``layer.weight`` is always
> unit-norm per row, so ``F.linear(unit_embedding, weight)`` is a cosine similarity.

It is not a cosine similarity on these checkpoints. With mean row norm 1.767 the "cosine" reaches
well past 1, and `cosine_to_zscore` **clamps to ±(1 − 1e-7)** — so everything above 1 saturates to
the same z-score.

## What is ruled out

- **Not the load path.** A fresh head saves and reloads with row norms exactly 1.0; the state dict
  carries `parametrizations.weight.original0/original1` and round-trips cleanly.
- **Not `dev/068`'s reading.** With the raw weight the reimplementation now agrees with the model's
  own forward at **0.9957** (ArcFace) and **0.9949** (plain), and both reproduce their known scores.
  So the model really does use non-unit rows.
- **Not `learn.unfreeze()` on its own.** A minimal Learner leaves `original0.requires_grad` False
  after `unfreeze()`.
- **Not Muon's optimiser step directly** — `muon_opt_func` passes fastai's groups through without
  filtering on `requires_grad`, but a standard AdamW skips parameters whose `grad` is None.

So something in the *full* training path moves `original0`, and the minimal reproductions do not
capture it. Candidates not yet tested: the interaction of fastai's parameter-group splitter with
`OptimWrapper.clear_state`, weight decay reaching a frozen parameter through `MuonAuxAdamW`, or the
bf16 wrapper re-creating parameters.

## What this does and does not invalidate

**Unaffected — every accuracy number.** All of them are measured through the model's own forward,
which is internally consistent whatever the row norms are. The port reproduced 0.9148, and the
centroid comparison is valid (0.995 agreement).

**Affected — the z-score's calibration argument.** §2.2 of the paper argues the transform maps a
concentrated cosine distribution onto an approximately standard normal one. That derivation assumes
the input *is* a cosine in [−1, 1]. With row norms up to 2.37, an unknown fraction of scores hit the
clamp and share an identical, maximal z-score. The transform is then not doing what the paper says
for those entries.

**Affected — the ArcFace margin round-trip.** `lepinet.loss.apply_arcface_margin_zscore` recovers
`cos = sin(Z/√(d−2))` from the emitted logit, rotates, and re-applies. That inverse is only correct
if the emitted `Z` came from a true cosine. For any clamped entry it does not, and the recovered
angle is wrong. The tests added on 2026-08-01 verified this round-trip **on synthetic unit-norm
input**, which is exactly the assumption now in question — a test that cannot fail the way the system
actually fails.

**Possibly explains something already observed.** ArcFace's rows are much closer to unit (1.081) than
the plain head's (1.767). If the margin term implicitly penalises magnitude growth, that would be a
mechanism — and it would also explain why the two heads' prototype spectra differed so sharply
before the fix.

## Next

1. **Measure the clamp rate**: what fraction of logits hit ±1 at inference, per head. Cheap, and it
   bounds how much of the calibration argument is affected. If it is a fraction of a percent, this is
   a documentation fix; if it is large, §2.2 and §2.3 need rewriting.
2. **Find the mechanism** by instrumenting a real training run — log `original0` mean and std every
   epoch. One short run answers it.
3. **Then decide**: enforce the invariant (assert row norms in `forward`, or renormalise), or accept
   learned magnitudes and *document* the head as a scaled-cosine classifier, updating the derivations
   accordingly. The second may be the better science — a learned per-class magnitude is a real
   modelling choice, and one head has evidently been benefiting from it.

**Not being done yet:** changing the head. Every published number in this project was produced with
whatever this behaviour is, and changing it before it is understood would invalidate the lot.
