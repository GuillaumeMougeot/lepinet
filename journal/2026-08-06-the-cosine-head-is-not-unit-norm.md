# The cosine head's prototypes are not unit-norm, and the docs say they are

**Kind:** incident · **Status:** **OPEN (2026-08-06) — but downgraded.** The clamp rate is now
measured and it is a **documentation problem, not a correctness one**: no ties, top-1 essentially
never saturates, so no reported accuracy is affected. The mechanism is still unknown. Original
status: Observation confirmed on two independently
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

## The clamp rate, measured (2026-08-06)

| | ArcFace × z-score | plain cosine |
|---|---|---|
| all logits clamped | **0.00003 %** | **67.4 %** |
| **top-1** clamped | 0.37 % | **0 %** |
| two or more clamped (argmax decided by index order) | **0** | **0** |
| top-1 raw score, mean / max | 0.766 / 1.107 | −0.678 / 0.353 |
| prototype row norms, mean / max | 1.081 / 1.711 | 1.767 / 2.374 |

**Accuracy is provably unaffected on both heads.** The tie rate is exactly zero, so the argmax is
never decided by index order, and the top-1 essentially never saturates. Every macro-F1 in this
project stands as measured.

**The two heads clamp at opposite ends, for different reasons, and that is the interesting part.**

*ArcFace* clamps only at the **top**, on 0.37 % of images: its row norms are close to 1 (mean 1.081),
so scores exceed +1 only for the very best matches. The consequence is narrow — for those images the
reported confidence is capped, so the calibration claim has a 0.37 % exception.

*Plain cosine* clamps **67 % of all logits at the bottom**. Its rows are much larger (mean 1.767) and
its top-1 raw score never reaches 0.36, so the score distribution sits low: two thirds of classes
fall below −1 and are pinned to an identical floor. That does not move the argmax, but it flattens
the entire tail of the softmax into one value.

**A hypothesis worth one cheap test, stated as a hypothesis.** The plain head's open-set AUROC is
0.601, near chance, and the rules that read the *shape* of the logit vector (MSP, entropy) were the
ones that behaved oddly for it. A distribution with 67 % of its mass pinned to a single floor value
has had its shape destroyed. This may be *why* the plain head cannot do novelty detection — not
because its embedding is poor, but because the transform discards the information the score needs.
That is testable by scoring novelty on the **raw** pre-clamp values, which costs one rerun of
`dev/061` and no training.

### What this settles

- **Documentation fix, not a rewrite.** The paper's §2.2 needs a footnote — the transform assumes a
  cosine input, the rows are not unit, and the practical consequence is bounded at 0.37 % of top-1
  confidences for the head we actually ship.
- **The ArcFace round-trip is safe** in 99.6 % of cases, which is the number I could not put on it
  yesterday.
- **The mechanism is still unknown** and is now the lower-priority half. It costs a 6.4 h
  instrumented run and buys an explanation rather than a correction, so it waits behind work that
  changes results.

## Next

1. ~~**Measure the clamp rate**~~ **DONE — see above. It is a documentation problem.**
2. **Score novelty on the pre-clamp values**, testing whether the plain head's near-chance open-set
   AUROC is caused by 67 % of its logits being pinned to one floor. One rerun of `dev/061`, no
   training.
3. **Then find the mechanism** — instrument a real run and log `original0` mean/std per epoch. This
   is now the *lowest*-priority piece: it buys an explanation rather than a correction, and it costs
   6.4 h.
4. **Then decide**: enforce the invariant (assert unit rows in `forward`), or accept learned
   magnitudes and *document* the head as a scaled-cosine classifier with the derivations updated. The
   second may be the better science — a learned per-class magnitude is a real modelling choice, and
   the evidence is that both heads have been using one.

**Not being done:** changing the head. Every published number was produced with this behaviour, and
changing it before it is understood would invalidate the lot for no measured gain.
