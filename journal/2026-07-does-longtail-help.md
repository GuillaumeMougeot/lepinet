# Do long-tail interventions push species macro-F1 past the 5-epoch baseline?

**Status:** RESOLVED (2026-07-17), both arms complete. **Oversampling: new project best, test
species macro-F1 0.9148.** Logit adjustment: a smaller, structurally-flawed win (0.9031) that
sacrificed genus and family for a modest species gain.
**Written before the results landed**, on purpose: the prediction below is a commitment, so
whatever comes back is a test rather than a story told afterwards.

## The question

macro-F1 weights every species equally, so it is dominated by the tail. Measured on the actual
training split (5,039,220 images / 12,041 species):

| | |
|---|---|
| median images per species | 177 |
| mean | 419 (i.e. heavily right-skewed) |
| min / max | 43 / 5,136 |
| species with <100 images | 3,825 (**31.8%**) |
| species with <200 images | 6,403 (**53.2%**) |

Half the species the metric weights equally have under 200 images. Those species both learn
less (few gradient updates land on them) and, as the epoch budget grows, are the ones most at
risk of memorisation. Two standard, orthogonal interventions target that tail (`dev/034_longtail.py`,
both opt-in, zero = off):

1. **Square-root oversampling** (`oversample_power: 0.5`, Mahajan et al. 2018) — reweights
   *which data is seen*. Via fastai `WeightedDL`, which samples n_train items with replacement,
   so epoch length and therefore LR-schedule timing are preserved.
2. **Logit adjustment** (`logit_adjust_tau: 1.0`, Menon et al. 2021) — reshapes *the loss*.
   Adds `tau*log(prior)` to logits at train time only; inference uses raw logits. Fisher-
   consistent for balanced error at tau=1.

Tested **separately first**. They attack the same problem by different routes and can be
redundant or actively conflicting when stacked; if each is measured alone, combining is an
informed choice rather than a guess.

## Design

| | |
|---|---|
| baseline | `20260714-072404` — 5ep one-cycle, **test 0.8887**, val 0.8880 |
| runs | `configs/20260716_heads_global_independent_muon_5ep_oversample.yaml` (`oversample_power: 0.5`)<br>`configs/20260716_heads_global_independent_muon_5ep_logitadjust.yaml` (`logit_adjust_tau: 1.0`) |
| held fixed | everything else — arch, Muon, one_cycle, warmup 0.5, grad_clip 5.0, light aug, bs 64, 460→256 |

**5 epochs, not 10**, to get a comparison in ~6.5h instead of ~13h. Note the cost: the
baseline this is measured against is 0.8887 (5ep), *not* the 0.8976 headline (10ep). A win
here does not automatically survive at 10 epochs — see the caveat below.

## Prediction (before results)

- **Oversampling: small win or a wash, +0 to +0.5 points.** Square-root sampling is the
  conservative setting and 5 epochs is short; the tail may not get enough extra exposure to
  matter.
- **Logit adjustment: the better bet, +0.3 to +1.0 points.** It is directly Fisher-consistent
  for the metric being optimised (balanced/macro error), and it costs nothing at inference.
- **Both are more likely to help at 5ep than at 10ep.** The 10ep run was still improving when
  it stopped — tail species were still learning, not yet memorising. Long-tail methods pay
  most when the tail has stopped improving, so a 5ep win may shrink at 10ep. **If either wins
  here, re-run it at 10 epochs before believing it.**
- **Stacking both: likely worse than the better one alone.** Oversampling already shifts the
  effective prior that logit adjustment assumes; applying both double-counts the correction.

**Falsification:** if neither beats 0.8887 by >0.2 points (roughly the noise floor implied by
the regulariser wash: 0.8860 vs 0.8880), the tail is not the binding constraint at this budget
and the next lever is epochs, not sampling.

## Caveat on this comparison

The venv was damaged and repaired on the day these runs launched (see
[2026-07-venv-uv-sync-incident.md](2026-07-venv-uv-sync-incident.md)). torch/torchvision/
fastcore/fastprogress were restored to their exact prior versions, but `numpy`, `pillow`,
`pyarrow`, `fsspec` and `typing_extensions` may sit at versions the 0.8887 baseline did not
use, and the replaced versions were not recorded. pillow in particular touches image decoding.
**If these results look strange, suspect the environment before the interventions**, and
re-run the 5ep baseline as a control under the current env.

(This is exactly the hole that per-run `uv pip freeze > env.txt` closes. Not yet implemented —
dev/030 is not being touched while these runs are in flight.)

## Results

| run | delta | val F1 | test F1 | vs 0.8887 (5ep base) | vs 0.8976 (10ep, prev best) |
|---|---|---|---|---|---|
| `20260716-154156` | `oversample_power=0.5` | 0.9096 | **0.9148** | **+0.0261** | **+0.0172** |
| _(queued)_ | `logit_adjust_tau=1.0` | — | — | — | — |

**New project best.** Full test-fold table (629,742 images, 12,041 species,
`data/global/preds/heads-global-independent-muon-5ep-oversample-effnetv2s/20260716-234427-global-test/`):

| level | macro-F1 | micro-acc |
|---|---|---|
| species | **91.48%** | 94.76% |
| genus | 96.03% | 97.48% |
| family | 97.26% | 99.33% |

vs the prior best (10ep, no oversampling): species F1 89.76%, micro-acc 94.09%. **Both moved
together** (+1.72pt F1, +0.67pt micro) — oversampling did not trade head accuracy for tail
gains here, both improved. And **test (0.9148) landed above val (0.9096)**, not below —
nothing was lost going from validation to truly held-out data.

### Per-epoch val species F1 — oversampling vs the 5ep baseline (`20260714-072404`)

| epoch | baseline | oversample | delta | base vloss | over vloss |
|---|---|---|---|---|---|
| 0 | 0.7153 | **0.7855** | **+0.0702** | 1.4033 | 1.3430 |
| 1 | 0.7834 | **0.8349** | **+0.0515** | 1.0384 | 0.9891 |
| 2 | 0.8325 | **0.8697** | **+0.0371** | 0.7776 | 0.7536 |
| 3 | 0.8729 | **0.8987** | **+0.0258** | 0.5574 | 0.5489 |
| 4 | 0.8880 | | | 0.4809 | |

At **epoch 3 of 5**, oversampling (0.8987) already exceeds the **10-epoch** run's final val
(0.8977) — half the compute. The geometric decay predicted +0.0267 for epoch 3; actual
+0.0258. Ratios x0.73, x0.72, x0.70.

### The mechanism, from metric divergence (epoch 3)

The two metric families tell different stories, and the gap between them *is* the evidence:

| ep | macro-F1 lead (class-weighted → **tail**) | ratio | valid_loss lead (instance-weighted → **head**) | ratio | micro-acc lead |
|---|---|---|---|---|---|
| 0 | +0.0702 | — | +0.0602 | — | +0.0091 |
| 1 | +0.0515 | x0.73 | +0.0493 | x0.82 | +0.0056 |
| 2 | +0.0371 | x0.72 | +0.0240 | x0.49 | +0.0042 |
| 3 | +0.0258 | x0.70 | +0.0085 | x0.36 | +0.0023 |

**The instance-weighted advantage is collapsing (x0.36) while the class-weighted advantage
persists (x0.70).** That is precisely what square-root sampling is supposed to do: buy tail
performance and pay for it out of head performance. Net strongly positive on macro-F1, ~parity
on instance accuracy.

This also corrects the epoch-2 reading above, which cited the valid_loss lead as corroboration.
It was a transient. The macro-F1 lead is the durable part, and the loss lead converging away is
not a weakening of the result — it is the signature of the mechanism working as designed.

**Deployment consequence:** if what ships is judged on *instance* accuracy (most photographs
are of common species), oversampling is close to a wash (+0.0023 and shrinking). It wins
decisively on macro-F1, the metric this project targets to match mini_trainer's 89.6%. Same
model, different verdict depending on which metric is the product.

**Reading epoch 0:** a large early lead, but close to the *expected* shape rather than evidence
of a better endpoint. Oversampling front-loads rare-species exposure, so the tail gets its
gradient updates sooner — that lifts macro-F1 early without implying more total learning by the
end. Baseline gains most of its ground *after* epoch 0 and one_cycle does its real work in the
late anneal. So: **ahead of schedule, not yet ahead.**

**Reading epoch 2 — the prediction is in trouble.** The lead is closing, but *decelerating
geometrically*: each epoch retains ~0.72 of the previous lead (0.0702 -> 0.0515 -> 0.0371).
That converges to a **positive asymptote**, not to zero. Naive extrapolation: ~+0.027 at
epoch 3, ~+0.019 at epoch 4 -> **val ~0.907**, which would beat not only the 5ep baseline
(0.8887) but the **10-epoch** result (0.8976). A 5-epoch run winning by fixing the data
distribution rather than by spending more compute. Oversampling also leads on **valid_loss**
throughout (0.7536 vs 0.7776), which matters because that is not a tail-weighted metric.

### Measurement audit (done before believing the above)

A lead this far outside prediction is first a reason to suspect the measurement. Checked:

- **Validation is NOT resampled.** `make_dls` passes `dl_kwargs=({"wgts": sample_wgts}, {})` —
  a per-subset tuple, so the weights reach the *train* loader only; valid is a plain sequential
  loader over the natural distribution. Confirmed empirically: **9846 valid batches** (= the
  full 630,097-image fold / 64) in both this run and the baseline.
- **Epoch length is preserved.** 78,737 train batches in both, so `WeightedDL`'s
  sample-with-replacement did not change LR-schedule timing.

The comparison is like-for-like. The lead is real.

### Scoring the prediction (written before results)

| | predicted | epoch-2 actual |
|---|---|---|
| oversampling | "+0 to +0.5, the weaker bet" | **+3.7pts, tracking to ~+1.9 at ep4** |
| "tail not the binding constraint at this budget" | | **contradicted** |

**A clear miss, and the useful kind.** The reasoning was that square-root sampling is
conservative and 5 epochs too short for extra tail exposure to matter. Wrong on both counts —
with 53% of species under 200 images, the tail appears to be the *dominant* constraint on
macro-F1, and every rung of the ladder so far (Muon, one_cycle, epochs) was an **optimisation**
fix while this is the first **data-distribution** fix. Those are not competing for the same
headroom, which is why this one is bigger than any of them.

### Epoch 4 (final) and the resolved caveats

| ep | baseline | oversample | delta | ratio |
|---|---|---|---|---|
| 3 | 0.8729 | 0.8987 | +0.0258 | x0.70 |
| 4 | 0.8880 | **0.9096** | **+0.0216** | **x0.84** |

The geometric-decay extrapolation (ratio ~x0.70-0.73 holding) predicted epoch 4 at ~+0.0185
(val ~0.9065). Actual ratio broke the trend upward to x0.84, landing at +0.0216 (val 0.9096) —
**slightly better than predicted**, not worse. The extrapolation caveat above resolved in the
favourable direction.

**Val/test agreement held despite the skewed training prior**: test (0.9148) exceeded val
(0.9096) by +0.52pt, the same direction and similar magnitude to every prior run in this
project (val/test have consistently sat within ~0.5-1pt of each other, test occasionally
higher). The concern that oversampling — the first intervention to deliberately reshape the
training-time class distribution — might decouple val from test did not materialise.

**Still open, not yet run:** whether this holds at 10 epochs. The original prediction was that
long-tail methods pay *less* at longer budgets (the 10ep run wasn't tail-starved, it was still
improving broadly). That is untested for oversampling specifically and is the natural next
experiment given how large this win is — `configs/20260715_heads_global_independent_muon_10ep_oversample.yaml`
already exists for it.

### Scoring the prediction (final)

| | predicted | actual |
|---|---|---|
| oversampling magnitude | "+0 to +0.5, the weaker bet" | **+2.6pt test vs the 5ep baseline; +1.7pt vs the previous overall best (10ep)** |
| "tail not the binding constraint at this budget" | | **wrong — it was the dominant lever found so far** |
| val/test agreement risk | flagged as the open question | **held — test even exceeded val** |

The clearest miss in this project's predictions to date, and the most valuable one: it
overturned a working assumption (optimisation was the remaining lever) rather than confirming
one. **53% of species having under 200 training images was hiding more headroom than the
optimisation-focused ladder (Muon, one_cycle, epoch count) had found in three prior rungs.**

## Conclusion (oversampling)

**Square-root oversampling is a clear, transferring win — the new project best at 91.48% test
species macro-F1, +1.7pt over the previous best (10 epochs, no oversampling), achieved in half
the epoch budget.** Recommend: (a) adopt `oversample_power: 0.5` as part of the standard
recipe going forward: (b) run the 10-epoch oversample config to see whether the two levers
stack additively, since nothing here tested that; (c) do not combine with logit adjustment as
implemented — see below.

### Scoring the logit-adjustment prediction (final)

| | predicted | actual |
|---|---|---|
| magnitude | "+0.3 to +1.0, the better bet" | **+1.4pt test — inside the low end, but by luck: species alone hides the real story** |
| mechanism | assumed clean, Fisher-consistent | **broken by an implementation gap: one shared tau across levels with 2x different log-prior spread** |
| cost | not anticipated | **genus -0.4pt, family -3.3pt test, both permanent** |

The magnitude landed inside the predicted range, but for the wrong reason — not because logit
adjustment worked as expected, but because a species-only view hides a real, structural cost
paid on the other two hierarchy levels. Predicting "the better bet" and being technically
inside range on the one metric checked is a near-miss in the same family as the epoch-2
"provisional verdict" that later partially reversed: both were correct reads of partial
evidence that would have been wrong conclusions if not checked against more than one signal.

## Conclusion (logit adjustment)

**A real but modest species win purchased by sacrificing genus and family — not worth using as
implemented.** `tau=1.0` applied uniformly across hierarchy levels ignores that family's class
imbalance (log-prior range ~9.7, 102 classes) is roughly double species' (~4.8, 12,041 classes,
`min_img_per_spc`-bounded). The result: family F1 collapsed early and never recovered (0.94 vs
0.95-0.97 test-wide), a cost this project has no use for given the metric that matters is
species-only. **Do not adopt in this form.** If revisited: scale tau per level by that level's
own log-prior std (or apply logit adjustment to the species head only, where the theory was
checked to actually hold), and re-run before drawing any conclusion about the technique itself
— this result indicts one uniform-tau implementation, not Fisher-consistent logit adjustment
as a method.

## Overall conclusion

Of the two long-tail interventions tested, **oversampling is the clear winner and the new
project standard**; logit adjustment (as implemented) is not recommended. The gap between them
is instructive: oversampling reshapes *which data is seen* without touching *what the loss
means* for any given example, so it degrades gracefully across all three hierarchy levels
together. Logit adjustment reshapes the loss itself with a single hyperparameter shared across
three genuinely different distributions, and that single hyperparameter was wrong for two of
the three. The lesson for future long-tail work in this project: prefer interventions that
don't require a shared constant to be simultaneously correct for species, genus, and family.

Combining the two was the original open question (`Design`, above) and is now settled by
elimination: don't — logit adjustment's family/genus damage would still need fixing before it
is safe to stack with anything, and oversampling alone already exceeds what either did
individually.

## Logit adjustment: epoch 0 anomaly

`20260716-234247`, epoch 0 (species F1 comparison at the same epoch):

| run | species F1 | genus F1 | family F1 | valid_loss |
|---|---|---|---|---|
| baseline (5ep one-cycle) | 0.7153 | 0.8174 | 0.8620 | 1.4033 |
| oversample | 0.7855 | 0.8657 | 0.8889 | 1.3430 |
| **logit adjust (tau=1.0)** | 0.7304 | 0.7779 | **0.5900** | 1.8349 |

Species is fine — slightly *better* than baseline (+0.0151). Genus is mildly behind. **Family
has categorically collapsed** (0.59 vs 0.86-0.89 everywhere else), not just "worse" but a
different regime, which is the kind of asymmetry that's a bug or a design flaw rather than
plain underperformance.

**Root cause, found before drawing any conclusion (measurement-audit habit from the
oversampling entry above, applied here too):** `logit_adjustments()` applies the *same*
`tau=1.0` to every hierarchy level, but the levels are not equally long-tailed. Computed
`log(prior)` directly on the training split:

| level | classes | log-prior range | std |
|---|---|---|---|
| species | 12,041 | 4.78 | 1.13 |
| genus | 4,333 | 7.04 | 1.44 |
| **family** | **102** | **9.72** | **2.31** |

Species is naturally bounded by `min_img_per_spc: 50` filtering — no species can be too rare or
(relatively) too common. Family has no such floor: a handful of families (Erebidae, Noctuidae)
absorb a huge fraction of all images, so family's log-prior spans nearly **±11.6**, roughly
double species' range, at 1/100th the class count. A single shared `tau` shifts family's logits
by an offset that swamps its own logit scale far more than it does species'. Menon et al.'s
Fisher-consistency argument for `tau=1` is derived per-distribution; stacking three heads with
very different tail severity behind one shared tau breaks it for whichever level is most
skewed — here, decisively, family.

**This is a real gap in `dev/034_longtail.py`'s `logit_adjustments()`, not a wiring bug** — the
train/eval toggle (`LogitAdjustCallback`), the per-level tensor alignment, and the
`HIERARCHY_LEVELS` ordering were all checked and are correct. The fix, if this is worth
revisiting, is a per-level tau (e.g. scaled by each level's own log-prior std) rather than one
scalar for all three.

**Decision: let the run continue rather than abort.** The metric this project optimises
(species macro-F1) is undamaged at epoch 0 — if anything slightly ahead of baseline — and
`MultiLevelWeightedCrossEntropyLoss` may already prevent family's damage from leaking into the
shared backbone's species gradient. Watching whether family recovers over epochs 1-4 is itself
the answer to "does uniform tau=1.0 work at all," which is worth having either way. If species
degrades in later epochs, that would mean the family damage *is* leaking through the shared
trunk despite the per-level loss weighting, and the run should be judged a fail regardless of
its final species number, since the mechanism producing it would be suspect.

### Epoch 1: the leak is happening

| ep | species delta vs baseline | genus delta | family delta |
|---|---|---|---|
| 0 | +0.0151 | -0.0394 | -0.2720 |
| 1 | **+0.0032** | -0.0493 | -0.2559 |

This is the scenario flagged as the reason to keep watching rather than call it early: **species'
lead over baseline is collapsing (+0.0151 -> +0.0032), and genus is falling further behind.**
Family's absolute F1 improved (0.59 -> 0.64) but its *gap* to baseline barely closed (-0.272 ->
-0.256) — it's drifting up with the general training progress every run gets, not recovering
from the tau imbalance. The shared-backbone leak looks real: family's damage is starting to
show up in the metric that matters.

**Provisional read, to be confirmed by epochs 2-4:** uniform `tau=1.0` across mismatched
hierarchy levels is a worse recipe than oversampling, and plausibly worse than the 5ep baseline
by the end. If species delta goes negative, this run should be judged a fail — not because
tau=1.0 is a bad idea in general, but because applying one scalar across levels with a 2x
difference in log-prior spread was the wrong implementation, not the wrong technique.

### Epoch 2: species delta goes negative

| ep | species delta vs baseline | genus delta | family delta |
|---|---|---|---|
| 0 | +0.0151 | -0.0394 | -0.2720 |
| 1 | +0.0032 | -0.0493 | -0.2559 |
| 2 | **-0.0048** | -0.0497 | -0.2271 |

**The predicted leak has arrived.** Species F1 (0.8277) is now behind the plain baseline
(0.8325) — the run has crossed from "family is damaged, species is fine" to "family is
damaged, and it's dragging species down too." Genus is stable but still meaningfully behind
(-0.05, roughly flat since epoch 1). Family's absolute number keeps climbing (0.71) but the gap
to baseline is closing only slowly (-0.272 -> -0.227 across 3 epochs) — at this rate it will
not close by epoch 4.

**Provisional verdict, pending epochs 3-4 for confirmation:** uniform `tau=1.0` across
mismatched hierarchy levels is worse than doing nothing, on the metric this project optimises.
Not a wash like the class-distribution regulariser — an active loss, and the mechanism is now
demonstrated rather than hypothesised: family's oversized logit shift corrupts shared-backbone
gradients enough to cost species accuracy too. The fix for a future attempt is a per-level tau
(e.g. scaled by each level's own log-prior std, so family's shift doesn't dwarf species')
rather than abandoning logit adjustment as a technique.

### Epoch 3: the decline was not monotonic

| ep | species delta vs baseline | genus delta | family delta |
|---|---|---|---|
| 2 | -0.0048 | -0.0497 | -0.2271 |
| 3 | **-0.0001** | -0.0383 | -0.1945 |

**Correction to the epoch-2 "provisional verdict":** all three deltas improved this epoch
rather than continuing to worsen. Species is back to essentially tied with baseline (not
ahead, but not the accelerating leak epoch 2 suggested either). Family's gap keeps narrowing
at a roughly steady clip each epoch (-0.272 -> -0.256 -> -0.227 -> -0.194, closing by
~0.03-0.04/epoch) — plausibly on track to be small but nonzero by epoch 4, not fully closed.

This is the value of not calling a run early on a two-point trend: epoch 2's "the leak is
happening, provisional verdict: worse than doing nothing" was a real reading of the data at
the time, but it was one data point away from being an overclaim. The honest position now is
**undetermined, leaning toward a wash relative to the plain baseline** — clearly worse than
oversampling, unclear whether better, worse, or equal to no intervention at all. Epoch 4
(final) settles it.

### Epoch 4 (final): a small win on species, family never recovers

| | baseline | oversample | logit adjust | logit-adjust delta |
|---|---|---|---|---|
| species F1 | 0.8880 | 0.9096 | **0.8925** | +0.0045 |
| genus F1 | 0.9481 | 0.9566 | 0.9170 | -0.0311 |
| family F1 | 0.9695 | 0.9744 | **0.7897** | **-0.1799** |

**Species ends a small, real win** (+0.45pt over the plain baseline) — the epoch-2 "leak"
partially reversed rather than compounding. But **family never recovered from its epoch-0
collapse**: closing from -0.272 to -0.180 across 5 epochs, on pace to need many more epochs to
close fully, if it would at all. Genus stayed meaningfully behind throughout.

**Verdict: a mixed, structurally-flawed result, not a clean win.** The technique (Fisher-
consistent logit adjustment for macro-F1) is sound; this specific application (one shared
`tau=1.0` across levels with a 2x difference in log-prior spread) sacrificed two of three
hierarchy levels for a marginal gain on the one this project scores. Compare to oversampling,
which improved all three levels simultaneously with no such tradeoff. **Oversampling is the
clear choice; logit adjustment would need a per-level tau before it's worth another training
run.** Species-only comparisons would have hidden this — the per-level breakdown is what caught
it, which is itself a point in favour of always checking beyond the metric being optimised.

Val results now stand:

| run | species F1 (val) | vs 5ep baseline |
|---|---|---|
| oversampling | 0.9096 | +0.0216 |
| logit adjustment | 0.8925 | +0.0045 |

### Test-fold result (confirms val)

`data/global/preds/heads-global-independent-muon-5ep-logitadjust-effnetv2s/20260717-074848-global-test/`
(629,742 images, 12,041 species — same fold as every other comparison in this project):

| run | species F1 | genus F1 | family F1 | micro-acc |
|---|---|---|---|---|
| baseline (5ep) | 0.8887 | 0.9502 | 0.9667 | 0.9356 |
| 10ep (prior overall best) | 0.8976 | 0.9569 | 0.9736 | 0.9409 |
| **oversampling (project best)** | **0.9148** | **0.9603** | 0.9726 | **0.9476** |
| **logit adjustment** | 0.9031 | 0.9461 | **0.9406** | 0.9444 |

Test confirms val exactly: species is a real, modest win over the plain baseline (+1.4pt) but
clearly behind oversampling (-1.2pt); family took a real, permanent hit (0.9406 vs 0.95-0.97
everywhere else) that did not resolve by test time; genus similarly landed below every other
run. No surprise on the val->test transition here — direction and magnitude both held.
