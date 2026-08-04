# Benchmarking imbalanced-learning methods properly (the long-tail reopening)

**Kind:** research · **Status:** OPEN (2026-08-01), design written before any run. Prompted by the
owner asking whether **balanced softmax** and similar imbalance methods deserve a benchmark, since
only oversampling and logit adjustment were ever tried.

## First, the awkward fact: we have already run balanced softmax

Balanced Softmax (Ren et al. 2020) trains with

$$\ell = -\log \frac{n_y e^{z_y}}{\sum_j n_j e^{z_j}} = -\log \frac{e^{z_y + \log n_y}}{\sum_j e^{z_j + \log n_j}}$$

Logit adjustment (Menon et al. 2021) trains with $z_j + \tau \log \pi_j$ where $\pi_j = n_j/N$. Since
$\log n_j = \log \pi_j + \log N$ and the constant cancels inside the softmax:

> **Balanced Softmax is exactly logit adjustment at $\tau = 1$.**

`dev/034`'s `LogitAdjustment` adds $\tau \log \pi$ at train time and predicts plain — Menon's recipe
— and `logit_adjust_tau: 1.0` scored **0.9031** against oversampling's 0.9148
([[2026-07-17-does-longtail-help]]). So the method was tested, and lost by 1.2 pt.

**But the reason it lost no longer applies.** The diagnosis was specific:

> one shared $\tau$ across three genuinely different distributions — 12,041 species, 4,333 genera,
> 102 families — and one constant cannot be simultaneously correct for all of them. It was wrong for
> two of the three.

That failure mode is **structurally impossible on the current architecture**. The single-head model
supervises *one* distribution; genus and family are marginals, not separately-adjusted heads. There
is no longer a shared constant to get wrong.

So the owner's instinct is right, but the experiment is not "try balanced softmax" — it is
**"re-run the one long-tail method we rejected, now that the reason we rejected it is gone."** That
is a much stronger motivation than a fresh benchmark would have been, and it would have been missed
by treating balanced softmax as an untried method.

## Second: one standard fix is already in the model, unnoticed

The decoupling literature (Kang et al. 2020) reports that **$\tau$-normalisation** — rescaling
classifier weights by $\lVert w_j \rVert^{-\tau}$ — is among the strongest long-tail interventions,
because a classifier trained on imbalanced data grows larger weight norms for frequent classes.

Our cosine head **constrains every prototype to $\lVert w_j \rVert = 1$** (weight-norm with the row
norm frozen). That is $\tau$-normalisation at $\tau = 1$, enforced during training rather than
applied afterwards. So the head already absorbs one of the toolbox's main entries.

This is worth stating because it reframes what remains: methods that fix the *classifier's norm bias*
should be expected to do little here, and only methods that change **which examples the loss weights**
(resampling, reweighting, margins) have room to act. It also offers a partial explanation for why
oversampling's +2.6 pt was as large as it was and logit adjustment's contribution as small.

## Third: the question worth asking is not "which method wins in-distribution"

The long-tail literature evaluates on in-distribution held-out splits, essentially without exception.
This project has now learned three separate times that in-distribution ranking does not transfer:

| intervention | in-distribution | under shift / open-set |
|---|---|---|
| single head vs multi-head | **+0.25** | **−2.10** under shift ([[2026-07-30-marginal-supervision]]) |
| max-logit vs MSP scoring | n/a | **inverts with model scale** ([[2026-08-01-the-scoring-rule-was-the-bug]]) |
| capacity vs augmentation | capacity wins | augmentation wins ([[2026-08-01-capacity-x-augmentation]]) |

Long-tail methods are a strong candidate for a fourth instance, and for a specific reason: they act
**on rare classes**, and rare classes are exactly where domain shift bites hardest — few images, so
the little that was learned is the most fragile. A method that reweights the tail may therefore look
very different on the shifted benchmark than on the held-out fold.

**Nobody in that literature reports this**, because the benchmarks (CIFAR-LT, ImageNet-LT, iNat)
supply no shifted test set. We have one. That makes this a contribution rather than a bake-off.

## The design: a 2×2, not a leaderboard

Resampling and loss reweighting are the two families that remain live, and they are usually presented
as alternatives without being crossed. Both act on class frequency, so whether they compose or
double-count is exactly the interesting question — and the capacity × augmentation factorial
([[2026-08-01-capacity-x-augmentation]]) showed a 2×2 answers more than two comparisons do.

All cells: `efficientnet_v2_s`, single species head + marginalisation, 5 epochs, everything else the
project baseline.

| | no balanced softmax | **balanced softmax** ($\tau=1$) |
|---|---|---|
| **no oversampling** | **L0** (control — never run on this architecture) | **L1** |
| **√-oversampling** ($p=0.5$) | **already have it: 0.9135** | **L2** |

Three new runs, ~1.5 h each. Each scored on the **triple** — in-distribution, shifted, open-set with
the rule named — which is now standard.

L0 matters more than it looks: oversampling's +2.6 pt was measured on the *multi-head*, at a time
when the recipe differed in other ways. **Nobody has measured what oversampling is worth on the
current architecture**, and the whole comparison is anchored to that number.

## Predictions (committed, before any run)

- **L0 ≈ 0.885–0.895.** Oversampling should still be worth ~2 pt, but I expect *less* than the
  historical +2.6, because the cosine head's unit-norm prototypes already remove part of the bias
  oversampling was correcting.
- **L1 ≈ 0.905–0.915**, i.e. **balanced softmax roughly matches oversampling** now that the shared-τ
  objection is gone. Its previous 1.2 pt deficit was, on this reading, mostly the coarse levels being
  mis-adjusted rather than a defect in the method. If L1 lands below 0.90 that reading is wrong and
  logit adjustment simply is worse here.
- **L2 < max(L1, 0.9135)** — I expect them **not** to compose. Both correct the same frequency bias:
  oversampling by showing rare classes more often, balanced softmax by lowering their effective
  threshold. Applying both should over-correct toward the tail and cost head/micro accuracy. ~65 % on
  a small loss, ~25 % on a wash, ~10 % on a genuine gain.
- **On the shifted benchmark I expect the ordering to differ from in-distribution**, and this is the
  prediction I would most like to be right about. Specifically: the more aggressively a method
  up-weights the tail, the worse it should do under shift, because tail classes carry the least
  robust evidence. Ordering guess: L0 ≥ 0.9135-cell ≥ L1 ≥ L2. **Requires >1.4 pt separation to be
  readable at all** (shifted noise floor 0.69, [[2026-08-01-how-noisy-are-our-numbers]]).

## Deliberately not in this round

- **LDAM** (Cao et al. 2019) — a *per-class* margin $\propto n_j^{-1/4}$. Genuinely interesting here
  because it is ArcFace with a frequency-dependent margin and would reuse the existing machinery, and
  because we now know margins interact with marginalisation
  ([[2026-07-30-does-arcface-compose-with-marginalisation]]). Deferred only because it needs
  per-class margin support in the loss, which is real work; queued behind the 2×2.
- **cRT / decoupled training** (Kang et al. 2020) — train the representation with *instance-balanced*
  sampling, then retrain only the classifier balanced. Their headline claim is that instance-balanced
  sampling gives the **best representations**, which would mean our oversampling is actively harming
  the backbone. That is the highest-value hypothesis in this whole area and it deserves its own entry
  rather than a cell in this table. It is also nearly free: stage 2 reuses a frozen backbone.
- **τ-normalisation** — already in the model (see above). Running it would measure nothing.
- **Class-balanced reweighting by effective number** (Cui et al. 2019) — expected to be dominated by
  balanced softmax, which is the same idea with a cleaner derivation.

## The cost the owner already accepted

Improving the baseline invalidates comparisons made against the old one. Two things make that cheaper
than it was a week ago: the **noise floors are measured**, so a re-run's difference is interpretable
immediately; and the **triple protocol** is automated, so re-scoring a model is three short jobs. The
real risk is not compute, it is drawing conclusions across a baseline change — which is what
`PLAN.md`'s hygiene section exists to prevent.

---

## L0: oversampling is an accuracy/robustness trade, not a free win (2026-08-01)

The control nobody had run. Single species head, everything at the project baseline, **oversampling
switched off.**

| effnetv2_s, 5 ep, single head | species | genus | family | **shifted** |
|---|---|---|---|---|
| **L0** — no oversampling | 0.8949 | 0.9514 | 0.9668 | **0.6445** |
| baseline — √-oversampling | **0.9135** | **0.9606** | **0.9739** | 0.6293 |
| Δ from oversampling | **+1.86** | +0.92 | +0.71 | **−1.52** |

### Both predictions confirmed

**"0.885–0.895, less than the historical +2.6"** — landed at 0.8949, at the top edge, and oversampling
is worth **+1.86 pt** here against +2.6 on the multi-head under the older recipe. The reasoning holds:
the cosine head's unit-norm prototypes are already τ-normalisation at τ=1, so part of the imbalance
correction oversampling used to supply is now built into the architecture. **A method's value is a
property of the system it sits in, not of the method.**

**"The more aggressively a method up-weights the tail, the worse it should do under shift"** — the
first cell agrees, and by more than expected. Turning oversampling *off* **gains 1.52 pt under
domain shift** (2.5× the noise floor).

### Oversampling is the fourth in-distribution/shift inversion

| intervention | in-distribution | shifted |
|---|---|---|
| single head vs multi-head | +0.25 | −2.10 |
| capacity vs augmentation | capacity wins | augmentation wins |
| max-logit vs MSP scoring | — | inverts with model scale |
| **√-oversampling** | **+1.86** | **−1.52** |

And this one lands on **the project's largest historical lever** — the change that took the baseline
from 0.8887 to 0.9148 and has been in every config since. It is not wrong, but it is not free: it
buys 1.9 pt of in-distribution macro-F1 by paying 1.5 pt of cross-source robustness.

**Why this is the expected direction.** Macro-F1 weights every species equally, so oversampling
targets exactly the classes with the fewest images. Those classes have the least evidence behind
them, so what the model learns for them is the most likely to be an artefact of their particular
photographs — photographer, background, equipment. Up-weighting them therefore up-weights the part of
the training signal that is *least* transferable. Under shift that is precisely what stops working.

### What this changes

**The default is now a decision, not a default.** For a model that will be deployed on someone
else's images, oversampling should be justified rather than assumed. It is worth noting that
`domain_aug: trap` buys **+4.0 pt** under shift while oversampling costs 1.5 — so the two together
still come out ahead, but the accounting was never done before.

**It also re-frames the rest of this benchmark.** The question is no longer "which imbalance method
gives the best macro-F1" — it is **"what does each method cost under shift for the macro-F1 it
buys?"** L1 and L2 are still running, and the exchange rate is now the number to read, not the
in-distribution column.

**It weakens no earlier conclusion**, because every architecture comparison in `RESULTS.md` held
oversampling fixed. But it does mean the shifted numbers throughout are measured *with* a lever that
suppresses them, which makes them floors — the second such qualifier after marginal supervision.

---

## The 2×2 completes: balanced softmax loses, and tail-reweighting is monotonically bad under shift (2026-08-02)

| cell | oversampling | balanced softmax | in-distribution | **shifted** | shifted micro |
|---|---|---|---|---|---|
| **L0** | — | — | 0.8949 | **0.6445** | 0.6589 |
| baseline | √ (p=0.5) | — | **0.9135** | 0.6293 | 0.6156 |
| **L1** | — | τ=1 | 0.8970 | 0.5726 | 0.5214 |
| **L2** | √ (p=0.5) | τ=1 | 0.8689 | **0.5492** | 0.4694 |

### Scoring the four predictions: three right, and the one I wanted is wrong

**L0 ≈ 0.885–0.895** → **0.8949**, at the top edge. Correct, including the reasoning: oversampling is
worth **+1.86 pt** here versus +2.6 historically, because the cosine head's unit-norm prototypes
already supply part of the correction.

**L1 ≈ 0.905–0.915, "balanced softmax roughly matches oversampling"** → **0.8970. Falsified**, and the
criterion was written explicitly: *"If L1 lands below 0.90 that reading is wrong and logit adjustment
simply is worse here."* It landed below 0.90.

So **the shared-τ explanation was wrong.** Balanced softmax was blamed in
[[2026-07-17-does-longtail-help]] on one τ spanning three level distributions; with a single level
and no shared constant it still loses to oversampling by 1.65 pt. The method is simply worse on this
problem, and the original diagnosis — which sounded mechanistic and satisfying — was a
rationalisation. Corrected there.

**L2 < max(L1, baseline)** → **0.8689**, worse than *every* other cell. Correct, and by more than
predicted (I put 65 % on "a small loss"; it is 4.5 pt below the baseline). The two levers do not
compose, they double-count.

**Shifted ordering L0 ≥ baseline ≥ L1 ≥ L2** → **0.6445 > 0.6293 > 0.5726 > 0.5492. Exactly right**,
all four cells in order, and the spread is **9.5 pt** — enormous next to the 0.69 pt noise floor.

### The result: shifted accuracy is monotone in tail-reweighting aggressiveness

Rank the cells by how hard they push probability mass toward rare classes — nothing, √-oversampling
(which deliberately *softens* the correction), full prior correction at τ=1, both — and the shifted
score falls monotonically at every step, while the in-distribution score does not order at all
(0.8949, 0.9135, 0.8970, 0.8689).

**The micro-accuracy column shows the mechanism directly.** For the un-reweighted cells macro < micro
(0.6445 < 0.6589), the normal pattern: rare species are harder. For both balanced-softmax cells the
sign **flips** — macro 0.5492 > micro 0.4694 for L2 — meaning the model now does *relatively better*
on rare species than common ones. It is not merely helping the tail; it is **over-predicting the
tail**, and under shift, where the tail's evidence was flimsiest to begin with, that collapses
ordinary accuracy by 19 pt of micro.

**Why this is the expected direction, stated before the runs:** macro-F1 weights every species
equally, so every one of these methods targets the classes with the fewest images. What a model
learns from 43 photographs is disproportionately about *those photographs* — one photographer, one
background, one camera. Up-weighting rare classes up-weights the least transferable part of the training
signal. In-distribution that is invisible, because the test fold shares those artefacts.

### What this settles, and what it opens

**Settled: balanced softmax is not worth pursuing here**, on either axis, and the earlier diagnosis
of why logit adjustment failed is retracted. √-oversampling stays the recommended long-tail
treatment — and the reason is now visible: it is the *mildest* of the interventions tested. The
square root is not a tuning detail, it is what keeps the method on the right side of this trade.

**Opened: the tail/robustness trade appears to be a law here, not a property of one method.** Four
points on a monotone curve is the strongest version of the hypothesis this project committed to, and
it is a result the long-tail literature cannot report, because CIFAR-LT / ImageNet-LT / iNat ship no
shifted test set. Worth its own paper section rather than a line in a bake-off.

**LDAM is now much more interesting than when it was deferred.** Its per-class margin
$\propto n_j^{-1/4}$ is a *fourth-root* softening, gentler even than √-oversampling — so the monotone
curve predicts it should sit **above** oversampling under shift. That is a real out-of-sample
prediction from the curve rather than a method to try, which is the right reason to run something.

**cRT/decoupled is the other one the curve speaks to:** it applies the rebalancing *only* to the
classifier and leaves the representation trained on the natural distribution. If the damage is to the
representation, cRT should escape the trade entirely.

---

## L5: oversampling's trade does not vanish at scale — it grows (2026-08-04)

Three interventions have behaved differently at 198 M than at 20 M
([[2026-08-02-f1-flagship]]), all in the same direction: capacity absorbed them. √-oversampling was
one of only two adopted interventions never re-tested across that boundary, and the larger.

F1's config with `oversample_power: 0.0`. One factor.

| | in-distribution | **probe** | probe held-out sp. |
|---|---|---|---|
| F1 — with √-oversampling | **0.9219** | 0.7209 | 0.7559 |
| **L5 — without** | 0.9055 | **0.7497** | **0.7641** |
| Δ | **−1.64** | **+2.88** (7.0× floor) | +0.82 (1.6× floor) |

**Prediction: in-distribution 0.902–0.907, probe 0.735–0.740.** In-distribution landed at **0.9055**,
inside the range. Probe landed at **0.7497**, *above* it — the effect is larger than predicted.

### Oversampling is the exception to the scale pattern, and the reason is what it acts on

| intervention | effect at 20 M | effect at 198 M |
|---|---|---|
| domain augmentation's in-distribution cost | −0.36 | **0.00** (absorbed) |
| marginal supervision's shifted benefit | +1.79 | **+0.02** (absorbed) |
| marginal supervision's coarse benefit | +0.27/+0.39 | +0.40/+0.74 (persists) |
| **√-oversampling's robustness cost** | **−1.52** | **−2.88** (**grows**) |

The three that were absorbed all act as **constraints on the optimisation** — auxiliary losses,
corrupted inputs — and a model with enough capacity satisfies them without giving anything up.
Oversampling is not a constraint. It **changes which data the model sees**, and no amount of capacity
makes a model learn from images it is shown less often. So its effect is not absorbed, and at higher
capacity the model fits the reweighted distribution *better* — which means it also fits its bias
better. That is why the cost grows rather than shrinks.

**Generalisable form:** interventions that constrain the *objective* weaken with capacity;
interventions that reshape the *data distribution* do not, and may strengthen. Predicting which of
the two an intervention is, is worth more than measuring it at one scale.

### The practical consequence is large

**L5's probe score of 0.7497 is the best of any model in the project**, including B3's self-training
(0.7370) and every 198 M variant. It was obtained by *deleting* a line from a config.

That is uncomfortable, because √-oversampling has been in every recipe since July and is the change
that took the baseline from 0.8887 to 0.9148. It is not wrong — it buys 1.64 pt of the headline
metric at this scale, and macro-F1 over 12,041 species is a legitimate thing to want. But **for a
model that will meet someone else's images, it is a net loss**, and that was invisible for six weeks
because nothing was measured off the training distribution.

### What it makes necessary

B6 (F1 + self-training, running) **has oversampling on**. The two best levers found —
self-training (+4.58 at 20 M) and dropping oversampling (+2.88 at 198 M) — have never been combined,
and both act on the shifted axis. **B7** is queued: L5's config plus the pseudo-labels.
