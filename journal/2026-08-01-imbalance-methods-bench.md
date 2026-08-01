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
