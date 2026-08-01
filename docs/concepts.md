# Concepts — the vocabulary this project keeps using

Written for someone who knows the *problem* well but not the deep-learning jargon the rest of these
documents lean on. Every term below appears in `RESULTS.md`, the journal or the paper, and each is
explained with the actual numbers from this project rather than in the abstract.

Read top to bottom the first time; the later sections build on the earlier ones.

---

## 1. Logits, softmax, and why the scale matters

The model ends by producing one number per species — 12,041 of them. Those raw numbers are
**logits**. They are not probabilities: they can be negative, and they do not sum to anything in
particular.

To turn them into probabilities you apply the **softmax**:

$$P(\text{species } j) = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Exponentiate each, then divide by the total so they sum to 1. The important consequence: **softmax
only cares about *differences* between logits, not their absolute values.** Adding 100 to every
logit changes nothing.

But the *spread* matters enormously. Logits `[10, 1, 1, …]` give a confident, peaked distribution.
Logits `[10, 9.9, 9.8, …]` — same maximum — give a nearly flat one. That distinction is the seed of
half the findings in this project.

## 2. The cosine head, and why a raw cosine is a bad logit

Most classifiers end in a plain linear layer. This project ends in a **cosine head**.

The backbone turns an image into a vector $e$ of 1280 numbers — an **embedding**, the model's
internal description of the photo. Each species owns its own vector $w_j$, called a **prototype** —
think "what a *Noctua pronuba* looks like, in the model's coordinates". Both are scaled to length 1,
and the score is the **cosine of the angle** between them:

$$\cos\theta_j = e^\top w_j \in [-1, 1]$$

1 means "pointing the same way" (perfect match), 0 means "unrelated", −1 means "opposite". So
classification becomes: *which prototype does this image point most closely toward?*

**The problem: in 1280 dimensions, all angles are nearly 90°.** This is genuinely unintuitive. Pick
two random directions in high-dimensional space and they are almost certainly close to
perpendicular — the cosine between them has a standard deviation of only $1/\sqrt{d} \approx 0.028$.

So all 12,041 cosines land inside a band about 0.1 wide. Feed *that* into a softmax and every
probability comes out nearly equal, no matter how good the embedding is: the differences are too
small for the exponential to separate. This is why cosine classifiers traditionally need a **scale
factor** $s$ — you multiply everything by ~30 to stretch the band. But $s$ is a hyperparameter
somebody has to guess.

## 3. The z-score transform — a scale you don't have to guess

Instead of an arbitrary multiplier, this project applies the transform that maps that
tightly-concentrated distribution onto an approximately **standard normal** one (mean 0, standard
deviation 1) — the statistician's **z-score**:

$$Z(\cos\theta) = \sqrt{d-2}\,\bigl(\arccos(-\cos\theta) - \tfrac{\pi}{2}\bigr)$$

Two things are worth knowing about it, and they are the reason it recurs everywhere:

**It is a scale factor, but a derived one.** Its slope near $\cos\theta = 0$ is exactly
$\sqrt{d-2} \approx 35.7$. So it does the same job as $s = 30$ — and lands in the same range — but
the number comes from the dimension of the embedding rather than from a search.

**It is bounded, and that turns out to be load-bearing.** Since $\cos\theta$ can only be in
$[-1, 1]$, $Z$ can only be in $\pm\sqrt{d-2}\cdot\pi/2 = \pm 56.15$. Nothing the model does can
produce a logit outside that. Section 5 explains why that matters.

## 4. A margin — making the model work harder than it has to

**ArcFace** is a training trick. During training only, before computing the loss, it rotates the
*correct* class's angle by a fixed extra amount $m$ (here 0.3 radians ≈ 17°), making that class look
*worse* than it really is:

$$\cos\theta_y \;\longmapsto\; \cos(\theta_y + m)$$

The model is then penalised for not getting the answer right *despite the handicap*. To satisfy that,
it must push each image well past the boundary rather than just barely over it — so classes end up
tightly clustered with clear space between them.

At **inference time the margin is switched off** ($m = 0$). It shapes training; it is not part of the
model you ship.

Why this project cares: tight, well-separated clusters mean an image belonging to *no* known species
sits in the empty space between them, where it can be detected. That is section 6.

## 5. "ArcFace × z-score" — what the composition actually is

The head emits $Z(\cos\theta)$. The loss needs to apply the margin, which is defined on
$\cos\theta$. So the loss must **undo** the transform, rotate, and **redo** it:

```
head  ->  Z(cos θ)                  the logits the model produces
loss  ->  cos θ = sin(Z/√(d−2))     invert the transform
      ->  cos(θ + m)                apply the margin to the true class only
      ->  Z(cos(θ + m))             re-apply the transform
      ->  softmax + cross-entropy   the actual loss
```

**Why bother inverting instead of applying the margin in the head?** Because the head must stay
**label-free** — it must not know the correct answer — or it could not be exported and run on a
phone, where there is no correct answer to give it. Keeping the labels inside the loss is what makes
the shipped model identical to the trained one.

**Why the inversion is safe.** $\cos\theta = \sin(Z/\sqrt{d-2})$ is only valid while the sine's
argument stays within $\pm\pi/2$ — beyond that, sine turns back on itself and the inversion returns
the wrong angle. It never happens, *by construction*: from section 3, $|Z| \le 56.15$, which is
exactly the boundary. (Measured logits run around 32.6 ± 7.8, so 3 standard deviations reaches ~56 —
alarmingly close, until you notice the limit is unreachable rather than merely un-reached.) This is
pinned by a test in `tests/test_heads.py`.

**Why it was worth doing.** Applying the margin to raw cosines instead (the textbook version) scored
open-set AUROC 0.732 and cost 3.3 points of accuracy. Composed with the z-score it reaches **0.9115
for 0.4 points**. The margin needs a scale with resolution to act on; squashed into the narrow raw
cosine band, it mostly cannot.

## 6. Open-set, and how you score "I don't know"

**Closed-set** means the model assumes every photo is one of the 12,041 species it was trained on.
**Open-set** means admitting it might be something else entirely — the normal case in the field.

There is no extra output for "unknown". Instead you compute a **novelty score** from the logits and
threshold it. **AUROC** (area under the ROC curve) measures how well that score separates known from
novel, and has one very concrete reading:

> **AUROC is the probability that a randomly chosen novel image gets a higher novelty score than a
> randomly chosen known one.** 0.5 is a coin flip. 1.0 is perfect.

So 0.601 (the plain cosine head) means barely better than guessing; 0.9068 means genuinely usable.

**The scoring rule matters more than expected.** Two candidates:

- **max-logit** — how strongly does the best prototype match? Reads *one* number.
- **MSP** (max softmax probability) — how much better is the best than all the others? Reads the
  *whole* vector.

They can rank two images in opposite orders:

| image | logits | max-logit | MSP |
|---|---|---|---|
| A — one clear winner | `[10, 1, 1, 1, 1]` | 10.0 | **0.9995** |
| B — five-way near-tie | `[12, 11.9, 11.8, 11.7, 11.6]` | **12.0** | 0.2419 |

B has the higher maximum but no winner — which is what a novel species looks like, resembling several
known prototypes about equally. max-logit calls B the more familiar one; MSP calls A. In this project
max-logit is the better rule on a 20 M-parameter model and **6–7 points worse** on a 198 M one, because
a better-fitted model matches *everything* strongly and only relative dominance still carries signal.

## 7. Fine, coarse, and marginalisation

The taxonomy runs species → genus → family. **Fine** means species (12,041 classes, most specific);
**coarse** means genus and family (4,333 and 102, more general).

Two ways to predict all three:

- **Separate heads** — one classifier per level, each trained on its own labels. This is the
  "multi-head" model. It has **coarse parameters** *and* **coarse supervision** (a loss at each level).
- **Marginalisation** — one species classifier, and the genus probability is simply the **sum of the
  probabilities of all species in that genus**. No extra parameters at all.

$$P(\text{genus } g) = \sum_{\text{species } s \in g} P(s)$$

Marginalisation makes the levels **probabilistically coherent** — the coarse number *is* the sum of
the fine ones, so both are statements from a single set of beliefs. Separate heads have no such
relation and can report a genus probability that contradicts their own species probabilities (they do,
on 1.8 % of images).

**Careful, though:** coherence does **not** mean the genus prediction always matches the parent of the
species prediction. `max` and `sum` do not commute — one confident species can be outvoted by many
mediocre siblings of a different genus:

| | | |
|---|---|---|
| species probabilities | `[0.40, 0.12, 0.12, 0.12, 0.12, 0.12]` | genus A = {first}, B = {rest} |
| best species | the first one, in **genus A** | |
| genus probabilities | A = 0.40, **B = 0.60** | so the best genus is **B** |

This repo claimed the opposite for weeks ("consistent by construction") before a test caught it.

**Coarse supervision** means applying the loss at genus and family *during training*, whether or not
separate parameters exist. That distinction turns out to matter a lot: coarse **parameters** hurt,
coarse **supervision** helps — but only visibly under domain shift.

## 8. The three numbers every model is reported with

A single accuracy number has repeatedly picked the wrong model here, so every model carries three:

| | what it asks | benchmark |
|---|---|---|
| **in-distribution** | how good on held-out images from the *same* source? | 629,742 images, 12,041 species |
| **shifted** | how good on images from a *different* source (a camera trap)? | 47,905 images, 486 species |
| **open-set** | can it flag a species it was never trained on? | AUROC, rule named |

Plus two supporting terms:

**macro-F1** — the headline metric. F1 balances precision and recall; **macro** means averaged over
species with every species weighted *equally*, no matter how many photos it has. That is deliberate:
the long tail is the hard part, and an average weighted by image count would let a handful of common
moths hide thousands of failures. (**micro-accuracy** is the image-weighted version, reported
alongside as a sanity check.)

**Noise floor** — retrain an identical configuration and the score moves a little. That movement is
the floor below which a difference means nothing. Measured here: **0.0000** species, 0.0005 genus,
0.0024 family, **~0.006** shifted. Species macro-F1 is nearly deterministic because it averages 12,041
noisy per-class scores; family is noisiest because it averages only 102. **Never quote a difference
smaller than its level's floor.**

## 9. The long tail

53 % of species have fewer than 200 images; some have 50. A model trained naively becomes excellent
at the common species and poor at the rest — and since macro-F1 weights them equally, that shows up
immediately.

Two families of fix, both in this repo's history:

- **Resampling** — show rare species more often. `√-oversampling` draws each species with probability
  proportional to $\sqrt{n}$ rather than $n$, softening the imbalance without erasing it. Worth
  **+1.86 pt** in-distribution.
- **Loss reweighting** — leave the sampling alone and make the *loss* more forgiving toward rare
  classes. **Balanced softmax** adds $\log n_j$ to each logit during training, which is exactly
  **logit adjustment** at $\tau = 1$ (the two names describe the same formula).

A finding worth carrying: **oversampling costs 1.52 pt under domain shift while buying 1.86
in-distribution.** Rare classes have the least evidence behind them, so what the model learns for them
is the most likely to be an artefact of their particular photographs — and up-weighting them
up-weights exactly the least transferable part of the signal.

---

## Where to go next

- The method, stated formally: [`paper/DRAFT.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/paper/DRAFT.md)
- Why each setting is what it is: [design decisions](design-decisions.md)
- What everything scored: [`RESULTS.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/RESULTS.md)
