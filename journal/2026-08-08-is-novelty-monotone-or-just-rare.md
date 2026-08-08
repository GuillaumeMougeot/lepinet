# C3b: is novelty detection monotone in taxonomic distance, or was C3 measuring rarity?

**Kind:** research · **Status:** **OPEN** (launched 2026-08-08). Retrains the exact C3 model on a
parquet with **common** taxa deliberately held out, to separate "unseen" from "rare".

## The confound

C3's contribution survived the scoring-rule retraction and is one of the paper's cleaner claims:
novelty detection is **monotone in taxonomic distance**, for both heads.

| stratum | plain (entropy) | ArcFace (max) |
|---|---|---|
| near — unseen species, known genus | 0.8527 | 0.8680 |
| mid — unseen genus, known family | 0.9342 | 0.9165 |
| far — unseen family | 0.9641 | 0.9444 |

But of what? C3 got its novel taxa for free, by scoring against the **unfiltered** catalogue: training
used `min_img_per_spc = 50`, so every species below that floor is unseen. That is 38,907 species and
it costs nothing — which is exactly why it was done, and why the caveat was written into
`dev/059`'s docstring at the time:

> these unseen species are the *rare* ones, so they may be systematically harder (fewer/worse images)
> than a random held-out species would be.

So "novel" and "rare" are perfectly confounded. A rare species may score as novel because the model
never saw it, or because a species with 12 photographs in a global citizen-science archive is
photographed differently from one with 800. **Nothing in C3 distinguishes those.** Worse, the
confound plausibly runs *with* the effect: far taxa are rarer than near taxa on average, so the
monotone ordering could be an image-quality gradient wearing a taxonomy costume.

## The design

`dev/072_holdout_common.py` builds a training parquet with **common** taxa removed at three ranks —
common meaning >= 200 training images, four times the 50-image floor:

| stratum | held out | scored | test images |
|---|---|---|---|
| near | 120 single species (genus keeps a sibling) | 120 species | 10,391 |
| mid | 40 whole genera (family keeps a sibling) | 83 species | 2,969 |
| far | 2 whole families | 28 species | 2,376 |

2.62 % of rows removed. Then the identical C3 recipe is retrained on it, and `dev/059` scores it
against the *unmodified* parquet: `stratify()` reads the checkpoint's vocabulary, so the held-out
taxa reappear as near/mid/far with no extra wiring.

**Two design points that took a false start each.**

*Whole taxa are removed, but only their common members are scored.* A family is only unseen if all
of it is gone — but this dataset has **no small family made of common species**. Families are either
rare singletons (the smallest has 44 images across one species) or enormous (the largest, 1,828
species and 868 k images). The first build sorted families by size and got three tiny all-rare ones:
4 scoreable species, 141 test images, and the rare-taxa confound straight back in. Holding out the
whole family while scoring only its >= 200-image members fixes it and keeps all three strata matched
on image count, which is the only thing C3b exists to do.

*`--holdout-manifest` asserts that the hold-out happened.* Point `dev/059` at a checkpoint trained on
the full parquet and every held-out species is in its vocab, `stratify` calls them all known, the
novel set is empty, and every AUROC comes back `nan` — or worse, gets computed over whatever rare
species happened to remain. The manifest path now refuses to run if more than half the scoreable
species are not novel for the given checkpoint. This project has been bitten too often by a
measurement that silently measures something else.

## Prediction (committed)

**Monotonicity survives — near < mid < far, with near above 0.80.** If the ordering is a property of
the embedding geometry and the taxonomy, rarity was never doing the work.

**But every stratum should be harder than C3's.** A held-out *common* species has hundreds of
well-photographed test images and ought to look far more like a confident known class than a
12-image rarity does. I expect `near` in the **0.78–0.85** band rather than C3's 0.853, and the
gaps between strata to narrow.

**Falsified if the ordering breaks** — any stratum out of order by more than one noise floor. That
would mean C3's monotonicity was an artefact of the rare-taxa population, and the paper's "novelty
has degrees" claim would need withdrawing rather than qualifying.

A companion closed-set eval runs on the same checkpoint. If removing 2.62 % of the data costs real
accuracy against the C3 model's 0.9148, C3b is measuring a weaker model rather than a fair hold-out,
and the AUROCs are not comparable. That check has to pass before the main result means anything.

## Why now

It is the last item on the backlog that changes what the paper can claim rather than adding a
robustness check to something already claimed, and it is the only one still gated on a full
retraining — 6.4 h, which is why it kept losing to experiments that reuse an existing checkpoint.
It runs alongside [[2026-08-08-self-training-does-not-iterate]]'s R4, which is a short chain; the
queue rule is to keep an independent job beside any chain, and C3b is that job.
