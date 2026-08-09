# C3b: is novelty detection monotone in taxonomic distance, or was C3 measuring rarity?

**Kind:** research · **Status:** **RESOLVED (2026-08-09). Monotone, and it was never rarity.** With
**common** taxa (>= 200 training images) deliberately held out, the ordering is unchanged and every
stratum is *slightly better* than C3's, not worse: near **0.8717**, mid **0.9463**, far **0.9726**.
The ordering prediction was right; the direction of the magnitudes was wrong.

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

---

## Result: the confound was not driving anything (2026-08-09)

231 deliberately held-out **common** species, 15,736 novel images against 8,000 known, plain head,
entropy rule — the same rule C3's re-score used.

| stratum | C3 (rare taxa, free) | **C3b (common taxa, held out)** | Δ |
|---|---|---|---|
| near — unseen species, known genus | 0.8527 | **0.8717** | +1.90 |
| mid — unseen genus, known family | 0.9342 | **0.9463** | +1.21 |
| far — unseen family | 0.9641 | **0.9726** | +0.85 |

**Prediction scored: the ordering held (correct), `near` cleared 0.80 (correct), but I predicted
`near` in 0.78–0.85 — *below* C3's 0.8527 — and it landed at 0.8717, above both.** So the
qualitative call was right and the directional one was wrong: holding out common taxa makes novelty
detection marginally *easier*, not harder.

The reasoning behind the wrong half was that a common held-out species has hundreds of
well-photographed test images and should therefore look more like a confident known class. That is
evidently outweighed by something else, and the mean max-cosine column suggests what: known images
sit at −0.471 and `near` at −2.297, a wide separation. A rare species in C3 is represented by a
handful of images that may be atypical *of that species* but not necessarily far from the known
manifold — whereas a well-sampled absent species occupies its own region of the embedding cleanly.
Rarity was adding noise to the novel population, not making it easier to detect.

**The claim is confirmed and strengthened.** Novelty detection being monotone in taxonomic distance
is a property of the embedding and the taxonomy, not an artefact of the rare-taxa population that
C3 got for free. It now rests on two populations chosen by opposite criteria — everything below the
50-image floor, and 231 species with at least 200 images each — which is a much better position than
the single free benchmark it had yesterday.

**The gaps also narrow slightly** (11.5 pt near→far here vs 11.1 in C3; near→mid 7.5 vs 8.2). Nothing
to read into: the two runs use different known sets and different novel populations, so only the
ordering transfers cleanly. The magnitudes should be reported as "comparable", not compared.

## The closed-set check, and why it is not yet interpretable

C3b scores **0.9110** species macro-F1 on its own test fold, against the C3 model's 0.9148 on the
full one. That looks like a −0.38 pt cost for removing 2.62 % of the data, and it is tempting to
call it acceptable and move on.

**It is not a valid comparison.** C3b's fold has 11,745 species; the full fold has 12,041. Macro-F1
does not decompose over subsets — this project has a finding about exactly this
([[2026-08-03-macro-f1-does-not-decompose]], START-HERE 4a), where the same model differed by 2.03 pt
between a set and a subset of the same images purely through the denominator. A 0.38 pt gap across a
296-species difference in denominator is well inside what that effect can manufacture.

`lepi-C3ref-eval` is running the only comparison that means anything: the **original C3 model** on
**C3b's fold**. Genus 0.9594 and family 0.9691 are reported above for completeness and carry the
same caveat.

Until that lands, the correct statement is: **the stratified AUROCs stand on their own** — they are
computed within one model and one benchmark, so the denominator problem does not touch them — and
the claim that C3b is a comparably-good model is unverified.

**Resolved (2026-08-09).** The C3 model on C3b's fold scores **0.9114**. C3b scores **0.9110**.

| | species | genus | family |
|---|---|---|---|
| C3 model, C3b's fold | 0.9114 | 0.9587 | 0.9698 |
| **C3b model, C3b's fold** | **0.9110** | **0.9594** | **0.9691** |
| Δ | **−0.04** | +0.07 | −0.07 |

**Removing 2.62 % of the training data and 243 taxa cost 0.04 pt** — nothing, at any level. The
apparent −0.38 pt against the full-fold 0.9148 was **entirely the denominator**, as suspected: 296
fewer species in the macro average, not a worse model.

So C3b is a fair hold-out and its AUROCs are interpretable. And this is the **fourth** time the
"check that both numbers mean the same thing" rule has paid in this project. It cost one 30-minute
eval to convert an unusable comparison into a clean one; the alternative was reporting a 0.38 pt
training-data cost that does not exist.
