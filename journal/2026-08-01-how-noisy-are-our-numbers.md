# How noisy are our numbers? (A5, the repeat run)

**Kind:** research · **Status:** **RESOLVED (2026-08-01) for the in-distribution metric.** Species
macro-F1 is **essentially deterministic** run to run — two independently trained copies of the same
config both scored **0.9135**, identical to four decimals, despite lepinet seeding nothing. But the
coarse levels are not: family moved **0.24 pt**. The noise floor is **level-dependent**, and it
scales the way class count predicts.

## Why this had to be run

This project routinely interprets 0.2–0.4 pt differences — the single-head win (+0.25), marginal
supervision's coarse gain (+0.27/+0.39), B1's in-distribution cost (−0.36) — and had **never measured
its own run-to-run spread**. Every one of those conclusions rested on an unstated assumption that the
spread is smaller than the effect.

## Result

`20260731-080034`, an exact repeat of the baseline config (single species head + marginalisation,
effnetv2_s, 5 ep). Nothing changed; there was nothing *to* change, because **lepinet has no `seed`
key and seeds nothing** — head initialisation, the oversampling sampler and the augmentation are all
free-running, so an identical config is already an independent draw.

| level | classes | original (`20260729-182718`) | repeat (`20260731-080034`) | **spread** |
|---|---|---|---|---|
| species | 12,041 | 0.9135 | 0.9135 | **0.0000** |
| genus | 4,333 | 0.9606 | 0.9601 | 0.0005 |
| family | 102 | 0.9739 | 0.9763 | **0.0024** |
| species micro-acc | — | 0.9344 | 0.9342 | 0.0002 |

## The pattern: macro-F1 noise scales inversely with class count

Species is 25× noisier than family in *class count* and about 25× **quieter** in variance. That is
not a coincidence — it is what averaging does.

Macro-F1 is a mean of per-class F1 scores. Each class's F1 is itself a noisy quantity (a handful of
borderline images flip), and averaging $C$ of them shrinks the mean's variance by roughly $C$. With
$C = 12{,}041$ the per-class noise is averaged into nothing; with $C = 102$ each family carries ~1 %
of the metric, so a few flipped images in one family move the headline.

**This inverts the intuition the project has been running on.** Species macro-F1 *looks* like the
fragile number — 12,041 classes, a long tail, tiny per-class support — and it is in fact the most
stable thing we measure. Family macro-F1 looks robust (102 easy classes, 98 % accurate) and is the
noisiest.

## What this retroactively licenses, and what it revokes

Applying the measured spread as a believability threshold:

| claim | effect | spread at that level | verdict |
|---|---|---|---|
| single head > multi-head (species) | +0.25 | 0.0000 | **safe** |
| ArcFace × z-score interference (species) | −1.00 | 0.0000 | **safe** |
| ArcFace interference (genus/family) | −1.15 / −1.11 | 0.0005 / 0.0024 | **safe** |
| B1 in-distribution cost | −0.36 | 0.0000 | **safe** |
| B1 shifted gain | +3.99 | *unmeasured* | see below |
| marginal supervision, genus | +0.27 | 0.0005 | **safe** (5× the spread) |
| marginal supervision, family | +0.39 | **0.0024** | **weakened** — only 1.6× the spread |

So one claim gets downgraded. **Marginal supervision's family gain (+0.39 pt) is not comfortably
outside noise**; its genus gain (+0.27) is. The headline of that entry — "species exactly unchanged,
coarse levels improve" — survives on genus, and family should be reported as suggestive rather than
established. Corrected in [[2026-07-30-marginal-supervision]].

Nothing else moves. Every other conclusion in `RESULTS.md` clears its level's floor by 4× or more,
which is a better outcome than this project had any right to expect.

## The gap this exposes: nobody has measured the *shifted* metric's noise

The shifted benchmark has **486 species**, not 12,041 — 25× fewer classes, the same ratio that makes
family 25× noisier than species. By the argument above its macro-F1 should be **substantially
noisier than anything measured here**, and yet it now carries the project's most important claims:
B1's +3.99, B4's +4.85, and the whole ranking inversion.

Those effects are large and probably survive. But "probably" is exactly the word A5 was run to
eliminate, and leaving it standing on the *other* benchmark would be a strange place to stop. Two
cheap evals close it: score both baseline copies (`20260729-182718` and A5's `20260731-080034`) on
the shifted set and read off the difference. Queued as `lepi-base-shift` and `lepi-A5-shift`.

## Caveats, because n = 2

One repeat gives **one difference**, not a standard deviation. The honest reading is an
order-of-magnitude estimate: species noise is $O(10^{-4})$ or below, family noise is $O(10^{-3})$.
A 0.0000 species spread is a *sample*, not a proof of determinism — a third run could show 0.0008
and nothing here would be wrong. What it does rule out is species noise of the size this project has
been treating as significant (0.2–0.4 pt); that is now excluded by more than three orders of
magnitude.

## Practical consequences

**Adopted as policy:** report deltas against the level's floor. Species and genus differences above
~0.05 pt are real; **family differences below ~0.25 pt are not reportable**, and neither is anything
on the shifted benchmark until `lepi-base-shift` / `lepi-A5-shift` land.

**A `seed` key is worth adding** — not for the noise floor (free-running is what made this
measurement possible) but for reproducing a specific run. It is a small change to
`lepinet.config.TrainConfig` plus a `set_seed` call, and its absence should be documented rather
than discovered.
