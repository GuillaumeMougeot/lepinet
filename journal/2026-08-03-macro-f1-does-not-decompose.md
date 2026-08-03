# F1 ties B4 on the full trap set and beats it by 2 pt on a subset of that same set

**Kind:** research · **Status:** RESOLVED (2026-08-03). Both numbers are correct and both are real —
**macro-F1 over different species subsets is not comparable across subsets**, so the full-set and
`probe` columns answer different questions and must never be read against each other. This corrects
how the morning's F1 result was stated, and tightens B3's falsification line.

## The apparent contradiction

| model | full set (47,905 / 486 sp.) | **probe** (15,200 / 368 sp.) | **probe held-out sp.** (2,455 / 58 sp.) |
|---|---|---|---|
| A4 (20 M) | 0.6616 | 0.6749 | 0.6992 |
| B1 (20 M) | 0.6836 | 0.6912 | 0.6974 |
| B4 (198 M) | 0.7101 | 0.7006 | 0.7101 |
| **F1 (198 M)** | 0.7103 | **0.7209** | **0.7559** |
| **F1 − B4** | **+0.0002** | **+0.0203** | **+0.0458** |

This morning's entry ([[2026-08-02-f1-flagship]]) concluded from the full-set column that *"F1 = B4 on
species and shift"* and that marginal supervision's robustness benefit does not survive scale. On
`probe` — a subset of the very same images — F1 wins by 2 pt.

## It is not noise

The full-set floor (0.0069) was measured on the full set and does not transfer to a smaller one, so
`probe` and `probe_heldout_species` floors were measured the same way: two independently trained
copies of the identical baseline config, scored on each.

| benchmark | draw 1 | draw 2 | **floor** |
|---|---|---|---|
| probe | 0.6250 | 0.6291 | **0.0041** |
| probe held-out species | 0.6386 | 0.6438 | **0.0052** |

So F1 − B4 is **5.0×** the floor on probe and **8.8×** on the held-out subset. Real on both.

Worth noting the floors came out *smaller* than the full set's 0.0069 rather than larger, which the
class-count argument had predicted. That argument still holds — it is just weak here: 368 species vs
486 is a 1.3× difference, not the 25× that made family macro-F1 noisy
([[2026-08-01-how-noisy-are-our-numbers]]). With one pair per benchmark these are all
order-of-magnitude estimates and all land at ~0.005.

## The resolution: macro-F1 does not decompose over subsets

Macro-F1 is a mean of per-class F1 scores, computed **within** whatever set it is given. The full set
has 486 species; probe has the 368 that appear in its images. So:

- a species' F1 on `probe` is computed from probe images only, not restricted from its full-set value;
- the 118 species absent from probe carry **zero** weight there and **1/486 each** in the full column;
- and every species' weight changes from 1/486 to 1/368.

A model better on the 368 probe species and worse on the 118 others therefore ties on the full set
and wins on probe. That is arithmetic, not a paradox — but the two columns are **not** a big and a
small measurement of the same thing, and reading one as a check on the other is a mistake.

## What changes

**The morning's claim is narrowed, not withdrawn.** "Marginal supervision's robustness benefit does
not survive scale" was measured on the full trap set and remains true *there*. On probe it does
survive, and substantially. The honest statement is that **the two benchmarks disagree about
marginal supervision at 198 M**, and the reason is which species each one weights — which is a fact
about the benchmarks, not about the model. Neither is more "correct"; they answer different
questions, and `probe` is the one that matters for anything trained on trap data.

**F1 is now clearly the model to ship**, and for a better reason than this morning's. It equals B4
in-distribution, beats it by 2.03 pt on held-out trap groups, and by 4.58 pt on trap species it has
never seen — the last being the closest thing this project has to a measure of genuine
generalisation.

**B3's falsification line moves.** It was B1's probe 0.6912 plus the *full-set* floor of 0.0069,
giving 0.6981. With probe's own floor it is **0.6912 + 0.0041 = 0.6953**. Corrected before B3 lands,
which is the only time such a correction is worth anything.

**Adopted:** every shifted number is now reported with the benchmark named — `full`, `probe`, or
`probe-heldout` — and differences are only ever taken *within* a column. `RESULTS.md` carries the
three floors so the rule is enforceable rather than remembered.
