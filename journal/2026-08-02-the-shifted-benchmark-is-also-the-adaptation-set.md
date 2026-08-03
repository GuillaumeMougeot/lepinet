# B3 cannot use any shifted number in RESULTS.md as its baseline

**Kind:** research · **Status:** RESOLVED as a protocol problem (2026-08-02); the corrected baselines
are running. Found while building B3's data splits, *before* running B3 — which is the only reason it
is a note rather than a retraction.

## The problem

B3 (self-training) pseudo-labels unlabelled trap images and mixes them into training. The trap
dataset has **47,905 images**. The shifted benchmark every model in `RESULTS.md` is scored on —
0.6293, 0.6445, 0.6616, 0.7101 — is **the same 47,905 images**.

So the moment B3 trains on any of them, every existing shifted number stops being a valid
comparison, and a B3 result quoted against 0.7101 would be measuring partly its own training set.
The gain could be arbitrarily large and entirely fake.

Nothing in the pipeline would have caught this. The eval would run, print a number, and the number
would be higher.

## The split, and why it is grouped the way it is

`dev/064` parses what the filenames already encode:

```
crop_TRAPNAME_LV3_IMAGENAME_20220811015000-88-snapshot_CROPNUMBER_5_UUID_<uuid>.jpg
               ^^^          ^^^^^^^^^^^^^^ ^^                      ^
               trap         timestamp      snapshot                crop
```

**Grouping is by (trap, night), not by image and not by snapshot.** Every crop of one snapshot is
the same moth; consecutive frames minutes apart on the same night are near-duplicates of each other.
Splitting at either finer level would put near-identical images on both sides — the classic way a
timelapse dataset manufactures a double-digit phantom gain. Nights run **midday to midday**, because
moths fly across midnight and a calendar date would cut one night in half.

| split | crops | groups | species | purpose |
|---|---|---|---|---|
| **adapt** | 27,230 | 1,304 | 387 | pseudo-labelled and trained on |
| **probe** | 15,200 | 588 | 373 | **the only set B3 may report on** |
| — of which held-out species | 2,455 | — | 58 | taxa `adapt` never saw at all |
| unused | 5,475 | 747 | 72 | held-out species that landed in adapt groups |

Totals: 12 traps, 316 nights, 1,960 groups, median 15 crops per group (max 304).

`dev/064` asserts the three ways this could silently break — no group spanning adapt/probe, no
held-out species present in adapt, and a non-empty held-out subset in probe — and exits non-zero
rather than writing a split that would inflate the result.

## The held-out-species subset answers the owner's specific worry

The stated risk for this direction was the model **specialising on the ~500 trap species** rather
than becoming robust. 15 % of species are withheld from `adapt` entirely, so `probe` contains 2,455
images of 58 taxa the adaptation never saw. If B3's gain appears on `probe` overall but **not** on
that subset, it specialised and did not generalise — and that is a distinction no aggregate number
can make.

## The baselines B3 must beat (landed 2026-08-03)

| model | full trap set (contaminated for B3) | **probe** (15,200) | **probe, held-out species** (2,455) |
|---|---|---|---|
| A4 -- best small architecture | 0.6616 | 0.6749 | 0.6992 |
| B1 -- best small robust | 0.6836 | 0.6912 | 0.6974 |
| B4 -- best overall | 0.7101 | **0.7006** | **0.7101** |

Two things to note before B3 runs.

**`probe` is not systematically easier or harder than the full set** -- A4 and B1 score higher on it,
B4 lower. So the split did not accidentally select a soft subset; it selected a different sample,
which is what makes it usable as an independent reference.

**Held-out species score *higher* than probe overall for all three models.** That is not a
generalisation signal -- none of these models has seen any trap data, so "held out from adapt" means
nothing to them and the 58 taxa are simply an easier subset. Measuring it beforehand is what makes it
usable: without these numbers, B3 scoring well on that subset would have looked like proof it
generalises, when the subset was easier all along. The quantity that will mean something for B3 is
the **change** in each column, not the level.

## The lesson, which is a familiar shape

The trap set has served as *the* external benchmark since July, and that role was never written down
as an exclusive one — so re-using it as adaptation data looked like using available images rather
than like burning a benchmark. The failure mode is the same as `-max_logit` and "consistent by
construction": **a thing whose role is assumed rather than recorded stops being visible as a
decision.**

`RESULTS.md` now states that the shifted column is measured on all 47,905 trap images, so the next
person to reach for them as training data sees the conflict in the same place they read the score.
