# Does the trap adapt/probe split leak individual moths across nights?

**Kind:** research · **Status:** **RESOLVED (2026-10-12).** Barely. The (trap, night) grouping holds:
capture runs 23:00-03:59, so the noon boundary never cuts a night. What it cannot see, a moth that
stays on the sheet into the next night, does happen: **13 of 15,200 probe crops (0.09 %)** have a
pixel-identical twin in `adapt`, 12 of them one night apart, mostly one *Phalera bucephala*.
Removing them moves every model's probe macro-F1 by **at most 0.0003**, and the self-training
gain (+7.94) is unchanged. Also confirmed: the +7.94 compares the same probe set with the same
command, and B1 and B3rep5x differ only in the training parquet.

## Why the question (the owner's, 2026-10-12)

All of the AMI trap data was annotated by one person, and it is a timelapse: the same moth appears
in dozens of frames. If frames of one individual sit in both `adapt` (pseudo-labelled and trained
on) and `probe` (scored), the self-training gain is partly memorisation of that individual.
[[2026-08-02-the-shifted-benchmark-is-also-the-adaptation-set]] designed the split against this,
grouping by (trap, night) with nights running noon to noon. Two things were never checked: that a
night is really separated from the next, and that no individual spans two nights.

The owner also asked whether "69 to 77" mixed the full trap set with a subset. It does not, but a
reader could think it does: the abstract's "falls to 0.69" is the full trap set (47,905 crops,
B1 0.6836), and the +7.94 is 0.6912 to 0.7706, **both on `probe`**, scored with the identical
`lepinet test` command (`archive/ucloud/lepinet-B1-probe.toml`, `-B3rep5x-probe.toml`). The B1 and
B3rep5x configs differ in the parquet path alone.

## Hypothesis (before the scan)

Within a night, the twin rate is high (that is why the grouping exists). Across nights it should be
at the look-alike background: two different individuals of one species photographed similarly. A
moth staying through the day would show up as an excess at one night apart only.

## Method

`dev/088_trap_split_twins.py`: for every crop, the most similar crop of the same species and trap
(normalised cross-correlation of 32x32 greyscale thumbnails, crop areas within 15 %), against three
populations. Instance-level on purpose: a species-level embedding would call every pair of
conspecifics similar.

## Result

**Capture times:** 23:00-03:59 only. Every adapt/probe boundary between consecutive frames of a trap
is at least 20.2 h wide (523 of the 665 such boundaries are consecutive nights).

| nearest same-species, same-trap crop is in | crops | NCC > 0.95 | NCC > 0.97 |
|---|---|---|---|
| the same night, another frame | 43,219 | 35.0 % | 29.1 % |
| the next night (any split) | 25,528 | 0.24 % | **0.157 %** |
| a night >= 30 nights away (cannot be the same moth) | 38,555 | 0.23 % | 0.005 % |

At 0.97 the next-night rate is 30x the background: a real effect, and tiny. The contact sheets
confirm it. The top next-night pairs are one *Phalera bucephala* on trap LV2, 18-19 July 2023, in
the same pose on the same patch of sheet. The top >=30-night pairs are visibly different moths.

**Probe crops with a twin in `adapt`:**

| threshold | probe crops | species | nights to the twin |
|---|---|---|---|
| NCC > 0.95 | 61 (0.40 %) | 18 | 14 at one night, ~25 a year apart (look-alikes, not individuals) |
| NCC > 0.97 | **13 (0.09 %)** | 3 | 12 at one night, 1 at five |

No species has more than 8 % of its probe crops twinned (*Phalera bucephala*, 11 of 137 at 0.95).

**Re-scored without them** (species macro-F1 recomputed from `predictions.csv` on known-label
rows; this lands about 0.001 from the official metric, a label-set convention, so compare
differences within a row):

| model | all probe | minus twins > 0.95 | minus twins > 0.97 |
|---|---|---|---|
| B1 (no trap data) | 0.6923 | 0.6920 | 0.6922 |
| B3rep5x (self-training) | 0.7719 | 0.7720 | 0.7720 |
| lab12230 (real labels, 1x) | 0.7581 | 0.7580 | 0.7580 |

## What it means

* **The self-training gain is not individual-level leakage.** The leak exists, and it is two
  orders of magnitude too small to matter.
* **The within-night rate shows the grouping was necessary.** 35 % of crops have a near-identical
  twin elsewhere in their night. A random or per-snapshot split would have put those on both sides.
* **What the probe does not test:** new traps or new seasons. Adapt and probe share the same 12 traps
  and the same summers, on different nights. The paper claims adaptation to *conditions*, and that
  is what this measures. A leave-one-trap-out split would test transfer to a new site; it is not
  run.
* **The single annotator is not a leakage channel.** Their labels on `adapt` measured pseudo-label
  accuracy only; the confidence gate (`dev/065`) never reads them. A shared annotator favours
  real-label training, not self-training ([[2026-08-05-label-budget]], correction).
