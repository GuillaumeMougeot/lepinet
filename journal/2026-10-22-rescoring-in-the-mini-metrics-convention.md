# What do our results say in mini_metrics' current convention?

**Kind:** research · **Status:** **RESOLVED (2026-10-23).** In-distribution numbers do not move. On trap images, comparisons within one label space survive (long tail, heads, self-training's gain, now +21.7 points). Anything that confines predictions to the local fauna gains 10-35 points: real labels lead self-training by +25.1 zero / +10.2 calibrated, and the regional checklist flips from -1.4 to +35.4 / +13.4. The dose optimum on probe disappears; it survives on held-out species only.

## Why

The owner decided (2026-10-22) that the merged paper with Svenning et al. reports macro-F1 the way
mini_metrics computes it today: the average runs over every species in the truth **or** in the
predictions (predicted-only species score F1 = 0), out-of-vocabulary rows are kept
(`known_only=False`), both without a threshold ("zero") and with the macro-F1-optimal threshold
fitted on a random 10 % calibration split ("calibrated"). Our numbers average over truth species
only, which is what mini_metrics did until commit `1132a10` (2026-07-27)
([[2026-10-22-why-the-trap-scores-of-the-two-drafts-differ]]). So every number the paper uses is
re-scored from its `predictions.csv`, and every conclusion resting on the old convention is
re-checked, not carried over.

## Method

`dev/093_rescore_mini_metrics.py`, mini_metrics `61db373` (exported to
`data/tools/mini_metrics-61db373`). The calibrated score is the mean (and sd) over 20 seeds of the
calibration split; its settings are those of the colleague's `boot_metrics.py` (`optimal=True`,
`opt_crit=MacroF1`, `use_quantiles=True`, `eps=5e-2`). The held-out-species column filters the probe
rows to the held-out species *before* scoring. That changes the class set as well as the rows, so in
this convention it is a different quantity from the old one, not just a subset of the probe score.

## Predictions (before any rescoring)

* **P1. Trap-image numbers drop a lot, in-distribution numbers barely move.** On trap images a
  12,041-species model predicts hundreds of absent species. In the held-out fold nearly every species
  is present, so predicted-only classes are few. Expect in-distribution within ~1 point, zero-threshold
  probe scores at a quarter to a half of the old values.
* **P2. Self-training's relative gain grows.** It teaches the model the local fauna, so it predicts
  far fewer absent species (B1 1,271, B3rep5x 391). Expect the B1 to B3rep5x gap to widen in absolute
  points as well as relatively, in both zero and calibrated.
* **P3. Replicated real labels (T1rep) gain most.** They closed the world hardest (91 % of held-out
  species' errors land in the labelled fauna), so they predict the fewest absent species. Expect
  T1rep to lead B3rep5x by more than the old +1.01 on probe.
* **P4. The held-out-species column changes meaning and may reorder.** Its class set now includes
  every species predicted on those 2,455 crops. Models that push unseen species into the labelled
  fauna get those predictions counted as predicted-only classes (F1 = 0), so the closed-world penalty
  of T1rep should show up even more strongly here.
* **P5. K1 flips.** A regional checklist removes predicted-only species by construction. In the old
  convention it hurt macro-F1 when the scored set is most of the checklist; in this one it should help
  by a large margin.
* **P6. In-distribution rankings (heads, long tail, capacity) do not change**, by P1.

## Result (trap images, 2026-10-22)

All 66 probe runs, the same 66 restricted to the held-out species, and the 16 full-trap runs on
UCloud were re-scored (`data/rescore/{probe,probe-ho,trap}.csv`; species level unless stated).
"old" is our convention, "zero" mini_metrics without a threshold, "cal" the calibrated threshold
(mean of 20 calibration splits; the split sd is 0.003-0.014 on probe, 0.015-0.05 on held-out species,
because the 10 % calibration split of 2,455 crops holds ~245 images). `X1`/`X2` are the pooled
hold-out folds rebuilt from their 15 fold predictions.

**Self-training and real labels (probe; probe-HO in brackets):**

| run | old | zero | cal | cal coverage |
|---|---|---|---|---|
| B1, no trap images | 0.692 (0.697) | 0.154 (0.098) | 0.407 (0.356) | 0.58 |
| B3rep1x, 1x | 0.736 (0.751) | 0.212 (0.126) | 0.518 (0.383) | 0.75 |
| **B3rep5x, 11x** | 0.772 (0.770) | 0.372 (0.166) | 0.621 (0.395) | 0.86 |
| B3, 27x | 0.738 (0.723) | 0.388 (0.154) | 0.615 (0.344) | 0.86 |
| B3rep26x, 58x | 0.718 (0.707) | 0.401 (0.158) | 0.607 (0.300) | 0.86 |
| lab500 / lab2500 / lab12230, 1x | 0.707 / 0.721 / 0.758 | 0.165 / 0.186 / 0.326 | 0.476 / 0.468 / 0.593 | 0.65 / 0.75 / 0.88 |
| **T1rep, real labels 11x** | 0.781 (0.701) | **0.623** (0.198) | **0.723** (0.320) | 0.94 |
| X1 pooled / X2 pooled | 0.743 / 0.711 | 0.333 / 0.300 | 0.615 / 0.596 | 0.84 / 0.83 |
| B8 / P5 / P5b | 0.782 / 0.783 / 0.773 | 0.311 / 0.396 / 0.389 | 0.641 / 0.616 / 0.605 | 0.84 / 0.84 / 0.84 |
| K1 checklist (B8, 486 species allowed) | 0.768 (0.845) | **0.665** (0.252) | **0.775** (0.480) | 0.90 |

**Heads and long tail (full trap set, 47,905 crops):**

| run | old | zero | cal |
|---|---|---|---|
| multi-head independent | 0.652 | 0.129 | 0.344 |
| single head + marginal supervision (marginal / its repeat M2) | 0.646 / 0.651 | 0.117 / 0.122 | 0.308 / 0.317 |
| single head + marginal inference (singlehead) | 0.627 | 0.113 | 0.279 |
| parent-conditioned | 0.623 | 0.094 | 0.270 |
| L0 no oversampling / singlehead √ / L1 balanced softmax / L2 both | 0.647 / 0.627 / 0.574 / 0.551 | 0.134 / 0.113 / 0.092 / 0.081 | 0.344 / 0.279 / 0.223 / 0.181 |
| A1 (+ margin) / B1 (+ trap augmentation) / A2 (198 M) / B4 (198 M + aug) | 0.646 / 0.686 / 0.662 / 0.712 | 0.098 / 0.112 / 0.091 / 0.103 | 0.361 / 0.403 / 0.394 / 0.433 |

**Retraining noise in the new convention** (identical repeats): singlehead vs A5 on probe, zero
0.145 / 0.145, cal 0.280 / 0.293; G3 vs G3b 0.346 / 0.354, 0.613 / 0.612; R5 vs R5b 0.390 / 0.376,
0.566 / 0.562; P5 vs P5b 0.396 / 0.389, 0.616 / 0.605. So about 1 point in either column, larger
than the calibration-split sd.

**Predictions scored (trap images):**

* **P1 (trap numbers drop a lot): holds.** Zero-threshold probe scores are 0.15-0.40 for models
  without the local fauna, a quarter to a half of the old values. Calibration recovers part of it at
  57-94 % coverage.
* **P2 (self-training's gain grows): holds, strongly.** B1 to B3rep5x is +8.0 points in the old
  convention, **+21.7 zero and +21.4 calibrated**.
* **P3 (T1rep gains most): holds, strongly.** T1rep leads B3rep5x by +1.0 old, **+25.1 zero and
  +10.2 calibrated**: replicated real labels stop the model naming absent species.
* **P4 (held-out-species column reorders): partly.** Calibrated, the closed-world penalty of T1rep
  remains (0.320 vs 0.395). Zero, it reverses (0.198 vs 0.166), because T1rep's mistakes go to a few
  labelled species instead of many absent ones. The column's verdict now depends on the threshold.
* **P5 (K1 flips): holds.** The checklist costs 1.4 points old and gains **+35.4 zero, +13.4
  calibrated**. The old conclusion, "a regional checklist hurts per-species macro-F1", does not
  survive the convention change.
* **P6 (in-distribution rankings): pending** (in-distribution set below).

**What survives and what does not, on trap images:**

* *Survives.* The long-tail ranking: L0 > √ > L1 > L2 holds in all three columns. The parent-
  conditioned head is last in all three, so the colleague's opposite result is not a metric effect
  on our data ([[2026-10-22-why-the-trap-scores-of-the-two-drafts-differ]]'s hypothesis is
  falsified here). Self-training transfers to a held-out trap (X1 cal 0.615 vs B3rep5x 0.621, X2
  0.596), more than the old convention said (64 % and 23 % of the gain kept become 97 % and 88 %
  calibrated, 82 % and 67 % zero).
* *Does not survive as stated.* **The interior optimum in the self-training share.** On probe, zero
  rises monotonically with replication (0.212, 0.372, 0.388, 0.401) and calibrated is flat beyond
  11x (0.621, 0.615, 0.607). Only the held-out-species column still falls with the share (cal 0.383,
  0.395, 0.344, 0.300), so "beyond 2.6 % adaptation turns into memorisation" holds for unseen species
  only. "Real labels give up the gain on unseen species" holds calibrated, not zero. "A regional
  checklist hurts" reverses. The margin (A1 vs singlehead) is +8.3 calibrated but −1.5 zero.

Rows of `RESULTS.md` §2 whose numbers change (all probe, probe-HO and full-trap columns; no row
rewritten yet): singlehead, marginal, M2 (external); A1, A2, A4, A5; B1, B3, B3rep1x/5x/26x, B4,
B6-B10; F1-F3; G2, G3, G3b; L0-L2, L4, L5, L7 (cap 250/500/1000, c1000b); R2-R5b; T1 (lab500,
lab2500, lab12230), T1rep, T2, T2b; X1, X2; P1b, P3c, P4, P4b, P5, P5b; K1; D2. §1 findings 4, 5
(cap part), 9 and 14 are affected.

## Result (in-distribution, 2026-10-23)

53 runs (`data/rescore/indist.csv`, 5 calibration seeds; peak memory 3.9 GB per process, one process
at a time). One file had been truncated by an interrupted download (A5, 8 MiB of 172 MB) and was
fetched again; the downloader's skip test only checks that the file exists.

* **On the full held-out fold, "zero" equals our number to four decimals in every run**: all 12,041
  species are in the truth, so no class is predicted-only, and there are no out-of-vocabulary rows.
* **On the decontaminated fold (217,856 rows) zero is 0.5-1.3 points lower**, because some species
  have no test image there and are still predicted. The BioCLIP-2 comparison keeps its sign and size:
  P3c minus our baseline is +1.25 old, +1.32 zero, +1.34 calibrated.
* **Calibrated moves every run by under ±0.6 points.** Two systematic shifts: raw-confidence B8-family
  runs (B7, B8, B10, L5) lose ~0.5, because their margin-head softmax saturates and the threshold
  cannot separate tied confidences; the staged 198 M runs (G1-G3b) gain ~0.3.

| run | old = zero | calibrated (coverage) |
|---|---|---|
| single head + marginal inference / + marginal supervision | 0.9135 / 0.9135 | 0.9117 / 0.9104 (0.94) |
| multi-head independent / parent-conditioned | 0.9110 / 0.8845 | 0.9076 / 0.8845 |
| L0 / √-oversampling / L1 / L2 | 0.8949 / 0.9135 / 0.8970 / 0.8689 | 0.8828 / 0.9117 / 0.9028 / 0.8877 |
| L4 (cRT) | 0.9068 | 0.9045 |
| cap 250 / 500 / 1,000 | 0.8783 / 0.8955 / 0.9060 | 0.8873 / 0.8983 / 0.9043 |
| A1 (margin) / B8 / P5 (clean fold) | 0.9035 / 0.9060 / 0.9113 | 0.9067 / 0.9005 / 0.8975 |

The folder `ucloud_preds/lepinet-heads-global-independent-muon-5ep-oversample-effnetv2s-ucloud`
holds a 484,299-row evaluation (old 0.9455, zero 0.5237) that matches no number in the paper; it is a
different evaluation set and is left out.

**P6 (in-distribution rankings do not change): holds.** Heads keep their order at species, genus and
family in all three columns. The long-tail cells still do not order in-distribution (calibrated
even moves L2 above L0), which is the original claim. cRT recovers more of oversampling's
in-distribution gain calibrated (75 %) than old (64 %). The cap sweep still rises monotonically.

## Answer

In mini_metrics' current convention, in-distribution results are unchanged, and the trap-image
conclusions split in two. The ones that compare models trained on the same label space survive: the
long-tail inversion, the heads ordering, self-training's gain (now three times larger), transfer to a
new trap or site (now mostly kept). The ones that depend on *which species a model names* change:
real labels and a regional checklist, which confine predictions to the local fauna, gain 10-35 points
and overtake self-training; and the dose optimum on probe disappears, surviving only on held-out
species. The merged paper must present §4.11's dose and label-budget claims, and K1, in the new
form, and state that the calibrated column fits its threshold on a random, not trap-and-night,
split.
