# Why does the colleague's draft score trap images so much lower than we do?

**Kind:** research · **Status:** **RESOLVED (2026-10-22).** It is mostly the metric. mini_metrics'
macro-F1 averages over every species a model *predicts*, not only the species in the truth. A
12,000-species model predicts about 1,300 species that never occur at the Danish traps. Each one
counts as a class with F1 = 0. The same predictions score **0.6923** in our convention and **0.1544**
in mini_metrics'. With the draft's macro-F1-optimal threshold they score 0.4116, at 57 % coverage.

**We did use mini_metrics' definition, until it changed.** `lepinet`'s macro-F1 was verified against
mini_metrics on 2026-07-16 and kept that definition: the average over species present in the truth.
On 2026-07-27 mini_metrics commit `1132a10` ("fixed macro f1 to work properly when some classes have
zero ground truth labels and/or predictions") added predicted-only species to the average. Scored with
mini_metrics just before that commit, B1 reads **0.6923**, exactly our number. Just after, 0.1566.
Every model predicts over all 12,041 species in both conventions; only the set of classes averaged
differs.

## The question (the owner's)

Svenning et al.'s draft (`~/Documents/hierarchical_classification.pdf`, October 2026) reports
species macro-F1 of 26.9-29.9 % on the expert-labelled AMI crops, at 41-49 % coverage. We report
0.63-0.69 for comparable models. The MAMBO v3 model card (mini_trainer,
`dev/releases/mambo_v3/`) reports 51.0 % for its EfficientNetV2-S on 52,788 of the same crops. All
three describe the same data. Before comparing any of them, check that they measure the same thing.

## Method

`dev/092_metric_conventions.py` re-scores one `predictions.csv` under each convention. The
mini_metrics settings are those of the draft's `publication/experiments/statistics/boot_metrics.py`
(`optimal=True`, `known_only=False`, threshold chosen to maximise macro-F1), at mini_metrics
`61db373`. Inputs: our probe predictions for B1 (no trap images) and B3rep5x (self-training), 15,200
crops, 368 truth species.

## Result

| convention | B1 | B3rep5x |
|---|---|---|
| A. ours: in-vocabulary rows, classes = truth species | **0.6923** | **0.7719** |
| B. plus out-of-vocabulary rows (59 rows, 2 species) | 0.6875 | 0.7664 |
| C. plus species predicted but absent from the truth | 0.1566 | 0.3797 |
| D. B and C: mini_metrics without a threshold | **0.1544** | 0.3716 |
| E. D with mini_metrics' optimal threshold | 0.4116 (coverage 0.570) | 0.6253 (coverage 0.854) |
| species predicted but absent from the truth | 1,271 | 391 |

Convention D reproduces mini_metrics' own output to four decimals.

* **Predicted-only species decide the number.** B1 predicts 1,271 species that do not occur in
  the probe set, more than three times the 368 that do. Each joins the average at F1 = 0.
* **The threshold then buys back F1 by abstaining.** Raising it removes low-confidence predictions
  of absent species, and their classes leave the average. So the optimum lies at low coverage, which
  is why the draft's coverages are 41-49 %.
* **Out-of-vocabulary truth matters little here** (0.005), because the probe set holds only 59 such
  rows.
* **Self-training improves this metric far more than ours** (0.15 to 0.37, against 0.69 to 0.77). It
  teaches the model the local fauna, so it stops predicting 880 absent species. This is the
  closed-world effect of [[2026-08-05-label-budget]], measured from the other side.

**Against the colleague's own numbers.** The draft's flat head scores 26.9 % at 41.5 % coverage. The
model card's model scores 50.8 % at 70.8 % in the same convention, and our B1 41.2 % at 57.0 %. The
card restricts predictions to a northern-European species list (`north_europe`), which removes most
predicted-only species. The draft does not say whether it used one, and its models are trained
differently. So the draft's numbers are plausible for a global model without a regional list. They
are not comparable with the card or with ours. Its raw predictions are not on this machine, so this
has not been checked directly.

**His trap set is also a different export.** The draft's AMI numbers come from
`flemming_helsing` (58,640 crops, 522 species), where 13.7 % of images carry a species key outside the
training vocabulary. [[2026-07-28-flemming-generalization]] traced most of that to GBIF key mismatch,
not to new taxa; our reconciled export (`data/flemming`, 47,905 crops, 486 species) has 0.5 %. With
`known_only=False` each mismatched species is a class the model can never get right. His
unthresholded table (`results/tables/main_comparison_flemming_zero.tex` in the Overleaf project)
gives species macro-F1 7.8-8.1 %, macro recall 51.6-56.2 % and accuracy 60.8-63.0 %. For comparison,
our B1 reaches macro recall 65.9 % and accuracy 65.7 % on the probe set.

**A hypothesis, untested.** In the draft the top-down (conditional) head wins on trap species
(29.9 %), and in ours it loses (0.6213 against 0.6503 on the full trap set). A head that constrains
the species by its predicted genus and family probably predicts fewer exotic species. That counts
for much more in convention D than in A. One run of `dev/092` on the draft's predictions would test
it.

## What it changes

* The merged paper must state the convention for every macro-F1. Both are defensible. A asks how well
  each species present is recognised. D also charges for naming species that are absent, which is
  an open-world error and matters in deployment.
* **Owner's decision (2026-10-22): the merged paper uses mini_metrics' current convention** (D, with
  and without the calibrated threshold, as in the colleague's tables). Every number of ours that enters
  the paper is re-scored from its `predictions.csv` with mini_metrics `61db373`. Conclusions resting
  on convention A must be re-checked, not carried over. Expected to move most: anything on trap
  images, and K1 (a regional checklist removes predicted-only species, so it should now help).
* Before any trap number is compared across the two drafts, both must use the same export
  (`data/flemming`, keys reconciled).
* The draft's noise estimate is a bootstrap over test images. It misses retraining noise, which is
  larger (see `RESULTS.md` §2, A5 and G3b).
