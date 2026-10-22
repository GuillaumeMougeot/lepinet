# What do our results say in mini_metrics' current convention?

**Kind:** research · **Status:** **OPEN** (2026-10-22). Predictions written before any rescoring.

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

## Result

Pending.
