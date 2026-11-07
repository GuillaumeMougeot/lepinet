# Otto: a production model for AMI traps

**Kind:** subproject · **Status:** OPEN (2026-11-07). Plan and predictions, written before any
candidate runs.

The owner's goal (2026-11-07): the most accurate model we can build for AMI traps across Europe
from the data we have now, trained in mini_trainer, and a comparison that colleagues will accept.
That comparison puts it next to Meghan (MAMBO_v2) and Nemo (MAMBO_v3) on trap crops none of the
three has seen. The working name is **Otto**.

Out of scope for now:
- the 50 M European AMI crops (backlog U, [[2026-10-21-self-training-on-european-traps]]), which
  come second;
- the Tree of Life run (W3), postponed by the owner the same day.

## What we already know (calibrated by trap and night, species, probe / probe-HO)

| lever | evidence |
|---|---|
| self-training on unlabelled trap crops | +22.2 probe on V2-S (0.371 to 0.593); a held-out site keeps 81 % of the gain (X2, lepinet) |
| larger backbone | DINOv3 self-trained 0.627 / 0.431 against V2-S 0.593 / 0.365; lepinet's P5 (BioCLIP-2) 0.622 / **0.497**, the best on unseen species |
| resolution and epochs | V2-S BU at 10 epochs, 384 px: +4.7 probe, +2.4 test; 30 epochs is running |
| trap-like augmentation | lepinet: +4.5 before adaptation, little after; c-v2s-trapaug is running |
| Danish real labels | T1rep 0.722 / 0.305: better on probe, worse on unseen species, because its errors land in the Danish fauna. Gated true labels 0.682 / 0.343 |
| teacher selection | a single global confidence gate hurt (R2: 55 % fewer species, -3.8 probe); per-species top-K helped (R3: +4.24) |
| regional list at inference | +5 accuracy (K1); Nemo full vocabulary 0.338 / 0.293, with the north_europe list and TTA 0.646 / 0.490 (VN) |

## The plan

1. **Settle the schedule and trap augmentation** from runs already in flight (V2-S BU at 30 epochs,
   384 px; v2s_bu_trapaug_s42).
2. **Check that BioCLIP-2 trains in mini_trainer.** The architecture loads, but P5 has never been
   reproduced there.
3. **Candidate:**
   - BioCLIP-2 ViT-L, BU head, all 12,632 species;
   - the pinned recipe, plus the winning schedule and trap augmentation if it pays;
   - self-training at 2.6 % of training rows, selected per species, with pseudo-labels from an
     ensemble teacher (P5 and the DINOv3 self-trained model, probabilities averaged);
   - a DINOv3 ConvNeXt-L twin as the fallback.
4. **Site folds:** the same recipe three times, each without one site's crops (LV, OH, SS), so
   that every Danish crop can be scored by a model that never saw its site.
5. **The comparison** below.
6. **Release** through mini_trainer's release path, with the regional lists, in int8 and fp16.

**The shipped model is the measured one.** The figure must show the model colleagues will run, so
Otto is the candidate, trained without the probe nights. A retrain on every night would be
unmeasured, and it would gain little: the pseudo-labels come from the adapt nights' pool anyway.

## The comparison with Meghan and Nemo

Rules, so that the result cannot be argued away:
- **One set, unseen by all three models.**
  - The probe nights of the reconciled Flemming export: Otto never saw them, labelled or not.
  - The whole Danish set, each crop scored by the site fold that never saw its site. This answers
    "home advantage", the objection to expect.
  - **To check:** that neither Meghan's nor Nemo's training data contains these crops.
- **Their pipeline and their figure.** Meghan and Nemo run through Asger's own evaluation
  (`dev/releases/mambo_v3`, `evaluate.py` and `legacy_evaluation.py`). The figure is his
  `comparison_charts.py`, extended by one model. Same metric (mini_metrics, all and known species),
  same regional lists (north_europe, europe, full).
- **Matched inference.** Each model uses its own preprocessing. Every model is shown with and
  without the regional list, and TTA is either on for all or off for all.
- **Speed and size as well.** A ViT-L is slower than an EfficientNetV2-S, and the figure should
  show it instead of hiding it.
- **Extra panels in our convention:** calibration grouped by (trap, night), probe-HO, and the
  site-fold score.

## Predictions (before any run)

- The Otto candidate beats P5 on probe (0.622) and matches it on probe-HO (0.497 +/- 0.02).
- With the same list and the same TTA setting, Otto beats Nemo by at least 5 points of species
  macro-F1 on probe nights, and by at least 3 on the site-fold score.
- **Falsified if** Otto is within 2 points of Nemo, or behind it, on the site-fold score. In that
  case the advantage is the home traps, not the model.
- The ensemble teacher beats a single teacher by less than 2 points. That is a guess: ensembles
  were never measured in this project.

## Log

- 2026-11-07: plan agreed by the owner; Tree of Life postponed. Name: Otto.
