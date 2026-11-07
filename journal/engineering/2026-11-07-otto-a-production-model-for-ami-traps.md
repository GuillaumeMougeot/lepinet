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

**Where the two models are:**
- **Nemo (MAMBO_v3):** EfficientNetV2-S, on Hugging Face (`asgersvenning/MAMBO-v3`), with the
  regional lists in `bundle/regions/`.
- **Meghan (MAMBO_v2):** BioCLIP-2 (`imageomics/bioclip-2`) with a 21 MB hierarchical head, one
  head per region (`hierarchical_bioclip2_ft_{neu,eu,}_v1.pt`). Public on ERDA
  (`anon.erda.au.dk/share_redirect/HE90eyuZCT/MAMBO/`), pinned with SHA-256 in mini_trainer's
  `dev/releases/mambo_v3/inventory.toml`. Asger's `legacy_evaluation.py` runs it from its original
  source (commit 32b3cd6).

Meghan shares Otto's trunk, so Otto against Meghan compares our recipe on the same backbone.

## Step 2 and the contamination check (mt, 2026-11-07)

**BioCLIP-2 trains in mini_trainer.**
- Smoke run on 4 species: loss 3.25 to 2.06, validation accuracy 72 to 91 % in 2 epochs, and the
  checkpoint reloads.
- **One bug fixed:** the preset fed BioCLIP-2 ImageNet-normalised pixels. BioCLIP's transform
  exposes no mean and std, so the preprocessing fell back silently. This is the same handicap
  `dev/075` corrected in lepinet. MT commit ce5b7e7.

**Differences from P5 that remain:**
- MT uses the projected 768-d embedding (Meghan's convention); P5 used the 1024-d pooled feature.
- The unused text tower is saved in the checkpoint (1.7 GB). Strip it at release.

**Cost:** about 430-500 img/s on a B200, so about 16 h for 5 epochs and 32 h for 10. Otto, the
DINOv3 twin and 3 site folds come to about 80 B200 hours at 5 epochs.

**No Flemming crops in Meghan's or Nemo's training data:**
- Meghan is stock BioCLIP-2 with a head trained on the same GBIF parquet as Nemo.
- None of the Danish training rows is a machine observation.
- Visually, the most similar of the 47,913 Flemming crops to any Danish or machine-observation
  training image is a different photo of the same species (cosine 0.989 at most).
- Not checked: non-Danish images, visually.

**Decisions (lepinet-ac, 2026-11-07; the first needs the owner's confirmation, since it departs
from the pinned recipe):**
- **Optimiser: Muon at lr 1e-5, as P3c, not AdamW at 1e-3.** On this trunk 1e-3 was the worst
  setting measured (P3a 0.8912 against P3c 0.9146 in-distribution). A per-backbone optimiser is
  already the practice (DINOv3 runs use Muon).
- **224 px**, BioCLIP-2's pretraining size. At 384 px a ViT-L/14 costs about 3x, so V2-S's 384 px
  result does not transfer.
- **10 epochs.** At 384 px, 10 epochs gave V2-S +4.7 probe and +2.4 test over 5 epochs at 256
  (confounded with size). 30 epochs on a ViT-L would take about 4 days per run.
- **The 768-d embedding is accepted.** The 1024-d option would need checkpoint metadata. The
  difference is unmeasured, and Otto differs from P5 in other factors anyway (no floor, BU head, no
  oversampling).
- **Teacher for the site folds:** P5 and the DINOv3 self-trained model saw every site's adapt
  crops, unlabelled. A fold's student still sees no pixel of its held-out site, since that site's
  pseudo-labels are dropped. The leak goes only through labels on the other sites' crops, so it is
  accepted and stated in the figure caption.
- **Averaging the ensemble:** by GBIF key. Species outside P5's 12,041 get P5 probability 0.

## The teacher check (mt, 2026-11-07)

Scored on the probe crops, which no teacher saw. Species; known set = the 12,632-key union for
every teacher, so P5 pays for the 591 species it cannot predict.

| teacher | top-1, probe / probe-HO | macro-F1 calibrated by (trap, night) | macro-F1 at zero |
|---|---|---|---|
| P5 | 0.836 / 0.758 | 0.622 / 0.488 | 0.394 / 0.206 |
| DINOv3 self-trained (st26) | 0.846 / 0.745 | 0.627 / 0.431 | 0.410 / 0.195 |
| **ensemble (probabilities averaged)** | **0.864 / 0.783** | **0.670 / 0.503** | **0.452 / 0.223** |

- **The ensemble teacher stays** (rule: at least the best single teacher on both sets).
- **Prediction falsified:** "less than 2 points" was wrong. The gain is +4.3 on probe over the best
  single teacher and +1.5 on probe-HO.
- On adapt crops the ensemble ranked below st26 (0.859 against 0.869). Both teachers had trained on
  those crops, so that ranking was biased, as expected.
- Pseudo-labels: per-species top-K, K = 35; 8,305 crops over 913 species, 71.3 % correct.
- The averaged pair is itself the best trap model we have. That makes it a bar Otto, a single
  model, should reach.

## Log

- 2026-11-07: plan agreed by the owner; Tree of Life postponed. Name: Otto. Both comparison models
  located (above). Otto is queued as soon as the settings are settled and GPUs are free (owner).
