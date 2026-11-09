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
- **Matched inference.** Each model uses its own preprocessing. Every model is scored in four
  settings: full vocabulary or the north_europe list, each with and without TTA.
  - **The headline against Nemo is north_europe with TTA**, Nemo's card setting (Nemo 0.646 / 0.490
    probe / probe-HO, RESULTS VN). Owner, 2026-11-07.
  - TTA is Nemo's default profile, `rotation30_pad25_3` (mini_trainer's
    `deployment/mambo_deploy/augmentation.py`), applied unchanged to Otto, P5 and Meghan. Not
    re-tuned for Otto.
  - Nemo's list and TTA were chosen on these crops (VN), so that setting favours Nemo slightly.
  - Also reported for every model, Otto included: **tta3bm**, the same three views with a border-mean
    fill instead of edge padding (same cost). It is the best fixed policy for all three models scored
    so far (log, 2026-11-07 late), so Nemo with tta3bm is the fair-best line next to its card setting.
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

## Trap augmentation: off (2026-11-07)

v2s_bu_trapaug_s42 (MT e32c25a, `domain_aug: trap`, one draw per batch) against the V2-S BU seeds.
Species macro-F1; traps calibrated by (trap, night).

| run | test | trap | probe | probe-HO |
|---|---|---|---|---|
| v2s_bu, seeds 42 / 43 / 44 | 0.8877 / 0.8866 / 0.8872 | 0.339 / 0.338 / 0.316 | 0.371 / 0.378 / 0.349 | 0.306 / 0.306 / 0.292 |
| v2s_bu_trapaug_s42 | **0.8802** | 0.361 | 0.405 | 0.321 |

- The rule, written before the result: on if probe and probe-HO are both at least v2s_bu_s42's and
  the test fold is within 0.3 points. Off otherwise.
- **Off.** The shift legs pass (+3.4 probe, +1.5 probe-HO), but the test fold loses 0.75 points,
  about 7 times the seed spread (0.11). So the cost on the test fold is real.
- The gain under shift is before adaptation. lepinet found little left after self-training, and Otto
  has the ensemble pseudo-labels. Trap augmentation stays opt-in in mini_trainer.

## TTA: what it fixes, and a cheaper way to get it (opened 2026-11-07, owner)

**What we know.**
- Nemo on our probe crops (VN), species macro-F1 calibrated by (trap, night): TTA adds +14 points on
  the full vocabulary (0.338 to 0.479) and +10 with the north_europe list (0.546 to 0.646).
- Asger's policy comparison on his trap crops (`docs/assets/mambo-composed-tta.json`, 52,788
  images, accuracy / macro-F1 at threshold 0): one view 0.713 / 0.259; 3 views of rotation +-30
  and padding 0.749 / 0.320; 5 views 0.755 / 0.322. The fourth and fifth views add almost nothing.
- On iNaturalist photos, the training domain, TTA adds nothing (macro-F1 0.834 against 0.833) and
  costs 3x (91 against 279 ms per image on CPU; Nemo's card).

**Reading.** TTA here is not a general accuracy trick. It corrects two differences between GBIF
photos and trap crops: framing (a trap crop is a tight box; padding zooms out) and orientation (a
moth on a trap sheet sits at any angle). It pays for that correction 3 times per image.

**Hypotheses, before measurement.**
- Self-training on trap crops teaches part of the framing, so TTA's gain on Otto is smaller than on
  Nemo: +3 to +5 probe, against Nemo's +10 with the list.
- One padded view (no rotation) recovers about half of the 3-view gain at 1x cost.
- Training with geometric augmentation (rotation and zoom-out, missing from today's trap
  augmentation, which is photometric only) removes most of the TTA gain at no inference cost.

**Plan, evaluation only (no training).**
- One forward pass per distinct view, logits saved: original, hflip, pad 0.08 / 0.15 / 0.25,
  rotations +-10 / +-30 with padding, the d4 views. Every policy is then an average of saved logits,
  so any combination costs no GPU.
- Models: Otto, the DINOv3 twin, P5, Nemo.
- Policies are chosen on the adapt nights and reported on probe and probe-HO, because choosing on
  probe crops would favour whichever model was tuned there. Nemo's own default is reported as is.
- Report per policy: probe / probe-HO, and time per image on CPU (int8) and GPU (fp16). The
  figure is F1 against time, so a policy is judged by what it gains per millisecond.

**Training, later (one factor, needs a run).** Geometric trap augmentation on V2-S BU, 5 epochs at
256 px: random rotation +-45 and zoom-out padding 0-25 %. Read: single-view probe up, and TTA's gain
down. The second option is TTA distillation: pseudo-labels from the teacher with TTA, so that the
student learns TTA's output at 1x cost.

## Log

- 2026-11-07: plan agreed by the owner; Tree of Life postponed. Name: Otto. Both comparison models
  located (above). Otto is queued as soon as the settings are settled and GPUs are free (owner).
- 2026-11-07: trap augmentation off. Submitted by mt (owner-approved there): c-otto-bioclip 12450502
  (4 B200: otto_bioclip_bu_s42 and the site folds x2lv, x2oh, x2ss) and c-otto-dinov3 12450503
  (1 B200: otto_dinov3_bu_s42, 320 from 384, bs 96, Muon 8e-4, head off Muon). Pseudo-labels at
  `target_frac` 0.02 of all parquet rows, i.e. x16, 2.6 % of training rows.
- 2026-11-07 16:20: c-otto-bioclip (12450502) waited 2 h 20 min for 4 B200 at once. Split into four
  1-B200 jobs with identical specs (owner-approved in mt): candidate 12450599 and x2lv 12450600
  RUNNING; x2oh 12450601 and x2ss 12450602 queued. 384x30 resume 12450604 queued. TTA tool ready
  (17 views, per-view logits); Nemo reproduces VN through it. Nemo on the adapt nights, calibrated
  species F1: full 0.326, + TTA 0.492, north_europe 0.556, both 0.614.
- 2026-11-07: AG submitted (owner-approved in mt; MT 988b29f): ag-otto-bioclip-geo 12450608,
  ag-v2s-384x10-geo 12450609, -geotrap 12450610, -trivial 12450611, each its control plus one key.
  Augmentation grid checked by eye: geometric as intended; TrivialAugmentWide includes strong colour
  ops (solarize, posterize, inversion). Meghan scored (species calibrated, probe / probe-HO): full
  0.321 / 0.248, + TTA 0.239 / 0.170, north_europe 0.509 / 0.403, both 0.402 / 0.322. TTA *costs*
  Meghan about 8 points with argmax nearly unchanged (zero-threshold 0.131 against 0.134); being
  checked (probability or logit averaging, confidence flattening) before it is reported. Logits are
  stored in fp32: fp16 moved calibrated F1 by up to 7e-4 at species, 3e-3 at genus.
- 2026-11-07 evening: **Meghan's TTA loss is real, and comes from the views.** The scoring path is
  the same for every model. Each of Nemo's rotated views alone drops Meghan's probe top-1 from 0.645
  to 0.16-0.17, while Nemo rises to 0.76. Rotation followed by *edge* padding replicates the border
  pixels where the rotated corners touch the canvas, which draws long stripes. Meghan breaks on them;
  Nemo tolerates them. Rotation with a plain fill does not hurt Meghan (white 0.701, grey 0.625).
  My earlier note "argmax nearly unchanged" was mt's first reading, since corrected: agreement is
  65 %. Consequence: the AG geometric augmentation trained on the same striped views. Decided: a
  constant fill per image (the mean colour of its outer border, i.e. more trap sheet), and the three
  geometric AG jobs are resubmitted before they start. A matched TTA policy with that fill (tta3bm)
  joins the study. Nemo's own tta3 stays its card setting.
- 2026-11-07 night: **TTA, first results** (17 views, north_europe list, species calibrated by
  (trap, night), probe / probe-HO):

  | policy | views | Nemo | P5 |
  |---|---|---|---|
  | none | 1 | 0.546 / 0.448 | **0.679 / 0.500** |
  | pad15 | 1 | 0.599 / 0.472 | 0.691 / 0.536 |
  | pad3 (none, pad 0.08, pad 0.15) | 3 | 0.613 / 0.485 | **0.709** / 0.532 |
  | rot30p15_3 | 3 | 0.637 / 0.497 | 0.683 / 0.507 |
  | tta3 (Nemo's card) | 3 | **0.645** / 0.491 | 0.663 / 0.486 |
  | d4 | 8 | 0.563 / 0.468 | 0.687 / 0.500 |
  | all17 | 17 | 0.628 / 0.496 | 0.707 / 0.544 |

  - **P5 with the list and one view (0.679 / 0.500) already beats Nemo's card setting** (0.645 /
    0.491), at a third of Nemo's TTA cost in views.
  - One padded view gives Nemo about half its TTA gain (+5.3 of +9.9), as predicted.
  - Nemo's tta3 hurts P5 (-1.6), as it hurt Meghan: both are BioCLIP-2, and both break on
    edge-padding stripes. tta3bm (border-mean fill) is being scored.
  - Quarter turns do not pay (d4).
  - Selection rule changed: adapt nights are biased for any model whose pseudo-labels came from them
    (P5, Otto, the twin). Every model is reported on a fixed set of policies without selection. A
    recommended policy is chosen on half of the probe nights and reported on the other half.
- AG jobs failed during image staging (all four, 12.5 min in; suspected full node disk), resubmitted
  with the border-mean fill: 12450732 (Otto geo), 12450733-35 (V2-S geo, geotrap, trivial).
- 2026-11-07 late: **TTA, fixed policy set** (mt; species macro-F1 calibrated by (trap, night); genus
  and family in `/data/au761367/otto_eval/fixed_table.json`). North_europe list, probe / probe-HO:

  | policy | views | Nemo | P5 | Meghan |
  |---|---|---|---|---|
  | none | 1 | 0.546 / 0.448 | 0.679 / 0.500 | 0.509 / 0.403 |
  | pad15 | 1 | 0.599 / 0.472 | 0.691 / 0.537 | 0.483 / 0.368 |
  | pad3 | 3 | 0.613 / 0.485 | 0.709 / 0.532 | 0.508 / 0.375 |
  | tta3 (Nemo's card) | 3 | 0.646 / 0.491 | 0.663 / 0.486 | 0.402 / 0.322 |
  | **tta3bm** (border-mean fill) | 3 | **0.660 / 0.519** | **0.711 / 0.546** | **0.549** / 0.402 |
  | all17 | 17 | 0.628 / 0.496 | 0.707 / 0.544 | 0.544 / 0.421 |

  Full vocabulary, tta3bm: Nemo 0.489 / 0.371 (tta3 0.478 / 0.358), P5 0.660 / 0.519 (none 0.622 /
  0.488), Meghan 0.408 / 0.313 (none 0.321 / 0.248).

  - **The stripes were costing every model, Nemo included.** Changing only the fill (same three
    views, same cost) gives Nemo +1.4 / +2.8, P5 +4.8 / +6.0 and Meghan +14.7 / +8.0 over tta3. For
    Nemo this is a free improvement to its released default; worth passing to Asger.
  - **TTA is now positive for all three models.** With tta3bm, Meghan gains +4.0 probe over no TTA
    instead of losing 10.7.
  - **P5 without any list equals Nemo with the list:** both 0.660 / 0.519 at tta3bm.
  - At the same policy (north_europe, tta3bm), P5 leads Nemo by +5.1 / +2.7.
  - **Choose on half A, report on half B** (seeded halves stratified by trap, split by (trap, night):
    A 293 groups / 7,983 crops, B 295 / 7,217). North_europe: Nemo picks tta3bm, B 0.676 / 0.527;
    P5 picks all17, B 0.724 / 0.577; Meghan picks tta3bm, B 0.571 / 0.420. Full vocabulary: Nemo
    tta3bm 0.530 / 0.431, P5 all17 0.675 / 0.522, Meghan tta3bm 0.440 / 0.347. Halves are scored
    over fewer species than the full probe, so compare halves only with halves.
  - **P5's all17 pick is a tie, not a preference:** on A, all17, pad3 and tta3bm are within 0.003
    (0.004 full). all17 costs 17 forward passes, tta3bm 3. The owner's objective is F1 per unit of
    prediction time, so the cost decides a tie. **Decided: tta3bm is the one deployment policy
    for all models,** never more than 0.004 below the best on A for any model. The rule's literal pick
    (all17 for P5) is reported too. Picking on A-HO instead would move Nemo and Meghan to all17.
  - Hypothesis "one padded view gives about half the 3-view gain": held for Nemo (pad15 +5.3 of
    tta3's +9.9). The other two (Otto's smaller gain, geometric training removing most of it) wait
    for Otto and the AG runs.
- 2026-11-08: the tta3bm decision is for trap crops. On iNat21 photos it costs 1.2 to 3.9 points
  (public benchmarks entry), as Nemo's card found for its TTA. The deployment default is per
  domain: tta3bm on trap crops, single view on photos.
- 2026-11-09: **Otto scored; it is below P5.** Both through one scorer (the TTA study's path reproduces the
  campaign autoscore on Otto to 0.0005; argmax agrees on 99.92 % of crops). Species calibrated, probe /
  probe-HO:

  | setting | Otto | P5 | Nemo |
  |---|---|---|---|
  | full, single view | 0.588 / 0.450 | 0.622 / 0.488 | 0.338 / 0.293 |
  | full, tta3bm | 0.633 / 0.495 | 0.661 / 0.519 | 0.489 / 0.371 |
  | north_europe, single view | 0.661 / 0.500 | 0.679 / 0.500 | 0.546 / 0.448 |
  | north_europe, tta3bm | 0.688 / 0.540 | 0.711 / 0.546 | 0.660 / 0.519 |
  | north_europe, Nemo's card TTA | | | 0.646 / 0.491 |

  - **Headline against Nemo** (north_europe with TTA): Otto +4.2 / +4.9 over Nemo's card setting, +2.8 / +2.1
    over Nemo with tta3bm.
  - **Prediction scored:** probe 0.70-0.75 and probe-HO 0.50-0.53, north_europe with TTA: probe falsified
    (0.688), probe-HO held (0.540).
  - **Otto trails P5 by 1.8 to 3.4 on probe in every setting;** on probe-HO from level to -3.9. P5 and Otto
    differ in many factors (recipe, floor, oversampling, teacher, pseudo-label count, epochs), so nothing is
    attributable. The site folds agree with the candidate (probe 0.583-0.592).
  - Site-fold pooled scoring and the release wait for the owner's choice between Otto and P5 as the
    production model.
- 2026-11-09: **why is Otto below P5? Hypotheses before any run** (owner asked whether mini_trainer or the
  dataset change is responsible).
  - **mini_trainer itself: unlikely to be most of it.** It reproduced lepinet where tested (T1rep 0.722 against
    0.723, C3b open-set), and the DINOv3 student self-trained in mini_trainer (0.627 / 0.431) is within 2 points
    of lepinet's B8 (0.647 / 0.452).
  - **The pseudo-label set: the prime suspect.** P5 trained on 12,230 confidence-gated crops, 98 % correct.
    Otto trained on a per-species top-K set (K = 35): 8,305 crops over 913 species, 71.3 % correct. The top-K
    rule came from R3 (+4.24 over a global gate in a second round), but here it trades accuracy for
    coverage at 29 % wrong labels, on a student that memorises what it is given (the dose results).
  - **The class set: a small part.** No floor adds 591 rare species to confuse at the traps, and every
    predicted-only species costs a calibrated macro-F1. Testable on CPU: mask Otto's logits to P5's 12,041
    species and rescore.
  - **The schedule.** P5 adapted in a second stage (2 epochs on the fine-tuned model); Otto mixed the
    pseudo-labels into one 10-epoch run. Not tested this week.
  - **Predictions:** the CPU mask recovers at most 1 point. Otto with gated pseudo-labels from the same
    ensemble teacher (one factor) recovers most of the gap: probe within 1 point of P5 or above.
