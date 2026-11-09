# Can augmentation, longer training and TTA be the generic adaptation method?

**Kind:** research · **Status:** OPEN (2026-11-07). Design and predictions, before any run.

The owner's question (2026-11-07): a broad set of augmentations, more epochs to pay for them, and a
matched TTA policy could be a method that adapts to any dataset, without target-domain data. If it
is, it belongs in mini_trainer as a shipped option. Self-training needs unlabelled target images;
this would not.

## What we already know

| evidence | result |
|---|---|
| B1: photometric trap augmentation (blur, low light, JPEG), lepinet | +4.0 under shift for -0.36 in-distribution; closes 17 % of the gap |
| B4: capacity x augmentation | they compose better than additively, on accuracy and under shift |
| RESULTS 1b #15 | the early baseline gained in-distribution partly from *lighter* augmentation, at a short schedule |
| VA: the same augmentation in mini_trainer, 5 epochs | +3.4 probe, +1.5 probe-HO, -0.75 test fold (7x the seed range) |
| VL: 10 epochs at 384 px against 5 at 256 | +2.4 test, +4.7 probe (size and epochs confounded) |
| Nemo's TTA (rotation +-30 with padding, 3 views) | +10 to +14 probe on trap crops; nothing on iNaturalist photos, at 3x the cost |
| self-training at 2.6 % (VX) | +22.2 probe; the largest lever, but it needs target images |

**Reading.** Augmentation helps under shift and costs in-distribution at 5 epochs. That cost may be a
schedule cost: an augmented task needs more steps. Nemo's TTA gains say which invariances trap crops
need and the training lacks: framing (zoom-out) and orientation. Our trap augmentation is
photometric only, so it teaches neither.

## The method to test

1. **Train with a broad augmentation policy and a longer schedule.**
2. **Use TTA to find what is still missing.** One forward pass per view on a small labelled target
   sample; a view that still gains names an invariance the training lacks.
3. **Feed that back:** add those transforms to training, and keep in TTA only the views that still
   pay for their cost.

Step 2 is the adaptive part. It needs a few hundred labelled target images, not an unlabelled pool.

## The runs (cut to the best candidates, owner, 2026-11-07)

The owner asked for few candidates, enough epochs, and a start as soon as possible, in parallel with
Otto. So no 5-epoch arms and no Pl@ntNet for now. The schedule is 10 epochs at 384 px, Otto's
schedule, whose plain run already exists as the control (v2s_bu_384x10_s42: test 0.9116, probe
0.418, probe-HO 0.326).

| run | augmentation | GPU | compared with |
|---|---|---|---|
| v2s_bu_384x10_geo | geometric: rotation +-45, zoom-out by edge padding 0-25 % | 1 B200, ~13 h | the control |
| v2s_bu_384x10_geotrap | geometric + photometric trap ("broad") | 1 B200, ~13 h | geo |
| v2s_bu_384x10_trivial | TrivialAugmentWide (generic, chosen for no dataset) | 1 B200, ~13 h | the control |
| otto_bioclip_bu_geo | Otto plus geometric, nothing else changed | 1 B200, ~35 h | Otto |

- The fourth run answers "does Otto benefit" directly, without waiting for V2-S. Geometric is the
  candidate with the strongest evidence (Nemo's TTA, +10 to +14 probe). Photometric trap
  augmentation was turned off for Otto (VA).
- Each run is a single-GPU job, since the lengths differ. About 75 B200 hours in all.
- Every run gets the per-view TTA logits on the trap crops, as in the Otto TTA study.
- Pl@ntNet and the 5-epoch arms only if a candidate wins.

## Predictions

- geo: test fold within 0.3 of the control, probe +3 to +6, and TTA's gain on it halves.
- trivial: smaller under shift than geo (+1 to +3 probe), because it does not target framing or
  orientation.
- geotrap: the best V2-S run without target data, about +5 to +8 probe over the control, still far
  below self-training (+22).
- otto_bioclip_bu_geo: +1 to +3 probe over Otto, less than on V2-S, since self-training on trap
  crops already teaches part of the framing; single-view score closer to TTA's.
- **Falsified if** the augmented V2-S runs lose more than 0.5 on the test fold at 10 epochs: then
  the schedule does not pay for augmentation, and the method costs accuracy.

## What would ship in mini_trainer

If the predictions hold:
- augmentation policies as registry entries (geometric, photometric trap, TrivialAugmentWide and
  their combinations), opt-in;
- TTA policies in the predictor;
- a diagnostic that ranks TTA views on a labelled target sample and says which transforms to add.

No default changes until the grid is read. Trap augmentation stays opt-in (owner, 2026-11-07).

## Log

- 2026-11-07: opened at the owner's suggestion. Implementation and plan asked of mt; submission
  through the owner in mt's session.
- 2026-11-08: ag-v2s-384x10-geo finished (778 min, exit 0); its eval failed at submission (UCloud 502)
  and was resubmitted (12450958). ag-v2s-384x10-trivial (12450735) was stopped before staging by the
  new disk check, 501 GB free against 510 needed on a shared node, so the suspected full disk is
  confirmed. Resubmitted as 12450957 (owner approved in mt's session). If the check stops it again,
  mt reports instead of retrying.
- 2026-11-09: **trainings done, single-view scores** (autoscore; TTA per view pending, the evals failed at
  submission with UCloud 502s). Test / trap / probe / probe-HO, calibrated: control 0.9116 / 0.388 / 0.418 /
  0.326; TrivialAugmentWide 0.8946 / 0.462 / 0.504 / 0.411; geometric 0.9026 / 0.299 / 0.314 / 0.281; geometric
  + trap 0.8920 / 0.353 / 0.384 / 0.323; Otto + geometric 0.9134 / 0.368 / 0.582 / 0.439 against Otto 0.9184 /
  0.359 / 0.589 / 0.450.
  - **Predictions scored.** geo +3 to +6 probe: **falsified, -10.5**. trivial +1 to +3: **falsified upwards,
    +8.6** (and +8.5 probe-HO), the largest gain without target images in the project. geotrap best: falsified.
    Otto + geo +1 to +3: falsified (-0.7). The bar "lose more than 0.5 on the test fold": crossed by every
    arm (-0.9, -2.0, -1.7).
  - **Reading, provisional.** A generic policy chosen for no dataset beat the policies built from what TTA
    said trap crops need. Geometric training hurts under shift while geometric TTA helps, which is
    surprising enough to check the augmentation as trained before reading it (probability, magnitudes, a
    grid of training batches).
