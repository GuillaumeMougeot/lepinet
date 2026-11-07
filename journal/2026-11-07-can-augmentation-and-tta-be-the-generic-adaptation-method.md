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

## The grid (V2-S BU, the pinned recipe, 256 px, seed 42; one factor per comparison)

| run | augmentation | epochs | compared with |
|---|---|---|---|
| A0 | the pinned recipe (exists: v2s_bu_s42) | 5 | — |
| A0-15 | the pinned recipe | 15 | A0: what epochs alone buy |
| G | + geometric: rotation +-45, zoom-out by edge padding 0-25 % | 5 | A0 |
| G-15 | the same | 15 | A0-15: does the schedule pay for it? |
| W | + TrivialAugmentWide (generic, chosen for no dataset) | 5 | A0 |
| W-15 | the same | 15 | A0-15 |
| GP-15 | geometric + photometric trap augmentation ("broad") | 15 | G-15 |

- **Lepidoptera:** test fold, trap / probe / probe-HO, and the iNaturalist set from Nemo's card as
  a second shifted domain (941 images).
- **Pl@ntNet:** A0, G, W and GP at 10 and 30 epochs, test fold. There is no shift set for plants, so
  this only checks that the method does not hurt another dataset.
- **TTA on every run:** the per-view logits of the Otto TTA study, so the policy and its gain are
  measured per training recipe.

Cost: about 70 B200 hours for Lepidoptera, a few for Pl@ntNet. Runs of matching length are packed
into the same job.

## Predictions

- A0-15 against A0: +1 test fold, probe within the seed range (+-1.5).
- G at 5 epochs: test -0.5 to -1, probe +3 to +6, and TTA's gain halves. G-15 brings the test fold
  back within 0.3 of A0-15 and keeps the probe gain.
- W: smaller under shift than G (+1 to +3 probe), because it does not target framing or orientation;
  it costs more in-distribution at 5 epochs (about -1).
- GP-15 is the best non-adapted model in the grid, about +6 to +9 probe over A0-15. It is still far
  below self-training (+22): augmentation complements target data, it does not replace it.
- On Pl@ntNet, nothing in the grid loses more than 0.5 at 30 epochs.
- **Falsified if** the augmented runs still lose more than 0.5 on the test fold at 15 epochs: then
  epochs do not pay for augmentation at this size, and the method costs accuracy.

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
