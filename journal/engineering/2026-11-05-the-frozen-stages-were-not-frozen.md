# The frozen stages were not frozen

**Kind:** incident · **Status:** OPEN (2026-11-05). Every conv-backbone "frozen trunk" stage trained
the whole network. The BioCLIP-2 frozen stages did not. Reruns of cRT and classifier-only adaptation
in mini_trainer are proposed; the affected claims are on hold.

Found by the `mt` session while porting L4 to mini_trainer, and independently here the same day.
The mechanism is the same as the head's "frozen" scale and bias
([[2026-11-03-moving-the-recipe-into-mini-trainer]], finding 1): fastai's freeze bookkeeping.

## What happened

- `dev/070_two_stage.py --freeze-body` calls `train(..., freeze_body=True)`. That sets
  `requires_grad=False` on the body (`src/lepinet/train.py:171`) and prints "Backbone frozen: ...
  (head only)".
- `_fit` then calls `learn.unfreeze()` for the one-cycle, flat-cos and front-loaded schedules
  (`:194`). The `fine_tune` path unfreezes too.
- fastai's `freeze_to(0)` sets `requires_grad=True` on every tensor in the optimiser's groups.
- For conv backbones, `vision_learner` creates the optimiser at construction, with every parameter
  (`default_split`). So the body is in the optimiser, and it trains.
- The ViT path (`vit=True`) builds a plain `Learner`, whose splitter is `trainable_params`. Its
  optimiser is created after `freeze_body`, without the frozen tensors, so `unfreeze` cannot reach
  them and the freeze holds.
- The log line that says "frozen" is printed before `_fit`, so every log looked right.

## Evidence

- **A toy model** (a two-layer body, lepinet's `muon_opt_func`, `default_split`): `freeze_body`
  leaves 0 of 4 body tensors trainable, `learn.unfreeze()` makes all 4 trainable, and `fit` changes
  them.
- **The checkpoints**, from the workstation mirror:

| run | init | trunk tensors changed | median relative change |
|---|---|---|---|
| L4 (V2-S, cRT) | L0 | all 170 conv/linear weights (all 450 weight tensors, all 220 BN statistics) | 0.68 |
| G1 (DINOv3 ConvNeXt-L, cRT) | L5 | 340 of 342 | 0.36 |
| P4 (BioCLIP-2 ViT-L/14, frozen adaptation) | P3c | 0 of 295 | 0 |

- **L4's epochs** took as long as L0's (1:22 and 1:24, against 1:28 to 1:33). Its first epoch fell
  from L0's validation F1 of 0.894 to 0.835 and recovered to 0.906: that is lr 1e-3 re-warmed on the
  whole network.

## Scope

- **Affected:** every `dev/070 --freeze-body` run on a conv backbone. On EfficientNetV2-S: L4, T2,
  T2b, F2, F3, and R2 to R5b (the self-training rounds started from L4). On DINOv3 ConvNeXt-L: L6, G1, G2, G3, G3b.
- **Not affected:** P1a, P1b, P4, P4b (BioCLIP-2, through `dev/075`'s ViT path).
- **What the affected runs measured instead:** a short second stage of full fine-tuning (2 epochs,
  lr re-warmed to 1e-3) from the first stage's weights, with that stage's data and sampling. The
  numbers stand as measurements of that; their descriptions do not.

## Claims on hold

- **RESULTS 1a #3, the spine (draft C2):** "interventions belong in the classifier, not the
  representation". Its two training legs trained the representation too: L4 (cRT) and T2 ("83 %
  from the classifier alone"). The prototype-matrix leg (H1) is not affected.
- **RESULTS 1a #5:** "cRT removes the trade". L4 was deferred re-sampling, i.e. two late epochs of
  oversampling on the whole network, not cRT. The trade's removal stands as measured, by another
  mechanism.
- **The staged recipe:** F2, F3, G2, G3 and G3b, and "staged against end-to-end is a tie at 198 M".
  The stages were short end-to-end fine-tunes.
- **The noise floor:** "~0.012 between frozen-trunk stages" (R5b) is the spread between short
  fine-tuning stages.
- **PLAN section 4:** "classifier adaptation subsumes LDAM and background suppression" (T2b).
- **Paper:** the draft's C2 and its cRT and classifier-stage sections; Merge Desk steps 06, 19 and 21
  (set to hold; none was pasted).

## Next

Proposed reruns in mini_trainer, from the cross-entropy teacher (`v2s_bu_s42`), 2 epochs each, two
variants per claim:

- **cRT:** (a) the whole network trains, as L4 ran; (b) a frozen trunk with BN statistics fixed.
- **Classifier-only adaptation** on the 2.6 % pseudo-labels: (a) and (b) likewise.

(b) decides whether the spine stands; (a) reproduces the old numbers.

The four runs are built as job `c-crt-t2` and wait for the owner's approval.
- They use a new mini_trainer option, `freeze_backbone` (MT e55b65f). It trains only the head, pins
  every trunk module in eval mode (BN statistics fixed; dropout and stochastic depth off) and keeps
  the precision.
- Checked on the real V2-S BU model for one epoch: 0 of 670 trunk tensors changed, weights and BN
  statistics included. It ran at 2,786 img/s against ~950 for full training. lepinet's `freeze_body` is not
fixed here, since training has moved to mini_trainer. Anyone reusing it must re-freeze after
`unfreeze()` and check that the trunk's checkpoint tensors are unchanged.

## The reruns (2026-11-06)

c-crt-t2, from the cross-entropy teacher (test fold 0.888; traps 0.339 / 0.371 / 0.306 for trap /
probe / probe-HO). Species macro-F1; traps calibrated by (trap, night); a T2 run's "trap" set holds
its own pseudo-labelled crops.

| run | test | trap | probe | probe-HO |
|---|---|---|---|---|
| cRT, frozen trunk | 0.899 | 0.360 | 0.397 | 0.329 |
| cRT, whole network (as L4 ran) | 0.897 | 0.425 | 0.447 | 0.373 |
| T2, frozen trunk | 0.847 | 0.465 | 0.509 | 0.326 |
| T2, whole network | 0.803 | 0.577 | 0.619 | 0.425 |

- **cRT survives, frozen:** +1.1 on the test fold and +2 to +2.6 under shift. The whole-network
  version gains more under shift (+7.5 on probe) at the same test score.
- **T2's test-fold losses are real** (checked: merge, splits, weights, class order). The likely cause
  is the design: a 2-epoch re-warm to lr 1e-3 without oversampling knocks the tail back. acc1 falls
  1.5 points, macro-F1 9. cRT has the same re-warm but with oversampling, and keeps the tail.
- So T2 against the teacher confounds pseudo-labels with the re-warm. The control is running: the
  same re-warm on global data only, frozen and whole network (v2s_bu_rewarm_{frozen,full}_s42).
- This teacher's gate kept 8,169 crops at 16x (lepinet: 12,230 at 11x), the same share.
