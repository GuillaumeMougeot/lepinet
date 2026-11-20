# Does the cosine head's clamp floor cost accuracy, and does the fix train as well?

**Kind:** research · **Status:** RESOLVED (2026-11-20): **the fix trains worse, at both scales** (CF on V2-S: test fold
-1.0, top-1 -0.50; cftol at 190 k classes: top-1 -1.12). The floor is not a defect but a learned gate, a non-linearity
worth more than the few images it strands; the next ToL run keeps the current head (owner: a negative result). Found at Lepidoptera scale by the
mt session on 2026-11-03 ([[2026-11-03-moving-the-recipe-into-mini-trainer]]); the floor itself was first seen in
August ([[2026-08-06-the-cosine-head-is-not-unit-norm]]).

## The problem, in plain words

The head scores an image against each species with a **cosine**: the agreement between the image's direction and
the species' direction, from +1 (same) to -1 (opposite). The cosine is then stretched into a score (the logit) by
`asin` and a factor sqrt(D-2) = 39.2 for D = 1,536, and the softmax turns the scores into probabilities. `asin` only
accepts inputs between -1 and +1.

The recipe adds a learned **magnitude** per species, `g`, and multiplies the cosine by it *before* the stretch:
`logit = 39.2 * asin(clamp(g * cos, -1, 1)) + b`. Once `g` is above 1, `g * cos` can leave [-1, 1], so the code clamps
it. Every species with `g * cos` below -1 gets the same minimum score (-61.6 plus its bias), whatever its cosine:
that is the **floor**. On the floor the score is flat, so its slope is zero, and training learns through slopes. A
species on the floor sends no "move closer" or "move away" signal to the image, and receives none.

For wrong species that is harmless, and the network uses it on purpose: growing `g` is an easy way to give most
wrong species the lowest possible score. On tol_full, `g` grew to 1.41 (median; it starts at 1), so any cosine below
-0.71 is floored. That is 79 % of all (image, species) pairs, and the median image has about 790 species off the
floor out of 190,199, a learned shortlist.

The defect is when the **true** species is on the floor. The loss wants its score raised, but the slope is zero, so
neither the image's embedding nor the species' prototype moves towards the other; only the species' bias rises, which
raises it for every image. That image stays wrong unless something else moves it.

![The transform and where tol_full's scores sit](figures/2026-11-19-clamp-floor.png)

Left: the score against the cosine at g = 1.41. The current head (red) is flat below -0.71; the fix (blue) is not.
At g = 1 both are the grey curve. Right: `g * cos` for 5,000 test images. Wrong species pile up just left of -1 and
true species just right of it, so the head separates them at the clamp's edge.

## Measured (`dev/113 stats`, fp32 head maths, 19,992 test images of trained species)

| checkpoint | g median | pairs on the floor | true class on the floor | top-1 | errors with the true class on the floor |
|---|---|---|---|---|---|
| ToL pilot (one pass) | 1.17 | 0 % | 0 % | 35.0 % | 0 % |
| tol_full (five passes) | 1.41 | 78.4 % | **0.88 %** | 81.3 % | **4.7 %** |
| Lepidoptera, B0 F / BU / V2-S F (mt, 2026-11-03) | 1.72 / 1.54 / 1.74 | 94 / 97 / 57 % | 0.33 / 0.31 / 0.09 % | | about 4 % (B0 F) |

The floor grows as `g` grows during training, and the stranded share grows with the number of classes. Every image
with its true class on the floor is misclassified, so the ceiling on what the fix can recover is 0.9 points of top-1
at 190 k classes (less in practice: some of those images would be wrong anyway).

The gradient, checked on one toy image (`dev/113 grad`; target at cos -0.80, two competitors at -0.45 and -0.50,
g = 1.5): the current head gives the target's cosine a gradient of exactly 0 (competitors +78.4 and +1.3); the fixed
head gives -97.9 (competitors +63.5 and +2.3). At g = 1 both forms give identical scores.

## The fix

Apply the magnitude after the stretch: `logit = g * 39.2 * asin(clamp(cos, -1, 1)) + b`. The cosine of two unit
vectors is always in [-1, 1], so the clamp becomes a guard against rounding only, every species keeps a slope, and
`g` becomes a per-species temperature. It is the mt session's "scale on z" from 2026-11-03, opt-in in mini_trainer so
no published number moves.

**What it could break.** The current head discriminates *at* the floor's edge (figure, right). The fixed head has
no floor to use, so it must separate species on `asin`'s steep part near -1 instead. That is a different solution,
not a corner fixed, and it may train worse. `asin`'s slope also grows without bound near +/-1 (capped at about 2,200
by the rounding guard), which can make large gradients for an image almost opposite its own species; the recipe's
gradient clipping (norm 5) bounds them. Alternatives if it fails: `g` frozen at 1 (mini_trainer's documented
default), or a clamp that passes the gradient through.

## The test (CF), before any longer ToL run

The pilot cannot test it: after one pass nothing is on the floor. The floor exists in trained Lepidoptera heads, and
three seeds give a noise band, so:

- **CF**: V2-S BU with the fix, otherwise exactly v2s_bu_s42, against v2s_bu_s42/43/44 (test fold species macro-F1
  seed range about 0.11 points). The three seeds ran on 5 Nov code, so the same job also reruns v2s_bu_s42 on today's
  code without the fix (mt's suggestion): CF is one factor from that control, and the control against the old s42
  shows whether the code moved anything. Built by mt as the opt-in head option `magnitude_after_angle` (default
  unchanged); its CPU test reproduces the gradients above (zero for a floored target before, non-zero after),
  identical logits at g = 1, and the option surviving a save and reload.
- **Predictions (2026-11-19, before any run):** nothing on the floor by construction; test fold within the seed range
  or up to 0.3 above (the stranded share, 0.1-0.3 % of images at Lepidoptera scale, bounds the gain); traps not below the seeds. If
  CF is below the range on the test fold, the floor is doing useful work and the next ToL run keeps the current head.
- **The 190 k-class check** (mt's design, about 2-3 B200-hours): the V2-S run cannot show the floor at ToL scale, so
  two head-only arms on tol_full's frozen trunk, both starting from its trained head (78 % of pairs on the floor), one
  with the current head and one with the fix, each trained for one pass of the pilot set. The two forms give
  different logits at the same parameters when g is not 1, so the fixed arm starts perturbed (a floored pair at cos
  -0.8 moves from -61.6 to about -51; a true class at cos 0.5 from 30.7 to about 28.9): this tests how training
  proceeds from there, not a drop-in swap.
- **Predictions for the 190 k check (supervisor, 2026-11-19, before it runs):** (a) the current arm keeps its floor
  (pairs and stranded targets within 0.2 points of 78.4 % and 0.88 %); (b) the fixed arm's top-1 on the known test
  images ends within 1 point of the current arm's, either way; (c) on the images whose true class started on the
  floor, the fixed arm gets at least 10 points more of them right than the current arm. If (b) fails by more than a
  point downward, the floor does useful work at 190 k classes and the next ToL run keeps the current head whatever
  CF says; if (c) fails, the stranded images are wrong for reasons the floor does not explain.

## 190 k-class check: cftol (mt, job 12469979, 2026-11-19)

Probe: the first 200,000 images of the pilot's validation split, the same images for all three models; paired 95 %
intervals. It is harder than the 19,992 images of the measurement table above (tol_full's top-1 67.6 % against 81.3 %).

| model | top-1 | true class on the current head's floor | top-1 on tol_full's floor subset (4,824 images) |
|---|---|---|---|
| tol_full (the start) | 67.58 | 2.41 % | 0.00 |
| current arm, one more head-only pass | 64.83 | 3.73 % | 0.10 |
| fixed arm, one more head-only pass | 63.70 | none (no floor by construction) | 4.66 |

Fixed minus current: -1.12 points (+/-0.15); on the floor subset +4.56 (+/-0.60); off it -1.26 (+/-0.16). Median g:
tol_full 1.41, current arm 1.23, fixed arm 2.18 (in the fixed form g multiplies the angle, so it means something else).

**My predictions, scored.** (a) Failed: the current arm did not keep its floor unchanged; the floor share fell from 76.5
to 75.7 % and the stranded share rose from 2.41 to 3.73 %, because the re-warmed pass degraded the head itself (-2.76
points against tol_full). (b) Failed by a hair: -1.12, not within 1. (c) Failed: +4.56 points on the floor subset, not
+10. mt's decision rule (fix more than 0.5 below the current arm = not supported) gives: not supported at this size.

**What it shows and what it does not.** The test was weighted against the fix, as its design said: both arms continued
from tol_full's head with the learning rate re-warmed to 8e-4, which cost the current arm 2.76 points, and the fixed
arm started from parameters trained for the other form (1.3 points lower at step 0) with one pass to re-scale g. So
the fix is not a free swap on a trained head. Whether a head trained with the fix from the start is worse is not
answered here; CF answers it at Lepidoptera scale. The floor subset shows the stranded images are recoverable (4.66 %
right instead of 0.1 %), but in this setting recovering them cost more on the rest. A fair 190 k test would train the
head from scratch on the frozen trunk in both arms (more costly). The owner (to mt, 15:45): wait for CF before
discarding the fix.

**CF moved to the shards** (cfsh, job 12470314). The staged cf-v2s job was stopped (owner) because staging global_lepi
to the node ran at 12-220 files/s and could not finish in 12 h. cfsh trains the fix and the control on
global_lepi_shards-v1, one B200 each, one factor apart; against the 5 Nov seeds the data path also differs (same
pixels, different stream order, a seed-like effect).

## CF: the fix trains worse (cfsh, job 12470314; evals 12470726 / 12470728; scored by mt 2026-11-19 21:30)

V2-S BU on the shards, fix and control one factor apart; the three 5 Nov seeds (staged data) as the noise band. Traps
are north_europe, line 1.

| | test fold species macro-F1, zero / calibrated | top-1 | probe tta3bm, calibrated / zero | probe-HO tta3bm, calibrated / zero |
|---|---|---|---|---|
| fix | 0.878 / 0.872 | 92.67 | 0.627 / 0.422 | 0.525 / 0.232 |
| control | 0.888 / 0.879 | 93.18 | 0.627 / 0.432 | 0.490 / 0.236 |
| v2s_bu s42-44 | 0.887-0.888 (zero) | | 0.624-0.633 / 0.433-0.439 | 0.466-0.485 / 0.237-0.246 |

Fix minus control: top-1 -0.50 (+/-0.04, paired), test fold -1.0, about nine times the seed spread. The control equals
the seeds, so the shard data path changes nothing. Floor counts (mt's `saturation.py`, 8,192 test images): control 58 %
of (image, species) pairs on the floor, the true class there for 0.13 % of images; the fix none.

**Predictions scored.** Nothing on the floor: held, by construction. Test fold within the seed range or up to 0.3
above: failed (-1.0). Traps not below the seeds: failed at zero threshold (three of four cells below). One calibrated
cell is above: probe-HO tta3bm, +3.5 over the control and +4 over the best seed (2,455 crops, one seed: a hint, not a
finding). By the rule set beforehand, the next ToL run keeps the current head.

## What it means: the floor is a gate, and the gate is worth more than what it strands

The premise of this entry was that the floor is a defect with a side benefit. Both tests say the side benefit is the
main effect. At 12 k classes the floor strands the true class for 0.13 % of images, so removing it could gain at most
0.13 points of top-1; it cost 0.50. At 190 k classes it strands 0.9-2.4 % and removing it cost 1.1 (cftol, on a test
weighted against the fix).

The owner's reading (2026-11-20): networks do better with a bit of non-linearity, ReLU being the example. It can be
made exact here. Write the logit as a function of the angle t between embedding and prototype. The fixed head is
linear in it: g * S * (pi/2 - t), so g is only a temperature. The current head is S * asin(g * cos t): flat beyond
t = acos(-1/g), steep near that edge. The flat part is a per-species gate whose position each species learns through g;
behind it the species is switched off for that image, with a constant output and zero gradient, which is a ReLU's
inactive side. The fix removed the head's only non-linearity in the angle, and the model got worse. The stranded true
classes are this gate's version of ReLU's dead units: a known cost of a non-linearity that pays overall. The analogy is
loose (the trunk keeps all its non-linearities), but the gate is real.

What the gate buys, from the floor counts (an interpretation; these runs do not separate the three):
- **A different geometry.** With the floor, the network parks wrong species on the far side of the sphere (median
  g * cos -0.78; 58 % of pairs floored) and keeps g moderate (median 1.81). Without it, wrong species stay near
  orthogonal (median cos +0.02), and the network separates by temperature instead: median g 2.42, logits up to about
  +/-149 instead of +/-62. In cftol, g went from 1.41 to 2.18 in one pass for the same reason.
- **Per-image sparsity.** Each image deals with a shortlist of species (about 790 of 190,199 for tol_full); the rest
  send no gradient, so it goes to the species that can be confused. This is learned hard-negative mining. Partial FC
  in face recognition (An et al., 2021) samples a fraction of the negative classes at random to a similar end and
  loses little; here the selection is learned, per image.
- **A bounded logit.** The current head's logit cannot leave +/-S * pi/2 whatever g does, a cap on confidence; the fixed
  head has none, and its g grew. The fix arm's lower zero-threshold trap scores, and its one higher calibrated cell,
  fit a calibration change. That fit is not yet a check.

Two cheap follow-ups would separate them, neither needed for a decision:
1. CPU, on the saved logits: compare the fix arm's confidence on its errors, and its calibration error, with the
   control's. If the fix arm is overconfident, the bounded logit carries weight.
2. The gate's LeakyReLU: keep the forward pass identical (inference unchanged) and let the gradient through the floor
   for the true class only. Wrong species stay gated; stranded targets can climb back. At 12 k classes its ceiling
   (0.13 %) is below the seed spread, so only a ToL-scale run could measure it. It is a candidate for a later ToL run,
   not for TL10, by the one-factor rule.

**Decision.** The next ToL run keeps the current head. Whether `magnitude_after_angle` stays in mini_trainer as an
opt-in option (default unchanged) or is dropped is the owner's call; mt recommends either.
