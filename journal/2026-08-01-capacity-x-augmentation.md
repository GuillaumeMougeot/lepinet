# Capacity × augmentation: the 2×2 (B4)

**Kind:** research · **Status:** **RESOLVED (2026-08-01).** With B4 the four cells of a clean
factorial are filled, and the two levers turn out to be **complementary on both accuracy axes and
jointly harmful on open-set**. The augmentation tax vanishes at scale (−0.36 pt → **0.00**) while its
shifted gain *grows* (+3.99 → **+4.85**). The prediction was right on the shifted axis and wrong on
open-set — in the direction that **confirms** the mechanism it was written to test.

## The factorial

All four use a single species head + ArcFace × z-score + marginalisation. Only capacity and
`domain_aug: trap` vary.

| | in-distribution | shifted | open-set AUROC |
|---|---|---|---|
| **A1** 20 M, no aug | 0.9035 | 0.6437 | **0.9068** |
| **B1** 20 M, aug | 0.8999 | 0.6836 | 0.9010 |
| **A2** 198 M, no aug | **0.9216** | 0.6616 | 0.8298 |
| **B4** 198 M, aug | **0.9216** | **0.7101** | 0.8132 |

Main effects and their interaction:

| effect | at 20 M | at 198 M | interaction |
|---|---|---|---|
| augmentation, in-distribution | −0.36 | **0.00** | tax **disappears** with capacity |
| augmentation, shifted | +3.99 | **+4.85** | gain **grows** with capacity |
| augmentation, open-set | −0.58 | −1.66 | cost **grows** with capacity |
| capacity, in-distribution | +1.81 | +2.17 | — |
| capacity, shifted | +1.79 | +2.65 | — |
| capacity, open-set | **−7.70** | **−8.78** | — |

## Three things worth keeping

### 1. The augmentation tax is a small-model problem

At 20 M, training on deliberately degraded images costs 0.36 pt in-distribution. At 198 M it costs
**nothing** — 0.9216 both times, identical to four decimals, and well inside the 0.0000 species noise
floor measured in [[2026-08-01-how-noisy-are-our-numbers]].

The reading is capacity-as-slack: a small model spends representational budget on the corrupted
variants and has less left for the clean distribution, while a large one absorbs both. So the usual
worry about robustness augmentation — that it trades clean accuracy for robust accuracy — is **a
consequence of under-parameterisation, not a law**. At sufficient capacity there is no trade to make.

### 2. Capacity and augmentation do not overlap

If both levers were fixing the same failures, the augmentation gain would *shrink* once capacity had
already handled some of them. It grows instead: +3.99 → +4.85. They attack different things —
capacity buys discrimination, augmentation buys nuisance invariance — and the shifted benchmark
punishes failures of both.

This is the first result in the project where two interventions **compose better than additively**,
after three consecutive cases where they composed worse ([[2026-07-30-does-arcface-compose-with-marginalisation]]).
Worth noting because it means "should compose" is not a reliable prior in either direction; the only
way to know has been to run the cell.

### 3. Everything helps accuracy and everything hurts novelty detection

The open-set column is monotone downward across the whole table: 0.9068 → 0.9010 → 0.8298 → 0.8132.
**Every intervention that improved accuracy on either axis cost AUROC**, and the two costs stack.

That is a coherent picture rather than four coincidences. Both levers make the model *more
confident on inputs it has not seen*: capacity by fitting known taxa more tightly, augmentation by
widening each known class to cover degraded variants — which is exactly the region a novel species
would otherwise occupy. Robustness augmentation buys tolerance of unfamiliar-looking *known* species,
and pays for it in tolerance of genuinely *unknown* ones. Those are the same operation seen from two
sides.

## Scoring the prediction

Committed in [[2026-07-31-best-model-is-not-the-best-model]]: shifted **0.70–0.73**, in-distribution
**0.915–0.920**, AUROC **0.83–0.86**, with *"above 0.88 falsifies the scale-hurts-open-set
mechanism"*.

| axis | predicted | actual | verdict |
|---|---|---|---|
| shifted | 0.70–0.73 | **0.7101** | correct, lower half |
| in-distribution | 0.915–0.920 | 0.9216 | marginally above the range |
| open-set | 0.83–0.86 | **0.8132** | **wrong — below the floor** |

The AUROC miss is the informative one. The falsification criterion was set at 0.88 *upward*, because
the risk being guarded against was that resolution or epoch budget, not capacity, explained A2's weak
novelty detection. It came back at 0.8132 — further from 0.88, not closer. **The mechanism survives,
and the prediction failed by underestimating how much augmentation costs on this axis**, which was
not something the prediction modelled at all: B1's −0.58 pt was treated as negligible and it nearly
tripled at scale.

## Where this leaves the deployable model

B4 shifted **0.7101** is the best external score in the project, and it matches the plain multi-head
DINOv3-ConvNeXt-L (0.7098) — i.e. **augmentation exactly repays the shift penalty the ArcFace margin
imposes**, and B4 gets open-set capability for free relative to that plain model. That is a clean
statement of what the architecture costs and what buys it back.

But there is still no single best model, and the split is now sharp:

| if you need… | ship | why |
|---|---|---|
| maximum accuracy on someone else's images | **B4** | shifted 0.7101, in-distribution 0.9216, no clean-accuracy cost |
| reliable novelty detection | **A1** | AUROC 0.9068, and 10× smaller |
| both, on a phone | **B1** | shifted 0.6836 with AUROC 0.9010 — the only cell that is decent at both |

**B1 remains the recommendation for the app**, and it is now recommended for a reason rather than by
default: it is the only cell that does not sacrifice one axis outright.

## What this makes worth running next

**The open-set degradation is now the binding constraint**, not accuracy. Three of four cells trade
AUROC for accuracy and none trades the other way, so the next experiments should attack that axis
directly rather than continue climbing accuracy:

1. **Does the margin need re-tuning at scale?** `m = 0.3` was chosen on a 20 M model. Larger
   embeddings concentrate cosines differently, so the margin that produced 0.9068 at 20 M may simply
   be too small at 198 M. This is the cheapest hypothesis and the one most likely to recover the 8.8
   pt — and note it contradicts the earlier decision to abandon margin tuning, which was made when
   the margin only had to justify itself in-distribution.
2. **Is the AUROC loss a calibration effect too?** Temperature-scaling the logits before scoring
   costs nothing and would separate "the geometry is worse" from "the confidences are miscalibrated".
   Pure post-processing on saved predictions, no GPU.
3. **B3 (self-training)** stays the highest-value robustness rung, but it should now be evaluated on
   the AUROC axis first, since that is where the headroom is.
