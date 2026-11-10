# Which recipe does the merged paper rerun? (decision sheet)

**Kind:** research · **Status:** **RESOLVED** (2026-11-09): one recipe, the campaign's; decisions at the end. Decisions to pin, with both drafts' settings
and the evidence; nothing is rerun until every row has an owner's decision.

> **Reframed 2026-11-03 (owner):** the code base is mini_trainer, with lepinet's recipe and findings
> integrated. The rows are settled by single-factor runs in the merged code, against a parity
> baseline: [[2026-11-03-moving-the-recipe-into-mini-trainer]]. Row 8: at 20 M lepinet's optimiser
> was AdamW, because Muon takes only 2-D tensors and EfficientNetV2-S has none.

## The question (the owner's, 2026-10-23)

The colleague's settings have priority, and some experiments will have to be redone anyway. Pin and
gather every decision first (the square-root oversampling, the number of epochs, EMLA...), then rerun
once. The evaluation side is already decided ([[2026-10-22-rescoring-in-the-mini-metrics-convention]]):
mini_metrics' current macro metrics, with and without a calibrated threshold, the calibration split
grouped by trap and night, and the reconciled trap export.

## The sheet

"Asger" = the colleague's draft and `mini_trainer/publication/experiments/training_ablations`;
"lepinet" = recipe L of this repo. Evidence in the old metric convention unless marked.

| # | setting | Asger | lepinet | evidence | recommendation |
|---|---|---|---|---|---|
| 1 | species floor | none: 12,632 species | 50 training images: 12,041 | the floor removes 591 species; it was also where lepinet's unseen species for open-set came from | **Asger's** (owner, 2026-10-23); open-set then needs withheld species (row 15) |
| 2 | images per species | corpus cap ~2,000 | same | a 1,000 cap helps under shift (+1.76 probe, +2.94 held-out, L7) | keep 2,000; the cap sweep stays a result |
| 3 | class sampling | none ("never oversample tail classes") | square-root oversampling | +1.9 in-distribution, -2.9 under shift at 198 M (L0, G rows); rebalancing only the classifier (cRT) keeps most of the gain without the cost (L4) | **none**, as Asger; cRT as a separate, reported classifier stage |
| 4 | loss | EMLA (entropy-gated logit adjustment) | cross-entropy | every prior-correcting loss tested lost under shift, monotonically in strength (balanced softmax, i.e. logit adjustment at tau 1: -7.2 pt on the full trap set against no reweighting, L0 vs L1); EMLA is untested under shift | decide on a measurement: CE against EMLA, same recipe, scored on probe and held-out species |
| 5 | prototype regulariser | lambda = 1e-3 (draft) / 0.1 (ablation table) | none | untested here; the two values in Asger's documents disagree | Asger to fix the value; include or drop with row 4's measurement |
| 6 | epochs | 30 | 5 | 10 epochs were still improving at 20 M (design-decisions, the ladder) | **cost decides** (row 16); at least 10 |
| 7 | image size | 384 | 256 (resize 460) | untested here | cost decides |
| 8 | schedule, optimiser | cosine, 1 epoch head warm-up; MuonAuxAdamW (backbone at lr/3) | one-cycle, 0.5 epoch warm-up; Muon backbone + AdamW head | one-cycle was worth +1.2 at 5 epochs (flat schedule graded mid-descent) | Asger's, if the schedule ends annealed |
| 9 | backbones | EfficientNet-B0, EfficientNetV2-S, ViT-L/16 | EfficientNetV2-S, DINOv3-ConvNeXt-L (198 M), BioCLIP-2 ViT-L/14 | BioCLIP-2's training set holds two thirds of the test fold (P1-P5) | Asger's three for the head comparison; one large backbone for the shift results |
| 10 | heads | F, BU, I, TD, AR | F (= single head + marginals), BU (= marginal supervision), I, TD | AR lost by 20 pt here | Asger's five |
| 11 | angular margin | none | ArcFace in several runs (B1-B8) | +0.78 AUROC for -1.00 accuracy; damages marginalisation (§4.7) | **none** by default; margin results to the appendix |
| 12 | precision | fp16 | fp16, head in fp32; bf16 for margin heads | margin heads overflow in fp16 | fp16 (no margin heads) |
| 13 | seeds | 3 (42, 43, 44) | mostly 1, a few 2 | identical lepinet runs differ by up to 3.74 pt in frozen-trunk stages | **3**, as Asger |
| 14 | self-training dose | — | 12,230 pseudo-labels x 11 (2.6 % of rows) | the probe-set optimum disappears in mini_metrics' convention; held-out species still prefer a low share | rerun a short dose sweep (1x, 11x, 26x) with the pinned recipe |
| 15 | open-set benchmark | — | species below the floor; C3b for common withheld taxa | without a floor, only withheld taxa give unseen species | one withheld-taxa split (C3b's design) for every head |
| 16 | budget | — | — | at 256 px a 20 M run reads ~1,100 img/s: one epoch of 5.06 M images is ~77 min, 30 epochs ~38 h; at 384 px roughly 2.25x | price the full grid (heads x backbones x seeds + the shift runs) before choosing rows 6 and 7 |

## What gets rerun, once the sheet is pinned

1. The head comparison (Asger's grid), with three seeds, scored on the test fold, Pl@ntNet and the
   trap probe set.
2. The shift results on one backbone with the pinned recipe: no adaptation, self-training (row 14),
   real labels (T1, T1rep), a trap and a site held out (X1, X2), cRT (row 3).
3. Open-set on the withheld-taxa split, museum-style and trap images (N2's design).

## Decisions (owner, 2026-11-09)

All five recommendations accepted:
1. **C1** keeps its in-distribution half; its trap half is restated: under shift the head ranking depends on
   the backbone, and the independent head stays at or near the top.
2. **C2** (interventions belong in the classifier) is dropped as the spine; self-training (C3) takes its place.
   Truly frozen trunks lose to whole-network adaptation (T2 probe 0.509 against 0.619, VK).
3. "cRT removes the trade" and the staged recipe are dropped, with one line in the appendix.
4. **One recipe**, the campaign's: cross-entropy, uniform sampling, no species floor (Merge Desk 04).
5. This weekend's results (augmentation, public benchmarks, Otto) stay out of this paper.

### Discussion point 1: oversampling, and when to rebalance

The owner's framing: a generic recipe, plus a short list of settings to change by dataset, each with a
measurable trigger ("heavy tail" quantified).

Rebalancing on Pl@ntNet (V2-S, seed 42 against the three-seed mean of no rebalancing, test fold species):

| head | none | sqrt-oversampling | EMLA |
|---|---|---|---|
| F | 0.466 | 0.479 | **0.514** |
| BU | 0.457 | **0.500** | 0.478 |
| I | 0.453 | **0.500** | 0.494 |
| TD | 0.407 | **0.504** | 0.345 |

- **Oversampling stays, and is the safer of the two:** it helps every head (+1.3 to +9.7) and closes TD's gap.
  EMLA is best for the flat head (+4.8) but breaks TD (-6.2). One seed per rebalanced arm; the F gain of
  oversampling (+1.3) is near the seed spread.
- On Lepidoptera, rebalancing pays little in-distribution and costs under shift: EMLA +0.6 on the test fold,
  -4 to -5.5 on trap images (VS). Oversampling was not rerun on Lepidoptera in mini_trainer. The lepinet
  evidence, from end-to-end runs, is +1.9 in-distribution and -2.9 under shift at 198 M (L0).

How heavy each tail is (training split, species counts):

| statistic | Lepidoptera (sets 2-9) | Pl@ntNet-300K (train) |
|---|---|---|
| classes / images | 12,632 / 5.06 M | 987 / 244 k |
| median images per class | 163 | 22 |
| largest / median class | 32 | 328 |
| classes under 50 images | 9.7 % | 62.4 % |
| Gini of class counts | 0.59 | 0.84 |
| effective classes, exp(entropy) / K | 0.54 | 0.19 |
| share of classes holding half the images | 13.4 % | 3.7 % |

The Lepidoptera corpus was capped at about 2,000 images per species when built, which already flattens its
head; Pl@ntNet has no cap.

**Proposed rule, a hypothesis from two datasets:** rebalance with square-root oversampling when the tail is
heavy (for example, more than half the classes under 50 images, or exp(entropy)/K below about 0.3), and not
when the model is deployed under a domain shift, where every rebalancing tested lost. Two datasets cannot
place the threshold. What would: the same three statistics and one oversampling run on more datasets (the
public benchmarks are candidates), and mini_trainer printing the statistics at the start of training.

### Discussion point 2: does marginal supervision, or marginalisation at inference, still pay?

This is the colleague's central point, so it needs evidence beyond the species score. F and BU both read genus
and family by marginalising species at inference; BU adds a loss at every rank (marginal supervision); I and
TD predict coarse ranks with their own parameters. Three-seed means, campaign:

| backbone | head | test species / genus / family | trap species / genus / family (calibrated) |
|---|---|---|---|
| V2-S | F | 0.889 / 0.949 / 0.969 | 0.354 / 0.468 / 0.522 |
| | BU | 0.887 / **0.956 / 0.976** | 0.331 / 0.459 / 0.511 |
| | I | 0.881 / 0.947 / 0.965 | **0.378 / 0.484 / 0.534** |
| | TD | 0.842 / 0.930 / 0.960 | 0.327 / 0.433 / 0.525 |
| B0 | F | **0.875** / 0.940 / 0.961 | 0.280 / 0.391 / 0.423 |
| | BU | 0.864 / 0.940 / **0.965** | **0.332 / 0.447 / 0.496** |
| | I | 0.854 / 0.923 / 0.940 | 0.326 / 0.433 / 0.455 |
| | TD | 0.788 / 0.887 / 0.934 | 0.267 / 0.360 / 0.482 |
| DINOv3 | F | **0.920** / 0.966 / 0.981 | 0.405 / 0.516 / 0.544 |
| | BU | 0.918 / **0.968 / 0.984** | 0.416 / **0.548 / 0.587** |
| | I | 0.917 / 0.966 / 0.981 | **0.424** / 0.534 / 0.571 |
| | TD | 0.906 / 0.963 / 0.980 | 0.402 / 0.528 / 0.572 |

- **Marginalisation at inference holds, in-distribution, everywhere:** F or BU is best at every rank on
  every backbone, and separate coarse parameters (I, TD) never help there.
- **Marginal supervision is a coarse-rank tool.** BU over F in-distribution: genus and family +0.2 to +0.7, at
  a species cost of 0.1 to 1.1. On trap images at genus and family: +5.6 / +7.3 on B0 and +3.2 / +4.3 on
  DINOv3, larger than the trap seed spread; on V2-S -0.9 / -1.1, within it.
- **At species under shift the ranking depends on the backbone,** and I is near the top on all three.
- **The argument for the colleague:** the claim survives where it matters for deployment. Coarse predictions
  are what back-off and abstention fall back on, and marginal supervision makes them better on two of three
  backbones under shift and on all three in-distribution. It is not a species-accuracy method.

Owner, 2026-11-09, later: **C6 (ArcFace x z-score, the margin relocating the open-set signal) is cut**: the paper is
already too long and the message must be clearer. The margin has no rerun; its appendix section goes too.
The open-set scoring-rule claim is restated from the rerun (step 17): entropy is best for every head at both
scales.

Owner, 2026-11-10: **the merged paper is split in two.** Paper A, hierarchical heads and metrics, Asger
leads, on his Overleaf draft. Paper B, species classifiers under domain shift, the owner leads. The
reasons: two storylines and two recipes in one draft (about 15,000 words); the hierarchy claim survives
only for coarse ranks and abstention, while the largest effect (self-training, +22.7) is unrelated to the
hierarchy. Pieces are tagged [A], [B] or [A+B]; PLAN.md section 1 holds both to-do lists. Open with Asger:
one head grid for A and EMLA's status.
