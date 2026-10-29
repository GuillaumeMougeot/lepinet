# Which recipe does the merged paper rerun? (decision sheet)

**Kind:** research · **Status:** **OPEN** (2026-10-23). Decisions to pin, with both drafts' settings
and the evidence; nothing is rerun until every row has an owner's decision.

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
