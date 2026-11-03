# Moving the recipe into mini_trainer

**Kind:** subproject · **Status:** **OPEN** (2026-11-03). Phase 0, the parity audit, is with the
mini_trainer session (tmux `mt`), on branch `feature/lepinet-integration` in the worktree
`~/codes/mini_trainer/.worktrees/lepinet-integration`; its technical notes go to
`.agents/notes/2026-11-03-lepinet-integration.md` there.

## The decision (owner, 2026-11-03)

One training code base: mini_trainer (MT), because Asger has time to maintain it. lepinet's recipe,
its speed practices and its findings move into a new MT branch; lepinet keeps running in parallel
for a while. In order:

1. reproduce lepinet's baseline in MT almost digit for digit, in accuracy and in convergence speed;
2. integrate the discoveries;
3. use MT's test set, and choose between the two recipes' variants (EMLA, oversampling, ...), or keep
   both where the difference is interesting;
4. choose which experiments to rerun for the paper;
5. compare with Nemo on the AMI trap crops (the Flemming set);
6. train the third production model, after Meghan and Nemo.

This settles the code-base question of the decision sheet
([[2026-10-23-which-recipe-does-the-merged-paper-rerun]]). Its rows are now settled by measurement in
the merged code (phase 3 below), not by preference.

## What moves, and what does not

**Phase 1, for parity.** Each piece is opt-in, so Nemo's recipe and MT's defaults stay untouched.

| piece | evidence | MT today |
|---|---|---|
| √-oversampling sampler: per-row weights, drawn with replacement, epoch length kept | +2.6 pt on the ladder; also the mechanism for self-training doses and cRT | "resampling is not supported" |
| presizing: aspect-preserving resize to 460, random crop (train) or centre crop (valid), bilinear; then a fastai-style GPU affine (dihedral p=0.5, rotate ±15°, zoom 1-1.1, reflection padding) resampled once to 256 | the lighter-augmentation part of the +4.7 bundle | squashes to the input size with NEAREST; salt-and-pepper, flips, rotation, translate |
| warm-up from lr/100 over 0.5 epoch, then cosine to 0, the same curve for every group from step 0; one learning rate; weight decay on every parameter | one-cycle +1.2 pt | backbone at lr 0 during a head warm-up, then lr/3; no decay on norms and biases |
| cross-entropy without smoothing or regulariser | lepinet's recipe | knobs exist; defaults are smoothing 1/C, regulariser 0.1 |
| head in fp32 under fp16 autocast; GradScaler settings | fp16 overflow in the cosine head | to check |

**Phase 2, the methods the paper and the third model need.**

| piece | evidence |
|---|---|
| single head with marginal inference (F), next to MT's BU (HierarchicalClassifier), I, TD and AR | F = 0.9135 / 0.9606 / 0.9739, the cheap reference |
| two-stage training from a checkpoint on a frozen backbone: cRT (class-balanced classifier) and classifier adaptation on pseudo-labels | cRT keeps the in-distribution gain of rebalancing without its shift cost; adaptation gets 87 % of self-training's gain in minutes (T2) |
| self-training: pseudo-label the crops of the adapt nights, merge them at a dose; the (trap, night) grouped split into adapt / probe / probe-HO | the largest lever (B3rep5x, T1rep, X1, X2) |
| trap-like augmentation presets (motion blur, low light, JPEG) | +4.5 pt on trap crops at 20 M, little once the classifier is adapted (B1) |
| per-species image cap | 1,000 helps under shift (L7) |
| evaluation: mini_metrics at zero threshold and calibrated, calibration grouped by (trap, night); probe and probe-HO; a withheld-taxa open-set split; the five scoring rules with AUROC by stratum; back-off policies (greedy per rank at 95 %, expected utility); temperature scaling | how every result in the paper is scored |

**Not moved, and why** (the owner asked for a reason on each):

| piece | why not |
|---|---|
| ArcFace margin | Costs 1.00 / 1.15 / 1.11 pt (species / genus / family) on the test fold. The coarse ranks lose most, because a margin distorts the spread of probability that marginalisation sums. It buys +0.78 AUROC best-vs-best. It needs bf16 and a temperature before any threshold (B8's softmax is exactly 1.0 on 70 % of trap crops), and it makes expected-utility back-off lose on traps. Consequence: lepinet's 198 M models (A2, B4, L5, B8) used it and will not be reproduced in MT. They stay recipe-L evidence in the appendix, and the new large model has no margin. |
| logit adjustment, balanced softmax | Lost in-distribution (0.9031 against 0.9148 for oversampling, and genus and family fell: one temperature for three label distributions) and under shift, monotonically in strength (traps, current convention: none 0.349, balanced softmax 0.224, balanced softmax plus oversampling 0.183). MT's EMLA is the same family with an entropy gate; phase 3 measures it instead of porting a second variant. |
| sampled-softmax, low-rank and proxy-free heads (H2-H4) | Built for a million-species head, and all lose 3-5 pt: 1,024 sampled negatives -3.25 with no plateau; taxonomy-aware negatives recover only 26 % of that; rank 128 -3.1; running centroids instead of a matrix -4.6. At 12,632 species the full matrix fits. At 1 M species the measured route is a data floor plus centroids at inference. |
| balanced pseudo-labels | Harmful end to end at 20 M (-6.4 probe, -4.1 held-out species, B9). On a frozen backbone at 198 M, +2.4 against a 2.2-point spread between identical runs. A balancing scheme is only a weight vector, which the ported sampler accepts, so no code is needed if it is ever wanted. |
| MixUp | Used only in the early 198 M explorations (alpha 0.2), never measured on its own, and in none of the models the paper uses (A2, B4, L5, B8, P5). It is incompatible with the margin head. There is nothing to reproduce and no finding behind it. |
| distillation for the app | Worth +0.44 pt to an EfficientNet-B0 student over training it from scratch (A3 against A6). It served only the phone app (D2), which keeps lepinet's bundle path. Neither the paper nor the production model needs it, and MT has its own teacher distillation if a small model is wanted. |
| int8 for the browser | ORT-Web has no ConvInteger kernel, so the int8 model cannot run there; fp16 ships. MT already has its own int8 paths (x86 PTQ/QAT, CUDA int8 training), and lepinet's adds nothing to them. |
| nvJPEG GPU decoding | A memory win (about 10x less host memory per loader worker), not a speed win. At 20 M the model is the ceiling (~1,100 img/s on a B200), and GPU decoding is slower unless overlapped with compute. MT sizes workers from the cgroup and has caching and CUDA prefetch. Revisit only if a profile shows a decode-bound run. |

**MT already has:** the hierarchical heads, ONNX export, the deployment adapter, regional presets,
TTA, DDP, compilation, caching, cgroup-aware worker counts, skipping of non-finite losses, and resume.

## Parity protocol

**Setting.** lepinet's: the 50-image floor, 12,041 species, and the test fold = set 0 restricted to
them (629,742 images). The target is then numbers that already exist, so no new lepinet run is
needed. Everything after the gate moves to MT's test set (no floor, 12,632 species).

**Gate.**
- Test macro-F1 within 0.003 of lepinet's at every rank, in mini_metrics' current convention at
  threshold 0: F 0.9135 / 0.9606 / 0.9739, BU 0.9135 / 0.9633 / 0.9778, I 0.9110 / 0.9587 /
  0.9708. Retraining noise on the test fold is about 0.000, so 0.003 is a real difference. The
  0.9148 of run `20260716-154156` is the I head at mini_metrics' optimal per-rank threshold, a
  different convention; the first brief to the `mt` session mixed the two, and was corrected the
  same day.
- The per-epoch validation curve within about 0.003 of run `20260716-154156`: 0.7855, 0.8349,
  0.8697, 0.8987, 0.9096 (species, argmax).
- Throughput at least lepinet's on the same GPU: 950-1,100 img/s on a B200, measured with both codes
  in the same job. Local timings are biased, because the RTX 5090 is shared (owner); lepinet's
  local run took 1:35 per epoch with validation, about 890 img/s of training.

**Steps.**
1. Audit: a component-by-component diff.
2. Port the phase-1 pieces.
3. A side-by-side on a subset on the RTX 5090. An implementation difference shows on any data, so a
   subset can find one, but only the full run can pass the gate.
4. The full run.

**Prediction, before the audit.** MT reaches parity only once every phase-1 piece is ported. Most
likely to matter, in order:
1. the frozen-backbone warm-up and the backbone at lr/3, which cost the backbone a large share of a
   5-epoch budget;
2. oversampling (+1.9 pt macro);
3. squash and NEAREST against crop and bilinear (aliasing on wing texture);
4. salt-and-pepper noise.

If the full run still misses by more than 0.003 with every piece ported, the rest is an
implementation detail (resampling kernels, GradScaler, initialisation), found by swapping one
component at a time.

## Phase 3: the recipe, settled by measurement

Single-factor runs against the parity baseline: 20 M, 5 epochs, scored on the test fold, probe and
probe-HO (calibrated, grouped), with two seeds wherever the probe spread matters.

| factor | lepinet | MT | prediction |
|---|---|---|---|
| sampler | √-oversampling | none | none wins under shift (-2.9 pt at 198 M) and loses ~1.9 in-distribution; cRT recovers most of that |
| loss | CE | EMLA | untested under shift; every prior-correcting loss lost there, monotonically in strength, so EMLA <= CE on probe |
| regulariser | 0 | 0.1 | within noise (0.001 was a wash) |
| schedule | one-cycle, all groups | head warm-up, backbone lr/3 | one-cycle wins at 5 epochs; the gap shrinks at 30 |
| resize and augmentation | crop, bilinear, affine | squash, NEAREST, salt-and-pepper | crop and bilinear >= squash in-distribution; salt-and-pepper may help on traps |
| epochs, image size | 5, 256 | 30, 384 | cost decides; 10 epochs were still improving |

## Phases 4-6

- **Paper reruns**, as the decision sheet listed them:
  - the head grid, 3 seeds, scored on the test fold, Pl@ntNet and probe;
  - the shift results on one backbone: no adaptation, self-training doses, real labels, a trap and a
    site held out, cRT;
  - open-set on a withheld-taxa split.
- **Nemo comparison.** Nemo's card numbers come from the old Flemming export (52,788 + 5,852 crops),
  whose GBIF keys were never reconciled: 13.7 % of its rows fall outside the vocabulary, against
  0.5 % in our reconciled export (owner, 2026-11-03: not the best set). So Nemo is re-run on the
  reconciled export (`data/flemming`, 47,905 crops) and scored like every other model:
  - mini_metrics' current convention, at zero threshold and calibrated, with calibration grouped by
    (trap, night);
  - the full set and the held-out probe nights. A self-trained model has seen the adapt nights'
    crops without labels, so it is compared on the probe nights only;
  - the northern-Europe preset and TTA as optional extra columns, for the card's view.
- **Third model**: the pinned recipe at the larger scale (ConvNeXt-L DINOv3 or BioCLIP-2, all
  12,632 species, self-training, BU head), released through MT's release path. The owner names it.

## Side finding (2026-11-03): at 20 M the optimiser was AdamW

`MuonAuxAdamW` sends only 2-D tensors to Muon. EfficientNetV2-S has 0 of 20.18 M body parameters
in 2-D, and lepinet routes the head to AdamW, so no 20 M run used Muon. This was a genuine error in
how we described the recipe; the conclusion that the optimiser was not the lever stands, and the
ladder is an AdamW ladder.

Corrected in `docs/design-decisions.md`, `docs/concepts.md`, `docs/user-guide.md`, `docs/index.md`,
`paper/DRAFT.md` and Merge Desk steps 04 and 23. Muon did act on the linear layers of the 198 M
ConvNeXt and the ViT trunks (B8, P5).

In MT's own loop, the head's projection matrix can receive Muon: only parameter names containing
`linear` or `layer` go to AdamW. That is one more difference for the audit.

## Decisions (owner, 2026-11-03)

1. Parity runs in lepinet's setting (the 50-image floor); everything after the gate uses MT's test set.
2. The branch stays local until the owner says it may go to Asger's repository.
3. Compute: use the local GPU as much as needed, but it is shared, so local speed tests are biased.
   Use UCloud GPUs freely, preferably for experiments in parallel.
4. Open: the third model's name.

## Log

- 2026-11-03: the decision; phase 0 assigned to the `mt` session; the Muon finding. The owner
  approved the selection and the parity setting. The reference conventions in the brief were
  corrected. Nemo's card set was judged unreconciled, so Nemo is re-scored on `data/flemming`.
