# Moving the recipe into mini_trainer

**Kind:** subproject · **Status:** **OPEN** (2026-11-03). Phases 0 and 1 are done: the subset
side-by-side (V1) and the full-scale gate (V2, 2026-11-04) passed for I, BU and F, every rank within
0.0025 of lepinet on the test fold. V0 is scored: dropping the floor costs nothing under shift. V2T:
mini_trainer's training loop is 1.185x lepinet's. The mini_trainer session (tmux `mt`)
does the work on branch `feature/lepinet-integration` in the worktree
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
| cross-entropy without regulariser (both codes smooth labels by 1/C) | lepinet's recipe | knob exists; default regulariser 0.1 |
| the head's per-class scale and bias, trained (lepinet trains them by accident; see the audit) | every lepinet number | frozen at 1 and 0 |
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

## Phase 0: what the audit found (2026-11-03)

The `mt` session's component-by-component diff is in mini_trainer's
`.agents/notes/2026-11-03-lepinet-integration.md` (branch `feature/lepinet-integration`, commit 0e93a0d,
local). What it adds to the prediction above:

- **lepinet's head trains its "frozen" scale and bias.** fastai's optimiser setup makes every parameter
  trainable; reproduced here on CPU. Species scales end at 1.6-2.6 and biases at -0.9 to 0.2, larger for
  rare species. This is the mechanism of [[2026-08-06-the-cosine-head-is-not-unit-norm]], now resolved.
- **mini_trainer's default augmentation draws once per batch:** the whole batch flips and rotates
  together. Its preprocess also crops the central 87.5 % of the squashed image, in training too. Both
  apply to Nemo's recipe; flagged to Asger.
- **Existing knobs already reproduce lepinet's learning-rate curve** (within 1.3e-8 over 393,685 steps).
- **mini_trainer routes the head's 1280 x 1280 hidden layer to Muon**, the only tensor Muon gets on
  EfficientNetV2-S. It decays neither BatchNorm, biases nor prototypes, runs the head in fp16 with TF32,
  and since 2026-08-17 initialises the prototypes almost at random.
- **Not a difference:** label smoothing. lepinet's `None` resolves to 1/C, mini_trainer's default too.
- **All three heads already exist in mini_trainer:** I = independent, BU = hierarchical, F = hierarchical
  with loss weights 1 0 0, which marginalises at inference.

**Phase 1 plan (approved 2026-11-03):** eight opt-in pieces, each defaulting to today's behaviour:
1. the class floor and a test vocabulary filter;
2. the weighted sampler;
3. head options (learned scale and bias, fp32 head, lepinet's initialisation);
4. a grouping that decays every parameter;
5. the crop reader;
6. the per-image affine with its resample-and-normalise preprocess;
7. a validation macro-F1 statistic;
8. a recipe builder with I, BU and F presets.

The order is GPU-free equivalence checks first (lepinet's checkpoint gives identical logits in
mini_trainer's head; the same images give the same pixels; lepinet's predictions re-score to 0.9110),
then the pieces, then the subset side-by-side (50-image floor, Asger's family x abundance selector,
512 species, seed 20261003, two seeds per code), then the gate. The tolerances are agreed: per-epoch
validation macro-F1 within 0.001, investigated above 0.002; test within 0.003 at every rank.

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
| head scale and bias | learned | frozen | unknown; they sharpen rare classes, so they may raise macro-F1 and change open-set scores |
| augmentation draws | per image | per batch | per image wins: a whole batch flipping together adds little diversity per step |

**Reduced (owner, 2026-11-04): no grid.** The paper is due next week. Proposed: prior evidence
settles every row except one.
- **Sampler: none.** At 20 M on all trap crops (calibrated by trap and night), no oversampling scores
  0.349 against 0.284 with it (L0 against F), for -1.9 in-distribution. cRT, as a separate stage,
  recovers the in-distribution part.
- **The one measurement: cross-entropy against EMLA**, both without oversampling: BU head,
  mini_trainer's test set, 5 epochs at 256 px, seed 0, otherwise the lepinet preset. EMLA is
  Svenning et al.'s loss, untested under shift, and the merged paper needs one recipe. The
  cross-entropy run against V0BU also measures the sampler on this test set, if the gate passes.
- **The rest, on evidence:** BU head (V0); one-cycle; crop, bilinear and per-image affine;
  regulariser 0; learned head scale and bias; 5 epochs at 256 px (V0BU matches Nemo's 30 at 384 on
  trap crops).

**Prediction, before the runs.** Cross-entropy without oversampling: about 0.895 on the test fold
(V0BU minus the sampler's 1.9 points) and 0.36-0.40 on all trap crops (V0BU's 0.330 plus part of
L0's +6.5). EMLA: 0.5-1.5 points higher on the test fold, because it corrects the prior as
oversampling does, and 2-6 points lower on probe and probe-HO. Every prior-correcting loss lost
under shift in proportion to its strength (balanced softmax -12.5 against L0), and the entropy
gate should weaken that, not reverse it. If EMLA trails by more than 1.5 points on probe and
probe-HO, the recipe is cross-entropy; inside 1.5, a second seed; if EMLA leads, EMLA.

## Nemo on the reconciled trap export (VN, 2026-11-03)

Re-scored by `dev/098` and `dev/093`: the same 47,905 crops, nights and calibration as every lepinet
model. Species macro-F1, mini_metrics' convention, zero threshold / calibrated by (trap, night), mean
over 20 splits:

| model | trap (47,905) | probe (15,200) | probe-HO (2,455) |
|---|---|---|---|
| Nemo, full vocabulary | 0.104 / 0.305 | 0.137 / 0.338 | 0.097 / 0.293 |
| Nemo + TTA | 0.142 / 0.449 | 0.200 / 0.479 | 0.140 / 0.359 |
| Nemo + north_europe list | 0.261 / 0.525 | 0.294 / 0.546 | 0.178 / 0.449 |
| Nemo + list + TTA (the card's setting) | 0.321 / 0.599 | 0.369 / 0.646 | 0.236 / 0.490 |
| B1 (20 M, margin + trap augmentation) | 0.112 / 0.409 | 0.154 / 0.424 | 0.098 / 0.356 |
| B8 (198 M, self-training) | - | 0.311 / 0.647 | 0.148 / 0.452 |
| P5 (BioCLIP-2, unfrozen) | - | 0.396 / 0.622 | 0.209 / 0.497 |

- **Like for like, Nemo sits with our plain 5-epoch heads.** On all trap crops its 0.305 matches F
  0.284 and BU 0.313, and is 10 points under B1. Six times the epochs and 2.25 times the pixels buy
  nothing under shift. That fits finding 2: in-distribution accuracy is saturated, and the headroom is
  under shift.
- **TTA is worth +14 points** on all crops and on probe, and +7 on held-out species. That is more than
  margin plus trap augmentation together (+12.5). Its recipe (`rotation30_pad25_3`) has three views,
  two of them rotated and zoomed out by 25 % padding. It was chosen on these same crops, so part of the
  gain is selection, and it needs a test on images that did not choose it. Zooming out may match the
  framing of the training photographs; that is a candidate augmentation for phase 3.
- **The regional list is worth +16 to +22 points**, the checklist effect of K1.
- The card's setting ties B8 on probe, but it pairs a regional list and TTA against unrestricted
  single-view models. The third model is reported in all four settings.
- Nemo lacks the same 6 trap species as lepinet (234 rows, 0.5 %). Its larger vocabulary costs little
  here: 2,499 species predicted but absent, against B1's 2,455.

## V0: lepinet's recipe on mini_trainer's class set

V0F and V0BU are recipe L without the 50-image floor. They cover 12,632 species, the vocabulary of
mini_trainer's test set and of Nemo, so they anchor phase 3 and are Nemo's like-for-like partner.
Training finished on 2026-11-03 (UCloud B200, 7.6-8 h each). Epoch 1 took about 1:55 in both runs
and the later epochs 1:18-1:37, at about 1,000 img/s per job.

| validation, species macro-F1 by epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| V0F (no floor) | 0.7884 | 0.8356 | 0.8691 | 0.8962 | 0.9103 |
| F (floor, `20260729-182718`) | 0.7979 | 0.8376 | 0.8717 | 0.9002 | 0.9129 |
| V0BU (no floor) | 0.7795 | 0.8317 | 0.8625 | 0.8930 | 0.9102 |
| BU (floor, `20260730-074913`) | 0.7856 | 0.8384 | 0.8691 | 0.8968 | 0.9133 |

The validation sets differ, because V0's includes the 591 added species. If the 12,041 floor species
keep their F1, the added species score about 0.86. V0BU's genus and family reach 0.9606 and 0.9785,
against 0.9614 and 0.9794 with the floor.

**Prediction, written on 2026-11-04 while the evaluations run.**
- **Test fold** (632,913 images), threshold 0: species 0.909-0.911 for both runs; genus and family
  within 0.002 of the floor runs.
- **Traps, calibrated:** at or slightly below the floor runs (F: trap 0.284, probe 0.287, probe-HO
  0.239; BU: trap 0.313). The added species include none of the trap species, because Nemo lacks the
  same 6 as lepinet, so they can only act as distractors. Rare classes learned the largest biases,
  which could pull a few trap crops their way.
- **Against Nemo** (0.305 / 0.338 / 0.293): level on all crops, as before. On probe and probe-HO,
  Nemo is 5 points above the floor F run; V0BU should close about half of that.

**Result (2026-11-04).** Both evaluations exited 0 and scored all 632,913 test images (12,632
species). Species macro-F1; the test fold at threshold 0, the trap sets calibrated by (trap, night),
mean over 20 splits (split sd 0.006-0.008 on trap and probe, 0.02-0.03 on probe-HO):

| species macro-F1 | test fold | trap | probe | probe-HO |
|---|---|---|---|---|
| V0F (no floor) | 0.9110 | 0.279 | 0.300 | 0.235 |
| F (floor; its test fold has 629,742 images) | 0.9135 | 0.284 | 0.287 | 0.239 |
| V0BU (no floor) | 0.9107 | 0.330 | 0.351 | 0.288 |
| BU (floor) | 0.9135 | 0.313 | - | - |
| Nemo, full vocabulary | - | 0.305 | 0.338 | 0.293 |

Scoring the predictions:
- **Test fold: right.** Species 0.9110 and 0.9107. Genus is within 0.002 in both runs (-0.0006,
  -0.0008); family is +0.0021 for V0F, just outside, and +0.0004 for V0BU. The 591 added species
  score 0.877 (F) and 0.873 (BU) on 3,171 test images, a median of 5 per species, near the 0.86
  inferred from validation. They cost the 12,041 shared species 0.0009 (0.9126 in both runs against
  0.9135): 0.06 % of the shared species' images go to an added species. So 0.0017 of the 0.0025 drop
  is the average taking in the added species, and 0.0009 is distraction. mini_metrics agrees at
  threshold 0; calibrated, the runs score 0.9091 and 0.9079 (floor 0.9117 and 0.9104), the same gap.
- **Traps: half wrong.** F is level (trap -0.005, probe +0.013, probe-HO -0.005), but BU is 1.8 points
  above its floor run, not below. Threshold-free in our old convention (rows with a known label,
  classes = truth species), both no-floor runs are above on every set: F +0.005 / +0.009 / +0.019, BU
  +0.013 on trap. These are single runs; the end-to-end probe floor is ~0.004. No trap species is
  among the added ones, so nothing explains a gain. My reading: dropping the floor costs nothing under
  shift, and the gains are not a result.
- **Against Nemo: wrong, in V0BU's favour.** With the same head, vocabulary and protocol, V0BU is +2.5
  points on trap, +1.3 on probe and -0.5 on probe-HO. It closes all of the probe gap and nine tenths
  of probe-HO's, where I said half. I anchored on the flat head; probe has no floor BU run. Five
  epochs at 256 px match or beat Nemo's 30 epochs at 384 px on trap crops, which strengthens VN.
- **The head matters under shift.** BU beats F by 5 points on every trap set (+5.2 / +5.1 / +5.3) and
  by 2.9 on trap with the floor, while the two are equal on the test fold. Finding 1 has the same
  sign at ~1.4 points. BU is mini_trainer's default head and Nemo's; phase 3 tests it for the third
  model.

So mini_trainer's test set (no floor) is safe for phase 3, and V0F / V0BU are its anchors:
0.9110 / 0.9600 / 0.9760 and 0.9107 / 0.9625 / 0.9782 at threshold 0. The evaluations read `/work`
at 80-94 img/s with 32 workers, two hours each for the test fold, against ~1,000 img/s in training
at 256 workers ([ucloud/README](../../ucloud/README.md)).

## V2T: speed on the same staged data (2026-11-04)

One B200 job ran both codes on a 10 % sample (566,932 rows, 12,035 species), one epoch each, in the
order L, M, M, L. lepinet ran as on UCloud, without `GCCallback`, and mini_trainer used 44 workers.

| | lepinet | mini_trainer |
|---|---|---|
| steady training img/s, batch 64 (two runs each) | 1,201 / 1,205 | 1,422 / 1,428 |
| peak GPU memory allocated | 10.4-10.6 GiB | 6.6 GiB |
| GPU busy | 72 % | 85 % |
| validation, 47,487 images | 29 s | 18 s |
| whole process, one epoch | 386-390 s | 508-552 s |

- **The training loop is 1.185x faster** in mini_trainer, on 40 % less memory. Each code's two runs
  agree within 0.4 %. The three gates shared the node, but L1 and L2 show no drift, so the load was
  steady.
- **Whole epochs looked slower**, with about 3 minutes outside the loops. The `mt` session found two
  causes:
  - **Worker start-up, once per run.** Python 3.14 starts DataLoader workers with forkserver; each
    launch pickles the dataset and waits for the child to import mini_trainer, about 2.1 s per worker
    on UCloud, one at a time. With persistent workers this happens once per loader, not every epoch.
    Preloading the module into the forkserver cuts a launch to 0.11 s.
  - **The per-epoch figures.** At 12,041 species the dendrogram takes 135-157 s per epoch, mostly
    resolving names over the network, and the confusion matrices 10-16 s: about 12 minutes per
    5-epoch run. They are now off in the lepinet presets (decision 9).
- The gates' own step rates (I ~1,290, BU ~1,420, F ~1,400 img/s at 32 workers) say the loaders are
  not the limit. They are not a comparison with lepinet's reference runs, which read `/work` with
  256 workers.
- mini_trainer exits 1.5 s after saving its weights on UCloud, so the 5 s per worker exit delay seen
  locally does not occur there.

## V2: the parity gate at full scale (2026-11-04)

The presets for heads I, BU and F in lepinet's setting (50-image floor, 12,041 species), one B200
each, images staged to node-local disk, 32 workers. Training took 355 min (I), 324 (BU) and 325 (F)
with the per-epoch figures still on; `last.pt` then predicted all 632,913 test images. The `mt`
session's `match_ref.py` kept lepinet's 629,742 images and checked that each is present with the same
labels at every rank (the 3,171 others belong to the 591 species under the floor). lepinet's
`dev/093` scored them.

**Test fold, macro-F1 at threshold 0** (mini_trainer minus lepinet's reference run in brackets):

| | species | genus | family | species, calibrated |
|---|---|---|---|---|
| I | 0.9107 (-0.0002) | 0.9588 (+0.0001) | 0.9726 (+0.0018) | 0.9071 (-0.0004) |
| BU | 0.9131 (-0.0004) | 0.9629 (-0.0004) | 0.9803 (+0.0025) | 0.9105 (+0.0000) |
| F | 0.9144 (+0.0009) | 0.9608 (+0.0002) | 0.9759 (+0.0020) | 0.9126 (+0.0009) |
| lepinet's repeat of F (A5) | 0.9135 (+0.0000) | 0.9601 (-0.0006) | 0.9763 (+0.0023) | 0.9115 (-0.0002) |

**All three pass**: every rank is within 0.003. Family carries the largest gaps, all positive (+0.0018
to +0.0025), and lepinet's own repeat moved family by +0.0023 on the same images. The protocol's
"retraining noise on the test fold is about 0.000" holds for species only; macro-F1 over 102 families
moves about 0.002 between runs.

**Validation curves** (set 1, species macro-F1, mini_trainer minus the reference run; lepinet's repeat
minus the same reference in brackets):

| epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| I | +0.0013 | -0.0014 | +0.0004 | +0.0000 | +0.0002 |
| BU | +0.0027 (+0.0018) | -0.0054 (-0.0028) | +0.0001 (-0.0006) | -0.0017 (-0.0013) | -0.0004 (-0.0007) |
| F | -0.0047 (-0.0060) | +0.0006 (+0.0024) | -0.0011 (-0.0007) | -0.0008 (-0.0006) | +0.0008 (-0.0007) |

Gaps reach 0.005 mid-training and close by epoch 5, as lepinet's repeat does. A per-epoch tolerance
of 0.001 would fail lepinet against itself (F species -0.0060 at epoch 1, BU family +0.0050 at
epoch 3), so the curves are judged against the repeat, and pass. Genus and family behave the same way
(epoch 5: I +0.0001 / +0.0015, BU +0.0004 / +0.0013 against the repeat's +0.0005 / +0.0014).

## The labels mix species (2026-11-04)

The mini_trainer session in tmux `nemo` found that the classes are GBIF `speciesKey`s, the species
that the legacy GBIF Backbone accepts for each record's matched name. Every Backbone synonymy error
is baked into the labels, ours included, because lepinet trains on the same parquet. Its crosswalk to
Catalogue of Life XR, GBIF's new taxonomy, has about 370 hand decisions and lives in mini_trainer's
ignored `.agents/local/nemo_species_colxr/`. It finds:
- **163 of 12,632 classes contain images of another species.** The worst is class 1924494, which
  holds 1,962 *Lycaena tityrus* and 1,836 *Brintesia circe*, a homonym. This one is confirmed here
  from `scientificName`, independently of the crosswalk.
- **281 classes share their species with another class**, so training splits one species over two
  classes. An example is *Pleuroptya ruralis* and *Patania ruralis*.
- **The quality filter** (7.49 M to 6.33 M images) removed 81 % of the images labelled as larvae, so
  species photographed mostly as caterpillars fell under the floor. The filter is the owner's: a
  classifier trained on the owner's annotations keeps adult-only images, dropping larvae, leaves,
  background, blur, subjects too far or too close, and microscopy. It lives in a local gbifxdl branch
  on another computer.

**On the trap set**, 7 of the 486 species (364 crops, 0.8 %) are in mixed classes and 12 (1,512
crops, 3.2 %) in duplicated ones, led by *P. ruralis* (682 crops) and *Euproctis similis* (472). Every
model so far saw the same labels, so comparisons between models stand; the absolute scores of these
species do not.

**Deferred until after the paper (owner, 2026-11-04).** The fix is a versioned label map from
`taxonKey` to species, built on the crosswalk with the epithet changes reviewed by hand. Nobody
involved is a moth expert, and every run would have to repeat.

## Pl@ntNet-300K on UCloud (2026-11-04)

For the Pl@ntNet head grid, `/12383016/plantnet300k/` holds Asger's own processed copy
(`/12348329/mini-trainer-ablations/datasets/plantnet`; converter `examples/plantnet/format.py` in
mini_trainer) in MT's format, labels and split unchanged. It is the data behind his Pl@ntNetV2
results. A delegated agent copied it; I re-checked the counts.

- **Version:** Pl@ntNet-300K **v2** (Zenodo 10419064: 1,000 classes, 306,087 images), not v1.
- **Asger's processing:** he renamed *Oreomecon* to *Papaver*, dropped 6 genus-only classes (158
  images) and merged classes that land on one legacy-Backbone key: *Ophrys sphegodes* (6 classes),
  *Ophrys fusca* (2), *Anemone baldensis* (2).
- **Contents:** 305,929 images, 512 px on the longest side (Asger trained at 384).
- **Split:** the official one. Test (`set` 0) 31,088, val (1) 31,097, train (2) 243,744; every
  species is in all three, with 2 to 7,207 training images (median 22).
- **Taxa:** 987 species, 325 genera, 111 families, up to 7 ranks.
- **Checked:**
  - the archives against his SHA256 manifest;
  - every JPEG decodes;
  - MT's adapter maps every label;
  - today's GBIF API gives his key for all 994 names;
  - the remote listing equals the local copy.
- **Calibration split:** the draft splits the test fold 10/90 into calibration and reporting at
  scoring time, so it belongs in the scorer, not the data.
- **Open for the owner** (Asger's choice stands until then): the 6-way *O. sphegodes* merge
  (subspecies under one GBIF key), and the 512 px ceiling (going above it means rebuilding from the
  41.8 GB originals).

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

**2026-11-04:**
5. The paper is to be finished next week; spend GPU minutes only on what it needs.
6. Phase 2 starts now, in parallel with the gate.
7. Phase 3 shrinks to its most important comparison, with instinct over a grid (see Phase 3).
8. The label correction waits until after the paper.
9. mini_trainer's per-epoch figure becomes optional and is off in every run from now on, after the
   name caching is tested.
10. With the merge to mini_trainer's main branch in view, the branch follows its agent rules
    (`AGENTS.md`) and keeps code and docs to the fewest words, without removing docs.
11. The paper reruns go ahead now, not after the paper, run in parallel over 11 days (until
    2026-11-15), while the owner writes the sections as results land.

**The campaign (2026-11-04).** One recipe, pinned on evidence (Phase 3), on mini_trainer's test set.
Multi-GPU jobs stage the images once and train one model per GPU. 9,087 B200 hours remain, against
about 650 needed.

| block | runs | starts |
|---|---|---|
| cross-entropy against EMLA (V2-S, BU) | 2 | now |
| head grid, Lepidoptera: F, BU, I, TD, AR x B0, V2-S, ViT-L/16, seed 42 (seeds 43 and 44 later, as confirmations) | 15 | after the gate |
| the same grid on Pl@ntNet-300K, seed 42; epochs and image size from Asger's Pl@ntNet runs (the data is 20x smaller) | 15 | data on UCloud (2026-11-04); specs next |
| shift on V2-S with BU, like the original evidence (B3, T1rep, X1, X2 and L4 were all at 20 M). The teacher is the cross-entropy arm; self-training at 0.24, 2.6 and 12.3 % of training rows; real labels at 2.6 %; X1 (12 folds) and X2 (3); cRT | ~20 | after the teacher's pseudo-labels |
| DINOv3 ConvNeXt-L with BU, the paper's large backbone at B8's settings (320 px, batch 96): no adaptation, then self-training at 2.6 % | 2 | after the gate |
| open-set: C3b's withheld taxa, 5 heads on V2-S | 5 | as soon as the specs are ready |

The owner approved every campaign submission, DINOv3 for the paper (BioCLIP-2 may come later for
the AMI deployment model), and the Pl@ntNet download.
The campaign uses channels_last (owner, 2026-11-04): 25-35 % faster for the convnets in V3, the same
function computed in another summation order, so differences should be run-to-run noise. One control
run checks it: the gate's BU spec with channels_last as the only change, against V2BU on the same
629,742 test rows. Pass: within 0.003 at every rank.

## Log

- 2026-11-03: the decision; phase 0 assigned to the `mt` session; the Muon finding. Phase 0 done the
  same day; phase 1 approved. Two mistakes of mine in the first brief were corrected: label smoothing
  is not a difference, and the flat head's F1 column is valid (I had cut the wrong CSV columns). V0F
  and V0BU started.
- 2026-11-03, evening: phase 1 implemented (six local commits, all opt-in, MT's tests pass); every
  CPU check matches lepinet exactly, augmentation draws and optimiser groups included. The subset
  side-by-side found that lepinet leaks GPU memory per batch without `GCCallback` (the port's D3 claim
  was wrong; a B200 hides it), so lepinet's subset runs use `extra_cbs=[GCCallback()]` (numerics
  unchanged), and the throughput job runs lepinet as on UCloud. The owner
  approved the selection and the parity setting. The reference conventions in the brief were
  corrected. Nemo's card set was judged unreconciled, so Nemo is re-scored on `data/flemming`.
- 2026-11-03, night: the subset side-by-side (V1) passed. On set 0, mini_trainer is within 0.0007 of
  lepinet at every rank at threshold 0, and 34 of 35 per-epoch validation cells are within the rule.
  The exception is family macro-F1 after epoch 2 of 5: mini_trainer +0.0111, tolerance 0.0089. I
  read it as noise, for three reasons:
  - With two seeds per code, a permutation test cannot give p below 1/6.
  - mini_trainer's family lead on validation (+0.002 to +0.019 per epoch, +0.0068 at the end) is
    +0.0004 on set 0, and family accuracy is equal (+0.0001). There are 40 families, so a few
    small ones move the macro average.
  - The two codes' learning-rate curves agree to four digits.

  Gate (b) is approved. Each job stages its images to node-local disk with `ucloud/stage.py` (5.67 M
  files in 12.6 min in July). mini_trainer submits I first, then BU and F, each after the previous
  job's staging finishes.

  One claim of mine was wrong. I told the `mt` session that `/work` caps at about 1,000 img/s
  across jobs. I took that from V0's first epoch, which was slower in both jobs for a reason not
  established (training 1:46 and 1:35, validation 8:50 and 20:00). From the second epoch on, each
  V0 job reads about 1,000 img/s, so two parallel jobs do not starve.
- 2026-11-04, early: V0 scored (see its section). The test-fold prediction held; the trap prediction
  was half wrong and the Nemo one wrong in V0BU's favour. The four mini_trainer jobs are not
  submitted: the "go ahead, submit all four" in the `mt` input box is Claude Code's dimmed
  suggestion, not the owner's message. V0's slow evaluations exposed a gap in the gate specs, which
  the `mt` session fixed before submission. Only sets 1-9 were staged, and predict would have read
  set 0 from `/work` with 32 workers, about 2 hours, past its 2-hour limit. Setup now stages set 0
  too. Predict also keeps the 3,171 rows whose species are outside the gate models' classes, so the
  gate is scored on lepinet's 629,742 rows, matched by file. Checking that predict handles those rows
  in a real run found a second gap. mini_trainer rewrites run names (`gate-I` saves to `gate_I/`), so
  the run guard would never have seen a finished run, and all three gates would have stopped without
  predicting after about 8 hours. The guard now reads the path from the trainer's own "Final weights
  saved at:" line. Tested on the real trainer for all three heads.
- 2026-11-04, evening: Pl@ntNet-300K is on UCloud (see its section). The `mt` session proposes the
  grid's specs. V3 (GPU options, F preset, one epoch) is done: channels_last, torch.compile
  reduce-overhead and the compiled optimizer with CUDA graphs reach 2,242 img/s against 1,335 (1.68x),
  with the same accuracy (64.48 %) and peak memory down from 6.6 to 4.3 GiB; channels_last alone is
  about 1.3x. Whether to compile waits for the owner. The full-scale gate passed for all three heads
  (see V2).
