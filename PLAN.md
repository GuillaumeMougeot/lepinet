# PLAN — what we are doing now

**Kind:** living · **Last updated:** 2026-11-18 · **Supersedes:** [2026-07-28-landscape-and-plan](archive/journal/engineering/2026-07-28-landscape-and-plan.md)

The one file about *today*. Experiment IDs resolve in [`RESULTS.md`](RESULTS.md); reasons are in
the linked journal entries; older versions of this file are in git history.

## 0. The focus: one code base, mini_trainer

Owner, 2026-11-03: training moves to mini_trainer (Asger maintains it); lepinet runs in parallel for
a while. The mini_trainer session (tmux `mt`) does the work on branch `feature/lepinet-integration`,
supervised from here. [[2026-11-03-moving-the-recipe-into-mini-trainer]]

| phase | what | state |
|---|---|---|
| 0 | parity audit: lepinet's recipe against MT's defaults, component by component | done (2026-11-03): every CPU check exact (initialisation, logits, decoding, augmentation, optimiser groups, scorer) |
| 1 | port the recipe as opt-in pieces (√-oversampling sampler, presizing and fastai-style affine, one-cycle for every group, CE with label smoothing 1/C and no regulariser, learned head scale and bias); gate: test F1 within 0.003 at every rank for I, BU and F, same per-epoch curve, same img/s | done (2026-11-04): the subset side-by-side (V1) and the full-scale gate (V2) passed for I, BU and F, every rank within 0.0025 of lepinet on the test fold; V2T: training loop 1.185x lepinet's |
| 2 | the methods: F head, frozen-backbone stages (cRT, adaptation), self-training on trap crops, trap-like augmentation, image cap, the evaluation (grouped calibration, probe, open-set, back-off) | started 2026-11-04, in parallel with the gate; the trap evaluation first |
| 3 | settle the recipe by single-factor runs: sampler, EMLA vs CE, regulariser, schedule, resize and augmentation, epochs and size; then the new baseline on MT's test set (12,632 species) | anchors done (V0F, V0BU, 2026-11-04): without the floor the test fold scores 0.9110 / 0.9107 and the traps are unchanged; BU is 5 pt above F under shift and level with or above Nemo. No grid (owner, 2026-11-04); proposed: one comparison, cross-entropy against EMLA without oversampling (two runs), the rest on evidence |
| 4-6 | the paper reruns; Nemo re-scored on the reconciled Flemming export (its card used the unreconciled one) next to the new models; the third production model | reruns now, in parallel, until 2026-11-15 (owner, 2026-11-04): head grid on Lepidoptera and Pl@ntNet, one seed now (12 + 12, four heads; seeds 43 and 44 after seed 42's results), shift results on V2-S (~21, like the original evidence) plus a DINOv3 pair at 198 M, open-set on C3b's withheld taxa (4) |

## 0b. Otto, the production model for AMI traps (owner, 2026-11-07)

On free GPUs, after the paper's jobs. [[2026-11-07-otto-a-production-model-for-ami-traps]]

| step | what | state |
|---|---|---|
| 1 | schedule and trap augmentation | done: 10 epochs (384x10 +4.7 probe); trap aug **off** (test -0.75, 7x seed spread) |
| 2 | BioCLIP-2 trains in mini_trainer | done (normalisation bug fixed, MT ce5b7e7); ensemble teacher checked (0.670 / 0.503 probe / probe-HO) |
| 3 | candidate (Otto, BioCLIP-2 ViT-L BU, ensemble pseudo-labels K=35) and DINOv3 twin | done (RESULTS OT): Otto trailed P5 on probe; cause found in the labels (71 % correct) |
| 4 | three site folds (LV, OH, SS) | done (RESULTS OT) |
| 4a | round 1, one factor each from Otto: OT1 gated labels (70 % gate, 98.6 % correct), OT2 TrivialAugmentWide, OT3 both, OT4 OT3 on DINOv3 | done (RESULTS OT1-OT4): OT3 best, probe / probe-HO 0.674 / 0.413 (ne, tta3bm, zero threshold), +0.14 over P5; OT4 ties on probe, below on probe-HO |
| 4b | round 2 (OT5): OT3's recipe, labels re-gated with the P5 + OT3 + OT4 teacher (19,061 crops, 99.0 % correct, 1,041 swapped), seed 42 | done (RESULTS OT1-OT5): within noise of OT3 everywhere (probe 0.688, probe-HO 0.415); label rounds have saturated. Owner to choose the production model (OT3 or OT5) |
| 4c | control (OT6): OT3's recipe with no pseudo-labels, to test whether self-training's gain is mostly the fauna prior (RESULTS Z9-Z15); OT3 is Otto (owner, 2026-11-16) | done: calibrated probe 0.700 against OT3's 0.749 (-4.9), probe-HO level (0.557 against 0.562); at zero threshold -15 points. The list + estimated-prior check is with lepinet-2a |
| 5 | comparison with Meghan (MAMBO_v2) and Nemo (MAMBO_v3): their pipeline, Asger's figure plus Otto, speed and size; every model with and without the north_europe list and TTA, headline north_europe + TTA (Nemo's card setting; owner) | after 4 |
| 6 | release through mini_trainer's release path, int8 and fp16, regional lists | after 5 |

Postponed (owner, 2026-11-07): the Tree of Life run (W3); first step when it resumes is the upload, the parquets and a one-epoch throughput pilot.

Side request (2026-11-11), done: OneStop figure, Nemo against MILA's UK & Denmark classifier on recent GBIF photos of invasive insects per Living Lab list, `data/onestop/onestop_per_list.png`. [[2026-11-11-onestop-nemo-against-mila-on-invasive-species]]

## 1. Two papers (owner, 2026-11-10: the merged paper is split)

The merged draft had two storylines and two recipes; the owner split it (memo to Asger, 2026-11-10).
Merge Desk pieces are tagged `[A]`, `[B]` or `[A+B]` in their title: [Merge Desk](https://claude.ai/artifact/KFNuRZrWnf39pLdUwvPyTP).
Numbers per piece: `/data/au761367/paper_numbers/NN.md`. Decisions: [[2026-10-23-which-recipe-does-the-merged-paper-rerun]].

### Paper A: hierarchical heads and metrics (Asger leads, his Overleaf draft)

Our pieces for it: 01, 02, 09, 10, 11, 12, 13, 16, 19, 22.

**Owner**
- [ ] Send the memo; agree with Asger: split, author order, deadline.
- [ ] Decide with Asger the one head grid (his EMLA runs, or the three-seed campaign) and EMLA's status.
- [ ] In Overleaf: delete the old self-training section (stale, moves to B) and the open-set TODO in Global Lepidoptera.
- [ ] Hand over the A pieces (paste, or let Asger paste) once the grid is decided.
- [ ] Verify the citations written from memory in A's pieces (Ren 2020, Menon 2021, Deng 2012, AMI hardware).

**Agent**
- [ ] If Asger keeps his grid: recompute the A pieces' analyses on his runs (13, 19, 22, 12; mini_metrics pipeline shared).
- [ ] Draft for Asger on request: results text around his figures, the hierarchy discussion.

### Paper B: species classifiers under domain shift (owner leads; base: our pieces and `paper/DRAFT.md`)

Pieces: 03, 04, 05, 07, 08, 14, 15, 17, 18, 20, 23 (plus 01, 02, 10, 11 shared).

**Agent**
- [ ] Outline and standalone LaTeX draft (intro, methods, results, discussion) from the pieces.
- [ ] Add: test-time augmentation (VY), TrivialAugmentWide (AG), AMI-Traps as a second, public shift set (PB), the label-noise paragraph.
- [ ] One external adaptation baseline (test-time adaptation or a FixMatch-style method): GPU, to plan.
- [ ] Full draft to the owner in about two weeks (target 2026-11-24).

**Owner**
- [ ] Target venue; author list.

## 2. Running now

| ID | what | state | next |
|---|---|---|---|
| **Campaign** | the paper reruns on mini_trainer, run by the `mt` session (journal: the campaign table) | done and scored: c-b0 and the Pl@ntNet grid, seed 42 (RESULTS VB0, VP); c-v2s, i.e. the V2-S grid, cross-entropy against EMLA (cross-entropy stays) and the channels_last control (passed) (VS, VC); c-t1rep, where T1rep reproduced (VT); c-openset, where C3b reproduced (VO); pp-dinov3, below V2-S on Pl@ntNet (VP). Running (2026-11-05, UCloud): c-vitl (lr 1e-4, for the appendix), c-openset, c-dinov3-r2 (no adaptation), c-dinov3-st, c-crt-t2, c-shift-c, the DINOv3 grid (c-dinov3-grid, pp-dinov3); the rest of the shift block resubmitted as four 4-GPU jobs (c-shift-a1, a2, b1, b2), since 8-GPU jobs wait for a whole node. Built, waiting for the owner's approval: c-crt-t2, i.e. cRT and classifier-only adaptation, each with the whole network and with a frozen trunk (4 runs, `--after c-v2s`), since lepinet's "frozen" stages were not frozen. Owner, 2026-11-05: reproductions first; seeds 43 and 44 after seed 42's results; AR dropped (it does not learn in mini_trainer); DINOv3 ConvNeXt-L replaces ViT-L in both grids (queued after c-v2s: c-dinov3-grid with F, I and TD, about 15 h each; pp-dinov3 with all four); EfficientNetV2-B0 shelved; the TD gap waits on mini_trainer's saturation test (a clamp after the learned scale), with a fix before the paper if confirmed | **Weekend 2026-11-07/08, unattended:** everything is queued in advance (no runtime decisions; dependencies via `--after`, 4-GPU jobs): seeds 43 and 44 of the V2-S and B0 grids on both datasets, Pl@ntNet with √-oversampling and EMLA, DINOv3 on Pl@ntNet at lr 2e-4, DINOv3 seeds after c-dinov3-grid, TD and F at 10 epochs and 384 px, one BU run at 30 epochs. `dev/100` scores every finished output into `data/campaign/summary.md` (cron, every 30 min) |
| **W1** | TreeOfLife-200M at short side 256, on the workstation (`/data/au761367/tol256s/`; zero UCloud core-hours) | crawl done: 64.16 M images plus ~0.54 M retried ([crawl](journal/engineering/2026-10-01-the-crawl-resized-the-wrong-side.md)); `retry3/` (9 servers, 72 h cap) | shared as `/data/datasets/tol256`: 65.2 M images, metadata with ToL's per-image fields ([share](journal/engineering/2026-10-21-sharing-the-tol-crawl.md)). **Upload re-planned** (1,053 shards); old shards in the UCloud Trash; upload paused on storage |

## 3. Backlog, in order

| ID | work | cost | why |
|---|---|---|---|
| AG | broad augmentation + TTA as a generic adaptation method (owner, 2026-11-07), cut to 4 runs, each 1 B200, in parallel with Otto: V2-S BU 384x10 with geometric, geometric + trap, TrivialAugmentWide (control v2s_bu_384x10_s42), and Otto + geometric | ~75 B200 h; mt implements the two augmentations, owner approves in mt | ships in mini_trainer if it keeps the test fold; Otto + geometric answers whether Otto benefits. [[2026-11-07-can-augmentation-and-tta-be-the-generic-adaptation-method]] |
| TAX | **after the paper is submitted (owner):** correct the training labels with one command, `.venv/bin/python dev/101_colxr_relabel.py apply --upload` (built and trial-run 2026-11-08); then measure the effect and retrain production models | CPU first; retraining decided from the measurement | 163 mixed and 281 duplicated classes come from legacy GBIF Backbone errors. [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]] |
| PB | a small check on public benchmarks after Otto (owner, 2026-11-07): iNat21 Lepidoptera val (closed-set on our species, the rest as an open-set slice), BioCLIP's Rare Species and its frozen-probe task suite, and a public light-trap set (AMI, to confirm); Otto, P5, Nemo, Meghan, BioCLIP-2 frozen as reference | eval only; mt plans it (sizes, label mapping, published protocols and numbers) | our numbers are all on our own data; this places them next to published ones. Plan: ~29 GB downloads, under 1 GPU-hour; AMI-Traps first. [[2026-11-07-where-do-our-models-stand-on-public-benchmarks]] |
| Q | the back-off policy on Pl@ntNet: Lepidoptera and traps done (no single rule wins; greedy with a lower genus target is the cheapest fix) | CPU only; needs the colleague's predictions (`dev/097 --plantnet`) | does the in-distribution ranking hold for another kingdom and five ranks? [[2026-10-23-what-is-the-best-back-off-policy]] |
| U | self-training on EU trap crops (>50 M): natural vs balanced, then share sweep, teacher B8 | 4-7 runs at 20 M, after the crop location and metadata are known | the largest lever, without copies. [[2026-10-21-self-training-on-european-traps]] |
| TF | **done 2026-11-09** on `/data/datasets/tol256` (`dev/102`): 2.6 M rows moved to `Genus_epithet` with ranks realigned, `taxon_rank` marks 2.6 M genus- and family-level rows (`species` empty), 74 k rows below the recomputed floor marked `below_floor`, 404 k images re-fetched for the corrected species, 4,791 rows with bad source labels removed, Lepidoptera names moved to COL XR (507,927 rows; `data/global` and TAX untouched); `dev/082` fixed. **Frozen 2026-11-12 as tol256-v1** (`dev/103`, applied by `dev/102 apply`): 65.71 M rows, 63.56 M usable, `split` 90/5/5 by observation, 1,810 held-out species, Rare Species overlap excluded; 111 k re-fetched images in; README in the dataset. Upload (`dev/104`): pilot at `/12383016/tol256-pilot` (262 shards, 261.5 GiB, 2026-11-13); full set at `/12383016/tol256` (1,832 shards, 1.79 TiB, 2026-11-15). Left: 12 k Riodinidae rows lack family and order (goes with TAX); 88 retry4 rows at genus rank that are species records | done | merged taxa and missing reptiles and fishes would have entered every ToL model. [[2026-11-08-does-a-margin-objective-make-a-better-pretrained-biodiversity-model]] |
| ME | after the paper, after TF and TAX: a cosine-margin objective against contrastive training on ToL-65M, same images and backbone; evaluated on BioCLIP-2's published suite (15 tasks), AMI-Traps and iNat21 | hundreds of B200-hours per arm at least; sharded loading first | is the objective what a better pretrained encoder needs? [[2026-11-08-does-a-margin-objective-make-a-better-pretrained-biodiversity-model]] |
| W3 | training parquets for the ToL run: **built 2026-11-12** (`/data/datasets/tol256/train/`: 60.9 M species rows, 190,199 species, seven levels keyed by full path, a tree by assertion, every species with train rows); mini_trainer shards on UCloud (`/12383016/tol256`, pilot `tol256-pilot`); **run plan fixed 2026-11-16** (owner: DINOv3 ConvNeXt-L, hierarchical head over seven ranks, no TrivialAugmentWide, no margin, one seed, one-arm pilot; W3 entry, fourth revision): lr 8e-4 (range test, RESULTS LF); smoke **passed** (12463902: finite loss at 7 ranks, weights equal on 8 ranks, resume works, ~5,650 img/s on 8 B200, so 5 passes ~14 h, ~110 B200-h); pilot **passed** (12464304: one pass of 8.09 M in 41 min, 5,730 img/s, val acc1 5.2 -> 38.1 %, about 7 B200-h); **full run done** 2026-11-17 (tol-full 12464522: 13.2 h, about 106 B200-h, GPU utilisation median 97 %, val species macro-F1 0.295 -> 0.640 over 10 half-pass epochs, acc1 80.7 %); TL3 scored (moth head below VD, confounded), TL2 open-set scored, TL1 scored (frozen ToL above BioCLIP-2 on Rare Species and held-out taxa); resolution check done (a larger test input is worse); TL4 done (ToL trunk fine-tuned: test fold +0.6 over VD's three seeds, traps level); pending Rare Species on TL4 (tol-full: 5 passes as 10 virtual epochs, warmup 1, GPU-utilisation log, 8 B200, 20 h limit without auto-extend, worst case 160 B200-h; expected 14-14.5 h, about 115 B200-h); full run from 23 Nov after the owner approves the measured spec | GPU, 8 B200, about 80-100 B200-hours (estimate; lr 8e-4) | the owner's question: does seeing the whole tree help open-set and shift? Also gives a trap order classifier. [[2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know]] |
| GS | global_lepi as mini_trainer shards (owner, 2026-11-16), so Lepidoptera jobs stop staging 5.7 M files (OT6 staged 6+ h at 220 files/s) | **uploaded** to `/12383016/global_lepi_shards-v1`: train 5,063,857 / val 633,224 / test 632,913 in 475 shards, 473 GiB, original bytes, hash-shuffled; class_spec byte-identical to the campaign's; 200/200 samples pixel-identical to the files. `verify --full` passed (job 12464416: 6,329,994 samples, 475 shards). Left: a one-epoch check against a finished staged run (gl-check, on hold by the owner). Plain runs only; Otto-style pseudo-label merges keep staging | GPU, none (built locally) | stops idle GPUs at job start |
| FL | the 50-image species floor, now that genus and family are supervised (owner, 2026-11-17) | **counted**: Lepidoptera +29-38 k species but +6 % images; ToL +0.1 % images on disk, 599 k species in the catalog 99 % undownloaded; C3's unseen set is these species. Recommended: coarse-only supervision (needs a per-rank loss mask, one run); owner to decide | CPU done | [[2026-11-17-should-the-species-floor-go-now-that-coarse-ranks-are-supervised]] |
| S1a | two organisms in one picture: composites (moth on moth, moth on trap background) through B8 and P5; does the open-set score flag them as novel? Per-location logits as a readout | workstation GPU, hours | the owner's question (2026-10-09); also decides how W3 is read out (whole tree vs subtree). [[2026-10-09-two-organisms-in-one-picture]] |
| W2 | our objective trained on ToL-10M, compared with BioCLIP-1 on the same data | ~2.5 h/epoch | isolates the objective from the data. [[2026-08-28-two-directions-checklists-and-our-objective-on-tol]] |
| O2 | open-set as enrolled taxa grow from 12 K to 204 K, on cached ToL embeddings | 1 GPU, no download | does the scoring-rule result hold across class count too? |
| K1+ | regional checklist **plus** abstention | eval only | should recover the tail damage K1 measured |
| Z | routes beyond scaling (2026-11-13 review). **First steps done 2026-11-15** (RESULTS Z5-Z8); the owner's question on label shift against domain shift answered the same day (Z9-Z15: the calibrated macro-F1, tta3bm and seeds, a prior-free measure of discrimination, crop-by-crop overlap, the Otto two-by-two, each site as a new site). **Owner to choose:** (a) paper B's baseline: self-training reported against the regional list, and against the list with an estimated prior; (b) what else would close the gaps in the evidence: two more seeds of `v2s_bu_st26`, a labelled trap set from another country (the control for Otto is running, row 4c; when its logits land, Z16 scores the list and the estimated prior on it); (c) routes 2, 5, 6 and the four 20 M runs. **2026-11-17, the first ToL model (owner's question, Z17-Z20, TL1 run here):** frozen it beats BioCLIP-2 as an encoder; its moth head trails VD through the label space, resolution and confidence; more ToL passes not recommended. **TL4 checked 2026-11-18 (Z21):** the moth recipe erases the ToL trunk's generality and it gains nothing on traps, so TL5 as planned is not recommended; a fine-tune that keeps the start (route 5) is the run worth doing | workstation only so far; training runs if wanted | most of self-training's probe gain is a prior on the fauna that a list and an estimated prior give with no training (RESULTS finding 4b): within 1.5 and 0.6 calibrated points of the self-trained twin with tta3bm; OT3 stays 3 to 6 points above every model that never saw a trap; marginal supervision makes congeners neighbours on three backbones (handed to the lepinet session for paper A). [[2026-11-13-beyond-scaling-what-would-make-a-better-biodiversity-backbone]] |
| — | trap order dataset | owner | owner is transferring it; needed to score W3's order classifier |

## 4. Decided not to do — each closed for a reason

- **More in-distribution accuracy**: saturated; the headroom is under shift.
- **LDAM, background suppression**: classifier adaptation subsumes that family (T2b; on hold, its trunk was not frozen: [[2026-11-05-the-frozen-stages-were-not-frozen]]).
- **Re-tuning the ArcFace margin**: two cheap proxies failed for principled reasons; the margin is worth ~0.8 pt.
- **The autoregressive head**: lost by 20 pt.
- **Uniform sampled softmax, low-rank or proxy-free heads for 1 M species**: all measured dead (H2-H4); a 50-image floor makes the matrix fit instead.
- **Re-litigating staged vs end-to-end**: a tie at 198 M once the noise floor was measured (G3b). The stages were short end-to-end fine-tunes, not frozen (2026-11-05).

## 5. Caveats to carry

- **The conv-backbone "frozen" stages trained the whole network** (2026-11-05): L4, T2, F2, the G and R series. The claims built on them are on hold until the mini_trainer reruns: the spine (C2), "cRT removes the trade", the staged recipe. BioCLIP-2's frozen stages held. [[2026-11-05-the-frozen-stages-were-not-frozen]]
- **Noise floors depend on the training regime.** Probe spread is ~0.004 between end-to-end runs and ~0.012 between short fine-tuning stages (the "frozen" stages above); identical 198 M staged runs differed by 3.74 pt once (G3b). On the test fold, retraining moves species macro-F1 by ~0.000 and family by ~0.002 (A5, V2). Claim nothing under ~3x the matching floor.
- **Thresholds do not transfer between models or embedding spaces.** Rankings do; abstention, novelty and temperature values must be refitted.
- **The labels mix species.** The classes are legacy GBIF Backbone `speciesKey`s: 163 contain images of another species and 281 share a species with another class. 19 of the 486 trap species are affected (1,876 crops). Comparisons between models stand; the absolute scores of those species do not. The correction waits until after the paper (owner, 2026-11-04). [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]] One result did not stand (2026-11-13): N2's near-novel trap stratum is 92 % *Pleuroptya ruralis*, a duplicate of *Patania ruralis*, which the model was trained on. Check any withheld-species set against COL XR before reading it. [[2026-11-13-beyond-scaling-what-would-make-a-better-biodiversity-backbone]]
- **Every lepinet head learned a per-class scale and bias.** fastai's optimiser setup trains the head's "frozen" `g` and bias (found 2026-11-03), so the rows are not unit-norm (species `g` mean 1.88). No reported number changes; mini_trainer keeps them frozen, so parity needs an option and phase 3 tests learned against frozen. [[2026-08-06-the-cosine-head-is-not-unit-norm]]

## 6. How to run things

Cluster rules are in [`ucloud/README.md`](ucloud/README.md). **Scale discipline (owner):** test a new mechanism at
20 M; promote to 198 M once, when the recipe stops moving. Costs: ~1,100 img/s at 20 M, ~480 img/s
at 198 M; a 5-epoch 20 M run is ~6.4 h.
