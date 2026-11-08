# PLAN — what we are doing now

**Kind:** living · **Last updated:** 2026-11-08 · **Supersedes:** [2026-07-28-landscape-and-plan](archive/journal/engineering/2026-07-28-landscape-and-plan.md)

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
| 3 | candidate: BioCLIP-2 ViT-L, BU, 12,632 species, Muon 1e-5, 224 px, 10 epochs, ensemble pseudo-labels K=35 at x16; DINOv3 twin | **running**: candidate 12450599, c-otto-dinov3 12450503 (1 B200 each); ~32-40 h |
| 4 | three site folds (LV, OH, SS), so that every Danish crop is scored by a model that never saw its site | x2lv 12450600 running, x2oh 12450601 and x2ss 12450602 queued (1 B200 each) |
| 5 | comparison with Meghan (MAMBO_v2) and Nemo (MAMBO_v3): their pipeline, Asger's figure plus Otto, speed and size; every model with and without the north_europe list and TTA, headline north_europe + TTA (Nemo's card setting; owner) | after 4 |
| 6 | release through mini_trainer's release path, int8 and fp16, regional lists | after 5 |

Postponed (owner, 2026-11-07): the Tree of Life run (W3); first step when it resumes is the upload, the parquets and a one-epoch throughput pilot.

## 1. The goal: a submittable paper

Every experiment [`paper/DRAFT.md`](paper/DRAFT.md) uses is closed. What remains is scoping, writing
and two evaluation-only runs. **Target: finished next week** (owner, 2026-11-04); GPU time goes only
to what the paper needs.

| # | step | who | state |
|---|---|---|---|
| 1 | **One paper, merged with Svenning et al.'s draft** (`~/Documents/hierarchical_classification.pdf`; owner, 2026-10-22: no second paper). Proposed outline: methods = data (Lepidoptera, Pl@ntNet, traps with the grouped split), HGLL head, the five heads, one recipe, evaluation (metric conventions, retrain noise); results = hierarchy (parameters vs supervision, alignment, novelty by taxonomic distance), evaluating under shift (conventions, scoring rules, contamination), long tail, target-domain data, classifier stage, abstention; appendix = margin, nuisance augmentation, head scaling, recipe ladder, AR decoder, EMLA and regulariser. Open: one training recipe for both drafts (EMLA vs √-oversampling), the colleague's raw trap predictions rescored with `dev/092` | owner + colleague | open |
| 1b | **Pin the rerun recipe before any rerun** (owner, 2026-10-23: the colleague's settings have priority; some experiments will be redone). 16 decisions with both drafts' settings and our evidence: species floor, sampling, loss (EMLA vs CE), regulariser, epochs, image size, backbones, heads, margin, seeds, dose, open-set split, budget. [[2026-10-23-which-recipe-does-the-merged-paper-rerun]] | owner + colleague | reframed 2026-11-03: settled by measurement in mini_trainer (§0, phase 3) |
| 2 | Insert our pieces into the Overleaf draft through the Merge Desk (`paper/merge/`, `dev/094`): all 23 steps drafted (02 pasted; 04 on hold for 1b). Results steps 13-22 and the appendix (23) carry recipe-L numbers in mini_metrics' current convention, trap scores calibrated on grouped splits; each says in its check line what the rerun replaces | owner, agent | ongoing |
| 3 | Re-score §4.5 with each head's own rule; measure open-set under shift for B8 and P5 | GPU, eval only | done (N2, N3; Merge Desk step 14) |
| 4 | Verify every citation in §1b and the references (written from memory) | owner | open |
| 5 | Consistency pass: every number against its journal entry | agent | — |
| 6 | Re-run A2's scoring rules: `rules-A2.json` was overwritten on 2026-08-03; §4.9 uses the 2026-08-01 file | GPU, eval only | open |

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
| TF | **before any ToL training:** correct the rank shift in the ToL labels: 2.6 M rows in 6,082 bare-epithet folders (mostly reptiles and fishes), recoverable from `scientific_name`; rename the files to their `Genus_epithet` folders and rewrite the rank columns, by dry-run plan then journalled apply, on the shared copy and on UCloud; audit the 12,151 capitalised single-word folders | CPU, no re-download | merged taxa and missing reptiles and fishes would enter every ToL model. [[2026-11-08-does-a-margin-objective-make-a-better-pretrained-biodiversity-model]] |
| ME | after the paper, after TF and TAX: a cosine-margin objective against contrastive training on ToL-65M, same images and backbone; evaluated on BioCLIP-2's published suite (15 tasks), AMI-Traps and iNat21 | hundreds of B200-hours per arm at least; sharded loading first | is the objective what a better pretrained encoder needs? [[2026-11-08-does-a-margin-objective-make-a-better-pretrained-biodiversity-model]] |
| W3 | training parquets for the ToL run: after TF; join the catalog by uuid, seven levels keyed by full path, check the hierarchy is a tree, drop the audit's 584 k excluded rows; then the 20 M run | GPU, after W1 | the owner's question: does seeing the whole tree help open-set and shift? Also gives a trap order classifier. [[2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know]] |
| S1a | two organisms in one picture: composites (moth on moth, moth on trap background) through B8 and P5; does the open-set score flag them as novel? Per-location logits as a readout | workstation GPU, hours | the owner's question (2026-10-09); also decides how W3 is read out (whole tree vs subtree). [[2026-10-09-two-organisms-in-one-picture]] |
| W2 | our objective trained on ToL-10M, compared with BioCLIP-1 on the same data | ~2.5 h/epoch | isolates the objective from the data. [[2026-08-28-two-directions-checklists-and-our-objective-on-tol]] |
| O2 | open-set as enrolled taxa grow from 12 K to 204 K, on cached ToL embeddings | 1 GPU, no download | does the scoring-rule result hold across class count too? |
| K1+ | regional checklist **plus** abstention | eval only | should recover the tail damage K1 measured |
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
- **The labels mix species.** The classes are legacy GBIF Backbone `speciesKey`s: 163 contain images of another species and 281 share a species with another class. 19 of the 486 trap species are affected (1,876 crops). Comparisons between models stand; the absolute scores of those species do not. The correction waits until after the paper (owner, 2026-11-04). [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]]
- **Every lepinet head learned a per-class scale and bias.** fastai's optimiser setup trains the head's "frozen" `g` and bias (found 2026-11-03), so the rows are not unit-norm (species `g` mean 1.88). No reported number changes; mini_trainer keeps them frozen, so parity needs an option and phase 3 tests learned against frozen. [[2026-08-06-the-cosine-head-is-not-unit-norm]]

## 6. How to run things

Cluster rules are in [`ucloud/README.md`](ucloud/README.md). **Scale discipline (owner):** test a new mechanism at
20 M; promote to 198 M once, when the recipe stops moving. Costs: ~1,100 img/s at 20 M, ~480 img/s
at 198 M; a 5-epoch 20 M run is ~6.4 h.
