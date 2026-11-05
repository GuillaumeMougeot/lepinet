# Results

Section 1 is what the project has established; section 2 resolves every experiment ID (A1, B8,
P5...) and is where new results go. The reasoning is in the linked [`journal/`](journal/README.md)
entries. The July local-run table is in [`archive/RESULTS-july-local-runs.md`](archive/RESULTS-july-local-runs.md).

## 1. What is established

One line each, with the paper section and the experiments (section 2) that carry it. Negative results count.

### 1a. Science

| # | finding | paper | IDs |
|---|---|---|---|
| 1 | Hierarchy-aware heads do not help; one species head plus marginalisation matches or beats them. Coarse *supervision* (a loss on the marginals) still buys ~1.4 pt under shift. | §4.1 | singlehead, marginal, A4 |
| 2 | In-distribution accuracy saturates near 0.93 but falls ~23 pt on trap images, and the three axes rank models differently. | §4.2, §4.10 | flemming, B4 |
| 3 | **On hold ([2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)): L4 and T2 trained the whole network.** **The spine:** interventions belong in the classifier, not the representation: rebalancing (cRT), domain adaptation (83 % from the classifier alone), the prototype matrix (centroids for 0.29 pt, for the margin head; 3.7 pt for the plain head). | §4.15 | L4, T2, H1 |
| 4 | Unlabelled trap images are the largest lever: +7.94 probe from self-training at 2.6 % of training rows, no target labels; more becomes memorisation. Real labels at the same dose score +1.01 but lose the gain on unseen species (91 % of their errors land in the labelled fauna). At a new trap accuracy keeps all the gain, macro-F1 64 % (new site 23 %). | §4.11 | B3rep5x, T1, T1rep, X1, X2 |
| 5 | Long-tail reweighting trades shift robustness for accuracy, monotonically; cRT removes the trade (on hold, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md): L4 was two late epochs of oversampling on the whole network); capping training at 1,000 images per species helps shift. | §4.13 | L0-L4, L7 |
| 6 | The best open-set scoring rule changes with model scale (6-7.6 pt); one rule for all models produced a false ranking. | §4.9 | E2 |
| 7 | The angular margin relocates open-set signal rather than creating it: +0.78 AUROC best-vs-best (an earlier "31 pt" compared rules, and is retracted). | §4.3 | A1, C3r |
| 8 | Novelty detection improves with taxonomic distance (near < mid < far), and not because unseen taxa are rare. | §4.4 | C3, C3b |
| 9 | Abstention under shift is expensive (13-19 % vs 0.8 % in-distribution) and yields a composition, not a score: tied B8 and P5 trade species answers (74.7 vs 67.4 %) for answers at any rank (77.2 vs 86.4 %). A margin head needs a temperature first. | §4.6a | O1, O3 |
| 10 | Two thirds of our test fold is inside BioCLIP-2's training data by GBIF occurrence id; a frozen probe understates that model by 7 pt; fine-tuned it is the better trunk, and our recipe closes the gap. | §4.14 | P1-P5 |
| 11 | A margin damages marginalisation more than classification, through calibration; replicated at 10x scale. | §4.7 | A1, A2, A4 |
| 12 | Augmentation that mimics trap conditions closes only 17 % of the gap. | §4.8 | B1 |
| 13 | Measurement: macro-F1 does not decompose over subsets; averaging over predicted-only species too (mini_metrics) turns 0.69 into 0.15 for the same predictions, and re-scoring every run that way changes the trap conclusions about labels, checklists and dose ([rescore](journal/2026-10-22-rescoring-in-the-mini-metrics-convention.md)); noise floors depend on the training regime (identical runs 3.74 pt apart); give every arm its own best configuration. | §3.2, §5 | A5, G3b, R5b, [conventions](journal/2026-10-22-why-the-trap-scores-of-the-two-drafts-differ.md) |
| 14 | A regional checklist helps the average observation (+5 pt accuracy) and hurts per-species macro-F1 when the scored set is most of the checklist. *(Not in the paper.)* | — | K1 |

### 1b. Engineering (argued in [`docs/design-decisions.md`](docs/design-decisions.md))

| # | finding |
|---|---|
| 15 | The baseline climbed 0.8297 → 0.9152 through the schedule, the sampler and lighter augmentation, not the optimiser. |
| 16 | Margin heads need bf16 (fp16 overflows); fp16 is fine elsewhere. |
| 17 | Audit the evaluation set before believing a metric: filtering rare species out of the test fold inflated a macro average by 3 pt. |
| 18 | Framework attributes can lie: fastai's reported `num_workers` caused a ~900x slowdown first blamed on hardware. |
| 19 | Deployment: int8 cannot run in ORT-Web, fp16 ships; the classifier head is half a small model, so its width is the size knob. |
| 20 | 9.3 % of TreeOfLife "species" keys are not species; audit a dataset's key composition before quoting a class count. |
| 21 | Derive stored image size from the training transform: the input crops the short side, so store the short side. |
| 22 | UCloud jobs see the host's 256 cores; size thread pools from the cgroup quota, and cost CPU jobs before submitting them. |

## 2. Every experiment

**Columns.** *in-dist* = species macro-F1 on the held-out fold of our own data. *probe* = macro-F1 on
held-out trap nights (the honest domain-shift benchmark). *probe-HO* = probe restricted to species
the adaptation never saw. *full trap* = all 47,905 trap images (before `probe` existed; not
comparable with it). *AUROC* = open-set detection of unseen species, with each
model's best scoring rule. Differences only mean something **within** a column. Noise floors:
in-dist ~0.000, probe ~0.004 end-to-end but ~0.012 for frozen-trunk stages, so a probe difference
under ~1.5 pt between staged runs is not a result.

> **Letters were reused before this registry existed** (app phases A-D, design decisions D1-D3,
> hypotheses H1-H4, W3's arms A-E); those are not the experiments below. The 2026-08-28 directions
> D1-D3 became **K1** and **W1-W3** (old names survive in paths such as `ucloud_preds/D1-probe`).
> New experiments take an unused letter.

### The ten rows that carry the paper

If you read only these, you have the paper's evidence: **singlehead** and **marginal** (heads do not
help, coarse supervision does), **A1** (the margin), **C3b** (novelty is graded by taxonomic
distance), **E2** (scoring rules do not transfer across scale), **L4** (cRT), **B3rep5x** and **T2**
(self-training; adaptation is a classifier problem), **B8** and **P5** (the two shipped models).

---

### Phase 0 — building the baseline (July, local GPU, cited by run id)

| run | question | result | where |
|---|---|---|---|
| `20260712-072542` | the starting point: multi-head cosine, Muon | in-dist **0.8297** | [ladder](journal/2026-07-16-why-was-fastai-behind-mini-trainer.md) |
| `20260714-072404` | + one_cycle schedule | 0.8887 | same |
| `20260716-105029` | 10 epochs instead of 5 | 0.8976 | same |
| `20260716-154156` | + square-root oversampling of rare species | **0.9148** — the recipe | [long tail](journal/2026-07-17-does-longtail-help.md) |
| `20260716-234247` | logit adjustment instead of oversampling | 0.9031 (broke genus/family) | same |
| port | the clean `src/lepinet` package reproduces it | 0.9152 | [port](journal/engineering/2026-07-24-src-lepinet-baseline-port.md) |
| bottleneck 128/256/512 | how much does the head width cost | 0.8843 / 0.9002 / 0.9058 | [compression](journal/engineering/2026-07-20-lepi-app-compression.md) |
| autoregressive, conditional heads | do hierarchy-aware heads help | 0.69-0.73 and 0.8845: **no** | [heads](journal/2026-07-16-why-was-fastai-behind-mini-trainer.md) |
| ConvNeXtV2-L @320 | does a bigger backbone help | **0.9316**, best in-dist ever | [bigger](journal/2026-07-24-bigger-everything.md) |
| flemming eval | does 0.9316 survive trap images | **0.6950** — the 23-pt gap that started the pivot | [flemming](journal/2026-07-28-flemming-generalization.md) |
| **singlehead** | one species head + marginalisation vs multi-head | 0.9135, beats multi-head at every level; **this is the cheap reference** | [marginal](journal/2026-07-30-marginal-supervision.md) |
| **marginal** | + loss on the marginalised genus/family | species unchanged, genus +0.27 / family +0.39; recovers 1.41 of the 2.10 pt lost under shift | same |
| distill T=4 / T=1 | does distillation help a small student | 0.8546 / **0.8786** vs 0.8692 from scratch; T must be ~1 | [bridge](journal/engineering/2026-07-25-teacher-student-app-bridge.md) |

### A — consolidate the architecture (closed)

| ID | question | change | result | where |
|---|---|---|---|---|
| **A1** | do the single head and ArcFace × z-score compose? | singlehead + angular margin, 20 M | in-dist 0.9035, full trap 0.6437, **AUROC 0.9068** (best open-set at 20 M) | [compose](journal/2026-07-30-does-arcface-compose-with-marginalisation.md) |
| A2 | the same at 198 M | DINOv3-ConvNeXt-L backbone | in-dist **0.9216**, full trap 0.6616, AUROC 0.8298 (max-logit; see E2) | same |
| A3 | distil the 198 M model into effnet-b0 | T=1 | 0.8833 | [bridge](journal/engineering/2026-07-25-teacher-student-app-bridge.md) |
| A4 | does marginal supervision help A1 under shift? | A1 + marginal loss | 0.8998, full trap 0.6616 (**+1.79** over A1) | [compose](journal/2026-07-30-does-arcface-compose-with-marginalisation.md) |
| A5 | how noisy are our numbers? | exact repeat | species spread **0.0000**, family 0.0024 | [noise](journal/2026-08-01-how-noisy-are-our-numbers.md) |
| A6 | control for A3: b0 single head from scratch | no teacher | 0.8789 — distillation's credit halves to +0.44 | [bridge](journal/engineering/2026-07-25-teacher-student-app-bridge.md) |

### B — robustness: augmentation, self-training, capacity (closed)

| ID | question | change | result | where |
|---|---|---|---|---|
| **B1** | does trap-like augmentation close the gap? | A1 + blur / low light / JPEG | full trap 0.6836 (+4.0 for −0.36 in-dist); probe 0.6912; closes 17 % of the gap | [domain shift](journal/2026-07-30-domain-shift.md) |
| B2 | background suppression | — | never run; retired after T2b | [F2](journal/2026-08-06-f2-capstone.md) |
| **B3** | self-training on unlabelled trap images | B1 + pseudo-labelled trap images at 6.1 % of training | probe **0.7370** (+4.58), in-dist unchanged | [B3](journal/2026-08-03-b3-self-training.md) |
| B3rep1x / **5x** / 26x | how much trap data? | actually 1x / **11x** / 58x: share of training rows 0.24 % / **2.6 %** / 12.3 % (formerly labelled 0.39 / 2 / 10 %) | probe 0.7354 / **0.7706** / 0.7159 — sharp optimum at 2.6 %; shipped as B3rep5x | [sweep](journal/2026-08-04-replication-sweep.md) |
| B4 | capacity x augmentation | A2 + augmentation, 198 M | in-dist 0.9216, full trap **0.7101**, AUROC 0.8132 | [2x2](journal/2026-08-01-capacity-x-augmentation.md) |
| B6 | self-training at 198 M | F1 + self-training at 6.2 % (27x) | 0.9225 / probe 0.7699 / probe-HO 0.7422 | [sweep](journal/2026-08-04-replication-sweep.md) |
| B7 | drop oversampling too | B6 − oversampling | 0.9050 / 0.7796 / 0.7712 | [imbalance](journal/2026-08-01-imbalance-methods-bench.md) |
| **B8** | B7 at the 2.6 % optimum | share 2.6 % (11x) | 0.9060 / **0.7798** / **0.7816** — best of our own models; **public on Hugging Face** | [sweep](journal/2026-08-04-replication-sweep.md) |
| B9 | balanced pseudo-labels, end-to-end, 20 M | balanced replication | probe 0.7635 — prediction falsified | [iteration](journal/2026-08-08-self-training-does-not-iterate.md) |
| B10 | the same at 198 M | balanced replication | probe 0.7800 — ties B8 | [balance](journal/2026-08-10-balance-is-oversampling-and-it-does-not-scale.md) |

### C — open-set and abstention (closed)

| ID | question | change | result | where |
|---|---|---|---|---|
| C1 | when should the model back off to genus? | rank abstention thresholds | needs **conditional** thresholds: genus precision 0.487 on hard images vs 0.970 overall | paper §4.6 |
| C2 | does open-set survive the move to a single head? | A1 vs multi-head ArcFace | 0.9068 vs 0.9115: yes | [compose](journal/2026-07-30-does-arcface-compose-with-marginalisation.md) |
| C3 | is novelty graded by taxonomic distance? | AUROC split near / mid / far | 0.8527 / 0.9342 / 0.9641 | [reframe](journal/2026-07-29-the-reframe-directions.md) |
| C3r | C3 with each head's own best rule | re-scoring | part of the retraction of the "31-point" claim | [retraction](journal/2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison.md) |
| **C3b** | or was C3 measuring rarity? | 231 **common** taxa withheld, retrained | **0.8717 / 0.9463 / 0.9726** — monotone, not rarity | [C3b](journal/2026-08-08-is-novelty-monotone-or-just-rare.md) |
| C3ref | matched control for C3b | C3's model on C3b's split | holding data out costs 0.04 pt, not 0.38 | same |

### D — product (closed)

| ID | question | result | where |
|---|---|---|---|
| D (bundle) | one command from checkpoint to app-ready ONNX | `lepinet bundle`; names, calibration, thresholds | [bridge](journal/engineering/2026-07-25-teacher-student-app-bridge.md) |
| D2 (student) | the shippable small model | fastvit_sa12 distilled from A2: **0.8967** | same |
| release | public models | P5, B8, B3rep5x on Hugging Face, 2026-09-29 | [release](journal/engineering/2026-09-29-public-hf-release.md), [P5 fix](journal/engineering/2026-09-30-full-fold-and-p5-resize.md) |

### E — is open-set the binding constraint? (closed: no)

| ID | question | result | where |
|---|---|---|---|
| E1 | re-tune the ArcFace margin at 198 M (~36 GPU-h) | **cancelled** by E2 | [rule](journal/2026-08-01-the-scoring-rule-was-the-bug.md) |
| **E2** | is the 198 M open-set loss in the model or in the scoring rule? | **the rule**: max-softmax beats max-logit by +6.1/+7.6 at 198 M; the capacity penalty falls 8.8 → 1.64 | same |

### F — the assembled recipe (closed)

| ID | question | change | result | where |
|---|---|---|---|---|
| F1 | compose every win at 198 M | B4 + marginal supervision | 0.9219 / full trap 0.7103 / probe 0.7209 / AUROC 0.8800 — identical to B4 on species and shift; coarse levels gain | [F1](journal/2026-08-02-f1-flagship.md) |
| **F2** | the staged recipe at 20 M (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 1 clean representation + cRT + adaptation, frozen trunk | in-dist **0.9081**, probe 0.7541 | [F2](journal/2026-08-06-f2-capstone.md) |
| F3 | joint vs sequential classifier stages (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | joint | 0.9061 / 0.7479 — a near-wash | same |

### G — the 198 M confirmation (closed)

| ID | question | result | where |
|---|---|---|---|
| G1 | cRT at 198 M (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | in-dist 0.9112 — confirms at half strength | [F2](journal/2026-08-06-f2-capstone.md) |
| G2 | the staged recipe at 198 M (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.9150 / probe 0.7648 / probe-HO 0.7600 | same |
| G3 | staged + balanced pseudo-labels (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.9138 / 0.7740 / 0.7518 | [balance](journal/2026-08-10-balance-is-oversampling-and-it-does-not-scale.md) |
| G3b | exact repeat of G3 (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7870 / 0.7892: **a 3.74 pt spread between identical runs** — retracted two claims; staged vs end-to-end is a tie at 198 M | [noise floor](journal/2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes.md) |

### H — scaling the head to ~1 M species (closed; answered by a data policy instead)

| ID | question | result | where |
|---|---|---|---|
| H1 | replace the trained matrix by class centroids at inference | costs **0.29 pt** — works | [scaling](journal/2026-08-05-scaling-the-head.md) |
| low-rank | factorise the matrix | rank 1035/1280 — dead | same |
| H2 (s256/1024/4096) | uniform sampled softmax | no plateau; 1024 negatives lose 3.25 pt — dead | same |
| H3 | taxonomy-aware hard negatives | recovers 26 % of H2's loss — weak | same |
| H4 | train against EMA centroids, no matrix | 0.8685 (−4.63) — falsified | [H4](journal/2026-08-09-can-centroids-be-trained-against.md) |

### L — long-tail learning (closed)

| ID | question | change | result (in-dist / shifted) | where |
|---|---|---|---|---|
| L0 | no oversampling on the single head | control | 0.8949 / 0.6445 | [imbalance](journal/2026-08-01-imbalance-methods-bench.md) |
| (singlehead) | square-root oversampling | — | 0.9135 / 0.6293 | same |
| L1 | balanced softmax | τ=1 | 0.8970 / 0.5726 | same |
| L2 | both | | 0.8689 / 0.5492 — robustness falls monotonically with tail weighting | same |
| **L4** | rebalance only the classifier (cRT) (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | L0's trunk, classifier oversampled | **0.9068 / 0.6539** — the trade disappears | same |
| L5 | does oversampling's cost grow at 198 M? | no oversampling, 198 M | 0.9055 / probe 0.7497 / probe-HO 0.7641 — costs 2.88 probe | same |
| L6 | cRT at 198 M (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | | launched 2026-08-06, **result never journalled**; G1 answers the same question | same |
| L7 (cap 250/500/1000) | where is the optimal cap per species? | cap training images per species | cap 1,000: **+1.57 probe, +3.26 probe-HO, −0.88 in-dist** vs ~2,000 | [L7](journal/2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts.md) |
| L7c1000b | repeat of cap 1,000 | | survives n = 2: +1.76 / +2.94 | same |
| M2 | repeat of the marginal-supervision shifted number | | the multi-head's apparent lead was noise | [marginal](journal/2026-07-30-marginal-supervision.md) |

### R — the self-training gate (closed)

| ID | change | probe / probe-HO | where |
|---|---|---|---|
| R2 | relabel with F2, keep top 30 % by confidence (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7161 / 0.7257 — falsified by 4.2 pt | [iteration](journal/2026-08-08-self-training-does-not-iterate.md) |
| R3 | per-species cap instead (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7585 / 0.7682 | same |
| R4 | no gate at all (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7674 / 0.7458 | same |
| R5 | no gate, balanced (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7692 / 0.7781 | same |
| R5b | repeat of R5 (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7573: frozen-stage probe spread is **0.0119**, 3x the old floor | [noise floor](journal/2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes.md) |

### T — what labels would have bought, and where adaptation happens

| ID | question | result | where |
|---|---|---|---|
| T1 (lab500 / 2500 / 12230) | real trap labels instead of pseudo-labels, each image once | 12,230 real labels: probe 0.7568, **+2.14 over 12,230 pseudo-labels at the same 1x dose** (0.7354). Not comparable with self-training at 11x (0.7706): different dose | [labels](journal/2026-08-05-label-budget.md) |
| T1rep | the 12,230 real labels at 11x, the dose of B3rep5x | probe **0.7807** (predicted 0.78-0.79), +1.01 over self-training; held-out species **0.7007** (self-training 0.7704): 91 % of its held-out-species errors land in the labelled fauna (self-training 42 %). Preds `T1rep-probe`, `T1rep-probeho` | same |
| **T2** | adapt only the classifier, trunk frozen, 2 epochs (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | probe 0.7572 / probe-HO 0.7621: **83 %** of self-training's gain | [T2](journal/2026-08-06-adaptation-is-mostly-a-classifier-problem.md) |
| T2b | the same from a trunk with no trap augmentation (**trunk was not frozen**, [2026-11-05](journal/engineering/2026-11-05-the-frozen-stages-were-not-frozen.md)) | 0.7515 — the features were already good | same |

### X — does self-training transfer to a trap or site it never saw?

| ID | question | result | where |
|---|---|---|---|
| X1 | B3rep5x minus one trap's pseudo-labels, 12 folds; each probe crop scored by the fold that never saw its trap | pooled **0.7430** (B1 0.6923, B3rep5x 0.7719, same crops): **64 %** of the gain; predicted 0.755-0.770, falsified. Accuracy keeps all (0.836 vs 0.835); held-out species 78 %. Lost: species pseudo-labelled only there (24 % kept), rare never-gated species (0 %). Preds `holdout-<trap>-probe` | [hold-out](journal/2026-10-12-self-training-on-a-trap-it-never-saw.md) |
| X2 | the same, one site's four traps removed, 3 folds | pooled **0.7105** (23 % of the gain; predicted 0.730-0.760, falsified); held-out species 0.7363 (53 %). Per site macro-F1 keeps 30 / 48 / 49 %, accuracy 97 / 90 / 79 % (LV / OH / SS) | same |

### U — self-training on European traps (opened 2026-10-21)

| ID | question | result | where |
|---|---|---|---|
| U1-U4 | pseudo-labels from EU traps (teacher B8, unique images, no Danish probe traps): natural vs balanced at 2.6 %, then a share sweep, then plus the Danish set | not run | [EU traps](journal/2026-10-21-self-training-on-european-traps.md) |

### N — knowing when a trap image is too poor, or a species is new (opened 2026-10-23)

| ID | question | result | where |
|---|---|---|---|
| N1 | the release back-off cascade by image sharpness and crop size (B8, P5, B3rep5x, probe) | species answers fall from 83-91 % on the sharpest fifth to 51-65 % on the blurriest; species precision there 0.931-0.935 (target 0.95); B8 and B3rep5x never answer at genus | [quality](journal/2026-10-23-does-the-model-know-when-a-trap-image-is-too-poor.md) |
| N2 | open-set on trap images: C3b (9 withheld trap species, 747 crops) plus the 234 out-of-vocabulary crops; five rules | near-novel AUROC **0.566** (0.8717 on museum images), out-of-vocabulary 0.781, all 0.619 (MSP); 38 % of near-novel crops get a confident wrong species name, family answers 0.988 correct | same |
| N3 | open-set on trap images for B8 and P5 (234 out-of-vocabulary crops) | B8 **0.857** (top-2 margin; 0.9153 on museum images); P5 **0.835** (max-logit; 0.9161 on museum images) | same |

### Q — the back-off policy (opened 2026-10-23)

| ID | question | result | where |
|---|---|---|---|
| Q1 | greedy per-rank 95 % thresholds (current) against an expected-utility rule with one error cost, calibrated in-distribution or on target data; Lepidoptera test fold, trap probe, Pl@ntNet, novel trap crops | at 95 % precision, expected utility wins in-distribution (+1.1 to +1.9 correct) and for the plain head on traps (C3b +5.2), loses on traps for the margin heads (B3rep5x -5.7, B8 -0.9 to -2.5): their coarse confidences equal the species one; policies fitted in-distribution miss 95 % on traps by 6-11 pt; Pl@ntNet pending | [back-off](journal/2026-10-23-what-is-the-best-back-off-policy.md) |

### P — foundation model (BioCLIP-2) as the trunk (closed)

| ID | question | result | where |
|---|---|---|---|
| P1a | frozen BioCLIP-2 + a fitted classifier | clean fold 0.8444 vs our 0.9021 | [P1](journal/2026-08-24-does-the-recipe-need-our-backbone.md) |
| P1b | + our adaptation stage | probe 0.5901 vs 0.7515 — falsified by 11 pt | same |
| P2 | centroids instead of a classifier | not run | |
| P3a / b / **c** | fine-tune it, lr 1e-3 / 1e-4 / **1e-5** | clean fold 0.8912 / 0.9025 / **0.9146** (+1.25 over ours); P3c probe 0.6630, probe-HO 0.6937 | [P3](journal/2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts.md) |
| P4 / P4b | P3c + frozen adaptation | probe 0.7199 / 0.7218 — below our recipe | same |
| **P5** | P3c + **unfrozen** adaptation | in-dist 0.9113, probe **0.7810**, probe-HO 0.7806 — ties B8; **public on Hugging Face** | [P5](journal/2026-08-28-frozen-adaptation-only-works-on-your-own-trunk.md) |
| P5b | repeat of P5 | 0.7703 / 0.7827; P5 n = 2: probe 0.7757 ± 0.0054 | same |

### O — deployment behaviour of the shipped models

| ID | question | result | where |
|---|---|---|---|
| **O1** | B8 vs P5 under a 95 %-precision back-off policy | reported 17.3 pt apart on "useful answers"; **superseded by O3** (rank-blind metric, fitted in-sample, B8 saturated). Best open-set rule is entropy for both | [O1](journal/2026-08-28-two-tied-models-differ-by-17-points-in-deployment.md) |
| O2 | open-set as the number of enrolled taxa grows to 204 K | not run | [PLAN](PLAN.md) |
| **O3** | is O1's gap a model property or a readout artefact? | neither: the metric was rank-blind. Held-out nights, published models, B8 / P5 / B3rep5x: correct at species **74.7 / 67.4 / 72.9 %**, at any rank **77.2 / 86.4 / 85.8 %**; raw B8 (69.7 % of softmax values exactly 1.0) answers 0 % at species | [O3](journal/2026-10-02-is-the-deployment-gap-a-readout-artefact.md) |

### K and W — directions opened 2026-08-28 (first named D1-D3)

| ID | question | status / result | where |
|---|---|---|---|
| K1 | restrict the label space to the regional (Danish) checklist | done: probe micro-accuracy **+5.06**, macro-F1 **−1.70**; held-out species macro-F1 +6.31 | [D1](journal/2026-08-28-a-regional-checklist-helps-the-user-and-hurts-the-tail.md) |
| W1 | download TreeOfLife-200M at our data policy | **running** (re-crawl at short side 256, on the workstation) | [crawl](journal/engineering/2026-10-01-the-crawl-resized-the-wrong-side.md) |
| W2 | train our objective on ToL-10M vs BioCLIP-1 | not started | [D2](journal/2026-08-28-two-directions-checklists-and-our-objective-on-tol.md) |
| W3 | does training on the whole tree of life help open-set and shift? | designed; waits for W1 | [D3](journal/2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know.md) |

### S — several organisms in one picture (opened 2026-10-09)

| ID | question | status / result | where |
|---|---|---|---|
| S1a | do our open-set scores flag a two-organism image as novel? Synthetic composites, published B8 and P5; per-location logits | designed | [S1](journal/2026-10-09-two-organisms-in-one-picture.md) |
| S1b | the same on the W3 model, with moth-on-plant, FathomNet frames and hand-checked photos; fixes W3's readout | after W3 | same |

### V — one code base: the recipe moves into mini_trainer (opened 2026-11-03)

| ID | question | status / result | where |
|---|---|---|---|
| V0F | recipe L's flat head without the 50-image floor: the anchor on mini_trainer's class set and test set (12,632 species, 632,913 test images); same vocabulary as Nemo | test fold at threshold 0 **0.9110 / 0.9600 / 0.9760** (floor F 0.9135 / 0.9606 / 0.9739): the 12,041 shared species 0.9126, the 591 added 0.877 (median 5 test images each). Species macro-F1 calibrated by (trap, night), trap / probe / probe-HO: **0.279 / 0.300 / 0.235** (floor F 0.284 / 0.287 / 0.239). The floor costs nothing under shift. Model `20261103-152940-lepinet-V0F-nofloor-singlehead-effnetv2s-ucloud` | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| V0BU | the same with marginal supervision (BU, mini_trainer's HierarchicalClassifier and Nemo's head) | test fold **0.9107 / 0.9625 / 0.9782** (floor BU 0.9135 / 0.9633 / 0.9778; added species 0.873). Trap / probe / probe-HO **0.330 / 0.351 / 0.288** (floor BU 0.313 on trap; Nemo 0.305 / 0.338 / 0.293): with Nemo's head and vocabulary, 5 epochs at 256 px match or beat its 30 at 384 px on trap crops, and BU is 5 pt above V0F on every trap set (single runs). Model `20261103-152945-lepinet-V0BU-nofloor-marginal-effnetv2s-ucloud` | same |
| V1 | parity on a subset, local RTX 5090: mini_trainer's lepinet preset against lepinet, I head, 5 epochs, two seeds per code run alternately (lepinet with `GCCallback`); 512 species picked by MT's family × abundance selector after the 50-image floor, 218,756 training images | **passed**. Set 0 (27,349 images, identical in all four prediction files), macro-F1 at threshold 0, MT minus lepinet: species -0.0001, genus +0.0007, family +0.0004 (calibrated +0.0016, -0.0012, +0.0004). Validation: 34 of 35 per-epoch cells within max(2 × seed range, 0.005). The exception, family macro-F1 after epoch 2 (+0.0111 against 0.0089), is noise: equal on set 0, equal family accuracy, 40 families. Learning-rate curves agree to four digits. Speed not comparable (shared GPU) | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md); MT note `.agents/notes/2026-11-03-lepinet-integration.md` (local commit 7009f39) |
| V2I, V2BU, V2F | the parity gate at full scale: mini_trainer's presets for heads I, BU and F in lepinet's setting (50-image floor, 12,041 species), one B200 each, images staged to node-local disk. Pass: test macro-F1 within 0.003 at every rank at threshold 0 (I 0.9110 / 0.9587 / 0.9708, BU 0.9135 / 0.9633 / 0.9778, F 0.9135 / 0.9606 / 0.9739), and the per-epoch curves of the clean-package runs | **passed** (2026-11-04; UCloud 12445754-56). Test fold, 629,742 images matched by file, threshold 0, species / genus / family: I 0.9107 / 0.9588 / 0.9726, BU 0.9131 / 0.9629 / 0.9803, F 0.9144 / 0.9608 / 0.9759; at most 0.0025 from lepinet, the largest at family, where lepinet's own repeat of F moved +0.0023. Validation curves differ by up to 0.005 mid-training and close by epoch 5 to within lepinet's repeat spread. Runs and predictions: UCloud `/12347837/repos/mini_trainer-lepinet/tmp-runs/` (`gate_{I,BU,F}/`, `eval/`) | same |
| V2T | speed: both codes in one B200 job on the same staged data, lepinet exactly as on UCloud (no `GCCallback`), peak GPU memory for both | **mini_trainer's training loop is 1.185x lepinet's**: steady img/s at batch 64, lepinet 1,201 / 1,205, mini_trainer 1,422 / 1,428 (10 % sample, one epoch per run, order L, M, M, L, no drift; job 12445759, node shared with the three gates). Peak GPU memory 10.4-10.6 GiB against 6.6; validation 29 s against 18. A whole epoch still took mini_trainer 508-552 s against 386-390 s: about 3 min outside the loops, from worker start-up (once per run, under forkserver) and the per-epoch figures (now off) | same |
| V3 | GPU options in mini_trainer, at the owner's request: cudnn.benchmark, channels_last, foreach or fused AdamW, the four torch.compile modes, the compiled optimizer with and without CUDA graphs; one B200, F preset, one short epoch per run (3,448 species, 40 training images each), base runs between phases | **done** (UCloud 12446034): channels_last, torch.compile reduce-overhead and the compiled optimizer with CUDA graphs reach 2,242 img/s against 1,335 (1.68x, 3 and 4 runs), same accuracy after the epoch (64.48 %), peak memory 6.6 to 4.3 GiB; channels_last alone ~1.3x. F preset only. The campaign uses channels_last (owner); compile waits for the owner | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VN | Nemo (mini_trainer's EfficientNetV2-S release: 12,632 species, 30 epochs at 384 px, BU head) re-scored on our reconciled trap export, our protocol (its card used the unreconciled export) | species macro-F1, calibrated by (trap, night), trap / probe / probe-HO: full vocabulary **0.305 / 0.338 / 0.293** (B1 0.409 / 0.424 / 0.356; our plain 5-epoch F 0.284, BU 0.313 on trap); with its TTA 0.449 / 0.479 / 0.359; with the north_europe list 0.525 / 0.546 / 0.449; both, the card's setting, 0.599 / 0.646 / 0.490, which ties B8 on probe (0.647) without trap images. TTA and list were chosen on these crops | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VB0 | the head grid on B0 (torchvision EfficientNet-B0, Svenning et al.'s small backbone), the pinned recipe (no floor, no oversampling), 5 epochs at 256 px, seed 42 (campaign job c-b0, UCloud 12447137) | test fold (632,913 images) at threshold 0, species / genus / family: F **0.8751** / 0.9405 / 0.9618, BU 0.8641 / 0.9408 / 0.9654, I 0.8533 / 0.9232 / 0.9409, TD 0.7878 / 0.8862 / 0.9326 (lepinet's A6, EfficientNetV2-B0 with √-oversampling and the floor: 0.8789 / 0.9386 / 0.9626; Svenning et al.'s B0, 10 epochs: 83.5 / 83.3 / 82.4 / 79.2 for F / BU / I / TD). Species calibrated by (trap, night), trap / probe / probe-HO: F 0.281 / 0.298 / 0.236, BU **0.329 / 0.357 / 0.307**, I 0.329 / 0.356 / 0.267, TD 0.270 / 0.291 / 0.212: BU at 5 M matches V0BU at 21 M with oversampling | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VP | the head grid on Pl@ntNet-300K, seed 42: F, BU, I, TD on V2-S (384 px), B0 (224 px) and ViT-L/16 (224 px, lr 1e-4); 10 epochs, the pinned recipe, five ranks (jobs pp-v2s, pp-b0-r2, pp-vitl) | test fold (31,088 images) at threshold 0, species for F / BU / I / TD: V2-S **0.467** / 0.462 / 0.451 / 0.406, B0 0.420 / 0.400 / 0.398 / 0.355, ViT-L 0.409 / 0.413 / 0.395 / 0.378, DINOv3 ConvNeXt-L (320 px, B8's lr 8e-4) 0.439 / 0.439 / 0.438 / 0.398: the 198 M backbone is below V2-S here. Svenning et al. (class-weighted loss): V2-S 48.2 / 47.3 / 47.8 / 37.5, ViT-L flat 38.2. The ImageNet-1k ViT-L is below V2-S here too. Genus to class in the journal | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VS | the head grid on V2-S with the pinned recipe (no floor, no oversampling), 5 epochs at 256 px, seed 42, and EMLA against cross-entropy on BU, the phase-3 comparison (campaign job c-v2s, UCloud 12447133) | test fold (632,913 images) at threshold 0, species / genus / family: F 0.8890 / 0.9488 / 0.9693, BU **0.8877 / 0.9556 / 0.9759**, I 0.8812 / 0.9471 / 0.9655, TD 0.8413 / 0.9293 / 0.9618; BU with EMLA 0.8936 / 0.9515 / 0.9683. Species calibrated by (trap, night), trap / probe / probe-HO: F 0.348 / 0.366 / 0.298, BU 0.339 / 0.371 / 0.306, I **0.381 / 0.405 / 0.307**, TD 0.315 / 0.347 / 0.271, EMLA 0.300 / 0.316 / 0.257. EMLA buys 0.6 in-distribution at species and loses 4-5.5 under shift: cross-entropy stays. Dropping oversampling costs 2.2-2.3 in-distribution (V0) and gains F 6-7 under shift, BU 1-2 | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VC | the channels_last control: the gate's BU spec (50-image floor, √-oversampling) with channels_last as the only change | **passed**: lepinet's 629,742 rows, threshold 0, 0.9135 / 0.9631 / 0.9790 against V2BU's 0.9131 / 0.9629 / 0.9803 (within 0.0013 at every rank). Traps 0.317 / 0.343 / 0.268 | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VT | the real-label arms on V2-S BU, uncompiled (job c-t1rep): T1rep's rerun (dev/067's round-robin labels, 12,205 crops at 11x) and the matched arm (the 12,224 gated crops with their true labels at 11x) | probe / probe-HO, species calibrated by (trap, night): T1rep **0.722 / 0.305** (lepinet's T1rep 0.723 / 0.312, same convention: **reproduced**); gated crops with true labels 0.682 / **0.343**; no adaptation (VS BU) 0.371 / 0.306. Test fold 0.8868 / 0.8873 at species. True labels on the gated crops keep a held-out-species gain that the round-robin draw does not | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
| VO | open-set on C3b's withheld taxa (dev/072's 231 common taxa), four heads on V2-S trained on c3b_train with the pinned recipe, scored on the unmodified set 0 with `dev/099` (job c-openset) | AUROC of known against near / mid / far (10,391 / 2,969 / 2,376 images), species rank, entropy rule: F **0.887 / 0.963 / 0.973**, BU 0.880 / 0.952 / 0.970, I 0.881 / 0.958 / 0.969, TD 0.843 / 0.904 / 0.922 (lepinet's C3b, entropy: 0.8717 / 0.9463 / 0.9726: **reproduced**, monotone in taxonomic distance for every head and rule but max and energy). Entropy is the best rule for all four; max and energy score 0.5-0.72 on these heads. At family rank, F and BU separate unseen families at 0.985-0.989 | [integration](journal/engineering/2026-11-03-moving-the-recipe-into-mini-trainer.md) |
