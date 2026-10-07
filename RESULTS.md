# Results

What the project has established, every experiment behind it, and the raw table of the July local
runs. Section 1 is the short version; section 2 resolves any experiment ID (A1, B8, P5...) and is
where new results go; section 3 is a frozen table of the July local runs. The reasoning behind
every number is in the linked [`journal/`](journal/README.md) entries.

## 1. What is established

One line each, with the paper section and the experiments (section 2) that carry it. Negative results count.

### 1a. Science

| # | finding | paper | IDs |
|---|---|---|---|
| 1 | Hierarchy-aware heads do not help; one species head plus marginalisation matches or beats them. Coarse *supervision* (a loss on the marginals) still buys ~1.4 pt under shift. | §4.1 | singlehead, marginal, A4 |
| 2 | In-distribution accuracy saturates near 0.93 but falls ~23 pt on trap images, and the three axes rank models differently. | §4.2, §4.10 | flemming, B4 |
| 3 | **The spine:** interventions belong in the classifier, not the representation: rebalancing (cRT), domain adaptation (83 % from the classifier alone), the prototype matrix (centroids for 0.29 pt, for the margin head; 3.7 pt for the plain head). | §4.15 | L4, T2, H1 |
| 4 | Unlabelled trap images are the largest lever: self-training gives +7.94 probe at a 2 % share and beats 12,230 real labels; above 2 % adaptation becomes memorisation. | §4.11 | B3rep5x, T1 |
| 5 | Long-tail reweighting trades shift robustness for accuracy, monotonically; cRT removes the trade; capping training at 1,000 images per species helps shift. | §4.13 | L0-L4, L7 |
| 6 | The best open-set scoring rule changes with model scale (6-7.6 pt); one rule for all models produced a false ranking. | §4.9 | E2 |
| 7 | The angular margin relocates open-set signal rather than creating it: +0.78 AUROC best-vs-best (an earlier "31 pt" compared rules, and is retracted). | §4.3 | A1, C3r |
| 8 | Novelty detection improves with taxonomic distance (near < mid < far), and not because unseen taxa are rare. | §4.4 | C3, C3b |
| 9 | Abstention under shift is expensive (13-19 % vs 0.8 % in-distribution), and a back-off policy yields a composition, not a score: tied B8 and P5 trade species answers (74.7 vs 66.0 %) for answers at any rank (77.2 vs 83.3 %). A margin head needs a temperature first: in float32 its softmax saturates and the species rank becomes unusable. | §4.6a | O1, O3 |
| 10 | Two thirds of our test fold is inside BioCLIP-2's training data by GBIF occurrence id; a frozen probe understates that model by 7 pt; fine-tuned it is the better trunk, and our recipe closes the gap. | §4.14 | P1-P5 |
| 11 | A margin damages marginalisation more than classification, through calibration; replicated at 10x scale. | §4.7 | A1, A2, A4 |
| 12 | Augmentation that mimics trap conditions closes only 17 % of the gap. | §4.8 | B1 |
| 13 | Measurement: macro-F1 does not decompose over subsets; noise floors depend on the training regime (identical runs 3.74 pt apart); give every arm its own best configuration. | §3.2, §5 | A5, G3b, R5b |
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
the adaptation never saw. *full trap* = all 47,905 trap images (used before `probe` existed, on
2026-08-02; not comparable with probe). *AUROC* = open-set detection of unseen species, with each
model's best scoring rule. Differences only mean something **within** a column. Noise floors:
in-dist ~0.000, probe ~0.004 end-to-end but ~0.012 for frozen-trunk stages, so a probe difference
under ~1.5 pt between staged runs is not a result.

> **Letters were reused before this registry existed.** The July app roadmap used A1-A4, B1-B3,
> C1-C4 and D1-D8 for app phases; the package-port entry used D1-D3 for design decisions; the
> domain-shift entry used H1-H4 for hypotheses; W3's design uses "arms A-E" internally. None of
> those are the experiments below. The 2026-08-28 directions were first called D1-D3, clashing with
> group D, and were renamed **K1** and **W1-W3** on 2026-10-02 (old names survive in UCloud output
> paths such as `ucloud_preds/D1-probe`). New experiments take an unused letter and get a row here
> when they are opened.

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
| B3rep1x / **5x** / 26x | how much trap data? | share 0.39 % / **2 %** / 10 % | probe 0.7354 / **0.7706** / 0.7159 — sharp optimum at 2 %; shipped as B3rep5x | [sweep](journal/2026-08-04-replication-sweep.md) |
| B4 | capacity x augmentation | A2 + augmentation, 198 M | in-dist 0.9216, full trap **0.7101**, AUROC 0.8132 | [2x2](journal/2026-08-01-capacity-x-augmentation.md) |
| B6 | self-training at 198 M | F1 + self-training at 6 % | 0.9225 / probe 0.7699 / probe-HO 0.7422 | [sweep](journal/2026-08-04-replication-sweep.md) |
| B7 | drop oversampling too | B6 − oversampling | 0.9050 / 0.7796 / 0.7712 | [imbalance](journal/2026-08-01-imbalance-methods-bench.md) |
| **B8** | B7 at the 2 % optimum | share 2 % | 0.9060 / **0.7798** / **0.7816** — best of our own models; **public on Hugging Face** | [sweep](journal/2026-08-04-replication-sweep.md) |
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
| **F2** | the staged recipe at 20 M | 1 clean representation + cRT + adaptation, frozen trunk | in-dist **0.9081**, probe 0.7541 | [F2](journal/2026-08-06-f2-capstone.md) |
| F3 | joint vs sequential classifier stages | joint | 0.9061 / 0.7479 — a near-wash | same |

### G — the 198 M confirmation (closed)

| ID | question | result | where |
|---|---|---|---|
| G1 | cRT at 198 M | in-dist 0.9112 — confirms at half strength | [F2](journal/2026-08-06-f2-capstone.md) |
| G2 | the staged recipe at 198 M | 0.9150 / probe 0.7648 / probe-HO 0.7600 | same |
| G3 | staged + balanced pseudo-labels | 0.9138 / 0.7740 / 0.7518 | [balance](journal/2026-08-10-balance-is-oversampling-and-it-does-not-scale.md) |
| G3b | exact repeat of G3 | 0.7870 / 0.7892: **a 3.74 pt spread between identical runs** — retracted two claims; staged vs end-to-end is a tie at 198 M | [noise floor](journal/2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes.md) |

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
| **L4** | rebalance only the classifier (cRT) | L0's trunk, classifier oversampled | **0.9068 / 0.6539** — the trade disappears | same |
| L5 | does oversampling's cost grow at 198 M? | no oversampling, 198 M | 0.9055 / probe 0.7497 / probe-HO 0.7641 — costs 2.88 probe | same |
| L6 | cRT at 198 M | | launched 2026-08-06, **result never journalled**; G1 answers the same question | same |
| L7 (cap 250/500/1000) | where is the optimal cap per species? | cap training images per species | cap 1,000: **+1.57 probe, +3.26 probe-HO, −0.88 in-dist** vs ~2,000 | [L7](journal/2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts.md) |
| L7c1000b | repeat of cap 1,000 | | survives n = 2: +1.76 / +2.94 | same |
| M2 | repeat of the marginal-supervision shifted number | | the multi-head's apparent lead was noise | [marginal](journal/2026-07-30-marginal-supervision.md) |

### R — the self-training gate (closed)

| ID | change | probe / probe-HO | where |
|---|---|---|---|
| R2 | relabel with F2, keep top 30 % by confidence | 0.7161 / 0.7257 — falsified by 4.2 pt | [iteration](journal/2026-08-08-self-training-does-not-iterate.md) |
| R3 | per-species cap instead | 0.7585 / 0.7682 | same |
| R4 | no gate at all | 0.7674 / 0.7458 | same |
| R5 | no gate, balanced | 0.7692 / 0.7781 | same |
| R5b | repeat of R5 | 0.7573: frozen-stage probe spread is **0.0119**, 3x the old floor | [noise floor](journal/2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes.md) |

### T — what labels would have bought, and where adaptation happens (closed)

| ID | question | result | where |
|---|---|---|---|
| T1 (lab500 / 2500 / 12230) | real trap labels instead of pseudo-labels | 12,230 real labels: probe 0.7568 — **below** self-training at 2 % (0.7706) | [labels](journal/2026-08-05-label-budget.md) |
| **T2** | adapt only the classifier, trunk frozen, 2 epochs | probe 0.7572 / probe-HO 0.7621: **83 %** of self-training's gain | [T2](journal/2026-08-06-adaptation-is-mostly-a-classifier-problem.md) |
| T2b | the same from a trunk with no trap augmentation | 0.7515 — the features were already good | same |

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
| **O3** | is O1's gap a model property or a readout artefact? | neither: the metric was rank-blind. Held-out nights, calibrated B8 vs P5: correct at species **74.7 vs 66.0 %**, at any rank **77.2 vs 83.3 %**; raw B8 (69.7 % of softmax values exactly 1.0) answers 0 % at species | [O3](journal/2026-10-02-is-the-deployment-gap-a-readout-artefact.md) |

### K and W — directions opened 2026-08-28 (first named D1-D3)

| ID | question | status / result | where |
|---|---|---|---|
| K1 | restrict the label space to the regional (Danish) checklist | done: probe micro-accuracy **+5.06**, macro-F1 **−1.70**; held-out species macro-F1 +6.31 | [D1](journal/2026-08-28-a-regional-checklist-helps-the-user-and-hurts-the-tail.md) |
| W1 | download TreeOfLife-200M at our data policy | **running** (re-crawl at short side 256, on the workstation) | [crawl](journal/engineering/2026-10-01-the-crawl-resized-the-wrong-side.md) |
| W2 | train our objective on ToL-10M vs BioCLIP-1 | not started | [D2](journal/2026-08-28-two-directions-checklists-and-our-objective-on-tol.md) |
| W3 | does training on the whole tree of life help open-set and shift? | designed; waits for W1 | [D3](journal/2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know.md) |

## 3. July local runs (frozen)

Generated on 2026-08-28 by the July run ledger (now `archive/dev/036_ledger.py`), which read run folders in the layout of the `dev/030` trainer on the workstation. Runs since then were on UCloud and are recorded in section 2.

Generated 2026-08-28 13:44 from `data/global/{models,preds}`, which is gitignored machine-local storage — this file is the only copy of this table that leaves the training box.

`delta` is the config keys differing from the baseline, i.e. what the run was testing. F1 is **species (level 0) macro-F1**: `val` from the run's own CSV, `test` from mini_metrics on the held-out fold (`set == '0'`). Reasoning behind these numbers lives in [`journal/`](journal/).

Baseline: `20260712-072542-heads-global-independent-muon-effnetv2s`

| run | model | ep | delta vs baseline | val F1 | test F1 | status |
|---|---|---|---|---|---|---|
| `20260723-042430` | backbone-effnetv2b2-h256-5ep-oversample | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 model_arch_name=tf_efficientnetv2_b2 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8815 | **0.8871** | done |
| `20260722-203133` | backbone-fastvit_t12-h256-5ep-oversample | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 model_arch_name=fastvit_t12 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8726 | **0.8800** | done |
| `20260722-203101` | backbone-mnv4_medium-h256-5ep-oversample | 0 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 model_arch_name=mobilenetv4_conv_medium oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | — | — | died |
| `20260722-142848` | backbone-repvit_m1_1-h256-5ep-oversample | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 model_arch_name=repvit_m1_1 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8743 | **0.8811** | done |
| `20260722-060901` | backbone-fastvit_sa12-h256-5ep-oversample | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 model_arch_name=fastvit_sa12 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8866 | **0.8920** | done |
| `20260722-004645` | backbone-effnetv2b0-h256-5ep-oversample | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 model_arch_name=tf_efficientnetv2_b0 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8701 | **0.8760** | done |
| `20260721-161210` | bottleneck-128-5ep-oversample-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=128 logit_adjust_tau=0.0 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8757 | **0.8843** | done |
| `20260721-074537` | bottleneck-512-5ep-oversample-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=512 logit_adjust_tau=0.0 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.9017 | **0.9058** | done |
| `20260720-200830` | bottleneck-256-5ep-oversample-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 hidden=256 logit_adjust_tau=0.0 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.8959 | **0.9002** | done |
| `20260718-121017` | heads-global-autoregressive-bf16-10ep-oversample-local-effnetv2s | 4 | aug_kwargs.*(6) grad_clip=5.0 head=autoregressive logit_adjust_tau=0.0 nb_epochs=10 oversample_power=0.5 precision=bf16 schedule=one_cycle warmup_epochs=0.5 | 0.7310 | — | died |
| `20260718-101317` | heads-global-autoregressive-bf16-fam9717-effnetv2s | 2 | aug_kwargs.*(6) family_filter=['9717'] grad_clip=5.0 head=autoregressive logit_adjust_tau=0.0 nb_epochs=2 oversample_power=0.5 precision=bf16 schedule=one_cycle warmup_epochs=0.5 | 0.6930 | — | done |
| `20260716-234247` | heads-global-independent-muon-5ep-logitadjust-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 logit_adjust_tau=1.0 oversample_power=0.0 schedule=one_cycle warmup_epochs=0.5 | 0.8925 | **0.9031** | done |
| `20260716-154156` | heads-global-independent-muon-5ep-oversample-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 logit_adjust_tau=0.0 oversample_power=0.5 schedule=one_cycle warmup_epochs=0.5 | 0.9096 | **0.9148** | done |
| `20260716-105029` | heads-global-independent-muon-onecycle-10ep-effnetv2s-resume | 2 | aug_kwargs.*(6) grad_clip=5.0 nb_epochs=10 schedule=one_cycle warmup_epochs=0.5 | 0.8977 | **0.8976** | done |
| `20260715-073321` | heads-global-independent-muon-onecycle-10ep-effnetv2s | 8 | aug_kwargs.*(6) grad_clip=5.0 nb_epochs=10 schedule=one_cycle warmup_epochs=0.5 | 0.8815 | — | died |
| `20260714-194321` | heads-global-independent-muon-onecycle-reg-effnetv2s | 5 | aug_kwargs.*(6) class_reg_strength=0.001 grad_clip=5.0 schedule=one_cycle warmup_epochs=0.5 | 0.8860 | — | done |
| `20260714-072404` | heads-global-independent-muon-onecycle-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 schedule=one_cycle warmup_epochs=0.5 | 0.8880 | **0.8887** | done |
| `20260713-164456` | heads-global-independent-muon-warmup-effnetv2s | 5 | aug_kwargs.*(6) grad_clip=5.0 warmup_epochs=0.5 | 0.8752 | **0.8769** | done |
| `20260712-184230` | heads-global-hierarchical-muon-effnetv2s | 5 | head=hierarchical | — | — | done |
| `20260712-072542` | heads-global-independent-muon-effnetv2s | 5 | (baseline) | — | **0.8297** | done |
| `20260711-154103` | multihead-v4-global-effnetv2s | 5 | decoder_nhead=unset decoder_num_layers=unset head=unset optimizer=unset precision=unset | — | — | done |
| `20260710-175046` | multihead-v4-global-effnetv2s | 5 | decoder_nhead=unset decoder_num_layers=unset head=unset optimizer=unset precision=unset schedule=one_cycle | — | — | done |
| `20260710-093034` | heads-global-hierarchical-effnetv2s | 1 | freeze_epochs=1 head=hierarchical nb_epochs=4 optimizer=adam schedule=fine_tune | — | — | done |
| `20260710-085450` | heads-global-independent-effnetv2s | 1 | freeze_epochs=1 nb_epochs=4 optimizer=adam schedule=fine_tune | — | — | done |

**Best test species macro-F1: 0.9148** (`20260716-154156` heads-global-independent-muon-5ep-oversample-effnetv2s, micro-acc 0.9476)
