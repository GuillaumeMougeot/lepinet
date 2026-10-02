# journal/ — one entry per question

Each entry asks one question, states a prediction before the result, and records the answer,
including dead ends. Files are named `YYYY-MM-DD-question.md` by the day the question was **opened**,
so `ls` shows the order things were asked. An entry is frozen once `RESOLVED`. Every entry declares
`**Kind:**` (research, subproject, infrastructure, incident) and `**Status:**`.

You do not read the journal front to back. Start from a finding in [`../RESULTS.md`](../RESULTS.md)
and follow its link here. Research entries are in this folder; side projects, cluster work and
incidents are in [`archive/`](archive/). The plan is [`../PLAN.md`](../PLAN.md).

## Research

| opened | question | answer |
|---|---|---|
| [07-16](2026-07-16-why-was-fastai-behind-mini-trainer.md) | Why was the new training loop 6 pt behind the old one? | Under-annealing; fixed, then overtaken (0.8976) |
| [07-17](2026-07-17-does-longtail-help.md) | Do oversampling or logit adjustment help the tail? | √-oversampling: 0.9148, the recipe; logit adjustment 0.9031 |
| [07-24](2026-07-24-bigger-everything.md) | Does a bigger backbone help? | ConvNeXtV2-L 0.9316, best in-distribution |
| [07-28](2026-07-28-flemming-generalization.md) | Does that model survive trap images? | No: 0.6950, a ~23 pt gap; the start of the pivot |
| [07-29](2026-07-29-the-reframe-directions.md) | What is the real bottleneck once heads are a null result? | Reliable prediction under shift, not accuracy (formerly `DIRECTIONS.md`) |
| [07-30](2026-07-30-domain-shift.md) | Does trap-like augmentation close the gap? (B1) | +4.0 pt under shift, only 17 % of the gap |
| [07-30](2026-07-30-marginal-supervision.md) | Does a loss on the marginals help? | Species unchanged; coarse levels up; +1.4 pt under shift |
| [07-30](2026-07-30-does-arcface-compose-with-marginalisation.md) | Do the single head and ArcFace compose? (A1, A2, A4) | Not on accuracy; open-set survives (AUROC 0.9068) |
| [07-31](2026-07-31-best-model-is-not-the-best-model.md) | Does the best in-distribution model deploy best? | No; later corrected by the scoring-rule entry |
| [08-01](2026-08-01-capacity-x-augmentation.md) | Do capacity and augmentation compose? (B4) | Better than additively on accuracy and shift |
| [08-01](2026-08-01-how-noisy-are-our-numbers.md) | How large is the run-to-run spread? (A5) | Species ~0.000; noise grows as class count falls |
| [08-01](2026-08-01-the-scoring-rule-was-the-bug.md) | Is the 198 M open-set loss in the model or the rule? (E2) | The rule: max-softmax beats max-logit by 6-7.6 pt |
| [08-01](2026-08-01-marginalisation-is-not-argmax-consistent.md) | Is marginalisation "consistent by construction"? | No; it is probabilistically coherent, not argmax-consistent |
| [08-01](2026-08-01-imbalance-methods-bench.md) | Do long-tail methods beat √-oversampling? (L0-L5) | They trade shift robustness for accuracy; cRT removes the trade |
| [08-02](2026-08-02-the-shifted-benchmark-is-also-the-adaptation-set.md) | Can self-training be scored on the trap benchmark? | No; built leak-free `probe` splits instead |
| [08-02](2026-08-02-f1-flagship.md) | Does composing every win at 198 M beat B4? (F1) | No; identical on species and shift |
| [08-03](2026-08-03-macro-f1-does-not-decompose.md) | Why do two benchmarks on the same images disagree? | Macro-F1 does not decompose over subsets |
| [08-03](2026-08-03-b3-self-training.md) | Does self-training on trap images help? (B3) | Yes, the largest lever: +4.58 probe |
| [08-04](2026-08-04-replication-sweep.md) | How much trap data? (B3rep) | Optimum at 2 % of training: probe 0.7706 |
| [08-05](2026-08-05-label-budget.md) | What would real labels have bought? (T1) | Less than self-training at its best dose |
| [08-05](2026-08-05-scaling-the-head.md) | How to reach 1 M species? (H) | No trained head works; centroids at inference, plus a data floor |
| [08-06](2026-08-06-adaptation-is-mostly-a-classifier-problem.md) | Does adaptation need the representation? (T2) | No: the classifier alone gets 83 % |
| [08-06](2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison.md) | Is the 31-pt open-set gain real? | No, retracted: 0.78 pt best-vs-best |
| [08-06](2026-08-06-f2-capstone.md) | Do the classifier-stage findings compose? (F2, F3, G1, G2) | Yes: in-dist 0.9081, probe 0.7541 at 20 M |
| [08-08](2026-08-08-self-training-does-not-iterate.md) | Does a second self-training round help? (R) | Yes, with no confidence gate and balanced classes |
| [08-08](2026-08-08-is-novelty-monotone-or-just-rare.md) | Is novelty graded by distance, or by rarity? (C3b) | By distance |
| [08-09](2026-08-09-can-centroids-be-trained-against.md) | Can a head train against centroids alone? (H4) | No: −4.63 pt |
| [08-10](2026-08-10-balance-is-oversampling-and-it-does-not-scale.md) | Does balanced replication survive 10x scale? (G3, B10) | Its effect shrinks with capacity |
| [08-24](2026-08-24-does-the-recipe-need-our-backbone.md) | Does the recipe work on any strong encoder? (P1) | Not frozen: BioCLIP-2 is 11 pt behind |
| [08-26](2026-08-26-bioclip2-has-seen-two-thirds-of-our-test-fold.md) | How much of our data is inside BioCLIP-2's training set? | 65 % of images, two thirds of the test fold |
| [08-26](2026-08-26-the-clip-projection-does-not-hurt-us.md) | Is the 768-d embedding cache usable? | Yes |
| [08-27](2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes.md) | Does G3's result reproduce? (G3b) | No: identical runs 3.74 pt apart; two claims retracted |
| [08-27](2026-08-27-tol-extra-data-is-almost-all-head.md) | How much usable data does TreeOfLife add for Lepidoptera? | 1.2x, almost all in common species |
| [08-27](2026-08-27-tol-at-our-policy-and-the-head-scaling-problem.md) | How big is TreeOfLife under our data policy? | 88 M images, ~185 k true species; the head fits |
| [08-28](2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts.md) | Fine-tuned BioCLIP-2; is our head cap right? (P3, P4, L7) | It beats us by 1.25; capping at 1,000 helps shift |
| [08-28](2026-08-28-frozen-adaptation-only-works-on-your-own-trunk.md) | Was P4's deficit real? (P5, P5b) | No: unfrozen, BioCLIP-2 ties B8 |
| [08-28](2026-08-28-two-directions-checklists-and-our-objective-on-tol.md) | Checklists at inference; our objective on TreeOfLife (K1, W1, W2) | Design; K1 resolved separately, W open |
| [08-28](2026-08-28-a-regional-checklist-helps-the-user-and-hurts-the-tail.md) | What is a regional checklist worth? (K1) | +5 pt accuracy, −1.7 pt macro-F1 on probe |
| [08-28](2026-08-28-two-tied-models-differ-by-17-points-in-deployment.md) | B8 vs P5 under abstention (O1) | Tied on accuracy, 17 pt apart on useful answers |
| [10-01](2026-10-01-what-the-blocked-servers-cost.md) | What do servers that refuse us cost? (W1) | 23 % of the selection, mostly herbaria; 89 % of Lepidoptera survive |
| [10-01](2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know.md) | Does training on the whole tree help open-set and shift? (W3) | **OPEN**, waits for the download |

## Archive: side projects, cluster work, incidents

| opened | kind | what |
|---|---|---|
| [07-16](archive/2026-07-16-gpu-hang.md) | incident | the training box hard-hung overnight (hardware) |
| [07-16](archive/2026-07-16-venv-uv-sync-incident.md) | incident | `uv sync` broke torch: never run it here |
| [07-17](archive/2026-07-17-ucloud-benchmark-oom.md) | infrastructure | UCloud jobs OOM-ing: dataloader workers |
| [07-18](archive/2026-07-18-ucloud-throughput.md) | infrastructure | making the B200 fast: decode-bound |
| [07-18](archive/2026-07-18-autoregressive-fp16-instability.md) | incident | the autoregressive head trained broken: fp16 overflow |
| [07-19](archive/2026-07-19-lepi-app.md) | subproject | can the model become an offline phone app? |
| [07-20](archive/2026-07-20-lepi-app-claude.md) | subproject | the app plan: size budget and decisions |
| [07-20](archive/2026-07-20-lepi-app-compression.md) | subproject | export, quantisation and calibration for a browser |
| [07-23](archive/2026-07-23-lepi-app-HANDOFF.md) | subproject | app handoff snapshot |
| [07-24](archive/2026-07-24-src-lepinet-baseline-port.md) | subproject | the clean `src/lepinet` package reproduces 0.9148 |
| [07-25](archive/2026-07-25-teacher-student-app-bridge.md) | subproject | distillation and the one-command bundle (A3, A6, D2) |
| [07-28](archive/2026-07-28-landscape-and-plan.md) | subproject | the July plan, superseded by `PLAN.md` |
| [07-30](archive/2026-07-30-ucloud-queue-daemon.md) | incident | the UCloud queue only advances when something ticks it |
| [08-06](archive/2026-08-06-the-cosine-head-is-not-unit-norm.md) | incident | the cosine head's rows are not unit-norm (no accuracy effect) |
| [08-24](archive/2026-08-24-three-week-report.md) | infrastructure | the report for 2-24 August: predictions scored, corrections |
| [08-24](archive/2026-08-24-work-storage-degraded.md) | incident | training stalled: `/work` read latency collapsed |
| [08-28](archive/2026-08-28-what-the-paper-is-still-missing.md) | infrastructure | paper audit: eight wrong numbers fixed |
| [09-29](archive/2026-09-29-public-hf-release.md) | subproject | public release of P5, B8, B3rep5x on Hugging Face |
| [09-30](archive/2026-09-30-full-fold-and-p5-resize.md) | subproject | P5's published preprocessing fixed |
| [10-01](archive/2026-10-01-the-crawl-that-spent-58-percent-of-the-cpu-budget.md) | incident | a crawler spent 58 % of the CPU budget |
| [10-01](archive/2026-10-01-the-crawl-resized-the-wrong-side.md) | incident | the ToL crawl stored the wrong side; a deadlock froze it (**OPEN**: re-crawl running) |
