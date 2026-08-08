# Start here — a guided tour of this repository

This is the **top of the map**. It exists so someone arriving with no context can reach any part of
the project — the code, the results, and the reasoning — in a few deliberate steps, without having
to guess which file matters.

The project is **hierarchical fine-grained image classification**: from one photo, predict a label at
every level of a taxonomy at once — for the reference dataset, the *species*, *genus* and *family* of
a moth or butterfly, over ~12,000 species with a heavy long tail.

---

## 1. Pick your entry point

| If you want to… | Go to | What you'll find |
|---|---|---|
| **Get up to speed on the vocabulary** *(logits, cosine head, margins, AUROC, marginalisation…)* | [`docs/concepts.md`](docs/concepts.md) | Every recurring term explained from the ground up, with this project's actual numbers |
| **Understand the problem & method** | [`README.md`](README.md) | What the task is, why it's hard (fine-grained + long-tailed), and the method (per-level cosine head + square-root oversampling) |
| **Use the package** (train / test / predict / export) | [`docs/user-guide.md`](docs/user-guide.md) | Install, the CLI (`lepinet train|test|predict|export|bundle|distill`), config reference |
| **Change the code** | [`docs/developer-guide.md`](docs/developer-guide.md) → then [`src/lepinet/README.md`](src/lepinet/README.md) | Architecture, module-by-module design, the lessons encoded in the code |
| **Know why a setting is what it is** *(before you change it)* | [`docs/design-decisions.md`](docs/design-decisions.md) | The ladder from 0.8297 to 0.9152, each recipe choice with what it was worth, and the things that didn't pay |
| **See what has been tried and what it scored** | [`RESULTS.md`](RESULTS.md) | Every run, its delta vs baseline, and its test score (+ a hand-kept table of the UCloud runs) |
| **Understand *why* decisions were made** | [`journal/README.md`](journal/README.md) | The master doc for *why*: how the project evolved in six phases, plus an index of every entry by kind |
| **Know what is running right now** | [`journal/PLAN.md`](journal/PLAN.md) | The status board: **§1 indexes every experiment group A–T with its state**, §2 what is running, §3 the ordered backlog, §6 one line per closed group. The one file meant to be true *today* |
| **Run experiments** | [`dev/README.md`](dev/README.md) *(if present)* / the numbered `dev/0NN_*.py` scripts | One script per experiment, importing the `lepinet` package |
| **Run on the GPU cluster** | [`ucloud/`](ucloud/) | One TOML per job (train / eval / benchmark), plus the shared `setup-lepinet.sh` |
| **See the phone app** | the companion repo `lepinet-app` + [`journal/2026-07-20-lepi-app-compression.md`](journal/2026-07-20-lepi-app-compression.md) | The browser PWA that consumes an exported bundle |

## 2. What this project has established

Each line is a result with a number, and links to the entry that argues it. Negative results are kept
deliberately — they cost GPU time to learn and are the first thing a newcomer would otherwise repeat.

The list is split because the two halves answer different questions. **§2a** is what the project
claims about the *problem* — findings that should hold on any long-tailed hierarchical dataset, and
the substance of the paper. **§2b** is what it learned building a working system on *this* one:
narrower, but the part that saves an engineer a week.

### 2a. Scientific findings

| # | finding | evidence |
|---|---|---|
| 1 | **Coarse *parameters* do not help; coarse *supervision* does — and only one of those is visible in-distribution.** Every head owning genus/family layers loses in-distribution to one that does not (conditional 0.8845, autoregressive 0.69–0.73, multi-head 0.9110, single head **0.9135**). But dropping the coarse *losses* costs **2.10 pt under domain shift** while costing nothing in-distribution. The four-head comparison is [paper §4.1](paper/DRAFT.md). | [heads](journal/2026-07-16-why-was-fastai-behind-mini-trainer.md), [marginal](journal/2026-07-30-marginal-supervision.md) |
| 2 | **One species head + marginalisation beats the multi-head at *every* level** (0.9135/0.9606/0.9739 vs 0.9110/0.9587/0.9708) — fewer parameters, and the coarse posterior is by definition the sum of the species one (**probabilistic coherence — not argmax agreement, which is not guaranteed**). **But it loses 2.1 pt under domain shift** unless the marginals are also *supervised during training*, which recovers 1.41 of it for free: **coarse supervision buys robustness, coarse parameters do not.** | [story](journal/DIRECTIONS.md), [marginal](journal/2026-07-30-marginal-supervision.md) |
| 3 | **The margin relocates open-set signal rather than creating it — the 31-point claim was a rule comparison.** Given each head its *best* scoring rule, ArcFace × z-score leads the plain cosine head by **0.78 pt** (0.9068 vs 0.8990), not 31, and trails it by 1.0 pt on accuracy. What the margin genuinely does: it moves the signal from the distribution's *shape* (plain head's best rule is entropy) into the *magnitude* of the top score (ArcFace's is max-logit), and makes the readout choice nearly free — its five rules span 1.2 pt against the plain head's 28.4. | [retraction](journal/2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison.md) |
| 4 | **An open-set scoring rule does not transfer across model scale.** `max-logit` is the best rule at 20 M and among the worst at 198 M, where `max-softmax-probability` beats it by **+6.1 to +7.6 pt**. Reading every model with one rule made a 1.6 pt capacity penalty look like 8.8 and produced a "ranking inversion" that was largely an artifact. As a model's fit improves, open-set signal moves from the *magnitude* of the top score to its *dominance* over the rest. | [scoring rule](journal/2026-08-01-the-scoring-rule-was-the-bug.md) |
| 4a | **Macro-F1 does not decompose over subsets, so two benchmarks on the *same images* can disagree and both be right.** F1 ties B4 on all 47,905 trap images (+0.0002) and beats it by **+2.03 pt** on a 15,200-image subset of them — because the full set weights 486 species at 1/486 and the subset weights its 368 at 1/368. Differences are only meaningful *within* a column. | [decomposition](journal/2026-08-03-macro-f1-does-not-decompose.md) |
| 4b | **A benchmark's exclusivity has to be written down or it gets spent twice.** The trap set was *the* external benchmark since July and also the only source of unlabelled trap images; nothing recorded that those roles conflict, so self-training would have trained on its own test set and reported a higher number. Caught before running, by building the split rather than by the pipeline. | [contamination](journal/2026-08-02-the-shifted-benchmark-is-also-the-adaptation-set.md) |
| 5 | **The three evaluation axes still disagree, but mildly.** With each model's best rule: A1 leads open-set (0.9068), B4 leads in-distribution (0.9216) and shift (0.7101) while giving up 1.75 pt of AUROC. In-distribution macro-F1 should not be the sole selection criterion — but the strong claim that the best in-distribution model is the *worst* deployable one did not survive scrutiny. | [inversion, corrected](journal/2026-07-31-best-model-is-not-the-best-model.md) |
| 6 | **Domain adaptation is mostly a *classifier* problem, not a representation one.** Freezing the backbone and retraining only the classifier for 2 epochs captures **83 %** of full self-training's gain (89 % of its transfer to unseen species). Adapting to a new camera is minutes of GPU, no labels, backbone untouched. **And it works from a trunk that never saw the target domain at all** — so `domain_aug` turns out to be a *substitute* for adaptation rather than a complement (+4.75 pt alone, +0.57 pt once the classifier is adapted). Same structural answer as cRT gave for the long tail: the representation is the robust, inert end and the classifier is the cheap, swappable one. | [T2](journal/2026-08-06-adaptation-is-mostly-a-classifier-problem.md) |
| 7 | **Unlabelled target data supplies *coverage of the input space* — and its *class distribution* is a second, separate design parameter.** Coverage first: 0.39 % of training buys 97 % of the gain, and trading 24.6 pt of label accuracy for 740 more species buys **+4.24 pt**. But the two shifted benchmarks then disagree — adding the head of the long-tailed trap distribution is worth **+0.89 probe and −2.24 held-out species**. Pseudo-labels are training data, so their class balance is a knob, not a property of the domain. | [iteration](journal/2026-08-08-self-training-does-not-iterate.md) |
| 8 | **Self-training on unlabelled target-domain images is the largest robustness lever, and the only free one.** **+7.94 pt** on held-out trap groups for ~0 in-distribution, with the gain transferring **essentially entirely** to species the adaptation never saw — at the right dose. How much target-domain data is a sharp interior optimum at **~2 % of training**: less works nearly as well, more converts adaptation into memorisation (transfer falls 121 % → 39 % as the share rises). A 20 M model with it beats a 198 M model without. It is also the first rung requiring nobody to *name* what differs between domains — unlike augmentation, which is bounded by imagination at ~4 pt. | [B3](journal/2026-08-03-b3-self-training.md) |
| 9 | **That trade is fixable: apply the rebalancing to the *classifier*, not the data.** cRT — train the representation on the natural distribution, then freeze it and rebalance only the classifier — recovers +1.19 pt of oversampling's in-distribution gain while scoring **+2.46 above** the fully-oversampled model under shift. The damage was in the representation all along. | [cRT](journal/2026-08-01-imbalance-methods-bench.md) |
| 10 | **Long-tail methods trade robustness for accuracy, monotonically.** Rank four cells by how hard they push mass toward rare classes — none / √-oversampling / balanced softmax / both — and the shifted score falls at *every* step: **0.6445 > 0.6293 > 0.5726 > 0.5492** (9.5 pt spread, 0.69 pt floor), while the in-distribution column does not order at all. Rare classes carry the least transferable evidence, so up-weighting them up-weights the part that breaks first. | [imbalance bench](journal/2026-08-01-imbalance-methods-bench.md) |
| 11 | **Domain-mimicking augmentation is a down-payment, not a fix**: three hand-named nuisances (blur, low light, JPEG) buy **+4.0 pt under shift for −0.36 in-distribution** — an 11:1 trade — yet close only **17 %** of the gap. What you can name is worth about four points. | [domain shift](journal/2026-07-30-domain-shift.md) |
| 12 | **An angular margin degrades marginalisation, not classification.** ArcFace × z-score costs ~1 pt at species but ~1.15 pt at genus/family, because summing a posterior is calibration-dependent and a margin sharpens boundaries at calibration's expense. **Replicated across a 10× backbone scale change.** The mirror image: supervising the marginals leaves species *exactly* unchanged while lifting coarse levels +0.27/+0.39. | [compose](journal/2026-07-30-does-arcface-compose-with-marginalisation.md), [marginal](journal/2026-07-30-marginal-supervision.md) |
| 13 | **In-distribution accuracy is near-saturated but generalisation is not**: 0.9316 in-domain → **0.6950** on an external source (~23 pt gap), and open-set AUROC falls 0.9115 → 0.7272 with it. Shift makes *known* species look unfamiliar. | [flemming](journal/2026-07-28-flemming-generalization.md), [domain shift](journal/2026-07-30-domain-shift.md) |
| 14 | **Knowledge distillation works, but the student is the ceiling.** T=1 beats from-scratch (0.8786 vs 0.8692); a 2 pt better teacher moved the student by ~0. **KD temperature is not head-agnostic** — the textbook T=4 *hurt* (0.8546). | [bridge](journal/2026-07-25-teacher-student-app-bridge.md) |
| 15 | **Scale pays in-distribution; for robustness, augmentation is the better buy.** ConvNeXtV2-L 0.9316 (+1.7 pt), and a DINOv3-distilled ConvNeXt matches it ~2× faster. But under shift a 20 M model with `domain_aug` beats a 198 M one without it, and the two compose better than additively (the augmentation tax vanishes at scale while its gain grows). | [bigger everything](journal/2026-07-24-bigger-everything.md), [factorial](journal/2026-08-01-capacity-x-augmentation.md) |
| 16 | **Methodological**: an `lr_find`-style range test is invalid for a margin (it mechanically raises the loss); and a 2-D projection is the wrong picture for an angular effect (silhouette barely moves while AUROC moves 30 pt). | [directions](journal/DIRECTIONS.md) |

### 2b. How the baseline was built — the engineering findings

The headline model did not arrive designed; it was climbed from **0.8297 to 0.9152** one change at a
time, and what moved it is not what one would guess. Full argument and the ladder table in
[`docs/design-decisions.md`](docs/design-decisions.md).

| # | finding | evidence |
|---|---|---|
| 17 | **The optimiser was never the lever.** Muon was in place at 0.8297 and stayed. What moved the number was the *schedule* (`one_cycle` over `flat_cos`, +1.2 pt), the *sampler* (√-oversampling, +2.6 pt) and *lighter* augmentation — what the model sees and for how long, not how gradients are applied. | [design decisions](docs/design-decisions.md), [ladder](journal/2026-07-16-why-was-fastai-behind-mini-trainer.md) |
| 18 | **A mechanism that explains a result is not the same as a tested one.** √-oversampling beating logit adjustment was blamed on one shared τ spanning three level distributions. Re-run on a single-head architecture where no shared constant exists, it still loses — the explanation was a rationalisation and is retracted. The satisfying story survived four weeks unchecked. | [long tail](journal/2026-07-17-does-longtail-help.md), [bench](journal/2026-08-01-imbalance-methods-bench.md) |
| 19 | **bf16 is not optional** for cosine heads — fp16 overflows them. The autoregressive head's "wiring bug" was this, and was misdiagnosed for days. | [fp16](journal/2026-07-18-autoregressive-fp16-instability.md) |
| 20 | **Before hunting a bug, check both numbers mean the same thing.** A "0.92 val vs 0.83 test" fold bug did not exist: one metric averaged three taxonomic levels, the other was species-only. Like-for-like, both were 0.83. | [fastai gap](journal/2026-07-16-why-was-fastai-behind-mini-trainer.md) |
| 21 | **Audit the eval set before believing the metric.** A port "beat" its own baseline 0.9455 vs 0.9148 — because the eval had filtered the long tail out of a *macro* average. Unfiltered: 0.9152, i.e. an exact reproduction. | [port](journal/2026-07-24-src-lepinet-baseline-port.md) |
| 22 | **Framework attributes can lie.** fastai hardcodes `num_workers` to 1; the true value is on `fake_l`. Reading the wrong one ran evaluations at ~1 img/s instead of 898 — a ~900× slowdown first misdiagnosed as a hardware problem. | [design decisions](docs/design-decisions.md) |
| 23 | **Deployment**: int8 cannot run in ORT-Web (no `ConvInteger` kernel) but **source-level fp16 can** (−28 % size, identical top-1); GitHub *release* assets send no CORS, so they cannot serve a browser (Hugging Face Hub can). The cosine head is ~51 % of a small model's parameters, so the bottleneck width (256) is the real size knob. | [bridge](journal/2026-07-25-teacher-student-app-bridge.md), [compression](journal/2026-07-20-lepi-app-compression.md) |

> **Where each list lives, and why.** The scientific findings (§2a) are stated formally in
> [`paper/DRAFT.md`](paper/DRAFT.md); the engineering ones (§2b) are argued in
> [`docs/design-decisions.md`](docs/design-decisions.md). Both appear here as one-line summaries for
> newcomers, and *chronologically* in [`journal/README.md`](journal/README.md). Four views of one
> truth, each with a different reader — no fifth copy.

## 3. The current baseline — what to compare against

**For any new experiment: `efficientnet_v2_s`, single species head, marginalisation, 5 epochs,
sqrt-oversampling → species macro-F1 0.9135.** Config:
[`configs/20260729_ucloud_singlehead_species_effnetv2s.yaml`](configs/20260729_ucloud_singlehead_species_effnetv2s.yaml).
Cheap, and every architectural comparison in the project is anchored to it.

There are **three different external benchmarks** and they are not interchangeable (finding 4a):

| column | images | species | what it is |
|---|---|---|---|
| **full trap** | 47,905 | 486 | every trap image. **Contaminated for anything trained on trap data** |
| **probe** | 15,200 | 368 | whole (trap, night) groups held out of adaptation. The honest column |
| **probe-HO** | 2,455 | 58 | probe restricted to species adaptation never saw. Tests *generalisation* |

| purpose | model | in-dist | full trap | probe | probe-HO | open-set |
|---|---|---|---|---|---|---|
| cheap reference | effnetv2_s, single head + marginals | **0.9135** | 0.629 | 0.627 | 0.641 | ~0.60 |
| best in-distribution, no open-set | ConvNeXtV2-L @320, multi-head | **0.9316** | 0.7122 | — | — | — |
| **best deployable — SHIP THIS** | **B8**: F1 minus √-oversampling, plus self-training at the 2 % dose | 0.9060 | — | **0.7798** | **0.7816** | — |
| best in-distribution, open-set capable | F1: DINOv3-cnx-L + ArcFace × z-score + marginal supervision + trap aug | **0.9219** | 0.7103 | 0.7209 | 0.7559 | 0.8800 |
| best robustness per parameter | **B3**: B1 + self-training on trap images (20 M) | 0.9003 | *n/a* | **0.7370** | 0.7231 | — |
| best novelty detection | A1: effnetv2_s + ArcFace × z-score | 0.9035 | 0.6437 | — | — | **0.9068** |
| best small robust model, no trap data | B1: A1 + `domain_aug: trap` | 0.8999 | 0.6836 | 0.6912 | 0.6974 | 0.9010 |
| shippable student | fastvit_sa12, distilled from A2 | **0.8967** | 0.6301 | — | — | — |

> **Three rules for reading this table**, each learned by getting it wrong:
>
> 1. **Take differences within a column only.** Macro-F1 averages per-class scores *within* the set
>    it is given, so the full-trap and probe columns weight different species sets. F1 ties B4 on
>    full trap and beats it by 2 pt on probe, and both are correct (finding 4a).
> 2. **Name the open-set rule.** AUROC is each model's *best* rule — `max-logit` at 20 M,
>    `max-softmax-probability` at 198 M, `entropy` for log-probability heads. One rule across all
>    models understated the large ones by 6–7.6 pt (finding 4).
> 3. **Check the noise floor before believing a difference.** Measured: species **0.0000**, genus
>    0.0005, family 0.0024, full trap **0.0069**, probe **0.0041**, probe-HO **0.0052**.

## 4. The 90-second version of the project's state

- A clean, fastai-only package (`src/lepinet`) reproduces the project-best baseline:
  **species macro-F1 0.9152**.
- Scaling up works: **ConvNeXtV2-L → 0.9316**; a DINOv3-distilled ConvNeXt matches it ~2× faster.
  In-distribution accuracy is essentially solved.
- **Knowledge distillation works** (`lepinet distill`): a small student beats its from-scratch
  equivalent — but caps at ~0.88 by its own capacity, not the teacher's quality.
- **`lepinet bundle`** turns any checkpoint into a deployable ONNX bundle, and such a bundle is
  **plug-and-play in the companion phone app** (validated in a real browser).
- **The open problem:** a model at **0.93 in-distribution drops to ~0.70 on external data**, and
  real datasets contain species the model was never trained on. So the current direction is
  *reliable prediction that knows what it doesn't know* — see
  [`journal/DIRECTIONS.md`](journal/DIRECTIONS.md).

## 5. How the layers fit together

```
START-HERE.md          <- you are here: the map
├── CLAUDE.md          <- the operating manual for an AI agent: invariants, culture, the
│                         documentation contract. Auto-loaded; humans may read it too.
├── README.md          <- the problem + the method (start reading here)
├── docs/              <- how to USE it, how to CHANGE it, and WHY it is this way
│   ├── concepts.md         <- the vocabulary, explained from the ground up
│   ├── user-guide.md       <- install + the CLI
│   ├── developer-guide.md  <- architecture + how to extend it
│   ├── design-decisions.md <- why each recipe choice is what it is, and what it was worth
│   └── (published as a website via MkDocs; see mkdocs.yml)
├── src/lepinet/       <- the package: the stable, tested implementation
│   └── README.md      <- module-by-module tour
├── dev/               <- experiments: numbered scripts that import the package
├── configs/           <- one YAML per training run (the source of truth for a run)
├── ucloud/            <- one TOML per cluster job
├── paper/DRAFT.md     <- the scientific claims, stated formally
├── journal/           <- WHY, as it happened: one entry per question, dated, negative results kept
│   ├── README.md      <- the master doc: how the project evolved + an index by kind
│   ├── PLAN.md        <- LIVING: the experiment-group index (A-T), what is running, the backlog
│   └── DIRECTIONS.md  <- LIVING: the research strategy
├── RESULTS.md         <- WHAT it scored (the numbers)
└── tests/             <- what must keep working (runs on CPU, no dataset needed)
```

**The rule of thumb:** `RESULTS.md` tells you *what* happened, `journal/` tells you *why*, `src/`
is *how*, and `dev/` is *what we're trying next*.

## 6. Conventions worth knowing before you dig in

- **The journal splits into living and archival.** `UPPERCASE.md` files (`PLAN`, `DIRECTIONS`) have
  no date and are kept current; `YYYY-MM-DD-question.md` files are frozen once `RESOLVED`, and the
  date is when the question was *opened*, so `ls journal/` reads in the order things were asked.
- **The journal is one file per _question_, not per run** — and a hypothesis is written *before* the
  result lands, so predictions are tested rather than rationalised. Negative results are kept on
  purpose; they cost real GPU time to learn.
- **Runs are cited by id** (`20260716-154156`), never by adjective.
- **`data/` is machine-local and gitignored** — a fresh clone has no runs. `RESULTS.md` is the only
  copy of those numbers that leaves the training box.
- **Metrics:** the headline is **species macro-F1** (every species weighted equally, so the long tail
  counts) on the held-out fold (`set == '0'`) over **all** species. Beware of filtering the test fold
  — see the eval-set lesson in
  [`journal/2026-07-24-src-lepinet-baseline-port.md`](journal/2026-07-24-src-lepinet-baseline-port.md).
