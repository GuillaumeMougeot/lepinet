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
| **Understand the problem & method** | [`README.md`](README.md) | What the task is, why it's hard (fine-grained + long-tailed), and the method (per-level cosine head + square-root oversampling) |
| **Use the package** (train / test / predict / export) | [`docs/user-guide.md`](docs/user-guide.md) | Install, the CLI (`lepinet train|test|predict|export|bundle|distill`), config reference |
| **Change the code** | [`docs/developer-guide.md`](docs/developer-guide.md) → then [`src/lepinet/README.md`](src/lepinet/README.md) | Architecture, module-by-module design, the lessons encoded in the code |
| **See what has been tried and what it scored** | [`RESULTS.md`](RESULTS.md) | Every run, its delta vs baseline, and its test score (+ a hand-kept table of the UCloud runs) |
| **Understand *why* decisions were made** | [`journal/README.md`](journal/README.md) | The reasoning, the dead ends, the negative results — starts with a big-picture summary of how the project evolved |
| **Run experiments** | [`dev/README.md`](dev/README.md) *(if present)* / the numbered `dev/0NN_*.py` scripts | One script per experiment, importing the `lepinet` package |
| **Run on the GPU cluster** | [`ucloud/`](ucloud/) | One TOML per job (train / eval / benchmark), plus the shared `setup-lepinet.sh` |
| **See the phone app** | the companion repo `lepinet-app` + [`journal/2026-07-lepi-app-compression.md`](journal/2026-07-lepi-app-compression.md) | The browser PWA that consumes an exported bundle |

## 2. What this project has established

Each line is a result with a number, and links to the journal entry that argues it. Negative results
are kept deliberately — they cost GPU time to learn and are the first thing a newcomer would
otherwise repeat.

| # | finding | evidence |
|---|---|---|
| 1 | **Hierarchical prediction heads do not help.** Parent-conditioned 0.8845 and autoregressive 0.69–0.73 both lose to a plain multi-head 0.9110, on identical configs. | [heads](journal/2026-07-why-was-fastai-behind-mini-trainer.md) |
| 2 | **One species head + marginalisation beats the multi-head at *every* level** (0.9135/0.9606/0.9739 vs 0.9110/0.9587/0.9708) — fewer parameters, and coarse levels cannot contradict the species call. | [story](journal/2026-07-story-and-directions.md) |
| 3 | **ArcFace × z-score turns novelty detection from chance into usable**: open-set AUROC 0.601 → **0.9115** for −0.4 pt accuracy. The margin *alone* costs 3.3 pt and reaches only 0.732 — the trade-off was an artefact of discarding the calibrated transform, not intrinsic. | [story](journal/2026-07-story-and-directions.md) |
| 4 | **In-distribution accuracy is near-saturated but generalisation is not**: 0.9316 in-domain → **0.6950** on an external source (~23 pt gap), and open-set AUROC falls 0.9115 → 0.7272 with it. Shift makes *known* species look unfamiliar. | [flemming](journal/2026-07-flemming-generalization.md), [domain shift](journal/2026-07-domain-shift.md) |
| 5 | **Knowledge distillation works, but the student is the ceiling.** T=1 beats from-scratch (0.8786 vs 0.8692); a 2 pt better teacher moved the student by ~0. **KD temperature is not head-agnostic** — the textbook T=4 *hurt* (0.8546). | [bridge](journal/2026-07-teacher-student-app-bridge.md) |
| 6 | **Scale pays, then plateaus.** ConvNeXtV2-L 0.9316 (+1.7 pt); a DINOv3-distilled ConvNeXt matches it at ~2× the training speed. | [bigger everything](journal/2026-07-bigger-everything.md) |
| 7 | **Deployment findings**: int8 cannot run in ORT-Web (no `ConvInteger` kernel) but **source-level fp16 can** (−28 % size, identical top-1); GitHub *release* assets send no CORS so they cannot serve a browser. | [bridge](journal/2026-07-teacher-student-app-bridge.md) |
| 8 | **Methodological**: an `lr_find`-style range test is invalid for a margin (it mechanically raises the loss); and a 2-D projection is the wrong picture for an angular effect (silhouette barely moves while AUROC moves 30 pt). | [story](journal/2026-07-story-and-directions.md) |

> Where this list lives, and why: **here** for newcomers (one line + a link, no argument),
> formally in [`paper/DRAFT.md`](paper/DRAFT.md) (the scientific claims), and *chronologically* in
> [`journal/README.md`](journal/README.md) (how it evolved). Three views of one truth, no fourth copy.

## 3. The current baseline — what to compare against

**For any new experiment: `efficientnet_v2_s`, single species head, marginalisation, 5 epochs,
sqrt-oversampling → species macro-F1 0.9135.** Config:
[`configs/20260729_ucloud_singlehead_species_effnetv2s.yaml`](configs/20260729_ucloud_singlehead_species_effnetv2s.yaml).
It replaced the old multi-head 0.9110 reference because it dominates it at every level while being
smaller.

| purpose | model | score |
|---|---|---|
| **cheap reference** (change one thing, compare here) | effnetv2_s single head | **0.9135** |
| best in-distribution / distillation teacher | ConvNeXtV2-L @320 | 0.9316 |
| best open-set | ArcFace × z-score (multi-head) | 0.9069 F1 / **0.9115** AUROC |
| shippable student | distilled b0, fp16 | 0.8786 |

> **Caveat worth knowing before trusting a comparison:** results 3, 5 and 6 above were obtained on
> the *old multi-head* baseline. The recommended architecture — **single head + ArcFace × z-score +
> marginalisation** — has therefore **never actually been trained**. That combination, and
> re-running distillation from a single-head teacher, are the highest-value pending experiments.

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
  [`journal/2026-07-story-and-directions.md`](journal/2026-07-story-and-directions.md).

## 5. How the layers fit together

```
START-HERE.md          <- you are here: the map
├── README.md          <- the problem + the method (start reading here)
├── docs/              <- how to USE it (user guide) and how to CHANGE it (developer guide)
│   └── (published as a website via MkDocs; see mkdocs.yml)
├── src/lepinet/       <- the package: the stable, tested implementation
│   └── README.md      <- module-by-module tour
├── dev/               <- experiments: numbered scripts that import the package
├── configs/           <- one YAML per training run (the source of truth for a run)
├── ucloud/            <- one TOML per cluster job
├── journal/           <- WHY: one entry per question, with the reasoning and the negative results
│   └── README.md      <- big-picture evolution + an index table of every entry
├── RESULTS.md         <- WHAT it scored (the numbers)
└── tests/             <- what must keep working (runs on CPU, no dataset needed)
```

**The rule of thumb:** `RESULTS.md` tells you *what* happened, `journal/` tells you *why*, `src/`
is *how*, and `dev/` is *what we're trying next*.

## 6. Conventions worth knowing before you dig in

- **The journal is one file per _question_, not per run** — and a hypothesis is written *before* the
  result lands, so predictions are tested rather than rationalised. Negative results are kept on
  purpose; they cost real GPU time to learn.
- **Runs are cited by id** (`20260716-154156`), never by adjective.
- **`data/` is machine-local and gitignored** — a fresh clone has no runs. `RESULTS.md` is the only
  copy of those numbers that leaves the training box.
- **Metrics:** the headline is **species macro-F1** (every species weighted equally, so the long tail
  counts) on the held-out fold (`set == '0'`) over **all** species. Beware of filtering the test fold
  — see the eval-set lesson in
  [`journal/2026-07-src-lepinet-baseline-port.md`](journal/2026-07-src-lepinet-baseline-port.md).
