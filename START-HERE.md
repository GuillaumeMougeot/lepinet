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

## 2. The 90-second version of the project's state

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

## 3. How the layers fit together

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

## 4. Conventions worth knowing before you dig in

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
