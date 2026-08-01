# The journal — the project's reasoning, in order

This is the **master document for why**. `RESULTS.md` says what scored what; `dev/036_ledger.py`
reads every run's config off disk; neither records what we were trying to learn, what we predicted,
or what we concluded. That is this directory.

```
dev/036_ledger.py   ->  reads every run's config + metrics off disk
RESULTS.md          ->  the numbers   (generated: `dev/036_ledger.py --snapshot`, then commit)
journal/            ->  the reasoning (written by hand — the part that can be lost)
```

`RESULTS.md` is generated but **tracked on purpose**: the ledger's source lives under `data/`, a
symlink to machine-local storage that is gitignored, so a clone anywhere else sees no runs at all.
The snapshot is the only copy of those numbers that leaves the training box, and
`git log -p RESULTS.md` is the project's result history.

---

## Where we are right now

**→ [`PLAN.md`](PLAN.md)** — the single living plan: the status board of every run in flight, the
ordered backlog, and what is deliberately *not* being done. It is the only file here that is expected
to be correct *today*; everything else is a record of a moment.

**→ [`DIRECTIONS.md`](DIRECTIONS.md)** — the living research strategy: what the results say the real
bottleneck is, and why the project reframed around open-set reliability rather than accuracy.

---

## Two kinds of file, and how to tell them apart

| | name | closes? |
|---|---|---|
| **Living** | `UPPERCASE.md`, no date | never — kept current, rewritten in place |
| **Archival** | `YYYY-MM-DD-question.md` | yes — opened `OPEN`, closed `RESOLVED`, then frozen |

The date is the day the **question was opened**, not the day it was answered — so `ls` gives the
order things were *asked*, which is the order the reasoning developed. Living documents carry no
date, because a creation date on a file that is continuously rewritten is a lie.

Archival entries carry a **kind**, because they are not all the same animal:

- **research** — a question settled by experiments. The scientific record.
- **subproject** — a build effort with phases and a definition of done.
- **infrastructure** — how to make the machines work: cluster, memory, throughput.
- **incident** — something broke. Root cause and fix, so it is not paid for twice.

## Conventions

**Name by question, dated:** `2026-07-16-why-was-fastai-behind-mini-trainer.md`. Not
`run-20260714.md` — a run number is not a thought.

**Write the hypothesis before the results land.** A file opened while the GPU is busy says what you
expect and why. When the number arrives you are testing a prediction instead of rationalising an
outcome. Open with `OPEN`, close with `RESOLVED` + the answer.

**Record negative results.** "The class-distribution regulariser is a wash" cost a full training run
to learn and is the first thing forgotten. A wash is a finding. So is a crash.

**Cite runs by id** (`20260714-072404`), never by adjective ("the good run"). Ids resolve in the
ledger and on disk; adjectives resolve nowhere.

**Keep the detail; don't transcribe.** Per-epoch numbers are in the CSV, metrics in `metrics.json`.
Link them. Write here only what those files cannot say: the reasoning, the dead ends, the thing you
would tell someone to save them a week.

---

## How the project evolved

Six phases, in order. The engineering lessons each one produced are consolidated in
[`../docs/design-decisions.md`](../docs/design-decisions.md); the scientific claims are in
[`../paper/DRAFT.md`](../paper/DRAFT.md).

1. **Catch up to the old pipeline** (Jul 16–18) — the new fastai loop started 6 pt *behind* the
   `mini_trainer` loop it was replacing. Closing that gap is what produced the recipe: annealing
   dominates, Muon helps, bf16 is mandatory for cosine heads, sqrt-oversampling beats logit
   adjustment. [[2026-07-16-why-was-fastai-behind-mini-trainer]] · [[2026-07-17-does-longtail-help]]
   · [[2026-07-18-autoregressive-fp16-instability]]
2. **Make the cluster usable** (Jul 17–18) — the B200 sat idle because the pipeline is CPU-decode
   bound and the dataloader workers were OOM-ing the node.
   [[2026-07-17-ucloud-benchmark-oom]] · [[2026-07-18-ucloud-throughput]]
3. **Ship it to a phone** (Jul 19–23) — can a 173 MB model become an offline browser app? Yes, at
   14 MB, via bottleneck + backbone swap + quantization.
   [[2026-07-19-lepi-app]] · [[2026-07-20-lepi-app-claude]] · [[2026-07-20-lepi-app-compression]] ·
   [[2026-07-23-lepi-app-HANDOFF]]
4. **Rebuild clean** (Jul 24) — reimplemented the whole pipeline as a fastai-only `src/lepinet`
   package and reproduced the project best from scratch (**0.9152 ≈ 0.9148**).
   [[2026-07-24-src-lepinet-baseline-port]]
5. **Scale** (Jul 24–25) — bigger backbones lift it to **0.9316**, and a teacher→student→app bridge
   makes shipping one command. **In-distribution accuracy is essentially solved.**
   [[2026-07-24-bigger-everything]] · [[2026-07-25-teacher-student-app-bridge]]
6. **The pivot** (Jul 28–30) — the head bake-off is a **null result**, and a model at 0.93 drops to
   **~0.70 on external data**. So the story is not heads or accuracy — it is *prediction that knows
   what it doesn't know*: open-set in image space, abstention in hierarchy space, both under domain
   shift. [[2026-07-28-flemming-generalization]] · [[2026-07-30-domain-shift]] ·
   [[2026-07-30-marginal-supervision]] · [`DIRECTIONS.md`](DIRECTIONS.md)
7. **The inversion** (Jul 31) — with all three axes finally measured on one architecture family, they
   rank our models in **opposite orders**. The best in-distribution model is the worst at novelty and
   loses under shift to one a tenth its size. In-distribution macro-F1 is not merely saturated; it is
   **anti-correlated** with the axes that describe deployment, and can no longer be the headline.
   [[2026-07-31-best-model-is-not-the-best-model]]
8. **Open-set becomes the binding constraint** (Aug 1) — the capacity × augmentation factorial closes,
   and *every* intervention that buys accuracy costs novelty detection (AUROC 0.9068 → 0.8132 across
   the four cells), with none trading the other way. Accuracy is no longer the thing in short supply.
   Meanwhile the noise floor is finally measured, and it is level-dependent.
   [[2026-08-01-capacity-x-augmentation]] · [[2026-08-01-how-noisy-are-our-numbers]]

## Index

### Living

| file | what it is |
|---|---|
| [PLAN.md](PLAN.md) | Status board of runs in flight + the ordered backlog. **Read this before picking up work.** |
| [DIRECTIONS.md](DIRECTIONS.md) | The research strategy, and why the project reframed around open-set reliability |

### Research

| opened | question | status |
|---|---|---|
| [07-16](2026-07-16-why-was-fastai-behind-mini-trainer.md) | Why did fastai+MT-heads score 0.83 when mini_trainer's own loop got 0.896? | RESOLVED — under-annealing; 0.8976, gap closed and overtaken |
| [07-17](2026-07-17-does-longtail-help.md) | Do oversampling / logit adjustment push species macro-F1 past 0.8887? | RESOLVED — oversampling **0.9148, project best**; logit adjustment 0.9031, not recommended (broke genus/family) |
| [07-24](2026-07-24-bigger-everything.md) | Does a bigger backbone beat 0.9148 and make a good distillation teacher? | RESOLVED — ConvNeXtV2-L **0.9316 (+1.68 pp)**, now the best teacher |
| [07-28](2026-07-28-flemming-generalization.md) | Does the 0.9316 model survive an external dataset? | RESOLVED — drops to **0.6950** (~23 pp gap); family robust; motivates the OOD work |
| [07-30](2026-07-30-domain-shift.md) | Is domain augmentation a fix for the 23 pp gap, or a treadmill? | **B1 RESOLVED** — a down-payment: **+4.0 pt shifted for −0.36 in-dist** (11:1), but closes only 17 % of the gap. B2/B3 still open |
| [07-30](2026-07-30-marginal-supervision.md) | Does supervising the marginals *during training* help? | RESOLVED — species **unchanged** (0.9135 → 0.9135), but genus +0.27 / family +0.39 pp. Free coarse accuracy via calibration of the sum |
| [07-30](2026-07-30-does-arcface-compose-with-marginalisation.md) | Do single-head marginalisation and ArcFace × z-score compose? | RESOLVED — **they do not**, but A1 stands: open-set survives (AUROC 0.9068). The interference replicates at 10× scale, so it is a calibration effect, not noise |
| [08-01](2026-08-01-marginalisation-is-not-argmax-consistent.md) | Is marginalisation really "consistent by construction"? | **RESOLVED — no.** `max` and `Σ` do not commute over a partition; a one-line counterexample breaks it. What is true is *probabilistic coherence*. The false wording had reached the paper's method section and 7 other files; all corrected |
| [08-01](2026-08-01-the-scoring-rule-was-the-bug.md) | Is the open-set loss in the embedding or the scoring rule? | **RESOLVED — the rule.** `msp` beats `max-logit` by **+6.1/+7.6 pt** at 198 M (and loses by ~1 pt at 20 M). The capacity penalty drops 8.8 → **1.64 pt**. **Corrects [[2026-07-31-best-model-is-not-the-best-model]]** and cancelled a 36 GPU-hour run |
| [08-01](2026-08-01-how-noisy-are-our-numbers.md) | How large is our run-to-run spread? | **RESOLVED** — species macro-F1 is essentially **deterministic** (0.0000 across a repeat) but family moves **0.24 pp**: noise scales inversely with class count. Downgrades one earlier claim; the shifted metric's noise is still unmeasured |
| [08-01](2026-08-01-capacity-x-augmentation.md) | Do capacity and domain augmentation compose? | **RESOLVED — better than additively on accuracy, jointly harmful on open-set.** The augmentation tax vanishes at scale (−0.36 → 0.00) while its shifted gain grows (+3.99 → +4.85). AUROC falls monotonically across all four cells |
| [07-31](2026-07-31-best-model-is-not-the-best-model.md) | Does the best in-distribution model deploy best? | **RESOLVED — no, the ranking inverts.** A 20 M model with augmentation beats a 198 M one under shift (0.6836 vs 0.6616), and the 198 M model is the *worst* at novelty (0.8298 vs 0.9068) |

### Subprojects

| opened | question | status |
|---|---|---|
| [07-19](2026-07-19-lepi-app.md) | Can the 165 MB model become a fast offline phone app? | PROPOSAL — heads are 51 % of the model; ≤8 MB is the honest target |
| [07-20](2026-07-20-lepi-app-claude.md) | ↳ the detailed plan: size budget, phases A–E, decisions | Decisions RESOLVED (§7); phases A+B done |
| [07-20](2026-07-20-lepi-app-compression.md) | Does the model export, quantize and calibrate for a browser? | RESOLVED — ONNX ok; int8 3.9× for −0.59 pp; marginalization proven; model is *under*confident |
| [07-23](2026-07-23-lepi-app-HANDOFF.md) | Self-contained handoff: state, env, how-to, open problems | snapshot of Jul 23 |
| [07-24](2026-07-24-src-lepinet-baseline-port.md) | How to port the 0.9148 baseline into a clean, fastai-only package? | **RESOLVED** — reproduces it (0.9152 vs 0.9148); that run is the milestone baseline |
| [07-25](2026-07-25-teacher-student-app-bridge.md) | How to make shipping a model (teacher→student→bundle→release) one command? | OPEN — distillation works (T=1); int8 dead in ORT-Web, fp16 ships |
| [07-28](2026-07-28-landscape-and-plan.md) | Global landscape and ordered backlog, as of Jul 28 | **SUPERSEDED** by [PLAN.md](PLAN.md) — kept for its execution log and the Q1–Q6 answers |

### Infrastructure

| opened | question | status |
|---|---|---|
| [07-17](2026-07-17-ucloud-benchmark-oom.md) | Why does the UCloud benchmark keep OOM-ing? | RESOLVED — image-pipeline anon × workers, not the dataframe; 128 workers safe |
| [07-18](2026-07-18-ucloud-throughput.md) | How to make the B200 fast (it is CPU-decode-bound)? | staging = the memory lever; GPU decode built but model-bound for effnetv2s |

### Incidents

| opened | what broke | status |
|---|---|---|
| [07-16](2026-07-16-gpu-hang.md) | The training box hard-hung overnight | RESOLVED as far as the evidence allows — hardware |
| [07-16](2026-07-16-venv-uv-sync-incident.md) | `uv sync` pruned the venv and broke torch | RESOLVED — **never run `uv sync` here**; known-good version set recorded |
| [07-18](2026-07-18-autoregressive-fp16-instability.md) | The autoregressive head trained broken | RESOLVED — fp16 backbone overflow; the bf16 default fixes it |
| [07-30](2026-07-30-ucloud-queue-daemon.md) | A 12-epoch run expired despite `auto_extend`, and a chained eval never launched | RESOLVED — `ucloud q` only advances when a daemon/cron ticks it; none was running. **Check the daemon before suspecting the cluster.** |
