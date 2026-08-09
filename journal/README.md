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
| [08-05](2026-08-05-scaling-the-head.md) | How to reach 1 M species without a 5 GB prototype matrix? | OPEN — options costed (low-rank, hierarchical softmax, sampled softmax, **taxonomy-structured fixed codes**, **retrieval**). Nothing run yet; the two cheapest tests are launched |
| [08-09](2026-08-09-can-centroids-be-trained-against.md) | Can the prototype matrix be replaced by EMA centroids during *training*, not just inference? | **OPEN** — removes 10.24 GB of optimiser state at 1 M classes. Predicted 0.900-0.912 vs 0.9148; falsified below 0.885 |
| [08-08](2026-08-08-is-novelty-monotone-or-just-rare.md) | Is novelty detection monotone in taxonomic distance, or was C3 measuring rarity? | **RESOLVED — monotone, not rarity.** With 231 **common** taxa withheld: near **0.8717** / mid **0.9463** / far **0.9726**, ordering unchanged and every stratum slightly *better* |
| [08-08](2026-08-08-self-training-does-not-iterate.md) | Does a second self-training round help? | **RESOLVED — yes, with no gate and balanced replication.** The quantile gate costs 3.80 pt; the per-species gate was doing **class balancing**, not filtering. Full coverage + balance: **probe 0.7692 / held-out 0.7781**, best on both shifted axes |
| [08-06](2026-08-06-f2-capstone.md) | Do the classifier-stage findings compose into one recipe? | **RESOLVED — yes.** One clean representation + 2×2 epochs of frozen-trunk classifier work gives in-dist **0.9081** / probe **0.7541**: better in-distribution than any end-to-end 20 M model, 1.65 pt behind on probe, for minutes of compute per deployment |
| [08-06](2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison.md) | Does ArcFace × z-score really take open-set from chance to usable? | **RESOLVED — no, and the claim is retracted.** 0.601 → 0.9115 compared ArcFace's best rule against the plain head's worst. Best-vs-best is **0.9068 vs 0.8990**, and the plain head is 1 pt better on accuracy |
| [08-06](2026-08-06-adaptation-is-mostly-a-classifier-problem.md) | Does domain adaptation need the representation, or just the classifier? | **RESOLVED — mostly the classifier.** 2 epochs on a frozen trunk captures **83 %** of self-training's probe gain and 89 % of its transfer. Corrects the mechanism claimed in [[2026-08-03-b3-self-training]] |
| [08-05](2026-08-05-label-budget.md) | What would real target labels have bought? | **RESOLVED** — real labels beat 98 %-accurate pseudo-labels by **+2.14 pt** at matched size, but **self-training at its own best dose beats 12,230 real labels** (0.7706 vs 0.7568). The held-out column is **not interpretable** for these arms and says so |
| [08-04](2026-08-04-replication-sweep.md) | How much pseudo-labelled data, and does replication help? | **RESOLVED — replication was never needed.** 0.39 % buys 97 % of the gain; the optimum is **2 %** (probe **0.7706**, project best) and transfer to unseen species falls monotonically with replication, 121 % → 39 %. **Falsifies the design argument in [[2026-08-03-b3-self-training]]** |
| [08-03](2026-08-03-b3-self-training.md) | Does self-training on unlabelled trap images help? | **RESOLVED — the largest robustness lever found.** probe **+4.58 pt** for **+0.04** in-distribution, 11.2x its floor; **56 % transfers to species the adaptation never saw**. A 20 M model with it beats a 198 M model without |
| [08-03](2026-08-03-macro-f1-does-not-decompose.md) | Why does F1 tie B4 on the full trap set but beat it by 2 pt on a subset of it? | **RESOLVED** — macro-F1 does not decompose over subsets; the two columns weight different species sets. Both real (5.0x and 8.8x their measured floors). Narrows the F1 claim and tightens B3's falsification line |
| [08-02](2026-08-02-f1-flagship.md) | Does composing every win at 198 M beat B4? | **RESOLVED -- prediction falsified.** Species and shift are identical to B4; marginal supervision's +1.79 pt shifted gain at 20 M is +0.02 here. But its coarse gain **grows** (+0.40 genus / +0.74 family). Regularisation benefits do not transfer upward; direct ones do |
| [08-02](2026-08-02-the-shifted-benchmark-is-also-the-adaptation-set.md) | Can B3 be compared against the existing shifted numbers? | **RESOLVED — no.** The shifted benchmark and B3's adaptation data are the same 47,905 trap images. Grouped (trap, night) splits built with leakage assertions; baselines re-scored on the 15,200-image `probe` set |
| [08-01](2026-08-01-imbalance-methods-bench.md) | Do imbalance methods (balanced softmax etc.) beat √-oversampling? | OPEN — 2×2 running. **Balanced softmax *is* logit adjustment at τ=1**, which we already rejected — but the reason it lost (one shared τ across three levels) is structurally impossible on the single-head architecture |
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
| [08-06](2026-08-06-the-cosine-head-is-not-unit-norm.md) | The cosine head's prototype rows are **not** unit-norm, though the design says they are | **OPEN** — confirmed on two checkpoints (mean 1.081 and 1.767); mechanism unknown. Accuracy numbers unaffected; the z-score calibration argument and the ArcFace round-trip may be |
| [07-16](2026-07-16-gpu-hang.md) | The training box hard-hung overnight | RESOLVED as far as the evidence allows — hardware |
| [07-16](2026-07-16-venv-uv-sync-incident.md) | `uv sync` pruned the venv and broke torch | RESOLVED — **never run `uv sync` here**; known-good version set recorded |
| [07-18](2026-07-18-autoregressive-fp16-instability.md) | The autoregressive head trained broken | RESOLVED — fp16 backbone overflow; the bf16 default fixes it |
| [07-30](2026-07-30-ucloud-queue-daemon.md) | A 12-epoch run expired despite `auto_extend`, and a chained eval never launched | RESOLVED — `ucloud q` only advances when a daemon/cron ticks it; none was running. **Check the daemon before suspecting the cluster.** |
