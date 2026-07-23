# Journal

One file per **question**, not per run. A run is a data point; a question is why you spent a
day of GPU on it. `dev/036_ledger.py` already answers *what was tried and what it scored* —
it reads every run's saved `config.yaml` and `metrics.json` and prints the table. Nothing on
disk records *why*, or what you concluded. That is what this directory is for.

```
dev/036_ledger.py   ->  reads every run's config + metrics off disk
RESULTS.md          ->  the numbers   (generated: `dev/036_ledger.py --snapshot`, then commit)
journal/            ->  the reasoning (written by hand, the part that can be lost)
```

`RESULTS.md` is generated but **tracked on purpose**: the ledger's source lives under `data/`,
which is a symlink to machine-local storage and gitignored, so a clone anywhere else sees no
runs at all. The snapshot is the only copy of those numbers that leaves the training box —
and `git log -p RESULTS.md` is the project's result history. Regenerate and commit it whenever
a run finishes.

## Conventions

**Name by question, dated:** `2026-07-why-was-fastai-behind-mini-trainer.md`. Not
`run-20260714.md` — a run number is not a thought.

**Write the hypothesis before the results land.** A file opened while the GPU is busy says
what you expect and why. When the number arrives you are testing a prediction instead of
rationalising an outcome. Open with status `OPEN`, close it with `RESOLVED` + the answer.

**Record negative results.** "The class-distribution regulariser is a wash" cost a full
training run to learn and is the first thing forgotten — six months on, nothing stops you
paying for it twice. A wash is a finding. So is a crash.

**Cite runs by id** (`20260714-072404`), never by adjective ("the good run"). Ids resolve in
the ledger and on disk; adjectives resolve nowhere.

**Keep the detail; don't transcribe.** The per-epoch numbers are in the CSV, the metrics are
in `metrics.json`. Link them. Write here only what the files cannot say: the reasoning, the
dead ends, the thing you would tell someone to save them a week.

## Index

| file | question | status |
|---|---|---|
| [2026-07-why-was-fastai-behind-mini-trainer.md](2026-07-why-was-fastai-behind-mini-trainer.md) | Why did fastai+MT-heads score 0.83 when mini_trainer's own loop got 0.896? | RESOLVED — 0.8976, gap closed |
| [2026-07-does-longtail-help.md](2026-07-does-longtail-help.md) | Do oversampling / logit adjustment push species macro-F1 past 0.8887? | RESOLVED — oversampling **new project best 0.9148**; logit adjustment 0.9031, not recommended (broke genus/family) |
| [2026-07-gpu-hang.md](2026-07-gpu-hang.md) | What killed the training box overnight on 07-16? | RESOLVED (as far as evidence allows) — hardware |
| [2026-07-venv-uv-sync-incident.md](2026-07-venv-uv-sync-incident.md) | What broke the venv, and what is the known-good version set? | RESOLVED — never run `uv sync` here |
| [2026-07-ucloud-benchmark-oom.md](2026-07-ucloud-benchmark-oom.md) | Why does the UCloud MT-head benchmark keep OOM-ing? | RESOLVED — image-pipeline anon ×workers; guard was miscounting reclaimable page cache; 128 workers safe |
| [2026-07-ucloud-throughput.md](2026-07-ucloud-throughput.md) | How to make the B200 fast (it's CPU-decode-bound)? | staging=memory lever; GPU decode fixed (overlap+gc+bigbatch) but model-bound for effnetv2s -> memory win not speed; co-locate copies to use the B200 |
| [2026-07-autoregressive-fp16-instability.md](2026-07-autoregressive-fp16-instability.md) | Why did the autoregressive head train broken? | RESOLVED — fp16 backbone overflow; the existing bf16 DEFAULT fixes it (bench config wrongly forced fp16) |
| [2026-07-lepi-app.md](2026-07-lepi-app.md) | Can the 165 MB model become a fast offline phone app? | PROPOSAL + review — heads are 51% of the model; ≤8 MB is the honest target, <1 MB is not |
| [2026-07-lepi-app-HANDOFF.md](2026-07-lepi-app-HANDOFF.md) | **START HERE to resume** — self-contained handoff: state, env, how-to, open problems | living |
| [2026-07-lepi-app-claude.md](2026-07-lepi-app-claude.md) | ↳ the detailed plan: size budget, phases A–E, decisions | Decisions RESOLVED (§7); phases A+B done, C+D open |
| [2026-07-lepi-app-compression.md](2026-07-lepi-app-compression.md) | Does the model export, quantize and calibrate for a browser? | RESOLVED — ONNX ok; int8 **3.9× for −0.59 pp**; marginalization proven; resize is a non-issue; model is *under*confident |
