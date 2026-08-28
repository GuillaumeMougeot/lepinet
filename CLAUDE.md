# CLAUDE.md — operating manual for an agent working on lepinet

You are reading this because it is loaded automatically at the start of every session in this
repository. It is written for you, not for a human. Its job is to make a cold start behave like a
warm one: what this project is, how it thinks, what is true right now, what must never be done, and
what you owe the documentation before you finish.

**This file is a router, not an encyclopedia.** It holds only what is (a) not derivable from the
code, and (b) needed to avoid a mistake. Depth lives in the files it points at. Do not copy content
here from those files; a duplicated fact is a fact that will go stale in one of its two homes.

---

## 1. What the project is

Hierarchical fine-grained image classification: from one photo, predict a label at every level of a
taxonomy at once. Reference dataset is Lepidoptera — 12,041 species / 4,333 genera / 102 families,
~3 M images, heavy long tail. The package is generic over the number and names of levels.

The headline metric is **species macro-F1** on the held-out fold (`set == '0'`) over **all** species.
Macro, because the tail is the point.

**The project has pivoted once, and this is the single most important thing to internalise.** It
began as a comparison of hierarchical prediction *heads*. That comparison is a **null result** —
architecture does not matter here. Meanwhile a model at 0.93 in-distribution scores ~0.70 on external
data, and real deployments constantly meet species outside the label set. So the actual subject is
now **reliable prediction that knows what it doesn't know**: open-set detection in image space, rank
abstention in hierarchy space, both under domain shift. If you find yourself optimising
in-distribution accuracy, you are working on the solved part.

## 2. Read this before acting

Always, in this order, before the first substantive action:

1. **`journal/PLAN.md`** — the status board. What is running, what it is blocked on, what is
   deliberately not being done. This is the only file guaranteed to be about *today*.
2. **`START-HERE.md` section 2** — the findings, one line each. Section 2a is the science, 2b is the
   engineering. This stops you re-deriving or contradicting a paid-for result.

Then, by task:

| task | read |
|---|---|
| explain the method to a non-specialist | `docs/concepts.md` — the vocabulary, ground up |
| change training behaviour | `docs/design-decisions.md` — every default has a measured reason |
| change the package | `docs/developer-guide.md`, then `src/lepinet/README.md` |
| run an experiment | `dev/README.md`, and the journal entry for the nearest prior question |
| write about results | `paper/DRAFT.md` for claims, `RESULTS.md` for numbers |
| understand a decision | `journal/README.md` — index by kind, chronological by open date |
| cluster work | `ucloud/README.md` |

**Do not reconstruct project history from the conversation.** The repo self-documents precisely so
that chat logs are not load-bearing. `dev/036_ledger.py` prints every run's config delta and score;
`journal/` says why. Prefer them over memory, including your own.

## 3. Invariants — violating these breaks something real

- **Never run `uv sync`** on the training venv. It prunes and breaks torch/torchvision. The venv is
  hand-managed. `journal/2026-07-16-venv-uv-sync-incident.md`.
- **bf16 for any margin head; fp16 elsewhere is what the project actually does.** The compressed
  form of this rule used to read "bf16, never fp16" and that is *false*: 63 of 115 configs set
  `precision: fp16`, including the headline baseline. The true rule is narrower — the cosine head
  overflows in fp16 unless forced to fp32 in an adapter (which `lepinet` does), and **ArcFace with a
  positive margin overflows anyway**, so every margin run is bf16 and `config.py` warns on the
  combination. A head that "trains broken" is still this. `docs/design-decisions.md` has the
  accurate version; corrected 2026-08-28 after the paper was found asserting the wrong one.
- **Never filter the test fold.** `--min-img-per-spc` on evaluation silently drops the tail out of a
  *macro* average and inflates the number by ~3 points. If a score jumps, audit the eval set before
  celebrating. `journal/2026-07-24-src-lepinet-baseline-port.md`.
- **`ucloud q` only advances when a daemon or cron ticks it.** Without one, `auto_extend` is inert
  and `--after` jobs never launch, while `ucloud q ls` still looks healthy. Check
  `ps aux | grep "ucloud q daemon"` before diagnosing any lost or stuck job.
  `journal/2026-07-30-ucloud-queue-daemon.md`.
- **`data/` is a symlink to machine-local storage and is gitignored.** A clone elsewhere has no runs.
  `RESULTS.md` is generated but **tracked**, because it is the only copy of those numbers that leaves
  the training box.
- **Read UCloud job logs, not job status.** A job reports SUCCESS while the script inside exits 1.
- **Change one factor per run.** The largest jump in the project's history (+4.7 pt) bundled three
  changes and is permanently unattributable. That mistake is preserved in the ladder as a warning.
- **Work on `main`** unless the owner asks otherwise, and never commit or push unless asked.
- **No emoji** in structural docs. Enforced by `dev/060_doc_health.py`.

## 4. How this project thinks

These are not style preferences. Each was learned expensively and each has changed a conclusion.

**Write the hypothesis before the result lands.** Open a journal entry while the GPU is busy, state
what you expect and why. When the number arrives you are scoring a prediction instead of
rationalising an outcome. Several entries score their own predictions explicitly; keep doing that.

**A negative result is a result.** Washes, crashes and abandoned directions are kept on purpose —
they cost GPU hours and are the first thing forgotten. `docs/design-decisions.md` section 3 is a list
of things not to buy twice. Add to it.

**Before hunting a bug, check that both numbers mean the same thing.** A "0.92 val vs 0.83 test" bug
was chased at length and did not exist: one metric averaged three taxonomic levels, the other was
species-only. This class of error has recurred more than any other in this project.

**Suspect the measurement before the model.** Three separate times the finding was in the harness,
not the network: the eval-set filter, fastai's hardcoded `num_workers` (a 900x slowdown misdiagnosed
as hardware), and `isinstance(True, int)` feeding a `sqrt`. When a number is surprising, the prior
should be "the measurement is wrong."

**A cheap proxy for an expensive experiment is usually invalid, and you must argue that it is not.**
Two attempts to tune the ArcFace margin cheaply failed for *principled* reasons — a range test
mechanically raises the loss when you add a margin, and a short-horizon grid anti-correlates with the
converged outcome. Both were abandoned with the reasoning written down, which is the correct outcome.

**Prefer mechanisms that do not need one constant to be right at every level at once.** This single
idea explains two separate results: why oversampling beat logit adjustment, and why marginalisation
beat per-level heads. It is the closest thing this project has to a design principle.

**A claim phrased as a *definition* still needs a test.** Two claims survived unchallenged for weeks
because they did not look like results: `-max_logit` (it looked like the definition of the score, and
was wrong across a scale change) and "marginalisation is consistent by construction" (it looked like
a theorem, and is false — `max` and `Σ` do not commute over a partition). Both were caught by writing
an assertion that took the words literally. If a property cannot be falsified by a test, it is not a
property, it is a hope.

**Prove what you can without a GPU first.** The clean package was made to load the old checkpoint
bit-exactly, so export, prediction and evaluation were all validated before any retraining.

**Extension points are registries.** `HEAD_REGISTRY`, `DOMAIN_AUG_REGISTRY`. A `dev/` experiment adds
an entry rather than editing the package, so the default recipe cannot drift under a published
number. Follow this pattern for anything new that is opt-in.

## 5. Working with the owner

Stated preferences, from repeated instruction. Treat as standing orders.

- **Journal everything.** Said more often than anything else. A session that produces results and no
  journal entry has not finished. `DEVELOPER.md` is the owner's own statement of this.
- **Plan and discuss before coding** on anything sizeable. The owner asks to "step back and redraw
  the landscape" and expects reasoning, alternatives and a recommendation — not a survey.
- **Report reasoning, not just outcomes.** What was surprising, what it means, what it changes.
- **Compute is not the constraint; experimental hygiene is.** Parallel jobs are fine. Do not fear a
  full GPU. Queue evals behind training with `--after`.
- **Limit background watchers.** They consume resources. Prefer telling the owner when to check back.
- **The paper matters.** Findings should be journalled in a form that can become `paper/DRAFT.md`.
- **Surface your own mistakes plainly.** Several corrections in this project came from the owner
  spotting an implausible number. Say "this was a genuine mistake" and what caused it.

## 6. The documentation contract

Six layers, each with one job and one owner. **No fact lives in two layers.** If content must appear
twice, one copy is a one-line summary that links to the other.

| layer | question | rule |
|---|---|---|
| `README.md` | what is the problem and the method | stays short |
| `START-HERE.md` | where is everything, what is established | one line per finding + link; never the argument |
| `docs/user-guide.md` | how do I run it | how-to only |
| `docs/developer-guide.md` | how do I change it | architecture and seams |
| `docs/design-decisions.md` | why is the recipe this way | every default, what it was worth, what failed |
| `journal/` | why, as it happened | one file per question; reasoning and dead ends |
| `RESULTS.md` | what scored what | generated by `dev/036_ledger.py --snapshot`; never hand-edited |
| `paper/DRAFT.md` | what do we claim | only claims that generalise beyond this dataset |

**`journal/` has two tiers.** Living documents are `UPPERCASE.md` with no date and are rewritten in
place: `PLAN.md` (status board), `DIRECTIONS.md` (strategy), `README.md` (the master index).
Everything else is `YYYY-MM-DD-question.md`, dated by when the question was **opened**, so `ls` reads
in the order things were asked, and frozen once `RESOLVED`. Every archival entry declares
`**Kind:**` — `research`, `subproject`, `infrastructure`, or `incident` — and `**Status:**`.

### What to update, and when

Maintenance is **event-driven, not calendar-driven**. Do these as part of the work, not as a chore
afterwards; a documentation pass scheduled for "later" is one that happens after the reasoning has
been forgotten.

| when this happens | do this, in the same session |
|---|---|
| a run finishes | move its row in `journal/PLAN.md`; regenerate and commit `RESULTS.md` |
| a question is answered | flip its entry to `RESOLVED` with the answer in the status line; update `journal/README.md`'s index row |
| a new question is opened | new dated entry with `**Kind:**`, `**Status:** OPEN`, and the hypothesis *before* results; link it from `journal/README.md` |
| a finding generalises | add a one-liner to `START-HERE.md` section 2a/2b, and a section to `paper/DRAFT.md` |
| a default changes | `docs/design-decisions.md`, with what it was worth |
| a trap costs you more than an hour | `docs/design-decisions.md` section 4, or an `incident` entry if it lost a run |
| the plan changes | `journal/PLAN.md`, including its `**Last updated:**` date |
| anything is renamed or moved | run `python dev/060_doc_health.py` before committing |

### The check that makes this survive

`python dev/060_doc_health.py` (also `tests/test_doc_health.py`, so CI enforces it) verifies what a
machine can: journal naming, `Kind`/`Status` headers, index completeness, every relative link and
wikilink resolving, no emoji in structural docs, and — the one that catches real drift — that
`PLAN.md`'s `Last updated` is not older than the newest journal entry. Run it before committing
documentation.

It cannot check whether a claim is still *true*. That is what section 7 is for.

## 7. Keeping this file true

This file rots the same way everything else does. Two mechanisms, and the first is the important one.

**Continuous, and preferred:** when you learn something that would have changed how you started the
session, add it here *then* — a new invariant, a corrected fact, a preference the owner stated. That
is the whole maintenance model in normal operation, and it costs almost nothing because you are
already holding the context that makes the edit correct.

**Periodic, and rare:** a full re-read is worth doing when the project changes shape — a research
pivot, a major refactor, a new subsystem — not on a schedule. The owner can trigger one by asking to
"refresh CLAUDE.md"; the procedure is: read `journal/PLAN.md` and every journal entry newer than this
file's last revision, reconcile sections 1, 3, 4 and 5 against them, delete anything superseded, and
run the doc-health check. Deletion is the part that gets skipped and matters most: an operating
manual that only accretes becomes a document nobody reads.

**What does not belong here:** current results (they live in `RESULTS.md`), the current plan (it
lives in `PLAN.md`), or anything a reader could get by opening the code. If you are tempted to add a
status update to this file, you want `PLAN.md` instead.

---

**Last revised:** 2026-07-30 · **Reconciled against:** `journal/PLAN.md` and all journal entries
through 2026-07-30.
