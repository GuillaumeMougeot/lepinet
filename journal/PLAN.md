# PLAN — where we are, and what runs next

**Kind:** living · **Last updated:** 2026-08-24 · **Supersedes:** [[2026-07-28-landscape-and-plan]]

The one file in `journal/` meant to be true *today*. Everything else is a record of a moment.

**Structure:** experiment groups first, because that is what people look for. Operating rules for the
owner's absence are at the bottom. Completed groups keep one line and a link — the reasoning lives in
the journal entry, not here. *(Restructured 2026-08-07: the groups had ended up below 220 lines of
operational text and were effectively unfindable.)*

---

## 1. Group index — the whole experiment programme at a glance

| group | question | state |
|---|---|---|
| **A** | consolidate the architecture (single head × ArcFace × distillation) | **closed** |
| **B** | robustness: augmentation, self-training, capacity | **closed** — B3 is the project's biggest lever |
| **C** | open-set: abstention, stratified novelty, AUROC | **closed** — C3b confirms monotonicity is not a rarity artefact |
| **D** | product: bundle, calibration, small student | **closed** |
| **E** | is open-set the binding constraint? | **closed — no**, the premise was a scoring-rule artefact |
| **F** | the assembled recipe | **closed** — F2 composes |
| **G** | the 198 M confirmation | **closed** — the staged/end-to-end verdict is capacity-dependent; B10 confirms the comparison is fair |
| **H** | scaling the head to ~1 M species | **closed** — all five training routes measured and dead; shard the matrix, use centroids at inference |
| **L** | imbalanced learning | **closed** — cRT is the answer |
| **P** | pretrained encoders (BioCLIP-2) as frozen trunks | **deferred** (owner) |
| **T** | what target labels would have bought | **closed** |

## 2. State (2026-08-26)

**Storage fixed by UCloud; everything is running again.** Parallel read throughput is back to
**992 files/s** and G3b trains at **509 img/s**, better than before the outage.
[[2026-08-24-work-storage-degraded]] is RESOLVED.

**Gate any future restart on `ucloud/lepinet-ioprobe-par.toml`** (64 threads, ~2 min), not on the
single-threaded `ioprobe3`. The single-threaded probe returned NO-GO at 13.2 files/s while the real
parallel throughput was 992 -- its threshold was borrowed from a parallel figure and is not
comparable. `ioprobe3` answers only "is the storage broken at all".

**Running, fully serialised** (one image-heavy job at a time, each `--after` the previous):

| run | predicted | falsified if |
|---|---|---|
| **G3b** | held-out 0.745-0.760 | above 0.7652 — G3's drop would be noise, and finding 7 loses its 198 M clause |
| **P1a** | in-dist 0.86-0.91 | below 0.80 |
| **P1b** | probe 0.72-0.78 vs T2b's 0.7515 | below 0.70 — the recipe would depend on our representation after all |

**Landed 24 Aug:** H4 falsified (-4.25 pt on a matched fold), B10 a tie. See
[[2026-08-24-three-week-report]].

## 3. Ordered backlog — take the top unblocked item

Costs from the measured formula: `images_per_epoch × epochs / 1100 img/s`, 5.04 M images/epoch.
A 5-epoch 20 M run is ≈6.4 h; a 2-epoch frozen-trunk stage is ≈30 min at 20 M, ≈1 h at 198 M.
**Throughput is ≈1100 img/s at 20 M but ≈480 img/s at 198 M** (measured on B10: 6 epochs x 5.18 M
images in 18 h). Using the 20 M figure for a 198 M run underestimates by 2.3x — it turned a
costed "7-10 h" into 18 h on 2026-08-09.

| # | work | GPU | why, and the gate |
|---|---|---|---|
| ~~1~~ | ~~read H4 and B10~~ | — | **DONE 2026-08-24.** H4 falsified, B10 a tie. |
| 2 | **decide the staged-vs-end-to-end question** | none | see section 7. It has flipped twice; recommendation is to freeze it |
| 3 | **seed-repeat of G3** | ~1 h at 198 M | the held-out drop at 198 M is n = 1, and two conclusions have already moved on single measurements |
| 4 | **P1 — BioCLIP-2 as a frozen trunk** | build + ~1 h | owner-deferred, reviewer-inevitable. T2b is what makes it plausible. See section 5 |
| 5 | **paper: fold in the four figures, fact-check `[VERIFY]` citations** | none | citations were written from memory and are not reliable |
| 6 | **C3b at 198 M** | ~6 h | the novelty-is-not-rarity result is currently 20 M only |

**Completed 2-10 August** — R2/R3/R4/R5 (the gate), B9, C3b + C3ref, G3, H4, B10, the figures.
Scores and predictions in [[2026-08-24-three-week-report]].

**Do not do** — each closed for a reason, not for lack of time:

- **LDAM, background suppression** — hand-authored *representational* fixes. T2b showed adaptation
  subsumes that family (`domain_aug` fell +4.75 → +0.57 once the classifier was adapted).
- **Re-tuning the ArcFace margin** — two cheap proxies failed for principled reasons, and the margin's
  measured benefit is now ~0.8 pt.
- **The autoregressive head** — lost by 20 pt.
- **More in-distribution accuracy** — saturated; the shifted axis is where the headroom is.
- **Uniform sampled softmax at scale** — H2/H3 measured no plateau, and taxonomy-aware negatives
  recovered only 26 % of the loss.

## 4. Group H — scaling the head to ~1 M species (open)

A 1280 × 1M prototype matrix is 5.1 GB plus 10.2 GB of optimiser state. Options and costings in
[[2026-08-05-scaling-the-head]].

| | status |
|---|---|
| **inference** | **solved** — class centroids replace the trained matrix for **0.29 pt**; mean beats k-means and medoid |
| low-rank factorisation | **dead** — rank 1035/1280; the margin *spends* dimensions |
| fixed / taxonomy codes | weakened by the same spectrum |
| uniform sampled softmax | **dead** — smooth, no plateau; 1024/1M is 0.1 % coverage |
| hard-negative sampling | weakened — taxonomy-aware negatives recovered only 26 % (H3) |
| **training** | **CLOSED — H4 falsified at 0.8685 (−4.63 pt)** — proxy-free: EMA centroids in a buffer, no matrix, no optimiser state. Removes 10.24 GB of the 15.36 GB. [[2026-08-09-can-centroids-be-trained-against]] |
| *constraint found* | a proxy-free head has **no trainable parameter on the species path** unless the bottleneck is kept, so it cannot serve a frozen-trunk stage (cRT, adaptation) without one |

## 5. Group P — pretrained encoders as trunks (deferred by the owner, 2026-08-06)

T2b showed classifier-only adaptation works from a trunk that never saw the target domain in any
form. If adaptation needs nothing from the representation, **it does not need *ours*.**

| id | work | why |
|---|---|---|
| **P1** | BioCLIP-2 or another strong biological encoder as a frozen trunk, classifier fitted then adapted | Reviewers will ask. It tests whether the pipeline reduces to "take the best encoder, fit a cheap classifier, adapt it" — a stronger claim than anything about our backbone. |
| P2 | the same with centroids instead of a trained classifier | No trained head at all: encoder + centroids + 2 epochs. |

Prerequisites are small — an encoder that runs here, and `train(init_from=, freeze_body=)`, which
exists.

## 6. Closed groups — one line each

Reasoning lives in the linked entries; numbers in [`../RESULTS.md`](../RESULTS.md).

| group | outcome |
|---|---|
| **A** | Single head + ArcFace do **not** compose (−0.59 species, −1.0 coarse); A2 is the best in-distribution model at 0.9216; distillation's credit halved once A6 supplied the missing control. [[2026-07-30-does-arcface-compose-with-marginalisation]] |
| **B** | Augmentation buys +4 pt under shift but closes only 17 % of the gap; **self-training is the largest lever** (+7.94 at the right dose) and costs nothing in-distribution. [[2026-08-03-b3-self-training]], [[2026-08-04-replication-sweep]] |
| **C** | Rank abstention needs *conditional* thresholds (genus 0.487 on the hard subset vs 0.970 overall); novelty is monotone in taxonomic distance **for both heads**. Two claims retracted — see below. |
| **D** | `lepinet bundle` emits names + calibration + thresholds; fastvit_sa12 is the shipped student at 0.8967. |
| **E** | Premise dissolved: "every intervention that buys accuracy costs open-set" was an artefact of reading every model with `max-logit`. [[2026-08-01-the-scoring-rule-was-the-bug]] |
| **F** | The staged recipe **composes**: one clean representation + 2 × 2 frozen-trunk epochs gives in-dist 0.9081 / probe 0.7541. [[2026-08-06-f2-capstone]] |
| **L** | Tail-reweighting trades robustness for accuracy monotonically — **but cRT fixes it**: rebalance the classifier, not the data. [[2026-08-01-imbalance-methods-bench]] |
| **T** | Self-training at its best dose beats **12,230 real labels**; adaptation is 83 % a classifier problem. [[2026-08-05-label-budget]], [[2026-08-06-adaptation-is-mostly-a-classifier-problem]] |

### Retractions and open incidents

- **The 31-pt open-set claim is retracted.** It compared ArcFace's best rule against the plain head's
  worst; best-vs-best is 0.9068 vs 0.8990, and the plain head is 1 pt better on accuracy.
  [[2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison]]
- **"Consistent by construction" is false** — `max` and `Σ` do not commute over a partition.
  [[2026-08-01-marginalisation-is-not-argmax-consistent]]
- **The cosine head's rows are not unit-norm** (mean 1.08 / 1.77). Downgraded to a documentation
  problem: zero ties, top-1 saturates on 0.37 % of images, **no accuracy number is affected**.
  Mechanism unknown; **do not change the head**. [[2026-08-06-the-cosine-head-is-not-unit-norm]]

---

## 7. Operating rules while the owner is away (2026-08-02 → ~2026-08-23)

**The queue survives unattended.** A crontab entry ticks `ucloud q` every 5 minutes. Before that, the
tick ran only inside a shell session, so `auto_extend` was inert and `--after` jobs never launched —
which already cost a 12-epoch run ([[2026-07-30-ucloud-queue-daemon]]). **If jobs look stuck, check
the cron entry first.**

**An agent session is not a monitor.** Work advances only when a session is invoked, so the queue is
loaded with runs that self-evaluate and the backlog above is ordered for pickup without asking.

### Queue discipline, each rule learned by breaking it

- **Run ONE image-heavy job at a time.** The pipeline is IO/CPU-decode bound. Three concurrent jobs
  with `num_workers` 128 + 256 + 32 collapsed `/work` read throughput on 2026-08-24 and cost two
  runs. [[2026-08-24-work-storage-degraded]]
- **Read the `[mem]` startup lines.** They printed `WARNING: 256 workers x ~1.2 GB anon is close to
  the 288 GB limit` before the run that then thrashed page cache for three hours. A startup warning
  is a result, not noise.
- **A frozen progress bar is usually starvation, not a hang.** Check GPU memory: 2-4 GB and
  fluctuating means the job is waiting on data. `ucloud/lepinet-ioprobe3.toml` measures cold random
  read throughput in about a minute and is the fastest way to tell.

- **Keep an independent job queued alongside any chain.** A blocked chain should cost one line of
  work, not all of it. (2026-08-03: five idle hours.)
- **Run a whole sweep at once**, not one arm at a time. Four arms is four jobs and one wait.
- **Put the untested step at the head of its own chain.** A script that has never run is the likeliest
  thing in the queue to fail.
- **After relaunching a failed job, remove and re-submit its dependents.** `BLOCKED` is sticky against
  the job *name*, so a fresh run under the same name leaves them frozen for ever.
- **Check `ucloud jobs list` before trusting `ucloud q logs`.** Logs are keyed by name, so a
  resubmitted job serves its predecessor's output — a re-run can appear to confirm the number it was
  launched to correct.
- **Clone a working job TOML; never hand-write the `run =` line.** C3b died instantly on
  `python -m lepinet train` -- the entrypoint is `lepinet train -c`. The same hand-written file also
  had `lepinet test --out` (it is `--out-dir`) and omitted `--min-img-per-spc 0`, which would have
  filtered the test fold and inflated a macro average by ~3 pt. Three errors in one file, all from
  not copying a file that already ran.
- **A `dev/`-registered head is invisible unless `dev/050` is imported.** Three scripts have died on
  this; the silent version (registering the head but not its callback) is worse than the crash.

### Scale discipline (owner, 2026-08-06)

**Validate at 20 M; promote to 198 M once, at the end.** Six 198 M runs in five days (42 GPU-hours)
while the "model to ship" changed three times — confirmation dressed as exploration. A new mechanism
is tested at 20 M however promising; 198 M is one run after the recipe stops moving. The caveat is
real — three interventions behaved *differently* at 198 M — but that is one question to ask
deliberately at the end, not a licence to run everything twice.

### The paper

Owner's direction: **let the story unroll**; the 23rd is not a deadline for the headline. Related work
is drafted with every citation marked `[VERIFY]` (written from memory — the owner fact-checks).
Figures were deferred until the storyline settled; it now has, so they are backlog item 4. **Keep §4
in step with the journal** — it drifted twice by treating the journal as primary.

### The 23 August report

Read every journal entry after 2026-08-02 plus this file and `RESULTS.md`. Cover: what ran and what
it scored; **every committed prediction and whether it held**; **every claim corrected or retracted**;
what the project now asserts that it did not on 2 August; what is still running; and the backlog with
reasons. **Include the dead ends** — four of the most useful results this fortnight were corrections.
Write to `journal/2026-08-23-three-week-report.md` and give a condensed version in chat.
