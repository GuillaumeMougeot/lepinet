# PLAN — where we are, and what runs next

**Kind:** living · **Last updated:** 2026-08-10 · **Supersedes:** [[2026-07-28-landscape-and-plan]]

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
| **G** | the 198 M confirmation | **G3 done — the staged/end-to-end verdict is capacity-dependent**; B10 running |
| **H** | scaling the head to ~1 M species | **running** — H4 tests the last training candidate |
| **L** | imbalanced learning | **closed** — cRT is the answer |
| **P** | pretrained encoders (BioCLIP-2) as frozen trunks | **deferred** (owner) |
| **T** | what target labels would have bought | **closed** |

## 2. Running now

| job | what | predicted |
|---|---|---|
| ~~G1 → G2~~ | **DONE. in-dist 0.9150 / probe 0.7648** — best in-distribution model in the project. Probe missed its 0.780–0.800 range, but **the staged-vs-end-to-end trade replicates across 10× scale**: +0.78/−1.65 at 20 M, +0.90/−1.50 at 198 M. |
| ~~F3~~ | **DONE. 0.9061 / probe 0.7479 / held-out 0.7703.** Both predicted ranges hit, but the probe falsification line (0.7500) tripped by 0.21. Near-wash that splits: sequencing +0.62 probe, joint **+1.09 held-out** at half the cost. **Joint is the better default.** |
| ~~R2~~ | **DONE — badly falsified. probe 0.7161, −3.80 pt.** Labels more accurate (99.84 %) but 156 species vs 346: a better labeller separates its confidences and the quantile gate keeps a narrower set. |
| ~~R3~~ | **DONE — predicted range hit. probe 0.7585 / held-out 0.7682.** Per-species gate instead of quantile, *nothing else changed*: −24.6 pt label accuracy, +740 species, **+4.24 pt**. Iteration now marginally positive (+0.44 over round 1). [[2026-08-08-self-training-does-not-iterate]] |
| **C3b** | **is novelty monotone, or was C3 measuring rarity?** Identical C3 recipe, retrained with **common** taxa held out at three ranks (2 families, 40 genera, 120 species; 2.62 % of rows). Predicted monotonicity survives with `near` 0.78-0.85 -- harder than C3's 0.853, because a common held-out species has hundreds of good photographs. [[2026-08-08-is-novelty-monotone-or-just-rare]] |
| ~~R4~~ | **DONE — predicted range hit, and it split the benchmarks. probe 0.7674 (+0.89 over R3), held-out 0.7458 (−2.24).** R3's k=35 cap was doing **class balancing**, not confidence filtering: R3 is R4 with the head of the trap distribution truncated. [[2026-08-08-self-training-does-not-iterate]] |
| ~~R5~~ | **DONE — both predictions inside range. probe 0.7692 / held-out 0.7781.** Best model in the project on both shifted axes. Coverage and balance are separable and compose. **The confidence gate is gone.** [[2026-08-08-self-training-does-not-iterate]] |
| ~~C3b~~ | **DONE — monotonicity confirmed, prediction half right. near 0.8717 / mid 0.9463 / far 0.9726.** Ordering held; magnitudes went *up*, not down as predicted. Novelty-has-degrees now rests on two populations chosen by opposite criteria. [[2026-08-08-is-novelty-monotone-or-just-rare]] |
| ~~B9~~ | **DONE — FALSIFIED, and it inverts. probe 0.7635 / held-out 0.7342.** Balance is worth +1.51/+1.87 to a frozen trunk and **−0.71/−3.62** to end-to-end. Best-vs-best, the staged recipe is **+0.71 in-dist, −0.14 probe, +0.77 held-out** — F2/G2's 1.65 pt trade was a configuration comparison. [[2026-08-08-self-training-does-not-iterate]] |
| ~~G3~~ | **DONE. in-dist 0.9138 / probe 0.7740 / held-out 0.7518.** Probe missed its range by 0.10 (cleared falsification); held-out missed by 1.8. **Balance is a TRADE at 198 M** (+0.92/−0.82), where it was free at 20 M. **End-to-end (B8) leads both shifted axes at 198 M** — yesterday's "the trade does not survive" holds only at 20 M. [[2026-08-10-balance-is-oversampling-and-it-does-not-scale]] |
| **B10** | RUNNING. end-to-end at 198 M, balanced. *Note: the 198 M verdict no longer depends on it — B10 can only raise end-to-end's best, not help staged.* **Prediction rewritten after B9, before launch: expected to LOSE** (probe 0.770-0.785 vs B8's 0.7798). Run anyway because three interventions have changed usable sign between 20 M and 198 M. |
| ~~R5-eval~~ | **DONE. in-dist 0.9074.** −0.07 vs F2 for +1.51 probe / +1.87 held-out: balance is **free** at the classifier stage. vs end-to-end B3rep5x: **+0.71 in-dist, −0.14 probe, +0.77 held-out**. |
| ~~C3ref-eval~~ | **DONE. 0.9114 vs C3b's 0.9110 — the hold-out cost 0.04 pt, i.e. nothing.** The apparent −0.38 was entirely the denominator. Fourth time the "do both numbers mean the same thing" rule has paid. |
| ~~bal3lvl-prep~~ | **DONE.** 3-level balanced parquet built: 135,296 rows, 151 per species x 896, no species growth. G3/B10 are unblocked on data. |
| ~~H4~~ | **trained cleanly, 5 ep, valid f1_species 0.8689; test eval re-running after a job-file bug.** **proxy-free head** — species prototypes are an EMA buffer, no trained matrix, no optimiser state. **15.36 GB -> 5.12 GB at 1 M classes.** Predicted 0.900-0.912 vs the baseline's 0.9148; falsified below 0.885. [[2026-08-09-can-centroids-be-trained-against]] |

## 3. Ordered backlog — take the top unblocked item

Costs from the measured formula: `images_per_epoch × epochs / 1100 img/s`, 5.04 M images/epoch.
A 5-epoch 20 M run is ≈6.4 h; a 2-epoch frozen-trunk stage is ≈30 min at 20 M, ≈1 h at 198 M.

| # | work | GPU | why, and the gate |
|---|---|---|---|
| ~~1~~ | ~~coverage-preserving gate~~ | — | **DONE as R3 — worked, +4.24 pt.** Its diagnostic opened R4 (no gate at all), now running. |
| ~~2~~ | ~~second adaptation round~~ | — | **running as R2.** |
| ~~3~~ | ~~joint vs sequential classifier stages~~ | — | **running as F3.** |
| 1 | **G3 + B10 — the 198 M pair, balanced** | ~1 h + ~7-10 h | **Owner-approved 2026-08-09, gated on B9 closing the gap.** Both G2 and B8 used unbalanced pseudo-labels, so re-running one arm would repeat at 198 M the unfairness B9 removes at 20 M. Configs, job files and the 3-level prep are built and waiting. G3 predicted probe 0.775-0.795 (falsified below 0.7689); B10 predicted 0.785-0.800 (falsified below 0.7798). |
| 3 | **paper: figures** | none | Deferred by the owner until the storyline settled. It has. The results now demand: the replication-share curve with transfer overlaid, the three-benchmark comparison, and the scoring-rule table. |
| 5 | **P1 — BioCLIP-2 as a frozen trunk** | build + ~1 h | Owner-deferred, reviewer-inevitable. T2b is what makes it plausible. See §5. |
| ~~6~~ | ~~H — the head-scaling build~~ | — | **running as H4.** |
| ~~7~~ | ~~C3b — hold out *common* taxa~~ | — | **running.** |

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
| **training** | **running as H4** — proxy-free: EMA centroids in a buffer, no matrix, no optimiser state. Removes 10.24 GB of the 15.36 GB. [[2026-08-09-can-centroids-be-trained-against]] |
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
