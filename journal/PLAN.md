# PLAN — where we are, and what runs next

**Kind:** living · **Last updated:** 2026-08-28 · **Supersedes:** [[2026-07-28-landscape-and-plan]]

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
| **H** | scaling the head to ~1 M species | **closed, and reframed** — five routes dead, but a min-50 floor turns 884 K species into 204 K and the matrix then fits (3.13 GB with Adam). The answer is a data policy, not an architecture |
| **L** | imbalanced learning | **closed** — cRT is the answer |
| **P** | pretrained encoders (BioCLIP-2) as frozen trunks | **deferred** (owner) |
| **T** | what target labels would have bought | **closed** |

## 2. State (2026-08-27)

**Two conclusions retracted today.** G3b — an exact repeat of G3 — scored held-out **0.7892 against
G3's 0.7518**, a 3.74 pt spread between identical runs and **7.2x the floor** we had been quoting.
The floors were measured on 5-epoch end-to-end 20 M runs and do not transfer to 2-epoch frozen-trunk
stages at 198 M. [[2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes]]

Dead as a result: **"balance is a trade at 198 M"** and **"end-to-end leads at 198 M"**. Staged is
either ahead or behind end-to-end depending on which of two identical draws you pick, so the honest
statement is that they are indistinguishable at 198 M.

**Still true:** the staged recipe's in-distribution advantage (+0.71 to +0.88 across four runs, on a
metric whose spread here is 0.0010), and everything with a margin above ~3 pt.

**R5b measured it (2026-08-27):** probe spread **0.0119**, held-out 0.0079. Probe is ~0.012 in both
frozen-trunk regimes and ~3x the old floor, which retracts "R5 beats R3 on probe" and "balance is
worth +1.51 probe". The core self-training story (coverage dominates, no gate) is untouched.

*(original note)* one repeat of R5 to measure the 20 M frozen-trunk spread. F2/F3/R3/R4/R5/T2 are *all* frozen-trunk stages scored against an end-to-end floor, so the
same problem may apply there; the margins are mostly larger, but R5-vs-R3 probe (+1.07) is exposed.

| run | result |
|---|---|
| **G3b** | probe 0.7870 / held-out 0.7892 / in-dist 0.9148. **FALSIFIED** its 0.745-0.760 prediction |
| **P1a** | full fold 0.8629, **clean fold 0.8444 vs baseline 0.9021 — a 5.77 pt gap**. Contamination was worth only ~0.58 pt; most of the drop is the harder denominator [[2026-08-24-does-the-recipe-need-our-backbone]] |
| ~~P1b~~ | **DONE — FALSIFIED by 11 pt. probe 0.5901 / held-out 0.5599 vs T2b's 0.7515.** The recipe is **not** trunk-agnostic; the backbone stays in the contribution list. [[2026-08-24-does-the-recipe-need-our-backbone]]. Originally failed — failed on a missing `flemming` mount (the pseudo-label parquet reaches trap images via `../../flemming/images/`). Third time this class of bug has hit; note added to all three P1b TOMLs |

## 2b. ToL-200M direction (opened 2026-08-26)

| | status |
|---|---|
| **contamination** | **BioCLIP-2 has seen 93.3 % of our species and 65.4 % of our images**, including 413,865 test-fold images, by exact GBIF occurrence id. In-distribution comparisons are contaminated; the shifted ones are not. [[2026-08-26-bioclip2-has-seen-two-thirds-of-our-test-fold]] |
| **768-d cache usable?** | **Yes.** Projected 768 matches pooled 1024 on centroid top-1 across three samples; the projection compresses the cosine scale but preserves ranking. No re-embedding needed. [[2026-08-26-the-clip-projection-does-not-hurt-us]] |
| **zero-shot vs probe** | fitted probe beats zero-shot by ~9 pt at matched class count; prompt template worth 0.3 pt. Zero-shot is the wrong instrument for representation quality. |
| ~~P3~~ | **DONE. Fine-tuned BioCLIP-2 = 0.9146 vs our 0.9021 (+1.25).** Frozen was −5.77, so unfreezing is worth **+7.02**. LR span 2.34 pt with our default the worst. **The owner's under-optimisation hypothesis is falsified in-distribution, and the ToL download is off the table** unless P3c's shifted arm reverses it. [[2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts]] |
| ~~L7~~ | **DONE. Shifted axes peak at cap 1,000**: +1.57 probe, +3.26 held-out vs our ~2,000, for −0.88 in-dist. Our cut was right in kind and too shallow. |
| ~~P3c shifted~~ | **DONE. probe 0.6630 (+3.60), held-out 0.6937 (+5.25) vs our baseline.** Fine-tuned BioCLIP-2 wins on **all three axes**, largest margin on held-out. Revises finding 7d: the backbone matters and **theirs is better than ours**. Our contribution is the recipe, not the encoder. |
| ~~P4~~ | **DONE — FALSIFIED. probe 0.7199** (predicted 0.76-0.80). **Crossover**: BioCLIP-2 is +3.60 ahead of our baseline before adaptation and −3.16 behind T2b after it. Adaptation is worth +10.78 on our trunk, **+5.69 on theirs**. |
| ~~P5~~ | **DONE — prediction correct. probe 0.7810 / held-out 0.7806.** Unfreezing is worth **+5.93 pt** over P4, so the crossover was a frozen-readout artefact, not substitution. **P5 ties B8, our best model.** Bounds the staged recipe to trunks trained with the same head. [[2026-08-28-frozen-adaptation-only-works-on-your-own-trunk]] |
| ~~L7c1000b, P4b~~ | **DONE. Both claims survive n = 2.** cap 1,000 vs uncapped: **+1.76 probe, +2.94 held-out**, ~3x the combined spread, n = 2 on both arms. P4 confirmed at 0.7218 (spread 0.0037). |
| *(superseded)* | our recipe, their backbone, UNFROZEN, 3 LR arms. Decides whether BioCLIP-2's representation is genuinely limited or merely badly read while frozen — the confound both P1 arms share. |
| **next** | open-set at scale on the cached embeddings: near/mid/far strata sampled across the whole tree, and AUROC as a function of how many taxa are enrolled. Embedding-only, one GPU. |
| **ToL at our policy** | **88.1 M images, 203,878 species** (min 50, cap 2,000): 15.5x our images, 16.1x our species. 22.2 h/epoch, 2.2 TB at 256 px, ~84 M images to download. **The 1280 x 204 K matrix is 3.13 GB with Adam and FITS** — a min-image floor solves what Group H could not solve by architecture. Upper ranks need a validity audit before "7 levels" is claimed. [[2026-08-27-tol-at-our-policy-and-the-head-scaling-problem]] |
| **head concentration** | **ToL's surplus is ~90 % head.** At our 2,000 cap it offers **+1.26 M images (1.2x)**, not 3.1x. Top 10 species = 1.63 M images; uncapped they would take 7.8 % of every epoch against 11.6 % for the 65,453 rarest combined. The ToL-data direction is much weaker than it looked. [[2026-08-27-tol-extra-data-is-almost-all-head]] |
| **L7 (running)** | **where is the optimal cap?** (reframed from "was cutting the head a mistake") Our construction capped at ~2,000 imgs/species for balance and never tested it. ToL did not: 19.1 M vs our 6.2 M for the same 12,494 species, with near-identical medians -- the difference is entirely the head, and only 1.76 M ToL images are in species we lack. Sweep caps 250/500/1000 to measure the slope with local data. A slope measured over 250-2,000 says nothing about caps of 10^5, so it can justify acquiring up to ~3,000-5,000/species, not restoring the full head. |
| **tail data** | the owner has the full Lepidoptera set on UCloud already — our 6.3 M is what survived the `min_img_per_spc` floor, so the extra ~17 M *is* the tail we deliberately removed. No download needed if we revisit it. |
| **do not** | pretrain a backbone on ToL-200M. It scales the *source* domain 43x, which our own factorial says does not buy shift robustness -- and BioCLIP-2 is a 200 M-image replication of exactly that. |

**Caveat to carry:** absolute cosines differ between the two spaces, so every threshold (abstention,
novelty, calibration temperature) must be refitted if the space changes. Ranking transfers, numbers
do not.

## 3. Ordered backlog — take the top unblocked item

Costs from the measured formula: `images_per_epoch × epochs / 1100 img/s`, 5.04 M images/epoch.
A 5-epoch 20 M run is ≈6.4 h; a 2-epoch frozen-trunk stage is ≈30 min at 20 M, ≈1 h at 198 M.
**Throughput is ≈1100 img/s at 20 M but ≈480 img/s at 198 M** (measured on B10: 6 epochs x 5.18 M
images in 18 h). Using the 20 M figure for a 198 M run underestimates by 2.3x — it turned a
costed "7-10 h" into 18 h on 2026-08-09.

**The paper is written.** The 2026-08-28 audit ([[2026-08-28-what-the-paper-is-still-missing]]) found
eight retracted or false numbers in the draft and six results with no home in it; all are fixed.
**§4.13** (long-tail rebalancing / cRT, resolving six dangling `§4.x` refs), **§4.14** (foundation
models and contamination), **§3.2** (the four benchmarks) and **§4.0** (summary of models) are
written, and the abstract, contributions, discussion, limitations and references are reconciled.

**Owner-owned remainder**, listed in the draft's own header: fact-check the `[VERIFY]` citations;
redraw `fig4` per-head-best-rule (it currently scores both heads with max-logit, which is the
retracted comparison); re-score §4.5 with each head's own rule.

**Two new directions opened (owner, 2026-08-28)**, scoped with committed predictions in
[[2026-08-28-two-directions-checklists-and-our-objective-on-tol]]:

- **D1 — regional checklist at inference.** Every number in the paper uses a 12,041-class global
  head; no deployment does. Mask the species logits to a national checklist (the trap corpus's 486
  Danish species), measure the accuracy-vs-checklist-size curve. Inference-only. `dev/081`.
- **D2 — our objective on Tree-of-Life.** Does cosine z-score + marginal supervision build a better
  *classifier trunk* than CLIP contrastive, on the same data? **Scoping found that ToL-200M hosts
  metadata only (0.05 TB) while ToL-10M hosts 2.0 TB of image tarballs** — so the 200M route needs an
  84 M-request crawler and the 10M route needs a download. **ToL-10M is also the better experiment**,
  because BioCLIP-1 was trained on exactly it, making the comparison matched-corpus with the
  objective as the only variable.

| # | work | GPU | why, and the gate |
|---|---|---|---|
| ~~**D1**~~ | **DONE (probe). micro-acc +5.06, macro-F1 -1.70.** The metrics disagree: restriction concentrates false positives onto the scored classes, so it helps the average observation and hurts the tail. macro-F1 peaks at ~4,000 labels, not at the true 464. probeho arm running. [[2026-08-28-a-regional-checklist-helps-the-user-and-hurts-the-tail]] | eval only | **next: restricted head + abstention**, which should recover the tail damage |
| **P5b** | seed repeat of P5 | running | "P5 ties B8" is n = 1 on a shifted axis, and two conclusions have died that way |
| ~~**O1**~~ | **abstention DONE: B8 and P5 tie on macro-F1 and differ by 17.3 pt on useful-answer rate** (73.3 % vs 92.8 % answered). Abstention under shift costs 26.7 % coverage against 0.82 % in-distribution. Open-set rules running. [[2026-08-28-two-tied-models-differ-by-17-points-in-deployment]] | eval only | **P5 is the model to ship**, and the reason is calibration, not accuracy |
| **D2a** | ~~ToL-10M~~ **ToL-200M crawler (owner chose 200M): `dev/082`, running** | CPU node | plan stage scanning 1,838 row-groups. Measured: 99.5 % success on S3, ~25 KB/image, iNat `medium` variant is **10.2x less transfer** than `original` for identical output |
| **D2b** | train our objective on ToL-10M, eval on Lepidoptera | ~2.5 h/epoch | the controlled objective comparison. Predicted: accuracy within ±1.5 pt, **frozen probe better by >3 pt** |
| **O2** | open-set at 12 K -> 204 K taxa on cached ToL embeddings | 1 GPU, no download | turns §4.9 from "rules do not transfer across model scale" into "nor across class-count scale" |
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
