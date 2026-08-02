# PLAN — where we are, and what runs next

**Kind:** living · **Last updated:** 2026-08-02 · **Supersedes:**
[[2026-07-28-landscape-and-plan]]

This is the one file in `journal/` that is meant to be true *today*. Everything else is a record of
a moment. Keep the status board current; when a run lands, move its result into `RESULTS.md` and its
reasoning into a dated entry.

---

## OWNER AWAY 2026-08-02 → ~2026-08-23

The owner delegated the project for three weeks. Two things follow, and the second is the one that
gets forgotten.

**The queue now survives unattended.** A cron entry ticks `ucloud q` every 5 minutes
(`crontab -l`). Before this, the tick ran only inside a shell session, so `auto_extend` was inert and
`--after` jobs never launched the moment the session ended — that failure already cost a 12-epoch run
([[2026-07-30-ucloud-queue-daemon]]). **If jobs appear stuck, check the cron entry first.**

**An agent session is not a monitor.** Work advances only when a session is invoked. So the queue is
loaded with runs that finish and self-evaluate (`--after` chains), and the backlog below is ordered
so any future session can pick the top item without asking. Do not design work that needs a human
mid-flight.

### Ordered backlog — take the top unblocked item

Costs use the measured formula, **not** guesses: `images_per_epoch × epochs / 1100 img/s`, where the
training fold is **5.04 M images/epoch**. A 5-epoch `effnetv2_s` run is **≈6.4 h**, not the "~1.5 h"
this file claimed for weeks.

| # | work | GPU | why it is here, and the gate |
|---|---|---|---|
| 1 | **collect F1 + D2**, journal, update `RESULTS.md`/`START-HERE` | — | Running now. F1 is the flagship; **falsified if shifted ≤ 0.717**. |
| 2 | **D1 — finish `lepinet bundle`**: calibration + thresholds + names in one artifact | none | Pure code, product-blocking, pending since July. Best use of a session while the GPU is busy. |
| 3 | **B3 — self-training on unlabelled trap images** | ~7 h + build | **The highest-value untested rung.** B1 established the name-a-nuisance ceiling at ~4 pt; B3 is the first rung that adapts to shifts nobody named. Needs grouped splits by capture event and held-out-*species* validation, or it will manufacture a phantom win. |
| 4 | **L4 — cRT / decoupled** | ~2 h (stage 2 only) | The 2×2 showed tail-reweighting trades robustness for accuracy monotonically. cRT rebalances *only the classifier*, so it tests **where the damage lives**. Reuses L0's checkpoint; stage 2 must not use Muon (it re-partitions param groups and breaks freezing). |
| 5 | **L3 — LDAM** | ~6.4 h + build | Its per-class margin ∝ n^(−1/4) is *gentler* than √-oversampling, so the monotone curve **predicts it beats oversampling under shift**. An out-of-sample prediction, which is the right reason to run something. |
| 6 | **B2 — background suppression** (flatbug-style) | ~6.4 h + build | Removes a nameable *category* rather than a nuisance dimension. Bounded by imagination like B1, but bounded higher. |
| 7 | **C3b — hold out *common* taxa** | ~6.4 h + build | The controlled version of C3, whose novel species were all rare and so possibly just harder. Needs a species-exclusion filter, which does not exist yet. |

**Do not do** (each has a reason on record): re-tune the ArcFace margin (E1, cancelled — the loss it
targeted was mostly a scoring-rule artifact); chase in-distribution accuracy; the autoregressive head;
balanced softmax at other τ (the 2×2 settled it on both axes).

### What to tell the owner on their return

The three findings that changed what the project claims, in order: **(1)** tail-reweighting trades
robustness for accuracy monotonically, and the long-tail literature cannot see this because its
benchmarks ship no shifted test set; **(2)** the open-set scoring rule does not transfer across model
scale or head convention, which invalidated a headline and cancelled ~80 GPU-hours; **(3)** coarse
*supervision* helps and coarse *parameters* hurt, so the original head question has two opposite
answers and the standard protocol sees only one.

Also flag, because they are corrections rather than results: "consistent by construction" was false
and is retracted; the shared-τ explanation for logit adjustment's failure was a rationalisation and
is retracted; runtime estimates in this file were 4× low until 2026-08-02.

---
## Status board (keep this current)

Updated 2026-08-01. `→` = chained eval. Every finished run must land in `RESULTS.md`.

| id | run | state | result |
|---|---|---|---|
| A1 | effnetv2_s, single head + ArcFace × z-score | **DONE (triple)** | in-dist **0.9035** / shifted **0.6437** / **AUROC 0.9068**. Prediction falsified — the effects do *not* compose — but **A1 stands**: open-set survives the single head (0.9068 vs 0.9115 multi-head, vs ~0.601 plain). Interference is −0.59 species / −1.0 coarse, i.e. the margin damages the *marginalisation*. [[2026-07-30-does-arcface-compose-with-marginalisation]] |
| **F1** | **flagship: DINOv3-cnx-L + ArcFace × z-score + marginal supervision + `domain_aug: trap`** | **running** (12362597) → triple chained | Every intervention that won, composed for the first time. Predicted in-dist 0.918–0.925, **shifted 0.720–0.745**, open-set 0.88–0.90. **Falsified if shifted ≤ 0.717** — that would mean marginal supervision's robustness benefit does not survive scale. |
| D2 | distil A2 → **fastvit_sa12** (not b0) | **running** (12362598) → eval + shift chained | b0 was the binding constraint on the shipped model (A6: student capacity dominates). Predicted 0.895–0.905. |
| A2 | DINOv3-cnx-L, single head + ArcFace × z-score | **DONE (triple)** | in-dist **0.9216** / shifted 0.6616 / **AUROC 0.8298**. Best in-distribution — and **worst deployable**: loses to B1 (10× smaller) under shift and to A1 on novelty by 7.7 pt. **No longer the final-model candidate.** [[2026-07-31-best-model-is-not-the-best-model]] |
| A3 | distil A2 → small single-head student | **DONE** | **0.8833** — best student yet (+0.47 over previous). Prediction (~0.88, <0.89) correct: **the ceiling claim survives**. A *worse* teacher (A2 0.9216) beat a better one (CnxV2-L 0.9316) by 0.77 pt, so teacher accuracy is near-irrelevant — target *shape* is what matters. |
| A6 | single-head b0 from scratch — A3's missing control | **DONE** | **0.8789** (predicted 0.870–0.878, just above). Splits A3's win: **head +0.97 pt, distillation +0.44** — the architecture is worth more than the teacher at b0 scale, and distillation's credit halves |

| A4 | **A1 + marginal supervision** (`marginal_arcface`) | **DONE** — open-set rescore rerunning | in-dist 0.8998 / **shifted 0.6616** / genus **0.9555** family **0.9725**. Shifted prediction **confirmed** (0.655–0.670); recovers **56 % genus / 87 % family** of the margin's damage, confirming the calibration mechanism. **Best shifted score of any effnetv2_s model, above the multi-head.** New recommended architecture. Marginal supervision is worth **+1.41 pt under shift** (2× the floor), recovering the single head's robustness deficit; every model in the table was trained without it, so every shifted number is a floor. Needs a head composing `marginal` with the ArcFace margin (needs a head that composes `marginal` with the ArcFace margin; the margin is applied loss-side today, but marginals are computed in the forward, so it needs a label path) | Two same-day results identified one mechanism from opposite sides: marginal supervision improves the summed posterior's calibration (+0.27/+0.39 coarse), the ArcFace margin degrades it (−1.15/−1.11 coarse). Composing them is the direct test. |
| A5 | repeat of the current baseline | **DONE** | Species **0.9135 → 0.9135 (spread 0.0000)**; genus 0.0005; **family 0.0024**. Noise scales inversely with class count. Downgrades marginal supervision's family claim; everything else clears its floor by 4×+. [[2026-08-01-how-noisy-are-our-numbers]] |
| A5b | shifted-metric spread | **DONE** | **0.0069** (0.6258 vs 0.6327, two trainings). ~10× the in-distribution floor, as the class-count argument predicts. Repeating the *eval* gives 0.0000, so all variance is from training. Every shifted claim ≥1.4 pt survives; nothing below ~0.7 pt is reportable. |
| — | **coarse supervision vs coarse parameters** (marginal head, shifted) | **DONE** | Plain single head is **−2.10 pt** under shift vs the multi-head (while +0.25 in-distribution); marginal supervision recovers **+1.41** of it with no coarse parameters. **Coarse supervision buys shift robustness; coarse parameters do not.** [[2026-07-30-marginal-supervision]] | The project has never measured its seed-to-seed spread, yet routinely interprets 0.2–0.4 pt deltas. 1.5 h, and it retroactively sets the believability threshold for every sub-half-point row in `RESULTS.md`. |
| B0 | more capacity / longer schedule | **deferred, deliberately.** The owner's 12-epoch DINOv3-cnx-L (job 12361261) expired mid-run — no queue daemon was ticking, see [[2026-07-30-ucloud-queue-daemon]]. Not restarted as-is: it trained the *old multi-head* architecture, so its number would land on a superseded baseline. Re-ask as **12 epochs of A2's config** once A2 lands, which isolates schedule length as the single factor. | — |
| B1 | domain-mimicking augmentation (`domain_aug: trap`) | **DONE (triple)** | in-dist 0.8999 / shifted **0.6836** / AUROC 0.9010. **+3.99 pt under shift for −0.36 in-dist — an 11:1 trade, the best in the project.** But closes only **17 % of the gap**: H1 confirmed, the shift is only partly nuisance. [[2026-07-30-domain-shift]] |
| B4 | A2 backbone + B1 augmentation | **DONE (triple)** | in-dist **0.9216** / shifted **0.7101** (project best) / AUROC 0.8132. Shifted prediction correct; **AUROC prediction wrong in the direction that confirms the mechanism**. Augmentation tax vanishes at scale; its shifted gain grows. [[2026-08-01-capacity-x-augmentation]] |
| B2 | background suppression (flatbug-style) | not started | — |
| B3 | self-training on unlabelled OOD images | not started — **now the highest-value untested rung** | B1 established the name-it-yourself ceiling at ~4 pt. B3 is the first rung that adapts to shifts nobody named. |
| C1 | rank-abstention curves (no GPU) | **DONE** | 99.18% answered at 95.04% precision; **coarse ranks must be calibrated conditionally** — genus is 0.487 on the hard subset vs 0.970 overall |
| C2 | OOD AUROC for A1/A2 | **DONE** | A1 **0.9068**, A2 **0.8298**. Open-set survives the single head — but **degrades with scale**, which no one predicted. |
| C3 | hierarchical OOD (near/mid/far), unfiltered parquet | **DONE** | Monotone in taxonomic distance for both heads. ArcFace×z-score: near 0.849 / mid 0.909 / far 0.941; plain 0.561 / 0.618 / 0.666. The hard, common case (`near`) is where the plain head is ~chance. |
| D1 | calibration + thresholds + names in `lepinet bundle` | not started | — |
| D2 | distil into fastvit_sa12 instead of b0 | not started | — |
| — | marginal supervision (owner's true hierarchical idea) | **DONE** | Species **unchanged** (0.9135 → 0.9135, H2 confirmed), but genus **+0.27** / family **+0.39** pp — free coarse accuracy, no parameters. Follow-up is `dev/058` on these predictions: does it help the *hard* subset, where conditional genus precision was 0.487? |

Done and folded in: H4 (backbone × shift, 3.0× spread), margin tuning (abandoned, two principled
failures), flemming OOD under shift (0.727 vs 0.574).

## H4, answered first: backbone choice matters 3× more under shift

| backbone | in-distribution | flemming (shifted) | gap |
|---|---|---|---|
| effnetv2_s (20 M) | 0.9110 | 0.6503 | 26.1 pt |
| ConvNeXtV2-L (198 M) | **0.9316** | **0.7122** | 21.9 pt |
| DINOv3-ConvNeXt-L (198 M) | 0.9311 | 0.7098 | 22.1 pt |
| **spread** | **2.1 pt** | **6.2 pt** | **3.0×** |

**H4 confirmed:** the in-distribution spread across backbones is 2.1 pt; under domain shift it is
6.2 pt. Ranking backbones on the in-distribution benchmark therefore *understates* what the choice is
worth by 3×, and every backbone comparison this project has run so far used the understating metric.

**But the mechanism is scale, not self-supervision.** DINOv3-ConvNeXt (0.7098) and ConvNeXtV2-L
(0.7122) are indistinguishable under shift despite completely different pretraining; the gain is
20 M → 198 M. That is a useful narrowing: it predicts a *bigger* model helps robustness again, and
that swapping pretraining at fixed size does not. It also means the cheapest robustness lever we
have is simply **more capacity in the teacher**, which we already know how to run.

**Consequence for evaluation policy:** from now on **every** model comparison reports the shifted
score alongside the in-distribution one. It is a 12-minute eval and it changes conclusions.

## The hygiene problem this plan fixes

ArcFace, distillation and the backbone sweep were all run on the **old multi-head** baseline, which
has since been superseded by single-head + marginalisation (0.9135 > 0.9110 at every level). So:

- the **recommended architecture — single head + ArcFace × z-score + marginalisation — has never
  been trained**; every claim about it is an inference from two separate experiments;
- the distillation results used a multi-head teacher *and* a multi-head student;
- nothing has been measured on the shifted benchmark except by accident.

Group A below closes that gap. Nothing in Group B or C should be believed until A lands, because
A changes what "the model" means.

---

## Group A — consolidate the architecture (blocking; run first)

| id | run | why | cost |
|---|---|---|---|
| **A1** | effnetv2_s, **single head + ArcFace × z-score**, m=0.3 | The recommended architecture, never trained. Tests whether the +0.25 pt from single-head and the +31 pt AUROC from ArcFace **compose**, or interfere. | ~1.5 h |
| **A2** | DINOv3-ConvNeXt-L, single head + ArcFace × z-score, 6 ep @320 | The **candidate final model**: best backbone × best head. Also the teacher for A3. | ~18 h |
| **A3** | distil A2 → small single-head student (T=1) | Re-runs distillation with both ends on the new architecture. | ~2 h |

Each is evaluated three ways: in-distribution macro-F1 (all levels via marginals), **flemming**, and
**open-set AUROC**. That triple is the new standard report.

**Predictions.** A1 ≈ 0.906–0.915 F1 with AUROC ≥ 0.90 (the two effects are mechanistically
independent — one changes which heads exist, the other the logit geometry — so they should compose).
A2 ≈ 0.93 in-distribution, ~0.71 shifted, AUROC ≥ 0.90. A3 ≈ 0.88, i.e. still student-capacity-bound
— **if A3 exceeds 0.89 my "the student is the ceiling" claim is wrong** and the whole distillation
section needs revisiting.

## Group B — robustness (the frontier; after A)

| id | run | why |
|---|---|---|
| **B0** | ConvNeXtV2-**H** (660 M) or a longer ConvNeXtV2-L | H4 says scale buys robustness. Cheapest test of whether the 6.2 pt spread keeps growing. |
| **B1** | domain-mimicking augmentation on A1 | Diagnostic: if it recovers most of 23 pt the gap is nuisance; if little, it is semantic. |
| **B2** | background suppression (flatbug-style) | Removes a whole nameable *category*, not one nuisance dimension. Also a user-facing knob. |
| **B3** | self-training on unlabelled flemming images | First genuinely general rung; uses OOD images, **no OOD labels**. Needs the grouped split. |

Protocol for all of B (from [[2026-07-30-domain-shift]]): grouped splits by capture event, validation on
**held-out species**, and the in-dist/shifted/AUROC triple every time.

## Group C — finish the open-set story (paper-blocking)

| id | work | why |
|---|---|---|
| **C1** | rank-abstention curves: per-rank thresholds → coverage/precision | The paper's §4.5 is empty and this is the *product* metric ("at 95 % species precision we cover X %, the rest resolve to genus"). Pure post-processing on saved predictions — no GPU. |
| **C2** | OOD AUROC for A1/A2 | Confirms open-set survives the architecture change. |
| **C3** | a larger novel set under shift | Today's flemming OOD has only 234 novel images (±0.03). Use the un-reconciled flemming_helsing OOD species, or hold out species from the trap data. |

## Group E — open-set is now the binding constraint (Aug 1)

Across the capacity × augmentation factorial, **every** intervention that bought accuracy cost
open-set AUROC (0.9068 → 0.8132) and none traded the other way. Accuracy is no longer scarce;
novelty detection is. These attack that axis directly.

| id | work | why | cost |
|---|---|---|---|
| ~~E1~~ | ~~re-tune the ArcFace margin at 198 M~~ **CANCELLED** — E2 showed two thirds of the loss it targeted was a scoring-rule artifact, found in 5 minutes instead of the ~36 GPU-hours estimated (really closer to 80, see the runtime note above) | `m = 0.3` was chosen on a 20 M model; larger embeddings concentrate cosines differently, so the margin may simply be too small at scale. Cheapest hypothesis for the 8.8 pt. **Note this reverses the earlier decision to abandon margin tuning** — that was taken when the margin only had to justify itself in-distribution. | ~18 h ×2 |
| **E2** | **DONE — `msp` beats `max-logit` by +6.1/+7.6 pt at 198 M.** Compare five OOD scoring rules (max / energy / msp / entropy / top-2 margin) on one forward pass — `dev/061` | Asks whether the 8.8 pt loss is in the *embedding* or in the *rule*. **Must run before E1**: if another rule reads the same embedding better, there is nothing to retune. *(The originally planned temperature scaling was dropped as vacuous — AUROC is a rank statistic and `max_logit/T` is monotone in `max_logit`, so T cannot change it. Verified in the script's self-test.)* | 4 × ~5 min, **running** |
| **E3** | B3 (self-training) | Still the highest-value robustness rung. Note the framing has changed: with the rule fixed, open-set is **not** the binding constraint after all — the shifted axis is, where B4 leads at 0.7101 against an in-distribution 0.9216. | large |
| **E4** | measure the **AUROC noise floor** | The capacity penalty is now 1.64 pt and no one knows the spread on this axis. Score the two baseline copies (A5 + original) with `dev/061`. | 2 × 5 min |

## Closed: the multi-head's 0.69 pt shifted lead was not real

M2 (a second marginal-supervision training) scored **0.6485** against draw 1's 0.6434 — mean 0.6460,
own spread 0.0051, and the multi-head sits 0.0043 away. **Indistinguishable, by measurement rather
than by citing a floor.** The auxiliary-coarse-heads follow-up this motivated is dropped.

`lepi-cond-shift` also landed: the conditional head is **0.6213**, worst of all four under shift as
well as in-distribution — the only head whose two rankings agree.

## Group L — imbalanced learning, benchmarked on the triple (Aug 1)

A 2×2 of resampling × loss reweighting, from [[2026-08-01-imbalance-methods-bench]]. The framing
that makes it worth running: **balanced softmax is logit adjustment at τ=1**, which this project
already rejected — but it lost because one shared τ spanned three level distributions, and the
single-head architecture supervises only one. Separately, τ-normalisation is *already in the model*
(the cosine head's unit-norm prototypes), so only frequency-reweighting methods have room to act.

| id | run | state |
|---|---|---|
| **L0** | no oversampling, no balanced softmax — the control nobody ran | **DONE**: 0.8949 in-dist / **0.6445 shifted**. Oversampling is **+1.86 in-dist and −1.52 shifted** — an accuracy/robustness trade, the 4th inversion |
| **L1** | balanced softmax instead of oversampling | **DONE**: 0.8970 in-dist / **0.5726 shifted**. **Prediction falsified** — the shared-τ explanation for logit adjustment's original failure is retracted |
| **L2** | both | **DONE**: 0.8689 / **0.5492** — worse than every other cell. They double-count, as predicted |
| — | √-oversampling alone | **have it: 0.9135** |

**Result: shifted accuracy is monotone in tail-reweighting aggressiveness** — 0.6445 (none) >
0.6293 (√-oversampling) > 0.5726 (balanced softmax) > 0.5492 (both), a **9.5 pt spread** against a
0.69 pt floor, while the in-distribution column does not order at all. The predicted ordering was
committed before the runs and came back exactly.

Next, and now motivated by the curve rather than by the literature:

| id | run | why |
|---|---|---|
| **L3** | **LDAM** (per-class margin ∝ n⁻¹ᐟ⁴) | A *fourth-root* softening — gentler than √-oversampling — so the monotone curve **predicts it should beat oversampling under shift**. An out-of-sample prediction to test, not a method to try. Needs per-class margin support in the loss. |
| **L4** | **cRT / decoupled** | Rebalances only the classifier, leaving the representation trained on the natural distribution. If the damage is to the *representation*, cRT escapes the trade entirely. Cheap: stage 2 reuses a frozen backbone. |

## Group D — product (independent of A–C)

| id | work |
|---|---|
| **D1** | calibration + thresholds + names into `lepinet bundle`, so a release is complete |
| **D2** | distil into **fastvit_sa12** (10.6 M, 0.892 in the old sweep) rather than b0 (0.876) — b0 is the binding constraint on the shipped model |

## What is *not* worth doing

- **Tuning ArcFace `m`.** Two attempts at a cheap proxy failed for principled reasons; m=0.3 already
  gives 0.9115, so this is a refinement needing full runs. Revisit only if A1/A2 disappoint.
- **The autoregressive head.** Lost by 20 pt; nothing has changed that would rescue it.
- **More in-distribution accuracy chasing.** Saturated at ~0.93 while the shifted number is ~0.71.

## Running order

A1 + A2 + A3 in parallel now → C1 while they run (no GPU) → B0/B1 → B2/B3 → D. Group A is the gate:
until it lands, "the model" is ambiguous and every downstream number inherits that ambiguity.
