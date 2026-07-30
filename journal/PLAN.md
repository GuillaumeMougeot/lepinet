# PLAN — where we are, and what runs next

**Kind:** living · **Last updated:** 2026-07-30 · **Supersedes:**
[[2026-07-28-landscape-and-plan]]

This is the one file in `journal/` that is meant to be true *today*. Everything else is a record of
a moment. Keep the status board current; when a run lands, move its result into `RESULTS.md` and its
reasoning into a dated entry.

Written after the documentation pass exposed that the *recommended*
architecture has never been trained. Compute is not the constraint; **experimental hygiene is** — the
danger is accumulating results on inconsistent baselines, which is exactly what happened already.


## Status board (keep this current)

Updated 2026-07-30. `→` = chained eval. Every finished run must land in `RESULTS.md`.

| id | run | state | result |
|---|---|---|---|
| A1 | effnetv2_s, single head + ArcFace × z-score | **in-dist DONE**; shift + OOD queued (`lepi-A1-shift`, `lepi-A1-ood`) | **0.9035** / 0.9491 / 0.9628 — **prediction falsified** (floor was 0.906). The two effects do *not* compose: interference is −0.59 pt at species but **−1.0 pt at genus/family**, so what breaks is the *marginalisation*, via calibration. See [[2026-07-30-does-arcface-compose-with-marginalisation]]. Verdict pending the AUROC, which is the axis ArcFace exists for. |
| A2 | DINOv3-cnx-L, single head + ArcFace × z-score (final-model candidate) | **queued** → eval queued | — |
| A3 | distil A2 → small single-head student | blocked on A2 | — |
| A4 | **A1 + marginal supervision** | not started — *newly indicated* | Two same-day results identified one mechanism from opposite sides: marginal supervision improves the summed posterior's calibration (+0.27/+0.39 coarse), the ArcFace margin degrades it (−1.15/−1.11 coarse). Composing them is the direct test. |
| A5 | **seed-repeat of the current baseline** | not started — *overdue* | The project has never measured its seed-to-seed spread, yet routinely interprets 0.2–0.4 pt deltas. 1.5 h, and it retroactively sets the believability threshold for every sub-half-point row in `RESULTS.md`. |
| B0 | more capacity / longer schedule | **deferred, deliberately.** The owner's 12-epoch DINOv3-cnx-L (job 12361261) expired mid-run — no queue daemon was ticking, see [[2026-07-30-ucloud-queue-daemon]]. Not restarted as-is: it trained the *old multi-head* architecture, so its number would land on a superseded baseline. Re-ask as **12 epochs of A2's config** once A2 lands, which isolates schedule length as the single factor. | — |
| B1 | domain-mimicking augmentation (`domain_aug: trap`) | **in-dist DONE**; shift queued (`lepi-B1-shift`) | 0.8999 in-distribution (−0.36 pt vs A1) — the expected cost. **Decides nothing until the shifted number lands.** |
| B2 | background suppression (flatbug-style) | not started | — |
| B3 | self-training on unlabelled OOD images | not started | — |
| C1 | rank-abstention curves (no GPU) | **DONE** | 99.18% answered at 95.04% precision; **coarse ranks must be calibrated conditionally** — genus is 0.487 on the hard subset vs 0.970 overall |
| C2 | OOD AUROC for A1/A2 | blocked on A | — |
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
