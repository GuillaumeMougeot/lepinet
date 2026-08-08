# Self-training iterates only if the gate preserves coverage — and the gate may not be needed at all

**Kind:** research · **Status:** **RESOLVED for R2/R3 (2026-08-08); R4 open.** Re-labelling with F2
instead of B4 produced **more accurate labels covering less than half the species** and cost **3.80
points** (R2). Swapping the confidence-quantile gate for a per-species one — *nothing else changed* —
recovered **+4.24** (R3), while label accuracy fell 24.6 points. But the diagnostic shows the
**ungated** set beats R3's on accuracy *and* coverage, so R4 tests removing the gate entirely.

## Result

| | round 1 (B4 labels) | round 2 (F2 labels) |
|---|---|---|
| images kept | 12,230 | 8,169 (−33 %) |
| **species covered** | **346** | **156 (−55 %)** |
| pseudo-label accuracy | 98.15 % | **99.84 %** (+1.69) |
| **probe** | **0.7541** | **0.7161 (−3.80)** |

**Predicted probe 0.758–0.772, falsified below 0.7582.** Landed at 0.7161 — falsified by 4.2 points,
the largest prediction miss of the project. The reasoning was that a 4.4-point-better labeller would
produce cleaner labels and eat into the gap to end-to-end training. **The labels did get cleaner. It
made things worse.**

## The mechanism

The gate keeps the top 30 % by confidence — a *quantile*, chosen because absolute softmax values are
not comparable across models ([[2026-08-03-b3-self-training]]).

Round 1's labeller was **saturated**: its confidence distribution had a median of 0.9999996 and ties
at exactly 1.0, so the 0.70 quantile landed *on* 1.0 and every tied image was kept — 44.9 % rather
than 30 %, spanning 346 species.

Round 2's labeller is better, and therefore **better separated**. Its cut falls at 0.99896, no ties,
exactly 30 % kept — and the top 30 % of a good model is a *narrower, easier, more homogeneous* slice
of the domain. It concentrated on 156 species.

So the improvement in the labeller **caused** the loss of coverage. The gate's behaviour depends on
the shape of the confidence distribution, and improving the model changes that shape in the direction
that shrinks the selection.

**This is not the failure mode the design was guarding against.** The stated risk was circularity —
the model reinforcing its own errors. Label accuracy went *up*, so that risk did not materialise. The
one that did was invisible in the design: **selection collapse.**

## Coverage beats label accuracy, and by a lot

+1.69 points of label accuracy did not pay for −55 % of species. Put beside the replication sweep
([[2026-08-04-replication-sweep]]), where 0.39 % of training bought 97 % of the gain, the two results
say the same thing from different directions:

> **What unlabelled target data supplies is *coverage of the input space*, not label information or
> gradient volume.** Anything that trades coverage for label quality is trading the wrong way.

That also explains why round 1 worked as well as it did *by accident*: its labeller was
overconfident, the gate saturated, and the saturation kept a broader set than the design intended.
A better-calibrated round-1 labeller would have scored worse.

## The fix, and the prediction it makes

The gate should target **coverage**, not a confidence quantile. Two candidates, both cheap:

1. **Keep a fixed number per species** — say the top *k* images of every species the model predicts,
   so coverage is guaranteed by construction and confidence only orders within a class.
2. **Raise `keep_frac` for later rounds** to hold the species count constant.

**Prediction (committed):** round 2 with per-species selection at matched coverage (346 species)
lands at probe **0.755–0.775**, i.e. at least matching round 1 and plausibly beating it, because the
labels really are 1.69 points cleaner. If it still loses, iteration genuinely does not work here and
the cause is circularity after all.

## What it changes now

**The recommended recipe stays at one round.** Round 1 at 2 % of training, and no iteration until
the gate is fixed. That is a real limitation of the headline result and belongs in the paper as one:
the method's largest lever works once, and the obvious way to compound it fails for a reason that
took a diagnostic to find.

**And it is a warning worth generalising.** A confidence-quantile gate is standard in self-training,
and its interaction with the labeller's calibration is the kind of thing that silently degrades an
iterated pipeline while every intermediate metric — label accuracy most of all — looks better.

---

## R3: the coverage-preserving gate works, and it is worth 4.24 points (2026-08-08)

Same labeller (F2), same 2 % dose, **only the gate changed**: top *k* = 35 images of every species the
model predicts, instead of the global top-30 % by confidence.

| | R2 — quantile 30 % | R3 — per-species k = 35 | Δ |
|---|---|---|---|
| pseudo-label accuracy | **99.84 %** | 75.22 % | **−24.62 pt** |
| species covered | 156 | **896** | **+740** |
| images kept | 8,169 | 7,801 | −368 |
| **probe** | 0.7161 | **0.7585** | **+0.0424** |
| **probe held-out** | 0.7257 | **0.7682** | **+0.0425** |

**Predicted probe 0.755–0.775, falsified below 0.7500. Landed 0.7585 — inside the range**, and the
first prediction in this thread to hold. Against round 1 (F2 labels, probe 0.7541) it is +0.0044 probe
(1.1× floor) and +0.0088 held-out (1.7× floor), so iteration is now marginally positive rather than
badly negative.

**The controlled comparison is the result.** Two runs identical except the selection rule, at nearly
identical dataset size: a **24.6-point drop in label accuracy bought 4.24 points of probe**. That is
about as clean a statement of the coverage-over-accuracy principle as this project is going to get,
and it was obtained for 40 minutes of GPU.

## But the diagnostic says my gate is worse than no gate

The gate log is the interesting part:

```
per-species gate: top 35 of each predicted species -> 7801 images over 896 species (min conf 0.0007)
pseudo-label accuracy: 0.8643 over all 27230, 0.7522 over the 7801 kept (gate bought -0.1121)
```

The **ungated** set — every adaptation image, no selection at all — is **86.43 %** accurate and covers
every species the model predicts. R3's kept set is 75.22 % accurate over 896 species. So the ungated
set **dominates R3 on both axes at once**: more accurate *and* broader.

My gate did not merely fail to help. It cost 11.2 points of label accuracy, because taking the top 35
of *every* predicted species forces it to scrape the bottom of spuriously-predicted classes — the
minimum kept confidence is 0.0007. Guaranteeing coverage per species means buying garbage for the
species that should not have been predicted at all.

**The gate has been in this pipeline since day one and has never been justified by a measurement.** It
was introduced against a circularity risk ([[2026-08-03-b3-self-training]]) that R2 then showed did not
materialise — label accuracy went *up* under the gate. Two runs have now varied the gate and the one
with worse labels won. The obvious arm was never run.

## R4: no gate at all

Queued. All 27,230 pseudo-labelled adaptation images, 86.43 % accurate, full predicted-species
coverage, same 2 % dose.

**Prediction (committed): probe 0.760–0.780** — at least matching R3, probably beating it, since the
ungated set dominates R3's selection on both of the axes this thread has identified as mattering.
**Falsified below 0.7544** (R3 minus one floor). If it *is* falsified, the coverage/accuracy framing is
incomplete in a specific and interesting way: it would mean confidently-wrong labels concentrated in a
few classes are less harmful than noise spread across many, which is not what either framing predicts
and would be worth a separate entry.

**Either result retires an unexamined design decision**, which is the thing I should have done before
building two rounds on top of it.

**The caveat, recorded before the number arrives.** All three arms are held at the same 2 % dose, and
`dev/066` reaches that dose by replicating the pseudo rows. So the replication factor is not free:

| | unique images | replication | pseudo rows |
|---|---|---|---|
| R2 | 8,169 | 16.7× | 136 k |
| R3 | 7,801 | 17.5× | 136 k |
| R4 | **27,230** | **5.0×** | 136 k |

At a fixed dose you cannot vary unique coverage without varying replication — they are the same knob
seen from two ends. R2 vs R3 was clean because both sit at ~17×; R4 changes both. The replication
sweep ([[2026-08-04-replication-sweep]]) already characterised the dose axis on its own and found the
gain saturating well below 2 %, which is why I am willing to read R4 as a coverage result. But if R4
lands high, "less replication" is a live alternative explanation and the matched-unique-count control
(R4 subsampled to 7,801 unique at 17.5×) is the run that separates them.

---

## R4: the gate is unnecessary on one benchmark and load-bearing on the other (2026-08-08)

**Predicted probe 0.760–0.780, falsified below 0.7544. Landed 0.7674 — inside the range.** And then
the held-out number arrived and made the prediction beside the point.

| | gate | unique | repl. | label acc | species | probe | **held-out** |
|---|---|---|---|---|---|---|---|
| F2 | quantile (round 1) | 12,230 | — | 98.15 % | 346 | 0.7541 | 0.7594 |
| R2 | quantile 30 % | 8,169 | 16.7× | 99.84 % | 156 | 0.7161 | 0.7257 |
| R3 | per-species k = 35 | 7,801 | 17.5× | 75.22 % | 896 | 0.7585 | **0.7682** |
| **R4** | **none** | **27,230** | **5.0×** | 86.43 % | 896 | **0.7674** | 0.7458 |

**The two shifted benchmarks now rank the arms differently.** R4 beats R3 on probe by +0.0089
(2.2× floor) and *loses* to it on held-out species by **−0.0224 (4.3× floor)** — R4 is worse on
held-out than doing nothing beyond round 1 (F2, 0.7594). Both gaps are well clear of noise, so this
is a real split rather than two draws.

## Why, and it is not the reason I queued the run for

R3's cap is `top 35 of each predicted species`, and its kept set averages **8.7 images per species**.
So for the large majority of species the cap never binds — R3 keeps *everything* the model predicted
for them. The 19,429 images that R3 discards and R4 keeps are, almost exactly, **the excess above 35
in the frequent species**.

That means R3 is not a higher-quality subset of R4, and it is not a narrower one either. **R3 is R4
with the head of the trap distribution truncated.** The difference between them is the *shape of the
pseudo class distribution*, and the coverage framing I used to justify R4 does not distinguish them
at all — they have identical species coverage, 896.

Read that way the split is exactly what you would expect. The trap data is long-tailed like
everything else here. Adding its head:

- **helps probe**, which is scored over species that are in the label set and where the frequent
  trap species carry weight (micro-accuracy 0.8430);
- **hurts held-out species**, 58 taxa the model was never trained on, because an over-represented
  known class is a stronger attractor and novel images get absorbed into it.

**So R3's gate was never doing confidence filtering — it was doing class balancing**, and that is why
it looked like a coverage device. The `k` in "top *k* per species" is a cap, and a cap on a
long-tailed distribution is oversampling's twin.

## What this corrects

**The coverage claim as stated is too strong.** [[2026-08-04-replication-sweep]] and R2/R3 supported:

> Anything that trades coverage for label quality is trading the wrong way.

R4 keeps that intact — it has R3's coverage and better labels and wins the benchmark those results
were about. What R4 adds is that **coverage is not the only property of the pseudo set that matters;
its class distribution matters too, and the two benchmarks disagree about which way.** START-HERE
finding 7 has been amended rather than withdrawn: the coverage-over-label-quality trade holds, and
the sentence claiming it is the *whole* story does not.

It also lands squarely on the project's through-line. Long-tail rebalancing has already been shown to
belong in the classifier ([[2026-08-01-imbalance-methods-bench]], cRT) rather than in the data the
backbone sees. This is the same lesson arriving from a third direction: **the pseudo-label set is
training data, so its class distribution is a design parameter, not a property of the trap.**

## R5, queued

Same 27,230 ungated images, replicated so **every species contributes the same number of rows** —
R4's coverage, a distribution flatter than R3's. `dev/066 --balance`, verified on a synthetic
long-tailed set (50:1 → 1:1, coverage preserved).

**Prediction (committed): probe 0.762–0.780 and held-out 0.765–0.785** — R4's probe *and* R3's
held-out, because the two effects are separable and balance is what held-out species need.
**Falsified if held-out lands below 0.7550** (R4 + one floor), which would mean the held-out loss
tracks unique coverage or label noise rather than distribution shape, and the balance reading above
is wrong.

If R5 lands as predicted, the recommended recipe becomes: **no confidence gate, balanced
replication** — one fewer hyperparameter than the pipeline started with, and the surviving one is a
rebalancing knob the project already understands.
