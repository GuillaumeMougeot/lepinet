# T1: what real labels would have bought — and why the answer favours using none

**Kind:** research · **Status:** **REOPENED (2026-10-12): the headline below is retracted.** At
equal dose real labels beat 98 %-accurate pseudo-labels by **+2.14 pt**. The claim that
"self-training at its own best setting beats 12,230 real labels" (0.7706 vs 0.7568) compared
self-training at 11x with real labels at 1x: different doses. The dose-matched arm (T1rep) is
running; see *Correction* at the end. The held-out-species column is **not interpretable at this
sample size** and that is itself a finding (unchanged).

## Result

Every arm: B3's recipe at 20 M, target rows merged at their natural share with no replication unless
stated. Labels drawn from `adapt` only.

| arm | target images | labels | **probe** | probe held-out sp. |
|---|---|---|---|---|
| B1 — no target data | 0 | — | 0.6912 | 0.6974 |
| lab 500 | 500 | real | 0.7060 | 0.7505 |
| lab 2500 | 2,500 | real | 0.7196 | 0.6970 |
| **lab 12230** | 12,230 | **real** | **0.7568** | 0.7518 |
| self-train 1× | 12,230 | machine (98.15 %) | 0.7354 | 0.7508 |
| **self-train @ 2 % share** | 12,230 (×5) | machine | **0.7706** | 0.7704 |

Floors: probe 0.0041, held-out 0.0052.

## Scoring the prediction

Committed: *"N=500 probe ~0.70–0.72, N=2500 ~0.72–0.74, N=12230 ~0.735–0.755 — real labels beat
98 %-accurate pseudo-labels by little. If N=12230 lands above 0.77 that reasoning is wrong."*

| arm | predicted | actual |
|---|---|---|
| 500 | 0.70–0.72 | **0.7060** correct |
| 2500 | 0.72–0.74 | **0.7196** just below |
| 12230 | 0.735–0.755 | **0.7568** just above |

Three near-misses in the right places, and the qualitative claim — *labels beat pseudo-labels by
little* — needs one correction: **+2.14 pt is not "little"**, it is 5.2× the floor. Label quality
does matter. What is true is that it matters **less than the dosage**, which is not what I predicted
and is the more useful finding.

## The two comparisons that answer the reviewer

**Matched comparison (same 12,230 images, same share, only the labels differ):** real 0.7568 vs
machine 0.7354. **Real labels are worth +2.14 pt.** So the pseudo-labels' 1.85 % error rate costs
something real, and anyone claiming pseudo-labels are as good as labels is wrong.

**Best-versus-best:** self-training at its own optimum (2 % share) scores **0.7706**, beating 12,230
*real* labels at 1× by **+1.38 pt** (3.4× floor). The label-free method wins because it has a free
hyperparameter — the share — that the labelled arm cannot exploit as cheaply: replicating 12,230
hand-labelled images to 2 % is possible, but you had to pay for those labels first.

**The practical reading, which is the answer to give a reviewer:** 12,230 labels buy less than
choosing the right dose of zero labels. Labelling is worth doing only if you can afford **more than
12,230 images**, and the curve's shape (+1.48 at 500, +2.84 at 2,500, +6.56 at 12,230) says the
return is still rising at that point — so the honest statement is *"labels help, super-linearly in
this range, and are not competitive with correctly-dosed self-training until well past 12 k."*

## The held-out column is not interpretable here, and that is worth stating

Read down it: 0.7505, **0.6970**, 0.7518. lab500 beats lab2500 by 5.35 pt — ten times the floor —
while their probe scores are correctly ordered. Something is wrong with using this column for these
arms.

The cause is structural. **The 75 held-out species are, by construction, absent from `adapt`**, so
*every* label budget contains exactly zero labels for them. Differences in that column are therefore
second-order effects of what the model learned from *other* species, measured over 2,455 images and
58 classes. The 0.0052 floor was measured on two draws of an identical config; it bounds run-to-run
noise, **not** the sensitivity of a 58-class macro-F1 to which unrelated species happened to be
labelled.

So: the held-out column did its job for self-training, where the question was *does the gain reach
species the adaptation never saw* and the answer was consistent across four arms. It cannot do the
same job here. **Reporting it for these arms would be reporting noise with a number attached** —
recorded rather than quietly dropped, and it is why the probe column carries the conclusion.

## What this leaves

**The no-label position is stronger than before, not weaker.** It survived a fair test — matched size,
matched share, real labels — and won on the axis that matters once each method is allowed its own
best setting.

**Still open (T2, T3):** whether *fine-tuning* on labels beats *mixing* them, which is the other half
of the reviewer's question and needs an `init_from` option the package lacks; and the cost side,
which is arithmetic rather than GPU time.

## Correction (2026-10-12): the "best versus best" comparison was not best versus best

The owner asked how machine labels could beat human labels when one person annotated both the
adaptation images and the probe. They cannot, and this entry never showed it. Two errors, both
mine, both visible in the configs:

**1. The doses differed.** `combined_b3_5x` (B3rep5x, 0.7706) replicates its 12,230 pseudo-labelled
images **11x** (`--target-frac 0.02`; the "5x" in the name is stale). `combined_lab12230` (0.7568)
has each real-labelled image **once** (`--target-frac 0`). The argument above, that self-training
"has a free hyperparameter the labelled arm cannot exploit as cheaply", is wrong: replicating
images that are already labelled costs nothing. The project's own rule (RESULTS #13: *give every
arm its own best configuration*) was broken in the comparison that fed the abstract.

**2. "Same 12,230 images" was false.** The pseudo arm holds the 12,230 images that cleared the
confidence gate (`dev/065`, 346 species): the teacher's easiest images. The real arm holds 12,230
images drawn round-robin over all 1,304 capture groups (`dev/067`, 352 species), hard ones
included. So the matched-dose +2.14 pt is label quality **plus** image selection: an upper bound
on what the 1.85 % label noise costs, not a measurement of it.

**What the single annotator does to the comparison.** Their labels on `adapt` only measured
pseudo-label accuracy; the gate never reads them. A shared annotator favours the *real-label* arm
(it learns the conventions the probe rewards), so it cannot explain a pseudo-label win; if anything
it inflates the real-label lead.

**What stands without the claim:** +7.94 pt on probe with no target labels, the gain transfers to
unseen species (held-out 0.7704), nothing lost in-distribution, 20 M with adaptation beats 198 M
without, and the sharp dose optimum.

**T1rep, committed before it runs.** The 12,230 real labels at 11x (`configs/20261012_T1rep_
lab12230rep11x_effnetv2s.yaml`, otherwise lab12230 exactly; job 12420216, probe and probe-held-out
queued after it). Package code is unchanged since August apart from docstrings, so 0.7706 is a valid
reference. **Prediction: probe 0.78-0.79**: real labels gain from the dose at least as much as
pseudo-labels did (+3.5 pt from 1x to 11x), minus some memorisation, since the dose optimum was found
for pseudo-labels. Falsified if probe <= 0.7747 (0.7706 + the 0.0041 floor), which would mean the
dose favours pseudo-labels specifically, plausibly because the gate's easy images tolerate
replication better.

**Rewritten:** abstract, §4.11, §5 "what we do not claim", RESULTS #4 and the T1 row.

**Found on the way, not yet fixed:** the share labels. Training is sets 2-9 after the 50-image floor,
5,039,220 rows, so 12,230 rows are ~0.24 % (not 0.39 % of "~3.1 M") and 134,530 are ~2.6 % (not 2 %:
`--target-frac` is computed against the whole parquet, test fold included). The ordering and the
interior optimum are unaffected; the numbers quoted in §4.11 need relabelling.
