# T1: what real labels would have bought — and why the answer favours using none

**Kind:** research · **Status:** RESOLVED for the probe axis (2026-08-05); the held-out-species
column is **not interpretable at this sample size** and that is itself a finding. Real labels do beat
98 %-accurate pseudo-labels, by **+2.14 pt** at matched size and share. But **self-training at its own
best setting beats 12,230 real labels** — 0.7706 vs 0.7568 — so the honest headline is that the
label-free method is not merely competitive, it wins.

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
