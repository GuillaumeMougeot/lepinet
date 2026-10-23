# What is the best back-off policy, stated so that it does not depend on moths or traps? (Q)

**Kind:** research · **Status:** **OPEN** (2026-10-23). Design and hypotheses; no measurement yet.

## The question (the owner's)

The release's back-off policy never answers at genus for B8 and B3rep5x on trap images; it goes from
species straight to family ([[2026-10-23-does-the-model-know-when-a-trap-image-is-too-poor]]). Is that
a property of the data, or of the policy? Which policy is best, if it has to work for any taxonomy and
any deployment, not only Danish moth traps?

## Why genus disappears (measured before opening, `data/hf_release/*/eval_probe.parquet`)

The current policy is greedy: a per-rank threshold at 95 % precision, each fitted on the images that
failed the rank below (paper §4.6 and `dev/058` for why conditional
fitting matters). On the probe crops that fail B8's species bar (3,536 of 15,200), the genus answer is
right 54 % of the time, and only 82 % at the highest genus confidences (6 % of those crops reach 0.95).
B3rep5x: 53 %, rising to 95 % only on 1.7 % of them. P5: 65 %, 94 % at 0.99. So a 95 % genus target is
out of reach for the margin-trained models: when the species is uncertain, the look-alikes are often in
different genera, and the genus posterior inherits the uncertainty. The family posterior aggregates
enough to reach 95 %. Nothing is broken; the policy's one constant (the same 95 % at every rank) makes
genus unreachable.

## Design: policies that need no dataset-specific tuning

All work from the per-rank posteriors a marginalising head already gives (the species posterior summed
up the taxonomy), so they apply to any hierarchy.

| policy | rule | knobs |
|---|---|---|
| P0, current | greedy, per-rank threshold at precision target t, fitted on the images that reach the rank | t (one), fitted per rank |
| P1 | P0 with a lower target at intermediate ranks | t per rank (sensitivity only) |
| P2, expected utility | answer the rank r maximising u_r p_r - c (1 - p_r), where p_r is the top posterior at rank r; abstain if all are negative ("hedging your bets", Deng et al. 2012 [VERIFY]) | utilities u_r (fixed by the user: what a genus is worth relative to a species) and one error cost c |
| P3 | P2 with c chosen on a calibration split to guarantee overall precision (risk control) | target precision (one) |

P2 needs calibrated posteriors; temperature per model is fitted on in-distribution validation data, as
shipped, and, separately, on target data with the split grouped by trap and night, to measure how much
calibration under shift matters.

**Evaluated everywhere, ranked consistently or not at all:** the in-distribution test fold
(Lepidoptera), the trap probe set (shift), Pl@ntNet (another kingdom and a 5-rank taxonomy, from the
colleague's predictions) and the novel trap crops of N2. Metrics: expected utility under two utility
vectors (species 1 / genus 0.5 / family 0.25, and species 1 / genus 0.8 / family 0.5), precision per
returned rank against its target, the composition of answers, and the rate of confident wrong species
answers on unseen species.

## Hypotheses (2026-10-23, before any measurement)

* **Q-H1. P2 (or P3) uses genus and beats P0's correct-answer rate at equal overall precision, on
  every dataset.** Falsified if P0 wins on any of the three datasets by more than 1 point.
* **Q-H2. Calibration is what limits P2 under shift.** With the in-distribution temperature, the
  per-rank precision on trap images misses its target by more than 3 points; with a temperature fitted
  on trap data it does not.
* **Q-H3. No back-off policy fixes unseen species in a known genus.** The rate of confident wrong
  species answers on near-novel trap crops stays above 30 % (it is 38 % under P0): that needs novelty
  detection, not a better back-off rule.

## Cost

CPU only, on per-image outputs that exist (B8, P5, B3rep5x on probe; in-distribution evaluation
parquets in `data/hf_release/`; C3b on all trap crops). Pl@ntNet needs the colleague's predictions.
