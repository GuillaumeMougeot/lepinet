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

## Result (2026-10-23, `dev/097`; Pl@ntNet pending)

Eleven fit/evaluation splits (trap images: halves of the (trap, night) groups, seed 0 as in dev/086
plus 1-10; in-distribution: random halves of the 629,742-image test fold). Means; the split-to-split
sd of the correct-answer rate is 0.4-2.8 points on traps, under 0.05 in-distribution. "Correct" =
share of all images that get a correct answer at the rank returned. P3 is tuned to 95 % overall
precision on the fit half, as P0 is per rank. Raw tables: `data/backoff/*.csv`.

Correct-answer rate at about 95 % precision (P0 against the best P3 variant):

| data, model | P0 (greedy) | P3, u = 1/.8/.5, shipped / target calibration | genus answers under that P3 |
|---|---|---|---|
| in-distribution, B8 | 94.4 % | 96.0 / 95.7 % | 1.3 % |
| in-distribution, P5 | 94.6 % | 96.4 / 96.6 % | 3.1-4.4 % |
| in-distribution, B3rep5x | 92.8 % | 94.7 / 93.9 % | 2.2-3.2 % |
| trap probe, B8 | **77.1 %** | 74.7 / 76.2 % | 0.2-0.5 % |
| trap probe, P5 | 84.5 % | 82.4 / 85.3 % | 5.7-7.8 % |
| trap probe, B3rep5x | **85.1 %** | 74.7 / 79.4 % | 0.8-1.2 % |
| trap, C3b (plain cosine head) | 79.3 % | **84.5** / 83.8 % | 22.5-25.8 % |

With u = 1/.5/.25 P3 is never better than this. P1 (a lower genus target) is the cheapest change:
P1-0.80 raises the correct rate on trap images by 1.5 (B8) and 2.1 points (P5) for 0.5-2.7 points of
precision.

* **Q-H1 is falsified.** The expected-utility rule beats the greedy one in-distribution (+1.1 to
  +1.9 points) and on trap images for the plain head (C3b, +5.2), ties for P5 on traps, and loses on
  traps for B3rep5x (-5.7) and B8 (-0.9 to -2.5). It uses genus only where the model's genus
  confidence differs from its species confidence: a quarter of C3b's trap answers, almost none of
  B8's.
* **Why it loses on margin heads under shift.** On the uncertain trap crops of B8 and B3rep5x the
  genus and family confidences are close to the species confidence (identical on 21-52 % of the crops
  that fail the species bar). With nothing to trade, expected utility always prefers species, so
  reaching 95 % means abstaining instead of backing off (family answers fall from 3-13 % to 0-4 %).
  The greedy policy's per-rank bars, each fitted on the images that reach the rank, find the family
  answers anyway.
* **A rule tuned on overall precision hides poor coarse answers.** P3 meets 95 % overall while its
  family answers are 0.70-0.72 precise in-distribution (B8, B3rep5x). P0 controls every rank.
* **Q-H2: the first half holds, the second does not.** Fitted in-distribution and applied to trap
  images, every policy misses 95 % by 6-11 points (P0: 0.871 B8, 0.860 P5, 0.895 B3rep5x). Recalibrating
  each rank on trap data does not rescue a cost fitted in-distribution: in-distribution the target is
  met at almost full coverage, so the fitted cost (0.01-0.24) carries no information. Setting the cost
  from theory instead (answer when the recalibrated confidence reaches 0.95, c = 19) overshoots to
  0.99 precision with 30-49 % abstention, because a per-image bar of 0.95 gives an average precision
  well above 0.95. Under shift the bar has to be fitted on target-domain images.
* **Q-H3 is falsified, narrowly.** Under P0, 37.6 % of the near-novel trap crops get a species answer
  (all wrong). P3 with u = 1/.8/.5 brings it to 26 %, by answering at species less often overall (33 %
  against 47 % of known crops). More than a quarter of an unseen species' crops still get a confident
  wrong name. On museum-style images it is worse: P0 names a species for 86-87 % of the 3,171 images of
  unseen species (B8, P5); P2 (c = 1) for 35-46 %.

**What it says so far.** No single rule wins everywhere, which is the generic answer: the rule that
compares ranks by expected utility needs coarse-rank confidences that carry information beyond the
species confidence, and margin-trained heads under shift do not provide it. The greedy per-rank policy
is the robust default; lowering the intermediate ranks' target is the cheapest way to use genus. The
expected-utility rule pays off in-distribution and for the plain head. Pl@ntNet will test whether the
in-distribution ranking holds for another kingdom and five ranks (`dev/097 --plantnet <predictions.csv>`).
