# Is the 17-point deployment gap between B8 and P5 a property of the models, or of how confidence was read?

**Kind:** research · **Status:** **RESOLVED (2026-10-02): neither — the metric was rank-blind.**
"Useful answers" counts a correct genus answer the same as a correct species answer. Calibrated and
measured on held-out trap nights, **B8 names the species correctly on 74.7 % of images against P5's
66.0 %, while P5 gives *some* correct answer on 83.3 % against B8's 77.2 %**. Neither dominates, and
uncalibrated B8 answers almost everything at genus and "wins" at 87.1 %. O1's "P5 is 17 pt more
deployable, because of calibration" does not survive; the paper's §4.6a and P5-first recommendation
need rewriting. Original framing: O1 found B8 and P5 tied on probe macro-F1
and 17.3 pt apart on useful-answer rate under a 95 %-precision back-off policy, and the paper
(§4.6a, contribution list) attributes the gap to "confidence calibration, not discriminative power".
Two things found while building the paper's figures put that in doubt.

## What prompted it

1. **B8's confidences saturate.** In the probe predictions O1 used, **69.7 % of B8's species
   confidences are exactly 1.0** (75.8 % above 0.999999) (float32 softmax rounds to 1 once the top logit leads by more than
   ~17, and B8's margin head scales cosines by 30). P5 has none. A threshold cannot separate images
   inside a tie, and the genus and family marginals inherit the saturation, so backing off cannot
   recover them. That is a property of the float32 readout, not of what the logits know.
2. **"Calibration, not discriminative power" is not what a precision-targeted policy measures.** The
   policy's coverage depends on how well confidence *ranks* right answers above wrong ones, which any
   monotone rescaling preserves; calibration (ECE) is exactly what such a rescaling changes. Measured
   on the same files: ECE 0.104 (B8) vs 0.085 (P5), a small difference, while B8's species-level
   coverage at 95 % precision is *higher* than P5's when ties are handled honestly (0.79 vs 0.73).
3. **The release entry already re-measured part of it.** With B8's temperature (T = 1.91, fitted on
   validation) and thresholds fitted on half the trap nights and verified on the other half, useful
   answers were B8 77.2 %, P5 83.3 %, B3rep5x 85.8 %
   ([[2026-09-29-public-hf-release]]). It did not run B8 *without* the temperature under that
   procedure, so it could not split O1's 17 points into "procedure" and "readout".

## The test

The release's own code (`dev/083`: `fit_thresholds`, `cascade`, the same seed-0 split of trap
nights) on three inputs: B8 at T = 1 (O1's predictions), B8 at T = 1.91 (the published model), and
P5. The last two must reproduce 77.2 % and 83.3 %, or the harness is wrong.

## Predictions (committed before running)

* **B8 at T = 1 under the split procedure: 69-74 % useful.** Its species threshold lands at 1.0
  either way, so the procedure change should barely move it, unlike P5 (88.4 -> 83.3).
* So of O1's 17.3 points, **~5 are procedure** (in-sample fitting flattered P5) and **~6-8 are the
  saturated readout**, leaving the ~6 the release measured as the model difference.
* If B8 at T = 1 scores near 77 %, the saturation story is wrong and the temperature's gain is not
  what moved B8.

## Result (`dev/086_deployment_gap.py`, same code and split as the release)

The harness reproduces the release: B8 calibrated 77.2 %, P5 83.3 % useful.

| held-out trap nights, 95 % target | species answered (precision) | genus | family | abstain | **correct at species** | **correct at any rank** |
|---|---|---|---|---|---|---|
| B8, T = 1 (O1's readout) | 0 % (threshold never reached) | 85.8 % (96.3 %) | 4.6 % | 9.5 % | **0 %** | 87.1 % |
| B8, T = 1.91 (published) | 78.3 % (95.4 %) | 0 % | 2.6 % | 19.0 % | **74.7 %** | 77.2 % |
| P5, T = 1 (published) | 68.7 % (96.1 %) | 6.1 % | 11.9 % | 13.4 % | **66.0 %** | 83.3 % |

Species-level only, with ties kept together: coverage at 95 % precision B8 0.79 (T = 1) / 0.785
(T = 1.91) vs P5 0.72; AURC 0.042 / 0.036 vs 0.037; ECE 0.104 / 0.073 vs 0.087.

**Predictions scored.** "B8 at T = 1 scores 69-74 %": **wrong**, 87.1 %. The saturation story was
right about the mechanism and wrong about its effect: saturation did stop B8 answering at species
(no threshold on the grid reaches 95 % once the tie group is included), but the fitter then moved
every image to genus, where B8's marginals are also near 1 and 96 % correct, and the rank-blind
metric scored that as a gain. "~5 pt is procedure, ~6-8 readout": not decomposable in that form,
because the three arms answer at different ranks.

## What it means

* **There is no single "deployability" number here.** A 95 %-precision back-off policy produces a
  composition (species / genus / family / abstain), and models trade one rank for another. B8 is the
  better species identifier at high precision (+8.7 pt correct at species); P5 backs off more
  gracefully (+6.1 pt correct at some rank). Which is better depends on what a genus answer is worth
  to the user, and the paper has to say so rather than pick a winner.
* **Calibration matters, but not the way O1 said.** For B8 the temperature is what makes the species
  rank usable at all (0 % -> 78 % species answers). The claim "the difference is calibration, not
  discriminative power" conflated two things: precision-targeted coverage depends on ranking, which
  B8 does as well as P5; B8's saturated float32 readout is what broke the species threshold.
* **Corrections owed:** paper §4.6a, the abstract's deployability sentence and the P5 recommendation
  in §4.0; `RESULTS.md` finding 9 and O1's row; `README.md` "P5 (recommended)"; the P5 model card's
  "+17.3 pt useful-answer rate" (on Hugging Face, so the owner decides). The release entry already
  had the 6-point version and the same rank-blind metric.
* **For the figure:** the honest picture is the stacked composition per model, not a bar of "useful".

