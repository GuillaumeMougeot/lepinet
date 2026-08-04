# B3: self-training is the largest robustness lever this project has found

**Kind:** research · **Status:** **RESOLVED (2026-08-04). Prediction beaten** — probe **0.7370**
against a predicted 0.695–0.715. Self-training on pseudo-labelled trap images buys **+4.58 pt** on
held-out trap groups for **+0.04 pt** in-distribution, and **56 % of that gain transfers to species
the adaptation never saw**. A 20 M model with it beats a 198 M model without it.

## Result

One factor against B1: the same backbone, the same trap augmentation, the same ArcFace × z-score
head. The only difference is 12,230 pseudo-labelled trap images (98.15 % label accuracy after the
strict gate), replicated 13× to ~6 % of training.

| | in-distribution | **probe** | **probe, held-out species** |
|---|---|---|---|
| B1 — 20 M, trap augmentation | 0.8999 | 0.6912 | 0.6974 |
| **B3 — B1 + self-training** | **0.9003** | **0.7370** | **0.7231** |
| Δ | +0.0004 | **+0.0458** | **+0.0257** |
| floor | 0.0000 | 0.0041 | 0.0052 |
| | negligible | **11.2× floor** | **4.9× floor** |

*(for scale)* F1 — 198 M, no trap data | 0.9219 | 0.7209 | 0.7559

**Committed prediction: probe 0.695–0.715, falsified below 0.6953.** Landed at **0.7370**, above the
range. Wrong in the useful direction, and worth being explicit about why the estimate was low: it
reasoned that pseudo-labels drawn from the model's own beliefs could only teach it what it already
knew. That reasoning is wrong, and the next section is why.

## Why it works, given the labels came from the model itself

The pseudo-labels are B4's own predictions, so they contain **no new label information**. Yet
training on them moves the shifted score by 4.58 pt. Three things are being supplied that the labels
are not:

**Target-domain pixels.** The model had never seen a trap image during training — only GBIF
specimens plus three hand-authored corruptions meant to imitate one (B1's `domain_aug: trap`). Real
trap frames carry the true nuisance distribution: actual sensor noise, actual backgrounds, actual
poses. B1 established that *naming* nuisances is worth ~4 pt ([[2026-07-30-domain-shift]]); B3
supplies them without anyone naming anything, and is worth about the same again.

**Selection, not just data.** The gate keeps the 45 % of images the model is nearly certain about, at
98.15 % label accuracy. So the model trains on target-domain images *where it is already right* —
which sharpens the decision boundary in the region of the target distribution it has a foothold in,
rather than teaching it new classes.

**This is the first rung that required no one to name anything.** B1 needed a human to guess that
blur, low light and JPEG artefacts were what differed. B3 needed only unlabelled images from the
deployment domain. That is the property that makes it worth more than its 4.58 pt: it transfers to
the *next* trap, the next country, the next camera, without repeating the guessing.

## Did it generalise, or specialise? Both, mostly the first

The risk stated for this direction was the model specialising on the ~500 trap species rather than
becoming robust. The held-out-species column exists to detect exactly that, and it was measured on
the baselines *before* B3 ran so the comparison could not be rationalised afterwards.

- Gain on probe overall: **+4.58 pt**
- Gain on the 58 species the adaptation never saw: **+2.57 pt** — **56 % of it transfers**

So the answer is nuanced and worth stating precisely: **most of the gain is genuine domain
adaptation, and a real minority is specialisation.** Both parts are large relative to their floors
(11.2× and 4.9×), so neither is noise. A single aggregate number would have hidden this entirely —
it would have reported +4.58 and implied it was all robustness.

## What it costs

**Nothing in-distribution**: +0.04 pt, inside the 0.0000 floor. Unlike oversampling
(+1.86 in-distribution / −1.52 shifted) and unlike the ArcFace margin, this is not a trade. It is the
first intervention in the project that improves the shifted axis for free.

## Adaptation beats scale

**B3 at 20 M scores 0.7370 on probe; F1 at 198 M scores 0.7209.** A model a tenth the size, with
access to unlabelled target-domain images, beats one that has ten times the capacity and none.

This is the sharpest form of a pattern that has now recurred all week
([[2026-08-01-capacity-x-augmentation]], [[2026-08-02-f1-flagship]]): **for cross-source robustness,
information about the target domain beats capacity, and it is not close.** Note the two are
complementary rather than exclusive — F1 still leads on held-out species (0.7559 vs 0.7231), so scale
buys generalisation to *unseen taxa* while adaptation buys generalisation to *unseen conditions*.
Combining them is the obvious next run.

## What this makes worth running next

1. **B3 at 198 M** — F1's config plus the pseudo-labels. If the two effects compose, this is the
   deployment model. On the week's evidence they should: they act on different failure modes.
2. **A replication sweep.** 13× on 12 k unique images is a memorisation risk that was stated in the
   design; with a result this size it now has to be controlled. 1×, 5×, 13×, 26×.
3. **A calibrated gate.** The confidence distribution was saturated (median 0.9999996), so "top 30 %"
   actually kept 45 %. D1's temperature scaling would give a gate with real resolution, and the gate
   is the one hyperparameter that decides label quality.
4. **A second round.** Self-training is usually iterated: pseudo-label again with B3 rather than B4,
   whose trap accuracy is now 4.58 pt higher. The gate's 98.15 % should rise, and with it the
   quality of everything downstream.
