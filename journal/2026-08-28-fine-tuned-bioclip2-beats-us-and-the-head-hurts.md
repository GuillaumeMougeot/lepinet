# P3 and L7: BioCLIP-2 was never under-optimised, and our head cut should have been deeper

**Kind:** research · **Status:** **RESOLVED (2026-08-28) for the in-distribution axis; P3's shifted
arm is running.** Two results that point the same way. **Fine-tuned BioCLIP-2 beats our task-trained
baseline by 1.25 pt** on a decontaminated fold — the 5.77 pt frozen deficit was a *readout* problem,
not a representation problem, and unfreezing is worth **+7.02 pt**. And the L7 sweep finds an
**interior optimum at cap 1,000**: capping harder than we do buys **+1.57 probe and +3.26 held-out**
for 0.88 pt in-distribution.

## P3: the owner's hypothesis, falsified

The hypothesis was that BioCLIP-2 under-exploits its 200 M images because its training lacks our
engineering, and that our recipe on ToL-scale data would beat it. P3 gives our recipe **their
backbone** — identical to the 0.9148 baseline config except `model_arch_name: bioclip2` and the
224/256 resolution, backbone unfrozen.

Decontaminated fold, 219,048 images / 11,998 species:

| | species macro-F1 | vs our baseline (0.9021) |
|---|---|---|
| P1a — frozen trunk | 0.8444 | **−5.77** |
| P3a — fine-tuned, lr 1e-3 *(our default)* | 0.8912 | −1.09 |
| P3b — fine-tuned, lr 1e-4 | 0.9025 | +0.04 |
| **P3c — fine-tuned, lr 1e-5** | **0.9146** | **+1.25** |

**Unfreezing is worth +7.02 pt.** The representation was fine all along; a cosine head reading frozen
CLIP features simply cannot get at it. That is the confound I flagged when P1a landed and it turned
out to be the whole story.

**And the three-arm gamble paid for itself.** Learning rate spans **2.34 pt**, monotone, with our own
default the *worst* of the three. A single run at `base_lr: 0.001` would have returned 0.8912, sat
1.09 pt below baseline, and been read as "BioCLIP-2 is behind" — the exact wrong conclusion, from a
plausible-looking number with no crash to warn us. The reason for three arms was written down before
the runs: *"that rate can destroy a pretrained ViT-L's representation in the first few hundred steps
— a failure that looks like a plausible-but-bad number rather than a crash."*

**What this closes.** Retraining a backbone on ToL-200M to beat BioCLIP-2 is not worth doing. They
already have the better representation; we can have it for the cost of a fine-tune. The ~84 M-image
download would buy us a worse version of something downloadable.

**The shifted arm landed too, and it is stronger than the in-distribution one.**

| | in-distribution | probe | held-out |
|---|---|---|---|
| our baseline | 0.9021 | 0.6270 | 0.6412 |
| **P3c — fine-tuned BioCLIP-2** | **0.9146** | **0.6630** | **0.6937** |
| Δ | **+1.25** | **+3.60** | **+5.25** |

**Fine-tuned BioCLIP-2 beats our trunk on all three axes, and by the largest margin on held-out
species** — the hardest and most deployment-relevant one. Against P1b's frozen 0.5901 it is +7.29 pt
on probe *without any adaptation stage at all*.

## This substantially revises finding 7d, which is four days old

P1b concluded the recipe is "not trunk-agnostic — the backbone stays in the contribution list", and
that is still literally true: the backbone matters a great deal. But the implicature was that **ours**
is the one worth keeping, and that is now wrong. Theirs is better on every axis once fine-tuned.

The accurate statement is: **the representation matters enormously, and the best available one is not
ours.** Our contribution is the recipe around it — the classifier stages, self-training, abstention,
open-set scoring — not the encoder. That is a *better* position for the paper than the one it
replaces, because the recipe is what transfers to other groups and other taxa; a backbone is not.

It also explains P1b without contradicting it. Frozen, BioCLIP-2's features are unreadable by a
cosine head (−16.14 under shift). Unfrozen, they are the best features we have. Both facts are about
the same encoder and only one of them is about its *quality*.

## What is still missing, and it is running

P3c has **no adaptation stage**. Our adapted models sit at probe 0.7515 (T2b) to 0.7706 (B3rep5x),
still ahead of P3c's 0.6630. So the ranking today is:

    our baseline 0.6270  <  P3c 0.6630  <  T2b 0.7515  <  R5 0.7692  <  B3rep5x 0.7706

**P4** adds our 2-epoch adaptation stage on top of P3c's trunk — the full recipe on the better
backbone. Predicted probe **0.76–0.80**, falsified below 0.7400. If it lands high the two effects
compose and we have a new best model; if it lands near T2b they are substitutes, which would say
adaptation and better pretraining buy the same thing, and that is the more interesting result.

## L7: the head hurts, and it hurts exactly where we care

| cap | train images | in-distribution | probe | held-out |
|---|---|---|---|---|
| 250 | 2.13 M | 0.8783 | 0.5776 | 0.6248 |
| 500 | 3.18 M | 0.8955 | 0.6281 | 0.6371 |
| **1,000** | **4.49 M** | 0.9060 | **0.6427** | **0.6738** |
| uncapped (~2,000) | 5.70 M | **0.9148** | 0.6270 | 0.6412 |

**In-distribution rises monotonically with the cap. The shifted axes peak at 1,000 and then fall.**

cap 1,000 against our current policy: **+1.57 probe (3.8x floor), +3.26 held-out (6.3x floor)**, for
**−0.88 in-distribution**. All three are well outside the end-to-end floors.

So head images beyond ~1,000 per species buy accuracy on the axis this project has already declared
saturated, and *cost* accuracy on the two axes that describe deployment. That is the same shape as
the self-training dose curve — an interior optimum, with the metric that looks like progress
continuing to rise past the point where the useful metrics turn over.

**It also settles the ToL-data question from the other direction.** Acquiring more head images would
move us further right on a curve whose shifted axes are already declining. The 84 M-image download is
now doubly unattractive: P3 says we do not need a better representation, and L7 says more head data
would make the shifted numbers worse.

**And it says our own cut was not deep enough.** The construction capped at ~2,000 on an untested
intuition about balance. The intuition was right and the number was wrong: 1,000 is better on both
deployment axes.

## Prediction scoring

**L7:** the qualitative call was right — in-distribution rises monotonically and is still rising at
1,000 — and the falsification line (cap 1,000 within 0.15 pt of uncapped) was not tripped, at 0.88 pt.
But **all three point ranges were too optimistic**: predicted 0.885–0.900 / 0.900–0.910 / 0.908–0.915,
landed 0.8783 / 0.8955 / 0.9060, every one below its range. A consistent bias worth naming: I
under-estimated how much in-distribution macro-F1 depends on sheer image count.

I explicitly declined to predict the shifted axes, writing that macro-F1 weights rare species equally
so head images might buy little there, but that P1b made representation quality look decisive. That
hedge was correct not to resolve — and the interesting result landed precisely in the part I refused
to guess.

**P3:** no range was committed for the arms individually, only the decision rule
(*">= 0.9021 means the representation was fine"*). P3c cleared it.

## What follows

1. **P3c on probe/held-out** — running. The only remaining difference between the backbones.
2. **The recommended cap becomes 1,000, not 2,000**, pending a repeat: this is n = 1 per arm, and
   [[2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes]] is a recent reminder that
   a 1.57 pt shifted difference deserves a second draw before it goes in the paper. These are
   end-to-end runs, so the end-to-end floors apply and 3.8x/6.3x is comfortable — but the floors
   themselves were n = 2.
3. **The ToL download is off the table** unless P3c's shifted arm reverses the picture.
