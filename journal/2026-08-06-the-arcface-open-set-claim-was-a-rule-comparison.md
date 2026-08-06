# The headline open-set claim compared ArcFace's best rule against the plain head's worst

**Kind:** research · **Status:** **RESOLVED (2026-08-06) — and it retracts the project's most-cited
number.** "ArcFace × z-score turns novelty detection from chance into usable, 0.601 → 0.9115" is not
a comparison between two heads. It is a comparison between two *scoring rules*. Give each head its
best rule and the gap is **0.78 pt**, not 31 — while the plain head is **1.0 pt better** on accuracy.

## The measurement

Both heads, same benchmark, same images, `dev/061`'s five rules on the logits each head actually
emits:

| rule | plain cosine | ArcFace × z-score |
|---|---|---|
| **entropy** | **0.8990** | 0.9047 |
| msp | 0.8819 | 0.8953 |
| margin | 0.8423 | 0.8979 |
| **max-logit** | **0.6258** | **0.9068** |
| energy | 0.6149 | 0.9064 |
| **best** | **0.8990** | **0.9068** |

**Accuracy:** plain 0.9135, ArcFace 0.9035.

So with each head scored at its best: ArcFace leads open-set by **0.78 pt** and trails accuracy by
**1.00 pt**.

## How the 31-point claim happened

`dev/052` — the first open-set script, written before any of the scoring-rule work — computes
**max-logit only**. Every early open-set number in this project came from it, including the plain
head's 0.601. When `dev/061` was written on 2026-08-01 to compare five rules, it was run on **A1, B1,
A2, B4 and A4** — every ArcFace variant, and never on the plain head, because the plain head's number
was "already known".

So the headline compared ArcFace at its best rule against the plain head at its **worst**. Max-logit
is the plain head's worst rule by 27.3 points, and it is the one number the project had for it.

This is the same failure as [[2026-08-01-the-scoring-rule-was-the-bug]], which established exactly
this hazard five days ago and fixed it for the capacity comparison. It was not applied backwards to
the claim that predated it — the number was old, load-bearing, and never re-derived. **A baseline
that everyone quotes is the one least likely to be re-measured.**

## A second correction: my clamp hypothesis was backwards

This morning's measurement found the plain head pins 67 % of its logits to the z-score clamp, and I
hypothesised that this destroyed the shape novelty detection reads. It does the opposite:

| | clamped (as the model emits) | pre-clamp |
|---|---|---|
| entropy | **0.8990** | 0.5991 |
| msp | **0.8819** | 0.7637 |

The clamp is worth **+30 pt** to entropy. Pinning the tail to a floor evidently *removes noise* from
the shape rather than removing signal — an accident of the transform that turns out to help. Stated
because I wrote the opposite this morning with a confident mechanism attached.

## What this retracts, and what survives

**Retracted:** "ArcFace × z-score turns novelty detection from chance into usable (0.601 → 0.9115)
for −0.4 pt accuracy." The plain cosine head detects novelty at **0.8990**; it was never near chance.

**Also affected, and not yet re-measured** — every open-set comparison that used `dev/052` for the
plain head:

- **C3, stratified novelty** (near/mid/far): ArcFace 0.849/0.909/0.941 vs plain 0.561/0.618/0.666.
  The plain numbers are max-logit.
- **Flemming shifted open-set**: 0.727 vs 0.574. Same.

Both are re-scoring now. Until they land, neither comparison should be quoted.

**What survives, and it is not nothing:**

- **ArcFace's best rule is `max-logit`; the plain head's is `entropy`.** That is itself a real
  finding: the margin puts the open-set signal into the *magnitude* of the top score, where a plain
  cosine head keeps it in the *shape* of the distribution. The margin does change the geometry — it
  changes where the information lives, not how much there is.
- **ArcFace is more robust to the choice of rule**: its five rules span 0.8953–0.9068 (1.2 pt), the
  plain head's span 0.6149–0.8990 (28.4 pt). For deployment that matters — a model whose open-set
  score is insensitive to the readout is one you can ship without tuning the readout.
- The stratified and shifted results may still favour ArcFace once re-scored. They are the harder
  benchmarks and the margin may earn its place there.

**What it costs the paper.** §4.3 is titled "ArcFace × z-score: the trade-off dissolves" and rests on
+31 pt for −0.4. That framing is gone. What can honestly replace it is narrower and more interesting:
the margin relocates open-set information from distribution shape to top-score magnitude, and makes
the readout choice nearly free. Whether that is worth 1 pt of accuracy is a judgement the paper
should present rather than assume.
