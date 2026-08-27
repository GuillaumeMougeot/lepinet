# G3b: the noise floors were measured in one training regime and do not transfer to another

**Kind:** research · **Status:** **RESOLVED (2026-08-27). Badly falsified, and it retracts two
conclusions.** An exact repeat of G3 — same config, same checkpoint, nothing changed — scored
**held-out 0.7892 against G3's 0.7518**. That is a **3.74 pt** spread between two draws of an
identical configuration, **7.2x the floor** this project has been using for that benchmark. The
"balance is a trade at 198 M" finding and the "end-to-end leads at 198 M" finding both die.

## Result

| | G3 | **G3b (repeat)** | spread | vs the floor in use |
|---|---|---|---|---|
| in-distribution | 0.9138 | 0.9148 | 0.0010 | 2.0x |
| probe | 0.7740 | **0.7870** | 0.0130 | **3.2x** |
| held-out species | 0.7518 | **0.7892** | **0.0374** | **7.2x** |

**Predicted held-out 0.745–0.760, falsified above 0.7652. Landed 0.7892** — falsified by 2.4 pt, the
second-largest prediction miss of the project after R2.

Note the shape: **in-distribution is quiet (0.0010) while the shifted metrics are wild.** The
representation is identical in both runs; only the 2-epoch classifier stage differs.

## Why the floors did not apply

They were measured honestly and they are not wrong — they were measured **in a different regime**:

| | floor measurement | G3 / G3b |
|---|---|---|
| model | 20 M effnetv2_s | 198 M ConvNeXt-L |
| training | **5 epochs, end-to-end** | **2 epochs, frozen trunk** |
| source | [[2026-08-01-how-noisy-are-our-numbers]], [[2026-08-03-macro-f1-does-not-decompose]] | this run |

A 5-epoch end-to-end run takes ~400 k gradient steps over the whole network, and run-to-run
differences average out. A **2-epoch frozen-trunk stage** touches only the head, has far fewer
effective updates, and its outcome depends much more on the free-running sampler order and head
initialisation — `lepinet` seeds nothing. There is simply less averaging, so more variance.

The original entry even flagged the danger and stopped one step short of it: *"the full-set floor was
measured on the full set and does not transfer to a smaller one, so probe and probe-heldout floors
were measured the same way."* The authors of that sentence (me) generalised across **benchmarks**
correctly and then failed to generalise across **training regimes** at all. A floor is a property of
(metric x benchmark x training procedure), and we had been treating it as a property of the benchmark
alone.

And the class-count argument predicts exactly where it would bite hardest: held-out species has **58
classes**. By the same reasoning that makes 102-class family macro-F1 25x noisier than 12,041-class
species, a 58-class benchmark is the noisiest thing we report — and it is the one carrying the
smallest margins.

## What this retracts

**1. "Balanced replication is a trade at 198 M" is dead.**
[[2026-08-10-balance-is-oversampling-and-it-does-not-scale]] rested on G3 vs G2: probe +0.92
(2.2x floor) and held-out −0.82 (1.6x floor). Both are **inside** the newly measured spread of
0.0130 / 0.0374. With G3b substituted the same comparison gives probe +2.22 and held-out **+2.92** —
the opposite sign on the axis the conclusion was built from. The truth is we cannot tell from n = 1.

**2. "End-to-end leads at 198 M" is dead**, and it was the correction I made on 24 August.

| staged draw | vs B8, probe | vs B8, held-out |
|---|---|---|
| G3 | **−0.58** | **−2.98** |
| G3b | **+0.72** | **+0.76** |

The staged recipe is either behind or ahead of end-to-end at 198 M depending on which of two
identical runs you pick. **The honest statement is that they are indistinguishable at 198 M given
n = 2 staged and n = 1 end-to-end.**

That also partially restores the 20 M picture: staged and end-to-end were within noise there too
(−0.14 probe). The story is no longer "capacity decides it" — it is "we never had the resolution to
say."

## What survives

- **The in-distribution advantage of the staged recipe.** +0.71 at 20 M, +0.78 (G3) / +0.88 (G3b) at
  198 M, on a metric whose spread here is 0.0010. Consistent in sign and size across four runs.
- **Everything resting on margins of 3 pt or more**: self-training's +7.94, the R2 collapse (−3.80),
  the R3 gate recovery (+4.24), the B9 end-to-end damage (−3.62 held-out), H4's −4.25.
- **All in-distribution conclusions.** The 0.0010 spread here matches the original floor.

## What is now suspect and needs n >= 2 before it is quoted

Anything in a **frozen-trunk regime** with a margin under ~3 pt on a shifted benchmark. That includes
several 20 M results, because F2, F3, R3, R4, R5 and T2 are *all* frozen-trunk stages while the floor
was measured on end-to-end runs:

| comparison | margin | status |
|---|---|---|
| F3 vs F2, probe | −0.62 | already reported as a near-wash; unaffected in substance |
| R5 vs R3, probe | +1.07 | **suspect** |
| R5 vs B3rep5x, probe | −0.14 | already reported as a tie |
| G3 vs G2, both axes | ≤0.92 | **retracted above** |

**Measured, same day (R5b).** An independent repeat of R5:

| | R5 | R5b | spread |
|---|---|---|---|
| probe | 0.7692 | **0.7573** | **0.0119** |
| held-out | 0.7781 | **0.7702** | 0.0079 |

Predicted probe within 0.010 (missed by 0.0019) and held-out within 0.015 (correct); the 0.020
falsification line cleared. **The probe spread is ~0.012 in both frozen-trunk regimes** -- 0.0119 at
20 M and 0.0130 at 198 M, strikingly consistent, and ~3x the floor we had been quoting. Held-out is
the axis that blows up with capacity: 0.0079 at 20 M against 0.0374 at 198 M.

**Two further retractions follow:**

- **"R5 beats R3 on probe" (+1.07) is inside the spread.** With the R5b draw, R5b (0.7573) sits
  *below* R3 (0.7585). Not established.
- **"Balance is worth +1.51 probe to a frozen trunk" (F2 -> R5) is not established** either: with
  R5b it is +0.32, against a two-sided spread of ~0.017.

**What survives:** self-training's +7.94, R2's -3.80, R3's +4.24, R4-vs-R5 held-out (+3.23 with R5,
+2.44 with R5b), and B9's -3.62. The core claim -- coverage dominates label quality, and no
confidence gate is needed -- is untouched. What does not survive is the finer probe-side claim that
*balancing specifically* helps a frozen trunk.

## The rule

**A noise floor is a property of (metric x benchmark x training procedure), not of the benchmark.**
Re-measure it whenever the procedure changes — different capacity, frozen vs unfrozen, different
epoch count. And prefer to state margins in points with the spread beside them rather than in
"x floor" units, because that notation hides which floor is being invoked.

Added to `docs/design-decisions.md` beside the companion rule from 10 August, which asked for two
capacities and should have asked for two *draws* as well.
