# The replication sweep: my justification for replicating was wrong, and 13× was the worst choice I could have defended

**Kind:** research · **Status:** **RESOLVED (2026-08-04).** Replication is **not** required — 12,230
pseudo-labelled images at **0.39 %** of training already buy +4.42 pt, essentially all of B3's gain.
The curve peaks at **2 %** (**probe 0.7706**, the project best) and falls away on both sides. And the
*transfer* to unseen species falls monotonically with replication, from 121 % at 1× to 39 % at 26 % —
so replication converts adaptation into specialisation. B3's 13× sits on the wrong side of both.

## Result

B3's config, four values of the pseudo-row share. Nothing else differs.

| arm | share of training | **probe** | **probe held-out species** | probe vs B1 | transfer |
|---|---|---|---|---|---|
| B1 — no trap data | 0 % | 0.6912 | 0.6974 | — | — |
| **1×** | 0.39 % | 0.7354 | 0.7508 | **+4.42** | **121 %** |
| **5×** | 2 % | **0.7706** | **0.7704** | **+7.94** | **92 %** |
| 13× (B3) | 6.1 % | 0.7370 | 0.7231 | +4.58 | 56 % |
| 26× | 10 % | 0.7159 | 0.7042 | +2.47 | 39 % |

*transfer* = (held-out-species gain) / (probe gain). Floors: probe 0.0041, held-out 0.0052.

## Scoring the prediction — the shape was right, every level was wrong

Committed: *"probe rises with the fraction and then flattens or falls as memorisation sets in — 1× near
B1's 0.6912, 5× ~0.72, 26× ≤ 13×'s 0.7370."*

| arm | predicted | actual | |
|---|---|---|---|
| 1× | ~0.6912 | **0.7354** | badly wrong |
| 5× | ~0.72 | **0.7706** | wrong, and it is the peak |
| 26× | ≤ 0.7370 | 0.7159 | correct |

**The 1× miss is the one that matters, because it falsifies the reasoning that produced the whole
design.** `dev/066` carries this comment, written before the sweep:

> at that share the pseudo rows appear in roughly one batch in 250, and B3 would return a null result
> that says nothing about self-training, only that 0.4 % more data changes nothing

That is wrong. **One batch in 250 buys 4.42 points** — 97 % of what 13× buys, and with *better*
transfer to unseen species. The argument sounded like arithmetic and was actually an assumption: that
an intervention's effect scales with how often the model sees it. For target-domain data it does not,
because what those images supply is not *more gradient* but *a different region of input space* the
training set never covered. A handful of samples from a region is enough to tell the model the region
exists.

## The transfer column is the real finding

Replication does not merely stop helping past 2 % — **it changes what is being learned**:

| share | transfer to species adaptation never saw |
|---|---|
| 0.39 % | **121 %** |
| 2 % | 92 % |
| 6.1 % | 56 % |
| 10 % | 39 % |

Monotone, across the whole range. At low share the model generalises to unseen taxa *more* than to the
ones it was adapted on; at high share it is largely memorising the 346 species it pseudo-labelled.

So **the dilution I was trying to avoid was the mechanism**. Seeing a trap image once or twice teaches
the domain; seeing it thirteen times teaches the image. B3's headline (+4.58, 56 % transfer) was
measured at a setting that had already traded away nearly half the generalisation, and I chose that
setting on an argument I never tested.

## Consequences

**The recommended self-training setting is ~2 % of training, not 6 %.** At 2 %, probe 0.7706 and
held-out-species 0.7704 are *equal* — the gain transfers essentially entirely. That is the strongest
robustness result the project has, and it is also the cleanest: no memorisation signature at all.

**B6 and B7 are running at 13 %.** Both use `combined_b3_3lvl` (13×), chosen before this was known,
so both are running at the setting that costs ~3.4 pt on probe and half the transfer. They still test
what they were built to test — composition with scale, and composition with oversampling removal — so
they are worth finishing. But **whatever wins should be re-run at 2 %**, and that is now the top
follow-up rather than an afterthought.

**A methodological note for the paper.** A hyperparameter sweep that a reviewer would read as routine
turned a +4.58 pt result into +7.94 and doubled its transfer. It was queued only because the
memorisation risk was written into the design as something to control later — the kind of caveat that
usually stays a caveat. Writing it down is what caused it to be run.


---

## B6 lands, and confirms the sweep's verdict on itself (2026-08-05)

B6 = F1's config (198 M, with oversampling) + self-training at **13x / 6 %**, launched before the
sweep landed.

| | in-distribution | probe | probe held-out sp. |
|---|---|---|---|
| F1 — no target data | 0.9219 | 0.7209 | 0.7559 |
| **B6** — F1 + self-training @ 6 % | **0.9225** | **0.7699** | 0.7422 |
| B3rep5x — 20 M, self-training @ 2 % | — | 0.7706 | **0.7704** |

**Scale and adaptation do compose**: +4.90 pt on probe over F1, for nothing in-distribution
(+0.06, inside the floor). The composition prediction (probe 0.750–0.775) was correct.

But the sweep's warning holds exactly as stated. **B6 at 198 M and 6 % share scores 0.7699 on probe —
statistically identical to a 20 M model at 2 % (0.7706) — and is 2.82 pt *worse* on held-out
species** (0.7422 vs 0.7704). Ten times the parameters, and the only thing it buys is undone by
running the share too high.

That is the memorisation signature again: probe holds up, transfer degrades. It is the third
independent confirmation, and the first at 198 M, that **share matters more than capacity** for this
lever. B8 (the same composition at 2 %) is the run that should settle whether scale adds anything
once the dose is right.


---

## B8: at 198 M, the 2 % dose no longer beats 6 % (2026-08-05)

B8 is B7's recipe (198 M, no oversampling, self-training) at the sweep's optimum share of 2 %
instead of 6 %.

| | in-distribution | probe | probe held-out sp. |
|---|---|---|---|
| B7 — same recipe @ 6 % | 0.9050 | 0.7796 | 0.7712 |
| **B8 — @ 2 %** | 0.9060 | **0.7798** | **0.7816** |
| Δ | +0.10 | **+0.02** (0.05x floor) | **+1.04** (2.0x floor) |

**Predicted probe 0.790–0.815; landed 0.7798, below the range.** Falsified — though the
falsification line (0.7747, B3rep5x + one floor) was cleared, so the composition itself holds.

**The dose effect is capacity-dependent, and that is the finding.** At 20 M, moving from 6 % to 2 %
was worth **+3.36 pt** on probe (0.7370 → 0.7706). At 198 M it is worth **+0.02** — nothing. What
survives is the *transfer* benefit: held-out species improve by 1.04 pt, two floors, in the same
direction as at 20 M but a third the size.

This fits the memorisation reading rather than complicating it. Over-replication hurts because the
model memorises the repeated images; a 198 M model has enough capacity to memorise them *and* fit
everything else, so the probe cost disappears. What it cannot avoid is that memorised examples do not
generalise to new taxa — hence the residual held-out gap.

**So the sweep's recommendation narrows: the 2 % dose matters at small scale and is optional at
large.** B8 is nonetheless the model to prefer — same probe, better transfer, and less data to
prepare.
