# P5: frozen-trunk adaptation only works on a trunk trained with the same head

**Kind:** research · **Status:** **RESOLVED (2026-08-28). Prediction correct, and it bounds one of
the project's central claims.** Unfreezing the adaptation stage on BioCLIP-2 is worth **+5.93 pt**
(probe 0.7218 → **0.7810**). So P4's crossover was a **frozen-readout artefact**, not substitution.
The consequence: *"adaptation is cheap because it can be done on a frozen trunk"* holds for **our**
trunk and not in general.

## Result

**Predicted probe 0.74–0.79, falsified below 0.7300. Landed 0.7810 — inside the range.**

| adaptation stage on BioCLIP-2 | probe | held-out |
|---|---|---|
| **frozen** (P4, n = 2) | 0.7218 | 0.7540 |
| **unfrozen** (P5) | **0.7810** | **0.7806** |
| Δ | **+5.93** | **+2.66** |

P4's two draws were 0.7199 / 0.7236, a spread of 0.0037, so the 5.93 pt gap is not a sampling
accident.

## The explanation that survives

Two hypotheses were on the table when P4 was falsified:

- **(a) substitution** — BioCLIP-2's pretraining already supplies what adaptation supplies, so less
  is left to gain. Predicts P5 ≈ P4.
- **(b) frozen-readout handicap** — a cosine head cannot read frozen BioCLIP features (P1b: probe
  0.5901, −16.14 vs ours), and P4's adaptation stage is a frozen-trunk stage, so it inherits that
  bottleneck.

**(b), decisively.** Unfreezing recovers almost the entire deficit.

**The mechanism is not mysterious, and it retro-explains T2b.** Our trunk was *trained with the very
cosine head we then re-fit*: its features are already arranged for that readout, so re-fitting the
head on frozen features works. BioCLIP-2's features are arranged for alignment with **text**. Freezing
them and attaching a cosine head asks the head to read a geometry it was never shaped for.

So the correct statement is not "adaptation is cheap" but:

> **Frozen-trunk adaptation is cheap when the frozen trunk was trained with the same head geometry
> you are re-fitting. On a foreign trunk it costs ~6 points, and the trunk must be unfrozen.**

That bounds T2b, F2, R5 and the whole staged-recipe story to trunks we trained ourselves. It does not
retract them — every number stands — but it removes an extrapolation the project had been making
implicitly since 6 August.

## Where this leaves the model ranking

| model | probe | held-out |
|---|---|---|
| **P5** — BioCLIP-2 fine-tuned + unfrozen adaptation | **0.7810** | 0.7806 |
| B8 — our 198 M, end-to-end, pseudo-labels | 0.7798 | **0.7816** |
| B3rep5x — our 20 M, end-to-end | 0.7706 | 0.7704 |
| R5 — our 20 M staged | 0.7692 | 0.7781 |
| P4 — BioCLIP-2 + frozen adaptation | 0.7218 | 0.7540 |

**P5 and B8 are tied** — +0.12 probe and −0.10 held-out, both far inside any plausible spread. So the
honest summary of the whole BioCLIP-2 investigation is:

- Before adaptation, their representation is **better** than ours (+3.60 probe, +5.25 held-out).
- After each is given its best treatment, the two **converge** to the same place.
- The advantage a 200 M-image encoder brings is real, and **our adaptation recipe closes it**.

That is a more useful result than either backbone simply winning. It says the ceiling here is set by
the target-domain data, not by the encoder — which is the same conclusion the replication sweep and
T2 reached from other directions.

## L7 cap 1000 survives its repeat

| | probe | held-out | in-distribution |
|---|---|---|---|
| cap 1,000 (n = 2) | **0.6446** ±0.0039 | **0.6706** ±0.0063 | 0.9064 |
| uncapped ~2,000 (n = 2) | 0.6270 ±0.0041 | 0.6412 ±0.0052 | **0.9148** |
| Δ | **+1.76** | **+2.94** | −0.84 |

Both differences are ~3x the combined spread, with n = 2 on **both** sides. **The recommendation
stands: cap at 1,000, not 2,000.** In-distribution pays 0.84 pt for it, which is the axis we have
repeatedly said is saturated.

## Scoring

- **P5: correct**, 0.7810 inside 0.74–0.79 (at the top edge).
- **P4: falsified** at 0.7199 against 0.76–0.80, and the repeat confirms it at 0.7236.
- **L7: qualitative call correct, all three point ranges too optimistic**, and the shifted result —
  which I declined to predict — turned out to be the finding.

Three predictions, one clean hit, one clean falsification, one half-right. The falsification (P4) was
the most productive of the three, which is now a pattern worth noting: R2, B9, H4 and P4 were all
falsified and all four changed what the project believes.
