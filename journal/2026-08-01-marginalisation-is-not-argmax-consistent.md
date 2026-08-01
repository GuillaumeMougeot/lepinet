# "Consistent by construction" is false as stated

**Kind:** research · **Status:** **RESOLVED (2026-08-01) — the claim is mathematically wrong and is
repeated in eight places including the paper's method section.** Marginalisation guarantees
*probabilistic* coherence, not *argmax* agreement. A counterexample fits on one line. The empirical
rate is being measured; the wording is corrected everywhere now.

## The claim

From `paper/DRAFT.md` §2.4, and echoed in `README.md`, `START-HERE.md`, `docs/design-decisions.md`,
`dev/042`, `dev/050` and two journal entries:

> This is exact, adds no parameters, and is **consistent by construction**: the genus argmax can
> never contradict the parent of the species argmax.

## The counterexample

Found while writing a self-test for `MarginalArcFaceHead` — the test asserted argmax consistency,
and it failed. Two genera, six species:

| | | |
|---|---|---|
| species posterior | `[0.40, 0.12, 0.12, 0.12, 0.12, 0.12]` | sums to 1 |
| parent map | `s0 -> A`, `s1..s5 -> B` | |
| top species | **s0**, so its parent is **A** | |
| genus posterior | `A = 0.40`, `B = 0.60` | |
| genus argmax | **B** | **contradicts A** |

One confident species can be outvoted by many diffuse siblings. The sum is exact; the *argmax* of
the sum is a different quantity from the parent of the argmax, and nothing forces them to agree.

`max` and `Σ` do not commute over a partition. That is all this is, and it should have been obvious.

## What is actually true, and still worth claiming

The property marginalisation really has is **probabilistic coherence**: the reported genus
posterior *is* the sum of the species posterior, by definition. Independent heads have no such
relation — a multi-head model can report `P(genus = Noctuidae) = 0.9` alongside a species
distribution that puts 0.2 on Noctuid species, and there is no sense in which those are the same
model's beliefs.

That distinction is the real contribution, and it survives intact:

| | probabilistically coherent | argmax-consistent |
|---|---|---|
| independent per-level heads | **no** | no (measured: 1.81 % contradict) |
| marginalisation | **yes** | not guaranteed (rate being measured) |

Coherence is also what the *downstream* machinery needs. Rank abstention
([[2026-07-30-domain-shift]] §C1) thresholds `P(genus)` against `P(species)`, and that comparison is
only meaningful if both come from one distribution. Argmax agreement was never the property being
relied on — it was just the property that got written down.

**And when they disagree, the genus answer is probably the better one.** A diffuse species posterior
spread across one genus is *evidence for that genus*, which is precisely the aggregation
marginalisation exists to perform. So the disagreement rate is not an error rate; a low one indicates
confident species predictions, not a correct implementation.

## How this got into the paper

The phrase entered in [[2026-07-20-lepi-app-claude]] as an informal argument for why the app could
drop the coarse heads, was carried into `dev/042`'s docstring when that experiment confirmed the
*accuracy* claim it sat next to, and propagated from there. **The accuracy claim was tested; the
consistency claim never was** — it read like a definition rather than a result, so nothing in this
project's conventions caught it. `dev/042` even measures the independent heads' 1.81 % contradiction
rate, and it did not occur to anyone to run the same measurement on the marginal path, because that
path was "consistent by construction".

The same failure shape as [[2026-08-01-the-scoring-rule-was-the-bug]], one day apart: **something
that arrives phrased as a definition escapes the scrutiny applied to results.** Both were caught by
writing a test that took the claim literally.

## Actions

- **Wording corrected** in `paper/DRAFT.md` (§2.4 and the abstract), `README.md`, `START-HERE.md`,
  `docs/design-decisions.md`, `dev/042`, `dev/050`. Coherence is claimed; argmax consistency is not.
- **Measurement queued** (`lepi-consistency`, `dev/062`): the actual species/genus argmax
  disagreement rate on the marginal path, on the same fold `dev/042` used for the 1.81 %. Expected
  to be small — well under 1 % — but it is now a number rather than an assumption.
- **Convention added to `CLAUDE.md`:** a claim phrased as a definition still needs a test. If it
  cannot be falsified by an assertion, it is not a property, it is a hope.
