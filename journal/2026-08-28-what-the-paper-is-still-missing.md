# What the paper is still missing — a top-down audit of DRAFT.md against the project

**Kind:** infrastructure · **Status:** **OPEN (2026-08-28).** A full re-read of `paper/DRAFT.md`
against `PLAN.md`, `START-HERE.md` §2 and every journal entry. The draft's header says 2026-08-05 and
that is accurate: **it has not absorbed anything from 6-28 August.** The audit finds **eight places
where the draft asserts something the project has since retracted**, **six results with no home in
the paper at all**, and **one structural hole** — the long-tail/cRT section is cited six times as
"§4.x" and does not exist.

No new experiment is needed for most of it. This is a writing debt, not a measurement debt.

## Why this happened, which is worth recording

`CLAUDE.md` §6 says a finding that generalises gets a line in `START-HERE.md` **and** a section in
`paper/DRAFT.md`, in the same session. The first half was done every time: START-HERE has findings
4c, 7d, 7e, 7f, 7g, all added on the day. The second half was skipped every time.

The reason is that a START-HERE line is *additive* — you append a row — while a paper section is
*integrative*: it has to be placed, and placing it usually contradicts something already written.
The cheap half of the contract got done and the expensive half accumulated. **PLAN.md even warned
about this** ("Keep §4 in step with the journal -- it drifted twice by treating the journal as
primary"), and it drifted a third time.

## Tier 1 — the draft states things the project has retracted

These are correctness bugs in a document meant to be submitted. Ordered by how visible they are.

| # | where | what it says | what is true |
|---|---|---|---|
| 1 | **Abstract** | margin turns "near-chance novelty detection (AUROC 0.601) into usable detection (0.9115) for 0.4 points of accuracy" | Retracted **in §4.3 of the same document**. Best-rule vs best-rule it is 0.8990 -> 0.9068, i.e. **+0.78 pt for -1.00 pt of accuracy** |
| 2 | §4.5 | open-set-under-shift table uses cosine 0.601 / ArcFace 0.9115 | The same retracted pairing (plain head's *worst* rule vs margin head's *best*), now propagated into the shift result. **Needs re-scoring, or an explicit caveat** |
| 3 | §4.7 | "it buys open-set AUROC 0.601 -> 0.9068" | Half-corrected: the corrected numerator against the retracted denominator |
| 4 | §4.13 | the 198 M staged-vs-end-to-end rows, and "+0.78 / -0.58 / **-2.98** at 198 M" | Dead. G3b, an exact repeat of G3, scored held-out **0.7892 against 0.7518** -- a 3.74 pt spread between identical runs. The -2.98 is inside the noise [[2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes]] |
| 5 | §4.13 | "staged, balanced" beats "staged, natural" by +1.51 probe / +1.87 held-out at 20 M | Retracted by R5b: the 20 M **frozen-trunk** probe spread is **0.0119**, not 0.0041. +1.51 is 1.3x its floor |
| 6 | §4.11 | "Run-to-run floors [...]: probe 0.0041, held-out 0.0052" stated once and used throughout | Those are **end-to-end 20 M** floors. Frozen-trunk 20 M is 0.0119/0.0079; 198 M frozen-trunk is 0.0130/0.0374. A floor is a property of (metric x benchmark x **training procedure**) |
| 7 | §3.1 | "**Numerics.** The cosine head overflows in fp16; all runs use bf16." | **False as written.** 63 of 115 configs set `precision: fp16`, including the headline baseline `20260729_ucloud_singlehead_species_effnetv2s.yaml`. `docs/design-decisions.md` has it right: the 0.9148 run "survived fp16 only because the head was forced to fp32 inside an adapter". `config.py` rejects only **ArcFace + positive margin** in fp16 |
| 8 | §6 | "the harder novelty-plus-shift case is _(pending)_" | §4.5 reports it |

Item 7 was found by accident while cloning a config, and it is the one worth pausing on. The
compressed form of the rule — "bf16, never fp16" — is in `CLAUDE.md` as an invariant and in the
paper as a methods sentence, and **neither matches what the code does**. The uncompressed version in
`docs/design-decisions.md` is accurate. This is the documentation contract failing in the direction
it is supposed to fail safely: the summary drifted from the source and the summary is what gets read.

**A ninth, in the figures.** `paper/figures/README.md` describes `fig4_openset_scores.png` as "the
main open-set figure", scoring both heads by max cos-theta and reporting 0.607 vs 0.898. That *is*
the retracted comparison — max-logit is the plain head's worst rule by 27 pt. The figure is not
wrong, but its caption frames a readout artefact as a representation difference, and it is currently
the paper's headline open-set image.

## Tier 2 — results with no home in the paper

Confirmed by grep: **`BioCLIP`, `contamination`, `GBIF`, `TreeOfLife` each appear zero times** in
DRAFT.md. `cRT` appears once, inside a sentence pointing at a section that does not exist.

| what | where it lives now | why the paper needs it |
|---|---|---|
| **Group L / cRT and the imbalance bench** | findings 9, 10; [[2026-08-01-imbalance-methods-bench]] | **The structural hole.** Six "§4.x" cross-references point here. §4.13's whole first pillar is cRT. The monotone robustness/accuracy trade (0.6445 > 0.6293 > 0.5726 > 0.5492) is one of the cleanest results in the project and is unwritten |
| **Group P / BioCLIP-2** (P1a, P1b, P3, P4, P5) | findings 7d, 7e, 7g | The headline has changed. A reviewer's first question — "why not use a foundation model?" — is now answered with a measurement, and the answer reframes the contribution as **the recipe, not the encoder** |
| **Contamination** | finding 4c; [[2026-08-26-bioclip2-has-seen-two-thirds-of-our-test-fold]] | Arguably the most transferable thing here. 93.3 % of our species and **413,865 of our 629,742 test-fold images** are in BioCLIP-2's training set **by exact GBIF occurrence id**, not name match. It generalises to every paper benchmarking a foundation model on a public archive |
| **L7 / the head-cap interior optimum** | finding 7f | Same shape as the dose curve, and a second instance of "in-distribution points the wrong way past the optimum" |
| **Group H at ToL scale** | [[2026-08-27-tol-at-our-policy-and-the-head-scaling-problem]] | §4.12 has the low-rank and centroid halves but not the resolution: a min-image floor turns 884,662 species into 203,878 and the matrix then **fits**. The scaling problem is answered by a data policy, not an architecture |
| **Macro-F1 does not decompose** | finding 4a | Cited in the Discussion as "§4.x". It is the reason the paper needs three benchmark columns and it is never stated |

## Tier 3 — structural

- **Six literal `§4.x` placeholders** (lines 74, 91, 599, 629, 639, 721).
- **The contributions list C1-C6 predates the project's biggest results.** It has no entry for
  self-training (the largest lever measured), for cRT, for contamination, or for the
  classifier-vs-representation through-line that §4.13 and §5 now treat as the paper's spine.
- **§3 never defines the three benchmarks.** `full trap` / `probe` / `probe-HO` are defined
  mid-§4.11; the reader meets "external" in §4.2, nine sections earlier. Finding 4a says differences
  are only meaningful within a column, and the paper does not give the reader what they need to obey
  that.
- **"Held-out" means two different things** — probe-held-out-*species*, and C3b's deliberately
  withheld common taxa. Renaming the latter to "withheld" costs nothing now and is confusing later.
- **Four of five figures are never referenced in the text**, and `figures/README.md` documents only
  two of them. Only fig5 has a caption.
- **No main results table.** `START-HERE.md` §3 has one; the paper makes the reader assemble it from
  eleven sections.
- **The abstract predates four of the paper's now-central results** and quotes a retracted number.

## Tier 4 — measurement gaps a reviewer would actually find

These need GPU. They are listed in cost order, not importance order.

1. **P5 is n = 1 on the shifted axes** and carries "P5 ties B8, our best model". Two conclusions have
   already died on single shifted draws. *(Launched as P5b, 2026-08-28.)*
2. **Open-set is measured on no model we would ship.** §4.3, §4.4, §4.9 are all on 20 M/198 M
   task-trained trunks; §4.6's abstention is on the *old* single-head model. B8 and P5 — the two
   models the paper recommends — have no AUROC and no abstention curve. This is eval-only, no
   training.
3. **Open-set at taxonomic scale.** §4.9 says scoring rules do not transfer across *model* scale.
   The cached ToL embeddings make "and not across *class-count* scale either" measurable from 12 K to
   204 K taxa, embedding-only, no image reads and no download of the corpus.
4. **C3b is 20 M only** (backlog #6) — the novelty-is-not-rarity result is single-scale, which is the
   exact shape of the two claims retracted this month.
5. **Per-rank taxonomy validity on ToL.** 191 kingdoms and 1,265 phyla are not credible; "seven
   levels" cannot be claimed until someone measures per-rank validity. Cheap, data already scanned.

## The plan

**Phase 1 — correctness, no GPU.** Tier 1 items 1-8, and the figures/README caption. The paper
should not contain a number the project has retracted, and eight of them is not a rounding error.

**Phase 2 — the structural hole, no GPU.** Write §4.x properly as **§4.14 long-tail rebalancing
belongs to the classifier**, resolving all six placeholders. This is the section §4.13 is built on.

**Phase 3 — the new arc, no GPU.** A **§4.15 foundation models** covering contamination, the frozen
probe understating a representation by 7 pt, and the bounded adaptation claim. Then reconcile the
abstract, contributions and limitations with everything above.

**Phase 4 — GPU, in parallel throughout.** Tier 4 in the order given. (1) is running.

Phases 1-3 are one sitting each and unblock the owner's `[VERIFY]` pass, which should happen against
a draft whose *numbers* are settled, not one where they are still moving.

## Prediction (committed)

That the audit's Tier 1 is the whole of the correctness debt — i.e. that a systematic pass over
every number in §4 against `RESULTS.md` and the journals finds **no more than two** further
retracted or stale figures beyond the eight above. Falsified at five or more, which would say the
drift is not a matter of a few late results but that §4 has been out of step for longer than the
2026-08-05 header suggests.
