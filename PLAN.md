# PLAN — what we are doing now

**Kind:** living · **Last updated:** 2026-10-21 · **Supersedes:** [2026-07-28-landscape-and-plan](archive/journal/engineering/2026-07-28-landscape-and-plan.md)

The one file about *today*. Experiment IDs resolve in [`RESULTS.md`](RESULTS.md); reasons are in
the linked journal entries; older versions of this file are in git history.

## 1. The goal: a submittable paper

Every experiment [`paper/DRAFT.md`](paper/DRAFT.md) uses is closed. What remains is scoping, writing
and two evaluation-only runs.

| # | step | who | state |
|---|---|---|---|
| 1 | **Choose the scope and venue.** 9 contributions, 16 results sections, 13 k words. Proposed spine: classifier-stage interventions (§4.15) + the evaluation protocol (three axes, scoring rules, contamination) + self-training (§4.11); margin (§4.3, §4.7) and head scaling (§4.12) to an appendix; TreeOfLife (W1-W3) a second paper | owner | open |
| 2 | Restructure §4 to ~6 sections; rewrite the abstract; write §1 (still an outline) | agent, after 1 | — |
| 3 | Re-score §4.5 with each head's own rule; measure open-set under shift for B8 and P5 | GPU, eval only | — |
| 4 | Verify every citation in §1b and the references (written from memory) | owner | open |
| 5 | Consistency pass: every number against its journal entry | agent | — |
| 6 | Re-run A2's scoring rules: `rules-A2.json` was overwritten on 2026-08-03; §4.9 uses the 2026-08-01 file | GPU, eval only | open |

## 2. Running now

| ID | what | state | next |
|---|---|---|---|
| **W1** | TreeOfLife-200M at short side 256, on the workstation (`/data/au761367/tol256s/`; zero UCloud core-hours) | crawl done: 64.16 M images plus ~0.54 M retried ([crawl](journal/engineering/2026-10-01-the-crawl-resized-the-wrong-side.md)); `retry3/` (9 servers, 72 h cap) | **upload paused on storage**: 28 of 1,048 shards up; ~550 more fit, the rest needs ~0.7 TB. Then merge `retry1/`/`retry3/` metas, upload the manifest, audit (`dev/085`) |

## 3. Backlog, in order

| ID | work | cost | why |
|---|---|---|---|
| U | self-training on EU trap crops (>50 M): natural vs balanced, then share sweep, teacher B8 | 4-7 runs at 20 M, after the crop location and metadata are known | the largest lever, without copies. [[2026-10-21-self-training-on-european-traps]] |
| W3 | training parquets for the ToL run: join the catalog by uuid, seven levels keyed by full path, check the hierarchy is a tree, drop the audit's 584 k excluded rows; then the 20 M run | GPU, after W1 | the owner's question: does seeing the whole tree help open-set and shift? Also gives a trap order classifier. [[2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know]] |
| S1a | two organisms in one picture: composites (moth on moth, moth on trap background) through B8 and P5; does the open-set score flag them as novel? Per-location logits as a readout | workstation GPU, hours | the owner's question (2026-10-09); also decides how W3 is read out (whole tree vs subtree). [[2026-10-09-two-organisms-in-one-picture]] |
| W2 | our objective trained on ToL-10M, compared with BioCLIP-1 on the same data | ~2.5 h/epoch | isolates the objective from the data. [[2026-08-28-two-directions-checklists-and-our-objective-on-tol]] |
| O2 | open-set as enrolled taxa grow from 12 K to 204 K, on cached ToL embeddings | 1 GPU, no download | does the scoring-rule result hold across class count too? |
| K1+ | regional checklist **plus** abstention | eval only | should recover the tail damage K1 measured |
| — | trap order dataset | owner | owner is transferring it; needed to score W3's order classifier |

## 4. Decided not to do — each closed for a reason

- **More in-distribution accuracy**: saturated; the headroom is under shift.
- **LDAM, background suppression**: classifier adaptation subsumes that family (T2b).
- **Re-tuning the ArcFace margin**: two cheap proxies failed for principled reasons; the margin is worth ~0.8 pt.
- **The autoregressive head**: lost by 20 pt.
- **Uniform sampled softmax, low-rank or proxy-free heads for 1 M species**: all measured dead (H2-H4); a 50-image floor makes the matrix fit instead.
- **Re-litigating staged vs end-to-end**: a tie at 198 M once the noise floor was measured (G3b).

## 5. Caveats to carry

- **Noise floors depend on the training regime.** Probe spread is ~0.004 between end-to-end runs and ~0.012 between frozen-trunk stages; identical 198 M frozen runs differed by 3.74 pt once (G3b). Claim nothing under ~3x the matching floor.
- **Thresholds do not transfer between models or embedding spaces.** Rankings do; abstention, novelty and temperature values must be refitted.
- **The cosine head's rows are not unit-norm** (mean 1.08 and 1.77 on two checkpoints). No accuracy number is affected; do not change the head. [[2026-08-06-the-cosine-head-is-not-unit-norm]]

## 6. How to run things

Cluster rules are in [`ucloud/README.md`](ucloud/README.md). **Scale discipline (owner):** test a new mechanism at
20 M; promote to 198 M once, when the recipe stops moving. Costs: ~1,100 img/s at 20 M, ~480 img/s
at 198 M; a 5-epoch 20 M run is ~6.4 h.
