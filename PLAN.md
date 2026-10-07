# PLAN — what we are doing now

**Kind:** living · **Last updated:** 2026-10-07 · **Supersedes:** [[2026-07-28-landscape-and-plan]]

The one file in `journal/` meant to be true *today*. Every experiment ID is resolved in
[`RESULTS.md`](RESULTS.md); the reasoning lives in the linked journal entries. Earlier
versions of this file (the August status boards, the rules for the owner's absence) are in its git
history.

## 1. The goal: a submittable paper

Every experiment the paper uses is closed. What stands between [`../paper/DRAFT.md`](paper/DRAFT.md)
and a submission is scoping and writing, plus two evaluation-only runs. No new training is needed.

| # | step | who | state |
|---|---|---|---|
| 1 | **Choose the scope and the venue.** The draft makes 9 contributions over 16 results sections, at 13 k words. Proposed spine: *interventions belong in the classifier* (§4.15) + *the evaluation protocol* (three axes, scoring rules, contamination) + *self-training* (§4.11); the margin analysis (§4.3, §4.7) and head scaling (§4.12) go to an appendix; TreeOfLife (W1-W3) becomes a second paper | owner | open |
| 2 | Restructure §4 to ~6 sections; rewrite the abstract; write the §1 introduction (still an outline) | agent, after 1 | — |
| 3 | Re-score §4.5 with each head's own scoring rule; measure open-set *under shift* for B8 and P5 (§6 says it is unmeasured) | GPU, eval only | — |
| 4 | Figures: six, in the text with captions (`dev/074_figures.py`); building them corrected §4.6a (the B8/P5 gap), §4.12 and §4.14.1 | agent | **done 2026-10-02** |
| 5 | Verify every citation in §1b and the reference list (all written from memory) | owner | open |
| 6 | Final consistency pass: every number against its journal entry | agent | — |
| 7 | Model cards and collection notes present the trade-off instead of recommending P5 (owner's call, 2026-10-07) | agent | **done** |
| 8 | `rules-A2.json` on the drive was overwritten on 2026-08-03 by a run reporting a different head; the paper's §4.9 row comes from the 2026-08-01 file. Re-run A2's rules to confirm | GPU, eval only | open |

## 2. Running now

| ID | what | state | next |
|---|---|---|---|
| **W1** | TreeOfLife-200M download at **short side 256**, on the workstation (`/data/au761367/tol256s/`, `crawl.log`; cores 0-5, nice 10; zero UCloud core-hours) | **2026-10-07: 63.26 M images done (2.0 TB)**; 2.08 M rows left on 7 slow servers at ~8.5 img/s (~3 days); no stall since the deadlock fix. 75 servers marked unreachable (28.3 M rows): probed, Harvard and Flickr had tripped on transient errors and run again in `retry1/` (cores 6-7); ecdysis.org refuses (403), the rest still time out | when both finish: a last pass over the main manifest (records the retry images from disk), upload and unpack on UCloud (1 vCPU), substitution pass for the unreachable servers, re-run the audit (`dev/085`) |

## 3. Backlog, in order

| ID | work | cost | why |
|---|---|---|---|
| W3 | training parquets for the ToL run: join the catalog by uuid, seven levels keyed by full path, check the hierarchy is a tree, drop the audit's 584 k excluded rows; then the 20 M run | GPU, after W1 | the owner's question: does seeing the whole tree help open-set and shift? Also gives a trap order classifier. [[2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know]] |
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
- **Re-litigating staged vs end-to-end**: a tie at 198 M once the noise floor was measured (G3b); frozen as "capacity-dependent".

## 5. Caveats to carry

- **Noise floors depend on the training regime.** Probe spread is ~0.004 between end-to-end runs and ~0.012 between frozen-trunk stages; identical 198 M frozen runs differed by 3.74 pt once (G3b). Claim nothing under ~3x the matching floor.
- **Thresholds do not transfer between models or embedding spaces.** Rankings do; abstention, novelty and temperature values must be refitted.
- **The cosine head's rows are not unit-norm** (mean 1.08 and 1.77 on two checkpoints). No accuracy number is affected; do not change the head. [[2026-08-06-the-cosine-head-is-not-unit-norm]]

## 6. How to run things

Cluster rules (CPU budget, the queue tick, logs over status, one image-heavy job at a time) are in
[`../ucloud/README.md`](ucloud/README.md). **Scale discipline (owner):** test a new mechanism at
20 M; promote to 198 M once, when the recipe stops moving. Costs: ~1,100 img/s at 20 M, ~480 img/s
at 198 M; a 5-epoch 20 M run is ~6.4 h.
