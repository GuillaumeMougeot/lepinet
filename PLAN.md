# PLAN — what we are doing now

**Kind:** living · **Last updated:** 2026-10-23 · **Supersedes:** [2026-07-28-landscape-and-plan](archive/journal/engineering/2026-07-28-landscape-and-plan.md)

The one file about *today*. Experiment IDs resolve in [`RESULTS.md`](RESULTS.md); reasons are in
the linked journal entries; older versions of this file are in git history.

## 1. The goal: a submittable paper

Every experiment [`paper/DRAFT.md`](paper/DRAFT.md) uses is closed. What remains is scoping, writing
and two evaluation-only runs.

| # | step | who | state |
|---|---|---|---|
| 1 | **One paper, merged with Svenning et al.'s draft** (`~/Documents/hierarchical_classification.pdf`; owner, 2026-10-22: no second paper). Proposed outline: methods = data (Lepidoptera, Pl@ntNet, traps with the grouped split), HGLL head, the five heads, one recipe, evaluation (metric conventions, retrain noise); results = hierarchy (parameters vs supervision, alignment, novelty by taxonomic distance), evaluating under shift (conventions, scoring rules, contamination), long tail, target-domain data, classifier stage, abstention; appendix = margin, nuisance augmentation, head scaling, recipe ladder, AR decoder, EMLA and regulariser. Open: one training recipe for both drafts (EMLA vs √-oversampling), the colleague's raw trap predictions rescored with `dev/092` | owner + colleague | open |
| 1b | **Pin the rerun recipe before any rerun** (owner, 2026-10-23: the colleague's settings have priority; some experiments will be redone). 16 decisions with both drafts' settings and our evidence: species floor, sampling, loss (EMLA vs CE), regulariser, epochs, image size, backbones, heads, margin, seeds, dose, open-set split, budget. [[2026-10-23-which-recipe-does-the-merged-paper-rerun]] | owner + colleague | open |
| 2 | Insert our pieces into the Overleaf draft through the Merge Desk (`paper/merge/`, `dev/094`): methods steps 01-11 drafted (02 pasted; 04 on hold for 1b); results steps wait for 1b | owner, agent | ongoing |
| 3 | Re-score §4.5 with each head's own rule; measure open-set under shift for B8 and P5 | GPU, eval only | — |
| 4 | Verify every citation in §1b and the references (written from memory) | owner | open |
| 5 | Consistency pass: every number against its journal entry | agent | — |
| 6 | Re-run A2's scoring rules: `rules-A2.json` was overwritten on 2026-08-03; §4.9 uses the 2026-08-01 file | GPU, eval only | open |

## 2. Running now

| ID | what | state | next |
|---|---|---|---|
| **N3-P5** | open-set on all 47,905 trap crops for P5 (N2 C3b and N3 B8 done, part B written) | resubmitted 2026-10-23 (12431255) after a missing open_clip | download `trap-openset-P5.{json,parquet}`, add its row to part B of [[2026-10-23-does-the-model-know-when-a-trap-image-is-too-poor]] |
| **W1** | TreeOfLife-200M at short side 256, on the workstation (`/data/au761367/tol256s/`; zero UCloud core-hours) | crawl done: 64.16 M images plus ~0.54 M retried ([crawl](journal/engineering/2026-10-01-the-crawl-resized-the-wrong-side.md)); `retry3/` (9 servers, 72 h cap) | shared as `/data/datasets/tol256`: 65.2 M images, metadata with ToL's per-image fields ([share](journal/engineering/2026-10-21-sharing-the-tol-crawl.md)). **Upload re-planned** (1,053 shards); old shards in the UCloud Trash; upload paused on storage |

## 3. Backlog, in order

| ID | work | cost | why |
|---|---|---|---|
| Q | the back-off policy: greedy 95 % thresholds vs expected utility, generic over taxonomies; Lepidoptera, traps, Pl@ntNet | CPU only, existing per-image outputs | B8 and B3rep5x never answer at genus on traps. [[2026-10-23-what-is-the-best-back-off-policy]] |
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
