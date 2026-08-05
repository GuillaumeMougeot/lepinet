# Scaling to a million species: options for a head that does not fit

**Kind:** research · **Status:** OPEN (2026-08-05), **ideas only, nothing run.** Owner's framing: the
Tree-of-Life 200M dataset has ~1 M species, and a 1280 × 1M prototype matrix is 1.28 B parameters —
5 GB in fp32, larger than the backbone, and the obvious fix (sharding across GPUs) is a tax on every
future experiment. What can be done that *exploits* the ArcFace × z-score work rather than
discarding it?

## The problem, sized

| classes | prototype matrix (d=1280) | fp32 | its share of a 198 M backbone |
|---|---|---|---|
| 12,041 (today) | 15.4 M | 62 MB | 8 % |
| 100,000 | 128 M | 512 MB | 65 % |
| **1,000,000** | **1.28 B** | **5.1 GB** | **650 %** |

Two distinct costs, and conflating them leads to the wrong fix:

- **Memory/storage** — the matrix itself, plus its optimiser state (Adam doubles or triples it).
- **Compute per step** — a 1 M-way softmax over the batch, and the gradient touching every prototype
  every step even though only ~batch-size classes are present.

A method can fix one and not the other. Sharding fixes neither; it just spreads them.

## The constraint that shapes everything

**One-hot is not the inefficiency.** The owner's instinct is that one-hot under-compresses, and the
information-theoretic version is right — 1 M classes need only 20 bits — but the prototype matrix is
not storing labels, it is storing **one direction in embedding space per class**. Compressing it means
asserting that class directions are *not* independent, i.e. that they lie on some lower-dimensional
structure. Whether that is true is an empirical question, and it is the question all the options below
are really asking. So the first thing to run is not a method, it is a measurement (option 0).

## Option 0 — measure the intrinsic dimension of the prototypes first

Take the trained 12,041 × 1280 prototype matrix and ask how many dimensions it actually occupies:
singular-value spectrum, participation ratio, and the accuracy retained when truncated to rank *k*.

**Why this is first.** Every compression option below has an implied rank. If the prototypes already
lie in ~200 dimensions, a low-rank factorisation is nearly free and the exotic options are
unnecessary. If they genuinely span 1280, then factorisation caps out early and the effort belongs
elsewhere. This is a **CPU-only job on an existing checkpoint**, takes minutes, and it tells us which
of the options is worth building.

**Prediction:** the taxonomy imposes strong structure (species within a genus should have similar
directions), so I expect effective rank well below 1280 — maybe 300–600 — with a long tail rather
than a sharp cutoff. If so, rank-512 factorisation is close to free.

## The options, and what each exploits or breaks

### A. Low-rank factorisation: `W = U V`, with `U: 1280×r`, `V: r×C`

Parameters drop from `1280·C` to `r·(1280+C)`. At r=256, C=1M: 1.28 B → 0.33 B, a **4× saving**.

- **Keeps ArcFace × z-score entirely.** The rows of `UV` are still directions; normalise them and
  every result transfers unchanged. This is the *only* option in this list that is fully compatible
  with the existing head.
- **Does not fix compute** — you still materialise C logits per step.
- Ceiling is set by option 0's answer.

### B. Hierarchical / two-stage softmax

Predict genus (4,333 today, maybe 100 k at ToL scale), then species *within* that genus. The species
matrix becomes block-sparse: only the true genus's children are scored.

- **Fixes both costs**, and is the classic answer.
- **This project has already measured that conditioning hurts** — the parent-conditioned head lost
  2.9 pt in-distribution and is the worst of four under shift ([[2026-07-16-why-was-fastai-behind-mini-trainer]],
  paper §4.1). At 1 M classes the memory argument might override that, but it is a known cost, not a
  free lunch, and the failure mode (parent errors propagating down) gets *worse* with a deeper tree.
- Worth revisiting only in the *sampled-softmax* form below, which keeps the flat head.

### C. Sampled softmax / negative sampling

Score the true class plus a sample of negatives each step. Standard in extreme classification and in
language modelling.

- **Fixes compute, not memory.** The matrix still exists.
- **Interacts badly with ArcFace in a specific way**: the margin's effect depends on the *hardest*
  negatives, and uniform sampling misses them. Would need hard-negative mining, which needs an index
  over the prototypes — at which point see option E.
- Cheap to try, and orthogonal to A.

### D. Encode class identity in the embedding space (the owner's binary idea)

Assign each class a fixed code — random ±1 in 1280 dims, or a binary code of the class index — and
train the embedding to match it. No learned prototype matrix at all; the "matrix" is a deterministic
function of the class id.

- **Fixes memory completely.** 1 M classes cost zero parameters.
- **The mathematics is favourable and underrated.** In 1280 dimensions, random unit vectors are
  near-orthogonal (§concepts: cosine sd ≈ 0.028), so 1 M random codes are almost as separable as 1 M
  learned ones *if the embedding can reach them*. Johnson–Lindenstrauss says 1280 dims comfortably
  supports 1 M near-orthogonal directions.
- **What it breaks:** the prototypes can no longer *move*, so the model cannot discover that two
  species look alike. That is exactly the structure option 0 is measuring — if the prototypes are
  low-rank *because the taxonomy is real*, freezing them at random codes throws that away.
- **The fix, and this is the interesting version:** make the codes **taxonomy-structured** rather than
  random — e.g. code(species) = code(genus) + small orthogonal offset. Then the code table is
  generated from the taxonomy, costs nothing to store, and *encodes* the hierarchy the project has
  spent weeks showing is useful. **ArcFace × z-score works unchanged**, because it only needs unit
  directions. This is my pick for the most interesting untested idea here.

### E. Retrieval instead of classification (the owner's second question, and it is the strong one)

At inference, replace the matrix multiply with **nearest-neighbour search over class centroids**
computed from the training set.

- **Fixes inference memory and compute**: an ANN index over 1 M centroids is ~a few GB on disk and
  sub-millisecond to query, and it never materialises 1 M logits.
- **Does not fix training** on its own — combine with A, C or D for that.
- **This is where ArcFace × z-score pays off most, and the project already has the evidence.** The
  margin exists to make classes tight and well-separated *in angle*; that is precisely the property a
  nearest-neighbour search needs. A plain cosine head's clusters are diffuse (mean max-cos 0.144 for
  the plain head vs 0.671 for ArcFace × z-score, [[DIRECTIONS]], the C3 stratified-OOD numbers) — kNN on
  those would be poor. **We have trained exactly the kind of embedding retrieval wants.**
- It also generalises the open-set story for free: a novel species is one whose nearest centroid is
  far, which is the same score the project already uses.

## What to run, in order

1. **Option 0 (CPU, minutes).** The prototype spectrum. Decides whether A is nearly free.
2. **Option E as an *evaluation*, no retraining (GPU, ~30 min).** Compute class centroids from the
   training fold, score the test fold by nearest centroid, and compare to the linear head. This is
   the owner's suggestion and it is the cheapest real experiment available: it needs no new training,
   it directly tests whether the ArcFace geometry supports retrieval, and a positive result makes the
   1 M-class inference path immediately plausible.
   **Prediction:** centroid-kNN lands within 1 pt of the linear head for the ArcFace × z-score model
   and *far* behind for the plain cosine one — because the margin is what makes centroids meaningful.
   If both are close, the margin is doing less than believed; if both are poor, the class conditional
   distributions are not unimodal and centroids are the wrong summary (in which case: medoids, or
   k centroids per class).
3. **Option D with taxonomy-structured codes**, at current scale first, as a drop-in head. If it
   matches the learned head at 12 k classes, it is the answer at 1 M.

Options B and C are documented above so they are not re-proposed, but neither is a priority: B has
already lost once in this project, and C fixes the cost we care about least.

---

## First results: centroid retrieval works, and the ArcFace prediction holds (2026-08-05)

`dev/068` on the ArcFace × z-score model, 12,041 species, test fold.

| method | species macro-F1 | vs linear head |
|---|---|---|
| **linear head** (the 15.4 M prototype matrix) | **0.9105** | — |
| **centroid, mean** | **0.9077** | **−0.29 pt** |
| centroid, k-means (k=3) | 0.8988 | −1.18 |
| centroid, medoid | 0.8960 | −1.45 |

**Predicted "centroid-kNN within 1 pt of the linear head for the ArcFace model". It landed at
0.29 pt.** The prototype matrix can be replaced, at inference, by the mean training embedding per
class — which is a *derived* quantity, not a learned parameter.

**Why this matters for 1 M species.** The prototype matrix does not have to be trained *or stored as
weights*: centroids are computed from data, can be built incrementally as species are added, and are
exactly what an ANN index wants. It does not fix training cost (option 0/A/C still apply), but it
removes the 5 GB matrix from the *inference* path and makes adding a species a matter of averaging
its images rather than retraining a head.

**Mean beats k-means and medoid**, which is worth noting because the opposite is often assumed. The
mean is the better summary here, so the classes behave like single tight blobs rather than multimodal
ones — consistent with what the margin is designed to produce, and it means the simplest option is
also the best one.

### The prototypes are *not* low-rank — option A is dead

| | 90 % energy | 99 % energy | participation ratio |
|---|---|---|---|
| ArcFace × z-score | **rank 1035** / 1280 | 1250 | 1152 |

Rank-truncating the head confirms it: rank 512 costs 0.35 pt, 256 costs 1.06, 128 costs 3.14, 64
costs 13.4. So a rank-512 factorisation is *nearly* free but only halves the matrix, and anything
aggressive enough to matter at 1 M classes destroys accuracy.

**My prediction was wrong**, and specifically: I expected effective rank 300–600 because the taxonomy
should make related species point in similar directions. It does not work out that way — the head
uses nearly the full 1280 dimensions. In hindsight the reason is visible in the training objective:
ArcFace *actively pushes classes apart*, so it spends dimensions rather than economising on them.
The margin and low-rank compression want opposite things.

**Consequence:** option A (low-rank factorisation) is not the answer, and **option D
(taxonomy-structured fixed codes) is now less attractive too** — it assumes class directions have
exploitable structure, and the spectrum says the trained ones do not. Retrieval (E) is the option
that survived contact with data.

## An unresolved discrepancy — the plain-head control is not reportable yet

The same script scored the **plain cosine** model's linear head at **0.5589**, against its known
0.9135. Its centroids scored 0.8809, i.e. **32 points above its own linear head**, which is not a
credible finding — it is a symptom.

Reporting it would have been reporting a bug as a result. The candidates, none yet eliminated: the
glob resolved to a different checkpoint than `lepinet test` used; the per-class test cap interacts
with that model differently; or the script's reimplementation of the head (`e @ W.T`) diverges from
the model's own forward for this head type.

`dev/068` now prints the **resolved checkpoint path**, the **prototype row norms**, and scores the
linear head **through the model's own forward** alongside the reimplementation, reporting their
agreement. If they disagree the bug is in the script; if they agree and the number is still 0.5589,
the checkpoint is not the one that scored 0.9135. Re-running both models with that instrumentation.

The ArcFace result above stands regardless — its linear head reproduced at 0.9105 against a known
0.9035, so the pipeline is sound for that model. But the *comparison* between heads, which is the
whole point of running both, waits for the control.
