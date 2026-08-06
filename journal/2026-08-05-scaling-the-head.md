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

## The discrepancy, resolved: the bug was mine, and it invalidates one number above

The instrumented re-run localised it in one line:

```
model forward vs e@W.T agreement: 0.5309
  linear_head          0.9182     <- the model's own forward, matching its known 0.9135
  linear_head_reimpl   0.5589     <- my e @ W.T
```

So the model is fine and the embeddings are fine (centroids from them score 0.8809); **the weight
matrix I read is not the one the model uses.** The cause is a single `F.normalize` in `dev/068`.

`cosine_to_zscore` is monotone, so `argmax(zscore(e @ W))` equals `argmax(e @ W)` *for the same W*.
The only way the argmax can change is if the true rows are **not** unit-norm — if the head holds
per-class magnitudes $g_c$, then $\arg\max_c\, g_c \cos_c \neq \arg\max_c \cos_c$, and
normalising strips exactly that $g_c$. `_normalize_layer` is supposed to freeze every row norm at 1;
this checkpoint's evidently did not stay there.

**Consequence for what is written above: the "rank 196" figure for the plain head is not a
measurement of direction structure.** The raw matrix's spectrum confounds direction spread with
row-norm variation, and a matrix with varying row norms has a small participation ratio (213) even
when its directions are spread. The ArcFace figure (rank 1035, participation 1152) is unaffected —
its forward agreed at 0.9801, so its rows *are* unit — and the conclusion drawn from it, that option
A is dead, stands.

`dev/068` now uses the weight as the model uses it, reports the row norms with a warning when they
are not unit, and computes the spectrum on the normalised matrix while truncating the raw one.
Both models re-running.

**Two things this cost, worth recording.** The first attempt reported a plain-head result that was
32 points above its own linear head — an impossible-looking number that was correctly *not*
published. The second attempt would still have been wrong, but silently, had it not printed the
agreement. **The instrumentation was worth more than the experiment.**

## The original discrepancy note (kept)

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


---

## Training is the unsolved half, and the spectrum narrowed it to one family (2026-08-05)

Retrieval solves *inference*. Training still needs the matrix, and at 1 M classes that is **5.1 GB in
fp32 plus 10.2 GB of Adam state = 15.4 GB** before a single activation. The spectrum killed the two
options that would have shrunk it:

| option | GPU memory | compute per step | status |
|---|---|---|---|
| E retrieval | inference only | inference only | **works** (0.9077 vs 0.9105) |
| A low-rank | partial | no | **dead** — rank 1035/1280 |
| D fixed codes | yes | no | weakened by the same spectrum |
| **C sampled softmax + CPU-resident matrix** | **yes** | **yes** | **untested** |
| **F proxy-free batch centroids** | **yes** | **yes** | untested, bigger build |

### Why sampled softmax is more attractive than it looked

The original note dismissed C as "fixes compute, not memory". That was wrong, because it assumed the
matrix must live on the GPU. It does not: **keep it in CPU RAM and gather only the sampled rows.**
At 4096 negatives that is 21 MB moved per step, against 15.4 GB resident. The optimiser state stays
on CPU too, updated only for the touched rows — which is exactly what sparse embedding tables do in
recommender systems, at far larger scale than 1 M.

So C fixes both costs, needs no new architecture, and keeps ArcFace × z-score intact. Its one real
risk is the one already flagged: **the margin's effect depends on the hardest negatives, and uniform
sampling misses them.**

### The experiment that decides it, and it can run now

**Does the ArcFace margin survive when most negatives are absent from each step?** That is answerable
at 12,041 classes today, and the answer transfers *pessimistically*: 1024 negatives here is 8.5 % of
classes, while 1024 of 1 M is 0.1 %. **If sampling fails at 8.5 % coverage it certainly fails at
0.1 %; if it survives, the 1 M case is plausible and worth building properly.**

Four arms, one factor, B3's recipe at 20 M: full 12,041-way softmax (the control, = A1's 0.9035),
then 4096, 1024 and 256 sampled negatives. Negatives are drawn uniformly *plus* the batch's own
classes, so the in-batch hard negatives are always present — the cheap half of hard-negative mining,
and the half that matters when a batch of 64 contains 64 different species.

**Prediction (committed):** 4096 within 0.5 pt of the control; 1024 within 1.5 pt; 256 loses more
than 3 pt. Reasoning: the margin needs to see the *confusable* classes, and confusability is
concentrated within a genus — with 12,041 species over 4,333 genera, a uniform sample of 1024 has a
~24 % chance of containing any given congener, so at 256 the margin is mostly acting against random
far-away classes and doing little. **If 256 does *better* than that, the margin is less
negative-dependent than believed and the 1 M case gets much easier.**

**If it works**, the follow-up is hard-negative sampling using the ANN index that option E already
justifies — sample negatives from the query's neighbourhood rather than uniformly, which restores
exactly the signal uniform sampling loses.

**Option F (proxy-free)** is the fallback if C fails: compute class centroids from the batch and a
MoCo-style queue of recent embeddings, so no prototype matrix exists at all. Higher risk — with 64
classes per batch the negative set is tiny, which is the known failure mode of contrastive losses at
extreme class counts — and a much bigger build. Worth doing only if sampling is shown not to work.

---

## H2: sampled softmax degrades smoothly and too fast (2026-08-06)

| negatives | coverage | species macro-F1 | vs control |
|---|---|---|---|
| full (12,041) — A1 control | 100 % | 0.9035 | — |
| 4096 | 34.0 % | 0.8940 | −0.95 |
| 1024 | 8.5 % | 0.8710 | −3.25 |
| 256 | 2.1 % | 0.8315 | −7.20 |

**Predicted 4096 within 0.5 pt, 1024 within 1.5, 256 losing more than 3. Two of three wrong**, and
wrong in the same direction: the cost is roughly **double** what I expected at every point. Only the
256 arm was predicted correctly, and for the wrong reason — it was supposed to be the cliff, and
instead the curve has no cliff at all.

**The shape is the finding.** The loss is smooth and roughly linear in log-coverage: halving coverage
costs a fairly constant amount (34 % → 8.5 % → 2.1 % gives −0.95, −3.25, −7.20). There is no plateau
where sampling is free, which is what the "just sample negatives" approach needs.

**This is bad news for the 1 M case, and the extrapolation is the point.** The answer transfers
pessimistically by construction: 1024 negatives here is 8.5 % coverage and costs 3.25 pt, while 1024
of 1 M is **0.1 %** — an order of magnitude below the worst point measured, which already costs 7.2.
Nothing in this curve suggests it flattens.

**Why, mechanically.** The margin's job is to push the true class away from its *confusable*
neighbours, and confusability is concentrated within a genus. With 12,041 species over 4,333 genera,
a uniform draw of 1024 contains a given congener with probability ~8.5 %, so most steps apply the
margin against classes the model was never going to confuse. The in-batch classes are always
included, but a batch of 64 supplies only 64 of them.

**What survives.** Uniform sampling is the wrong sampler, not necessarily the wrong idea — the
obvious fix is to draw negatives from the *query's neighbourhood* using the ANN index that the
centroid result (option E) already justifies. That restores exactly the signal uniform sampling
loses, and it is now the only version of option C worth building. But it adds an index-maintenance
loop to training, so it is no longer the cheap option it appeared to be.

**Status of the head-scaling direction after this:** inference is solved (centroids, −0.29 pt).
Training is not. Low-rank is dead, fixed codes are weakened by the same spectrum, uniform sampling
degrades too fast, and what remains is hard-negative sampling with a maintained index, or option F
(proxy-free, no matrix at all). Both are real builds rather than experiments.


---

## The recommendation, consolidated (2026-08-06)

Everything above is exploration. This is what I would actually build, and why the evidence points
there rather than anywhere else.

### The two problems are separable, and one is already solved

| | status | cost at 1 M |
|---|---|---|
| **inference** | **solved** — centroids cost 0.29 pt, and are built from data | ANN index, no 5 GB matrix |
| **training** | open | 5.1 GB matrix + 10.2 GB optimiser state |

That separation is the useful part. A deployed model never needs the learned matrix, so whatever
training does, it does not have to produce an artifact that ships.

### What the measurements eliminated

- **Low-rank factorisation** — rank 1035/1280. Dead.
- **Fixed / taxonomy-structured codes** — rests on the class directions having exploitable
  structure. The same spectrum says they largely do not.
- **Uniform sampled softmax** — smooth degradation with no plateau; 0.1 % coverage at 1 M is far past
  the worst point measured.
- **Hierarchical two-stage prediction** — measured worst of four heads here, on both axes.

The eliminations are worth as much as the survivors: four plausible directions closed by three cheap
measurements, and each was closed for a *reason* that generalises rather than by a disappointing
number.

### The recommendation: EMA centroids as the classifier, with ANN-mined hard negatives

One design, assembled from three measured results rather than from the literature:

1. **No learned prototype matrix at all.** Maintain a centroid per class as an exponential moving
   average of its embeddings, updated whenever a class appears in a batch. Zero learned parameters,
   zero optimiser state, and the table lives on CPU. *Justified by:* centroids match trained
   prototypes to 0.29 pt.
2. **Score against a small active set each step** — the batch's own classes plus the *k* nearest
   centroids to each query, retrieved from an ANN index over the centroid table. *Justified by:*
   uniform negatives fail (H2) while the margin's effect depends on confusable classes; ANN
   retrieval over these centroids is exactly what the inference result showed works.
3. **Keep ArcFace × z-score unchanged.** It operates on unit directions, which is what centroids are.
   Every open-set result carries over.

Memory at 1 M classes: the centroid table is 5.1 GB of *data* on CPU with no gradient or optimiser
state, against 15.4 GB of GPU-resident parameters — and the per-step GPU cost is `batch × (64 + k)`
rather than `batch × 1M`.

**The honest risks**, in order:

- **Cold start.** Centroids are meaningless before the embedding is trained. Mitigation: warm-start
  the backbone from an existing model — which this project has — and use random negatives for the
  first epoch.
- **Index staleness.** The centroids move while the index is fixed. Mitigation: rebuild periodically;
  the retrieval only has to be approximately right, since it is choosing negatives, not predictions.
- **Feedback loop.** Mining negatives from a model's own neighbourhood can reinforce its errors. This
  is the same shape as the self-training failure mode, where the fix was a strict gate and an
  evaluation the training could not reach.

### What to run next, and it is small

**The cheapest decisive test is at 20 M and needs no new infrastructure: taxonomy-aware negative
sampling.** H2's failure was attributed to uniform draws missing the *confusable* classes, and
confusability is concentrated within a genus. So sample negatives as *all congeners of the batch's
classes* plus a uniform fill to the same budget, and compare to H2's uniform arm at matched budget.

- If taxonomy-aware 1024 recovers most of the 3.25 pt that uniform 1024 lost, the hard-negative
  hypothesis is confirmed and the full ANN design is worth building.
- If it does not, the loss is not about *which* negatives but *how many*, and the whole sampling
  family is dead — which would leave option F (proxy-free) as the only survivor and is worth knowing
  before building an index.

It uses the taxonomy already in every checkpoint, costs one 20 M run per arm, and settles the
mechanism behind the design above.
