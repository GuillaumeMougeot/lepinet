# H4: can a classifier be trained against centroids instead of a prototype matrix?

**Kind:** research · **Status:** **OPEN** (launched 2026-08-09). Group H's last open training
candidate: replace the species prototype matrix with a buffer of EMA class centroids, removing the
optimiser state that is the actual barrier at 1 M species.

## The problem, and why every other route is closed

Scaling to Tree-of-Life size (~1 M species) is not a compute problem, it is a memory one. A
1280-wide prototype matrix at 1 M classes is:

| | fp32 |
|---|---|
| parameters | 5.12 GB |
| Adam state (m, v) | 10.24 GB |
| **total** | **15.36 GB** |

**Two thirds of it is optimiser state**, and that is the part that will not fit alongside a
backbone. [[2026-08-05-scaling-the-head]] costed the alternatives and closed all but one:

| route | verdict |
|---|---|
| low-rank factorisation | **dead** — the matrix has rank 1035/1280; the margin *spends* dimensions |
| fixed / taxonomy codes | weakened by the same spectrum |
| uniform sampled softmax | **dead** — H2 measured no plateau; 1024/1M is 0.1 % coverage |
| hard-negative sampling | weakened — H3's taxonomy-aware negatives recovered only 26 % |
| **inference by centroids** | **solved** — H1: class centroids replace the trained matrix for **0.29 pt** |

H1 is the one that suggests this experiment. If centroids are good enough to *predict* with, are
they good enough to *train against*? That removes the gradient, and with it the optimiser state:
**15.36 GB becomes 5.12 GB, a 3x reduction, and the component removed is the one that scales worst.**

## The mechanism

`dev/073_proxy_free.py`. The species layer becomes a `register_buffer` holding one unit-norm
centroid per class, updated as an EMA of that class's own embeddings under `no_grad`. Genus and
family keep ordinary trained layers — 4,333 and 102 classes are not the memory problem, and changing
them would confound the measurement.

The backbone still learns: the loss gradient flows into the embedding, which is pulled toward its own
centroid and away from the others. What is removed is the matrix's freedom to move *independently of
the data*. Whether that freedom was load-bearing is the question.

## Three things the selftest caught that a run would have paid for

**1. The in-place/autograd collision — this one was a real bug.** The obvious implementation updates
the centroids at the end of `forward`. That crashes: `F.linear` saves the centroid tensor to compute
the gradient with respect to the embedding, so mutating the buffer before `backward()` raises
*"one of the variables needed for gradient computation has been modified by an inplace operation"*.
The tempting fix — clone the centroids for scoring — would allocate a **second 5.12 GB matrix per
forward** at 1 M classes and destroy the only reason the head exists. The update is therefore
deferred: the embedding (B x d, negligible) is stashed at forward and the callback applies the update
in `after_backward`.

**2. Cold start.** A randomly-initialised centroid starts orthogonal to its own examples, and an EMA
at momentum 0.9 needs tens of updates to escape. A species with 50 images would spend most of an
epoch pointing at noise — and the rare species are the ones this project's macro-F1 is *about*. So a
class's **first** observation replaces its centroid outright; the EMA engages from the second onward.

**3. A constraint on where this can ever be used.** With `hidden: false` the species path contains
**no trainable parameter at all** — `preclassification` is just a normalize and the prototypes are a
buffer. The species loss can then only train the backbone, which is fine when the backbone is
unfrozen and fatal when it is not. **A proxy-free head cannot serve a frozen-trunk stage unless the
bottleneck is kept**, and both of this project's cheap classifier stages (cRT, adaptation) freeze the
body. That bounds the deployment story regardless of how the accuracy lands, and it was found by
writing an assertion rather than by a run.

Verified end to end on the real pipeline before submitting: `Head=proxy_free, hidden=True -> 1.82 M
head params` (bottleneck plus the two coarse layers — no species matrix) with `_CentroidContext`
attached, training past the point where the in-place error fired.

## Prediction (committed)

**Species macro-F1 0.900–0.912**, against the baseline's **0.9148** on the identical recipe.

The reasoning: centroids cannot separate classes beyond where the data already sits, so I expect a
real but modest loss — *larger* than H1's 0.29 pt inference cost, because in H1 the trained matrix
had already done the separating and the centroids only had to approximate the answer. Here they have
to find it.

**Falsified below 0.885.** That would put the cost above one point per GB saved and close Group H's
training half for good — which is a perfectly good outcome, since it would mean the honest answer to
head scaling is "shard the matrix" and the project should stop looking for a clever alternative.

## What each outcome means

- **Inside the range** — proxy-free is the recommended route to 1 M classes, with the frozen-trunk
  caveat above, and Group H closes with a positive answer.
- **Above 0.912** — surprising, and would say the prototype matrix's independent freedom is worth
  nothing, which sits oddly beside the margin result (ArcFace *spends* 1035 dimensions doing
  something). Would need a second run before being believed.
- **Below 0.885** — closed. Sharding is the answer and the paper says so in one sentence.
