# Knowing what you don't know: calibrated open-set hierarchical classification for long-tailed species identification

**Status:** working draft (2026-08-01). Numbers are from `RESULTS.md`; the reasoning behind each is
in [`../journal/`](../journal/). Sections marked _(pending)_ await runs that are in flight.

---

## Abstract (draft)

Automated species identification is usually framed as fine-grained classification over a fixed label
set. We argue this framing is the wrong one for deployment, and support the argument with three
results on a 12,041-species Lepidoptera benchmark. First, **hierarchical prediction heads do not
help**: an independent multi-head cosine classifier, a parent-conditioned hierarchical head and an
autoregressive head are all matched or beaten by a *single* species head whose genus and family
predictions are obtained by marginalising its own posterior (0.9135 vs 0.9110 species macro-F1, and
better at every coarser level) — while being smaller and consistent by construction. Second,
in-distribution accuracy is close to saturated (0.9316 with a modern large backbone) yet collapses to
**0.70 on data from a different source**, so the remaining error is dominated by distribution shift
rather than by classifier design; we further decompose that gap, showing that hand-named nuisances
account for only about one sixth of it. Third, and most usefully, we show that composing an **additive
angular margin with a dimension-aware z-score transform** turns a classifier that is near-chance at
detecting unseen species (AUROC 0.601) into a reliable one (**0.9115**; **0.9068** on the single-head architecture we
recommend) at a cost of 0.4 to 1.0 points of closed-set accuracy — where the margin *alone* costs 3.3 points and yields only 0.732. Taken
together, these give a single-head architecture that (i) predicts species, (ii) degrades gracefully
to genus/family by marginalisation when the image cannot support a species call, and (iii) flags
taxa it has never seen. We release the package, the models and the reproduction recipes.

---

## 1. Introduction _(outline)_

- Species ID from images: fine-grained, long-tailed (53 % of species have < 200 images), and
  **open-set in deployment** — a field tool meets taxa outside its label set constantly.
- The literature's instinct is to encode the taxonomy in the *architecture* (hierarchical softmax,
  conditional heads, autoregressive decoders). We test that instinct and find it does not pay.
- Reframe: the taxonomy's value is in **abstention and novelty handling**, not in the classifier.
  Two orthogonal kinds of "unknown":
  1. **Image-space / open-set** — the taxon was never trained on. Detect it.
  2. **Hierarchy-space / rank abstention** — the photo cannot resolve the species. Back off to the
     rank the evidence supports ("unknown species, but *Noctuidae*").
- Contributions: (C1) a negative result on hierarchical heads with a like-for-like protocol;
  (C2) the marginalisation-beats-coarse-heads result; (C3) **ArcFace × z-score**, with the analysis
  of *why* the two compose; (C4) a measured generalisation gap, **decomposed** into the part
  removable by naming nuisances (17 %) and the part that is not; (C5) an interaction result — an
  angular margin degrades *marginalisation* more than classification, because summing a posterior is
  calibration-dependent, replicated across a 10× scale change; (C6) an open, reproducible package.

## 2. Method

### 2.1 Cosine classification head

A backbone $f_\theta$ maps an image to $z \in \mathbb{R}^{d}$ through a bottleneck, then
$e = z/\lVert z\rVert$. Each class $j$ owns a prototype $w_j$ constrained to $\lVert w_j\rVert = 1$
(weight-norm with the row norm frozen), so the pre-activation is a cosine similarity

$$\cos\theta_j \;=\; e^{\top} w_j \;\in\; [-1, 1].$$

### 2.2 The z-score transform (and why a raw cosine is a poor logit)

In high dimension the cosine between (near-)random unit vectors concentrates sharply: for $e$
uniform on $S^{d-1}$, $\cos\theta$ has mean 0 and standard deviation $1/\sqrt{d}$, so with
$d = 1280$ virtually all class scores fall inside a band of width $\approx 0.1$. Feeding that into a
softmax gives a nearly uniform posterior regardless of the embedding's quality — the classic reason
cosine classifiers need a scale factor.

Instead of an arbitrary multiplier we apply the transform that maps this concentrated distribution
onto an approximately standard normal one:

$$Z(\cos\theta) \;=\; \sqrt{d-2}\;\Bigl(\arccos(-\cos\theta) - \tfrac{\pi}{2}\Bigr).$$

Its derivative at $\cos\theta = 0$ is $\sqrt{d-2}$, i.e. the transform *is* a scale — but a
principled, dimension-aware one rather than a tuned constant. Logits are $Z(\cos\theta_j) + b_j$
with $b_j$ frozen at 0.

### 2.3 ArcFace × z-score

ArcFace [Deng et al. 2019] sharpens class boundaries by rotating the *true* class by a margin $m$
before the softmax, $\cos\theta_y \mapsto \cos(\theta_y + m)$, and rescaling by a constant $s$:
$\ell_j = s\cos\theta_j$ for $j \ne y$, $\ell_y = s\cos(\theta_y + m)$.

We instead **compose the margin with the z-score transform**:

$$\ell_j \;=\; \begin{cases} Z\bigl(\cos(\theta_y + m)\bigr) & j = y \\[2pt] Z(\cos\theta_j) & j \ne y \end{cases}$$

with, as usual, the margin applied **only during training** (at inference $m = 0$, so the forward
pass is label-free and exports to ONNX unchanged).

Two implementation notes make this practical:

1. **The head stays label-free.** Because $Z$ is *invertible*,
   $\cos\theta = \sin\bigl(Z/\sqrt{d-2}\bigr)$ (verified to $10^{-7}$), the head can emit
   $Z(\cos\theta)$ and the **loss** recovers the cosine, rotates the true class, and re-applies $Z$.
   No labels enter the model's forward pass.
2. **No $s$ to tune.** For the softmax to express confidence $p$ over $C$ classes it needs
   $s \gtrsim \log\bigl((C-1)p/(1-p)\bigr)$ [Wang et al. 2017]; at $C = 12{,}041$, $p = 0.9$ that is
   $s \ge 11.6$. The z-score's implicit scale is $\sqrt{d-2} = 35.7$ at $d = 1280$ — **already above
   the floor**. The transform supplies for free the scale that raw-cosine ArcFace must guess.

### 2.4 Marginalisation instead of coarse heads

Let $\pi(\cdot)$ map a class to its parent. Coarse posteriors are obtained from the species
posterior rather than from dedicated heads:

$$\log P(g) \;=\; \operatorname*{log\,sum\,exp}_{\{\,s\,:\,\pi(s)=g\,\}} \log P(s),$$

applied recursively up the taxonomy. This is exact, adds no parameters, and is **consistent by
construction**: the genus argmax can never contradict the parent of the species argmax (measured at
1.81 % of images for independently trained heads).

## 3. Experimental setup

12,041 species / 4,333 genera / 102 families; ~3 M training images; a held-out fold of
**629,742 images**. Backbone efficientnet\_v2\_s unless stated; Muon + AdamW, one-cycle, 5 epochs,
460→256 px, square-root class oversampling ($p=0.5$). Metric: **species macro-F1** (each species
weighted equally, so the tail counts). All comparisons change exactly one factor; configs are
released. Open-set benchmark: species with < 50 images, excluded from training but drawn from the
**same image distribution**, isolating novelty from domain shift.

### 3.1 How the baseline recipe was chosen

Every comparison in §4 is made against a baseline, so it matters how that baseline was arrived at.
It was not designed; it was built by single-factor ablation from a weak starting point, and we
report the path because the *ordering* of what mattered is itself informative.

| change | species macro-F1 | Δ |
|---|---|---|
| Muon + flat-cosine schedule, heavy augmentation | 0.8297 | — |
| + warmup, **lighter** augmentation, looser gradient clip | 0.8769 | +4.7 |
| flat-cosine → **one-cycle** schedule | 0.8887 | +1.2 |
| 5 → 10 epochs | 0.8976 | +0.9 |
| 5 epochs + **square-root class oversampling** | **0.9148** | +2.6 |

Three observations, each of which shaped the rest of this work.

**The optimiser was not the lever.** Muon is present in every row, including the 0.8297 one. What
moved the number was the annealing schedule, the augmentation strength and the sampler — decisions
about *what the model sees and for how long*, not about how gradients are applied. The one-cycle gain
in particular is a measurement artefact turned real: under a flat schedule the model was still
descending when the epoch budget expired, so it was being graded mid-convergence.

**Square-root oversampling is preferred to logit adjustment on structural grounds, not just
empirical ones.** Logit adjustment reached 0.9031 and *degraded* genus and family to get there. Its
single temperature $\tau$ is shared across three label distributions with very different class
counts and tail shapes (12,041 / 4,333 / 102), and no single value is simultaneously correct for
all of them. Oversampling changes which examples are drawn without changing what the loss means for
any example, so it acts consistently at every level. This is the same argument that later favours
marginalisation over per-level heads (§2.4): **in a hierarchy, prefer mechanisms that do not require
one constant to be right at every level at once.**

**Numerics.** The cosine head overflows in fp16; all runs use bf16. This is not a tuning detail — an
autoregressive baseline trained visibly broken under fp16 and the failure presents as a modelling
bug rather than a numerical one.

Two caveats are stated rather than hidden. The +4.7 row bundles three changes, so their individual
contributions are unrecoverable; and the 10-epoch run was still improving when it stopped, so the
5-epoch budget used throughout §4 is a comparison convention, not a converged optimum — absolute
numbers here understate what the architecture can reach, while the *differences* between arms, which
is what we claim, are measured at matched budget.

## 4. Results

### 4.1 Hierarchical heads do not help (C1, C2)

| head | species | genus | family |
|---|---|---|---|
| multi-head independent | 0.9110 | 0.9587 | 0.9708 |
| parent-conditioned hierarchical | 0.8845 | 0.9471 | 0.9683 |
| autoregressive | 0.69–0.73 | — | — |
| **single head + marginalisation** | **0.9135** | **0.9606** | **0.9739** |

The single head wins at *every* level. Marginalisation also beats separately trained coarse heads in
a matched comparison (+0.7 pp genus, +3.1 pp family). Marginal supervision *during* training
_(pending)_.

### 4.2 Accuracy saturates; generalisation does not (C4)

| model | in-distribution | external dataset |
|---|---|---|
| efficientnet\_v2\_s | 0.9110 | — |
| ConvNeXtV2-L @320 | **0.9316** | 0.6950 |
| DINOv3-ConvNeXt-L @320 | 0.9311 (≈2× faster to train) | — |

A ~23-point drop on data from a different source. Distillation into a small student saturates at
~0.88 **regardless of teacher quality** (0.8786 from a 0.911 teacher; 0.8756 from a 0.9316 teacher),
i.e. student capacity, not teacher accuracy, is the binding constraint.

### 4.3 ArcFace × z-score: the trade-off dissolves (C3)

| head | species macro-F1 | **open-set AUROC** | known $\max\cos$ | novel $\max\cos$ |
|---|---|---|---|---|
| cosine (z-score, no margin) | **0.9110** | 0.601 | −9.27 ± 7.44 | −11.46 ± 6.82 |
| ArcFace ($s\cos$, $m{=}0.3$) | 0.8784 | 0.732 | 26.00 ± 13.03 | 23.47 ± 10.84 |
| **ArcFace × z-score** ($m{=}0.3$) | 0.9069 | **0.9115** | 32.58 ± 7.83 | 18.17 ± 6.38 |

The composition beats **both** components on the axis each was meant to own: +31 points of AUROC over
the plain cosine head for −0.4 points of accuracy, and +18 AUROC *and* +2.9 accuracy over plain
ArcFace. The trade-off reported for margins is therefore not intrinsic — it is an artefact of
discarding the calibrated transform.

**Mechanism.** Measured on held-out images against each model's own prototypes:

| head | intra ($\cos$ to own) | inter (max $\cos$ to wrong) | margin | silhouette |
|---|---|---|---|---|
| cosine | −0.154 | −0.336 | 0.182 | 0.617 |
| ArcFace × z-score | **+0.667** | 0.056 | **0.610** | 0.641 |

Silhouette barely moves: *separability was never the problem* — closed-set accuracy is equal. What
changes is **absolute** angular position (Fig. 3, `figures/fig3_embedding_tsne.png`; the score
distributions that carry the result are Fig. 4, `figures/fig4_openset_scores.png`). A novelty score $\max_j \cos\theta_j$ is only meaningful if
"close to a known class" has an absolute scale; the plain head places everything near-orthogonal to
everything (intra −0.15), so novel and known look alike. This is why a 2-D projection (UMAP/t-SNE) is
the wrong visualisation: it is invariant to exactly the property that carries the effect.

### 4.4 Novelty is graded by taxonomic distance

Treating open-set detection as binary hides the structure that matters. Splitting novel taxa by how
much of their lineage the model has seen (unfiltered catalogue; species below the training floor are
genuinely unseen, in-domain):

| stratum | n | cosine | **ArcFace × z-score** |
|---|---|---|---|
| **near** — unseen species, known genus | 8,000 | 0.5606 | **0.8493** |
| **mid** — unseen genus, known family | 8,000 | 0.6177 | **0.9094** |
| **far** — unseen family | 399 | 0.6656 | **0.9411** |

Both heads are **monotone in taxonomic distance**: the further a novel taxon lies from the training
set, the easier it is to flag. The embedding therefore places unfamiliar taxa at a distance that
tracks the taxonomy rather than merely displacing them, which is what makes a single scalar novelty
score meaningful across ranks.

It also disciplines the headline. `far` is rare (399 images) while `near` — a new species in a
familiar genus — is both the most common and the operationally decisive case, so a pooled AUROC is
flattered by the easy strata. Stated honestly: **0.94 for a novel family, 0.85 for a novel species in
a known genus** — against a plain cosine head that is near-chance (0.56) on precisely that case.
ArcFace's advantage is uniform across strata (+27.6 to +29.2 points), i.e. it improves the entire
difficulty range rather than only the easy end.

### 4.5 Open-set under domain shift

Repeating the benchmark where the novel species *also* come from a different camera (flemming):

| head | novelty only | novelty + domain shift |
|---|---|---|
| cosine | 0.601 | 0.574 |
| ArcFace × z-score | **0.9115** | **0.7272** |

The advantage survives but halves (+31.1 → +15.3 points). The logits show why: under shift the
*known* mean falls (32.6 → 20.6) toward the novel one (14.1), i.e. shift makes familiar species look
unfamiliar rather than making novel ones look more distinct. Novelty detection is therefore **not
domain-robust**, and domain adaptation is upstream of open-set rather than parallel to it. (Only 234
of 47,905 images are novel here, so treat this as directional, ±0.03.)

### 4.6 Rank abstention — and why backing off is subtler than it looks

Marginalisation makes rank abstention a *threshold*, not a second model: back off to genus when the
species posterior is unconfident. Evaluated on 629,742 held-out images with the single-head model:

| rank | precision at threshold 0 |
|---|---|
| species | 0.9344 |
| genus | 0.9701 |
| family | 0.9922 |

The obvious policy is to pick each rank's threshold from these global curves. **That is wrong**, and
the error is instructive. Conditioned on the images where the species head was *unconfident*
(15,453 images, species conf < 0.40), genus precision is **0.4874**, not 0.9701 — and family is
0.7935, not 0.9922. The coarse posterior is a deterministic function of the species posterior, so it
**inherits exactly the uncertainty that triggered the back-off**. "Genus is 97 % accurate" is true on
average and badly misleading for the cases where you actually need it.

Calibrating the coarse thresholds *on the subset they will serve* fixes this. At a 95 % precision
target throughout:

| returned rank | coverage | precision |
|---|---|---|
| species | 97.55 % | 0.9506 |
| genus | 0.41 % | 0.9502 |
| family | 1.22 % | 0.9366 |
| abstain | 0.82 % | — |

**99.18 % of images get an answer, 95.04 % of those are correct**, and the fallback ranks carry their
promised precision instead of a 49 % one. The practical lesson generalises beyond this dataset: in
any hierarchical back-off, thresholds must be calibrated **conditionally on reaching that rank**, or
the coarse levels' apparent reliability is borrowed from the easy cases they never see.

(Figure 5, `figures/fig5_rank_abstention.png`: per-rank precision/coverage curves, and the rank the
user receives as the species bar rises.)

Coverage/precision per rank from marginalised posteriors with per-level thresholds.

### 4.7 The margin and marginalisation interact — through calibration

Sections 4.1 and 4.3 present two independent contributions: coarse ranks by marginalisation, and an
angular margin composed with the z-score transform. Composing them is not free, and the way it fails
identifies the mechanism.

| | species | genus | family |
|---|---|---|---|
| single head, plain cosine (efficientnet\_v2\_s) | 0.9135 | 0.9606 | 0.9739 |
| single head + ArcFace × z-score | 0.9035 | 0.9491 | 0.9628 |
| Δ | −1.00 | **−1.15** | **−1.11** |
| same Δ, DINOv3-ConvNeXt-L (198 M, 10× larger) | −0.95 | **−1.21** | **−1.13** |

**The coarse ranks lose more than the fine rank they are derived from.** That is the diagnostic. If
the margin merely cost discriminative power, the species decision would absorb the damage and the
marginals would inherit it proportionally; instead the derived quantities degrade *further*.

The explanation is that marginalisation is **calibration-dependent** in a way argmax classification is
not. The coarse posterior $P(g) = \sum_{s \in g} P(s)$ depends on how mass is distributed across all
children of a parent, not only on which child ranks first. An additive angular margin optimises
against a deliberately harder target than the true label, which tightens the decision boundary while
distorting the posterior it induces. Sharper boundaries, worse sums.

Two pieces of evidence support this over the alternatives. First, the effect **replicates at 10×
parameter scale** (last row), so it is not a capacity artefact and will not be scaled away. Second,
the *converse* intervention produces the mirror image: supervising the marginals during training
(§2.4) leaves species macro-F1 **exactly unchanged** at 0.9135 while lifting genus by +0.27 and
family by +0.39 — a change with no discriminative component at all, acting purely on the sum.

**Practical consequence.** The margin is still worth its cost: it buys open-set AUROC 0.601 → 0.9068
on this architecture (§4.3), and a classifier that cannot flag an unseen taxon fails in a way
closed-set macro-F1 does not measure. But a system that both marginalises and uses a margin should
**supervise the marginals as well**, since the two interventions act on the same quantity in opposite
directions. We report this as an interaction to be managed rather than a solved problem; the combined
model is future work.

### 4.8 How much of the domain gap is nameable nuisance?

Section 4.2 measures a ~26-point drop from in-distribution to an external source. That number alone
does not say whether the shift is *nuisance* (blur, illumination, compression — removable by
augmentation) or *semantic* (pose, background, taxon-mix, labelling conventions). We separate them by
augmenting training with three hand-named nuisances and re-measuring.

| efficientnet\_v2\_s, single head + ArcFace × z-score | in-distribution | external | gap |
|---|---|---|---|
| standard augmentation | 0.9035 | 0.6437 | 25.98 |
| + motion blur, low light, JPEG quantisation | 0.8999 | **0.6836** | **21.63** |

**Nameable nuisance is worth about four points, or 17 % of the gap.** The trade is unusually
favourable — 0.36 points of in-distribution accuracy for 3.99 under shift, an 11:1 ratio, at no
parameter or inference cost — and larger than any architectural change we measure in this paper. It
is also *bounded by construction*: each transform encodes a guess about what differs between the
domains, so the method cannot address a shift nobody anticipated, and the residual is invisible.

The result is therefore best read as a **measurement of the gap's composition** rather than a method
contribution. Roughly one sixth of cross-source degradation is removable by naming nuisances; five
sixths are not. That is what motivates treating cross-source generalisation, rather than closed-set
accuracy, as the open problem.

### 4.9 Open-set scoring rules do not transfer across model scale

All open-set numbers above use $-\max_j z_j$, the standard max-logit score. Measured that way, a 10×
larger backbone appears to *lose* 7.7 points of AUROC, and the three evaluation axes appear to rank
models in opposite orders. Both conclusions are largely artifacts of that choice.

Recomputing five rules from the same forward pass, on the same images and embeddings:

| model | params | `max` | `energy` | `entropy` | `margin` | `msp` |
|---|---|---|---|---|---|---|
| efficientnet\_v2\_s | 20 M | **0.9068** | 0.9064 | 0.9047 | 0.8979 | 0.8953 |
| + domain aug | 20 M | **0.9010** | 0.9005 | 0.9008 | 0.8945 | 0.8917 |
| DINOv3-ConvNeXt-L | 198 M | 0.8298 | 0.8287 | 0.8813 | 0.8807 | **0.8904** |
| + domain aug | 198 M | 0.8132 | 0.8118 | 0.8802 | 0.8789 | **0.8893** |

**The best rule inverts with scale.** Max-logit wins at 20 M and is beaten by max-softmax-probability
by 6.1–7.6 points at 198 M. Using each model's best rule, the apparent capacity penalty falls from
**7.70 to 1.64 points**.

The mechanism is a saturation argument. Since $\mathrm{msp} \approx z_{\max} -
\operatorname{logsumexp}(z)$, max-logit asks *how strongly does the best prototype match*, while MSP
asks *how much better than the alternatives*. In a well-fitted embedding every input — known or novel
— attains a high cosine to some prototype, so the absolute maximum saturates and stops
discriminating; what still separates them is whether one prototype **dominates**. Only the
shape-sensitive rules (MSP, entropy, top-2 margin) see that, and they are exactly the three that gain.
Energy tracks max-logit to within 0.15 points everywhere, as it must: $\operatorname{logsumexp}$ is
dominated by its largest term for peaked logits and is therefore near-monotone in $z_{\max}$.

**Recommendation.** Open-set results should be reported *with the scoring rule named*, and rules
should be re-selected whenever model capacity changes materially. A rule validated on a small model
is not a property of the method; carrying one silently across a scale change is enough to invert a
published ranking.

### 4.10 What the axes then say

With each model's best rule, the picture is a mild disagreement rather than an inversion:

| model | in-distribution | external | open-set |
|---|---|---|---|
| efficientnet\_v2\_s | 0.9035 | 0.6437 | **0.9068** |
| + domain aug | 0.8999 | 0.6836 | 0.9010 |
| DINOv3-ConvNeXt-L | **0.9216** | 0.6616 | 0.8904 |
| + domain aug | **0.9216** | **0.7101** | 0.8893 |

The largest augmented model leads two axes and gives up 1.75 points on the third; the smallest model
leads open-set detection. Selecting on in-distribution macro-F1 alone would still pick a model 6.6
points worse on external data, which is the practical argument for reporting all three — but it would
not, as an earlier version of this analysis claimed, pick the worst deployable system.

We report this correction explicitly because it illustrates the failure mode the section above
describes: an inherited default that was never visible as a decision propagated through every
downstream comparison until it was checked directly.

## 5. Discussion

- Encoding a taxonomy in the *architecture* buys nothing here; using it for *inference-time
  reasoning* (marginalisation, abstention) buys consistency for free and better coarse accuracy.
- Metric-learning margins are usually sold for retrieval and face verification. Their real value in
  a classification pipeline may be **calibration of the open-set score**, provided the logit scale is
  dimension-aware.
- The field's benchmark culture rewards closed-set accuracy on the training distribution; our numbers
  suggest that is close to exhausted while the deployment-relevant quantities (domain shift, novelty)
  are not.

## 6. Limitations

Single taxonomic domain and one 3-level hierarchy. Open-set results are on a *no-domain-shift*
benchmark; the harder novelty-plus-shift case is _(pending)_. $m = 0.3$, $s = 30$ were first guesses,
so 0.9115 is a floor, not a tuned optimum. Distillation experiments use one student family.

## References _(to complete)_

Deng et al., *ArcFace*, CVPR 2019 · Wang et al., *NormFace*, ACM MM 2017 · Liu et al., *ConvNeXt V2*,
CVPR 2023 · Oquab et al., *DINOv2* / *DINOv3* · Jordan & Jacobs, hierarchical mixtures · Hinton et
al., *Distilling the knowledge in a neural network*, 2015.
