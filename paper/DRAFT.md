# Knowing what you don't know: calibrated open-set hierarchical classification for long-tailed species identification

**Status:** working draft (2026-08-05). Numbers are from `RESULTS.md`; the reasoning behind each is
in [`../journal/`](../journal/). Sections marked _(pending)_ await runs that are in flight.

---

## Abstract (draft)

Automated species identification is usually framed as fine-grained classification over a fixed label
set. We argue this framing is the wrong one for deployment, and support the argument with results on
a 12,041-species Lepidoptera benchmark evaluated on three axes: a held-out fold, an external
camera-trap source, and open-set detection of unseen taxa. First, **hierarchical prediction heads do
not help** — an independent multi-head cosine classifier, a parent-conditioned head and an
autoregressive head are all matched or beaten by a *single* species head whose coarse ranks come from
marginalising its own posterior — but the finer statement is that coarse **parameters** hurt while
coarse **supervision** helps, and only the second is visible off the training distribution. Second,
in-distribution accuracy is close to saturated (0.9316) yet falls to **0.69 on data from a different
source**, and interventions selected on the first axis routinely invert on the others: square-root
resampling buys 1.9 points in-distribution and costs 2.9 under shift; the best open-set scoring rule
changes with model capacity, a 6–7.6 point effect that inverted a published ranking of our own.
Third, and most usefully, **unlabelled** target-domain images are the strongest lever we find:
self-training on machine-generated labels beats 12,230 human labels, costs nothing in-distribution,
and lets a 20 M model outperform a 198 M one — with a sharp interior optimum in the *share* of target
data, beyond which adaptation silently becomes memorisation. We also show an **additive angular
margin composed with a dimension-aware z-score transform** turns near-chance novelty detection
(AUROC 0.601) into usable detection (**0.9115**) for 0.4 points of accuracy, and that the resulting
classifier matrix can be replaced at inference by class centroids for 0.29 points — though it is not
low-rank, because the margin spends dimensions rather than economising on them. We release the
package, the models and the reproduction recipes.

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

## 1b. Related work _(drafted from memory — every citation marked [VERIFY] needs checking against the source before submission; author lists, years and venues are not reliable)_

**Cosine classifiers and angular margins.** Normalised-prototype classifiers and additive angular
margins are standard in face recognition — SphereFace, CosFace and ArcFace [Deng et al., CVPR 2019,
VERIFY] — where the goal is an embedding whose *distances* are meaningful, not merely a decision
boundary. NormFace [Wang et al., ACM MM 2017, VERIFY] gives the scale-factor lower bound we invoke in
§2.3. Our contribution is not the margin but its **composition with a dimension-aware calibration**:
applied to raw cosines the margin costs 3.3 points of accuracy for AUROC 0.732, and composed with the
z-score transform it costs 0.4 for 0.9115. We are not aware of prior work reporting that the margin's
accuracy/open-set trade-off is an artefact of the scale it is applied on.

**Hierarchical classification.** The literature encodes the taxonomy in the architecture —
hierarchical softmax, conditional/parent-gated heads, autoregressive decoders over the label path
[VERIFY: representative citations needed]. We test three such heads against marginalisation and find
none of them pay (§4.1). The distinction we draw — coarse *parameters* hurt while coarse
*supervision* helps, visible only off the training distribution — does not appear in that literature,
which to our knowledge evaluates in-distribution throughout.

**Long-tailed recognition.** Square-root resampling [Mahajan et al., ECCV 2018, VERIFY], logit
adjustment [Menon et al., ICLR 2021, VERIFY], Balanced Softmax [Ren et al., NeurIPS 2020, VERIFY],
LDAM [Cao et al., NeurIPS 2019, VERIFY], class-balanced reweighting by effective number [Cui et al.,
CVPR 2019, VERIFY] and the decoupling/τ-normalisation line [Kang et al., ICLR 2020, VERIFY]. Two
observations. Balanced Softmax and logit adjustment at τ=1 are the **same objective** (§4.x), which
the two papers do not note. And the cosine head already implements τ-normalisation at τ=1 by
construction, so results transferred from a linear-classifier setting should not be expected to hold.
Our contribution here is the **evaluation axis**: these methods are benchmarked on in-distribution
held-out splits (CIFAR-LT, ImageNet-LT, iNaturalist) essentially without exception, and we show their
ranking inverts under source shift.

**Open-set recognition and OOD detection.** Max-softmax-probability [Hendrycks & Gimpel, ICLR 2017,
VERIFY], energy scores [Liu et al., NeurIPS 2020, VERIFY], and the max-logit family. §4.9 reports
that the best rule among these **changes with model capacity and with the head's output convention**
— max-logit at 20 M, MSP at 198 M, entropy for log-probability heads — with a 6–7.6 point spread. We
have not seen this reported, and it is the kind of result that invalidates comparisons rather than
adding to them.

**Domain adaptation and self-training.** Pseudo-labelling with confidence thresholds is long
established [Lee, ICML workshop 2013, VERIFY], with FixMatch and noisy-student as modern
representatives [Sohn et al. 2020; Xie et al. 2020, VERIFY]. Our finding of a sharp interior optimum
in the *share* of target-domain data (§4.x) — and specifically that transfer to unseen classes falls
monotonically as that share rises — is, as far as we know, not documented; the usual concern is
label noise rather than dosage.

**Calibration.** Temperature scaling [Guo et al., ICML 2017, VERIFY]. We use it as shipped, and note
in §4.7 that marginalisation makes calibration *load-bearing* rather than cosmetic: summing a
poorly-calibrated posterior gives a wrong parent even when the top-1 is right.

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

*Implementation note, stated because the derivation assumes it.* $Z$ is defined on $[-1,1]$ and is
implemented with a clamp. Our prototype rows are constrained to unit norm by construction, but in
trained checkpoints they drift (mean norm 1.08 for the margin head, 1.77 without it), so a fraction
of pre-activations fall outside $[-1,1]$ and saturate. We measured the consequence: **no image has
two saturated logits**, so the prediction is never affected, and the top-1 saturates on **0.37 %** of
images for the margin head and none for the plain one. The accuracy results are therefore exact; the
calibration argument of this section carries that 0.37 % exception.

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

applied recursively up the taxonomy. This is exact, adds no parameters, and makes the levels
**probabilistically coherent**: the reported coarse posterior *is* the sum of the fine one, so the
two are statements from a single distribution. Independent per-level heads have no such relation and
can report a genus probability incompatible with their own species distribution — they contradict
each other's argmax on 1.81 % of images.

Coherence is what the downstream machinery needs: rank abstention (§4.6) compares $P(\mathrm{genus})$
against $P(\mathrm{species})$, which is only meaningful within one distribution. Note that coherence
does **not** imply argmax agreement — $\max$ and $\sum$ do not commute over a partition, so a
confident species can be outvoted by many diffuse siblings of another genus. When that happens the
coarse answer is arguably the better one, since aggregating sibling evidence is exactly what the
marginal is for.

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

### 4.1 Hierarchical heads: what the taxonomy should and should not touch (C1, C2)

Four ways of getting a label at every level, all sharing the same backbone, bottleneck and cosine
prototypes, differing only in how the taxonomy enters. Fine = species (12,041 classes); coarse =
genus (4,333) and family (102).

| head | coarse **parameters** | coarse **supervision** | information flow |
|---|---|---|---|
| multi-head independent | yes, one layer per level | yes, one CE per level | none between levels |
| parent-conditioned ("conditional") | yes | yes | **top-down**: parent conditions child |
| autoregressive | yes (decoder) | yes | top-down, sequential |
| single head + marginal *inference* | **no** | **no** | bottom-up, at test time only |
| single head + marginal *supervision* | **no** | yes | **bottom-up**, during training |

The parent-conditioned head corrects each level's independent logits $M^{(i)}$ by the parent's
already-conditioned score, so a child cannot be more confident than its parent's evidence allows:

$$C^{(\text{top})} = M^{(\text{top})}, \qquad
C^{(i)} = M^{(i)} + \operatorname{gather}_{\pi}\!\Bigl(C^{(i+1)} - \operatorname*{log\,sum\,exp}_{\text{siblings}} M^{(i)}\Bigr)$$

i.e. $P_{\text{cond}}(\text{child}) = P(\text{child}) \cdot P_{\text{cond}}(\text{parent}) /
P(\text{siblings})$ in log-space. Marginalisation (§2.4) is the exact mirror: information flows
*up*, and no coarse parameters exist at all.

| head | species | genus | family | **external (shifted)** |
|---|---|---|---|---|
| multi-head independent | 0.9110 | 0.9587 | 0.9708 | **0.6503** |
| parent-conditioned | 0.8845 | 0.9471 | 0.9683 | 0.6213 |
| autoregressive | 0.69–0.73 | — | — | — |
| single head + marginal inference | **0.9135** | 0.9606 | 0.9739 | 0.6293 |
| single head + marginal supervision | **0.9135** | **0.9633** | **0.9778** | 0.6434 |

Run-to-run spread, measured by retraining one configuration unchanged: **0.0000** species, 0.0005
genus, 0.0024 family, **0.0069** external.

Three readings, and the third is the one that matters.

**Coarse *parameters* do not help.** Every head that owns genus/family layers is beaten
in-distribution by one that does not. Conditioning is worse still (−2.9 pp species) **and worst of
all four under shift** (0.6213): constraining a child by its parent propagates the parent's errors
downward, and the parent is the *easier* problem only because it is coarser, not because it is more
reliable. Domain shift degrades the parent too, so the conditioning amplifies a now-unreliable prior
— the one head whose in-distribution and shifted rankings agree, both last.

**Coarse *supervision* does help, but only where in-distribution accuracy cannot see it.** Adding the
marginal losses leaves species **exactly unchanged** (0.9135 → 0.9135, four decimals) and lifts genus
by 0.27 pp — a change with no discriminative component, acting purely on how mass is distributed
within a parent. On the shifted benchmark the same intervention is worth **+1.41 pp**, twenty times
its in-distribution species effect.

**The two questions therefore have opposite answers, and the standard protocol can only see one.**
Dropping coarse parameters is right (+0.25 pp in-distribution, smaller, coherent); dropping coarse
supervision is wrong (−2.10 pp externally, 3× the noise floor). An evaluation restricted to a
held-out fold of the training distribution — the norm in this literature — measures the first and is
blind to the second. Marginal supervision keeps the parameter saving and recovers most of the
robustness, landing within one noise floor of the multi-head; whether the residual 0.69 pp is real
is below what our replication can resolve.

### 4.2 Accuracy saturates; generalisation does not (C4)

| model | in-distribution | external dataset |
|---|---|---|
| efficientnet\_v2\_s | 0.9110 | — |
| ConvNeXtV2-L @320 | **0.9316** | 0.6950 |
| DINOv3-ConvNeXt-L @320 | 0.9311 (≈2× faster to train) | — |

A ~23-point drop on data from a different source. Distillation into a small student saturates at
~0.88 **regardless of teacher quality** (0.8786 from a 0.911 teacher; 0.8756 from a 0.9316 teacher),
i.e. student capacity, not teacher accuracy, is the binding constraint.

### 4.3 What the angular margin actually does for open-set detection

> **This section was rewritten on 2026-08-06 and its earlier claim is retracted.** It previously
> reported that composing the margin with the z-score transform takes open-set AUROC from 0.601 to
> 0.9115 for 0.4 points of accuracy. That compared the margin head's *best* scoring rule against the
> plain head's *worst*: the plain head's 0.601 came from a max-logit-only script, and max-logit is
> its weakest rule by 27 points. The corrected comparison is below.

Scoring both heads with five rules on the logits each actually emits (§4.9):

| rule | plain cosine | ArcFace × z-score |
|---|---|---|
| entropy | **0.8990** | 0.9047 |
| max-softmax-probability | 0.8819 | 0.8953 |
| top-2 margin | 0.8423 | 0.8979 |
| max-logit | 0.6258 | **0.9068** |
| energy | 0.6149 | 0.9064 |
| **best** | **0.8990** | **0.9068** |
| closed-set macro-F1 | **0.9135** | 0.9035 |

**A plain cosine head detects unseen species at 0.899.** It was never near chance; it was being read
with the wrong rule. Given both heads their best readout, the margin is worth **0.78 points of AUROC
and costs 1.00 point of accuracy** — which is not the trade the earlier version of this section
claimed, and arguably not a trade worth making on this benchmark.

**What the margin does do is relocate the signal.** The plain head's best rule is *entropy* and its
worst is *max-logit*, a 28.4-point spread; the margin head's best is *max-logit* and its spread is
1.2 points. So the margin moves open-set information out of the **shape** of the logit distribution
and into the **magnitude** of the top score, and in doing so makes the choice of readout nearly
irrelevant. For a deployed system that insensitivity has real value — a score that does not depend on
picking the right rule is one that can be shipped without tuning it — but it is a different and
smaller claim than "the trade-off dissolves".

*(A related artefact, reported because it was predicted backwards: the plain head's logits are pinned
to the transform's clamp on 67 % of entries, and we expected this to destroy the shape its best rule
reads. Scoring on pre-clamp values instead **lowers** entropy AUROC from 0.899 to 0.599. The clamp
removes noise from the tail rather than signal.)*

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

### 4.11 Unlabelled target-domain data beats labels, capacity, and augmentation

The gap of §4.2 is a *source* gap: training images are museum-style specimen photographs, deployment
images are camera-trap frames. Three interventions address it, and we can price them against each
other because all three were measured on the same held-out split.

The split matters and is not the one used elsewhere in this paper. Self-training consumes the same
trap images that serve as the external benchmark, so we partition them by **capture group (trap,
night)** — nights running midday-to-midday, since moths fly across midnight — into an *adaptation*
set (27,230 images) and a **probe** set (15,200) that no training run touches. Fifteen percent of
species are withheld from adaptation entirely, giving a **held-out-species** subset (2,455 images,
58 taxa) that separates domain adaptation from specialisation on the taxa being adapted to.
Run-to-run floors, measured by retraining one configuration unchanged: probe 0.0041, held-out 0.0052.

| intervention (efficientnet\_v2\_s, 20 M) | in-distribution | probe | held-out species |
|---|---|---|---|
| none | 0.8999 | 0.6912 | 0.6974 |
| hand-named nuisance augmentation (§4.8) | — | — | — |
| 500 real target labels | — | 0.7060 | — |
| 2,500 real target labels | — | 0.7196 | — |
| 12,230 real target labels | — | 0.7568 | — |
| **self-training, 12,230 pseudo-labels** | **0.9003** | **0.7706** | **0.7704** |

**Self-training on machine-generated labels beats 12,230 human labels** (0.7706 vs 0.7568) and costs
nothing in-distribution (+0.04, inside the 0.0000 floor). At matched size and share, real labels *are*
better than 98.15 %-accurate pseudo-labels by 2.14 points — label quality is not free — but the
label-free method has a hyperparameter the labelled one cannot exploit as cheaply, which is the
subject of the next paragraph.

**There is a sharp interior optimum in the *share* of target data, and it is low.** Replicating the
pseudo-labelled images to a target fraction of the training set:

| share of training | probe | held-out species | fraction of gain transferring |
|---|---|---|---|
| 0.39 % (no replication) | 0.7354 | 0.7508 | 121 % |
| **2 %** | **0.7706** | **0.7704** | **92 %** |
| 6 % | 0.7370 | 0.7231 | 56 % |
| 10 % | 0.7159 | 0.7042 | 39 % |

Two things here are, to our knowledge, unreported. First, **0.39 % of training — one image in 250 —
already buys 97 % of the gain**: what unlabelled target data supplies is not gradient volume but
coverage of a region of input space the source distribution never visits, and a handful of samples
suffices to mark that region. Second, **the fraction of the gain that transfers to classes the
adaptation never saw falls monotonically with the share**, from 121 % to 39 %. Over-replication does
not merely stop helping; it converts domain adaptation into memorisation of the specific images.
At 2 % the probe and held-out-species scores are equal (0.7706, 0.7704) — the gain transfers
entirely, with no memorisation signature.

**Adaptation dominates capacity.** A 20 M model with unlabelled target data (0.7706) beats a 198 M
model without it (0.7209) on the probe benchmark. They are complementary rather than substitutes —
the larger model leads on held-out species (0.7559 vs 0.7231), so capacity buys generalisation to
unseen *taxa* while adaptation buys generalisation to unseen *conditions* — and combining them, while
also removing the resampling of §4.x, gives the best model we obtain: **probe 0.7798, held-out
0.7816**.

*(The dose effect is itself capacity-dependent: moving from 6 % to 2 % is worth 3.36 points at 20 M
and 0.02 at 198 M. A larger model can memorise the replicated images and fit everything else, so the
probe cost disappears — but the transfer cost does not, and the held-out gap remains at 1.04 points.)*

### 4.12 The classifier matrix is replaceable at inference, and not compressible

A practical obstacle to scaling this method to the ~1 M species of a global taxonomy is the
classifier itself: at $d = 1280$ a prototype matrix is 5.1 GB in fp32, plus twice that in optimiser
state. Two measurements bear on it.

**Class centroids match trained prototypes.** Replacing the learned matrix at inference with the mean
training embedding per class costs **0.29 points** (0.9077 vs 0.9105). Single means outperform both
k-means centroids ($k=3$, 0.8988) and medoids (0.8960), indicating classes are unimodal blobs rather
than multimodal — which is what an angular margin is designed to produce, and it means the cheapest
summary is also the best. Centroids are computed from data rather than trained, can be extended
incrementally as taxa are added, and are directly indexable for approximate nearest-neighbour search,
so the inference-time matrix need not exist.

**But the matrix is not low-rank.** Its singular spectrum requires **rank 1035 of 1280** for 90 % of
the energy; truncating to rank 512 costs 0.35 points, 256 costs 1.06, and 128 costs 3.14. Factorising
the head is therefore not a route to a 1 M-class model. The reason is visible in the objective: an
angular margin *pushes classes apart*, so it spends dimensions rather than economising on them. **The
margin and low-rank compression want opposite things** — a trade-off worth stating for anyone
combining metric learning with extreme classification.

### 4.13 Almost everything that matters is a classifier-stage concern

The results of §4.1, §4.x (resampling), §4.11 and §4.12 were obtained separately and answer different
questions. Placed together they say the same thing, and it is the strongest through-line we have.

**Coarse supervision.** Per-level classifier *parameters* hurt (§4.1); the per-level *losses* help,
and only visibly off the training distribution.

**Long-tail resampling.** Applied to the data the backbone sees, square-root oversampling buys 1.9
points in-distribution and costs 2.9 under source shift. Applied only to a classifier retrained on a
frozen representation (cRT; Kang et al. [VERIFY]), it recovers **+1.19** of the in-distribution gain
while scoring **+2.46 above** the fully-resampled model externally. The accuracy/robustness trade of
§4.x is therefore not intrinsic — it is an artefact of where the rebalancing is applied.

**Domain adaptation.** Freezing the representation and adapting only the classifier for 2 epochs on
pseudo-labelled target images captures **83 %** of what full self-training gives on held-out capture
groups, and **89 %** of its transfer to unseen species. From a representation trained *without* any
domain-mimicking augmentation the figure is the same (0.7515 vs 0.7572), so the features were already
adequate: hand-authored augmentation and classifier adaptation are **substitutes**, not complements —
the augmentation is worth +4.75 points alone and +0.57 once the classifier is adapted.

**The classifier itself.** Its learned prototype matrix can be replaced at inference by class
centroids for 0.29 points (§4.12).

**Assembled, the recipe simplifies rather than accumulating:**

| stage | cost | in-distribution | external (probe) |
|---|---|---|---|
| representation: no resampling, no domain augmentation | 5 epochs | 0.8949 | 0.6445 |
| + classifier rebalanced (frozen trunk) | 2 epochs | 0.9068 | 0.6539 |
| + classifier adapted on pseudo-labels (frozen trunk) | 2 epochs | **0.9081** | **0.7541** |
| *end-to-end training with the same pseudo-labels, for comparison* | *5 epochs* | *0.9003* | *0.7706* |

The staged recipe is **0.78 points better in-distribution** and 1.65 worse externally than training
end to end, at a fraction of the cost — and, more usefully, its two adaptation stages are *repeatable
per deployment*. A new camera does not require a new model; it requires a new classifier, which is
minutes on a frozen trunk and needs no labels.

We report this as an empirical regularity on one problem rather than a law. But four independent
questions resolving the same way, with the constructive assembly then behaving as predicted, is a
stronger form of evidence than any one of them alone — and it suggests that for fine-grained
recognition under shift, **the representation is the robust, expensive, inert component and the
classifier is the cheap, swappable one where interventions belong.**

## 5. Discussion

**What we set out to test, and what we found instead.** The project began as a comparison of
hierarchical prediction heads. That comparison is a null result, and the interesting structure turned
out to lie one level down: not *how the taxonomy enters the architecture*, but *which component of
the model each intervention should act on*. Four independent questions — coarse supervision,
long-tail resampling, domain adaptation, and the classifier matrix itself — all resolved toward the
classifier (§4.13).

**The methodological finding may outlast the empirical ones.** Three times, a conclusion this project
held was overturned not by a better model but by a better measurement:

1. An open-set scoring rule chosen on a 20 M model does not transfer to a 198 M one; reading every
   model with one rule understated the large ones by 6–7.6 points and inverted a ranking (§4.9).
2. That same oversight, applied to our own headline, produced a claimed 31-point advantage for an
   angular margin that is **0.78 points** when each head is read with its best rule (§4.3).
3. Macro-F1 does not decompose over subsets, so two benchmarks over the *same images* can rank two
   models oppositely with both results correct (§4.x).

Each was a comparison made with a default nobody had recorded as a decision. We report them because
the corrected numbers are the paper's numbers, and because the failure mode is not specific to us:
**a baseline that everyone quotes is the one least likely to be re-measured.**

**What we would tell a practitioner.** Train one representation, cleanly, without resampling and
without hand-authored domain augmentation. Then do the cheap work: rebalance the classifier if the
in-distribution metric matters, adapt the classifier on unlabelled target images if deployment does,
and read novelty with a rule chosen on *your* model rather than inherited. Report accuracy, external
accuracy, and open-set detection separately, because they disagree — and never compare two of them
across different evaluation sets.

**What we do not claim.** That the classifier/representation split generalises beyond fine-grained
recognition under source shift; that self-training's advantage over labels survives at much larger
label budgets (it is still rising at 12,230); or that the angular margin is not worth its cost — we
show only that its cost and benefit are both about one point, and that its real advantage is
insensitivity to the readout rather than a better score.

## 6. Limitations

Single taxonomic domain and one 3-level hierarchy. Open-set results are on a *no-domain-shift*
benchmark; the harder novelty-plus-shift case is _(pending)_. $m = 0.3$, $s = 30$ were first guesses,
so 0.9115 is a floor, not a tuned optimum. Distillation experiments use one student family.

## References _(to complete)_

Deng et al., *ArcFace*, CVPR 2019 · Wang et al., *NormFace*, ACM MM 2017 · Liu et al., *ConvNeXt V2*,
CVPR 2023 · Oquab et al., *DINOv2* / *DINOv3* · Jordan & Jacobs, hierarchical mixtures · Hinton et
al., *Distilling the knowledge in a neural network*, 2015.
