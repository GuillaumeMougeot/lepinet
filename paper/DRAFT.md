# Knowing what you don't know: calibrated open-set hierarchical classification for long-tailed species identification

**Status:** working draft (2026-08-28). Numbers are from `RESULTS.md`; the reasoning behind each is
in [`../journal/`](../journal/). Sections marked _(pending)_ await runs that are in flight.

Audited against the journal on 2026-08-28
([`../journal/2026-08-28-what-the-paper-is-still-missing.md`](../journal/2026-08-28-what-the-paper-is-still-missing.md));
§4.13 (long-tail rebalancing) and §4.14 (foundation models and benchmark contamination) were written
in that pass, and all six unresolved section placeholders now point somewhere. **Remaining for the
authors:** fact-check the `[VERIFY]` citations in §1b and the reference list, redraw `fig4`
per-head-best-rule (see `figures/README.md`), and re-score §4.5 with each head's own scoring rule.

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
source**, and interventions selected on the first axis routinely invert on the others: the long-tail
literature's methods degrade cross-source accuracy **monotonically in how hard they reweight the
tail** (a 9.5-point spread over which the in-distribution column does not order at all), and the best
open-set scoring rule changes with model capacity, a 6–7.6 point effect that inverted a published
ranking of our own. Both inversions are invisible to the standard protocol, which evaluates on a
held-out fold of the training distribution. Third, and most usefully, **unlabelled** target-domain
images are the strongest lever we find:
self-training on machine-generated labels beats 12,230 human labels, costs nothing in-distribution,
and lets a 20 M model outperform a 198 M one — with a sharp interior optimum in the *share* of target
data, beyond which adaptation silently becomes memorisation. We also show that an **additive angular
margin composed with a dimension-aware z-score transform** does not so much improve novelty detection
as **relocate its signal**: given each head its own best scoring rule the margin is worth 0.78 points
of AUROC (0.8990 → **0.9068**) for 1.00 point of accuracy, but it collapses the spread across
scoring rules from 28.4 points to 1.2, so the readout no longer has to be tuned. Finally, the
resulting classifier matrix can be replaced at inference by class centroids for 0.29 points — though
it is not low-rank, because the margin spends dimensions rather than economising on them. Finally, we
report a contamination result with implications beyond this dataset: **two thirds of our held-out
test fold, 413,865 images, lie inside a biological foundation model's training set by exact
occurrence identifier** rather than by species name, which is the check the field actually performs.
On a decontaminated fold that model, fine-tuned, is the better representation — a *frozen* probe
understates it by 7 points — and once our adaptation recipe is applied to each, the two converge.
Our contribution is therefore the recipe rather than the encoder, and the ceiling is set by
target-domain data. We release the package, the models and the reproduction recipes.

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
- Contributions:
  - **(C1)** a negative result on hierarchical heads with a like-for-like protocol, refined into the
    distinction that coarse *parameters* hurt while coarse *supervision* helps — and only the second
    is visible in-distribution (§4.1);
  - **(C2)** **the classifier, not the representation, is where interventions belong.** Four
    independent questions — coarse supervision, long-tail rebalancing, domain adaptation, and the
    prototype matrix itself — resolve the same way (§4.15). This is the paper's spine;
  - **(C3)** **unlabelled target-domain data is the strongest and cheapest lever**, beating human
    labels, capacity and augmentation, with a sharp interior optimum in its *share* beyond which
    adaptation becomes memorisation (§4.11);
  - **(C4)** **the long-tail literature's ranking inverts under source shift**, monotonically in how
    hard a method reweights the tail — and the trade is an artefact of *where* the rebalancing is
    applied, not of rebalancing (§4.13);
  - **(C5)** **a benchmark-contamination result and a cheap method for detecting it**: two thirds of
    our test fold is inside a foundation model's training set by exact occurrence identifier, and a
    frozen probe understates that model by 7 points (§4.14);
  - **(C6)** **ArcFace × z-score**, with the analysis of why the two compose, and the finding that
    the margin *relocates* open-set signal rather than creating it (§2.3, §4.3);
  - **(C7)** a measured generalisation gap **decomposed** into the part removable by naming nuisances
    (17 %) and the part that is not (§4.8);
  - **(C8)** an interaction result — an angular margin degrades *marginalisation* more than
    classification, because summing a posterior is calibration-dependent, replicated across a 10×
    scale change (§4.7);
  - **(C9)** an open, reproducible package, with every negative result and retraction recorded.

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
observations. Balanced Softmax and logit adjustment at τ=1 are the **same objective** (§4.13), which
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
in the *share* of target-domain data (§4.11) — and specifically that transfer to unseen classes falls
monotonically as that share rises — is, as far as we know, not documented; the usual concern is
label noise rather than dosage.

**Biological foundation models, and benchmark contamination.** BioCLIP and BioCLIP-2 [Stevens et al.,
CVPR 2024; 2025, VERIFY] train CLIP-style encoders on TreeOfLife-10M/200M, assembled largely from
GBIF, iNaturalist and EOL. The standard evaluation of such a model on a downstream taxonomic
benchmark is a frozen linear probe. We report two problems with that protocol (§4.14). First, the
benchmarks are drawn from the same public archives as the pretraining corpus, and we measure a
**65.4 % image-level overlap with our own held-out test fold by exact GBIF occurrence identifier**;
the overlap checks we have seen reported are taxonomic, which is both weaker and more reassuring.
Data-contamination audits of this kind are routine in NLP [Dodge et al., EMNLP 2021, VERIFY] and, as
far as we know, not yet routine here. Second, the frozen probe itself understates the representation
by 7 points relative to fine-tuning, which is large enough to reverse a published-style comparison.

**Calibration.** Temperature scaling [Guo et al., ICML 2017, VERIFY]. We use it as shipped, and note
in §4.7 that marginalisation makes calibration *load-bearing* rather than cosmetic: summing a
poorly-calibrated posterior gives a wrong parent even when the top-1 is right.

## 2. Method

### 2.1 Cosine classification head

A backbone $f_\theta$ maps an image to $z \in \mathbb{R}^{d}$ through a bottleneck, then
$e = z/\|z\|$. Each class $j$ owns a prototype $w_j$ constrained to $\|w_j\| = 1$
(weight-norm with the row norm frozen), so the pre-activation is a cosine similarity

$$
\cos\theta_j  =  e^{\top} w_j  \in  [-1, 1].
$$

### 2.2 The z-score transform (and why a raw cosine is a poor logit)

In high dimension the cosine between (near-)random unit vectors concentrates sharply: for $e$
uniform on $S^{d-1}$, $\cos\theta$ has mean 0 and standard deviation $1/\sqrt{d}$, so with
$d = 1280$ virtually all class scores fall inside a band of width $\approx 0.1$. Feeding that into a
softmax gives a nearly uniform posterior regardless of the embedding's quality — the classic reason
cosine classifiers need a scale factor.

Instead of an arbitrary multiplier we apply the transform that maps this concentrated distribution
onto an approximately standard normal one:

$$
Z(\cos\theta)  =  \sqrt{d-2} \Bigl(\arccos(-\cos\theta) - \tfrac{\pi}{2}\Bigr).
$$

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

$$
\ell_j  =  \begin{cases} Z\bigl(\cos(\theta_y + m)\bigr) & j = y \\ Z(\cos\theta_j) & j \ne y \end{cases}
$$

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

$$
\log P(g)  =  \mathrm{logsumexp}_{\{ s : \pi(s)=g \}} \log P(s),
$$

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

**Numerics.** Mixed precision interacts with the cosine head and the interaction is easy to
misdiagnose. Runs use fp16 autocast with **the head forced to fp32 inside an adapter**; without that
protection the head overflows, and an autoregressive baseline trained visibly broken until its
precision was changed — a failure that presents as a modelling bug rather than a numerical one.
ArcFace with a positive margin is fp16-unstable even with the adapter, because $s\cos\theta$
overflows and $\arccos$ then returns NaN, so **every margin run in this paper uses bf16** and the
configuration layer warns on the unsafe combination rather than trusting the operator to remember.

Two caveats are stated rather than hidden. The +4.7 row bundles three changes, so their individual
contributions are unrecoverable; and the 10-epoch run was still improving when it stopped, so the
5-epoch budget used throughout §4 is a comparison convention, not a converged optimum — absolute
numbers here understate what the architecture can reach, while the *differences* between arms, which
is what we claim, are measured at matched budget.

### 3.2 Four benchmarks, and why they are not interchangeable

Results are reported on several evaluation sets. They answer different questions and **differences
are only meaningful within a column.**

| name | images | species | what it is |
|---|---|---|---|
| **in-distribution** | 629,742 | 12,041 | held-out fold of the training distribution |
| **full trap** | 47,905 | 486 | every camera-trap image. *Contaminated for anything trained on trap data* |
| **probe** | 15,200 | 368 | whole (trap, night) capture groups held out of adaptation. The honest shifted column |
| **probe held-out species** | 2,455 | 58 | probe restricted to species adaptation never saw. Separates domain adaptation from specialisation on the adapted taxa |
| **open-set** | 3,000 | 566 unseen | novel species drawn from the *same* image distribution, isolating novelty from shift |

**Macro-F1 does not decompose over subsets, so two benchmarks over the same images can rank two
models oppositely with both results correct.** One model ties another on all 47,905 trap images
(+0.0002) and beats it by **2.03 points** on a 15,200-image *subset* of them, because the full set
averages over 486 species at $1/486$ each and the subset averages over its 368 at $1/368$. Neither
number is wrong. This is a property of any per-class average, and it is why we never quote a
difference computed across two evaluation sets — a discipline that took one misdiagnosed
"regression" to adopt.

**A benchmark's exclusivity has to be recorded in the split, not assumed in the pipeline.** The trap
corpus was simultaneously our external benchmark and the only source of unlabelled target images, and
nothing recorded that those roles conflict; self-training would have trained on its own test set and
reported a better number for it. The grouped (trap, night) partition above exists to make that
conflict structural rather than remembered. §4.14 reports the same failure one level up, where the
corpus serving two roles is a public archive and the second role is somebody else's pretraining.

## 4. Results

### 4.0 Summary of models

The sections below each vary one factor. For orientation, the models the paper refers to repeatedly:

| model | what it is | in-dist | probe | probe-HO | open-set | useful-answer |
|---|---|---|---|---|---|---|
| baseline | effnetv2_s, single head + marginals, √-oversampling | 0.9135 | 0.6270 | 0.6412 | 0.8990 | — |
| A1 | baseline + ArcFace × z-score | 0.9035 | 0.6437 | — | 0.9068 | — |
| best in-distribution | ConvNeXtV2-L @320, multi-head | **0.9316** | — | — | — | — |
| **B8 — best deployable (ours)** | 198 M, no √-oversampling, self-training at the 2 % dose | 0.9060 | 0.7798 | **0.7816** | 0.9153 | 71.17 % |
| **P5 — SHIP THIS** | BioCLIP-2 fine-tuned + unfrozen adaptation | 0.9113 | **0.7810** | 0.7806 | **0.9161** | **88.44 %** |
| shippable student | fastvit_sa12, distilled | 0.8967 | — | — | — | — |

**B8 and P5 tie on accuracy** (§4.14.4) **and are 17.3 points apart on useful-answer rate** (§4.6a),
which is why P5 is the recommendation. Open-set AUROC is each model's *best* scoring rule (§4.9);
useful-answer is the fraction of probe images given an answer that is correct, under a 95 %-precision
back-off policy.

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

$$
C^{(\text{top})} = M^{(\text{top})}, \qquad
C^{(i)} = M^{(i)} + \mathrm{gather}_{\pi}\Bigl(C^{(i+1)} - \mathrm{logsumexp}_{\text{siblings}} M^{(i)}\Bigr)
$$

i.e. $P_{\text{cond}}(\text{child}) = P(\text{child}) \cdot P_{\text{cond}}(\text{parent}) / P(\text{siblings})$ in log-space. Marginalisation (§2.4) is the exact mirror: information flows
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

### 4.4 Novelty is graded by taxonomic distance — for any head

Novelty is not binary in a taxonomy. We stratify unseen species by how much of their lineage the
model has seen: **near** (unseen species, known genus), **mid** (unseen genus, known family), **far**
(unseen family). Each head is scored with its own best rule (§4.3, §4.9).

| stratum | n | plain cosine (entropy) | ArcFace × z-score (max-logit) |
|---|---|---|---|
| near | 8,000 | 0.8527 | **0.8680** |
| mid | 8,000 | **0.9342** | 0.9165 |
| far | 399 | **0.9641** | 0.9444 |

**Both heads are monotone in taxonomic distance.** That is the result: the embedding places
unfamiliar taxa at a distance that *tracks the taxonomy* rather than merely displacing them, and it
does so whether or not an angular margin was used. It is therefore a property of the learned
representation and the taxonomy, not of the margin.

The operational reading is the ordering, not the average. The easy stratum (`far`, an unseen family)
is rare — 399 images — while the hard one (`near`, a new species in a familiar genus) is both the
most common field case and 11–12 points worse. A single pooled AUROC is flattered by the easy strata.

**The ordering is not an artefact of rarity.** The novel taxa above are obtained for free, as every
species below the 50-image training floor — so "unseen" is perfectly confounded with "rare, and
plausibly photographed differently", and the confound runs *with* the effect, since far taxa are
rarer than near ones. We therefore retrain the identical model on a corpus with **common** taxa
deliberately withheld at all three ranks (231 species, each with >= 200 training images: 2 whole
families, 40 whole genera, 120 single species; 2.62 % of rows removed) and repeat the measurement.

| stratum | novel population = rare (free) | novel population = **common (withheld)** |
|---|---|---|
| near | 0.8527 | **0.8717** |
| mid | 0.9342 | **0.9463** |
| far | 0.9641 | **0.9726** |

The ordering is unchanged and every stratum is marginally *better*, so rarity was if anything adding
noise to the novel population rather than manufacturing the gradient. The claim now rests on two
novel populations selected by opposite criteria — everything below a 50-image floor, and 231 species
above a 200-image one. Absolute values across the two columns are not strictly comparable (different
known sets and different novel populations); the ordering is what transfers.

*(An earlier version of this section reported the plain head at 0.561/0.618/0.667, a 29–31 point
deficit. Those numbers used max-logit, which §4.3 shows is that head's worst rule by 27 points. The
corrected comparison above has the two heads within 2 points and trading places by stratum.)*

### 4.5 Open-set under domain shift

Repeating the benchmark where the novel species *also* come from a different camera (flemming):

| head | novelty only | novelty + domain shift |
|---|---|---|
| cosine | 0.601 | 0.574 |
| ArcFace × z-score | **0.9115** | **0.7272** |

> **Caveat, and it is the same one §4.3 raises.** Both columns score both heads with `max-logit`,
> which §4.9 shows is the plain cosine head's *worst* rule by 27 points. The cosine row is therefore
> a lower bound on that head, and **the "+31.1 → +15.3" reading below is not supported** — it is the
> retracted best-vs-worst comparison carried into the shifted setting. What survives is the
> within-row comparison: the margin head loses 18.4 points of AUROC when novelty is compounded with
> domain shift. Re-scoring this benchmark with each head's best rule is outstanding.

The absolute *drop* is what the section is about. The logits show why: under shift the
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

### 4.6a Under domain shift, abstention is expensive — and two models with equal accuracy are not equally deployable

The numbers above are in-distribution. Repeating the same 95 %-precision policy on the shifted probe
fold, for the two models this paper recommends:

| | B8 (198 M, ours) | P5 (BioCLIP-2 fine-tuned) |
|---|---|---|
| species: coverage / precision | 69.72 % / 0.9737 | 71.48 % / 0.9530 |
| genus | 1.43 % / **0.8073** | **0.00 %** |
| family | 2.14 % / 0.9939 | 21.31 % / 0.9537 |
| **abstain** | **26.70 %** | **7.21 %** |
| answered | 73.30 % | **92.79 %** |
| **useful (answered and correct)** | **71.17 %** | **88.44 %** |

**Reaching 95 % precision costs 0.82 % abstention in-distribution and 26.70 % under source shift.**
The mechanism of §4.6 is the same and its magnitude is not: a back-off policy that looks nearly free
on a held-out fold withholds an answer on a quarter of images from a different camera. Any paper
reporting abstention coverage on an in-distribution split is reporting the easy case.

**The conditional-calibration failure of §4.6 also sharpens.** On the 4,603 images where B8's species
confidence falls below threshold, genus precision is **0.5570** against 0.8511 overall and **never
reaches 95 % at any threshold**. The genus rung is therefore unusable for that model at that target —
the ladder loses a step exactly where §4.6 predicts it should.

**And the result we did not expect: these two models are statistically tied on probe macro-F1
(0.7798 vs 0.7810) and 17.3 points apart on the fraction of images that receive a usable answer.**
The difference is not discriminative power — it is confidence calibration. B8 buys its higher
precision-among-answered (97.10 % vs 95.31 %) by declining to answer, which is what a model does when
its confidence does not cleanly separate its correct predictions from its errors. P5's calibration
also lets it skip the genus rung entirely and fall straight to family, where the evidence actually
supports the promised precision.

The practical consequence is that **a per-class accuracy metric cannot rank deployable systems.**
Selecting on probe macro-F1 would have called this pair a coin flip; one of them answers 93 % of
photographs and the other 73 %. We report useful-answer rate alongside accuracy for that reason, and
recommend it for any system that is permitted to abstain.

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

**Practical consequence.** The margin's open-set benefit is small once each head is read with its own
best rule — 0.8990 → 0.9068 (§4.3) — so on this architecture it does *not* pay for the 1.15 pt it
costs the coarse ranks. What it buys is insensitivity to the readout, which is worth something in
deployment and nothing on a benchmark. A system that both marginalises and uses a margin should
**supervise the marginals as well**, since the two interventions act on the same quantity in opposite
directions. We report this as an interaction to be managed rather than a solved problem; the combined
model is future work.

### 4.8 How much of the domain gap is nameable nuisance?

The gap between in-distribution and external accuracy is ~23 points for the best model of §4.2 and
25.98 points for the 20 M model used in this section's table. That number alone
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

The mechanism is a saturation argument. Since $\mathrm{msp} \approx z_{\max} - \mathrm{logsumexp}(z)$, max-logit asks *how strongly does the best prototype match*, while MSP
asks *how much better than the alternatives*. In a well-fitted embedding every input — known or novel
— attains a high cosine to some prototype, so the absolute maximum saturates and stops
discriminating; what still separates them is whether one prototype **dominates**. Only the
shape-sensitive rules (MSP, entropy, top-2 margin) see that, and they are exactly the three that gain.
Energy tracks max-logit to within 0.15 points everywhere, as it must: $\mathrm{logsumexp}$ is
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

Among *these* models the largest augmented one leads two axes and gives up 1.75 points on the third,
and the smallest leads open-set detection.

**That last clause does not survive scoring the models we actually recommend.** Scored on the same
open-set benchmark with each model's own best rule (§4.9):

| model | in-distribution | probe | open-set |
|---|---|---|---|
| efficientnet\_v2\_s (A1) | 0.9035 | 0.6437 | 0.9068 |
| **B8** — our best | 0.9060 | 0.7798 | **0.9153** (entropy) |
| **P5** — BioCLIP-2 fine-tuned | 0.9113 | **0.7810** | **0.9161** (entropy) |

**Both recommended models beat the small one at open-set detection**, so the capacity/novelty
tension reported above is a property of the four models in that factorial rather than a general one.
**P5 leads or ties on all three axes at once.** The practical argument for reporting three axes
survives — they measure different things and a reader needs all of them — but the stronger claim,
that they *rank models oppositely*, does not. We state this plainly because an earlier version of
this section made the stronger claim, and the correction runs in the direction of the boring answer.

The disagreement that *does* survive is between accuracy and **deployability**, and it is large:
§4.6a shows B8 and P5 tied on probe macro-F1 and 17.3 points apart on the fraction of images that
receive a usable answer.

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

**Noise floors are reported per training regime, and this is not a formality.** Retraining one
configuration unchanged and scoring the repeat gives:

| regime | probe | held-out species |
|---|---|---|
| end-to-end, 20 M, 5 epochs | 0.0041 | 0.0052 |
| frozen-trunk stage, 20 M, 2 epochs | **0.0119** | 0.0079 |
| frozen-trunk stage, 198 M, 2 epochs | **0.0130** | **0.0374** |

A floor is a property of the (metric × benchmark × **training procedure**) triple, not of the
benchmark alone. We state this because we got it wrong: floors measured on the first row were quoted
against results from the third, and two conclusions in an earlier version of this paper did not
survive the correction. A 2-epoch stage on a frozen representation has far more run-to-run freedom
than a 5-epoch end-to-end run, and the 198 M held-out floor — 0.0374 on a 2,455-image, 58-species
benchmark — is large enough to swallow most differences reported at that scale.

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
also removing the resampling of §4.13, gives the best model we obtain: **probe 0.7798, held-out
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

**Nor is training against centroids, or sampling the softmax.** We measured four further routes and
all fail: uniform sampled softmax degrades smoothly with no plateau (at 1M classes, 1024 negatives is
0.1 % coverage); taxonomy-aware hard negatives recover only 26 % of that loss; fixed taxonomy codes
are weakened by the same spectrum; and a proxy-free head that keeps EMA class centroids in a buffer
with no matrix and no optimiser state — removing 10.24 GB of the 15.36 GB — costs **4.63 points**. So
**centroids can *replace* a trained matrix at inference but cannot *substitute* for training one.**

**The resolution is a data policy, not an architecture.** The obstacle is stated above as ~1 M
species, but that count includes taxa with a handful of images. Applying the same minimum-image floor
we already use for unrelated reasons ($\geq 50$ images) to TreeOfLife-200M reduces **884,662 species
to 203,878** — a 4.3× cut that drops 680,784 taxa but only ~7 % of images — and with it a 15.36 GB
parameter-plus-optimiser footprint to **3.13 GB, which fits on one device.** The extreme-classification
problem at Tree-of-Life scale is therefore largely self-inflicted: the honest answer is to apply an
image floor and train an ordinary matrix, and the interesting residue is what the floor costs in
coverage rather than how to compress what it removes.

### 4.13 Long-tail rebalancing trades robustness for accuracy — and the trade is an artefact of where it is applied

Every method in the long-tailed-recognition literature is validated on an in-distribution held-out
split: CIFAR-LT, ImageNet-LT and iNaturalist ship no shifted test set. We have one, and the ranking
does not survive it.

**Two preliminaries, because they change what there is to test.** First, Balanced Softmax [Ren et al.
2020] trains with $-\log\bigl(n_y e^{z_y} / \sum_j n_j e^{z_j}\bigr)$; since $\log n_j = \log \pi_j + \log N$ and the constant cancels inside the softmax, **it is exactly logit adjustment [Menon et al.
2021] at $\tau = 1$**. The two papers do not note this. Second, $\tau$-normalisation — rescaling
classifier weights by $\|w_j\|^{-\tau}$, one of the decoupling line's strongest
interventions [Kang et al. 2020] — is **already in our architecture**: the cosine head constrains
every prototype to $\|w_j\| = 1$, which is $\tau$-normalisation at $\tau=1$ enforced during
training rather than applied afterwards. Methods that correct the classifier's *norm* bias therefore
have nothing left to do here, and only methods that change **which examples the loss weights** have
room to act. That is a useful reminder that a method's value is a property of the system it sits in,
not of the method.

So the live question is a 2×2 over the two families that remain — resampling and loss reweighting —
which are usually presented as alternatives and rarely crossed.

| | oversampling | balanced softmax | in-distribution | **shifted** | shifted **micro** |
|---|---|---|---|---|---|
| L0 | — | — | 0.8949 | **0.6445** | 0.6589 |
| baseline | √ ($p=0.5$) | — | **0.9135** | 0.6293 | 0.6156 |
| L1 | — | $\tau=1$ | 0.8970 | 0.5726 | 0.5214 |
| L2 | √ ($p=0.5$) | $\tau=1$ | 0.8689 | **0.5492** | 0.4694 |

**Rank the cells by how hard they push probability mass toward rare classes — nothing,
√-oversampling, full prior correction, both — and the shifted score falls monotonically at every
step, a 9.5 point spread against a 0.69 point floor. The in-distribution column does not order at
all** (0.8949, 0.9135, 0.8970, 0.8689). The ordering was committed as a prediction before the runs
and all four cells landed in the predicted order.

**The micro column shows the mechanism directly.** For the un-reweighted cells macro < micro (0.6445
< 0.6589), the normal pattern — rare species are harder. For both balanced-softmax cells the sign
**flips**: L2 scores macro 0.5492 against micro 0.4694. The model is not merely helping the tail, it
is *over-predicting* it, and under shift — where the tail's evidence was flimsiest to begin with —
that costs 19 points of ordinary accuracy.

Why this is the expected direction: macro-F1 weights every species equally, so all of these methods
target the classes with the fewest images. What a model learns from 43 photographs is
disproportionately about *those photographs* — one photographer, one background, one camera. **Tail
reweighting up-weights the least transferable part of the training signal.** In-distribution that is
invisible, because the held-out fold shares the same artefacts.

**The cost grows with capacity rather than being absorbed by it.** Removing √-oversampling from the
198 M model costs 1.64 points in-distribution and buys **+2.88 on probe** (7.0× floor), against
−1.52 at 20 M:

| intervention | effect at 20 M | effect at 198 M |
|---|---|---|
| domain augmentation's in-distribution cost | −0.36 | **0.00** (absorbed) |
| marginal supervision's shifted benefit | +1.79 | **+0.02** (absorbed) |
| **√-oversampling's robustness cost** | **−1.52** | **−2.88** (**grows**) |

The two that vanish are *constraints on the optimisation* — auxiliary losses, corrupted inputs — and
a model with enough capacity satisfies them without giving anything up. Oversampling is not a
constraint: it **changes which data the model sees**, and no amount of capacity makes a model learn
from images it is shown less often. At higher capacity the model fits the reweighted distribution
better, which means it fits its bias better too. **Interventions that constrain the objective weaken
with capacity; interventions that reshape the data distribution do not, and may strengthen.**
Predicting which of the two an intervention is, is worth more than measuring it at one scale.

**The trade is not intrinsic.** cRT [Kang et al. 2020] — train the representation on the natural
distribution, then freeze it and rebalance only the classifier — separates *what* is rebalanced from
*where*:

| | in-distribution | probe |
|---|---|---|
| L0 — no oversampling at all | 0.8949 | 0.6445 |
| baseline — oversampling throughout | **0.9135** | 0.6293 |
| **L4 — cRT: L0's representation, classifier rebalanced** | 0.9068 | **0.6539** |

cRT recovers **+1.19** of oversampling's +1.86 in-distribution gain while scoring **+2.46 above** the
fully-oversampled baseline under shift — and above L0 itself by +0.94. Rebalancing the classifier
costs nothing under shift; rebalancing the *data the backbone sees* is what cost 1.52 points. The
damage was in the representation all along, and freezing it closes that channel while the classifier,
which only has to re-scale decision boundaries, gets the benefit for free.

**The recommendation is therefore not "leave oversampling off" but "apply it to the classifier
only"** — and it composes with domain adaptation rather than competing with it. At 198 M, removing
oversampling and adding self-training are worth +0.97 probe and **+2.90 held-out species** over the
same recipe with oversampling on, so neither subsumes the other.

#### 4.13.1 The same shape appears in the *data*: truncating the head has an interior optimum

Rebalancing acts on classes the model already has. The dual question is how many images per class to
collect at all. Our corpus was built with a cap of ~2,000 images per species, chosen for balance on
an untested intuition. Sweeping it:

| cap | train images | in-distribution | probe | held-out species |
|---|---|---|---|---|
| 250 | 2.13 M | 0.8783 | 0.5776 | 0.6248 |
| 500 | 3.18 M | 0.8955 | 0.6281 | 0.6371 |
| **1,000** | **4.49 M** | 0.9060 | **0.6446** | **0.6706** |
| uncapped (~2,000) | 5.70 M | **0.9148** | 0.6270 | 0.6412 |

**In-distribution accuracy rises monotonically with the cap. Both deployment axes peak at 1,000 and
then fall.** Capping at 1,000 rather than 2,000 buys **+1.76 probe and +2.94 held-out species** for
−0.84 in-distribution ($n=2$ on both arms, roughly 3× the combined spread).

This is the same shape as the self-training dose curve of §4.11 — an interior optimum, with the
metric that looks like progress continuing to rise past the point where the useful metrics turn over
— and it is a third instance of the section's theme: **images of species that already have thousands
of them buy accuracy on the axis this problem has already saturated, and cost accuracy on the two
axes that describe deployment.** Our own cut was right in kind and too shallow.

It also answers the acquisition question from the other side. Measured on TreeOfLife-200M's own
per-species counts, the Lepidoptera surplus over our corpus is **3.1× uncapped but only 1.2× at our
2,000 cap** — about 90 % of it is head images beyond the cap. Uncapped, the ten most-photographed
species would take 7.8 % of every epoch while the 65,453 rarest combined take 11.6 %. Acquiring that
data would move us further right on a curve whose deployment axes are already declining.

This is the second of the paper's four independent questions to resolve toward the classifier, and
§4.15 assembles them.

### 4.14 Foundation models: contamination, a frozen probe that understates by 7 points, and what the recipe is actually contributing

A reviewer's first question about any specialist model is why not use a foundation model instead.
BioCLIP-2 [Stevens et al., VERIFY] — a ViT-L/14 CLIP model trained on TreeOfLife-200M, i.e. this
exact domain at 43× our data — makes the question sharp. Answering it properly required first
discovering that the obvious comparison is invalid.

#### 4.14.1 The benchmark is inside the foundation model's training set

Our corpus is GBIF-derived. So is TreeOfLife-200M. Joining the two on **GBIF occurrence identifier**
— photograph identity, not species name:

| | |
|---|---|
| our species also in ToL-200M | **11,916 of 12,772 — 93.3 %** |
| our images in ToL-200M, by exact occurrence id | 4,141,385 — **65.4 %** |
| **our held-out test-fold images in ToL-200M** | **413,865 of 629,742 — 65.4 %** |

**Two thirds of our test fold is BioCLIP-2's training data.** Every in-distribution comparison against
it is contaminated in its favour, so all in-distribution numbers below are re-scored on a
**decontaminated fold** — the 219,048 images / 11,998 species carrying no ToL occurrence id — with
*both* arms re-scored on it, since a clean-subset score cannot be compared against a full-fold one
either.

**The generalisable point is the method, not our number.** *"Pretrained on a public archive"* and
*"evaluated on a public archive"* are the same sentence far more often than anyone checks, and the
reason it goes unnoticed is that the usual check is **taxonomic** overlap — "does the model know
these species?" — which is the weaker question and reassuringly answerable at 93 % without anyone
realising the images are literally the same. The check costs nothing: the parquet footers locate the
relevant shards, and reading only the taxonomy and provenance columns skips the embeddings (~99 % of
the bytes), so the whole join runs on a laptop in twenty minutes against a 350 GB cache with nothing
downloaded. We recommend it as routine for any benchmark evaluated against a biological foundation
model.

The shifted benchmarks are **not** contaminated: they are camera-trap imagery with no GBIF occurrence
ids, a different camera and a different collection process. The shifted comparison is clean on both
sides, which makes it the decisive one.

#### 4.14.2 A frozen probe can understate a representation by 7 points

The standard way to compare representations is a linear (here, cosine) probe on frozen features. On
the decontaminated fold:

| BioCLIP-2 trunk | species macro-F1 | vs our baseline (0.9021) |
|---|---|---|
| frozen, classifier fitted | 0.8444 | **−5.77** |
| fine-tuned, lr 1e-3 *(our default)* | 0.8912 | −1.09 |
| fine-tuned, lr 1e-4 | 0.9025 | +0.04 |
| **fine-tuned, lr 1e-5** | **0.9146** | **+1.25** |

**Unfreezing is worth +7.02 points**, and the learning rate spans 2.34 of them — with *our own
default the worst of the three arms*. A single-arm run at the rate tuned for our CNN would have
returned 0.8912, sat 1.09 points below baseline, and supported the conclusion opposite to the truth,
from a plausible number with no crash to warn anyone. We report the three arms because the reason for
running three was written down beforehand: a rate that suits a CNN can destroy a pretrained ViT-L's
representation in the first few hundred steps, and that failure looks like a bad number rather than
an error.

Under shift the fine-tuned model leads by more, not less:

| | in-distribution | probe | held-out species |
|---|---|---|---|
| our task-trained baseline | 0.9021 | 0.6270 | 0.6412 |
| **BioCLIP-2, fine-tuned** | **0.9146** | **0.6630** | **0.6937** |
| Δ | +1.25 | +3.60 | **+5.25** |

**The best representation for this task is not ours**, and the margin is largest on the hardest and
most deployment-relevant axis. A frozen probe reported the opposite.

#### 4.14.3 Frozen-trunk adaptation is bounded to trunks trained with the same head

§4.15 will claim that the classifier stages are cheap because they run on a frozen trunk. That claim
has a boundary, and a foreign trunk finds it.

| adaptation stage | probe | held-out species |
|---|---|---|
| our trunk, frozen (T2b) | 0.7515 | — |
| BioCLIP-2, frozen (P1b: fitted then adapted, never unfrozen) | 0.5901 | 0.5599 |
| BioCLIP-2 fine-tuned, then **frozen** adaptation | 0.7218 | 0.7540 |
| BioCLIP-2 fine-tuned, then **unfrozen** adaptation | **0.7810** | **0.7806** |

Unfreezing the adaptation stage is worth **+5.93 points**. So the apparent crossover — BioCLIP-2
leading our baseline by 3.60 before adaptation and trailing by 3.16 after it — is a **frozen-readout
artefact**, not evidence that adaptation and pretraining are substitutes.

The mechanism is not mysterious. Our trunk was trained *with the very cosine head we then re-fit*, so
its features are already arranged for that readout. BioCLIP-2's are arranged for alignment with
**text**. Freezing them and attaching a cosine head asks the head to read a geometry it was never
shaped for. The correct statement is therefore:

> **Frozen-trunk adaptation is cheap when the frozen trunk was trained with the same head geometry
> you are re-fitting. On a foreign trunk it costs about six points, and the trunk must be unfrozen.**

This bounds §4.15 to trunks trained with the head in question. It retracts none of its numbers; it
removes an extrapolation.

#### 4.14.4 What this says the contribution is

Fine-tuned BioCLIP-2 with our unfrozen adaptation reaches probe 0.7810 / held-out 0.7806. Our own
best model reaches 0.7798 / 0.7816. **They are tied.** The honest summary is:

- before adaptation, their representation is **better** than ours (+3.60 probe, +5.25 held-out);
- after each is given its best treatment, the two **converge**;
- the advantage a 200 M-image encoder brings is real, and **the adaptation recipe closes it**.

That is more useful than either backbone winning. It says the ceiling here is set by **target-domain
data**, not by the encoder — the same conclusion §4.11 and §4.13 reach from other directions — and it
locates our contribution in the **recipe** (classifier stages, self-training, abstention, open-set
scoring) rather than in the representation. A recipe transfers to other groups and other taxa; a
backbone does not.

It also settles a costly question in the negative: **retraining a backbone on ToL-200M is not worth
doing.** A ~84 M-image download would buy a worse version of something downloadable. Two independent
measurements agree — the fine-tune already exceeds what we could train, and (§4.13) the extra data is
almost entirely head images, which our own cap policy says to discard.

### 4.15 Almost everything that matters is a classifier-stage concern

The results of §4.1, §4.13 (resampling), §4.11 and §4.12 were obtained separately and answer different
questions. Placed together they say the same thing, and it is the strongest through-line we have.

**Coarse supervision.** Per-level classifier *parameters* hurt (§4.1); the per-level *losses* help,
and only visibly off the training distribution.

**Long-tail resampling.** Applied to the data the backbone sees, square-root oversampling buys 1.9
points in-distribution and costs 2.9 under source shift. Applied only to a classifier retrained on a
frozen representation (cRT; Kang et al. [VERIFY]), it recovers **+1.19** of the in-distribution gain
while scoring **+2.46 above** the fully-resampled model externally. The accuracy/robustness trade of
§4.13 is therefore not intrinsic — it is an artefact of where the rebalancing is applied.

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

**Both arms must be tuned, and the answer depends on capacity.** The table above gives both arms the
same pseudo-label set, which inherits the *natural, long-tailed* class distribution of the target
camera. Rebalancing that set so every species contributes equally is not a neutral choice:

| | | in-distribution | probe | held-out species |
|---|---|---|---|---|
| 20 M | staged, natural | 0.9081 | 0.7541 | 0.7594 |
| 20 M | **staged, balanced** | 0.9074 | **0.7692** | **0.7781** |
| 20 M | **end-to-end, natural** | 0.9003 | **0.7706** | 0.7704 |
| 20 M | end-to-end, balanced | — | 0.7635 | 0.7342 |
| 198 M | staged, natural | **0.9150** | 0.7648 | 0.7600 |
| 198 M | **staged, balanced** | 0.9138 | **0.7740** | 0.7518 |
| 198 M | **end-to-end, natural** | 0.9060 | 0.7798 | **0.7816** |
| 198 M | end-to-end, balanced | 0.9058 | **0.7800** | 0.7741 |

**Read this table against the floors of §4.11, and most of it disappears.** The frozen-trunk probe
floor is 0.0119 at 20 M and 0.0130 at 198 M, and the 198 M held-out floor is **0.0374**. Applying
them:

- **Balancing harms a trainable trunk at 20 M.** −0.71 probe / **−3.62** held-out for end-to-end; the
  held-out effect is 7x its floor and is the one solid cell in the table. It concentrates replication
  on the classes with fewest unique images, which are also those with least reliable pseudo-labels,
  and a trainable representation memorises them.
- **Balancing's benefit to a frozen trunk does not survive.** +1.51 probe is 1.3x a 0.0119 floor.
  Directionally consistent with the mechanism above, but we do not claim it.
- **Nothing at 198 M is resolvable.** All four staged/end-to-end differences at that scale sit inside
  the floors, and an exact repeat of the staged 198 M configuration scored held-out **0.7892 against
  0.7518** — a 3.74 pt spread between identical runs.

We therefore make the weaker and better-supported claim: **staged and end-to-end training are
indistinguishable under shift at both scales we tested**, while the staged recipe's in-distribution
advantage (+0.71 to +0.88 across four runs, against a species-level spread of 0.0010) is real and
stable across the 10x change. The result is a **cost and redeployability** one. Two adaptation stages
are repeatable per deployment: a new camera does not require a new model, it requires a new
classifier, which is minutes on a frozen trunk and needs no labels.

An earlier version of this section claimed the external trade was capacity-dependent — parity at
20 M, a 2.98 pt deficit at 198 M. That was a single-draw measurement compared against a floor
borrowed from a different training regime, and it did not survive a repeat. It is retracted here
rather than quietly dropped, because it is the third time in this work that a difference smaller than
its true noise floor was reported as a finding, and the pattern is more useful than the number.

We flag the tuning point explicitly because it is the second time in this work that a headline gap
moved once both arms were tuned — the first being the open-set head comparison of §4.3, where a
31-point margin advantage became 0.78 once each head was read with its own best scoring rule.

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
classifier (§4.15).

**The methodological finding may outlast the empirical ones.** Five times, a conclusion this work
held was overturned not by a better model but by a better measurement:

1. An open-set scoring rule chosen on a 20 M model does not transfer to a 198 M one; reading every
   model with one rule understated the large ones by 6–7.6 points and inverted a ranking (§4.9).
2. That same oversight, applied to our own headline, produced a claimed 31-point advantage for an
   angular margin that is **0.78 points** when each head is read with its best rule (§4.3).
3. Macro-F1 does not decompose over subsets, so two benchmarks over the *same images* can rank two
   models oppositely with both results correct (§3.2).
4. A run-to-run noise floor is a property of the (metric × benchmark × **training procedure**)
   triple. Floors measured on 5-epoch end-to-end runs were quoted against 2-epoch frozen-trunk
   stages, where the true spread is 3–7× larger; an exact repeat of one configuration differed from
   itself by 3.74 points, and two conclusions did not survive it (§4.11).
5. A frozen linear probe — the standard way to compare representations — understated a foundation
   model by **7 points**, and the fine-tuning learning rate spanned 2.34 more. The default tuned for
   our own architecture was the worst of three arms and would have supported the opposite conclusion,
   from a plausible number with no error to warn us (§4.14).

Each was a comparison made with a default nobody had recorded as a decision. We report them because
the corrected numbers are the paper's numbers, and because the failure mode is not specific to us:
**a baseline that everyone quotes is the one least likely to be re-measured.** The pattern is
consistent enough to state as a working rule — *give every arm its own best configuration, or you are
comparing configurations rather than methods* — and twice a headline gap in this paper collapsed
under it, from 31 points to 0.78 and from 1.65 to 0.14.

**What we would tell a practitioner.** Start from the best available pretrained encoder and
**fine-tune it** — do not judge it by a frozen probe, and check whether your benchmark is inside its
training set before you believe any in-distribution comparison. Then do the cheap work: rebalance the
classifier if the in-distribution metric matters, adapt the classifier on unlabelled target images if
deployment does, and read novelty with a rule chosen on *your* model rather than inherited. Cap the
head of your training distribution more aggressively than feels comfortable. Report accuracy,
external accuracy, and open-set detection separately, because they disagree — and never compare two
of them across different evaluation sets.

**What we do not claim.** That the classifier/representation split generalises beyond fine-grained
recognition under source shift; that self-training's advantage over labels survives at much larger
label budgets (it is still rising at 12,230); that the angular margin is not worth its cost — we show
only that its cost and benefit are both about one point, and that its real advantage is insensitivity
to the readout rather than a better score; or that our recipe would beat a foundation model given
comparable pretraining data. On the contrary: §4.14 finds the better representation is not ours, and
the claim we make is that the **recipe closes the gap**, which is a claim about the recipe.

## 6. Limitations

Single taxonomic domain and one 3-level hierarchy. $m = 0.3$, $s = 30$ were first guesses, so the
margin's measured effect is not a tuned optimum. Distillation experiments use one student family.

**Open-set under shift is measured for no model we recommend.** §4.6a and §4.10 now report
abstention and open-set AUROC for both recommended models, but on the *no-domain-shift* novelty
benchmark; §4.5 shows novelty detection is not domain-robust, and the compounded case is unmeasured
for B8 and P5.

**Statistical power is uneven, and we say where.** Differences are quoted against the regime-matched
floors of §4.11, but several results rest on n = 2 and a few on n = 1. Where a difference is under
3x its floor we describe it rather than claim it. The novelty-plus-shift benchmark of §4.5 has only
234 novel images among 47,905 and is directional.

**In-distribution comparisons against foundation models are contaminated** (§4.14), and we do not
have an uncontaminated version of that axis — only a decontaminated fold that removes the overlap we
could detect by exact identifier.

## References _(to complete)_

All entries below are drafted from memory and carry the §1b `[VERIFY]` caveat: author lists, years
and venues must be checked against the sources before submission.

Deng et al., *ArcFace*, CVPR 2019 · Wang et al., *NormFace*, ACM MM 2017 · Liu et al., *ConvNeXt V2*,
CVPR 2023 · Oquab et al., *DINOv2* / *DINOv3* · Jordan & Jacobs, hierarchical mixtures · Hinton et
al., *Distilling the knowledge in a neural network*, 2015 · Kang et al., *Decoupling representation
and classifier for long-tailed recognition*, ICLR 2020 · Menon et al., *Long-tail learning via logit
adjustment*, ICLR 2021 · Ren et al., *Balanced Meta-Softmax*, NeurIPS 2020 · Cao et al., *LDAM-DRW*,
NeurIPS 2019 · Cui et al., *Class-balanced loss based on effective number of samples*, CVPR 2019 ·
Mahajan et al., *Exploring the limits of weakly supervised pretraining*, ECCV 2018 · Hendrycks &
Gimpel, *A baseline for detecting misclassified and out-of-distribution examples*, ICLR 2017 · Liu et
al., *Energy-based out-of-distribution detection*, NeurIPS 2020 · Lee, *Pseudo-label*, ICML workshop
2013 · Sohn et al., *FixMatch*, NeurIPS 2020 · Xie et al., *Self-training with noisy student*, CVPR
2020 · Guo et al., *On calibration of modern neural networks*, ICML 2017 · Stevens et al., *BioCLIP*,
CVPR 2024 · Stevens et al., *BioCLIP 2* / *TreeOfLife-200M*, 2025 · Dodge et al., *Documenting large
webtext corpora*, EMNLP 2021.
