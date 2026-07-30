# The story & the research directions — what is the actual bottleneck?

**Status:** strategic (2026-07-29), owner-driven. We set out to compare **heads** for hierarchical
image classification. The results say that is *not* the interesting story. This entry records the
reframe and the directions that follow, so the paper is written around the real finding.

## What the experiments actually settled

1. **Head structure does not help.** On identical conditions (same recipe, same sqrt-oversampling
   0.5, effnetv2_s, only the head swapped — verified by config diff), on the clean stack:
   - independent (multi-head cosine): **0.910** val species F1
   - hierarchical (parent-conditioned): **0.885** — worse
   - autoregressive: historically 0.69–0.73 — much worse (deferred, unchanged conclusion)
   The independent head wins; conditioning/sequencing don't earn their complexity. **A "which head
   wins" paper has a null result.**
2. **The coarse heads are redundant — marginalisation beats them.** From `dev/042`
   ([[2026-07-lepi-app-compression]] §3): deriving genus/family by log-sum-exp over the *species*
   posterior beats separately-trained genus/family heads (+0.74 pp genus macro-F1, **+3.1 pp
   family**), and is *consistency-guaranteed* (no genus-argmax that contradicts the species-argmax's
   parent — 1.8 % of images otherwise). So the honest architecture is **one species head +
   marginalise up**, not a multi-head. The multi-head's extra heads are not justified.
3. **In-distribution accuracy is basically solved.** 0.9316 (ConvNeXtV2-L), DINOv3-ConvNeXt-L matches
   it (val 0.930) and ~2× faster. Scale gives diminishing returns *for the shipped model* because
   the small student caps at ~0.88 regardless of teacher (distilling from 0.911 vs 0.9316 gave the
   b0 the same ~0.88).
4. **The real gap is generalisation.** Best model in-distribution **0.93**, on external data
   **0.70** (flemming). A ~23 pp cliff. *This* is the unsolved problem.

## The reframe — the paper's actual contribution

**Hierarchical classification is valuable not for accuracy, but as the framework for *reliable
prediction that knows what it doesn't know* — along two orthogonal axes of "unknown":**

- **Open-set in image space** (novel species): the input is a taxon the model was never trained on.
  Detect it (don't force a confident wrong species). Tools: ArcFace/cosine **max-similarity** or
  energy/max-logit as a novelty score; the margin is meant to widen the known↔novel gap. Eval:
  flemming OOD species, and — cleanly, no domain shift — the **global_lepi species with <50 images**
  that were *removed from training but are still on UCloud* (owner's idea; a pure open-set set from
  the *same* distribution).
- **Open-set / abstention in hierarchy space** (unknown *rank*): the image simply cannot determine
  the species (occlusion, a genus that isn't separable from a photo, a hard pair). The right answer
  is not a guessed species but **"back off to the rank you can support"** — report genus, or family,
  with calibrated confidence. Marginalisation gives this *for free*: `P(genus)=Σ P(species∈genus)`
  is more confident and consistent than the species head, so a per-rank threshold yields graceful
  degradation ("I don't know the species, but it's *Noctuidae*"). This is the **useful, honest**
  output for a field tool, and it is exactly what the taxonomy is *for*.

So the hierarchy earns its place in **calibrated abstention and novelty handling**, not in the
classification head. That is a stronger, more honest, and more useful story than a head bake-off —
and it turns the null result (heads don't matter) into the *setup* for the real contribution.

The framing also unifies the two "OOD"s the owner named: an image can be unknown because the
**species is novel** (image-space) or because the **evidence is insufficient** (hierarchy-space);
a good system distinguishes and handles both.

## Research directions (prioritised)

1. **Two OOD benchmarks, then two detectors.**
   - Benchmarks: (a) global_lepi <50-image species (in-distribution image domain, unseen species —
     isolates *novelty*); (b) flemming OOD species (novelty **+** domain shift). Comparing (a) vs (b)
     separates "novel taxon" from "different camera".
   - Detectors: max-cosine (`dev/052`) for cosine/ArcFace vs energy/max-logit for the independent
     head; report AUROC on both benchmarks. Does ArcFace's margin actually widen the gap? (pending run).
2. **Hierarchy abstention as a first-class output.** Marginalise; fit per-rank thresholds
   (`dev/044`-style) targeting a precision; report the **accuracy/coverage-per-rank** curve — "at
   95 % species precision we cover X %, the rest resolve to genus at Y % precision". This is the
   product-facing metric and the paper's headline figure.
3. **Domain robustness without over-fitting the 500 (owner's constraints).**
   - **Prefer image-only / label-free**: (a) **domain-mimicking augmentation** — make GBIF training
     images look like trap/timelapse frames (motion blur, background clutter, low light, JPEG
     artefacts); cheapest, uses no OOD data. (b) **Unsupervised domain adaptation / self-training**
     — pseudo-label OOD images (images, not labels) with confidence gating; keep the 12 k-species
     head so the model doesn't specialise on 500.
   - **Splitting matters** (owner): flemming is timelapse ⇒ heavy near-duplicates ⇒ a random split
     leaks. Any use of these images needs a **grouped split by capture event / location / time**,
     never random. Guard against the model specialising on the 500 species by validating on *held-out
     species*, not held-out images.
   - Use OOD *labels* as little as possible (owner) — so measure with them, adapt without them where
     we can.
4. **Bigger/longer teacher (owner still believes; cheap now).** 6 epochs was budget, not principle;
   DINOv3-cnx trains ~3 h/epoch, so **double to ~12 epochs** and see if the teacher climbs. Improving
   the *teacher's* robustness matters more than the student (the student is capacity-capped).
5. **A slightly bigger student.** b0 caps ~0.88; the compression sweep already showed
   **fastvit_sa12 (10.6 M) → 0.892** and effnetv2b2 (8.7 M) → 0.887 beat b0 (5.9 M → 0.876). For a
   higher-quality small model, distil into fastvit_sa12 / effnetv2_b2, not b0. Newer options to try:
   mobilenetv4, efficientvit, the DINOv3-distilled convnext_tiny/small.

## Notes on the owner's other observations

- **Fair comparison?** Yes — verified identical configs (incl. oversampling); hierarchical is a wash.
- **TTA** (4-flip): ~+0.3 pp. Not worth its 4× eval cost as a default; keep it optional.
- **Evals "crazy slow" (>5 h).** The big-model evals run on a **`b200-1-mig.1g`** slice — the
  *smallest* MIG — so a 198 M model streams 630 k images on a fraction of one GPU; TTA makes it 4×.
  Fix going forward: **eval big models on a bigger MIG (2g/3g) or a full GPU**, and drop TTA for the
  big-model sweeps (its +0.3 pp isn't worth 4×).

## First OOD result: ArcFace clearly beats the plain cosine head (2026-07-29)

Benchmark (a) from the plan — **global_lepi's <50-image species**: removed from training but still on
UCloud, so they are genuinely *unseen species* in the **same image domain** (no domain-shift
confound). Score = `-max species logit`; metric = AUROC(known vs novel).

| head | in-distribution species F1 | **OOD AUROC** | known max-logit (μ±σ) | novel max-logit (μ±σ) |
|---|---|---|---|---|
| independent (plain cosine) | **0.9110** | 0.601 | −9.27 ± 7.44 | −11.46 ± 6.82 |
| **arcface** (species margin 0.3) | 0.8784 | **0.732** | 26.00 ± 13.03 | 23.47 ± 10.84 |

632,913 images, 3,171 of them novel-species.

**Reading it.** The plain cosine head is *barely better than chance* at knowing what it doesn't know
(0.601): its max-logit distributions for known and novel species overlap almost completely. ArcFace
moves that to **0.732 — a +13.1 pt improvement in open-set detection** — for a −3.3 pt cost in
in-distribution species F1. **That is the trade the new story is about**, and it is the first
evidence that the margin does what the face-recognition literature claims: it shapes an embedding
where "distance to the nearest known class" is *meaningful for classes never trained on*.

Caveats worth keeping: (i) 0.732 is useful but far from solved — a deployable "unknown species"
gate probably wants ≥0.85, so this is a direction, not a finished result; (ii) the two heads have
different logit scales (arcface is `s·cosθ` ≈ +26, the cosine head's z-score ≈ −9), so only the
*ranking* (AUROC) is comparable, not the raw thresholds — calibration per head is required before
any UI threshold; (iii) this isolates **novelty**; benchmark (b) (flemming OOD species = novelty **+**
domain shift) is the harder, more realistic case and still to run.

**What it changes.** ArcFace graduates from "a head that costs accuracy" to **the enabling component
of the open-set half of the story**. Next levers: energy / max-cosine variants of the score, the
margin as a hyperparameter (0.3 was a first guess), per-rank abstention (fall back to genus/family
when the species score is low), and the same measurement under domain shift.

## Two results that matter (val, 2026-07-30)

Both trainings finished; the fold-0 test evals are queued behind them. Val species macro-F1:

| run | val f1_species | vs its reference |
|---|---|---|
| multi-head independent (reference) | 0.9096 | — |
| **single species head (flat)** | **0.9129** | **+0.33 pp** |
| plain arcface (`s·cos`, m=0.3) | 0.8781 | −3.15 pp |
| **arcface × z-score** (`z(cos(θ+m))`, m=0.3) | **0.9065** | **+2.84 pp over plain arcface** |

**1. The multi-head is redundant — confirmed on the training side too.** A model trained with *only*
a species head matches (slightly beats) the three-head model on species. Combined with `dev/042`
(marginalising genus/family from the species posterior *beats* the trained coarse heads: +0.7 pp
genus, +3.1 pp family, and consistency-guaranteed), the conclusion is now end-to-end: **the extra
heads help neither the backbone during training nor the predictions at inference.** The honest
architecture is **one species head + marginalise up**. That is a clean, publishable negative result
and it simplifies the shipped model (fewer parameters, no possibility of contradictory levels).

**2. The z-score composition works.** Plain ArcFace cost 3.15 pp of in-distribution accuracy;
composing the margin with `cosine_to_zscore` recovers almost all of it (0.9065 vs the 0.9096
reference, −0.31 pp) — i.e. the two mechanisms were **fighting**, not trading off. The margin's
open-set geometry appears to be nearly free once the logits keep their calibrated, dimension-aware
scale. The decisive number is the **OOD AUROC** on the same benchmark: if it holds near 0.73 while
in-distribution F1 returns to ~0.91, the accuracy-vs-open-set trade-off largely dissolves and this
becomes the default head. That eval is next.

**A second N=1 metric bug, same root cause as the first.** `StreamingF1MultiHead` did
`zip(learn.pred, learn.y)`; with a single level fastai passes bare tensors, so `zip` iterated the
batch *rows* instead of the levels and reported `F1(macro) 0.0075` — which looked like a
catastrophically broken run until the per-level column (`f1_speciesKey 0.9129`, computed through the
already-fixed `level_pred_targ`) showed otherwise. Fixed and regression-tested. Lesson: when a
framework overloads a container for N=1, *every* consumer needs normalising — fixing one call site
is not enough.


## ArcFace × z-score: the trade-off dissolves (2026-07-30)

The decisive number. Same open-set benchmark (global_lepi <50-image species: unseen species, same
image domain; 632,913 images, 3,171 novel), same score (`−max species logit`), test macro-F1 on the
full fold-0:

| head | species F1 (in-dist) | **OOD AUROC** | known logit μ±σ | novel logit μ±σ |
|---|---|---|---|---|
| independent (plain cosine) | **0.9110** | 0.601 | −9.27 ± 7.44 | −11.46 ± 6.82 |
| arcface (`s·cos`, m=0.3) | 0.8784 | 0.732 | 26.00 ± 13.03 | 23.47 ± 10.84 |
| **arcface × z-score** (`z(cos(θ+m))`) | **0.9069** | **0.9115** | 32.58 ± 7.83 | 18.17 ± 6.38 |

**This is the headline result of the open-set direction.** Composing the margin with the z-score
transform is not a compromise between the two earlier heads — it beats *both* on the axis each was
supposed to own:

- **+31 pt AUROC over the plain cosine head** (0.601 → 0.9115), for **−0.4 pt** in-distribution F1.
  The plain head is near-chance at knowing what it doesn't know; this one is genuinely usable.
- **+18 pt AUROC over plain ArcFace**, *and* +2.9 pt in-distribution. So the earlier
  accuracy-vs-open-set trade-off was **not intrinsic to the margin** — it was an artefact of dropping
  the calibrated transform.

**Why (mechanism).** The distributions tell it: the known/novel logit gap goes 2.2 (plain cosine) →
2.5 (arcface) → **14.4** (arcface × z-score), while the *spread* simultaneously narrows (σ 7.8 vs
13.0). `cosine_to_zscore` stretches the tightly-concentrated cosines of high-dimensional unit
vectors into an approximately standard-normal scale; the margin then operates where that scale has
resolution, instead of being squashed into the narrow band where raw cosines live. Margin and
transform are complementary — one shapes the angles, the other makes the angle *differences* legible.

**Consequence for the architecture.** With single-head + marginals winning on accuracy at all three
levels and arcface × z-score winning on open-set at ~no accuracy cost, the recommended model is now
**one ArcFace × z-score species head + marginalisation up the taxonomy**. Both halves of the paper's
story — reliable *novelty* detection and calibrated *rank* abstention — are served by the same
single-head architecture. That is a much cleaner claim than a head bake-off.

**Caveats.** 0.9115 is on the *no-domain-shift* benchmark; the flemming OOD set (novelty **+**
different camera) is the harder case and is still to run. `m=0.3`/`s=30` remain untuned, so this is
a floor, not a ceiling.


## The embedding geometry, measured (dev/053, 2026-07-30)

*Why* arcface × z-score detects novelty so much better, measured directly on the embeddings —
12 species × 40 held-out images, cosine against each model's **own trained prototypes**:

| head | intra ↑ | inter ↓ | **margin** ↑ | silhouette |
|---|---|---|---|---|
| independent (plain cosine) | −0.154 | −0.336 | 0.182 | 0.617 |
| **arcface × z-score** | **0.667** | 0.056 | **0.610** | **0.641** |

- **intra** = mean cosine to its own prototype, **inter** = mean max cosine to a *wrong* prototype,
  **margin** = intra − inter (the separation ArcFace is designed to create).
- The margin is **3.3× larger** (0.610 vs 0.182), and the absolute geometry is transformed: the plain
  head's embeddings sit at cosine **−0.15** to their own prototype — barely aligned, essentially
  relying on *relative* ordering — while ArcFace pulls them to **+0.67**, genuinely clustered around
  the prototype, with wrong classes pushed to ≈0 (orthogonal).
- **Silhouette barely moves (0.617 → 0.641).** That is the honest caveat and the reason this script
  reports numbers before pictures: *cluster separability* was already fine — closed-set accuracy is
  0.911 either way, so of course it was. What changes is the **absolute angular position** relative
  to the prototypes, which is exactly what an open-set score reads. A t-SNE panel would have shown
  "two similar-looking blobs" and hidden the effect entirely.

**This explains the OOD result mechanistically.** A novelty score of `max cos θ` only works if
"close to a known class" has an absolute meaning. In the plain head everything is roughly orthogonal
to everything (intra −0.15, inter −0.34): a novel species looks no different from a known one, hence
AUROC 0.601. ArcFace × z-score gives known images high absolute similarity (0.667) while leaving
novel ones near the inter level, so a threshold separates them — AUROC 0.9115.

Plot: `data/emb_compare/embeddings.png` (t-SNE; umap not installed, and it would not change the
conclusion since the effect is angular, not topological).
