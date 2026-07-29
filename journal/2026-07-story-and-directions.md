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
