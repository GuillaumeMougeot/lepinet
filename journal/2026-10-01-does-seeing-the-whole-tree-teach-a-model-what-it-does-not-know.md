# D3: does seeing the whole tree of life teach a model what it does not know?

**Kind:** research · **Status:** **OPEN (2026-10-01).** The question the owner wants the TreeOfLife
crawl to answer: does a model trained on far more data, **including taxa outside Lepidoptera**, get
better at the two things this project is about -- **open-set detection** (flagging species it was
never trained on) and **robustness to domain shift** (camera-trap photographs) -- than the same
model trained on Lepidoptera alone? Designed here, with predictions, before any training run.

## Why ask it

BioCLIP-2 already hints at a yes, and could not answer it. Trained on all of TreeOfLife-200M and
fine-tuned with our recipe (P5), it **ties our best Lepidoptera-only model on accuracy** and is
**17.3 points better on useful-answer rate** under abstention, because its confidence is better
calibrated ([[2026-08-28-two-tied-models-differ-by-17-points-in-deployment]]); fine-tuned, it also
beat our baseline under shift by +3.60 probe and +5.25 held-out species before any adaptation
([[2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts]]). But BioCLIP-2 differs from our
models in four ways at once -- corpus breadth, corpus size, objective (CLIP contrastive vs cosine
z-score) and architecture (ViT-L/14 vs ConvNeXt/EfficientNet). Any of them could be the cause. The
crawl makes it possible to change **one** at a time.

## The design: separate breadth from volume

"More data from outside Lepidoptera" bundles two things, and they make different predictions:

* **volume** -- more images, more of them per species;
* **breadth** -- more classes, from across the tree, most of them unlike any moth.

So a factorial on the same architecture, objective and recipe, with only the training corpus
changing:

| arm | corpus | images | classes | isolates |
|---|---|---|---|---|
| **A** | our `global_lepi` (current baseline) | ~6 M | ~12 k Lepidoptera | -- |
| **B** | ToL Lepidoptera + `global_lepi` fill | ~9 M | ~16 k Lepidoptera | volume, same breadth |
| **C** | ToL, all taxa, binomial species only | ~70 M | ~185 k | volume + breadth |
| **D** | C, subsampled to B's image count | ~9 M | ~185 k | **breadth at fixed volume** |

B vs A is volume. **D vs B is breadth with volume held fixed** -- the comparison that answers the
owner's question. C vs D is volume again at full breadth. Every arm then gets the same downstream
treatment: Lepidoptera species head, the staged recipe, the same pseudo-labelled adaptation stage.

BioCLIP-2 (P5) is the external reference: it shares C's corpus and differs in objective and
architecture, so **C vs P5 isolates the objective** -- which is D2's question, answered for free.

## What is measured, and on what

All on the benchmarks the paper already uses, so every number lands in an existing column:

* **open-set**: AUROC on the novel-species benchmark, each model with its own best scoring rule
  ([[2026-08-01-the-scoring-rule-was-the-bug]]), **stratified near / mid / far** by taxonomic
  distance ([[2026-08-08-is-novelty-monotone-or-just-rare]]);
* **domain shift**: probe and probe-held-out-species macro-F1, before and after adaptation;
* **calibration**: useful-answer rate at 95 % precision on probe, the measure on which P5 and B8
  diverged;
* **accuracy**: species macro-F1 on the decontaminated held-out fold.

## Two contamination traps, both of which would make the answer look better than it is

1. **The test fold is inside ToL.** 413,865 of our 629,742 held-out images are in TreeOfLife-200M by
   GBIF occurrence id ([[2026-08-26-bioclip2-has-seen-two-thirds-of-our-test-fold]]). Arms B-D must
   **exclude every test-fold occurrence from training** -- `data/tol_overlap/tol_gbif_ids.parquet`
   and `test_ids_clean_of_tol.parquet` already hold the join -- and in-distribution numbers come from
   the decontaminated fold only.
2. **The "novel" species are not novel to ToL.** The open-set benchmark's unseen species are
   everything under our 50-image floor; ToL may hold hundreds of images of them. Training on ToL
   would turn the open-set test into a closed-set one. **Every benchmark novel species, and the 231
   withheld common taxa of C3b, must be removed from B-D's training data at every rank they are
   novel at** (a "far" novel family must have no images of that family at all).

Neither trap applies to the probe benchmarks -- camera-trap images are not in ToL -- which makes
the shift comparison the cleanest of the four.

## Label hygiene

Train on the **184,853 binomial species** only; genus-only records (12,915 keys) can supervise the
genus level of a hierarchical head; the 6,110 bare-epithet keys are excluded -- they merge unrelated
genera ([[2026-10-01-what-the-blocked-servers-cost]]). Labels come from the metadata, never the
species folder names.

## Predictions (committed)

* **Domain shift: breadth helps before adaptation, much less after.** D beats B on probe by
  **+1 to +4 points** before adaptation; after the adaptation stage the gap falls under 1.5 points.
  This is the P3 -> P5 pattern (BioCLIP-2 ahead before adaptation, tied after), and our own result
  that the ceiling is set by target-domain data. Falsified if D trails B on probe before adaptation.
* **Open-set: breadth helps far novelty, not near.** D beats B by **+2 to +5 AUROC on the far
  stratum** (unseen families: a model that has seen 185 k species has a better sense of "not
  anything I know") and by **less than 1 point on near** (an unseen species in a known genus looks
  like its congeners whatever else the model has seen). Falsified if near improves by more than
  far.
* **Calibration is where breadth pays most.** D's useful-answer rate exceeds B's by **more than 5
  points**. This is the prediction most directly suggested by P5, and the one I would most like
  tested, because it would say BioCLIP-2's deployability advantage comes from its corpus rather
  than its objective.
* **Accuracy: no meaningful gain** (within 1 point on the decontaminated fold). In-distribution is
  saturated; nothing in this project has moved it by data breadth.
* I decline to predict **C vs D** (volume at full breadth). Every volume experiment here so far
  found an interior optimum -- the self-training dose, the head cap -- and there is no basis for
  guessing where it sits at 185 k classes.

## Cost, and order of work

Validate at 20 M parameters before any 198 M run (the owner's scale discipline). At ~1,100 img/s,
arm D (~9 M images) is ~2.3 h/epoch and arm C (~70 M) ~18 h/epoch on one GPU; GPU-hours are the
less scarce resource (9,685 of 11,000 left), CPU-hours are not, so data loading must not move to
CPU nodes.

1. **Finish the crawl and the `global_lepi` fill** -- arms B-D need the images.
2. **Build the training parquets** with both contamination exclusions and the label hygiene, and
   write the exclusion counts down before training (how many images each trap removes).
3. **Arm D vs arm B at 20 M** first: it is the cheapest arm pair and answers the question.
4. Arm C, then the 198 M promotion of whichever arm wins, then the comparison against P5.

## Open questions this does not settle

Whether breadth must come from *related* taxa (other insects) or from anywhere; a follow-up arm
restricted to Insecta would separate those if D beats B.
