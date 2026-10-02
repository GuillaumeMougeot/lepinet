# W3: does seeing the whole tree of life teach a model what it does not know?

**Kind:** research · **Status:** **OPEN (2026-10-01).** The question the owner wants the
TreeOfLife crawl to answer: does a model trained on far more data, **including taxa outside
Lepidoptera**, get better at **open-set detection** and **robustness to domain shift** than the same
model trained on Lepidoptera alone? A second objective joined it: **one model that names the order
of any insect on the trap and the species of a moth.** **Decision (owner, third revision below): the
first run is a plain training on the crawled ToL subset with every taxonomic level it has, kingdom
to species, no special treatment of Lepidoptera.** Arm E and the A-D factorial are kept as later
options; the earlier revisions are kept as the record of how the design moved.

> *IDs renamed 2026-10-02 to end a clash with group D: D1 -> **K1** (checklist), D2 -> **W1** (the ToL
> download) and **W2** (our objective trained on ToL), D3 -> **W3** (training on the whole tree).*


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
architecture, so **C vs P5 isolates the objective** -- which is W2's question, answered for free.

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

## Revision (2026-10-01): arm E first, an order level, and a storage confound

### The owner's arm, and why it is better as a first experiment

> have ToL training modified so the lepi set includes only our training set during training (...
> just on the selection). So we could compare to A directly in one experiment?

**Arm E: A's Lepidoptera training split, exactly (folds 1-9: 5.70 M images, 12,632 species keys),
plus ToL's non-Lepidoptera.** It is better than D-vs-B as a first experiment, for three reasons:

* **One factor.** The Lepidoptera supervision is identical to A, row for row. E vs A changes only
  "what else the model sees". D vs B changes the Lepidoptera corpus *and* the class count *and*
  requires a subsampling scheme -- three levers where E has one.
* **Both contamination traps vanish by construction.** E's Lepidoptera are A's, so the held-out fold
  and the open-set benchmark's novel Lepidoptera are exactly as unseen as they are for A. The audit
  below shows how much exclusion machinery B-D would otherwise need.
* **It answers the question as the owner asked it** -- "more data, even from outside Lepidoptera" --
  rather than my reformulation into breadth-at-fixed-volume, which is a finer question best asked
  only if E shows an effect.

What E gives up: it confounds breadth with the volume of *non*-Lepidoptera data. That is acceptable
for a first answer: if E does nothing, neither volume nor breadth of other taxa helps, and D is
unnecessary; if E helps, D (or a subsampled E) attributes it.

**Insecta first (E1), all taxa later (E2).** ToL's selection holds 20.9 M Insecta images, of which
8.78 M are Lepidoptera: **~12.1 M non-Lepidoptera insect images over ~33.5 k species** before the
unreachable-server loss. E1 = ~17 M images per epoch, a third of E2's cost, and the only variant that
serves the order classifier. E2 (adding plants, vertebrates, fungi...) asks whether *unrelated*
breadth helps too -- BioCLIP-2's regime -- and waits on E1.

### The storage confound (found while sizing E; it applies to B-D as well)

The crawler stores ToL images at **<= 256 px on the long side** (`verify_and_encode`); A trains from
our ~512 px store, through the recipe's 460 px item resize and 256 crop. Mixing the two, every
non-Lepidoptera image is upsampled ~2x and every moth is sharp: **image sharpness would predict the
order label**, a shortcut aimed precisely at the order-level and open-set signals this experiment
measures. The original factorial had the same flaw -- B vs A was supposed to isolate volume but also
changed storage resolution -- and I did not see it when I wrote it. It is fixed by **A256**: A's
split re-encoded by the crawler's own function (`file://` rows, the path the `global_lepi` fill
already uses), trained with the unchanged recipe. A256 vs A then measures what 256 px storage costs
on its own (worth knowing before any ToL arm), and **E vs A256 is the clean comparison.** The images
are not edited, only stored the way every other image in the run is.

Cost: ~6.3 M re-encodes on 1 vCPU, of order 10 core-hours; within the budget guard, no approval needed.

### An order level: the trap's order classifier from the same model

The trap pipeline today is detector -> order classifier -> species classifier
(`dev/007_ami_pipeline.ipynb`). lepinet's head is hierarchical and generic over levels, and
marginalisation gives every coarser rank from the species probabilities. So E gets taxonomy levels
species -> genus -> family -> **order**, one species head over Lepidoptera **and** non-Lepidoptera
insect species, and the order prediction is the order marginal -- no second head and no second
network. Rank abstention ([[2026-08-28-two-tied-models-differ-by-17-points-in-deployment]]) extends
for free: a beetle confidently answered "Coleoptera" is a useful answer, a novel moth answered
"Lepidoptera, Noctuidae" is one too.

This changes what open-set means at the trap. For A, a beetle is out-of-distribution and the best
A can do is reject it. For E, it is in-distribution at order rank. Both are scored on the same crops:

* **moth vs non-moth** -- A: AUROC of its best open-set score; E: AUROC of P(order = Lepidoptera);
* **order accuracy / macro-F1** -- E only (A has one order);
* **Lepidoptera species macro-F1** on the held-out fold -- E with the full head, and E with the head
  masked to Lepidoptera species (`dev/081`'s `RestrictedHead`), the latter directly comparable to A.

Needed and not yet in the repo: **order-labelled trap crops** for evaluation (asked of the owner).

### Two recipe points fixed in advance

* **Equal Lepidoptera exposure.** An epoch is one pass of the recipe's oversampled Lepidoptera
  sampling plus the non-Lepidoptera rows (same per-species floor and cap); E runs the same number of
  epochs as A256. E costs ~3x the steps; the Lepidoptera signal is identical. Matching steps instead
  would give E a third of A's Lepidoptera gradient and confound breadth with undertraining.
* **Head width held at 256.** 256 was the knee at 12 k classes; at ~46 k it may not be. Changing it
  in the same run would be a second factor; widen only if E loses on Lepidoptera accuracy.

### Predictions for E1 vs A256 (committed)

* **Lepidoptera species macro-F1, masked head: within 1 point.** Unchanged reasoning from above.
  Open head: up to 1 point lower, from moths confused with look-alike non-Lepidoptera
  (Trichoptera, some Diptera) that A's head cannot even express.
* **Moth vs non-moth: E clearly better.** Turning a rejection problem into a classification problem
  should be worth more than any scoring rule; I expect E's AUROC above 0.97 and A's best rule
  10+ points below.
* **Open-set on novel Lepidoptera: far stratum +1 to +4 AUROC, near unchanged.** Weaker than the D
  prediction, because the added breadth is insects, not the whole tree.
* **Probe (domain shift): +0.5 to +3 before adaptation, gap under 1.5 after.** Trap backgrounds,
  lighting and by-catch are shared across orders; seeing other insects photographed in the field
  should help a little.
* **A256 vs A: -0.5 to -1.5 points** of species macro-F1 (the 128/256/512 width ladder suggests
  resolution matters at this scale, but less than that ladder's range).

### The audit (`dev/085_tol_audit.py`, run 2026-10-01 over the 94.3 M manifest rows)

| check | rows | |
|---|---:|---|
| duplicate uuids, excluding fills | **0** | fills reuse the uuid of the unreachable row they replace -- 265,774, one-to-one, verified |
| duplicate URLs (scheme-insensitive) | 2,857 | 0.003 %; inside ToL itself |
| our `extend` images sharing an occurrence with a reachable ToL row | 518 occurrences; 446 rows dropped where one image each side | `dev/084` already matched almost everything |
| **rows from lepinet's held-out fold** | **591,594** | 418,081 ToL-selected, 20,680 substitutes, 152,833 our fills/extends |
| occurrences split across lepinet's own folds | 0 | the fold split is clean by occurrence |
| pixel duplicates (sha1 of the 256 px pixels) among 58.2 M downloaded | 17,624 | 1,526 of the hashes span two species |
| placeholders (one hash across >= 20 species) | 40 images, 1 hash | almost none; servers mostly return errors, not stand-ins |
| **excluded** | **584,329** | 565,098 test fold, 15,938 pixel duplicates, 2,857 URLs, 396 occurrences, 40 placeholders |

`exclude.parquet` (uuid, reason) and `audit_report.json` are at
`/12383016/treeoflife_200m/manifest/audit/`, to be re-run when the crawl ends (62.9 M of 94.3 M rows
attempted at snapshot). The pixel hash only catches identical images after resizing; re-encoded or
cropped copies need a perceptual hash, not worth it at 0.03 %. For E none of the test-fold exclusion
matters -- its Lepidoptera are A's -- which is one more argument for doing E first.

My first version of the cross-source rule dropped 213,080 of our rows as "same occurrence as ToL":
most were fills, i.e. our copy of a ToL photograph on a server that refuses us -- the only reachable
copy. A genuine mistake, caught by the duplicate-uuid count matching the fill count exactly; the
rule now compares only `extend` rows against reachable ToL rows.

Seven more servers tripped the crawler's breakers since the substitution pass (20 unreachable now).
The largest, `api.idigbio.org` (now HTTP 410), had already delivered 442,984 of its 443,387 images;
the genuinely new loss is ~120 k images, 79 k of them on `intermountainbiota.org`. A second
substitution pass covers them after the crawl.

## Second revision (2026-10-01, owner): all taxa first; the resolution confound, stated properly

**All taxa first.** E is now our Lepidoptera training split + **every non-Lepidoptera binomial
species in the crawl**, not insects only. Sizing from the audit: 72.2 M manifest rows are on
reachable servers; minus Lepidoptera (~9.3 M with the fill) and the 9 % non-binomial keys, that is
**~57 M non-Lepidoptera images over ~165 k species**, so E ~ 63 M images and ~177 k species with
our 12 k. Consequences:

* **Mixing ratio, fixed in advance: one non-Lepidoptera image per Lepidoptera image.** Each batch is
  half ours, half the rest; the run lasts as many passes over our Lepidoptera split as A (10), which
  is about one pass over the rest -- ~114 M samples, **~29 GPU-hours at 20 M parameters**, 2x A.
  Equal Lepidoptera exposure holds; a natural-proportion mix (9 % Lepidoptera) would either starve
  our classes or cost 11x A.
* **The head is bigger than the trunk.** ~177 k classes x width 256 = 45 M parameters against a
  20 M backbone. Still fixed at 256 for the first run (one factor); the width question is the
  first follow-up if E loses on Lepidoptera.
* **Order level across all taxa.** Kingdom/class/order are in the catalog for every row, so the
  order marginal covers spiders, slugs and plants as well as insects -- strictly more than the trap
  needs, and the reason all-taxa costs the order classifier nothing.
* Insecta-only becomes the follow-up that asks whether *related* breadth carries the effect.

**The resolution confound, stated properly.** The owner is right that every model sees 256 x 256:
the pipeline is `Resize(460)` (short side to 460, square crop) then `aug_transforms(size=256)`. My
first text said non-Lepidoptera images would be "upsampled ~2x" -- that described the intermediate
460 step and was misleading. What matters is **how many original pixels feed the 256 input**:

| | stored (typical) | short side | into the 256 input | |
|---|---|---:|---|---|
| our Lepidoptera (`global_lepi`) | 512 x 384 | 384 | 384 -> 256 | **downsampled 1.5x: crisp** |
| ToL rows (crawler, long side <= 256) | 256 x 192 | 192 | 192 -> 256 | **upsampled 1.33x: soft** |
| trap crops (`flemming`, measured) | 64-300 px | ~100-250 | ~150 -> 256 | **upsampled ~1.7x: softer** |

In E's training data, softness would correlate perfectly with "not Lepidoptera": every moth crisp,
every other organism soft. A CNN detects blur trivially, so it may use it. The trap row is why this
matters rather than being a nicety: **trap crops are softer than either, so a model that learned
"soft = not a moth" would push trap moths toward other orders** -- exactly the error the order
classifier exists to avoid, and one that would hide inside "E is worse at the trap".

**The fix is not a different training resolution; it is giving both sources the same *source*
resolution.** Downsample our Lepidoptera images to the crawler's <= 256 px long side *as they are
loaded* (an item transform ahead of `Resize(460)`, added through the registry pattern; the stored
files stay untouched), so moth and non-moth arrive with the same detail. The baseline has to get the
same transform, otherwise E vs A changes two things at once -- that is **A256**: A, retrained, with
only that transform added. My first version re-encoded files; the load-time transform does the
same without writing 6 M images and without any CPU job. A256 vs A also measures what 256 px
sources cost, and may *help* on probe, since trap crops are low-resolution.

The diagnostic that runs regardless: score E's P(Lepidoptera) on held-out moths at full source
resolution and after downsampling to 192 px short side. A shortcut shows as P(Lepidoptera) falling
on the soft copies.

**The trap order dataset exists** (owner; to be transferred). Meanwhile, Kim Bjerge (Aarhus) has
published trap work with order-level or hierarchical labels that may come with data -- from memory,
to be checked rather than cited: a hierarchical multitask classifier with anomaly detection for
moth-trap images (Ecological Informatics, 2023) [VERIFY], and detection/classification of
flower-visiting insects from camera traps (PLOS Sustainability and Transformation, 2023) [VERIFY].
Web search was unavailable in this session.

## Third revision (2026-10-01, owner): a plain ToL training, every level

> Just do a normal training on the tol dataset with as many taxonomic levels as possible (do not
> stop at order level) and ignore the moth problem for now.

The owner overrode the arm-E design, and the override is reasonable: E was built to make the
Lepidoptera comparison airtight before anyone knew whether ToL training works at all with this
recipe. The plain run answers that first, and is arm C of the original factorial.

* **Corpus:** everything the re-crawl delivers (short side 256 --
  [[2026-10-01-the-crawl-resized-the-wrong-side]]), ToL Lepidoptera and the `global_lepi` fill
  included, no mixing ratio, the recipe's oversampling as usual.
* **Levels: kingdom, phylum, class, order, family, genus, species** -- all seven in the catalog,
  joined by uuid (the manifest carries only the species key). The order classifier is the order
  marginal; so are the other six.
* **The resolution confound is gone by construction**: every image, ours included, now goes through
  the crawler's short-side-256 encode. A256 is no longer needed for this run.
* **Kept, because it is a selection and costs nothing:** the 565,098 test-fold rows in the audit's
  `exclude.parquet` stay out of training, so the model can still be scored on our held-out fold
  later without contamination. The open-set benchmark's novel species are *not* removed ("ignore the
  moth problem"): open-set numbers on our benchmark from this model will be optimistic, and must be
  labelled so if reported.
* **Label hygiene:** species level from binomials only (184,853); genus-only rows are dropped from
  the first run rather than given a partial-label loss (no code for that yet); bare-epithet keys
  excluded. **Every level keyed by its full path** (e.g. `Plantae|...|Moraceae|Morus`), because
  genus names repeat across kingdoms -- *Morus* is a mulberry and a gannet -- and a name-keyed genus
  level would merge them and break the tree that marginalisation assumes.
* **To check before the first epoch:** that lepinet's hierarchy and marginalisation handle seven
  levels and ~185 k species (the head alone is ~47 M parameters at width 256, larger than a 20 M
  trunk), and that the hierarchy built from the catalog is a tree (each child under one parent).

