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


## Fourth revision (2026-11-16, owner): the run plan

> no TrivialAugmentWide, DINOv3 ConvNeXt-L, one-arm pilot

**Standing rule (owner, 2026-11-16): on this dataset, one training run per configuration and one seed;
no arms, no seed repeats.** So every choice below is fixed before the pilot, and the pilot tests the
machinery, not the recipe. The 2026-11-09 pilot design (four arms: baseline, margin 0.3,
TrivialAugmentWide, ViT-L) is replaced.

### Data (frozen, nothing changes)

tol256-v1 (`dev/103`; README in the dataset): 60,914,607 training samples in 1,832 shards at
`/12383016/tol256` (1,648 train, 92 val, 92 test; SHA-256 verified where training reads it, job 12462945),
190,199 species, seven ranks keyed by full path (kingdom to species), short side 256. Out of training:
the 1,810 held-out species (`heldout_taxa.csv`: 1,200 species with a trained sibling, 300 whole genera,
20 whole families), Rare Species near-duplicates (6-bit dHash within genus), lepinet's test fold, below-floor
and duplicate rows. Genus- and family-level rows (2.6 M) are not trained on (no partial-label loss). Pilot
set: `/12383016/tol256-pilot`, 9,337,505 samples in 262 shards, every species, up to 50 images each.

### Recipe (one configuration)

| choice | value | why |
|---|---|---|
| trunk | DINOv3 ConvNeXt-L (`timm:convnext_large.dinov3_lvd1689m`, 198 M), self-supervised start | has seen no labels and no text, so a later contrastive arm (ME) starts equal; not BioCLIP-2, trained on this same corpus |
| head | hierarchical (species classifier, every coarser rank its sum, a loss at all seven ranks), cosine classifier as in the campaign | marginal supervision makes congeners neighbours on every backbone (RESULTS Z7) and gives the trap's order classifier for free |
| loss | cross-entropy at every rank, no margin, no EMLA | the margin is ME's question; one factor at a time |
| sampling | natural frequencies under the corpus's 2,000-per-species cap (`class_sampling_power` 0) | the cap already flattens the head; tail weighting costs robustness under shift (finding 5, Z1d) |
| augmentation | the recipe's affine without dihedral turns (horizontal flip, rotation up to 15 degrees, zoom 1.0-1.1); **no TrivialAugmentWide** (owner) | TAW traded -1.7 on photos for +8.6 on traps (AG); a general backbone is judged on photos too |
| resolution | 224 training crop from the 256 short side (read 256, train 224, as Otto and OT3) | storage is 256; 320 would upscale (mt agrees) |
| batch | 96 per GPU, 8 GPUs, global 768 | the setting run on this trunk (OT4); 224 would fit 128 |
| optimiser | MuonAuxAdamW, one-cycle, **lr 8e-4 unscaled** (below) | the one choice a one-arm pilot cannot settle |
| seed | 42, one run | owner's rule |
| epochs | set from the pilot's throughput and the GPU budget; target 5 passes of 60.9 M | the ME design's scale |

**The learning rate (owner to choose before the pilot).** Evidence on this exact trunk, both at batch 96 on
one GPU: 8e-4 on Lepidoptera (VD, OT4: 5 M images, 12 k species) and 2e-4 on Pl@ntNet-300K (VD: 4-8 points
better than 8e-4 there; 306 k images, 1 k species). The small dataset wanted the lower rate; ToL (55 M training
images, 190 k species) is far closer to Lepidoptera. mini_trainer does not scale lr with world size, so 8e-4 at
global batch 768 is a lower rate per sample than OT4's. **Recommendation: 8e-4, unscaled.** (The plan's first
draft said 2e-4 from the Pl@ntNet result; that read the dataset-size difference the wrong way round.)

**Decided 2026-11-16: lr 8e-4, unscaled.** The owner asked for an lr finder, validated on our two datasets
with measured rates and used on ToL only if credible (`dev/108_lr_finder.py`, RESULTS LF, plots in
`data/lr_finder/`). Range test from the recipe's initial state, 200 steps, 1e-6 to 1e-1, mini_trainer's own
training step with gradient accumulation to the recipe's batch:

| dataset | batch | lr at smoothed minimum | minimum/10 | valley | measured |
|---|---|---|---|---|---|
| Pl@ntNet-300K (987 species) | 96 | 7.8e-4 | 7.8e-5 | 8.6e-5 | best 2e-4; 8e-4 cost 4-8 points |
| Global Lepidoptera (12,632) | 96 | 8.7e-4 | 8.7e-5 | 9.1e-5 | 8e-4 trained (VD, OT4) |
| ToL (190,199; 200 k sampled rows) | 768 | 1.1e-3 | 1.1e-4 | 1.1e-4 | none |

Critical reading. The finder gives the same answer on Pl@ntNet and Lepidoptera, where the measured best rates
differ 4x, so the difference is a long-horizon effect (a small dataset over 10 epochs) that 200 steps from the
initial state cannot see; this is the project's rule about cheap proxies again. fastai's suggestion rules land
near 1e-4 everywhere, 2x below Pl@ntNet's best and 9x below the rate Lepidoptera trained well at, so they are not
used. The one thing the test does show is that ToL's 8x larger batch and 190 k-species head leave the stability
edge where it was (raw loss lowest near 6e-4 and rising from 1e-3 on all three). ToL is a large dataset like
Lepidoptera, so 8e-4, the default, stands.

**Corrections from mt's spec build (2026-11-16).** `lepinet-BU.yaml` weights the loss of three ranks, and the loss
zips weights with levels, so a seven-rank run would have trained species, genus and family only: the specs set
`[1] * 7` (trainer log "[1.0 x 7]"). The recipe's standard augmentation includes dihedral turns and vertical flips;
general photos have an upright orientation, so `dihedral: false` (horizontal flip, rotation up to 15 degrees, zoom
1.0-1.1, affine_p 0.75 kept), which is what this plan meant by "crop, flip". Warmup counts virtual epochs (four per
pass): half a virtual epoch in the pilot, 10 % of training (half a pass) in the full run, the campaign's proportion.
The smoke also asserts identical weights on all ranks after its last step.

### The pilot (one arm, 16-20 Nov)

First a smoke of a few hundred steps (minutes). Then the same recipe on `tol256-pilot` (8.09 M training
samples), on the 8 B200 planned for the full run, for one pass (about 20-35 min at the expected speed). It must show, before the full run is submitted:
1. throughput (img/s per GPU and overall) at the full class count, with the head on GPU (1.1 GB, 5.75 GB peak
   measured with optimiser state);
2. DDP correctness: the head's gradient sync (about 3 ms per step at 4 GPUs, shard-bench-ddp) and identical
   weights across ranks after a step;
3. loss falling at all seven ranks, no non-finite values (fp16 vs bf16: no margin, so mt's default applies);
4. checkpoint and resume across a job boundary (a multi-day run will be extended by `ucloud q`), and the
   20-minute log-stall guard;
5. the evaluation pipeline end to end on the pilot checkpoint (below), so no eval is first written after
   the full run.
Its scores are not compared with anything and do not choose anything.

### Evaluation, fixed before training

| what | measure | answers |
|---|---|---|
| val and test shards | macro-F1 and accuracy at every rank | does the run train |
| the 1,810 held-out species | placement: right genus (1,200 species), right family (300 genera), right order (20 families), by the coarse marginals; few-shot species accuracy (1 and 5 shots, class centres) | a backbone's job on taxa it never saw |
| Rare Species | 1- and 5-shot, BioCLIP-2's protocol (reproduced: 60.4 / 77.0 against 59.5 / 77.2) | comparison with BioCLIP-2 on a decontaminated set |
| BioCLIP-2's other suite tasks | as published | to be built during training; not a blocker |
| lepinet's test fold (Lepidoptera) | species macro-F1, against DINOv3 BU trained on Lepidoptera alone (VD) | does breadth cost or help in-distribution (W3 prediction: within 1 point) |
| AMI trap crops | probe and probe-HO species macro-F1 (north_europe, tta3bm, calibrated by (trap, night)), before any adaptation; trap order accuracy | does breadth help under shift (W3 prediction: +1 to +4 on probe before adaptation) |
| open-set | C3b's benchmark stratified near / mid / far, each with its best scoring rule | W3 prediction: far gains, near does not; labelled optimistic (novel species not removed) |
| geometry | congener-neighbour rate (Z7), logged every pass with held-out 5-shot | the taxonomy as geometry at 190 k classes |

### Compute

Measured so far (mt, 2026-11-16): the loader alone reads 26-29 k img/s on a 96-vCPU node (330-490 img/s per
CPU worker, decode-bound; 256-px shards decode faster than the 384-px benchmark); the 190 k-species head
costs 1.11 GB (about 4.5 GB with gradients and AdamW states) and its all-reduce +3.0 ms per step at 4 GPUs,
about 2 % of a step. Not measured: the trunk on shards, and anything at 8 GPUs; the pilot measures both.
Nearest model number: OT4 (this trunk at 320 px, 1 B200) ran at about 509 img/s, so about 900-1,000 at 224
(extrapolated). At 8 GPUs and 6,500-8,000 img/s: one pass of the 54.8 M training samples takes 1.9-2.3 h, five
passes 10-12 h, about 80-100 B200-hours; the pilot a few B200-hours. The full spec goes to the owner with
the pilot's measured numbers.

### Still to build before the pilot

- **A DDP launcher for shards** (mt): campaign jobs run one process per GPU and the production path reads
  parquet only. A torchrun job spec (`--nproc_per_node 8 -m mini_trainer.train --input <shard root>
  --class_spec class_spec.json`, seven levels) with the 20-minute stall guard. DDP on shards has never run end
  to end, hence the smoke.
- **class_spec.json** (101 MB) staged next to the shards or mounted, and passed explicitly.
- **Evaluation on ToL class names** (lepinet side): ToL classes are full taxonomic paths, not GBIF keys, so
  the trap scorer (keys -> north_europe list -> `flemming.py score`) and the Lepidoptera test fold need a
  name-to-key map. ToL's Lepidoptera are on COL XR names and lepinet's on legacy keys, so the map goes
  through `dev/101`'s tables. Also: held-out placement and few-shot, Rare Species and C3b's open-set from a
  ToL checkpoint. All written and run on the pilot checkpoint, not after the full run.

### Risks written down now

- **The queue's login.** The UCloud refresh token expired twice (30 Aug, 11 Nov) and silently stopped
  `ucloud q tick`, so `auto_extend` stops working. A multi-day run needs `ucloud q tick; echo $?` checked
  daily, or `hours` set to the whole run.
- **A silent freeze** (OT5 attempt 1): the stall guard now stops the job; resume from the last checkpoint.
- **/work throughput** at 4 GPUs reading 1.8 TiB of shards: measured by mt's shard benchmark; the pilot
  confirms it at scale.
- **The levels caveat** (`levels` < 7 truncates the head): not hit by a seven-rank run.

### Not in this run

No margin (ME), no TrivialAugmentWide, no self-training on traps, no held-out-observer list (Z6: the
observer factor re-ranks nothing), no second seed, no arm E or the A-D factorial.

### Evaluation code for the pilot (2026-11-16)

`dev/109_tol_eval.py`, written ahead of the pilot as the run plan asks; outputs in `data/tol_eval/`.
- **Map** ToL species -> lepinet legacy keys (dev/101's `name_map.csv`, COL XR name, then legacy name):
  11,879 of lepinet's 12,041 vocabulary species (98.7 %), 626,118 of the 629,742 test-fold images (99.4 %),
  475 of 486 trap species (47,320 of 47,905 crops), 1,966 of 1,977 north_europe keys. All 162 unmapped
  vocabulary species are absent from ToL training (138 not in tol256, 23 below its floor, 1 test fold only),
  so the map misses nothing ToL trained on. 264 legacy keys share a ToL class with another (COL XR merged
  them); `relabel` answers with the key that has the most training images. Genus and family keys by majority
  (169 of 4,450 genus paths split over several keys).
- **Route** for test fold and traps: `mt_hpredict --data_index index_<set>.json --class_list lepinet.tol.classes`
  (or `north_europe.tol.classes`, 1,939 classes), then `relabel` to three GBIF-keyed ranks for `score.py` and
  `flemming.py score`. `map_species.csv` also carries `tol_index` for the logit-level tta3bm path.
- **Held out**: 89,706 images (at most 50 per species) of all 1,810 species. Placement at genus (1,200
  species), family (300 genera) and order (20 families) by each rank's argmax, macro over species; 2 of the
  1,200 have a genus that is not a model class (their trained sibling left in the freeze), reported apart.
  Few-shot 1 and 5 with BioCLIP-2's SimpleShot, ported from the Rare Species run (60.4 / 77.0).
- **Rare Species**: the same SimpleShot on the 11,983 images, split by whether the species is a ToL class.
- **Z7** on the 190 k species prototypes, chunked, raw and centred.
- Tested: `selftest` (known answers for few-shot, congeners, placement, names), a `relabel` round trip (all
  three ranks exact), every index's image paths, and `embed` on CPU with OT3 on 16 test-fold images (15 right;
  768-d head input, prototypes from the cosine head's weight direction). Not run: `embed` on a ToL checkpoint,
  which waits for the pilot.
- `embed` defaults to `min(4, effective_cpus())` loader workers (mt's default of one per host core, each with a
  copy, reached 10 GB at 16 images and was killed; CLAUDE.md's oomd rule). On UCloud pass `--num_workers` with
  the job's cgroup count.

### Smoke (2026-11-16): passed; measured throughput

tol-smoke (job 12463902, 8 B200, 21 min): train loss falling at all seven ranks over three short epochs (species
11.29 -> 9.03, kingdom 0.31 -> 0.14); "DDP equality: OK over 8 ranks, 349 tensors"; a stopped launch resumed from
`checkpoint_last.pth` and finished. **Throughput about 5,650 img/s on 8 B200 (about 710 per GPU)**, below the
6,500-8,000 assumed above; GPU memory 27-30 GB of 180. Per-epoch boundary (eval, checkpoint, loader respawn) about
4 min, a slow tail of about 2 min per epoch, 78 s to the first step. Full run at this speed: one pass of 54.8 M
about 2.7 h, five passes about 14 h, about 110 B200-hours. 710 per GPU is about two thirds of what OT4's 320 px rate
scales to at 224 px, so the pilot should log GPU utilisation to tell a loader limit from a compute limit before the
full spec. Pilot submitted: tol-pilot 12464304 (one pass of tol256-pilot, worst case 24 B200-hours).

### Pilot (2026-11-16): passed

tol-pilot (job 12464304, 8 B200): one pass of the pilot set's 8.09 M samples in 41.0 min of training (four virtual
epochs). Steady 5,709-5,753 img/s overall (714-719 per GPU), flat across epochs; each boundary (eval of 156 k val
samples, checkpoint, loader respawn) 2.4 min; a tail of about 1.9 min lost at the end of every epoch, fixed in time.
Losses finite and falling at all seven ranks; val species loss 6.41 -> 3.23, val acc1 5.2 -> 38.1 %, val macro-F1
0.03 -> 0.34; weights equal on 8 ranks; 28 GB of GPU memory; about 7 B200-hours. mt's reading of the speed
(arithmetic, not a measurement): compute-bound, since part of the step does not shrink with resolution (Muon's
Newton-Schulz, the float32 190 k-species head, the gradient all-reduce on 8 GPUs); the full run logs GPU utilisation
to settle it. Full run as specified (5 passes in 10 virtual epochs): about 14-14.5 h, about 115 B200-hours, 20 h limit
without auto-extend (worst case 160). The evaluation scripts (`dev/109`) run next on the pilot checkpoint.

**Full run submitted 2026-11-16 15:20** (owner, in mt): tol-full, job 12464522. Pilot weights for the eval test:
`/12347837/repos/mini_trainer-campaign/tmp-runs/tol_pilot/weights/last.pt` (2.06 GB; loads with `Classifier.build`, seven outputs).

### After the full run: proposed next steps, predictions written before any result (2026-11-16)

Not approved; for the owner. The one-run rule holds: evaluations first, then at most two trainings.

| step | what | cost | prediction |
|---|---|---|---|
| E1 | the frozen ToL model against BioCLIP-2 and against raw DINOv3 ConvNeXt-L (its starting point), same protocols: BioCLIP-2's suite (1- and 5-shot species tasks; the five non-species tasks), Rare Species, held-out 1,810 taxa (few-shot and placement) | eval only, a few GPU-hours | ToL model far above raw DINOv3 on species few-shot (+10 or more on the mean); below BioCLIP-2 on the suite mean (3x its data, longer training, text), within 5 points on Insects and Rare Species; further below on the non-species tasks, where a classification objective discards within-class variation (route 5, Z review). BioCLIP-2 has seen the held-out taxa, so that column favours it |
| E2 | open-set on the held-out taxa: held-out images against val images of trained species, near (genus kept) / mid (genus withheld) / far (family withheld), each scoring rule | eval only | graded near < mid < far, as C3b; far above 0.95. This replaces C3b's benchmark for ToL models, whose novel species were trained on |
| E3 | the ToL model's Lepidoptera head, no fine-tuning, on lepinet's test fold and the trap probe (dev/109 map) against DINOv3 BU trained on Lepidoptera alone (VD) | eval only | W3's committed predictions stand: test fold within 1 point; probe +1 to +4 before adaptation |
| E4 | fine-tune the ToL model on global_lepi with VD's recipe (campaign, shards), one run, against VD (same trunk, DINOv3 start) | one run, 1 B200, about 14 h | test fold +0.5 to +1.5 over VD; probe and probe-HO +2 to +5 before any trap adaptation |
| E5 | Otto's recipe (OT3/OT4: gated labels, TrivialAugmentWide) from the ToL model, one run, against OT4 (same trunk from DINOv3) and OT3 (production) | one run, 1 B200, about 28 h | probe level with OT4 and OT3; probe-HO above OT4 (OT4's weak point was species the adaptation never saw), so a candidate to replace OT3 only if both hold |
| E6 | the trap order classifier (the order marginal) on the owner's trap order dataset | eval only, when the dataset arrives | |

Order: E1-E3 the day the run lands (they decide whether E4 and E5 are worth running); E4 before E5. A model
"better as an embedding" means E1's few-shot and placement, not accuracy on its own label set.

**The owner's view on E1 (2026-11-16), recorded beside my prediction so both can be scored.** BioCLIP-2's extra data
is mostly what our dataset cuts on purpose (the head above the 2,000-per-species cap, the tail below the 50-image
floor), so it may add little, and our labels may be cleaner (COL XR, rank fixes, decontamination). Its text is
largely the taxonomy, which our hierarchical head already supervises. The one difference the owner expects could
matter is training length (epochs), to be measured if E1 shows a gap. So the owner's expectation is that the
ToL model is level with BioCLIP-2 on species tasks, unless training length decides it. E1-E3 are queued to run as
soon as tol-full ends (owner, 2026-11-16).

### dev/109 end to end on the pilot checkpoint (2026-11-16): every step runs

The pilot's `last.pt` (one pass of 8.09 M, val species macro-F1 0.34), locally on the shared 5090, 4 loader workers.
The numbers test the code, not the model.
- **embed**: held out 89,706 images in about 2 min, Rare Species 11,983; 1,536-d head input. The species
  prototypes are the head's weight-norm direction (`head.fc.linear.parametrizations.weight.original1`,
  190,199 x 1,536); the direction is what a cosine neighbour needs, so this is right for Z7.
- **placement** (macro over species), falling from kingdom to the placement rank as it should. Near (1,200 species):
  genus 46.9 %, family 70.1, order 80.1, class 94.9, kingdom 98.2. Whole genus (402): family 47.3, kingdom 97.9.
  Whole family (208): order 41.1, class 83.4, kingdom 97.3. Held-out few-shot 1/5-shot 46.1 / 68.7 %
  (near 50.9 / 74.0, mid 40.9 / 64.0, far 27.3 / 44.6).
- **Rare Species** 1/5-shot 42.8 / 60.2 % (BioCLIP-2 in our port 60.4 / 77.0); 356 of its 400 species are ToL classes.
- **Z7**: nearest prototype a congener 90.2 % (centred 92.2), chance 0.03 %; not comparable with lepinet's Z7
  (12 k species, DINOv3 BU 74.4).
- **Test-fold route** on a 3,000-image sample (2,281 species): `mt_hpredict --data_index --class_list
  lepinet.tol.classes` ran unchanged, `relabel`, then mini_trainer's `dev/lepinet/score.py`: species accuracy
  55.8 %, macro-F1 0.40 / 0.61 / 0.88 at species / genus / family (macro on a sample this size is noisy).
- **Bug found and fixed**: `relabel` set `known_label` to "the ToL model can name this label", and `score.py` drops
  every `known_label` 0 image, so the 1.4 % of test images whose species ToL lacks would have been silently removed:
  a filtered test fold, about +0.8 points of accuracy here. `known_label` now means lepinet's vocabulary, as for
  every lepinet model, and those images count wrong; the nameable share is printed only. `flemming.py score` drops nothing, and its `known_label` now
  means lepinet's vocabulary too, as for lepinet's own trap scores.
- mt writes two extra columns (`prediction_made`, `correct`); `relabel` drops them, since `correct` is stale
  after relabelling. mt appends `_1` to an output folder name that is new (`pred_testfold_s3000_1`).
