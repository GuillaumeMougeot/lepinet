# Where do our models stand on public benchmarks?

**Kind:** research · **Status:** OPEN (2026-11-07). Plan and predictions, before any download or
run.

Every number in this project so far is on our own data: the test fold, the Danish trap crops, the
probe nights. That shows what changed between our runs, but not where the models stand against
published work. The owner asked for a small check (2026-11-07). mt planned it; the plan file is in
mt's scratchpad (`bench/PLAN.md`), and the substance is below.

## Setup

- **Models:**
  - Otto, once it lands;
  - P5, Nemo (MAMBO_v3), Meghan (MAMBO_v2);
  - BioCLIP-2 frozen (`imageomics/bioclip-2`, unmodified) as the reference.
- **Vocabularies:**
  - Nemo and Meghan share 12,632 GBIF keys;
  - P5 has 12,041, a subset of those;
  - the north_europe list has 1,977.
- **Single view everywhere,** because the published numbers are single view.
- **No eval set is filtered.** Leakage is reported as a partition, never removed.

## The three benchmarks, in order

### 1. AMI-Traps (Rolnick lab, ECCV 2024, arXiv 2406.12452)

The only public light-trap benchmark with published numbers, so the one that matters for Otto.

- **Data:** Zenodo record 12554005 (v2), 16.0 GB, MIT licence; 14,105 moth crops from trap images.
- **Labels:** GBIF accepted keys, so the mapping to ours is exact.
  - **W-Europe (UK, Denmark):** 2,205 crops labelled to species, covering 243 species; 1,940 crops
    and 227 species are in Nemo's vocabulary.
  - **NE-America:** 3,157 crops, 271 species; 3,090 crops and 255 species are in Nemo's vocabulary.
  - **C-America:** genus and family only.
- **Protocol: the paper's.**
  - Each crop is scored at its labelled rank and above.
  - Top-1 micro and macro per rank and region; coarser ranks are summed species probabilities.
  - A label outside our vocabulary counts as an error.
  - W-Europe uses the north_europe list (their analogue is a 2,603-species checklist); the other
    regions use the full vocabulary (12k-way against their ~2.5k).
- **Published (ConvNeXt-B trained on AMI-GBIF), micro / macro:**
  - W-Europe: species 82.87 / 76.41, genus 82.45 / 75.98, family 91.27 / 79.39.
  - NE-America: species 76.75 / 68.56, genus 85.46 / 78.28, family 91.07 / 76.87.
  - C-America: genus 51.92 / 68.63, family 72.24 / 44.31.
- **Leakage check first.** The 892 Danish images may come from the same traps as our adapt, trap and
  probe nights. Otto self-trains on the adapt nights.
  - Match by file hash and by timestamp and trap ID.
  - If any overlap, Denmark is reported separately, not removed.
- **AMI-GBIF's test split is unusable:** 44 % (W-Europe) and 67 % (NE-America) of its GBIF ids are in
  our training table.

### 2. iNat21, the Lepidoptera part of the official validation set

- **Data:** 8.4 GB `val.tar.gz`; non-commercial research licence. 1,430 species x 10 images.
- **Vocabulary match:** 1,288 species are in Nemo's vocabulary and 142 are outside, which makes an
  open-set slice. The full vocabulary is used: north_europe covers only 262 of these species.
- **Heavy leakage:**
  - 80 % of our training rows are iNaturalist research-grade.
  - 42 % of the validation images match one of our training observations on (rights holder, day,
    lat/lon); 56 % on (holder, day) alone.
  - BioCLIP-2's training data contains iNaturalist too.
  - So the full slice is reported, and the seen/unseen partition is a diagnostic.
- **Protocol:**
  - argmax over the full vocabulary; BioCLIP-2 zero-shot with taxonomic-name prompts over the same
    vocabulary;
  - top-1 and macro-F1 at three ranks;
  - for the 142 out-of-vocabulary species: genus and family accuracy, and max-softmax AUROC of in- against
    out-of-vocabulary.
- **Published:** nothing comparable for a Lepidoptera slice; BioCLIP-2 is the reference.

### 3. Rare Species, plus Meta-Album Insects

- **Data:** `imageomics/rare-species`, 11,983 images, 400 IUCN species. Only 2 species are Lepidoptera.
- **What it measures:** how general our frozen features stay after fine-tuning on moths, not moth ID.
- **Protocol: BioCLIP's own** (arXiv 2311.18803):
  - SimpleShot nearest-centroid on frozen embeddings, 1 and 5 shots, 5 seeds;
  - zero-shot for BioCLIP-2 only.
- **Published:** BioCLIP 1-shot 44.9, 5-shot 65.7, zero-shot 38.0. BioCLIP-2's values are to be taken
  from its paper (arXiv 2505.23883, Table 1); our BioCLIP-2 run also checks that we reproduce the
  protocol.
- **Meta-Album Insects** (117 pollinator species) only, with the same protocol; the rest of BioCLIP's
  suite says nothing about moths. Licence to check.

## Cost

- **Downloads:** about 29 GB to `/data/au761367/benchmarks/`.
- **GPU:** about 40 k images, one view per model; under one GPU-hour for all five.
- **CPU:** minutes.
- Needs the owner's approval for the downloads.

## Predictions (before any run)

- **AMI-Traps, W-Europe, species:**
  - P5 within 3 points of ConvNeXt-B's macro (76.4); its top-1 on our probe crops is 0.836 (VE).
  - Otto above P5 by 2 to 5 points.
  - Nemo 5 to 10 below P5, as on our probe nights.
  - If the Danish images overlap our adapt nights, Otto's Denmark number is optimistic and the UK
    number is the honest one.
- **AMI-Traps, NE-America, species:** every model below ConvNeXt-B by 5 to 15 points. Ours choose among
  12k species against their ~2.5k, and none was adapted to North American traps. Genus and family
  closer.
- **iNat21:**
  - in-vocabulary species top-1 of 0.85 to 0.92 for P5 and Nemo;
  - seen images 2 to 5 points above unseen;
  - BioCLIP-2 zero-shot at least 15 points below our fine-tuned models.
- **Rare Species, 5-shot:** BioCLIP-2 frozen first. P5, Otto and Meghan, all fine-tuned from
  BioCLIP-2, lose a little generality (within 5 points of BioCLIP-2). Nemo, an EfficientNetV2-S
  trained on moths only, loses 15 or more.

## Log

- 2026-11-07: opened; mt's plan journalled. Metadata only fetched so far (~1.1 GB, mt's scratchpad);
  nothing downloaded or submitted. AMI-Traps confirmed. Order: AMI-Traps, iNat21, Rare Species and
  Meta-Album Insects.
- 2026-11-07 late: owner approved the ~29 GB downloads in mt's session (AMI v2 with md5 check,
  iNat21 val, Rare Species, to `/data/au761367/benchmarks/`). Meta-Album not approved; mt checks its
  licence and asks separately. Danish overlap: our 47,905 crops (12 traps, LV1-4, OH1-4, SS1-4,
  2022-2023) cannot be hashed against AMI's full frames, so the match is on trap name and the
  snapshot timestamp in the filenames. Calibrated macro-F1 on AMI needs (trap, night) groups from its
  metadata; mt reports if they are missing.
