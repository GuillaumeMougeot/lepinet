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
- 2026-11-08: **AMI-Traps scored** (mt, local GPU) for P5, Nemo, Meghan and BioCLIP-2 zero-shot. Files
  in `/data/au761367/benchmarks/ami/` (`scores/<model>/summary.json`, `we_species_by_flemming_split.json`,
  `flemming_name_matches.json`).
  - **The Danish overlap is real.** 448 of AMI's 892 Denmark frames carry the same filename as a
    Flemming source frame (10 of our 12 traps); template matching confirms 13 of 15 sampled crops
    (correlation 0.93-1.00, against at most 0.70 for random frames). Of the 2,205 W-Europe
    species-labelled crops, 1,089 come from our adapt-night frames, 373 from probe-night frames, 58
    mixed or neither, and 685 from frames not in Flemming (UK and other Danish frames). Nothing
    filtered; the split is a diagnostic.
  - Protocol as run: the paper's 14,105 crops; top-1 micro / macro (macro = mean per-class recall)
    at the labelled rank and above; north_europe list for W-Europe, full vocabulary elsewhere;
    out-of-vocabulary labels count as wrong (Nemo knows 92.3 % of W-Europe species crops). Also
    calibrated macro-F1 (calF1, `dev/093` `score_grouped`, 20 seeds, (trap, night) groups).

  | W-Europe species, micro / macro / calF1 | no TTA | tta3bm |
  |---|---|---|
  | paper, ConvNeXt-B on AMI-GBIF, ~2.6k checklist | 0.829 / 0.764 | |
  | P5 | 0.664 / 0.638 / 0.498 | **0.740 / 0.712 / 0.547** |
  | Nemo | 0.616 / 0.600 / 0.426 | 0.701 / 0.665 / 0.509 |
  | Meghan | 0.586 / 0.578 / 0.397 | 0.667 / 0.629 / 0.467 |
  | BioCLIP-2 zero-shot | 0.362 / 0.403 / 0.272 | |

  | tta3bm, micro / macro | W-Eu genus | W-Eu family | NE-Am species | NE-Am genus | NE-Am family |
  |---|---|---|---|---|---|
  | paper | 0.825 / 0.760 | 0.913 / 0.794 | 0.767 / 0.686 | 0.855 / 0.783 | 0.911 / 0.769 |
  | P5 | 0.754 / 0.752 | 0.909 / 0.770 | 0.654 / 0.591 | 0.761 / 0.730 | 0.908 / 0.683 |
  | Nemo | 0.718 / 0.696 | 0.859 / 0.653 | 0.594 / 0.526 | 0.700 / 0.648 | 0.826 / 0.626 |
  | Meghan | 0.682 / 0.664 | 0.848 / 0.672 | 0.557 / 0.498 | 0.664 / 0.622 | 0.852 / 0.620 |

  - W-Europe species by the Flemming split of the frame, no TTA, micro / macro: P5 adapt 0.735 /
    0.678, probe 0.729 / 0.688, not in Flemming 0.518 / 0.549. P5 scores the same on probe frames,
    which it never trained on, as on adapt frames, so its advantage on our traps is domain match, not
    memorised nights. The independent W-Europe number is the not-in-Flemming line: P5 tta3bm 0.664 /
    0.692.
  - P5 with tta3bm is our best model everywhere; family micro ties the paper (0.909 / 0.913,
    0.908 / 0.911). At species, every model is 9-15 points below on micro. tta3bm adds 5-11 points
    at species and genus for every model.
  - **Predictions scored, provisionally:** P5 within 3 of the paper's W-Europe macro: **falsified**
    (-5.2 with tta3bm, -12.6 without). Nemo 5-10 below P5: just under (-4.7 tta3bm, -3.8 none).
    NE-America 5-15 below the paper: held (-9.5 macro for P5); genus and family closer: held.
  - **Suspect the measurement before the model.** On probe-night frames, P5 scores 0.729 top-1 with
    the north_europe list on AMI's crops, against 0.836 with the full vocabulary on our own probe
    crops (teacher check, Otto journal). Same traps, same nights, a smaller list, and 10 points
    lower. That points at the crops (AMI's YOLO boxes against our crops, PNG mode, resizing) or the
    labels, not at the model. The gap to the paper is not read as a model gap until a matched
    check is done: the same insects in the 448 shared frames, AMI crop against Flemming crop, and
    AMI label against Flemming label.
- 2026-11-08: **matched check: the anomaly was mostly our label mapping.** Files in
  `/data/au761367/benchmarks/ami/` (`pairs_checked.parquet`, `species_truth_resolved*.json`; first
  scores kept in `scores_v1_keymap/`).
  - **Cause.** AMI's keys come from a newer GBIF backbone. For 170 W-Europe species crops neither
    AMI key was in our vocabulary, and the name fallback missed every genus change (Eilema to
    Manulea, Wittia, Katha; Spilosoma lutea, ...). Those crops got a truth outside the list and
    counted as wrong, and their genus truth was wrong too. Fixed by resolving in order: AMI's keys,
    our table's synonym usages, our table's original names, a unique (family, epithet), else
    unresolved (kept, counted wrong). W-Europe species crops in vocabulary: 92.3 % to 99.6 %
    (98.5 % on north_europe). NE-America barely changed (98.4 % direct). Lesson: two GBIF backbone
    versions make a key join look complete while silently re-labelling renamed genera; join through
    synonym usages, never on accepted keys alone.
  - **Pairs.** 898 AMI crops paired with their Flemming crop in the 448 shared frames (IoU median
    0.85), 823 labelled to species by AMI. AMI crops are exactly their YOLO boxes, RGB, no alpha;
    Flemming boxes are 16 % larger by area.
  - **Labels.** After the fix the species labels agree on 93.2 % of pairs (family 98.8 %). The 56
    disagreements are confusable pairs (Chilo phragmitella / Schoenobius gigantella, Pheosia tremula
    / gnoma, Eilema lurideola / complana, ...). On them P5's prediction on the AMI crop matches the
    Flemming label 73 % of the time and the AMI label 16 %.
  - **Crops.** Against a common label (767 pairs) P5 scores 0.906 on the AMI crop and 0.913 on the
    Flemming crop: the crop costs under 1 point. Preprocessing unchanged (the paper pads to square
    with zeros; we keep each model's own, identical for both sources).
  - **Corrected W-Europe species, micro / macro / calF1:**

  | | no TTA | tta3bm |
  |---|---|---|
  | paper (single view) | 0.829 / 0.764 | |
  | P5 | 0.712 / 0.656 / 0.525 | **0.793 / 0.732 / 0.576** |
  | Nemo | 0.666 / 0.623 / 0.453 | 0.752 / 0.688 / 0.541 |
  | Meghan | 0.628 / 0.594 / 0.417 | 0.707 / 0.645 / 0.492 |
  | BioCLIP-2 zero-shot | 0.393 / 0.414 / 0.285 | |

  tta3bm, micro / macro: W-Europe genus P5 0.795 / 0.758 (paper 0.825 / 0.760), family 0.910 /
  0.800 (0.913 / 0.794). Not-in-Flemming frames (n = 685), species: P5 0.739 / 0.718, Nemo 0.714 /
  0.630, Meghan 0.653 / 0.573. NE-America unchanged within 0.006 (P5 species 0.654 / 0.594).
  - **Reading.**
    - Like for like (single view), P5 is about 11 points below the paper on W-Europe species. With
      tta3bm (3x the cost) the gap is 3.6 micro / 3.2 macro. P5 ties the paper at genus macro and at
      family.
    - Part of what remains is AMI's labels: on 7 % of shared crops AMI and Flemming's experts
      disagree, and P5 sides with Flemming. The paper also drops order-rank crops, key -1 and 16
      species; we keep all (0.4 % of W-Europe species crops out of vocabulary).
    - NE-America keeps an 11-point species gap: 12,632-way against a ~2,500-way checklist, and their
      model trained on that region's data. The two causes are not separated.
  - **Predictions rescored:** P5 within 3 of the paper's W-Europe macro: falsified narrowly with
    tta3bm (-3.2), clearly without (-10.8). Nemo 5-10 below P5: held at the margin (-4.4 micro,
    -4.4 macro with tta3bm). NE-America 5-15 below: held.
- 2026-11-08: **iNat21 scored** (Lepidoptera val, same key resolution as AMI; files
  `/data/au761367/benchmarks/inat21/scores/<model>.json`).
  - Resolution of the 1,430 species: accepted name 1,288; our table's original name usage 105; a
    unique (family, epithet) 12 (all genus moves, e.g. Argynnis adippe to Fabriciana adippe);
    unresolved 25. So the closed set is 1,405 species and the open set 25 (250 images), not 142.
  - Closed set, top-1 / macro-F1, raw argmax over the full vocabulary:

  | | species | genus | family |
  |---|---|---|---|
  | P5 | **0.918 / 0.933** | 0.956 / 0.959 | 0.990 / 0.971 |
  | P5 tta3bm | 0.900 / 0.915 | 0.940 / 0.945 | 0.969 / 0.937 |
  | Nemo | 0.907 / 0.929 | 0.953 / 0.959 | 0.988 / 0.967 |
  | Nemo tta3bm | 0.895 / 0.916 | 0.941 / 0.946 | 0.979 / 0.950 |
  | Meghan | 0.874 / 0.906 | 0.937 / 0.947 | 0.986 / 0.961 |
  | Meghan tta3bm | 0.835 / 0.874 | 0.905 / 0.919 | 0.964 / 0.931 |
  | BioCLIP-2 zero-shot | 0.854 / 0.887 | 0.928 / 0.936 | 0.986 / 0.945 |

  - Seen against unseen images (42 % seen), species top-1, single view: P5 0.940 / 0.903, Nemo
    0.933 / 0.887, Meghan 0.887 / 0.864, BioCLIP-2 0.857 / 0.851. The unseen line is the fair one.
  - Open set (25 species), genus / family top-1 / max-softmax AUROC closed against open: P5 0.452 /
    0.740 / 0.768, Nemo 0.432 / 0.752 / 0.764, Meghan 0.456 / 0.752 / 0.733, BioCLIP-2 0.468 /
    0.756 / 0.723. Renames can undercount genus here.
  - **TTA hurts on photos:** tta3bm costs P5 and Nemo 1.2 to 1.8 points, Meghan 3.9. It adds 5 to 11
    on AMI and Flemming. TTA corrects trap-crop framing and orientation; on photos there is nothing
    to correct. So the deployment rule is per domain: tta3bm on trap crops, single view on photos.
  - **BioCLIP-2 zero-shot is the contrast that matters:** 0.854 on curated photos, 0.393 on AMI's
    W-Europe trap crops. Its loss under shift (-46 points) is far larger than P5's (-13 single
    view), so the shift, not the domain knowledge, is what the foundation model lacks.
  - **Predictions scored:** P5 and Nemo at 0.85-0.92: held (0.918, 0.907). Seen 2 to 5 above
    unseen: held (3.7, 4.6). BioCLIP-2 zero-shot at least 15 below our models: **falsified** (6.4
    below P5).
  - **A data flag from mt, checked in our training table:**
    - A few GBIF-backbone species keys put butterflies into the wrong family. Examples: Lycaena
      hyllus, helloides, epixanthe and rubidus, and Biblis hyperia, sit under Sesia (Sesiidae);
      Papilio agestor under Plebicula; Brintesia circe under Loweia (Lycaenidae).
    - **Corrected the same day; my first reading was wrong.** I wrote that this was about 4 k rows over
      fewer than 10 keys and that species classes stay coherent. That only counted cross-family
      names whose genus occurs elsewhere. The full investigation finds 163 mixed classes (23,306
      images), among them *Loweia tityrus* with 1,836 *Brintesia circe*, so some classes are not
      coherent. [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]]
    - Not fixed now; the correction waits until the paper is submitted.
- 2026-11-08: **Rare Species scored** (11,983 images, 400 species, only 2 Lepidoptera; BioCLIP-2's
  own code ported, SimpleShot on frozen features, folds 0-4, top-1; files
  `/data/au761367/benchmarks/rare_species/results.json`).
  - **The protocol reproduces.** BioCLIP-2 ours against its paper's Table 1: 1-shot 60.4 / 59.5,
    5-shot 77.0 / 77.2, zero-shot 76.7 / 76.8. The zero-shot match needs the text built without the
    class rank and common name (the Hugging Face columns are named differently from what its
    `Taxon` class reads); with both fields present zero-shot is 78.7. That the paper's run lost those
    fields is an inference from the match.

  | frozen features | dim | 1-shot | 5-shot |
  |---|---|---|---|
  | BioCLIP-2 | 768 | 60.4 +- 0.7 | 77.0 +- 0.3 |
  | Meghan (head's pre-classification output) | 768 | 60.5 +- 0.8 | 78.0 +- 0.2 |
  | P5 | 1024 | 25.4 +- 0.7 | 44.6 +- 0.4 |
  | Nemo | 1280 | 13.7 +- 0.5 | 27.7 +- 0.3 |

  - **Predictions scored:** "BioCLIP-2 first": falsified narrowly (Meghan +1.0 at 5-shot). "P5 and
    Meghan within 5 points": held for Meghan, **falsified for P5** (-32.4). "Nemo loses 15 or more":
    held (-49.3).
  - **Not read yet.** P5's 1024-d feature is taken before the projection that gives BioCLIP-2 and
    Meghan 768-d, so the -32 mixes fine-tuning with the feature layer. Check: BioCLIP-2's own 1024-d
    pre-projection feature through the same protocol. Only 2 species here are moths, so this says
    nothing against P5 on moths (AMI, iNat21).
