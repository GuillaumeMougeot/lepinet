# Does a cosine-margin objective make a better pretrained biodiversity model?

**Kind:** research · **Status:** OPEN (2026-11-08). Design only; the work starts after the paper is
submitted. Extends W2 ([[2026-08-28-two-directions-checklists-and-our-objective-on-tol]]).

## The question

Large pretrained biodiversity encoders are currently trained with hierarchical contrastive learning
(BioCLIP, BioCLIP-2). In face identification, the strongest pretrained encoders come from large-scale
margin softmax (ArcFace and its descendants) on millions of identities. Species identification has the
same structure: a very large closed set of names. So: **on the same images, does a cosine-margin
objective with hierarchical supervision give an encoder that performs as well or better on the published
biodiversity benchmarks?**

No direction is predicted. In particular, whether margin training keeps within-species variation
(traits, life stage, habitat) is open: an identity model may need to represent that variation in order
to recognise identity across it. The non-species tasks below measure it.

## Design

- **Data:** ToL-65M (`/data/datasets/tol256`, 65.2 M images, 202 k keys), after the two corrections
  below.
- **Arms, one factor:** cosine margin with hierarchical supervision against CLIP-style contrastive
  training with taxonomic text, on the same images, backbone and compute.
- **Starting point:** a self-supervised trunk (DINOv3, which has seen neither labels nor text) so that
  both arms start equal; from scratch if the result justifies its cost. A rough scale from Otto's
  planned throughput: a few hundred B200-hours per arm at 5 epochs, and over a thousand from
  scratch. To be computed when planned.
- **Engineering first:** sharded loading in mini_trainer. A sampled or distributed classifier
  (PartialFC) only if the class count makes the full classifier the bottleneck.

## The evaluation

BioCLIP-2's own suite (arXiv 2505.23883, Tables 1 and 2), with its protocols. Reference values as
read from the paper's HTML (top-1 %, to be confirmed against the PDF when the suite is built):

| species task | zero-shot | 1-shot | 5-shot |
|---|---|---|---|
| NABirds | 74.9 | 82.4 | 92.4 |
| Plankton | 3.9 | 32.0 | 50.5 |
| Insects | 55.3 | 74.6 | 89.3 |
| Insects 2 | 27.7 | 28.4 | 44.3 |
| Camera Trap | 53.9 | 48.1 | 67.7 |
| PlantNet | 96.8 | 85.8 | 94.4 |
| Fungi | 83.8 | 70.3 | 85.0 |
| PlantVillage | 25.1 | 67.6 | 83.9 |
| Medicinal Leaf | 57.8 | 92.0 | 98.4 |
| Rare Species | 76.8 | 59.5 | 77.2 |
| **mean** | **55.6** | **64.1** | **78.3** |

Non-species tasks: FishNet 39.8, NeWT 89.1, AwA2 69.5, Herbarium 19 48.6, PlantDoc 40.4 (mean 57.5).

Plus AMI-Traps and the unseen part of iNat21 Lepidoptera, where the field conditions are.

Conditions:
1. Reproduce every reference value with stock BioCLIP-2 first. Rare Species is done (60.4 / 77.0 /
   76.7 against 59.5 / 77.2 / 76.8).
2. Remove the benchmarks' images from training, as BioCLIP-2 did (near and exact duplicates).
3. A separate set of taxa never trained on, for the strict unseen-taxa reading. BioCLIP-2's paper does
   not say whether its benchmark taxa were held out, and our attempt to tell from the catalog was
   invalid (public-benchmarks entry).
4. Read both the backbone and the head embedding; they differ (Rare Species layer check).

**Zero-shot without a text tower.** A class with no images needs side information. Three routes: the
learned centre for taxa in the vocabulary; genus and family centres below that (BioCLIP-2's zero-shot
on Rare Species has the right genus on 83.1 % of images and the right species on 76.7 %); a name-to-centre
model fitted after training, for labels that are not taxa.

## Prerequisites, before any ToL training

### 1. The rank shift in the ToL labels (2026-11-08)

Rows whose taxonomy lacks a rank have their columns shifted one rank up in the catalog, as our crawl
read it. A row for *Pleydellia aalensis* has `genus` = aalensis, `family` = Pleydellia, `order` =
Hildoceratidae. For reptiles, `class` = Squamata and `order` = Colubridae. `dev/082` builds its key from
genus and species, so these rows became bare-epithet keys.

Measured on `metadata.parquet`:
- **2,599,474 rows in 6,082 bare-epithet folders.** Mostly reptiles and ray-finned fishes (Colubridae
  282 k, Scincidae 103 k, Cyprinidae 88 k, ...), which is why the crawl has no folder for *Anolis* or
  *Thunnus*.
- **The labels are recoverable.** Every one of these rows has `scientific_name` = "Genus epithet ...".
- **2,914 of the 6,082 keys merge several genera** (up to 92). After correction these are 17,824
  species.
- **The per-species cap acted on the merged keys.** 154 keys reached the cap (307,010 rows). True species
  under them got a share of 2,000 images, and the floor was applied to merged counts too.

The fix needs no re-download for what we hold: rename `images/<epithet>/<uuid>.jpg` to
`images/<Genus_epithet>/<uuid>.jpg` (same filesystem, 2.6 M renames) and rewrite `path`, `species`,
`genus`, `family`, `order` and `class`. Which rank is missing varies, so the shift needs a lookup, not a
fixed offset. To do with a dry-run plan and a journalled apply as in `dev/090`, on the shared copy and on
the UCloud copy. Species under-sampled by the cap on merged keys would need a re-crawl to fill up,
and rows with two or more missing ranks were dropped by the crawl (empty key) and are not counted here.

Corrected on 2026-11-09, with the capitalised single-word folders audited (genus- and family-level
records, not shifted): see the Log.

### 2. The Lepidoptera labels

Legacy GBIF Backbone errors: [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]].

## Log

- 2026-11-08: opened (owner). Rank shift found while checking which Rare Species taxa are in the
  catalog.
- 2026-11-09: **TF done: the rank shift is corrected on `/data/datasets/tol256`** (`dev/102`, applied
  with the owner's approval). Build, logs and the previous metadata: `/data/au761367/tol256s/fix_build`.
  - **Cause: the published catalog, not our reading.** TreeOfLife-200M's `catalog.parquet` left-packs
    its rank columns: when a rank is missing, every lower name moves up one column and the epithet
    lands in `genus`. GBIF's backbone has no order for reptiles (Squamata etc. are classes), no class
    for ray-finned fishes, no order for gastropods, no family for many fungi. Of 233 M catalog rows,
    6.51 M are shifted by one rank, 99,791 by two to four. `dev/082 realign` now puts each name back at
    its rank (exception list, then the name's usual column among complete rows, then its ending), so
    the crawl keys on "Genus epithet"; `tests/test_tol_crawler.py` covers fish, reptile, ammonite,
    fungus and a two-rank shift. 206,424 species reach the 50-image floor under the corrected key,
    against 203,878 before.
  - **The 12,151 capitalised one-word folders are not shifted**: genuine genus-level records
    (`scientific_name` is the genus alone), plus 7,525 rows where the source copied the family into
    `genus` (Sciaridae, Apidae). They now have `species` empty and a new column `taxon_rank` =
    genus (2,607,888 rows) or family; their folders stay. Nothing below species passes as a species.
  - **Moved:** 2,599,474 rows from 6,082 bare-epithet folders to `Genus_epithet`, ranks realigned;
    empty ranks filled from the genus's path where every complete row of the genus agrees, and class
    Reptilia / Actinopterygii given to reptile orders and fish `-iformes` orders (the only names not
    taken from the catalog; reversible). A rank the source does not give stays empty.
  - **Floor, recomputed on true species:** 9,558 species have fewer than 50 rows in the whole catalog,
    which the crawl's rule would not have taken. Their 74,482 rows are kept on disk and marked
    `exclude = below_floor`.
  - **Cap, recomputed:** the corrected keys select 415,766 rows never queued, for 1,911 of the 8,652
    corrected species (the merged keys had shared one 2,000 cap, and rows with two or more missing
    ranks had an empty key). Fetched on the workstation under the crawl's policy (4 cores, no UCloud
    CPU): **404,453 ok**; lost 3,249 (404), about 120 other failures, and 7,945 given up: NHM throttled
    5,868, n2t.net and indiabiodiversity.org forbidden, paleo-attachments.colorado.edu dead. Their
    `exclude`: 1,306 pixel duplicates, 1 duplicate URL, none in the test fold. Earlier failures were not
    retried: retries 1-3 had exhausted them, the rest being blocked hosts and the slow servers the owner
    chose to give up on.
  - **Counts, before -> after:** rows 65,203,076 -> 65,607,529; folders 202,269 keys (6,082 of them
    lowercase) -> 214,343 (0 lowercase), 214,338 after the removal below; species-level rows
    62,992,116 in 201,894 species; 192,310 species keep at least one usable row, 137,477 at least 50. `exclude`: below_floor 74,482,
    lepinet_test_fold 548,584, pixel_duplicate 16,543, duplicate_url 2,857, same_occurrence_as_tol 396,
    placeholder 43.
  - **Tree check** (names keyed by full path): 39 species names under several paths (13 across
    kingdoms), 661 genera (467 across kingdoms: homonyms), 16 families, 8 orders, 1 class. What is not
    a homonym is the source disagreeing with itself (*Cephalaria* in Caprifoliaceae and Dipsacaceae).
    Reported, not harmonised; list in `fix_build/check_taxa.parquet`. W3 must key by full path.
  - **Interacts with TAX (not done, owner to decide):** 60,371 of the re-fetched rows still lack a rank
    that no source we hold gives. About 12 k are Riodinidae butterflies (*Calephelis*, *Caria*,
    *Mesosemia*...) with no family or order: these genera have no complete catalog row and are absent
    from our GBIF Lepidoptera data too. Filling them needs COL or the GBIF taxonomy, i.e. TAX.
  - **Mistakes of mine.** (1) `dev/090` enrich left order, family and genus empty on our 1.41 M GBIF
    Lepidoptera rows (`GL_COLS` omitted them), which gave 10,408 species two paths; `dev/102 plan`
    fills them from global_lepi and from each species' complete path. (2) The re-fetch's stall
    watchdog (20 min) was shorter than NHM's throttle backoff, so it restarted 20 times over 7 h and
    the crawl's 6 h give-up never fired (little CPU spent). (3) `add` peaked at 13.6 GB (97 GB free,
    no memory pressure), above the few-GB rule.
  - **UCloud:** there is no ToL copy (the old one is in the trash; the new upload never started,
    storage full). The `dev/087` shard lists are re-planned on the corrected layout (65,607,529 files,
    1,059 shards; the 2026-10-23 lists kept in `upload/superseded_2026-10-23`); the upload waits for
    storage.
  - **Validation on the shared copy after the apply:** 65,607,529 rows, 0 paths missing, 0 files
    without a row, 0 duplicate paths, 0 lowercase one-word folders. Every link, swap and unlink is in
    `fix_build/applied.tsv`; the previous metadata is `fix_build/metadata_before.parquet`.
  - **Bad source labels removed (owner's decision, 2026-11-09).** Among the 12,449 genus- and family-level
    folders, five are GBIF name-matching errors, not taxa: `Genus` (2,000, the placeholder word matched
    as a genus), `Crustacea` (2,000, filed as a bee genus in Apidae), `Symphyta` (53, a sawfly suborder
    filed in Lasiocampidae), and `Antipatharia` (735) and `Scutigeromorpha` (3), order names copied into
    `genus`. Their 4,791 rows, images and folders were deleted from the shared copy (list in
    `fix_build/dropped.tsv`, metadata before in `fix_build/metadata_before_drop.parquet`). The rest are
    genuine genus-level records (one-word `scientific_name`); names that look like ranks (`Leptostales`,
    `Xerophyta`, `Ceropales`) are real genera. A re-crawl with `dev/082` would fetch these keys again.
    **Final:** 65,602,738 rows, 0 missing, 0 extra, 0 duplicate paths; 214,338 folders; genus-level
    2,603,097 rows; shards re-planned (1,059).
- 2026-11-09: **the Lepidoptera labels in ToL moved to COL XR** (owner; `dev/102 colxr`, on `dev/101`'s
  frozen tables, `/data/au761367/tol256s/colxr_build`). `data/global` and TAX are untouched: until TAX
  runs, ToL and the Lepidoptera models name these species differently.
  - `dev/101` works per image from (speciesKey, taxonKey), which only our GBIF download has. Of 8,853,386
    Lepidoptera rows, 5,482,609 are occurrences in that download (by file, else gbifID) and take its
    decision for their own record. Catalog rows take their class's name change where at least 90 % of
    their name's occurrences sit in one legacy class (2,534,254 rows), so 10,286 species with both
    kinds of row do not split. Rows with no class keep their name and take their genus's COL XR family
    where unique.
  - **Result, applied and validated:** 507,927 rows renamed and moved, 569,625 changed in all; Lepidoptera
    species 18,201 -> 17,952 (1,621 old names, 1,586 new, 702 new folders); family changed on 81,054 rows,
    order on none. 134 species exceed the 2,000 cap after merging (largest 5,369; 64,892 rows above
    it), all kept (owner). 1,256 rows newly `below_floor`, counted in this dataset for touched names.
    **Final:** 65,602,738 rows in 201,645 species, 214,089 folders, 0 paths missing, 0 extra files, 0
    duplicate paths. Genus names under several paths 663 (660 before).
  - **Not fixed:** 436,332 rows in 5,539 species are outside the 12.6 k classes the frozen tables cover
    and stay on legacy names; catalog rows cannot be checked for mixed classes (they carry no original
    name); 69 classes `dev/101` leaves unresolved or low-confidence keep theirs. The Riodinidae gap
    above stands: those genera are not in the tables either.
  - Undo: `colxr_build/metadata_before.parquet`, `colxr_build/applied.tsv`. The `dev/087` lists now name
    old paths and wait for the freeze.
- 2026-11-09: target start of the full ToL training: Mon 23 Nov 2026, after a pilot on 16-20 Nov at the full class
  count (192 k species, up to 50 images each, about 8 M images; arms: baseline DINOv3 ConvNeXt-L, margin 0.3,
  TrivialAugmentWide, ViT-L). In mini_trainer (branch, 6f6a3cf): shard loading with virtual epochs, shard predict,
  and an opt-in species margin applied in the loss only, so genus and upper ranks are unchanged (a CPU test
  asserts it; not lepinet's `marginal_arcface`, which applied the margin before marginalising). The head at 192 k
  classes and seven ranks: 1.1 GB, 5.75 GB GPU peak with optimiser state. Open: /work throughput and DDP head
  sync (benchmark jobs 12452221 and after).
