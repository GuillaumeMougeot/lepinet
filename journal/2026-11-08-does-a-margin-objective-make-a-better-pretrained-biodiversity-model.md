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
- 2026-11-10: ToL freeze checks (`dev/103_tol_freeze.py`, build dir `tol256s/freeze_build`), steps 1, 7, 8 and 4
  of the freeze list.
  - Decode: all 65,602,738 images decode; 0 corrupt, so no integrity exclusions.
  - Names: 9,133 looked up in COL XR (5,536 Lepidoptera outside `dev/101`'s tables, 3,597 species with an
    empty rank after filling from the same genus or family in the dataset). Accepted 7,922 (639 k rows):
    exact species match, subgenus ignored, not an ambiguous synonym, kingdom equal to the dataset's, and order
    Lepidoptera for a Lepidoptera name. 1,211 unresolved (71 k rows) keep their names. The kingdom check
    rejected 11 (e.g. *Euglypha*: Protozoa here, Chromista in COL XR), the order check 1 (*Enantia spinifera*
    came back as a flatworm).
  - Shortfall: the crawl selected 94.3 M rows; 65.5 M are held. Of the 28.8 M missing, about 20 M sit on hosts
    that refused us (403 bot blocks: mediaphoto.mnhn.fr, observation.org, EOL, naturalis, ALA, ...), which only
    an access request fixes; 7.7 M on hosts dead on 1-3 Oct; 0.76 M retryable on open hosts; 0.6 M permanent
    (404, size). A 5-request probe per dead host found 6 serving images again (Harvard, cch2, Colorado,
    TACC, biodiversity.bt, rdu.unc). Re-fetch `tol256s/retry4` (workstation, plain UA): 725 k rows, species
    below 50 images first (190 k rows, 19,219 species; 2,000 would reach 50), then under half of target
    (535 k rows, 12,756 species). Arrivals by Thursday evening go into the freeze.
  - My mistake: the first `shortfall` used `np.isin` against the 65 M held hashes, which re-sorts them for
    each of 2,574 manifest parts; it ran 20 h on one core at 9 GB without passing the first part. With
    `searchsorted` on the sorted array it took minutes.
  - Held-out taxa (owner's choice): seeded draw (seed 20261113), stratified by kingdom share, common taxa only
    (each species at least 200 usable images), each keeping a trained sibling, Lepidoptera not drawn so all of
    lepinet's families stay in pretraining: 1,200 species (genus keeps a sibling), 300 whole genera (402
    species; family keeps a sibling genus), 20 whole families (208 species; order keeps two others). 1,810
    species (0.94 %), 1.50 M images (2.4 % of usable). List `freeze_build/heldout_taxa.csv`, draw
    `freeze_build/heldout_draw.py`.
  - Benchmark overlap: the dataset holds no EOL rows (EOL refused the crawl), so Rare Species' EOL content ids
    match nothing; the same photographs arrive through GBIF instead. A 64-bit difference hash within the
    same genus (`dev/103 bench`, 1.47 M dataset images in Rare Species' 316 genera, 5 min) finds 3,775 at
    distance 0 and 5,883 within 6 bits, covering 4,995 of the 11,983 benchmark images (42 %). Pairs checked by
    eye: up to 3 bits the same photograph, except museum specimen sheets on one printed template (QR code,
    ruler, label) that match different specimens at 3-5 bits; 6-7 bits mixed. Cutoff 6 bits: the false
    positives cost a few hundred of 62 M images, a missed duplicate biases every Rare Species score. Any
    Rare Species number from a model trained on the pre-freeze ToL is contaminated.
  - First dry run (`dev/103 plan`, before the re-fetched rows): 65,602,738 rows; 17,963 Lepidoptera files
    move to their accepted names (138 new folders); exclude adds heldout_taxa 1,497,064, benchmark_overlap
    5,834, bad_kingdom 235, below_floor +735. Usable species rows 60.85 M in 190,102 species (135,355 with
    50 or more), plus 2.60 M genus- and 7.5 k family-level rows. 157 species had every observation hashed
    into val or test; the plan now gives such a species its lowest-hash observation in train.
- 2026-11-11: second dry run (`freeze_build/plan`): identical numbers, `species_moved_to_train` 157, and a pass
  over the output finds every one of the 190,125 usable species keys (full path) with train rows. Owner
  approved the apply (after the re-fetch fold-in, delta shown first), the upload (pilot first to
  `/12383016/tol256-pilot`, then `/12383016/tol256`, nothing replaced) and the pilot caps (50 train, 5 val,
  5 test per species).
  - Harvard (data.huh) in retry4: 138,683 of its first 150,000 rows rejected by the 25 MB cap. They are 50 MP
    baseline JPEGs of 25-38 MB on Harvard's S3 (`huhspecimenimages/JPG/<barcode>.jpg`); the bucket has no smaller
    rendition we could find (an older `herbaria2/.../full/` set has 250 px thumbs, below 256), and baseline
    encoding rules out reading a low-resolution prefix. Worth: 957 species reach 50 only with Harvard. The
    narrowest pull (905 species, 12,847 images, about 0.4 TB; `tol256s/retry5`) ran at 6 images a minute, about
    41 h, so it was stopped after 5 minutes (owner: Harvard is known slow, give up quickly). Its 32 images are
    kept. These species stay below the floor in this freeze.
  - Consumer check on the smoke set. mini_trainer (`feature/lepinet-integration`; shard format unchanged since
    the pinned 6f6a3cf): `verify --full` passes, the 7-rank class spec loads with its hierarchical parser, and
    `ShardDataset` streams every val/test sample with 7-rank labels. lepinet: `train.parquet` had no
    `image_path`, so lepinet built `<speciesKey>/<filename>` from the full-path key and found 0 of 2,636 images;
    `dev/103 train` now writes `image_path` (relative to `images/`) and lepinet finds all of them. lepinet has no
    shard reader and its `gen_df` holds about 33 GB per copy of the 61 M-row frame (three copies), so it reads
    the freeze from the image tree on the workstation, or a subset; the uploaded shards are for mini_trainer.
- 2026-11-12: re-fetch stopped at 2 images/s (owner: not worth it against 63 M); an `--offline` pass recorded
  the 31,807 images on disk that unfinished parts had not written to their meta. 111,220 images kept (TACC
  39 k, unimus 28 k, Colorado 19 k, Harvard 15 k, others), all decode, none within 6 bits of Rare Species.
  The fold-in found three defects, each in my code or the crawler's, none in the shared copy yet:
  - The crawler records a row recovered from disk with `content_hash` "" and width/height 0. `dev/102 add`
    took "" for a hash, so all 31,866 recovered rows matched each other as pixel duplicates; now null, with the
    size from the file (27 real pixel duplicates remain). The original crawl has 1,636,542 such rows: their
    duplicate check was right, but their size was 0; `dev/103 sizes` read them and the plan fills them in.
  - `dev/102 add` marked every added row species-rank, so 42,767 genus-level rows (3,311 one-word names such as
    *Bryum*) would have entered species training; now species empty and rank genus, as `dev/102 plan` does.
  - The plan's "every species has a train row" was keyed by kingdom|species, training by full path, so two
    names on two paths (*Dianella*, *Lycoperdon*) had a path with no train row; now keyed by full path.
  - Checked and fine: the floor. `below_floor` means the crawl would not have selected the species (catalog
    under 50), not fewer than 50 held; none of the 18,066 species with added rows has below-floor rows.
- 2026-11-12: final plan (`freeze_build/plan_final`, after the three fixes) checked and applied.
  - Checks on `metadata_final.parquet`: all 190,199 usable species paths have train rows; 0 rows of width 0;
    0 empty hashes (1,668,381 null: no hash known); 0 usable one-word names at species rank; usable genus-rank
    rows 2,600,088 -> 2,642,775 (+42,687).
  - Against dry run 2: rows 65,602,738 -> 65,713,958 (+111,220); usable 63,454,710 -> 63,564,904 (+110,194;
    Plantae train +95,392, every other kingdom under +2.2 k); held-out taxa +999, pixel duplicates +27, all
    other exclusions unchanged (below_floor 76,473, benchmark_overlap 5,834); 17,963 file moves; 148 species
    moved to train (was 157, now keyed by full path).
  - Applied to `/data/datasets/tol256` (owner approval of 2026-11-11, delta shown first): 17,963 files linked
    at new paths, 111,220 re-fetched files linked in, `metadata.parquet` swapped atomically (previous copy
    hard-linked at `freeze_build/plan_final/metadata_before.parquet`), 17,963 old links removed, 139 emptied
    folders removed. Log: `freeze_build/apply.log`, `plan_final/applied.tsv`.
  - `dev/102 check` on the shared copy: 65,713,958 rows, 0 paths missing, 0 duplicate paths, 0 extra files;
    201,694 species; tree as in the plan. New: 64 lowercase one-word folders (88 rows). They are retry4 rows
    whose catalog key held only the epithet (*morhua*, *natrix*, *vulgata*); my one-word rule in `add` read
    the epithet as a genus-level name, so they sit at genus rank with the right genus and empty species.
    Not in training (species rank only); left as a known defect in the README rather than a second swap.
  - Version **tol256-v1**: `metadata.parquet` sha256 `cea244d4...c5021e8d`; README rewritten (split, every
    exclusion, `train/`, history); `train/` (hard links of `freeze_build/train`) and `heldout_taxa.csv`
    added beside it.
  - `dev/103 train`: 60,914,607 rows in 190,199 species (set 2: 54,818,719, 1: 3,050,136, 0: 3,045,752), 0
    species without train rows; classes 190,199 / 40,797 / 5,039 / 1,144 / 313 / 64 / 7 (species to
    kingdom); `none:` classes for missing ranks: 77 families, 751 orders, 202 classes, 5 phyla.
  - Pilot (50/5/5 per species and set): 9,337,505 images, about 70 GB at the smoke set's 7.6 KB a sample;
    `dev/103 shards --pilot` into `tol256s/shards/tol256-pilot`, `dev/104 --free --watch` to
    `/12383016/tol256-pilot`.
  - My mistake, caught before any upload: the pilot picked each species' 50 rows by the same uuid hash that
    orders the stream, so large species kept only low-hash rows and the stream ran from head species to tail
    species (first of 64 buckets 856 k rows in 131 k species, last 28 k rows in 21.5 k species; even buckets
    would hold 126 k). Each shard would have held a different species mix. The pilot now picks by a second,
    independent hash (`pilot<seed>`); the full set selects nothing and was never affected. Local pilot shards
    deleted and rebuilt.
  - UCloud's files API returned 502 for every drive from 15:44 (login, drive list and the job queue worked);
    nothing was created on `/12383016`. The pilot upload now waits for the files API and resumes from its
    ledger (`freeze_build/upload_pilot.sh`).
- 2026-11-13: my mistake: the upload script's outage probe called `ucloud` by name, which is not on the
  systemd service's PATH, so it reported "files API down" all night. The outage was real at 15:44, but I
  cannot tell when it ended; at 09:15 the API answered. Fixed with the absolute path; upload started 09:20.
  - Pilot rebuilt with the independent pick hash: 64 buckets of 125,734-127,108 rows, first and last
    each about 90.5 k species. 9,337,505 samples in 262 shards (226 train, 18 val, 18 test), 260.5 GiB:
    29 KiB a sample, not the 7.6 KiB I took from the smoke set (my 70 GB estimate was wrong by 3.7x).
  - Upload rate 108 MiB/s (first batch: 16 shards, 16.1 GiB in 152 s). Pilot about 41 min; full set
    about 1.66 TiB, about 4.5 h of upload. Shard build about 13 k samples/s (full set about 80 min), so the
    upload sets the pace. Full build started; its upload (`freeze_build/upload_full.sh`) starts after the
    pilot's, so the two do not share bandwidth.
  - Pilot uploaded 09:20-10:03: 262 shards and `manifest.json` at `/12383016/tol256-pilot`, 261.5 GiB at
    105.9 MiB/s; every remote `.tar` has the manifest's byte size (sha256: mini_trainer `verify --full` where it is
    read). Paths, format, branch and the levels/num_classes caveat sent to mini-trainer-f4. Full upload to
    `/12383016/tol256` started 10:03.
  - mini-trainer-f4 confirmed the levels caveat at c4731a6: with `levels` < 7, `parse_class_spec` keeps the
    7-entry `num_classes`, so the loss gets 7 levels against a truncated head; expected to crash at the first
    step, not to mistrain silently. A 7-level run is unaffected. Fix (slice `num_classes` to `levels`, with a
    test) deferred until a run with fewer levels is planned.
- 2026-11-13 10:30: the full upload's websocket dropped mid-batch after 144 verified shards. UCloud kept the
  in-flight files as partials (e.g. `train/000144.tar`, 543 MB of 1.07 GB); the retry, with overwrite off, landed
  the full files beside them as `000144(1).tar`, and the size check stopped the run (8 such pairs). My
  oversight: `dev/104` assumed an interrupted upload leaves nothing behind. The shard build then sat at its
  200 GB floor until 2026-11-15, with `/data` at 98 %.
- 2026-11-15: the owner approved deleting the partials. The 8 partials were trashed (drive Trash), the 8
  full `(1)` copies renamed to their names (sizes equal to local), no `(1)` left. `dev/104` now overwrites a
  remote file only when it is not in the ledger and smaller than the local one (its own interrupted
  upload); a larger remote file stops the run. Upload resumed 12:17 at 105 MiB/s.
  - Full shard build done: 60,914,607 samples in 1,832 shards (1,648 train, 92 val, 92 test), 1.79 TiB.
  - 12:37 the uploader was OOM-killed at its 3 GB cgroup cap after 272 shards (cause not reconstructed; at
    6 GB its anon memory is 116 MB, the rest reclaimable page cache from reading the tars, no OOM events).
    Restarted at 6 GB; the new rule overwrote the 8 partial files the kill left, and the batch verified.
    Remaining 1,512 shards, about 4 h.
  - Full set complete 17:29: 1,832 shards and `manifest.json` at `/12383016/tol256`; every remote `.tar` has
    the manifest's byte size, no renamed copies; 93.8 MiB/s over the run. Local `.tar` files all freed
    (`/data` back to 1.9 TB free). mini-trainer-f4 told; sha256 is its `verify --full`.
  - mini-trainer-f4: `verify --full` (sha256 of every shard) runs in a UCloud job where the shards are read;
    it will put that job to the owner as the first step of the ToL run, not before.
  - Full check passed where training reads it (mini-trainer-f4, owner-approved UCloud job 12462945, 1 MIG
    slice, 6 vCPU, `/12383016/tol256` mounted read-only): per-shard sample counts, every sample's tar header
    and CRC, every tar's SHA-256 against `manifest.json`: "60,914,607 samples OK (all 1832 shards)", exit 0,
    1.788 TiB in 6.1 min. The 8 replaced files pass. Log:
    `/12347837/repos/mini_trainer-campaign/tmp-runs/tol256_verify/verify.log`. TF closed.
