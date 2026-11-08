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

Not yet analysed: the 12,151 capitalised single-word folders (genus-only records, or shifted the same way,
e.g. `Sciaridae`).

### 2. The Lepidoptera labels

Legacy GBIF Backbone errors: [[2026-11-03-the-training-labels-inherit-gbif-backbone-errors]].

## Log

- 2026-11-08: opened (owner). Rank shift found while checking which Rare Species taxa are in the
  catalog.
