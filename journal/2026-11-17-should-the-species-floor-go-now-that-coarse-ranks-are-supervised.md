# Should the species floor go, now that the coarse ranks are supervised?

**Kind:** research · **Status:** OPEN (2026-11-17). Counts done for Global Lepidoptera and ToL; no decision yet.

## The question (owner, 2026-11-17)

Both datasets drop every species with fewer than 50 images. With a hierarchical head, a species under the floor
whose genus (or family) has many images could still be trained: its images would at least supervise genus and
family. How many species and images would that add, for a genus rule and for a family rule? And with no floor at
all, how few images would the smallest families have?

## Global Lepidoptera (`dev/110_species_floor.py`)

From the GBIF download before the quality filter (7,492,060 images, 50,948 species, 127 families, legacy keys).
Trained today: 12,632 species, 6,329,994 images (>= 50 images after the quality filter, which keeps 89.8 % of a
trained species' rows; its exact rules are not documented in this repo).

| rule | species added | images added (after a quality filter at 89.8 %) | new genera / families | median images per added species |
|---|---|---|---|---|
| species under the floor whose genus has >= 50 images | +29,160 | +372,520 (~334 k) | 665 / 5 | 6 (5,754 single-image species) |
| same, family >= 50 | +38,279 | +441,753 (~397 k) | 5,921 / 8 | 5 (8,177 single-image) |
| no floor | +38,316 | +441,994 (~397 k) | | |

With no floor: 127 families, images per family min 1, 1st percentile 1, median 3,213; 15 families under 50 (today
all 104 trained families have at least 50). 5,280 of 10,421 genera have under 50 images.

So lifting the floor triples the species (12.6 k to 41-51 k) and adds about 6 % images; most added species have one
to six images.

## ToL (ToL session, `dev/111_tol_species_floor.py`)

Full-path keys; held-out taxa excluded throughout, including under-floor species inside a held-out genus or family.

**On disk (tol256-v1):** 9,662 under-floor species (76,244 rows) outside held-out taxa.

| rule | species added | images added |
|---|---|---|
| genus has >= 50 usable images | 4,442 | 43,849 |
| family has >= 50 usable images | 7,712 | 66,022 |

4,797 under-floor species have an untrained genus, 1,658 an untrained family. The family-rule rows were never
screened by the freeze for every exclusion: 34 benchmark overlaps, 50 duplicates, 1,671 with no content hash, and
1,993 Lepidoptera rows that need the lepinet test-fold check first.

**In the TreeOfLife-200M catalog** (not downloaded): 655,084 species under 50 catalog rows, not trained, not held
out (6.32 M rows).

| rule | species | candidate rows | already on disk |
|---|---|---|---|
| genus >= 50 usable on disk | 431,026 | 4,756,070 | 40,678 (0.9 %) |
| family >= 50 usable on disk | 598,779 | 6,026,357 | 61,897 (1.0 %) |

**No floor, images per family** (species capped at 2,000; held-out families out):

| set | families | min | median | under 50 |
|---|---|---|---|---|
| disk, species-rank rows | 5,558 | 1 | 453 | 975 (17.5 %) |
| disk, plus genus- and family-rank records | 5,897 | 1 | 436 | 1,059 (18.0 %) |
| catalog, species-rank rows | 11,849 | 1 | 62 | 5,604 (47.3 %) |
| catalog, plus genus- and family-rank rows | 21,495 | 1 | 17 | 13,821 (64.3 %) |

On disk, lifting the floor adds only 66 k images (+0.1 %), because the crawl fetched the floor's species only. The
catalog holds 6.0 M candidate rows in 599 k species, 99 % not downloaded: a re-crawl of about 10 images per species,
which not every host would serve.

## What this changes, and what must be decided first

1. **Species classes or coarse labels only.** Added as species classes, the under-floor species turn species
   macro-F1 over 12 k species into macro-F1 over 41-51 k, most with one to six images, which changes the headline
   benchmark and every number in both papers. Trained with genus and family labels only, they strengthen the coarse
   ranks (and, through the marginals, their sibling species) without new classes; that needs a loss that ignores the
   species rank for those images, which neither lepinet nor mini_trainer has yet (W3 noted the same gap for ToL's
   2.6 M genus-rank records).
2. **The C3 open-set benchmark is made of these species.** C3 (paper finding 8, near / mid / far) scores against the
   unfiltered catalogue: every species under the floor is its "unseen" set (38,907 species,
   [[2026-08-08-is-novelty-monotone-or-just-rare]]). Training on them removes that benchmark; C3b (withheld common
   taxa, retrained) and ToL's held-out taxa (TL2) do not depend on the floor.
3. **Images.** The Lepidoptera download marks all 7.49 M as uploaded, but the local image folder holds the 12,632
   trained species only; where the under-floor images are has to be checked. For ToL, 99 % would be a new crawl.
4. **Keys.** The Lepidoptera counts use legacy GBIF keys; COL XR moves some genera and families.

## Recommendation (not a decision)

The gain is breadth of the label space far more than images (+6 % for Lepidoptera, +0.1 % on ToL's disk). The cheap
and clean version is **coarse-only supervision**: under-floor images with genus and family labels, no new species
classes, so every benchmark stays as it is (except C3, to be replaced by C3b and TL2). It needs one code change, a
per-rank loss mask, and one run to measure it on the coarse ranks, on traps (unseen trap species fall back to genus)
and on held-out placement. Adding the species as classes is a separate, larger decision about what the papers'
species metric means.
