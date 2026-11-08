# The training labels inherit errors from the legacy GBIF Backbone

**Kind:** research · **Status:** OPEN (2026-11-08). Diagnosed; the correction waits until the paper is
submitted (owner, 2026-11-04 and 2026-11-08).

An investigation run from the mini_trainer checkout (2026-11-03 and 04) mapped every Nemo class to
GBIF's new taxonomy, the Catalogue of Life eXtended Release (COL XR), and checked each class against
the names behind its training images. lepinet uses the same table
(`..._quality_filtered.parquet`, 6,329,994 images), so this applies to P5, Otto and every model
trained on it. Its files are git-ignored and exist only on this machine:
`~/codes/mini_trainer/.agents/local/nemo_species_colxr/` (`JOURNAL.md`,
`nemo_species_gbif_colxr.csv`, `nemo_training_names.csv`).

## The cause

The images are labelled with the legacy Backbone's accepted `speciesKey`, not with the name each record
was identified under (`taxonKey`). So the Backbone's synonymy errors became class definitions. GBIF
has since frozen that Backbone (last built 2023) and moved to COL XR.

## What is wrong

| problem | extent | examples |
|---|---|---|
| **Mixed classes:** a class holds images named as another species | 163 classes; 23,306 images (0.4 %); 39 classes at 20 % or more; 53 classes in a Europe list | *Loweia tityrus* holds 1,836 *Brintesia circe* (homonym, about half the class); *Eudonia truncicolella* 1,018 *E. pallida*; *Idaea seriata* 980 *I. minuscularia*; *Pyrgus oileus* 1,941 *Burnsius orcus* |
| **Duplicated classes:** several classes are one species in COL XR | 281 classes collapse onto 135 species; 186,168 images. Includes genuine lumps | *Anarta trifolii* / *Hadula trifolii*; *Agrochola helvola* / *Anchoscelis helvola* |
| **Wrong display names** | 2 classes in the Europe list, more elsewhere | the second "*Leptidea sinapis*" holds only *L. juvernica*; the second "*Euphydryas aurinia*" only *E. beckeri* |
| **Wrong family or genus above a correct class** | about 1,900 rows found on 2026-11-08, likely a lower bound | Lycaena *hyllus*, *helloides*, *epixanthe*, *rubidus* and *Biblis hyperia* sit under *Sesia* (Sesiidae) |
| **Species absent because of the image filter** | 34 EU27 Red List species with 50 or more photos before the filter | the adult-only filter removes species photographed mostly as larvae |

- **Trap set:** 19 of the 486 species, 1,876 of the 47,905 crops: 364 (0.8 %) in mixed classes, 1,512
  (3.2 %) in duplicated ones.
- **Mapping status:** 12,578 of the 12,632 classes (99.6 %) have a COL XR species. 54 do not; 3 of those
  are in a Europe list.
- **COL XR has errors too:** 48 of GBIF's crosswalk results were overridden by hand after checking
  them against the images (*Euphydryas aurinia* to *E. beckeri*, for instance), and 67 classes meet
  duplicate COL XR records for one species. 15 hand decisions are low confidence, and the 375
  genus-only renames were not audited individually.
- **A limit of the evidence:** a mixed class says what name a record carried, not that the photo is
  misidentified. For cryptic pairs the photos may not separate the species.

## What it does to our numbers

Train and test share these labels, so comparisons between models stand. The absolute scores of the
affected species do not:
- two classes for one species cannot both be predicted, which lowers macro-F1 on those classes;
- a mixed class teaches the wrong species and is then scored against the same wrong label.

The size and sign of the net effect on macro-F1 are not measured. The caveat is carried in `PLAN.md`
section 5.

Label joins inherit it too. Resolving outside names through our table's synonym usages (AMI-Traps,
iNat21) gives the label the models trained on, which is correct for scoring them today, but it sends
*Brintesia circe* to the *Loweia tityrus* class.

## The correction, after submission

1. Relabel every image from `taxonKey` to its COL XR species, starting from the two CSVs above. Keep
   the 48 overrides and review the 15 low-confidence decisions.
2. Decide the rule for merging duplicates from key-based evidence, and record the taxonomy source and
   version with every resolution (mini_trainer's roadmap: a versioned change to
   `mini_trainer.integrations.gbif`). COL XR is re-released monthly, so pin one release.
3. Re-map the trap labels (19 species) and rebuild the regional lists.
4. Add the check that the hierarchy is a tree, and flag cross-family placements.
5. Measure the effect before retraining everything: re-score existing models on the corrected labels,
   then retrain the production models (Otto's successor first). Which others to retrain is decided
   from that measurement.
6. Re-run the public benchmarks with the new labels.
7. Any new training at Tree-of-Life scale starts from the corrected taxonomy.

## The one command (2026-11-08)

`.venv/bin/python dev/101_colxr_relabel.py apply --upload`, offline, from tables frozen in `dev/colxr/`
(COL XR as served on 2026-11-08). It writes new files to `data/global/colxr/` and to a new UCloud folder, and
changes nothing that exists. The images stay in their folders: `legacySpeciesKey` names the folder, and
`ucloud/stage.py` copies to `<new speciesKey>/<filename>` on the node. To change a decision, add a row to
`dev/colxr/overrides.csv`. Afterwards a training spec only needs to point at the new parquet.

Decisions:
- A class goes to its COL XR species. An image goes elsewhere only if its own name is another species
  (23,306 rows, the mixed classes), not for homonyms that only carry the class's name, and not for duplicate
  COL XR records of the class's species.
- Classes sharing a COL XR species are merged. COL XR's duplicate records are not merged automatically:
  no rule tried was safe (epithet and author match unrelated species; *Heliconius erato* and *Tirumala erato*
  are linked by a shared record). The confirmed one, the Nutmeg, is in the overrides.
- The 54 unresolved classes and the 15 low-confidence decisions stay as they are, keyed `gbif:<legacy>`.
- Every row is kept, test fold included. The training floor stays with training.
- 140 species have no chain from COL XR under their key (usageKey lookup returns nothing, or a homonym in
  another order): parents come from a genus of the same name, or the legacy genus or family.
  `parents_from` in `taxonomy.csv` records which. 6 trap species outside the training table go through
  GBIF's crosswalk.

Trial on the real data (2026-11-08, 26 s, 1.3 GB peak), output checked against the source: 6,329,994 rows, order,
`filename` and `set` identical; no null keys; species to genus and genus to family are trees; 12,632 classes
become about 12,635 (139 merged away, 143 new from mixed classes). *Brintesia circe*'s 1,836 images get
their own key while staying in folder 1924494. 686 new species fall under the 50-image floor. Trap
species 486 to 484; `north_europe` 1,977 to 1,988. mini_trainer's side (string keys, staging of set 0) is
on the branch.

## Log

- 2026-11-03 and 04: investigation (mapping, manual review of 325 classes, audit of GBIF's
  crosswalk, check against the training names). Decision: correct after the paper.
- 2026-11-08: mt's iNat21 scoring flagged family errors; written up here at the owner's request.
