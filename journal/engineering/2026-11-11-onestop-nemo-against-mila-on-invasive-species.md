# OneStop: Nemo against MILA on invasive alien insects

**Kind:** subproject · **Status:** RESOLVED (2026-11-11): Nemo 0.52 against MILA 0.34 species macro
accuracy over the 64 listed species with images; calibrated macro-F1 0.23 against 0.11. Otto (OT3), scored
for us only, 0.55 and 0.29.

A colleague's request for the OneStop project (invasive alien species monitoring): Nemo (MAMBO_v3) next to
the Rolnick lab's "UK & Denmark Species Classifier - Apr 2024" (RolnickLab/ami-data-companion,
`UKDenmarkMothSpeciesClassifier2024`: timm ResNet-50, 128 px, 2,603 species), per OneStop Living Lab species
list (`data/onestop/onestop_species_list.csv`; Belgium, Portugal, UK, Finland; 85 insect rows, 67 taxa).
The intended message (owner): a model trained on a better species list is better. Figure:
`data/onestop/onestop_per_list.png`. Code: `dev/105_onestop_plan.py`, `dev/106_onestop_eval.py`.

## Test set

- GBIF images with an event date after 2025-04-26 (Nemo's GBIF download date; MILA's model is from April
  2024), any country (owner), up to 2,000 per species spread over months in proportion to records. As a
  second guard, 93 occurrences already in Global Lepidoptera were dropped despite the date.
- Queried through **COL XR**, GBIF's current backbone, not the legacy one: the legacy key of Oegoconia
  novimundi (1845962) lumps four Oegoconia species, among them the native O. quadripuncta (388 recent
  records against 171). The owner caught it from GBIF's taxon pages. Other list names differ only by
  renaming (Tuta absoluta = Phthorimaea absoluta, and so on).
- Fetched with dev/082's fetch stage on 4 cores. observation.org and artsobservasjoner.no answer 403,
  image.laji.fi times out and moth.tbn.org.tw has a bad certificate; their images were skipped (owner:
  discard what hosts block), and a Flickr stall left ~570 out. 43,771 images, 29,006 occurrences, 64 taxa.
- Not filtered by life stage: larvae, leaf mines and nests are in (Cameraria ohridella, Phyllonorycter
  platani, Lymantria dispar score low for every model partly for that reason).

## Scoring (mini_metrics, owner)

- Pinned mini_metrics 70cc69a, canonical mini_metric rows at species level, as Asger's release charts.
- **macro-F1 at threshold 0 is useless here**: 0.013 (Nemo) and 0.014 (MILA). Over truth ∪ predicted classes,
  800-1,800 predicted-absent species count 0 against 8-47 listed ones; about 8,600 photos of non-moths
  (Harmonia, Halyomorpha, Vespa, Agrilus, Orientus) are spread over moth classes. Its occurrence bootstrap
  was also biased (resamples lose rare predicted classes), so it was dropped.
- Reported instead: **macro accuracy** at threshold 0 (stratified occurrence bootstrap, so the species set
  is fixed; the unstratified one dropped 1-3-image species and biased up) and **calibrated macro-F1**
  (OptimalConfidenceThreshold(MacroF1, eps 0.01, quantiles) on half of the occurrences, stratified by
  species, scored on the other half).

| list | Nemo acc | MILA acc | Otto acc | Nemo F1 cal | MILA F1 cal | Otto F1 cal |
|---|---|---|---|---|---|---|
| Belgium (14) | 0.62 | 0.40 | 0.66 | 0.24 | 0.08 | 0.41 |
| Portugal (47) | 0.53 | 0.34 | 0.56 | 0.34 | 0.14 | 0.32 |
| UK (8) | 0.74 | 0.46 | 0.81 | 0.38 | 0.12 | 0.52 |
| Finland (12) | 0.49 | 0.38 | 0.53 | 0.15 | 0.07 | 0.24 |
| all (64) | 0.52 | 0.34 | 0.55 | 0.23 | 0.11 | 0.29 |

## What it says

- **The species list is not most of the gap.** Nemo names 40 of the 64 species, MILA 38; on the 32 both
  name, mean recall is 0.82 against 0.65 and Nemo is better on 30. Species only Nemo names include Clepsis
  peritana, Spodoptera frugiperda, Isturgia deerraria; only MILA: Prays citri, Grapholita molesta,
  Blastobasis maroccanella.
- **MILA never predicts Epiphyas postvittana** (once in 43,771 images; 0 of 2,000 correct, Nemo 0.92)
  although the class is in its map. Either that class had almost no training images or the map does not
  match the weights; worth asking the Rolnick lab before the figure is shown outside.
- **Otto (OT3) beats Nemo on photos too** (+3 macro accuracy, +6 calibrated F1), although its gain is on trap
  crops; not in the OneStop figure (owner).
