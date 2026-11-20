# Can one frozen trunk with specialisation heads stand in for specialist models?

**Kind:** research · **Status:** OPEN (2026-11-20). The question behind paper B's new story (owner, 2026-11-20); claims,
evidence so far, the work still missing, and predictions written before any of it runs.

## The question, in the owner's words

"One model to rule them all. A very good frozen trunk combined with a set of very good specialization techniques to
bridge the gap between foundation model/embedding models and real life application, which are domain-shifted,
label-shifted and full of unknown objects." Paper B's deadline moved to 2027, so the story comes first.

The specialist is the reference: Otto (OT3/OT4) on moths, and published or in-house specialists elsewhere. The
generalist is a ToL trunk, frozen, plus heads and test-time tools. Fine-tuning closes gaps but erases the trunk's
generality (Z21), so the paper measures how far the frozen route goes and what is left.

## The claims, and where each stands

| | claim | evidence now | missing |
|---|---|---|---|
| C1 | mini_trainer trains a foundation model at a fraction of BioCLIP-2's cost | tol-full: about 2 % of BioCLIP-2's training compute (106 B200-hours, 0.30 B images seen against 6.4 B), one attempt, 97 % GPU utilisation ([[2026-11-19-what-the-tree-of-life-model-is-for]] section 3) | TL10's cost, same accounting |
| C2 | on BioCLIP-2's own benchmarks, the frozen ToL trunk is in BioCLIP-2's range or above | Rare Species only: 1- / 5-shot 62.9 / 79.1 against 60.4 / 77.0 (Z19, BioCLIP-2's protocol reproduced first) | the other nine species tasks and the five non-species tasks ([[2026-11-08-does-a-margin-objective-make-a-better-pretrained-biodiversity-model]] lists them with BioCLIP-2's values); an image-overlap audit per benchmark |
| C3 | on AMI traps, one trunk with two heads (order, then moth species for crops predicted Lepidoptera) is in the range of two specialists (the pipeline's order classifier, Otto) | moth species: TL8b 0.696 calibrated with the list against Otto's 0.749 (Z23); test fold 0.895 against 0.916 | the trap order dataset (owner, being transferred, PLAN), the specialist order classifier's score on it, the order head |
| C4 | the same holds outside moths: camera-trap animals and plants | none | iWildCam (WILDS: its fixed out-of-distribution split and macro-F1 make the comparison with published specialists direct); Pl@ntNet-300K (on disk, 56 GB, GBIF-mapped; Asger's specialists from paper A's head grid as the reference) |
| C5 | the gap that remains is class bias and confidence, which specialisation tools address, not discrimination | Z23: TL8a's pairwise error under every fine-tuned model's; TL8b ranks its answers best of all; 4b: most of self-training's gain is a fauna prior the list and EM give without training | the same decomposition on C3 and C4 |

**Overlap is the trap in C2 and C4.** ToL's training rows come from GBIF. Pl@ntNet publishes its observations to GBIF,
and several benchmarks draw on iNaturalist or EOL images, so test images may be training images. BioCLIP-2 removed
its benchmarks' images from training; we have not. Each benchmark needs an audit before its number is a claim: GBIF
occurrence ids where the benchmark has them, otherwise perceptual hashes on a sample. Where the overlap is real, the
claim is made on the clean part.

## Work, in order

1. **TL10 (running, job 12471243, started 2026-11-20 09:51).** The trunk every claim is finally read on; tol-full is
   used meanwhile, and each frozen evaluation below is rerun on v2 when it lands.
2. **C2, BioCLIP-2's suite.** Download, reproduce BioCLIP-2's published values with stock BioCLIP-2 first (as for
   Rare Species), then score tol_full: 1- and 5-shot (SimpleShot) on the ten species tasks and the non-species tasks
   with their protocols, and zero-shot. Zero-shot is not BioCLIP-2's privilege: BioCLIP-2 is trained on a closed set
   of taxa and its text encoder encodes their hierarchical labels, so its zero-shot on taxa it trained on (Rare
   Species: 76.8, above its own 1-shot 59.5) is classification by label prototypes. Ours is the same with the head's
   rows (coverage per task reported), and coarser-rank marginals for taxa outside the vocabulary. The framing for the
   paper: both are embedding models trained in a closed-set fashion, as ArcFace is for faces; the one difference
   worth stating is that a text encoder can make a prototype for a name it never trained on. Downloads: the owner
   asked for the space check (2026-11-20); the Meta-Album licence is still open (2026-11-07 entry).
3. **C3, AMI order and species.** With the owner's order dataset: the specialist order classifier's accuracy; the
   ToL head's own order marginal (no training); an order head on the frozen trunk; then the pipeline, order then
   moth head on crops predicted Lepidoptera, scored end to end against the two specialists.
4. **C4, iWildCam then plants.** iWildCam: a head on the frozen trunk on WILDS's train split, scored on its
   out-of-distribution test split with its metric, then the tools (a prior estimated per camera location, the
   trap prior's analogue; TTA; open-set for empty or unknown frames). Pl@ntNet-300K: a head on the frozen trunk
   against Asger's specialists, after the overlap audit.

Each item is a head or an evaluation on a frozen trunk: a few B200-hours each, no trunk training.

## Predictions (supervisor, 2026-11-20, before any result)

- **C2.** Mean 5-shot over the ten species tasks within 3 points of BioCLIP-2's 78.3, either side; above on Insects,
  Insects 2 and Rare Species; below on Plankton, PlantVillage and Medicinal Leaf (microscopy and lab leaf images, far
  from GBIF photographs). Non-species mean 2 to 8 points below BioCLIP-2's 57.5 (TL7: ToL's coarse structure is
  flatter, and attributes such as habitat favour a model trained with text).
- **C3.** Order accuracy from the ToL head's order marginal within 3 points of the specialist order classifier; an
  order head on the frozen trunk within 1. Species: the TL8b gap (5 points under Otto on the trap probe) shrinks with
  TL10 but does not close. So "the same range" holds for order now, and for species only if TL10 reaches about 0.73.
- **C4.** iWildCam: frozen trunk and a linear head above WILDS's ERM baseline and below the best fine-tuned entries,
  with the location prior adding the most of the tools (label shift dominates camera traps). Pl@ntNet-300K: within 3
  points of Asger's specialists on his metric, if the overlap audit leaves a clean test set.
- **The weakest link** is C3's species half: if TL10 leaves TL8b-v2 more than 3 points under Otto on traps, the paper
  says "a frozen generalist reaches the specialists' range on order, on benchmarks and on other domains, and on
  moth species closes all but N points, with the tools", which is still the story, told honestly.

## Log

- 2026-11-20: opened (owner's message: new story first; C1 to C4 stated by the owner; C5 from Z23). TL10 already
  running. Nothing downloaded or submitted for C2 to C4.
