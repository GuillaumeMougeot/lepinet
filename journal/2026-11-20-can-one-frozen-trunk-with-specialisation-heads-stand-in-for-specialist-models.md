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

## The specialisation candidates (supervisor, 2026-11-20)

Everything the project has measured that adapts a model to a deployment, grouped by what it needs from the target.
"Fixes" says which part of the gap it addresses: the label space, bias (which classes the target holds, label
shift), confidence, discrimination, or domain (how the target's images look). Numbers are the project's own, with
their IDs; the last column says where each stands on the AMI broad-taxon set (C3, BT1) and is the to-do list for C3
and C4.

| | candidate | needs from the target | fixes | measured | on C3 (MCC24) |
|---|---|---|---|---|---|
| 1 | map the head to the target's labels by taxonomy (marginals; mass off the target's taxa set aside) | its label names | label space | Rare Species 70.2 against BioCLIP-2's text 76.8 (BS3) | done: V0-V2, 0.765 against Kim's 0.970 (BT1) |
| 2 | target list: restrict the answer to the taxa expected there (a regional checklist) | a list | bias | the largest single prior on traps: calibrated probe V2-S 0.372 to 0.553 (Z9); +5.1 accuracy, -1.7 macro-F1 when the list is most of the scored set (K1) | built into 1 (16 categories) |
| 3 | test-time augmentation, three views with border-mean fill (tta3bm) | nothing | domain | calibrated probe +5 to +11 (VY); halves the small-crop deficit (Z22) | **not tried** |
| 4 | read the crop as it is: pad to square instead of cutting the centre | nothing | domain | not measured | **not tried** (Kim's crops are elongated; the square read cuts some) |
| 5 | open-set score chosen for the model's scale, rank abstention | a precision target | unknowns | the best rule changes with scale, 6-8 points (E2, finding 6); trap out-of-vocabulary AUROC 0.84-0.86 (N3); expected-utility back-off (Q1) | V2 only (Vegetation as 'everything else') |
| 6 | prior estimated by EM (Saerens et al. 2002) on unlabelled target images, applied as p(y\|x) x prior(y) | unlabelled images | bias | with the list, a never-trap model comes within 1.5 / 0.6 points of its self-trained twin (finding 4b, Z14); at a new site it equals self-training on two other sites (Z15); it **harms** models trained on trap labels (Z21, Z23), so it is for never-trap models such as ToL's head | **not tried**: next, as rung L0 of the C5 ladder, on all crops and per site |
| 7 | self-training on gated pseudo-labels (OT3's recipe) | unlabelled images, GPU | bias, domain | +7.94 probe (finding 4); on the frozen trunk +4.9 calibrated (TL8b against TL8a, Z23); 73-78 % of its repairs are what list and EM give without training (Z13, Z16) | **not tried** (a CPU version: pseudo-label a head on the frozen embedding) |
| 8 | temperature on the target | a few labels or a held-out split | confidence | a margin head needs one before any threshold (finding 9) | **not tried** |
| 9 | nearest class mean on the frozen embedding (SimpleShot) | a few labelled images per class | discrimination | 10 images per species: 84.6 % test-fold accuracy against 89.3 for a trained head (Z20); Rare Species 79.1 at 5 shots (Z19) | planned, BT1 (b) |
| 10 | the C5 ladder on the head's own outputs: L1 class biases, L2 a linear map of the category masses, L3 a linear model on the family posterior | labelled images | bias, then discrimination | new | planned (rungs fixed before the scores, 12:10 above) |
| 11 | linear head on the frozen embedding | labelled images | discrimination | new | planned, BT1 (a) |
| 12 | head trained through the frozen trunk with augmentation (TL8's recipe) | labelled images, GPU | discrimination, domain | test fold 0.895 against Otto's 0.919; probe 0.647 (TL8a) | **not tried** |
| 13 | class-balanced classifier retraining (cRT), frozen trunk | the training labels | long-tail bias | +1.1 on the test fold with a frozen trunk (VK) | not relevant at 16 classes |
| 14 | a small dose of real target labels | labelled target images | bias | +1.0 probe, and the gain is lost on unseen species (finding 4) | the trained heads above are this, at full dose |

Two more change the trunk, so they are the generalist's training recipe rather than specialisation: resolution
augmentation (MixRes, MR; Z22 puts never-trap models 13-18 accuracy points lower on their smallest crops than on their
largest; Z24, 2026-11-25: shrinking the largest crops to the smallest's size costs the frozen ToL head nothing, so the deficit is blur and framing, not resolution, and MixRes is ruled out) and domain augmentation (AG). Fine-tuning the trunk (TL4) is the specialist reference, not a candidate:
it erases the trunk's generality (Z21).

**On Kim's split.** Rungs 8-12 are scored twice: on his list, to compare with his 0.970, and site held out, where only
a specialist retrained on the same folds is a fair reference; candidates 1-6 use no labels. *Corrected 2026-11-21:* this
paragraph first called his list "close to a resubstitution test" because 62 % of validation crops share a source frame
with a training crop of the same class. That was wrong: a frame holds about 35 insects, so sharing one does not mean the
same insect. In embedding space only 10.3 % of validation crops have a training crop at cosine 0.95 or more (0.9 % at
0.98). What the random split hides is site shift (a new site costs 7-9 points of macro F1), not leakage (a trap-night
split within the sites costs about 1); section C3 below.

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

## C3: AMI broad taxa (MCC24)

The order half of C3, on Kim Bjerge's published broad-taxon dataset rather than the owner's (still in transfer).
Broad-taxa agent, 2026-11-20: CPU (4 cores) and, from 13:58, two B200 MIG slices. Scripts `dev/115_broad_taxa.py`
(CPU) and `dev/117_broad_taxa_gpu.py` (GPU job); outputs `data/broad_taxa/`; RESULTS BT1-BT5.

### The data, Kim's split and his metric (established before any model ran)

- **Data.** Zenodo 15105293 (CC-BY-4.0), `broadTaxonClassifier.zip`, md5 `aed0d68c...` as published, in
  `/data/au761367/benchmarks/mcc24`. 125,135 crops in 16 folders (plus 3 `Thumbs.db`), from 12 AMI traps at three
  Danish sites (LV, OH, SS), 2022-2023, 10-minute time-lapse. Non-Lepidoptera files are named by trap, source frame
  and crop number; Lepidoptera files by a species name, trap, day of year and time (the names agree with the folder at
  family level in 98 % of the 45,181 files whose genus ToL knows, so they look like a species classifier's output,
  not labels).
- **Sources.** The bioRxiv v2 text (Wiley's page returned 403; the published version is rse2.70007) and the
  repository `kimbjerge/MCC24-trap` (`orderClassifier.py`, `resnet50.py`, `model_order_100524/`), copied to
  `/data/au761367/benchmarks/mcc24/kim_repo`.
- **Model.** ResNet50 (ImageNet weights, a 512-unit hidden layer, dropout 0.5), fine-tuned; crops squashed to
  128 x 128 without keeping aspect; Adam 1e-4, batch 256; anomaly rule: top logit below its class's mean minus 2.5
  standard deviations.
- **Split.** "The dataset was split in 80 % for training and 20 % for validation" (Table 2 lists 20 % per class); no
  test set, no split lists, no mention of trap, night or time. The split is recoverable anyway: the repository has
  his model's logits on all 125,135 crops (`predictLabelsTestTrain.pkl`, in ImageFolder order: checked by running
  his weights on two whole folders, 100 % nearest-row agreement) and on the 25,034 validation crops
  (`predictLabelsTest.pkl`); each validation row equals one full-set row exactly. So **Kim's validation list is
  known file by file** (`index.csv`, column `split`). Per class it holds exactly ceil(0.2 n): a random per-class
  split, by crop. His checkpoint is `dhc_best`, so the validation set also picked the checkpoint.
- **What the split means.** 86 % of the validation crops that carry a frame time stamp have a same-class training
  crop from the same trap within one time-lapse frame (11 min), and 62 % share their very source frame with a
  same-class training crop. 83 % of validation Lepidoptera have a training file with the same species name, trap
  and night. 24 files form 12 byte-identical pairs, 3 of them split across training and validation. I first read
  this as "close to a resubstitution test" (insects sit still on the screen for several frames). **That was a
  genuine mistake**, corrected at 19:30 by the measurement: a source frame holds about 35 insects, so sharing a
  frame or a night does not mean sharing an individual. The embedding measures it directly (Results, leakage): 10 %
  of validation crops have a training crop at cosine 0.95 or more, 0.9 % at 0.98.
- **Metric.** Table 4 reports precision 0.974, recall 0.965, F1 0.970 on 25,034 samples, without saying how they are
  averaged. His own logits give **macro** (unweighted over the 16 classes) P 0.9742, R 0.9652, F1 0.9695 (accuracy
  0.9839); with his anomaly thresholds 24,799 samples remain and P / R / F1 are 0.9772 / 0.9707 / 0.9738, his second
  column (0.977 / 0.971 / 0.974) exactly. So the metric to beat is **macro F1 0.970, accuracy 0.984**, on that list.

### The mapping (written before any score; taxonomy only)

The ToL head gives a softmax over 190,199 species; each category's mass is the sum over its species (equivalently
over its families: every rule below is at family level or above). The prediction is the category with the largest
mass among the 16; mass on taxa outside the 16 (Plecoptera, Odonata, molluscs...) is set aside, and its size is
reported. Orders and families are matched by name, so ToL's few mis-pathed nodes still map (Riodinidae appears under
Cnidaria; a fish family, Asterolepididae, under "Lepidoptera" is excluded).

| category | ToL taxa |
|---|---|
| Araneae, Coleoptera, Ephemeroptera, Hemiptera, Neuroptera, Opiliones, Trichoptera | the order of that name |
| DipteraTipulidae | family Tipulidae |
| DipteraTrichoceridae | family Trichoceridae |
| DipteraNematocera | the lower Diptera families of Pape, Blagoderov & Mostovski 2011 (Zootaxa 3148: 222-229) other than those two: Anisopodidae, Bibionidae, Bolitophilidae, Cecidomyiidae, Ceratopogonidae, Chaoboridae, Chironomidae, Culicidae, Cylindrotomidae, Ditomyiidae, Dixidae, Keroplatidae, Limoniidae, Mycetophilidae, Pachyneuridae, Pediciidae, Psychodidae, Ptychopteridae, Scatopsidae, Sciaridae, Simuliidae, Tanyderidae (those in ToL) |
| DipteraBrachycera | the other 81 Diptera families in ToL, each listed in the script (it fails on an unlisted family) |
| HymenopteraVespidae | Vespidae, with Eumenidae and Masaridae, which current classification treats as Vespidae subfamilies and ToL keeps apart |
| HymenopteraOther | every other Hymenoptera family |
| LepidopteraMacro | Hepialoidea, Cossoidea (with Sesiidae), Zygaenoidea (the families of the British and Danish macro-moth convention), the Macroheterocera (Mimallonoidea, Drepanoidea incl. Thyatiridae, Lasiocampoidea, Bombycoidea, Geometroidea, Noctuoidea) and Papilionoidea; family placement after van Nieukerken et al. 2011 (Zootaxa 3148: 212-221); 50 families |
| LepidopteraMicro | every other Lepidoptera family (69) |
| Vegetation | kingdom Plantae |

Kim gives no family list ("grouping species of Lepidoptera at the family level"). His file names, read before any
model ran, agree with this split for every family with more than 30 files except Cossidae (29 of 32 in Micro; a
handful of files either way). Pre-registered variants, reported beside the primary (V0):

- **V1, broad readings:** DipteraTipulidae = Tipuloidea (adds Limoniidae, Pediciidae, Cylindrotomidae: "craneflies"
  in the field sense); Neuroptera = Neuropterida (adds Megaloptera, Raphidioptera).
- **V2, Vegetation as open-set rejection:** Vegetation = everything not in the 15 arthropod categories (other orders,
  other animals, plants, fungi), instead of Plantae alone.

### Protocol

1. **Zero training:** V0 (and V1, V2) on all 125,135 crops (nothing was trained on them) and on Kim's validation list;
   macro P / R / F1 as Kim computes it, accuracy, per-category F1, the confusion matrix, the set-aside mass.
2. **Heads on the frozen trunk**, fit on Kim's training list only, scored once on his validation list: (a, primary)
   multinomial logistic regression on ToL's normalised embedding, C picked from {0.1, 1, 10, 100} on a random 10 % of
   the training list, then refit on all of it; (b) nearest class mean by cosine (no hyperparameter); (c) (a) on the
   trunk's features (before the head's hidden layer). No temperature or threshold is used.
3. **The split's effect:** 1-NN from the training list (how much the validation set is near-duplicates), and (a)
   trained on two sites and scored on the third (all crops, three folds), against (a) on Kim's random split.

The trunk is the `mt_release` export of tol_full (`trunk.onnx`, fp32; the head scored by the release's `predict.py`),
which reproduces mini_trainer's evaluation on 300 images at every rank (journal 2026-11-19): largest centred square,
read at 256, resampled to 224. Kim's crops are elongated and often small, so the square read cuts some.

### Predictions (before any score)

Supervisor's (above): zero-training order accuracy within 3 points of the specialist; a trained head within 1. On
Kim's metric that is macro F1 at least 0.940 from the marginals and 0.960 from the head.

Agent's:

- **Zero training, V0, Kim's validation list:** macro F1 0.78 (80 % interval 0.65-0.88), accuracy 0.86 (0.75-0.93),
  so the supervisor's 3-point prediction fails by more than 10 points. The full set within 1 point of the validation
  list (the same distribution). Best categories Ephemeroptera, Lepidoptera Macro, Araneae, Coleoptera (F1 at least
  0.9); worst Trichoceridae (winter crane flies are rare in photographs), Hymenoptera Other, Vegetation and the
  Brachycera / Nematocera boundary on small crops; Lepidoptera Micro against Trichoptera the largest confusion. The
  reasons: ToL learned from photographs taken by people, these are night trap crops of 30-100 px, and three of the
  16 labels are conventions that are not monophyletic (Nematocera, micro-moths, "Hymenoptera Other").
- **V1** raises Tipulidae's recall and macro F1 by 1-2 points; **V2** lowers macro F1 by 0-2 points (Vegetation
  absorbs every off-list arthropod).
- **Trained head (a):** macro F1 0.955 (0.93-0.97), accuracy 0.975: within 3 points of Kim, and within 1 point only
  with probability about 0.4. (b) 5-10 points below (a); (c) within 1 point of (a).
- **The split:** 1-NN accuracy at least 0.95; (a) trained on two sites and scored on the third loses at least 5
  points of macro F1 against the random split, so the random split flatters every trained model and the
  zero-training number is the only one in this section that measures transfer.

Added at 12:10, after the zero-training score on the validation list and before any number from the training list:
a ladder for C5, from the ToL head's own output to a head on the embedding, each rung fit on Kim's training list
only. L1: 16 biases on the zero-training log masses (the class prior alone); L2: a linear map of the 16 log masses
(the mapping relearned); L3: logistic regression on the log family posterior (families with mean mass above 1e-4).
Predicted macro F1 on the validation list: L1 0.80-0.84, L2 0.84-0.88, L3 0.90-0.93, then (a) at 0.955. If C5
holds here, L1 and L2 close most of the gap; if the deficit is discrimination, only (a) does.

### Wider scope (owner, 13:30): leakage, specialisation candidates, a GPU job

The owner asked whether Kim's split is simply leakage, and for the specialisation candidates above (rows 3, 4, 6,
7, 8, 12) on this set, with Kim's specialist retrained so that a leak-free comparison has a specialist. IDs:

- **BT1:** zero training and every labelled rung of the ladder ((a), (b), (c), L1-L3), each scored twice: on Kim's
  validation list, and site held out (fit on two sites, scored on the third; three folds, pooled over all crops).
  Plus near-duplicates across each split: cosine of the nearest training embedding at least 0.98 and 0.95, 1-NN,
  byte-identical pairs.
- **BT2 (GPU):** Kim's recipe retrained from his repository's model code, on his split (a reproduction check against
  0.970) and on the three site folds. His training script is not public, so the augmentation ranges are ours.
- **BT3 (GPU, then CPU):** the trunk re-run for tta3bm (mean of the species logits over r0 and the two rotated,
  border-mean-padded views) and pad-to-square (padsq: the crop padded to a square with its border mean, instead of
  the centre cut), for zero training and head (a).
- **BT4 (GPU):** a head trained through the frozen trunk with augmentation (TL8's recipe: hidden layer, cosine head,
  TrivialAugmentWide, 10 epochs), on his split and the site folds.
- **BT5 (CPU, no labels):** L0, an EM prior (Saerens et al. 2002, as dev/107: uniform source prior, 50 iterations)
  on the zero-training category posterior, estimated on all crops and per site, applied as p(y|x) x prior(y); a
  temperature on the 16 category log masses, fit on Kim's training list (argmax unchanged, so macro F1 and accuracy
  cannot move, only calibration, and the EM prior through it); a pseudo-label head: (a) fit on the zero-training
  answers of unlabelled crops (the training part of each fold), gated to the 70 % most confident (OT1's gate).

Jobs (`ucloud/lepinet-bt-a.toml`, `lepinet-bt-c.toml`, running `dev/117`): lepinet-bt-a (12472000: BT2 then BT3)
and lepinet-bt-c (12472004: BT4), one B200 1g MIG slice each, submitted 13:58. Job a was restarted at 14:09 as
12472031 after 11 minutes: a genuine mistake. The crop squash ran in a `multiprocessing.Pool` whose workers
sized torch's thread pool from the host's 256 cores (CLAUDE.md warns of exactly this), so six workers crawled on
6 vCPUs; the local check had shown it (25 crops/s) and was not read. Fixed with one thread per worker and
`torch.set_num_threads` from the cgroup quota (800 crops/s locally). The trunk path in the job reproduces the CPU ONNX embeddings (r0 cosine at least 0.99992 on 27 crops).

**Predictions (14:05, before any of these numbers; the zero-training score on the validation list was known):**

- **Leakage:** at least 30 % of Kim's validation crops have a training crop at cosine 0.95 or more, at least 10 % at
  0.98; site held out, under 5 % at 0.98.
- **BT2:** the reproduction lands at 0.960-0.975 macro F1 on his list; site held out, his recipe drops 7-12 points
  (pooled 0.85-0.90).
- **BT1 site held out:** (a) drops 5 points or more (as predicted above); L1 and L2 drop 2 points or less (16 to 272
  parameters cannot memorise). Against BT2 on the same folds, (a) is within 3 points of the retrained specialist.
- **BT4:** on his list 0.960-0.970 (level with or above (a)); site held out, within 2 points of BT2 or above it.
- **BT3:** zero training with tta3bm +1-3 points macro F1 over r0; padsq +0-2 (keeps legs, tails and long wings
  that the centre cut loses on elongated crops; Tipulidae, Ephemeroptera, Neuroptera gain most).
- **BT5:** L0 (EM prior, all crops pooled) +2-5 points on zero training (0.79-0.81 on Kim's list), per site no
  better than pooled (+-1: three Danish sites share most of their fauna); the temperature changes nothing in macro
  F1 by construction and improves calibration; the pseudo-label head 0.78-0.83, inheriting zero training's
  systematic errors (mayflies as micro-moths, bibionids as Nematocera) rather than repairing them.

**Added 19:35, after the leak and GPU numbers and before this split ran: a trap-night split (post hoc).** Kim's
80/20 again, but by (trap, night) within the same sites (3,532 trap-nights, 20 % held out, seed 0; 99.8 % of his
validation crops share a trap-night with training crops). It removes what his random split shares within a night
(any individual seen on several frames, the night's conditions) and keeps the sites, so it separates leakage from a
new site's shift. Prediction: 1-NN and (a) on it land within 2.5 points of their values on his list (1-NN there:
0.947), far above site held out (1-NN 0.856): his split's advantage is mostly the site, not shared individuals.

### Results

**Zero training (V0) on Kim's validation list, 25,034 crops** (`data/broad_taxa/zeroshot.json`):

| | macro F1 | macro P | macro R | accuracy | 11 orders, macro F1 | Lepidoptera or not, F1 |
|---|---|---|---|---|---|---|
| Kim's ResNet50 (his logits) | **0.970** | 0.974 | 0.965 | 0.984 | 0.985 | 0.998 |
| ToL, zero training, V0 | **0.765** | 0.762 | 0.815 | 0.825 | 0.792 | 0.916 |
| V1 (Tipuloidea, Neuropterida) | 0.742 | 0.739 | 0.817 | 0.806 | 0.790 | 0.917 |
| V2 (Vegetation = everything off the 15) | 0.752 | 0.769 | 0.803 | 0.807 | 0.773 | 0.914 |

Per category (V0 F1 against Kim's): Araneae 0.85 / 0.97, Coleoptera 0.81 / 0.97, Brachycera 0.82 / 0.99,
Nematocera 0.86 / 0.98, Tipulidae 0.67 / 0.96, Trichoceridae 0.39 / 0.78, Ephemeroptera 0.61 / 1.00, Hemiptera
0.85 / 0.98, Hymenoptera Other 0.70 / 0.96, Vespidae 0.95 / 1.00, Lepidoptera Macro 0.90 / 0.99, Micro 0.85 / 0.99,
Neuroptera 0.75 / 0.99, Opiliones 0.92 / 0.98, Trichoptera 0.86 / 0.99, Vegetation 0.46 / 0.99. The set-aside mass
(taxa outside the 16) averages 0.058 and is spread thin (termites, silverfish, Odonata, Orthoptera at 0.001 each).

Where it goes wrong, by the confusion matrix and a look at the crops:

- **Ephemeroptera recall 0.44 (precision 0.99).** 673 of 2,030 mayflies go to Lepidoptera Micro and 93 to Macro.
  They are small, pale, blurred subimagos (56-90 px wide), given to Tineidae or Crambidae at low confidence (top
  species 0.08-0.42); the larger, sharper ones are recognised (Cloeon dipterum, Hexagenia at 0.3-0.8). A domain
  gap: photographs of mayflies are of sharp adults.
- **Brachycera recall 0.71.** 364 go to Nematocera, 297 of them to Bibionidae, a nematoceran family by taxonomy. The
  crops are dark, robust flies; Kim's groups were sorted "based on the content of the recorded images", so a
  bibionid may well sit in his Brachycera. Part of this gap is the convention, not the model.
- **Trichoceridae 0.39; Nematocera to Trichoceridae 390.** Winter crane flies, crane flies (Tipulidae, Limoniidae)
  and mosquitoes trade places at this size; Kim's own model is weakest here too (0.78).
- **Vegetation precision 0.33.** Plant mass wins on 141 of 179 vegetation crops, but also on 290 arthropod crops
  (Nematocera 78, Lepidoptera 62, Ephemeroptera 44, Trichoptera 36); grasses (Poaceae) are the usual plant answer,
  plausibly for thin, pale, blurred bodies.
- **Predictions scored.** Agent: macro F1 0.78 (0.65-0.88), measured 0.765, inside; accuracy 0.86 (0.75-0.93),
  measured 0.825, inside; the worst categories named (Trichoceridae, Hymenoptera Other, Vegetation) were among the
  five worst, but Ephemeroptera, predicted among the best, is the third worst. V1 +1-2 points: wrong, -2.3 (Limoniidae
  is ToL's usual answer for small Nematocera and Trichoceridae, so widening Tipulidae draws them in). V2 0-2 points
  lower: right (-1.3). **Supervisor: within 3 points of the specialist (0.940): falsified, 20.4 points below.**
- **All 125,135 crops** (nothing trained on them): V0 macro F1 0.760, accuracy 0.822 (V1 0.738, V2 0.747); within
  1 point of the validation list, as predicted.

**Leakage across Kim's split** (`leak.json`; nearest training crop by cosine of ToL's embedding):

| scored part | n | cos >= 0.98 | cos >= 0.95 | median | 1-NN macro F1 (acc) | byte-identical twins |
|---|---|---|---|---|---|---|
| Kim's validation list | 25,034 | 0.9 % | 10.3 % | 0.910 | 0.947 (0.968) | 3 |
| site held out (LV / OH / SS) | 125,123 | 0.0 % | 1.4 % | 0.881 | 0.856 (pooled) | 0 |

So his split is not a resubstitution test: one validation crop in ten has a near-twin in training, one in a hundred
an almost exact one. Predictions: "at least 30 % at 0.95 and 10 % at 0.98" falsified (10.3 % and 0.9 %); "site held
out under 5 % at 0.98" right (0.0 %); "1-NN accuracy at least 0.95" right (0.968). Between his split and a new site,
1-NN loses 9 points of macro F1; how much of that is shared nights and how much the site is the trap-night split's
question (below, when it lands).

**BT2, Kim's recipe retrained** (lepinet-bt-a, 76 min on a 1g slice; `gpu.json`). On his list it moves between 0.913
and 0.964 macro F1 from epoch to epoch (accuracy 0.937-0.981); his protocol, the best epoch on that list, gives
**0.964** (epoch 12, accuracy 0.981), half a point under his own published checkpoint (0.970, chosen the same way);
the last epoch 0.925, the mean of the last five 0.947. Site held out, last epoch: **0.870 pooled** (accuracy
0.929); per site LV 0.828, OH 0.844, SS 0.865, each moving 5-12 points between epochs (LV 0.742-0.850). So a
single number for a ResNet50 trained this way carries several points of epoch noise, and his 0.970 is a best-of-25
on the list it is reported on. Predictions: reproduction 0.960-0.975, right (0.964); site held out 0.85-0.90, a 7-12
point drop, right (0.870, 9.4 points).

**BT4, a head trained through the frozen trunk with augmentation** (TL8's recipe; lepinet-bt-c, 131 min): his list
**0.962** (accuracy 0.980), level with his recipe's best epoch and without any epoch choice; **site held out 0.897**
(accuracy 0.943; LV 0.871, OH 0.871, SS 0.873), **2.7 points above the retrained specialist**, and above it at every
site (+4.2, +2.7, +0.7 against its last epoch; +7.1, +2.8, +2.0 against the mean of its last five). By category, site
held out, it is ahead on 13 of 16 (Araneae 0.95 against 0.86, Tipulidae 0.94 against 0.89, Vegetation 0.70 against
0.65) and behind on Ephemeroptera (0.92 / 0.95), Neuroptera (0.95 / 0.97) and Trichoptera (0.96 / 0.97); both fail
on Trichoceridae (0.32 and 0.28: winter crane flies, half of them caught in November and two thirds at SS, so
holding SS out leaves 547 to learn from). The routing decision
C3's pipeline needs, Lepidoptera or not: F1 0.990 (BT4) against 0.991 (BT2) site held out, 0.997 against 0.996 on
his list. Predictions: 0.960-0.970 on his list, right; site held out within 2 points of BT2 or above, right (above).

**The trap-night split** (`night.json`; 99,534 training crops, 25,589 scored, from 706 held-out trap-nights): near-twins
at cosine 0.95 or more 3.7 % (his split 10.3 %), at 0.98 0.1 %; **1-NN 0.937 and (a) 0.944**, against 0.947 and
0.953 on his split and 0.856 and 0.878 site held out. Zero training on the same crops: 0.750 (0.765 on his list),
so part of even that point is which crops are scored. **His random split is worth about 1 point; a new site costs
7-9.** The owner's question, "is Kim's split just leakage?", has a measured answer: no; its weakness is that every
site is in training, not that individuals cross the split. Prediction (within 2.5 points of his list, far above
site held out): right.

**Everything on one page** (macro F1, accuracy in brackets; "site held out" pools the three folds; for rungs without
labels it is simply all crops):

| rung | needs from the target | Kim's list | site held out | Lepidoptera or not (his list / site) |
|---|---|---|---|---|
| Kim's ResNet50, his published checkpoint | his labels; epoch chosen on the list | 0.970 (0.984) | - | 0.998 / - |
| BT2: his recipe retrained (best epoch / last) | labels, GPU | 0.964 / 0.925 (0.981 / 0.950) | 0.870 (0.929) | 0.996 / 0.991 |
| zero training, centre read (V0) | nothing | 0.765 (0.825) | 0.760 (0.822) | 0.916 / 0.914 |
| zero training, tta3bm (BT3) | nothing | 0.829 (0.888) | 0.825 (0.886) | 0.952 / 0.950 |
| zero training, pad to square (BT3) | nothing | 0.826 (0.887) | 0.821 (0.885) | 0.959 / 0.957 |
| L0, EM prior pooled / per site (BT5) | unlabelled crops | 0.732 / 0.775 | 0.734 / 0.774 | 0.867 / - |
| pseudo-label head, 70 % gate (BT5) | unlabelled crops | 0.801 (0.841) | 0.776 (0.830) | 0.920 / 0.916 |
| L1, 16 biases | labels | 0.820 (0.870) | 0.785 (0.844) | 0.950 / 0.937 |
| L2, linear map of the 16 masses | labels | 0.861 (0.905) | 0.817 (0.870) | 0.968 / 0.953 |
| L3, linear on the family posterior (families above 1e-4 mean mass) | labels | 0.933 (0.959) | 0.857 (0.910) | 0.991 / 0.977 |
| (b) nearest class mean on the embedding | labels | 0.793 (0.809) | 0.758 (0.775) | 0.912 / 0.906 |
| 1-NN on the embedding | labels | 0.947 (0.968) | 0.856 | - |
| (a) linear on the embedding | labels | 0.953 (0.970) | 0.878 (0.923) | 0.995 / 0.983 |
| (a), tta3bm at test | labels | 0.941 (0.970) | 0.891 (0.936) | 0.994 / 0.985 |
| (a), pad to square | labels | **0.961** (0.978) | **0.900** (0.943) | 0.998 / 0.991 |
| (c) linear on the trunk features | labels | 0.958 (0.975) | 0.887 (0.927) | 0.996 / 0.982 |
| BT4: head trained through the frozen trunk with augmentation | labels, GPU | 0.962 (0.980) | 0.897 (0.943) | 0.997 / 0.990 |

What it says:

- **A frozen generalist with a trained order head is in the specialist's range on his split and ahead of it at a new
  site.** On his list the best frozen-trunk heads sit 0.8-0.9 under his published, list-selected checkpoint and level
  with his recipe retrained (0.961-0.962 against 0.964 best epoch, 0.925 last). Site held out, (a) on the padded read
  (0.900, a linear model on CPU) and BT4 (0.897) beat the retrained specialist (0.870) by 2.7-3.0 points. That is C3's
  order half, with the honest caveat that 16 conventional groups need labels: with none, the generalist is 14 points
  below at best.
- **C5's decomposition, on his list.** Of the 19.9 points between zero training and his recipe, reading the whole
  crop (pad or tta3bm) gives 6; the class prior alone (L1) 5.5; relearning the mapping of the 16 category masses (L2)
  9.6; a linear readout of ToL's own family posterior (L3, still nothing but the head's output) 16.8, 84 % of the
  gap; the embedding (a) 18.8. So the head already discriminates these groups; what zero training lacks is the
  mapping from taxa to Kim's conventions (bibionids are "Brachycera" to his annotators), the fauna's prior and a read
  of the whole insect. Discrimination proper is the last 1-3 points.
- **Reading the crop is the largest label-free lever.** The centre square of an elongated crop loses wings, legs and
  tails: padding to square or tta3bm (whose views are padded 25 %) adds 6 points with no training, Ephemeroptera most
  (F1 0.61 to 0.80-0.84: the tails), then Neuroptera (0.75 to 0.83-0.88), Coleoptera and Brachycera; Trichoceridae
  and Vegetation do not move. For a trained head, padding helps everywhere (+0.8 on his list, +2.1 site held out, against (a) on the job's r0);
  tta3bm on a head fit on centre reads helps at a new site (+1.3) and costs on his list (-1.2), the views being
  foreign to it. Neither combination (padded and rotated) nor a head trained on padded crops through the trunk (BT4
  with the padded read) was run.
- **The unlabelled-data tools barely help, or hurt, because zero training's errors are systematic.** EM drove the
  Trichoceridae prior to 0.000 (true share 0.013), Ephemeroptera to 0.022 (0.081) and Lepidoptera Micro to 0.329
  (0.225): it reads the mayflies-as-micro-moths confusion as fauna and amplifies it (-3.3 pooled; per site +1.0). Its
  uniform source prior is also wrong here: a category's implied prior grows with the number of ToL species in it.
  The pseudo-label head (gate 91.7 % correct) inherits the same errors (mayfly recall 0.44 to 0.48) and repairs only
  Vegetation (F1 0.46 to 0.82, whose false plant answers were unconfident). The temperature fit on his training list
  is 1.06: the category posterior was already calibrated (ECE 0.036 at T = 1, 0.046 at 1.06; NLL 0.7195 to 0.7180).
- **Caveats.** The 11 non-Lepidoptera categories are conventions of one annotation team; Kim's training script is
  not public (BT2's augmentation ranges are ours); one seed everywhere, and BT2's epoch-to-epoch spread (5-12 points)
  is larger than most differences in the table, BT4's is unmeasured; (a) and L3 chose C = 0.1, the edge of the
  pre-registered grid, on every fold (not extended after the fact); BT4 and (a) differ in more than augmentation
  (hidden layer, cosine head, epochs), so the table does not attribute BT4 to augmentation.

**Predictions scored** (the 10:50, 12:10 and 14:05 sets; "right" means inside the stated range):

| prediction | outcome |
|---|---|
| zero training 0.78 (0.65-0.88), accuracy 0.86 (0.75-0.93); full set within 1 point | right: 0.765, 0.825; 0.760 |
| V1 +1-2; V2 0-2 lower | wrong (-2.3); right (-1.3) |
| (a) 0.955 (0.93-0.97); within 1 point of Kim with probability 0.4 | right (0.953); not within 1 (-1.7) |
| (b) 5-10 below (a); (c) within 1 of (a) | wrong (16 below); right (+0.5) |
| L1 0.80-0.84, L2 0.84-0.88, L3 0.90-0.93 | right, right, just above (0.933) |
| 1-NN accuracy at least 0.95 | right (0.968) |
| near-twins: at least 30 % at 0.95, 10 % at 0.98; site held out under 5 % at 0.98 | wrong (10.3 %, 0.9 %); right (0.0 %) |
| BT2 0.960-0.975; site held out 0.85-0.90 | right (0.964); right (0.870) |
| site held out: (a) drops 5 or more; L1, L2 drop 2 or less; (a) within 3 of BT2 | right (7.5); wrong (3.5, 4.5); right (+0.8) |
| BT4 0.960-0.970; site held out within 2 of BT2 or above | right (0.962); right (+2.7) |
| BT3: tta3bm +1-3, padsq +0-2 on zero training | wrong both: +6.3, +6.0 |
| BT5: L0 +2-5 pooled, per site within 1 of pooled; T changes no macro F1; pseudo-label head 0.78-0.83, inheriting the errors | wrong (-3.3), wrong (+4.3 over pooled); right by construction; right (0.801), right (Vegetation the exception) |
| trap-night split within 2.5 points of his list | right (-1.0, -0.9) |
| supervisor: zero training within 3 points of the specialist; a trained head within 1 | wrong (-20.4, -14 with padding); right for BT4 (-0.8) and (a) padded (-0.9), wrong for (a) (-1.7) |

**What C3 still needs.** (1) The pipeline end to end: an order head from this section routing crops to TL8b's moth
head, scored against the two specialists on one labelled set; MCC24 cannot score the species half (its moth names
look like a classifier's output, not labels), and the trap probe has no order labels, so this waits for the owner's
order dataset. (2) The best order head is probably BT4 or (a) on a padded read with tta over padded views; not run.
(3) TL10's trunk, when it lands (rerun `dev/115 embed` with its release). (4) If EM is to be tried again, with the
head's implied category prior as source prior.

GPU cost: lepinet-bt-a 12472031 about 1.7 h and the aborted 12472000 0.2 h, lepinet-bt-c 12472004 about 2.2 h, on
1g slices: about 4.1 slice-hours, roughly 0.6 B200-hours. On UCloud, `/12347837/repos/broad_taxa` holds the 0.52 GB
staging folder and 1.5 GB of outputs (nothing deleted; the owner can remove it). Local CPU: 6.6 h for the trunk on
all crops (5.4 img/s on 4 cores, 3.9 GB), then 1.5 h of scoring; the largest step (`head`) peaked at 7.6 GB, above
the few GB asked for and under the 12 GB cap.

### BT6: the reverse direction (owner, 2026-11-24; supervisor, `dev/118`, CPU)

The owner's question: score Kim's model on ToL's test images of the same taxa, against our best ToL model. C3 scored
ToL on Kim's ground (zero training 0.765, 0.829 with views, against his 0.970); BT6 scores his model on ToL's ground.
Sample (drawn before any score): ToL test rows with no exclude flag, in the 15 arthropod categories by the V0 mapping,
Vegetation left out (Kim's Vegetation is trap background, not plant photos), at most 2,000 per category at random:
27,898 images of 10,035 species. 20,734 are citizen-science photos, 6,572 museum specimens and 592 unknown; Trichoptera
and Ephemeroptera are mostly museum specimens. Three classifiers, each scored as trained (16-way, so a Vegetation answer
is wrong) and restricted to the 15 categories:
- Kim's published weights, with his preprocessing;
- tol_full with zero training (dev/115 V0);
- C3's head (a), fitted on Kim's training crops on ToL's embedding: the trap-trained head on the general trunk.

The test is lopsided by design. Kim's model classifies detector crops of trap insects; ToL's photos are whole field
photos and pinned specimens. It measures how far each model travels from its own domain, the mirror of C3, not which
classifier is better.

**Predictions (before the score; the 48-image smoke test was seen first: all spiders, ToL right on 47, Kim's model
Vegetation on 36, Araneae on 7).** Macro-F1 over the 15 categories:
- ToL zero training: 0.88-0.95 (in domain; the weak spots should be Nematocera against Brachycera, macro against micro
  moths, and Trichoceridae on 72 images);
- Kim as trained: 0.25-0.45, with Vegetation absorbing many photos; restricted to 15: 0.40-0.60;
- Kim on museum specimens above citizen-science photos by 5-15 points (15-way): a pinned insect on a plain background
  is closer to a trap crop than a field photo is;
- ToL head (a): 0.75-0.88, below zero training because its category prior and boundaries were fitted on traps.

If these hold, the symmetric reading for paper B: the specialist loses 40 to 60 points off its domain, the generalist
about 14-20 on the specialist's domain with zero training and about 1 with a cheap head (C3).

**BT6 scored (2026-11-24; `data/broad_taxa/reverse/scores.json`, 200 bootstrap draws over images).** Macro-F1 over the
15 categories present, with its 95 % interval:

| classifier | as trained | restricted to 15 | accuracy (15) | citizen / museum (15) |
|---|---|---|---|---|
| ToL, zero training | **0.974** [0.970, 0.978] | 0.975 | 0.983 | 0.974 / 0.950 |
| Kim's published model | **0.130** [0.124, 0.137] | 0.137 | 0.174 | 0.158 / 0.065 |
| ToL trunk + head (a), trap-fitted | 0.357 [0.351, 0.362] | 0.368 | 0.376 | 0.357 / 0.300 |

The measurement was checked first. On 320 of Kim's own crops, dev/118's model and preprocessing reproduce his published
predictions:
- same answer on 320 of 320;
- accuracy 0.9875 in both;
- largest logit gap 0.38.
So the 0.13 is his model, not the harness.

What Kim's model does on photos:
- It answers Vegetation for 59 % of them: 65 % of citizen-science photos (leaves and plants behind the insect) and 42 %
  of museum specimens.
- With Vegetation removed, it calls 49 % of all images a macro-moth.
- It is confident while wrong: median maximum softmax 0.988, and 67 % of images above 0.9.
ToL's weakest categories are Trichoceridae (0.86, 72 images) and Nematocera (0.95). Macro- against micro-moths was not a
weak spot (0.99 each).

Predictions scored:
- ToL zero training 0.88-0.95: wrong, above (0.974).
- Kim 0.25-0.45 as trained, 0.40-0.60 on 15: wrong, far below (0.130 / 0.137).
- Kim on museum specimens above citizen science: wrong direction (0.065 against 0.158). Group macro-F1 is confounded by
  each group's category mix, so this is not read further.
- Head (a) 0.75-0.88: wrong, far below (0.357).
I underestimated how specialised a trap-trained classifier is, whether it is Kim's whole network or a linear head on
our trunk.

Reading for paper B, the symmetric comparison (C3 and BT6):

| | on Kim's trap crops | on ToL's test photos |
|---|---|---|
| Kim's model (the specialist) | 0.970 | 0.13 |
| ToL, zero training (the generalist) | 0.765 (0.829 with views) | 0.974 |
| ToL trunk + trap head (a) | 0.953, site held out 0.878 (BT1) | 0.36 |
| ToL trunk + trap head, augmented (BT4) | 0.962, site held out 0.897 | not scored |

- The specialist loses 84 points off its domain, and loses them confidently.
- The generalist loses 21 points on the specialist's domain with zero training, and none with a cheap head.
- Head (a) shows the specialisation lives in the head. On the same trunk features the generalist's own head scores
  0.974 and the trap head 0.357, so one trunk with both heads keeps both abilities.
- A likely cause of head (a)'s collapse, not tested: its inputs are standardised with trap statistics, which amplifies
  directions that are quiet in traps.
- The comparison is lopsided by design. Kim's model never saw a photo; it shows what a domain-specific model costs
  outside its domain, not that ToL is the better trap classifier.

### BT7: the owner's box tree moth crops (supervisor, 2026-11-25, CPU, `dev/120`)

The owner gave a mothbot export of Kim's order classifier ("nemo") run on snapshots selected for the box tree moth
(*Cydalima perspectalis*, an invasive Crambidae): 4,933 crops from 2022-2025 and about 60 traps across Europe (most in
the IAS project; 2022 crops from Kim's own LV, OH and SS sites are included), with a parquet of the annotations. The
owner's view: Kim's classifier is particularly bad on this species, probably because his training set has few. The
task: the same zip with our best order classifier's answers in place of Kim's.

The classifier: head (a) on the pad-to-square read (0.961 on his list, 0.900 site held out; level with BT4 and
CPU-only), refit on his training list from dev/117's embeddings (0.9608, the recorded score). The CPU read reproduces
the GPU embedding it was fit on (48 of Kim's crops: cosine 0.99999 or more, the same answer on all). ToL's own
species head gives an independent-of-labels species read of each crop: the box tree moth is row 28774,
*Cryptographis perspectalis*, the Catalogue of Life build's name (about 49 k crawled images).

| | crops | Kim's classifier | ours |
|---|---|---|---|
| ToL's species answer is the box tree moth | 2,428 (2,346 above 0.5) | Vegetation 1,078, Lepidoptera Micro 763, Macro 510, Tipulidae 74, other 3: **52.4 % Lepidoptera** | Micro 2,358, Macro 67, other 3: **99.9 % Lepidoptera** |
| the other crops | 2,505 | | agreement with Kim 63.7 % |
| Kim's own dataset (125,135 crops, ToL centre read) | 1 box tree moth (probability below 0.5; labelled Vegetation) | | |

- **The owner is right on both counts.** Kim's classifier calls 44 % of the box tree moths Vegetation and 3 % crane
  flies, with a median score of 1.000 on those it calls Vegetation, and his training set holds essentially none (1 of
  46,816 Lepidoptera crops by ToL's read, 3 in the top 5).
- **Checks.** The output zip has the input's entries, Arrow schema and row order, the same image bytes, and differs only
  in label, labelid, score, abovethreshold, algorithm and orderlabel; filenames follow the input's pattern
  (`label_score_project_partner_snapshot_crop_detection.jpg`, score to 3 decimals with trailing zeros cut), checked on
  every crop before renaming. score_order and abovethreshold_order are left as they were: they are not Kim's order
  score (they anti-correlate with it) and their meaning is unknown. abovethreshold keeps the input's rule
  (score >= 0.045), which every 16-way softmax maximum meets.
- **I looked at a few crops.** The three box tree moth crops opened (two Kim-Vegetation, one Kim-Tipulidae, all at
  about 1.0) are unmistakable. The crops ours calls Vegetation are empty background, the closest category Kim's scheme
  has; some empty crops go to Diptera Nematocera.
- **Caveats.** No human labels: ToL's species read is the reference, and it shares the trunk with our order head, so
  the two are not independent (the visual check is small). Five crops have a near-twin in Kim's dataset (cosine 0.98 or
  more), negligible. Box tree moths split Micro 97 % / Macro 3 % here; Crambidae are micro in the convention dev/115
  wrote down, and Kim's labels for them are unknown.

Output (outside the repo, the owner's folder): `/home/au761367/Documents/mothbot/AMI_IAS_tol.zip`, and a side table
`AMI_IAS_tol.csv` (per crop: Kim's answer, ours, ToL's species top 1 and its probability, the box tree moth's
probability, the nearest crop of Kim's dataset).

**The owner's review of head (a)'s zip, and two more versions (2026-11-25).** The owner saw box tree moths in both
Lepidoptera folders, snails among the macros, crops with several insects or a branch called macro with confidence, and
dirt confidently in Ephemeroptera and Vegetation (Kim's scheme has no dirt class). ToL's species read agrees: of head
(a)'s 266 macros, 90 are Crambidae (65 with box tree moth probability above 0.2) and about 30 more are other micro
families, 4 are land snails and 11 are read as tardigrades (dirt); of its 60 Ephemeroptera, 9 are read as mayflies and
25 as moths. **Kim's labels are not the cause:** on his own Lepidoptera crops his macro / micro follows the family
convention dev/115 wrote down 98-99 % of the time (Crambidae 3,792 micro, 8 macro). The box tree moth is simply absent
from his training set, and a head fit to it calls a large white crambid macro.

- **Raw ToL** (`raw`): ToL's own head on the padded read, family posterior summed into Kim's 16 categories by the V0
  mapping, every other family into its ToL order (snails "Stylommatophora"). Box tree moths 2,427 of 2,428 micro.
  191 crops land outside his 16, median mass 0.35 against 0.95 inside. ToL's open-vocabulary names on dirt are
  nonsense, sometimes confident: four crops opened show a real snail (0.99, Kim's macro), and blurry streaks or blobs
  read as a gecko (0.82, at a Danish site), a songbird (0.86) and an amphipod (0.92). It also calls 435 crops winter
  crane flies, where head (a) says Nematocera for 368; 280 of them are from October and November, the family's
  season, so part may be right; no labels settle it. On Kim's data raw ToL is 13 points below head (a) (0.826 /
  0.821, BT3).
- **The product of the two** (`combo`, supervisor's idea, checked on Kim's data before use, no prediction written):
  head (a)'s probabilities times raw ToL's, renormalised over the 16. On Kim's data **0.949 on his list (-1.2) and
  0.899 site held out (-0.0)**, the components reproducing their recorded 0.961 / 0.900 and 0.826 / 0.821. Where
  ToL puts more than half its mass outside the 16, the crop is labelled **Other** (labelid 16; ToL's name only in
  the side table). On the mothbot crops: box tree moths 2,423 micro, 4 macro; 213 Other (snails, dirt; head (a) had
  34 of them in macros, 47 in Vegetation); Ephemeroptera 60 to 25, macros 266 to 170; it keeps head (a)'s answer
  on 89 % of crops.
- **Not fixed by any classifier:** crops holding several insects, or an insect on a branch, are detection errors; a
  classifier names the dominant object. They need the detector, or a "several objects" class with labels.
- **BT4** (the augmented head) was not run: dev/117 saved its logits, not its weights.

Outputs in `/home/au761367/Documents/mothbot/`: `AMI_IAS_tolraw.zip` and `AMI_IAS_tolcombo.zip`, each with its side
table (`.csv`); the trunk pass is cached in `data/broad_taxa/mothbot_cache.npz`, so further variants cost seconds.

## Log

- 2026-11-21: C2 scored (lepinet-2a; section "C2: scores" below, BS1-BS4): BioCLIP-2 reproduced with its own code
  (5-shot, zero-shot, NeWT, AwA2 40/10, PlantDoc within 2; 1-shot Fungi 4.2 below); frozen ToL +1.1 at 5-shot and
  +0.6 at 1-shot over nine species tasks, head zero-shot 54.8 against text 53.4, NeWT and AwA2 1.4 to 3.5 below,
  PlantDoc above; overlap large on Fungi (517/1,000 by id) and NeWT (3,278 of 6,000 audited) but it does not
  inflate ToL.
- 2026-11-21: C3's order half scored (broad-taxa agent; section C3 above, BT1-BT5): frozen-trunk heads level with
  Kim's specialist on his split (0.961-0.962 against 0.964 retrained, 0.970 published) and 2.7-3.0 points above it
  site held out (0.897-0.900 against 0.870); zero training 0.765, 0.83 with a padded or tta3bm read; EM prior and
  pseudo-labels do not repair systematic errors; his split is not leakage (trap-night split within 1 point).
- 2026-11-20: C3's order half started on Kim Bjerge's MCC24 broad-taxon data (section above): split recovered,
  metric reproduced, mapping and predictions written before scoring; embeddings running on 4 CPU cores.
- 2026-11-20: opened (owner's message: new story first; C1 to C4 stated by the owner; C5 from Z23). TL10 already
  running. Nothing downloaded or submitted for C2 to C4.

## C2: BioCLIP-2's suite, sources and sizes (benchmarks agent, 2026-11-20)

The owner asked (2026-11-20) whether BioCLIP-2's whole benchmark suite fits on disk. It does, with room to spare.
Beyond Rare Species (already on disk) the suite is about 37 GB as archives, about 70 GB with the archives kept
beside their extraction, against 1.2 TB free on `/data`. Nothing needs cleaning. Everything whose licence allows
research use and that needs no account is now in `/data/au761367/benchmarks/bioclip2_suite/<name>/` (38.5 GB,
archives kept under each `_archives/`; they can go once the evaluation has run). Three datasets need the owner:
NABirds and Herbarium 19 sit behind a form or Kaggle terms, and FishNet's only image link is dead. No model was run;
`/data` had 1,209 GB free afterwards.

**Which version BioCLIP-2 used.** Read from its evaluation code and metadata (github.com/Imageomics/bioclip-2,
`slurm/eval.sh`, `slurm/eval_other.sh`, `data/annotation/`, `src/evaluation/*.py`) and from the paper (arXiv
2505.23883, section 4.1, Tables 1-2 and appendices F.4 and H). The species tasks are scored zero-, 1- and 5-shot on
the whole image set: k images per class sampled as support, the rest as query, 5 folds, SimpleShot. That is the
protocol already reproduced on Rare Species.

| dataset | version used by BioCLIP-2 | source | size | licence | on disk |
|---|---|---|---|---|---|
| NABirds | all 48,562 images, **555 visual categories** (400 species; 288 categories carry a sex, age or morph qualifier), labelled by common name; repo `data/annotation/nabirds/metadata.csv` | dl.allaboutbirds.org/nabirds: HubSpot form (name, e-mail, affiliation), then a terms page | about 9 GB (estimate; the link appears only after the form) | Cornell Lab terms: non-commercial research only, no redistribution, a fixed acknowledgement sentence in every publication | no: owner |
| Plankton | Meta-Album PLK **Mini**, OpenML 44282: 86 classes x 40 = 3,440 | OpenML store, `data.openml.org/datasets/0004/44282/PLK_Mini.zip` | 37 MB | MIT (WHOI) | yes |
| Insects | Meta-Album INS Mini, OpenML 44306: 104 classes x 40 = 4,160 | OpenML, `.../44306/INS_Mini.zip` | 40 MB | CC BY-NC 2.0 (SPIPOLL) | yes |
| Insects 2 | Meta-Album INS_2 Mini, OpenML 44292: 102 classes x 40 = 4,080 (IP102 pests) | OpenML, `.../44292/INS_2_Mini.zip` | 36 MB | IP102 "free for academic usage"; Meta-Album release CC BY-NC 4.0 | yes |
| Camera Trap | IDLE-OO Camera Traps (HF `imageomics/IDLE-OO-Camera-Traps`, revision 45e92a3): 5 balanced LILA subsets, 2,586 images (desert-lion 352, ENA24 1,120, island 310, ohio-small-animals 468, orinoquia 336); the score is the mean of the five | Hugging Face, public, not gated | 1.2 GB | CDLA-Permissive-1.0 | yes |
| PlantNet | Meta-Album PLT_NET Mini, OpenML 44293: 25 x 40 = 1,000 | OpenML, `.../44293/PLT_NET_Mini.zip` | 12 MB | CC BY 4.0 | yes |
| Fungi | Meta-Album FNG Mini, OpenML 44302: 25 x 40 = 1,000 | OpenML, `.../44302/FNG_Mini.zip` | 11 MB | BSD-3-Clause (Danish Fungi 2020) | yes |
| PlantVillage | Meta-Album PLT_VIL Mini, OpenML 44286: 38 x 40 = 1,520 | OpenML, `.../44286/PLT_VIL_Mini.zip` | 14 MB | CC0 1.0 | yes |
| Medicinal Leaf | Meta-Album MED_LF Mini, OpenML 44299: 25 x 40 = 1,000 | OpenML, `.../44299/MED_LF_Mini.zip` | 5 MB | CC BY 4.0 | yes |
| Rare Species | HF `imageomics/rare-species`: 11,983 images, 400 species | Hugging Face | 11 GB | CC0 1.0 (per-image licences in the card) | yes, since 2026-11-08 (`benchmarks/rare_species/`) |
| FishNet | original split, 75,631 train / 18,901 test; 9 binary habitat labels; two-layer probe; correct only if all 9 are right | Google Drive file `1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3` (images), github.com/faixan-khan/FishNet `anns/` (labels) | 5.4 GB zip (BioBench's figure) | none stated for the data (the project site is CC BY-SA 4.0, the repo has no licence) | no: link dead |
| NeWT | NeWT 2021: 164 binary tasks, linear SVM per task, mean accuracy | AWS Open Data, `ml-inat-competition-datasets/newt/` | 4.2 GB | repo MIT; iNaturalist images under their own CC licences | yes |
| AwA2 | AwA2-data: 37,322 images, 50 classes, 85 attributes; linear probe on attributes, mean F1 over attributes on the unseen classes | cvml.ista.ac.at/AwA2 | 13.9 GB | images chosen as free to use and redistribute, licence file per image | yes |
| Herbarium 19 | the **small resized** release (longer side at most 1,024 px): `small-train` and `small-validation`; generalized category discovery with BioCLIP-2's Old/New class split (`herbarium_19_class_splits.pkl`), semi-supervised k-means, clustering accuracy | Kaggle `herbarium-2019-fgvc6` only: the public bucket (`storage.googleapis.com/nybg/...`) now returns 403 | 1.7 GB + 135 MB | non-commercial research and education (2019 competition terms) | no: Kaggle |
| PlantDoc | the GitHub classification release (`pratikkayal/PlantDoc-Dataset`, train/ and test/ folders), **not** Meta-Album PLT_DOC; 1-shot SimpleShot from train, scored on all of test, 5 repeats | GitHub | 1.0 GB (+0.9 GB `.git`) | CC BY 4.0 | yes |

**What is on disk, and how it was checked** (`/data/au761367/benchmarks/bioclip2_suite/`; scripts in the agent's
scratchpad, never inside the data folders; extraction refuses absolute paths, `..` and links):

- `meta_album_mini/` (PLK, INS, INS_2, PLT_NET, FNG, PLT_VIL, MED_LF, and PLT_DOC for completeness; 355 MB):
  each zip's multipart S3 ETag recomputed and matched (`_archives/etags.txt`), the OpenML parquet kept beside it.
  Each holds `images/` and `labels.csv`; BioCLIP-2's own `metadata.csv` is copied in as `bioclip2_metadata.csv`.
  **The file sets are identical to BioCLIP-2's for all eight, image for image, and so are the labels** once both
  sides are NFC-normalised: BioCLIP-2's CSVs store accented labels decomposed (NFD), so a plain string join drops
  1,360 Insects and 80 Fungi images. Normalise before joining.
- `camera_trap/` (1.2 GB): 2,589 LFS files matched against their sha256, the CSVs and README against their git blob
  ids; images per subset equal the balanced CSVs' rows (total 2,586).
- `newt/` (8.0 GB with the archives): both md5s equal the published ones (`_archives/CHECKSUMS.txt`). 36,032
  images, all present in `newt2021_labels.csv`; 164 tasks, 18,008 train and 18,024 test rows.
- `awa2/` (27 GB with the archive): 13,923,888,724 bytes, the server's length; no checksum is published, but every
  zip member's CRC-32 passed on extraction. 37,322 images, 50 classes, 85 predicates.
- `plantdoc/` (1.9 GB): shallow git clone at commit 5467f60, object hashes checked by git. 2,342 train images in 28
  classes, 236 test images in 27.

**What the owner must do** (nothing was accepted, signed or logged into on the owner's behalf):

- **NABirds.** Fill the form at dl.allaboutbirds.org/nabirds (name, e-mail, affiliation) and accept the Cornell Lab
  terms it redirects to; put `nabirds.tar.gz` in `bioclip2_suite/nabirds/_archives/` (or pass the link to an
  agent). The terms bind the paper: non-commercial use only, and the acknowledgement sentence "Data provided by the
  Cornell Lab of Ornithology, with thanks to photographers and contributors of crowdsourced data at
  AllAboutBirds.org/Labs. This material is based upon work supported by the National Science Foundation under Grant
  No. 1010818." Third-party copies on Hugging Face exist and were not used: the terms forbid redistribution.
- **Herbarium 19.** With a Kaggle account, accept the rules of competition `herbarium-2019-fgvc6`, then download the
  small-train and small-validation archives (for example `kaggle competitions download -c herbarium-2019-fgvc6`).
  The organisers' md5s: small-train `8bf1985008dea8f79da1d9668b065543`, small-validation
  `d7e8670238a54f13b8b729823a988004`.
- **FishNet.** The Drive file returns 404 to an anonymous request (both the view page and the download endpoint, on
  2026-11-20). Try the link in a browser signed in to Google; if it asks for access, request it, or write to the
  first author (Faizan Farooq Khan, KAUST), and ask at the same time about the data's licence, which is not stated
  anywhere. The labels are public and match the paper exactly (75,631 / 18,901 rows), so only the 5.4 GB of images
  is missing.

**Where this differs from the 2026-11-08 list** ([[2026-11-08-does-a-margin-objective-make-a-better-pretrained-biodiversity-model]]):

- **The reference values are right.** Tables 1 and 2 in the arXiv HTML give the same numbers as the 2026-11-08 table,
  task by task (species means 55.6 / 64.1 / 78.3, non-species mean 57.5).
- **Camera Trap is IDLE-OO, five LILA subsets, 2,586 images**, and Table 1 reports their mean (caption; per-subset
  numbers in appendix F.4). It is a new set built for BioCLIP-2, not from LILA's own splits.
- **Insects is not "117 pollinator species"** (2026-11-07 entry). The Mini release has 104 classes at mixed ranks:
  45 are family names (Apidae, Andrenidae...), the rest French common names at species, genus, family or group
  level ("Les Eristales (autres) (Eristalis)", "Le Moro-sphinx (Macroglossum stellatarum)"). A taxon head reads it only through a hand mapping, and part of it only
  at family rank.
- **NABirds' 555 classes are not species.** A species-level prototype cannot separate the male, female and juvenile
  categories of one species, so our zero-shot on NABirds has a ceiling below 100 % that BioCLIP-2's text prompts
  ("Lazuli Bunting (Adult Male)") do not have. 1- and 5-shot are unaffected.
- **PlantDoc in Table 2 is the GitHub release**, 2,578 images now (the paper says 2,598), not Meta-Album's PLT_DOC.
  Train has one class more than test ("Tomato two spotted spider mites leaf"); it sorts last, so the two
  `ImageFolder` label indices still agree.
- **AwA2: the paper and the code disagree.** Appendix H says 45 classes train the attribute classifier and 5 unseen
  ones test it; `awa2.py` reads `trainclasses.txt` / `testclasses.txt`, which in the official archive hold 40 and
  10. The stock-BioCLIP-2 reproduction (69.5) settles which split was used; run it with the official files first.
- **Herbarium 19 is the small resized release with a class split shipped in BioCLIP-2's repo**, and since the public
  bucket closed it is only on Kaggle.
- **The Meta-Album licence question from 2026-11-07 is closed.** Every task allows research use. OpenML records all
  eight Mini datasets as CC BY-NC 4.0; the original releases are MIT, CC BY-NC 2.0, IP102 academic-only,
  CC BY 4.0, BSD-3-Clause and CC0. Insects and Insects 2 are non-commercial, which a paper is.

**For the overlap audit (C2's missing piece), the likely GBIF overlaps are these.** Fungi (Danish Fungi 2020 comes
from the Danish fungal atlas, which publishes to GBIF), PlantNet (Pl@ntNet publishes to GBIF), NeWT (iNaturalist
images), Herbarium 19 (NYBG specimens) and Rare Species (EOL, partly iNaturalist). Camera Trap (LILA), Plankton
(WHOI), PlantVillage and Medicinal Leaf (lab photographs) and AwA2 (Flickr) are unlikely to. This is a reading of
the sources, not a measurement; the audit should test it.

## C2: scores on BioCLIP-2's suite (lepinet-2a, opened 2026-11-20)

Supervisor's request (2026-11-20): reproduce stock BioCLIP-2 on each task with its own code first, then
score the frozen tol_full with the same protocols, zero-shot as the head's rows, and an overlap audit.
CPU only (the local GPU is another user's service). Script `dev/116_bioclip2_suite.py`; outputs
`data/bioclip2_suite/`; RESULTS BS1-BS4.

**Protocol, read from BioCLIP-2's code** (github.com/Imageomics/bioclip-2 at 49e1cf5, cloned read-only to
`benchmarks/bioclip2_suite/_code/`; ported, not run):
- Species tasks (`classification.py`): features from `encode_image`, L2-normalised; per fold `random.seed(fold)`,
  per class in order of first appearance shuffle and take k as support (a class is never skipped); subtract the
  support mean, L2-normalise, class centroids, nearest centroid; 5 folds, k = 1 and 5. Zero-shot: 80 OpenAI
  ImageNet templates per class name, mean of the normalised text embeddings. Class names: as they stand for
  Meta-Album (`asis`); taxonomy plus common name for Camera Trap (`taxon_com`), whose CSVs carry the right
  columns. Camera Trap is the mean of five subsets, each scored on its own.
- NeWT: features L2-normalised; per task, centre on the train mean and L2-normalise; `RandomizedSearchCV` (100
  draws over C, kernel, gamma; 5-fold) on a scaled SVC; accuracy pooled over all 164 tasks' test images; 3 runs.
- AwA2: features not normalised; Linear(d, 512), Dropout 0.5, Linear(512, 85); BCE, Adam 1e-5, 50 epochs, batch
  256, step decay 0.5 every 20; trained on `trainclasses.txt`, scored on `testclasses.txt`; prediction is
  logit > 0.5 (not the probability); macro-F1 over attributes; 3 runs.
- PlantDoc: 1 support image per class from train/, SimpleShot (sklearn NearestCentroid), accuracy on all of
  test/; 5 repeats. Its feature order comes from a shuffled loader, so our draws cannot be BioCLIP-2's.
- The ToL model through its release (`data/release_checks/release_tol`, verified against mini_trainer to the top
  class at every rank), on CPU: its own reader (largest centred square, 256 then 224 px). Few-shot on the head's
  embedding (as TL1) and, as a second row, on the trunk's pooled features. Zero-shot: each benchmark class mapped
  to a ToL taxon by name (species, else genus, else family; Latin names as given, plus a written table for
  common names), scored by the head's marginal probability of that taxon, minus the mass of any benchmark class
  nested inside it; classes that map to one taxon tie (their ceiling is reported).

**Predictions (lepinet-2a, before any embedding ran).**
- (a) *Reproduction.* Stock BioCLIP-2 within 1.5 points of the paper on every species task at 1- and 5-shot
  (Rare Species reproduced to 0.9 and 0.2), within 2 on Camera Trap (small subsets); zero-shot within 1 on
  Meta-Album, within 2 on Camera Trap. NeWT within 1 of 89.1. **AwA2 with the official 40/10 files within 2 of
  69.5**, so the code, not the appendix's 45/5, made the table. PlantDoc within 3 of 40.4.
- (b) *Few-shot, head embedding, 5-shot, ToL against BioCLIP-2's reproduced value:* Fungi +2 to +6, PlantNet 0
  to +3 (ceiling), Rare Species +2 (measured), Camera Trap 0 to +5, Insects -2 to +3 (45 of its classes are
  families, and TL7 found ToL's family structure flatter), Insects 2 -3 to +3, Plankton -5 to +1, PlantVillage
  -8 to 0 and Medicinal Leaf -3 to 0 (diseases and lab leaves are not taxa). Mean over the nine species tasks on
  disk within 2 points of BioCLIP-2's, either side; 1-shot mean 0 to 3 above.
- (c) *Non-species:* NeWT 1 to 4 below 89.1, AwA2 2 to 10 below, PlantDoc 0 to 10 below (all three reward what a
  species head is trained to ignore: age, behaviour, attributes, disease).
- (d) *Layer:* the trunk's pooled features beat the head embedding on the non-species tasks and on PlantVillage
  by 2 to 6 points, and lose to it on the taxon tasks.
- (e) *Zero-shot:* where the classes are Latin names in ToL's vocabulary (PlantNet, Fungi, Rare Species,
  Medicinal Leaf, Camera Trap) the head is above BioCLIP-2's text zero-shot; on Plankton and Insects 2 coverage
  is the limit, and on PlantVillage the ties cap it near 37 % (14 crops for 38 classes), still above BioCLIP-2's
  25.1.
- (f) *Overlap:* Fungi, PlantNet, NeWT and Insects 1 to 20 % of images in ToL's training rows; Camera Trap,
  Plankton, PlantVillage, Medicinal Leaf, Insects 2, PlantDoc, AwA2 under 1 %; Rare Species 0 by construction.
  Where the overlap is real, removing it moves ToL's few-shot by less than 1 point.

**First results (2026-11-20, before the embedding queue started).**
- *The port reproduces.* On TL1's Rare Species embeddings the ported SimpleShot gives BioCLIP-2 60.4 / 77.0 and
  ToL 62.9 / 79.1, as before (a class is never skipped here, unlike dev/109's port; no Rare Species class is
  small enough for that to matter).
- *BioCLIP-2's PlantDoc number carries a label bug.* `plantdoc.py` builds a torchvision `ImageFolder` per split,
  and each indexes its own folders in code-point order. Train has a 28th class, "Tomato two spotted spider mites
  leaf". It does not sort last: the lowercase "grape leaf" and "grape leaf black rot" sort after every
  capitalised name. Their test indices are therefore one lower than their train indices, and those 20 of 236 test
  images (8.5 %) are scored against the wrong class. The benchmarks agent's "it sorts last" read `ls`, which sorts
  by locale. Both readings are scored: as BioCLIP-2's code ran (for the reproduction) and with labels joined by
  name (for the claim).
- *Zero-shot without text, Rare Species (BS3).* The head's marginals: **70.2**, against BioCLIP-2's text zero-shot
  76.8 (our reproduction 76.7), while ToL's 5-shot is above (79.1 against 77.0). 356 classes sit on species rows
  (73.5 % right), 42 on genus, family or order marginals (45.0 %), 2 are unmapped; 97.3 % of queries are
  reachable. Dropping the head's class biases changes little (70.5). Scoring by the prototype's angle alone, the
  literal "row as text embedding", gives 34.4: the rows are prototypes only with their learned magnitudes.
- *Overlap, exact ids (BS4).* Danish Fungi file names start with the GBIF occurrence id: **607 of the 1,000 Fungi
  images come from occurrences that are in ToL, 517 of them in its training rows** (2.1 ToL rows per occurrence;
  the genus agrees on all 607). Whether the very photograph is among them is for the hash audit; at least the same
  specimen and observation are. PlantNet: 27 of 1,000 images by their Pl@ntNet image hash in ToL's media URLs,
  all trained. Insects (SPIPOLL ids): none found in ToL's URLs. Fungi is therefore read on its clean part only.
  The candidate pool for the hash audit: 11.0 M trained ToL rows in the audited genera and families, 7.7 M of
  them in the 45 Insects classes that are whole families (Geometridae alone 1.3 M rows in all splits).

**The queue, and two fixes it forced (2026-11-21, written before the reruns).** The embeddings ran on cores 0-3
from 17:24 on 20 November to 10:03 on 21 November (ToL 6 img/s, BioCLIP-2 2.5 img/s on four threads). Zero-shot,
the hash audit, ToL's NeWT and ToL's AwA2 ran between them; the few-shot step stopped on an error. Two things
were wrong in my code, not in the models:
- *The clean part crashed.* Removing flagged images from support and query alike left a Fungi class with 2 images,
  and BioCLIP-2's protocol never skips a class, so 5-shot cannot be drawn. Changed to **clean queries**: the same
  draws as the full run (the shuffle is over row indices, which gives the same permutation), accuracy over the
  queries not found among ToL's training rows. A flagged image can still be a support image. Both models are
  scored on the same queries. NeWT gets the same split of its test images.
- *AwA2's recipe does not transfer across feature scales.* ToL scored macro-F1 0.0 on its head embedding and 30.8
  on its trunk features. BioCLIP-2's code feeds unnormalised features (BioCLIP-2's mean norm 47.6) to a probe
  trained at learning rate 1e-5 that predicts an attribute when the logit exceeds 0.5; ToL's embedding has norm
  1.0 and its trunk features 1.9, so the probe barely leaves its initialisation and predicts almost nothing. This
  is a measurement artefact. Kept as "as is" (the protocol taken literally), and added a second variant: each
  model's features times one constant, to BioCLIP-2's mean norm (identity for BioCLIP-2).
- *The hash audit misses cropped copies.* Meta-Album images are 128 px squares cut from the source photographs.
  On Fungi, where the GBIF id gives the ground truth, the hash finds 29 images within 6 bits against 517 trained
  occurrences by id. So on the Meta-Album sets the hash counts are lower bounds, and only Fungi (ids) and PlantNet
  (ids, 27) have a usable clean part; for NeWT, AwA2, Camera Trap and PlantDoc, which keep whole frames, the hash
  is the right tool.

Predictions for the reruns: AwA2 with features at BioCLIP-2's norm, ToL trunk features 55 to 68 and head
embedding 50 to 65, against BioCLIP-2 near 69.5 (prediction (c) stands). Clean queries move ToL's few-shot by
under 1 point on every task but Fungi, where removing 517 trained occurrences costs ToL 1 to 4 points and
BioCLIP-2 under 1. On NeWT's audited tasks (inat_unobserved, 6,000 images, 3,286 flagged), ToL is 1 to 5 points
better on flagged test images than on clean ones; BioCLIP-2 shows a smaller gap or none, unless its own training
set holds the same iNaturalist photographs (TreeOfLife-200M draws on GBIF too; not audited here).

**Results (2026-11-21).** All on CPU, cores 0-3. Tables in `data/bioclip2_suite/results.md`; BS1-BS4.

*Reproduction (BS1).* Stock BioCLIP-2 through the port, against the paper:

| task | 1-shot | paper | 5-shot | paper | text zero-shot | paper |
|---|---|---|---|---|---|---|
| Plankton | 32.9 | 32.0 | 51.3 | 50.5 | 3.7 | 3.9 |
| Insects | 73.5 | 74.6 | 89.1 | 89.3 | 55.2 | 55.3 |
| Insects 2 | 28.2 | 28.4 | 44.7 | 44.3 | 27.6 | 27.7 |
| Camera Trap | 47.6 | 48.1 | 67.3 | 67.7 | 53.8 | 53.9 |
| PlantNet | 84.3 | 85.8 | 95.0 | 94.4 | 96.9 | 96.8 |
| Fungi | 66.1 | 70.3 | 85.3 | 85.0 | 83.6 | 83.8 |
| PlantVillage | 68.1 | 67.6 | 84.1 | 83.9 | 25.2 | 25.1 |
| Medicinal Leaf | 91.1 | 92.0 | 98.4 | 98.4 | 57.9 | 57.8 |
| Rare Species | 60.4 | 59.5 | 77.0 | 77.2 | 76.7 | 76.8 |

NeWT 88.9 (paper 89.1). AwA2 with the official 40/10 class files 67.6, runs 69.0, 68.8, 65.0 (paper 69.5): the
code's split made the table, not the appendix's 45/5. PlantDoc 41.3 as BioCLIP-2's code runs (paper 40.4), 45.9
with the labels joined by name. Every value is within 1.5 except 1-shot Fungi (4.2 below; the five folds spread
by 3.7) and 1-shot PlantNet (1.5 below; spread 5.8), the two 25-class tasks where one drawn image per class
decides much.

*Frozen ToL on the same protocols (BS2), head embedding; trunk features in brackets where they differ by more
than 1:*

| task | ToL 1-shot | minus BioCLIP-2 | ToL 5-shot | minus BioCLIP-2 |
|---|---|---|---|---|
| Plankton | 38.1 | +5.2 | 58.1 (59.2) | +6.8 |
| Insects | 67.8 | -5.7 | 87.0 | -2.1 |
| Insects 2 | 26.7 | -1.4 | 43.5 (42.1) | -1.3 |
| Camera Trap | 48.3 | +0.8 | 69.5 | +2.2 |
| PlantNet | 81.4 | -2.9 | 94.3 (93.2) | -0.7 |
| Fungi | 63.1 (61.0) | -3.0 | 83.1 (82.0) | -2.2 |
| PlantVillage | 72.6 | +4.4 | 88.1 | +4.0 |
| Medicinal Leaf | 96.5 | +5.4 | 99.5 | +1.1 |
| Rare Species | 62.9 | +2.6 | 79.1 | +2.1 |
| **mean of nine** | **61.9** | **+0.6** | **78.0** | **+1.1** |

Non-species tasks: NeWT 86.9 (trunk 87.6) against 88.9; AwA2 with features scaled to BioCLIP-2's norm 64.1
(trunk 65.3) against 67.6, and 0.0 / 30.8 with the code as is (the scale artefact above); PlantDoc 47.1 as the
code runs (trunk 46.9) against 41.3, 53.6 by name against 45.9.

*Zero-shot (BS3).* The head's marginals against BioCLIP-2's text (our reproduction), with ToL's coverage
(classes on species rows / on genus, family or order / unmapped):

| task | ToL head | BioCLIP-2 text | coverage | ceiling |
|---|---|---|---|---|
| Plankton | 5.9 | 3.7 | 1 / 9 / 76 of 86 | 10.5 |
| Insects | 73.1 | 55.2 | 20 / 67 / 17 of 104 | 80.8 |
| Insects 2 | 19.0 | 27.6 | 17 / 21 / 64 of 102 | 37.3 |
| Camera Trap | 48.4 | 53.8 | 116 / 32 / 2 of 150 | 97.6 |
| PlantNet | 96.0 | 96.9 | 25 / 0 / 0 of 25 | 100 |
| Fungi | 86.2 | 83.6 | 23 / 2 / 0 of 25 | 100 |
| PlantVillage | 24.0 | 25.2 | 28 / 2 / 8 of 38 | 28.9 |
| Medicinal Leaf | 70.8 | 57.9 | 18 / 6 / 1 of 25 | 96.0 |
| Rare Species | 70.2 | 76.7 | 356 / 42 / 2 of 400 | 97.3 |
| mean of nine | 54.8 | 53.4 | | |

Insects is the head's largest win (+17.9): 46 of its classes are families, and a family marginal is something
the head has and a text prompt of a family name is not. Camera Trap is its largest loss among covered tasks
(-5.4 with a 97.6 ceiling; species queries 54.3): night infrared frames of mammals, where the head's 5-shot is
still ahead (+2.2), so the prototypes, not the embedding, fall short there.

*Overlap (BS4).* Exact ids: Fungi 517 of 1,000 images from occurrences in ToL's training rows, PlantNet 27,
Insects 0. Hash audit over 11.0 M trained rows in the audited genera and families, each match confirmed by
thumbnail correlation (`verify`): the 6-bit rule alone flagged 239 Camera Trap and 100 AwA2 images, and by eye
every Camera Trap match and most AwA2 matches are different frames (night infrared, sea, plain backgrounds);
copies correlate 0.96 or more, the different frames 0.94 or less. Confirmed copies: **NeWT 3,278 of the 6,000
images of its inat_unobserved tasks** (the only NeWT tasks whose labels name a genus; the rest are not audited),
AwA2 9, PlantDoc 1, PlantNet 1 and Fungi 12 (all inside the id matches), Camera Trap, Insects, Insects 2,
PlantVillage 0. The hash audit is a lower bound on Meta-Album, whose 128 px crops hash differently (Fungi: 12
by hash against 517 by id).

*Does the overlap inflate ToL?* No measurable effect. Fungi on clean queries (483): ToL 84.6 and BioCLIP-2 86.1 at
5-shot, a gap of -1.5 against -2.2 on all; the flagged queries are harder for both models, not easier for ToL.
PlantNet clean: 94.3 against 94.9. NeWT's inat_unobserved test images: ToL 96.6 on the 1,364 clean, 96.9 on the
1,636 copies (trunk 96.1 / 97.0); BioCLIP-2 97.5 / 97.2. Whether BioCLIP-2's own training set holds the same
photographs was not audited.

*Predictions scored.*
- (a) Reproduction: **held** for 5-shot, zero-shot, NeWT (0.2), AwA2 (1.9, with the 40/10 files) and PlantDoc
  (0.9); **missed** for 1-shot Fungi (4.2) and, by 0.01, 1-shot PlantNet.
- (b) Per task at 5-shot: held on Camera Trap, Insects 2 and Rare Species; missed on Fungi (-2.2, predicted +2 to
  +6), PlantNet (-0.7), Insects (-2.1, narrowly), and the wrong way on Plankton (+6.8, predicted -5 to +1),
  PlantVillage (+4.0, predicted -8 to 0) and Medicinal Leaf (+1.1, predicted -3 to 0). The means held: +1.1 at
  5-shot (within 2), +0.6 at 1-shot (0 to 3). My per-task picture was backwards: I expected ToL to win where
  its taxa are well sampled (fungi, plants) and lose on non-taxon images; it loses slightly on the well-sampled
  taxa and wins on plankton, crop leaves and lab leaves.
- (c) Non-species: NeWT 2.1 below **held**; AwA2 3.5 below with the features at BioCLIP-2's norm **held** (2.3 for
  the trunk; with the code as is the comparison is not meaningful); PlantDoc **wrong way** (+5.8 as run, +7.7 by
  name).
- (d) Layer: **missed** on size. The trunk's features beat the head's embedding by 0.7 on NeWT, 1.2 on AwA2, 0.2 on
  PlantVillage and 1.1 on Plankton, and lose by 0.2 on PlantDoc; on the taxon tasks they lose by 0.3 to 1.4. The
  head costs the non-species tasks about a point, not 2 to 6.
- (e) Zero-shot: **held** on Fungi (+2.6) and Medicinal Leaf (+12.9), and on coverage limiting Insects 2 and
  Plankton; **missed** on Rare Species (-6.5), Camera Trap (-5.4) and PlantNet (-0.9); PlantVillage's ceiling is
  28.9, not 37, and the head is 1.2 below the text.
- (f) Overlap: Fungi (52 % by id) and NeWT's audited tasks (55 %) far above the 1 to 20 % predicted; Insects 0,
  not 1 to 20; the rest under 1 % as predicted. Removing it moves ToL's gap to BioCLIP-2 by under 1 point
  (Fungi +0.7, NeWT 0.3 the other way): **held**.

**What C2 says.** On BioCLIP-2's own suite and code, one frozen trunk trained only on species labels matches
BioCLIP-2 on few-shot species recognition (+1.1 at 5-shot, +0.6 at 1-shot over nine tasks, ahead on five) and
its head matches BioCLIP-2's text zero-shot on average (54.8 against 53.4) without a text encoder, winning
where coarse taxa are the classes (Insects) and losing where the vocabulary is common names or night camera
traps. It is 1.4 to 3.5 points behind on the two tasks that ask for traits rather than taxa (NeWT, AwA2), and
ahead on plant disease (PlantDoc, PlantVillage). The overlap with ToL's training rows is large on Fungi and NeWT
but does not explain any of it. Two protocol facts for the paper: BioCLIP-2's AwA2 recipe only works at its
own feature scale, and its PlantDoc code mislabels 8.5 % of the test images.
