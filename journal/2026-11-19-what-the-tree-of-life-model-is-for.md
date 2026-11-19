# What is the Tree of Life model for?

**Kind:** research · **Status:** OPEN (2026-11-19). A step back on the ToL run and the project around it: cost
against BioCLIP-2 measured from its paper, the embedding claim measured (`dev/112`), the cosine head's behaviour at
190 k classes checked, and a path from the generalist to a model that beats Otto, for the owner to decide.

## Why this entry (owner, 2026-11-19)

The owner's critique, recorded because it is right: the TL series asked a narrow question, "is the ToL trunk a
better start for the moth recipe?" (TL3-TL5). I framed BioCLIP-2's contrastive recipe as the presumptive winner (my
TL1 prediction put the ToL model below it; falsified) and closed the TL line on "fine-tuning erases it, TL5 not worth
running". That reads the run as an ingredient. The run had three larger purposes:

1. **mini_trainer at scale, in practice**: one run, 60.9 M images, 190 k classes, nothing guaranteed it would work.
2. **The cosine-and-softmax head at 190 k classes, in theory**: does the z-scored cosine head stay sound there?
3. **One generalist model plus specialisation**: a trunk that is also an embedding model, with list, prior (EM),
   cRT, self-training and only then a little fine-tuning on top, eventually replacing per-domain specialists.

## 1. In practice: the run worked as built

tol-full (job 12464522): 5 passes over 60.9 M images on 8 B200 in 13.2 h, one attempt, no restart, weights equal on
the 8 ranks, GPU utilisation median 97 % (compute-bound, so the shard loader kept up at about 5,850 img/s), validation
species macro-F1 rising every half pass to 0.640. The lr was set by a range test checked on two datasets first (LF).
The same shards then served TL4 at 320 px on Lepidoptera, above VD in distribution. W3 holds the detail.

## 2. In theory: the cosine head at 190 k classes

mini_trainer's head (`Classifier.forward`): the embedding `h` is L2-normalised; each class has a direction `v_c` and a
learned magnitude `g_c` (weight norm) and bias `b_c`; logit `= sqrt(D-2) * asin(clamp(g_c cos(h, v_c), -1, 1)) + b_c`,
D = 1,536. Two consequences checked on tol_full's weights and 20,000 known-species test images:

- **The temperature is set by the dimension, and it is large enough.** Near zero the transform is
  `sqrt(D-2) cos`, an effective scale of 39.2, with logits bounded to +/-61.6. A normalised softmax over C classes needs
  a scale of order ln C to reach low loss (NormFace's bound); ln 190,199 = 12.2, so the head has about 3x headroom and
  would only run short near e^39 classes. Measured: train species loss 0.77, top-1 81 % on held-back known images.
- **The clamp is used, at the bottom.** `g_c` learned to 1.41 (median; 1st-99th percentile 1.32-1.97), so
  `g_c cos` leaves [-1, 1]: no top-1 ever reaches the top clamp (0 %), but 79 % of (image, class) pairs sit on the
  floor, every one with the same logit and no gradient. The median image has 750 classes off the floor (10th
  percentile 91; for at least 10 % of images all 190 k). The geometry is a cone and its opposite: 74 % of embedding
  coordinates are positive (the leaky ReLU before the normalisation), 97 % of prototype coordinates negative, median
  top cosine -0.47. Lepidoptera's plain cosine head showed the same floor in August (67 %,
  [[2026-08-06-the-cosine-head-is-not-unit-norm]]).
- **One defect follows, small and fixable.** For 0.88 % of known-species images the true class itself is on the
  floor, so cross-entropy cannot pull it up (all are misclassified). The fix keeps the scale and loses the dead zone:
  apply `g_c` after the transform (scale the z-score, not the cosine), or a straight-through clamp. It belongs in
  mini_trainer (mt / Asger) and in the next large run, not in a retrofit; no reported number depends on it.

So: sound in practice, sound in theory for 190 k classes, with one dead zone to remove before a larger run.

## 3. Cost against BioCLIP-2

BioCLIP-2 figures from its paper (arXiv 2505.23883, Table 3 and section 4); per-image compute from timm's model
cards (forward, 224 px). Both start from an expensive pretrained model that neither count includes.

| | BioCLIP-2 | ToL model (tol-full) |
|---|---|---|
| start | CLIP ViT-L/14, LAION-2B | DINOv3 ConvNeXt-L (LVD-1689M) |
| image tower | 304 M parameters, 77.8 GMACs | 198 M parameters, 34.4 GMACs |
| data | 214 M images, 952 k taxa, + 26 M LAION replay | 60.9 M images, 190,199 species (7 ranks) |
| epochs; images seen | 30; about 6.4 B (+ replay) | 5; 0.30 B |
| batch | about 90 k (contrastive needs in-batch negatives) | 768 |
| hardware | 32 H100 x 10 days = 7,680 H100-hours | 8 B200 x 13.2 h = 106 B200-hours, about 240 H100-hour equivalents |
| training compute, image tower, forward + backward | about 1.5e21 MAC | about 3.1e19 MAC (**about 50x less**) |

**Why it is cheaper.** 21x fewer images seen and a 2.3x cheaper trunk per image. A classifier is also a
proxy-based metric learner: every image is compared with all 190 k class prototypes at every step, for a 190 k x
1,536 product (0.29 GMACs, under 1 % of the trunk). A contrastive model only sees as many negatives as its batch, which
is why BioCLIP-2 needs batches of about 90 k and many epochs.

**What the money bought.** TL1 (same protocol for both): Rare Species 1/5-shot 62.9 / 79.1 against 60.4 / 77.0;
held-out taxa that BioCLIP-2 trained on and we did not 68.9 / 86.3 against 66.2 / 85.0; Z20, moths outside both
training sets, +1.9 on photos and +8.5 on trap crops with the list. So: about 2 % of the compute, level or above on
species-level tasks. Not measured: BioCLIP-2's other suite tasks (including non-species ones, where a classifier
should lose), text zero-shot (we have none: a new species enters by a few images, not by its name), one seed.

## 4. Is it an embedding model? Measured (`dev/112`, TL7)

The argument: with a normalised embedding and normalised prototypes, the head is nearest-prototype retrieval with
190 k learned proxies. The closed label set is the set of proxies, not a limit of the embedding: a new species is one
more prototype (the mean of a few images), with no retraining. TL1 is that test (1-shot 68.9 % on 1,810 species never
trained). What the owner asked next is whether the space is *structured*. Species means on the saved embeddings
(held-out taxa, 1,810 species; Rare Species, 400), centred and normalised:

| encoder | nearest species in the same genus (held-out / Rare) | same family | cosine of species pairs: same genus / same family, other genus / different class or above (held-out) |
|---|---|---|---|
| **ToL, frozen** | **84.5 / 81.1 %** | **74.8 / 80.5 %** | 0.51 / **0.10** / 0.00 |
| BioCLIP-2 | 79.3 / 78.3 % | 69.1 / 78.4 % | 0.62 / 0.22 / -0.02 |
| DINOv3, raw | 67.2 / 69.9 % | 71.0 / 80.5 % | 0.69 / 0.41 / -0.04 |
| TL4 (ToL, then moths) | 47.1 / 51.0 % | 44.3 / 45.7 % | 0.52 / 0.30 / -0.04 |
| VD (DINOv3, then moths) | 44.5 / 49.0 % | 41.7 / 46.1 % | 0.54 / 0.35 / -0.05 |

- **Finer structure: yes, the most of the five.** A species' nearest neighbour is a congener 84.5 % of the time on
  taxa it never trained on (+5.2 over BioCLIP-2), and a confamilial one 74.8 % (+5.7).
- **Coarse structure: no, the least.** Beyond the genus, species are near-orthogonal: same-family pairs 0.10
  against BioCLIP-2's 0.22 and raw DINOv3's 0.41. Rank correlation of cosine with tree distance over all pairs:
  ToL -0.16, BioCLIP-2 -0.25, raw DINOv3 -0.65 (dominated by kingdom-level visual similarity). A softmax over species
  pushes every non-congener apart; the families live in the head's marginals (placement at mid 59.9 %), not as
  clusters. Clustering families from the embedding alone, BioCLIP-2 would serve better.
- **Fine-tuning on moths flattens both** (TL4, VD: congener neighbours 45-51 %), as Z21 found for few-shot.
- **Claim we can make:** "a hierarchical classifier trained with a cosine head is an embedding model": level or
  above BioCLIP-2 on species-level retrieval and few-shot, with sharper local and weaker global taxonomic structure,
  at about 2 % of its training compute. Paper A's heads were a null on accuracy; whether they differ on *this*
  geometry (independent per-rank prototypes would put families in the space) is a free check on the campaign's F,
  BU, I and TD checkpoints.

## 5. The generalist, and how it could beat Otto

Today it does not: on north_europe probe crops (tta3bm, calibrated), Otto (OT3) 0.749, the ToL model's own moth head
0.590 (TL3), TL4 0.691. But Otto's margin decomposes into pieces that need not live in the trunk: the regional list,
the site prior (Z16: self-training on top of list and prior is worth 1.2-2.8 points), TrivialAugmentWide (+8 at zero
threshold), the BioCLIP-2 start (probe-HO). The bet is that a generalist trunk plus specialisation layers recovers
them all, and the generalist then carries every other taxon for free. A ladder, each step deciding the next, all
predictions written now:

| step | what | cost | prediction |
|---|---|---|---|
| G1 | specialisation only, trunk frozen: tol_full features of Global Lepidoptera cached once (224 px), a BU moth head trained on them (cRT-style), then list, estimated prior, and Otto's gated pseudo-labels in the head | about 1 B200-hour to cache, minutes per head | probe 0.64-0.70 calibrated with list and prior: above the 190 k head (0.590), below Otto; probe-HO near OT6 (0.557). Generality kept by construction |
| G2 | a little fine-tuning (route 5): G1's head as the start, trunk at a tenth of the head's rate, TrivialAugmentWide, gated labels | 1 run, about 14-28 B200-hours | probe within 2 points of Otto (0.73-0.77), probe-HO above it; Rare Species 5-shot stays above 60 (TL4 kept 31). Both together = the generalist path matches the specialist |
| G3 | the leap, a second generalist (ToL v2): the trap domain inside pretraining (the trap crops we hold and the owner's trap order dataset, with coarse labels only), the under-floor species and ToL's 2.6 M genus-rank records with coarse labels, TrivialAugmentWide, 320 px, the clamp fixed | 1 run, about 300 B200-hours | its frozen moth head (G1 on v2) reaches Otto on probe without per-region training; held-out few-shot not below tol-full |

One code change unlocks three of G3's data sources: **a per-rank loss mask** (an image supervises only the ranks it
is labelled at). It is also what the species-floor entry recommends ([[2026-11-17-should-the-species-floor-go-now-that-coarse-ranks-are-supervised]])
and what W3 noted for the genus-rank records. G3 is the scientific leap; G1 is nearly free and says how far
specialisation alone goes; G2 tests the owner's "only maybe a bit of fine-tuning". The one-run rule holds: one
seed, one run per configuration. A third paper becomes possible if G1-G2 hold: the ToL model as a biodiversity
embedding model, needing BioCLIP-2's full suite first (evaluation only).

## What I got wrong

My TL1 prediction (below BioCLIP-2) and my framing after TL4 both treated the contrastive recipe as the reference
and the ToL model as an ingredient to be tested against our moth specialist. The measurements say the opposite
reading is open: the classification run is the cheaper and stronger embedding, and the specialist is what should
be rebuilt on top of it. Owner's decision on G1-G3.
