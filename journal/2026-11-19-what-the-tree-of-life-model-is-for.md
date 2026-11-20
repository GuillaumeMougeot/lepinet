# What is the Tree of Life model for?

**Kind:** research · **Status:** OPEN (2026-11-19). A step back on the ToL run and the project around it: cost
against BioCLIP-2 measured from its paper, the embedding claim measured (`dev/112`), the cosine head's behaviour at
190 k classes checked, and a path from the generalist to a model that beats Otto (TL8 approved), and how long the next ToL run should be.

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
  floor, so cross-entropy cannot pull it up (all are misclassified; 4.7 % of top-1 errors). The mt session found the
  same at Lepidoptera scale on 2026-11-03 (0.09-0.33 % of images) and deferred the fix until after the paper; it
  grows with the class count. The fix keeps the scale and loses the dead zone: apply `g_c` after the transform. The
  plain explanation, the figure and its test (CF) are in [[2026-11-19-does-the-cosine-heads-clamp-floor-cost-accuracy]].

So: sound in practice, sound in theory for 190 k classes, with one dead zone to test a fix for before a larger run.

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

(The ladder was first written as G1-G3; those IDs are taken in RESULTS, so it is TL8-TL10.)

| step | what | cost | prediction |
|---|---|---|---|
| TL8 | specialisation only, trunk frozen. **Approved 2026-11-19, built by mt.** TL8a: OT4's recipe (Otto's, on DINOv3 ConvNeXt-L BU, TrivialAugmentWide, 10 epochs) with tol_full's trunk frozen, 224 px, no pseudo-labels; TL8b: TL8a plus OT3's gated labels. Scored with list and estimated prior (lepinet-2a) | two 1-GPU runs on the mini_trainer shards (owner's choice, 2026-11-19: TL8a at once, TL8b once the gated-label merge is in the shard loader, so the two differ by the labels only); the trunk runs forward only, so well under OT4's training time; first sketched as cached features, which would drop the augmentation | TL8b probe 0.64-0.70 calibrated with list and prior: above the 190 k head (0.590), below Otto (0.749); probe-HO near OT6 (0.557). Rare Species unchanged from tol_full (the trunk is frozen; a check that it is) |
| TL9 | a little fine-tuning (route 5): TL8's head as the start, trunk at a tenth of the head's rate, TrivialAugmentWide, gated labels | 1 run, about 14-28 B200-hours | probe within 2 points of Otto (0.73-0.77), probe-HO above it; Rare Species 5-shot stays above 60 (TL4 kept 31). Both together = the generalist path matches the specialist |
| TL10 | a second generalist (ToL v2), section 6 | 1 run, 200-320 B200-hours | its frozen moth head (TL8 on v2) closer to Otto than TL8; held-out few-shot not below tol-full |

**TL8 scored (2026-11-20; mt's score sheet and lepinet-2a's Z23).** Calibrated species macro-F1, tta3bm, regional
list, probe / probe-HO: TL8a 0.647 / 0.482 (test fold 0.895), TL8b 0.696 / 0.429 (test fold 0.895, equal to TL8a; OT3 0.916), against OT6
0.700 / 0.557, OT3 0.749 / 0.562 and OT4 0.760 / 0.539. My prediction: TL8b 0.64-0.70 with the list held, at the top
of the range; with an estimated prior it is falsified (0.576-0.613): an estimated prior harms any head trained on
trap labels (Z23), so it is for models that never saw a trap. Probe-HO near 0.557 is falsified (0.429): the gated
labels cost the frozen trunk 5.3 points there, where they cost OT3 nothing. **A frozen generalist plus a head does
not reach Otto** (TL8b 5.3 under OT3, 6.4 under OT4), so TL9 is the next rung. What the frozen trunk does well: TL8a
tells trap species apart pairwise as well as the fine-tuned 320 px models (one-view pairwise error 0.00039 against
VD's 0.00045-0.00058), and TL8b ranks its own answers best of any model measured (AUROC right against wrong 0.935
with tta3bm). Its deficit is class bias and confidence, not discrimination. Both trunks are unchanged (342 of 342
tensors), so their general embedding is tol_full's. One seed each; 224 px against 320 px for OT4 and VD.

**Owner, 2026-11-19: no AMI trap data in the ToL training set yet, and no 320 px**: ToL is stored at 256 px (short
side), so 320 would only upsample, and most trap crops are smaller than that too (probe crops: median side 178 px,
10th percentile 124, 90th 268, 96 % under 320). TL10 stays at read 256 / crop 224. My first TL10 sketch had both; it
is corrected here.

One code change unlocks two of TL10's possible data sources: **a per-rank loss mask** (an image supervises only the
ranks it is labelled at), for the under-floor species and ToL's 2.6 M genus-rank records. It is what the species-floor
entry recommends ([[2026-11-17-should-the-species-floor-go-now-that-coarse-ranks-are-supervised]]). TL8 is nearly
free and says how far specialisation alone goes; TL9 tests the owner's "only maybe a bit of fine-tuning". A third
paper becomes possible if TL8-TL9 hold: the ToL model as a biodiversity embedding model, needing BioCLIP-2's full
suite first (evaluation only).

## 6. A longer ToL run: how long, and when (owner, 2026-11-19)

**Not 30 epochs.** BioCLIP-2's 30 is a contrastive model's count; section 3's argument (every image meets every
prototype at every step) suggests a proxy classifier needs fewer passes. The closest evidence we own is Lepidoptera, with a similar density (about 400 training images per
species against ToL's 320): V2-S BU went up from 5 epochs to 10 (+2.1 to +3.4 test, confounded with 256 to 384 px)
and *down* from 10 to 30 (test 0.9116 to 0.9036, 7x the seed range; probe 0.418 to 0.409, probe-HO 0.326 to 0.333;
RESULTS VL). tol-full's own signals agree that 5 passes is short of the optimum but not far: validation was still
rising when the one-cycle rate reached zero, while train loss had dropped below validation from epoch 7, and the
congener-neighbour rate fell from 90 to 78 % as few-shot rose.

**Proposal: 10 passes, one factor against tol-full**, judged on TL1 (few-shot), TL2 (open set by depth), Z20 and
TL8 on the new trunk, not on validation loss (lepinet-2a's advice in W3). About 26 h on 8 B200, about 212 B200-hours.
A cheaper way to learn the whole curve in one run, if mt can add it: a warmup-stable-decay schedule with short
cooldown branches at 5, 10 and 15 passes (each about 10-15 % of its length), so the run itself answers "how many
passes" and every branch is a usable model.

**What goes in, and what does not.** Changing one factor keeps the result readable; the project's +4.7 lesson
(three changes, one jump, never attributed) applies to a 200 B200-hour run more than anywhere. In: length, and the
clamp fix only if CF passes (it would then be a correction measured on its own; **it did not**, 2026-11-20, so the current head stays). Out until tested separately:
coarse-only labels for under-floor species (needs the loss mask, then its own run), MixRes (section 7), trap data
(owner).

**When.** Ready when (1) CF has scored (code plus one V2-S run, about a day once built), (2) TL8 has scored (it is
the yardstick for v2's frozen moth head; about two days with staging), and (3) mt has the schedule (one-cycle at 10
passes needs nothing new; the branched schedule needs code and a resume test). Realistically the start of next week.

**Decided (owner, 2026-11-20).** TL9 on hold; TL10 first. The owner's goal for it: a frozen trunk as close as possible
to Otto on the test fold, towards "one model to rule them all": a very good frozen generalist plus specialisation
techniques that bridge an embedding model and a real deployment (domain shift, label shift, unknown objects). CF and
TL8 have scored, so the conditions above are met except the branched schedule, which is dropped to start sooner:
TL10 is tol-full with one change, 10 passes instead of 5, one-cycle, everything else equal (current head, 256/224, no
trap data, one seed). Its pass-5 checkpoint is mid-schedule (learning rate not annealed): a progress diagnostic, not a
matched comparison with tol-full. Yardstick: TL8a's exact recipe on the frozen v2 trunk (TL8a-v2), plus the
generalist suite (TL1 few-shot, TL2 open-set, Z20 moth retrieval, TL7 structure).

**Predictions for TL10 (supervisor, 2026-11-20, before the run).**
- Validation species macro-F1 0.68-0.71 (tol-full 0.640).
- TL8a-v2 test fold 0.900-0.907 at zero threshold, line 1 (TL8a 0.895; OT3 0.916, VD 0.912): the gap to Otto shrinks
  by a quarter to a half but does not close. Basis: Lepidoptera gained 2-3 points from 5 to 10 epochs (confounded
  with resolution) and nothing beyond; a frozen trunk passes on only part of a better trunk to one group.
- TL8a-v2 trap probe, calibrated with the list, tta3bm: 0.65-0.68 (TL8a 0.647).
- Generality kept: Rare Species and held-out 5-shot within 1.5 points of tol-full's 79.1 and 86.3, either way.
- If TL8a-v2 gains under 0.3 on the test fold, length is not the lever for the frozen-trunk goal, and the next
  generalist changes something else (data, resolution or the species floor), each tested on its own.

**Beyond 10 passes, and the schedule (owner's questions, 2026-11-20).** TL10 started at 09:51 (job 12471243), so
the choice for now is between letting it run and restarting it longer. tol-full was far from memorising its data:
at the end, training top-1 83.3 % against validation 80.7 %, and validation species macro-F1 still rising (0.590,
0.619, 0.634, 0.640 over the last four half-passes, the last steps mostly from the annealing). The Lepidoptera case
where 30 epochs did worse than 10 was a model that had fitted its training set. So more than 10 passes may well pay
on ToL, but 10 first measures the slope from 5 to 10 at half the cost of 20, and the slope decides. A small
schedule test first would not help: a short-horizon comparison does not predict the converged outcome (CLAUDE.md,
the margin lesson). The schedule matters for the run after TL10: a finished one-cycle run cannot be extended
(re-warming tol_full's head cost 2.76 points in cftol), while warmup-stable-decay can continue its stable phase and
branch cooldowns at several lengths. mt prepares it as opt-in code during TL10 (no GPU job); TL10, one-cycle at 10
passes, is the reference a WSD run at 10 passes must match.

## 7. Resolution augmentation (MixRes), owner's question

Jain et al. (ECCV 2024, the AMI dataset) downscale training images to 25 or 50 % of the input size (probability 0.25
each) and scale them back up, because many trap crops are under 64 px. Their ablation (ResNet50, NE America, 128 px
input, micro top-1): +7.0 on AMI-Traps alone, +4.0 on top of RandAugment and random-resized crop, for -0.3 to -1.8
in distribution; and training at 224 instead of 128 gained 3.6 in distribution but lost 3.5 on traps.

What we know: our trap augmentation (motion blur, low light, JPEG) had no resolution change, nor does
TrivialAugmentWide, so the factor is untested here. Our crops are larger than theirs (none under 64 px, median 178),
and we train at 320 (VD, OT4) or 224 (Otto), so a matched version would downscale to roughly 40-100 % of the input
instead of 25 / 50 %. Their 224-against-128 result is the train-test resolution mismatch: a trap crop upscaled to the
input looks softer than any training photo. Penalty: their in-distribution cost (up to 1.8) is the size of our trap
augmentation's (0.75), which our pre-registered rule turned off, so for the paper recipe it would need the same rule;
for Otto, whose goal is traps, a small in-distribution cost is acceptable if probe and probe-HO rise.

Plan: first lepinet-2a's CPU check (requested 2026-11-19): is the trap deficit concentrated on small crops, and does
training resolution interact with crop size? If yes, one moth run, **MR**: OT3's recipe plus the matched resolution
augmentation, one factor, scored by crop-size quintile. Into ToL only after MR is positive, because the one-run
rule allows no ToL ablation and the generalist's few-shot benchmarks are high-resolution photos that it could cost.

**Check done (Z22, lepinet-2a, 2026-11-19): no on both counts for Otto.** The small-crop deficit is a domain gap, not a
resolution one. Models never trained on trap crops lose 13-18 points on the smallest fifth of crops with one view;
the self-trained OT3 and OT4 lose nothing (-0.7, +2.3), and three-view TTA shrinks the others' loss to 2-3 points.
Training resolution does not interact with crop size (224 and 320 lose alike). So MR is not run for Otto, which has
nothing to recover. It could still matter on the generalist path (TL8a, the frozen ToL head), which sees no trap
crop; before any run, an eval-only test (largest crops downscaled to the smallest's size and back, scored by a
never-trap model) separates resolution from framing and posture. Details in the beyond-scaling entry, Z22.

## What I got wrong

My TL1 prediction (below BioCLIP-2) and my framing after TL4 both treated the contrastive recipe as the reference
and the ToL model as an ingredient to be tested against our moth specialist. The measurements say the opposite
reading is open: the classification run is the cheaper and stronger embedding, and the specialist is what should
be rebuilt on top of it. TL8 approved 2026-11-19; TL9, TL10 and MR are the owner's decisions.
