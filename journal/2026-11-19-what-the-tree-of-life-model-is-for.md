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

**The s3 decision rule (supervisor, 2026-11-21 18:00, before any d5 or d10 number).** TL10 became a WSD run (mt's plan,
`plans/wsd/README.md`): stable segments s1 (passes 1-5) and s2 (6-10) are done, and the decay branches d5 (5 + 1 pass)
and d10 (10 + 2 passes) each get TL8a-v2 and the standard eval. The owner delegated the call on s3 (passes 11-20 with a
4-pass decay, about 212 B200-hours): "take the right decisions depending on perfs". The rule follows the owner's goal for
TL10 (a frozen trunk as close as possible to Otto on the test fold) and the 0.3 threshold written above:
- Queue s3 if TL8a-v2 on d10 beats TL8a-v2 on d5 by **0.3 points or more** on the test fold at zero threshold, and
  d10's Rare Species and held-out 5-shot are not more than 1.5 points below d5's.
- Otherwise stop: TL10 is d10, and the next generalist changes something other than length.
- Validation species macro-F1 alone does not decide: a gain there without a TL8a-v2 gain is not the goal.

Both branches decay for the same share of their length (one sixth), so d10 against d5 measures one doubling of
compute, and s3 would be the next doubling (12 to 24 passes), expected to return less than this one.

Predictions: validation species macro-F1 d5 0.64-0.66, d10 0.68-0.71; TL8a-v2 test fold d5 0.895-0.899, d10
0.900-0.907, so the rule most likely queues s3, by a small margin; trap probe (calibrated, list, tta3bm) d5 0.64-0.66,
d10 0.65-0.68.

**d5 scored (mt, 2026-11-21 18:00; job 12472455, DDP equality OK).** Validation species macro-F1 **0.603** (tol-full
0.640): my 0.64-0.66 is wrong, below the range. Rare Species 1- / 5-shot 63.2 / 77.5 (tol-full 62.9 / 79.1), held-out
68.8 / 84.9 (68.9 / 86.3), placement genus / family / order 66.3 / 59.8 / 49.3 (67.1 / 59.9 / 45.6), open-set max
softmax near / mid / far 0.772 / 0.823 / 0.851 (0.788 / 0.854 / 0.883). So at slightly more compute (6 passes against
5), WSD with a one-sixth cooldown lands below tol-full's one-cycle cosine: -3.7 on validation, -1.6 and -1.4 at 5-shot,
level at 1-shot. The literature's rule of thumb (a cooldown of about a fifth matches cosine) does not hold here, or not at
one pass. TL8a-v2 on d5 is running.

**Rule amended (supervisor, 2026-11-21 19:00, after d5, before any d10 number).** s3 keeps the one-sixth cooldown share
(4 of 24 passes), the share that just lost to cosine, so 212 B200-hours on it need a guard. If d10's validation species
macro-F1 is below tol-full's 0.640, or its Rare Species or held-out 5-shot below tol-full's 79.1 / 86.3, the cooldown is
the suspect and s3 is not queued; instead one branch from e20 with a 4-pass cooldown (10 + 4 passes, about 85
B200-hours) tests the cooldown length, and s3 waits for it. Otherwise the rule above applies unchanged.

**The stable phase was flat (owner spotted it, 2026-11-21 18:50; `tmp-runs/tol_v3/logs/summary.csv`).** Validation species
macro-F1 reached 0.50 by pass 2.5 and stayed at 0.498-0.503 from virtual epoch 5 to 19 (passes 3-10); training loss moved
from 2.88 to 2.76 on matched half-passes, validation loss from 2.943 to 2.853 and validation top-1 from 69.8 to 70.5 (mt):
a small residual slope in loss and accuracy, none in macro-F1. d10 against d5 measures what that slope is worth once decayed. The config differs from tol-full only in the schedule, so this is the schedule:
8e-4 was tol-full's one-cycle peak, held briefly there, and held for 10 passes here it pins the model to its noise floor
(in tol-full, lowering the rate from 8e-4 to 5.4e-4 alone was worth +3.7 by epoch 4). **s3 as designed is off whatever d10
shows** (it would hold 8e-4 for 10 more passes); this supersedes the rule above. If d10 lands at or below about 0.64, the next
generalist is the 10-pass one-cycle run first approved. A 1-pass cooldown from e20 (separating stable passes from cooldown
length) waits for d10 (owner). Prediction for d10's final validation species macro-F1: 0.615-0.635, between d5 and tol-full.

**d10 scored (mt, 2026-11-21 23:31; job 12472710, DDP equality OK).** Validation species macro-F1 **0.618**, top-1 80.0,
loss 1.817 (d5 0.603 / 78.8 / 1.932; tol-full 0.640 / 80.7 / 1.879). Predictions: the pre-run 0.68-0.71 is wrong; the 18:50
revision 0.615-0.635 is right. Passes 6-10 plus a cooldown twice as long bought +1.5 over d5, and d10 stays 2.2 under
tol-full's 5-pass cosine. But d10's validation **loss is below tol-full's** (1.817 against 1.879) while its top-1 is 0.7 lower
and its macro-F1 2.2 lower: d10 fits the common species as well or better and loses on the tail. That points at the rare
species' head rows, which get a few updates per pass and so need passes at a low rate (tol-full spent about 2.5 passes below
4e-4; d10 about 1). The 4-pass cooldown from e20 tests exactly that, one factor against d10.
d10's generalist suite (mt, TL1/TL2 protocol; d10 / d5 / tol-full): Rare Species 1-shot 63.7 / 63.2 / 62.9, 5-shot
77.8 / 77.5 / 79.1; held-out 1-shot 69.8 / 68.8 / 68.9, 5-shot 85.3 / 84.9 / 86.3; placement genus / family / order 67.3 / 60.8 /
49.6 (tol-full 67.1 / 59.9 / 45.6); open-set max softmax 1-2 points under tol-full. The embedding is at least tol-full's
(1-shot and placement above it); what lags is what depends on the head and the tail (macro-F1, open set) and 5-shot.

**TL8a-v2 on d5 scored (mt, 2026-11-22; e-tl8a2d5 12472992, trunk check 342/342).** Test fold species macro-F1, zero /
calibrated: **0.893 / 0.883** against TL8a on tol-full 0.895 / 0.884 (prediction 0.895-0.899: just under). Trap probe, list,
tta3bm, calibrated / zero: **0.664 / 0.463** against 0.647 / 0.432 (prediction 0.64-0.66: just over); probe-HO 0.471 / 0.274
(0.482 / 0.267); single view 0.578 / 0.330 (0.557 / 0.330). So for the owner's goal (a frozen trunk close to Otto), d5's
trunk is level with tol-full's on the test fold and 1.7 points better on the trap probe, although its own ToL head is 3.7
points worse on validation macro-F1. The WSD deficit sits in the ToL head's tail, which TL8a-v2 replaces; the trunk is not
worse. That lowers the stake of a longer cooldown for the frozen-trunk claims (it would mostly repair the ToL head).

**TL8a-v2 on d10 scored (mt, 2026-11-22 06:45; all TL10 pieces done).** Test fold zero / calibrated **0.897 / 0.886**
(d5 0.893 / 0.883, TL8a 0.895 / 0.884; OT3 0.916, OT4 0.919); trap probe tta3bm calibrated **0.667** (d5 0.664, TL8a 0.647),
probe-HO 0.486 (0.471, 0.482). Predictions: test fold 0.900-0.907 wrong (under); trap probe 0.65-0.68 right. d10 against d5
(one doubling of length): +0.4 test fold, +0.3 probe, +1.5 validation, +0.3 / +0.4 at 5-shot. The withdrawn s3 rule would have
queued s3 (+0.41); it stays off. **Answer to this section's question:** against tol-full, ten passes buy +0.2 on the test fold,
under the 0.3 written before TL10 ran, and the gap to Otto is still 1.9-2.2 points. At about +0.4 per doubling, length cannot
close it: **length is not the lever for the frozen-trunk goal.** The next generalist changes something else, one factor at a
time: model size (the model underfits, train and validation top-1 within 3 points) or data (species floor, coarse-only labels);
not resolution (owner: the data stay at 256 px).

**Next (owner, 2026-11-23).** Queued through mt: the training-set check without augmentation by species-frequency bin
(tol-full, d10) and the overfit test on 1 % of tol256 (augmentation off, one-cycle cosine). The 4-pass cooldown from e20 is not
queued. The owner is reconsidering constant learning rates; my view: the next generalist uses tol-full's one-cycle cosine at
its chosen length, so it changes one factor against a known reference, and branching stays an exploration tool (lower
held rate, cooldown counted in passes), not the schedule of a model we release.

**The two checks: design (mt) and predictions (supervisor, 2026-11-23, before any number).** (1) Training-set check:
up to 10 training and 10 validation images per species (1.85 M and 1.01 M images), eval transform, tol-full and d10, top-1 and
mean per-species recall by training-frequency bin (<100, 100-200, 200-500, 500-1000, >=1000). mt's smoke already hints: tol-full
gets 0.657 top-1 on a species-balanced training sample, against 83 % on the natural training mix, so the rare species are not
fitted even on their own training images. Predictions: per-species recall train minus validation 8-15 points in the <100 bin and
under 3 at >=1000; training recall under 70 % in the <100 bin; d10's deficit against tol-full concentrated in the <100 bin (2-4
points there, under 1 at >=1000). (2) Overfit test: 1 % of set 2 (546,436 images, 112,716 species), tol-full's recipe with
augmentation off (dropout 0.1 and weight decay 0.01 kept, so augmentation is the only change), batch 768 and lr 8e-4 on one B200,
30 epochs, one-cycle cosine, about 8 B200-h. Predictions: training top-1 in eval mode reaches 95-99 % (not 100: duplicates with
conflicting labels and the clamped cosine head); validation macro-F1 on the subset's species peaks before epoch 15 and then
falls. If training top-1 stalls under 90 %, the recipe caps fitting (dropout, weight decay, the head) before the data do.

**Can the generalist match OT6 on its own ground? (owner, 2026-11-23).** OT6 (Otto's recipe without pseudo-labels,
BioCLIP-2) scores 0.920 / 0.910 (zero / calibrated) on the test fold. Its config against TL10's: the same resolution (256 read,
224 crop), loss and head type, **no class oversampling in either** (`class_sampling_power: 0`); OT6 adds TrivialAugmentWide and
dihedral views, fine-tunes at 1e-5 from BioCLIP-2 (whose pretraining saw about two-thirds of our test fold, with labels), on 12 k
classes. Filtering ToL's own head to the Lepidoptera species was measured as TL3: 0.875. Two jobs, in parallel (mt): (1)
filtering plus a post-hoc prior correction (logits minus tau log prior, tau chosen on fold 1), on tol-full's and d10's ToL heads
and on the TL8a-v2 heads; (2) TL8a-v2 on d10 with square-root class oversampling, one factor against 0.897. Predictions (before
any number): plain filtering of d10's head under tol-full's 0.875; the prior correction adds 1-2 points to a filtered ToL head and
0.3-1.0 to a TL8a-v2 head; the oversampled head reaches 0.902-0.912. None reaches 0.920 alone: the gap closes to about 1 point at
best, and a stronger tail in the generalist itself (sqrt oversampling in the ToL run, one factor) is the next lever.
Design note (mt, supervisor's call): mini_trainer refuses class-weighted sampling on shard sets, and TL8a trains on shards,
so the oversampled head runs on the folder path (WeightedEpochSampler, count^-0.5, same draws per epoch, with replacement), with a
control on the same path at power 0: sqrt minus control is the sampling effect, control minus tl8a2d10 the loader's. Weighted
sampling in the shard loader (half a day) waits for the owner's choice of the next ToL run, which also trains on shards.
e-prior (job 12474204) validated: at tau 0 it reproduces TL8a-v2 d10's 0.8971 / 0.8862 exactly.

**Check 1 scored (mt, 2026-11-23; `plans/fit/README.md`, `/data/au761367/lepinet/fit/c1_report.json`).** Mean
per-species recall, train / val / paired gap, tol-full: <100 46.8 / 44.4 / +8.8; 100-200 75.5 / 68.7 / +7.0; 200-500 82.6 / 77.5 /
+5.1; 500-1000 87.3 / 83.5 / +3.8; >=1000 90.4 / 87.9 / +2.5. d10 minus tol-full, train / val: <100 -4.9 / -3.5, at most -0.9
elsewhere. Predictions: gap 8-15 at <100 and under 3 at >=1000, right (+8.8, +2.5); training recall under 70 % at <100, right
(46.8); d10's deficit concentrated at <100 (2-4 there, under 1 above), right (-3.5 val). **Reading: the generalist underfits the
tail.** Species under 100 training images are 103 k of 190 k, so they carry species macro-F1, and the model names under half of
their own training images. Regularisation is not the lever (the gap is small); what gives the tail more fitting is: sqrt
oversampling (more draws of rare species), more passes at a low rate (what d10 lacked against tol-full), and capacity. Whether
the recipe can memorise at all is the overfit test's question.

**Check 3 scored (mt, e-prior 12474204, about 2 B200-h; `ot_scores.md` "Check 3").** Tau chosen on fold 1: tol-full
ToL head 0.25, d10 ToL head 0, TL8a-v2 heads 0.25. Test fold line 1, zero / calibrated, plain then corrected: tol-full ToL head
0.875 / 0.869 -> 0.877 / 0.874; d10 ToL head 0.874 / 0.869 (tau 0); TL8a-v2 d5 0.893 / 0.883 -> 0.900 / 0.892; d10 0.897 / 0.886
-> **0.903 / 0.896**; OT6 (line 1, as the sheet) 0.917 / 0.912. Best generalist route: 1.4 / 1.6 under OT6, the prior closing
30-40 % of the gap. Traps: the Global Lepidoptera frequency prior does not help (calibrated -0.5 to +0.1), as expected: it is
not the trap prior. d10's own ToL head scores 0.628 on the trap probe against tol-full's 0.590, level with it on the test fold.
Predictions: plain d10 filtering under 0.875, right only just (0.874); the prior adds 1-2 to a filtered ToL head, **wrong** (+0.2
and 0); 0.3-1.0 to a TL8a-v2 head, right (+0.6, +0.7). Reading: the ToL head's tail is under-fitted (check 1), not biased by
its prior, so a post-hoc prior cannot repair it; a trained head's prior bias is real but small.

**The next ToL run: a decision rule to apply when the evidence lands (supervisor, 2026-11-23, owner: "keep it in mind,
we'll see tomorrow").** One factor against tol-full, on tol-full's cosine schedule. If the sqrt-oversampled TL8a-v2 head
gains 0.5 or more on the test fold over its control, and the label-shift test shows its trap deficit vanishing once class
frequencies are corrected: tol-full plus sqrt oversampling. Otherwise, if the ViT-L overfit test fits clearly faster than
ConvNeXt-L at matched epochs: ViT-L on tol-full's recipe. Otherwise: tol-full at 10 passes. Class-weighted sampling in mt's
shard loader is being built now (no GPU) so the first branch is not delayed. Start estimate: best case 2026-11-24 afternoon,
realistic 11-25, worst case 11-27 or the week after.

**Schedule test (owner, 2026-11-23; mt `plans/sched`, job 12475208).** EfficientNetV2-S BU on Global Lepidoptera, 5 epochs
at 256 px, from the shards, against the shard control v2s_bu_ctl_sh_s42 (cosine; equal to the staged seeds 42-44, 0.8866-0.8877,
which give the noise band): arm A poly 0.9 with the control's warmup (shape only), arm B poly 0.9 with no warmup (the owner's
nnU-Net-style package; B - A is the warmup). SGD dropped (owner): heavy-tailed classes favour Adam-type optimisers. Predictions
(before any number): A within 3x the seed range of cosine on test-fold species macro-F1 (|difference| under 0.3); B at or
below A, by 0 to 0.8, the cost, if any, from the randomly initialised head's first steps at full rate reaching the pretrained
backbone; per-bin, any difference concentrated in the rare bins.

**(C) scored: what sqrt oversampling does for a specialist (mt, no GPU; `ot_scores.md` (C), `/data/au761367/lepinet/bank/bank.json`).**
Three clean bank pairs, oversample_power 0.5 against 0, test fold line 1 zero / calibrated, trap single view without the list
(no logits kept): 20 M (L0 / singlehead) +1.9 / +2.9 on the test fold, calibrated probe -9.4, probe-HO -8.0; 198 M (L5 / F1)
+1.6 / +2.2, probe -4.6, probe-HO -4.8; 198 M with self-training (B7 / B6) +1.8 / +2.3, probe -0.3, probe-HO -2.4. So for a
specialist sqrt buys a steady +1.6 to +1.9 in distribution at every size, and costs the traps unless the model learns the
trap's class prior (self-training removes most of the cost): the label-prior reading again. **The fair specialist bar is
therefore about 0.933-0.936** (OT6's 0.917 plus that gain), not 0.917: with sqrt on both sides, the generalist route's gap is
nearer 2-3 points than 1.

**Check 2 scored: the recipe memorises completely (mt, fit-overfit 12474050; `/data/au761367/lepinet/fit/results/`).**
ConvNeXt-L, tol-full's recipe without augmentation, 546,436 images of 112,716 species, 30 epochs: eval-mode training top-1 on
the fixed 10 % 0.590 at epoch 5, 0.993 at 10, 0.9997 at 15, 1.0000 from 21; the whole subset at epoch 30 546,434 of 546,436
(100 % in every frequency bin; the two misses are cross-kingdom, an orchid named as a fish and a red alga as a brown alga,
label or image problems). Validation macro-F1 (3 images per subset species, not comparable to 0.640) 0.128 at 10, still
rising to 0.147 at 25, flat to 30, while validation loss was lowest at epoch 5. Predictions: 95-99 % wrong (100 %); a
validation peak before 15 then a fall wrong (still rising after memorisation). **Reading:** nothing in the recipe (cosine head
with its clamp, dropout 0.1, weight decay 0.01, lr 8e-4) caps fitting, so the tail at full scale (46.8 % of its own training
images, check 1) is underfit for lack of exposures, not for lack of a mechanism. Rough count: here an image needed about 10
exposures under a decaying rate to be fitted; tol-full gave each rare-species image 5, and sqrt oversampling would give about
15 (x3.1 draws). Whether capacity also binds at 61 M images is the ViT-L test's question (fit-vitl 12475364, started). And the
owner's point holds: validation kept improving after training reached 100 %, so the gap alone is no reason to stop.

**Check 4 scored: sqrt oversampling on the frozen d10 head, and its trap cost is label shift (mt; `ot_scores.md` Check 4).**
Test fold line 1 zero / calibrated: sq (power 0.5) **0.906 / 0.906**, its folder-path control 0.896 / 0.884, TL8a-v2 d10 (shards)
0.897 / 0.886: sampling +1.08 / +2.20, loader -0.16 / -0.18 (seed level). Trap, sq minus control, calibrated / zero, probe then
probe-HO: (a) as they are -0.86 / -2.66, -2.70 / -0.44; (b) sq's implicit prior (count^0.5) replaced by the natural one
-0.13 / -0.28, -1.69 / -0.36; (c) each head's own prior divided out, then the list and an EM trap prior +2.10 / +1.59,
+1.32 / +1.69 (absolute (c): sq 0.679 calibrated probe, 0.537 probe-HO). Prediction for the sq head 0.902-0.912: right (0.906).
**Reading: the owner's hypothesis holds.** Oversampling's trap cost is label shift: it vanishes on the probe once the training
prior is put back, and with the trap's own prior estimated the oversampled head is better under shift too, so its
representation is not damaged (the 2026-08 'trades shift robustness for accuracy' finding is superseded; its two supports had
already fallen). Against the decision rule above: test-fold gain +1.08 >= 0.5, and the trap deficit vanishes once class
frequencies are corrected: **the next ToL run is tol-full plus sqrt oversampling**, subject to the owner's review tomorrow with
the ViT-L test. Gap of the best generalist head to OT6: -1.1 zero, -0.6 calibrated; to a fair sqrt specialist about 2.7-3.

**ViT-L overfit test scored (mt, fit-vitl 12475364; `/data/au761367/lepinet/fit/results_vitl`).** DINOv3 ViT-L/16, same 1 %
subset and recipe, batch 640, lr 8e-4 (guard OK, no NaN). Eval-mode training top-1, ViT / ConvNeXt-L: e2 19.9 / 9.6, e4 50.2 /
35.5, e8 95.1 / 94.6, e10 98.4 / 99.3, 100 % from e24 / e21; whole subset at e30 546,435 / 546,434 of 546,436 (the one ViT miss
and one ConvNeXt miss look like one image under two labels). Validation macro-F1 at e30 **0.210 / 0.146**, top-1 26.1 / 18.8 %,
ahead in every bin (<100 11.7 / 8.1, >=1000 52.2 / 40.0); throughput 791 / 1,185 img/s (0.67x). Reading: both memorise; the
ViT generalises far better from about 5 images per species, which is the tail's regime (it measures pretrained features plus
fine-tuning from few examples, not capacity at 61 M images). The 5-point lr rule did not trigger. A 15-pass ViT-L ToL run would
cost about 480 B200-h (about 2.5 days on 8 B200).

**Label-shift test, test-fold half (mt, e-shift 12475365).** Test fold line 1 zero / calibrated, sq against control: (a) as
trained 0.906 / 0.906 against 0.896 / 0.885 (+1.0 / +2.2); (b) sq's prior reset to natural 0.905 / 0.896 (+0.9 / +1.1); (c)
own prior out and an EM prior estimated on the test fold 0.898 / 0.885 against 0.880 / 0.867 (+1.8 / +1.8; EM costs both heads
0.8-1.7 here, since that fold already follows the training prior: a trap-side correction only). So about half of sqrt's
calibrated test-fold gain is its implicit prior and the rest a better head (+0.9 / +1.1 with the prior removed), while on the
traps its deficit is the prior. Sqrt oversampling improves what the head learns, not only where it puts its prior mass.

**Schedule test scored (mt, SC1).** Test fold species zero / calibrated: cosine control 0.8880 / 0.8793; A poly with
warmup 0.8885 / 0.8816; B poly without warmup 0.8632 / 0.8507; seeds 42-44 0.8866-0.8877. Per-bin species F1 (<100 ... >=1000):
control 0.844 ... 0.944, A 0.848 ... 0.944, B 0.778 ... 0.940. Predictions: A within 0.3 of cosine, right (+0.05 / +0.23);
B at or below A by 0-0.8, wrong (-2.5, -6.6 in the rarest bin). B's calibrated trap probe rose to 0.665 (control 0.627, seeds
0.624-0.633): an under-fitted tail predicts fewer rare species, which helps on a catch of common species, the same label-shift
mechanism that made oversampling look harmful. Conclusion: keep cosine; warmup is mandatory (more so for a ViT).

**The next ToL run, agreed (owner, 2026-11-24): a bundle, not one factor.** ViT-L, sqrt oversampling, 15 passes, tol-full's
cosine with warmup (about 480 B200-h); the owner accepts that the bundle cannot be fully attributed afterwards. Augmentation is
tested first, universally (the generalist serves other uses than traps: no dihedral rotations, no trap-specific degradations):
ViT-L on the 1 % subset, 30 epochs, arms LIGHT (tol-full's augmentation), H1 (+ TrivialAugmentWide), H2 (H1 + random erasing +
Mixup/CutMix at timm's defaults, new opt-in code), against fit-vitl without augmentation (0.210). Predictions (before any number):
LIGHT above none (0.215-0.235), H1 above LIGHT by 1-3 points, H2 within 2 points of H1 either way at 30 epochs (mixing pays over
long training, so a narrow H2 loss would not rule it out at full scale). Label priors need no target labels: the list comes
from outside knowledge and the EM prior from unlabelled target images (averaged soft pseudo-labels), and the oversampled head
takes the EM prior better than its control (+2.1 probe).

**GPU idle at phase starts (mt's diagnosis, 2026-11-24).** On the 1 % runs the GPU is idle about a quarter of the time
(nvidia-smi dmon: fit-vitl 27.1 %, the three augmentation arms 23-27 %). The idle time comes as two stalls of about 2.5 min,
before the first batch of every training and every evaluation phase. Staging the shards to local disk changed nothing, so the
reading of 2026-11-23 (re-reads over the network; mt's, and I endorsed it) was wrong. The stall has two pieces of 60-90 s each.
- (A) Workers start one after another. With the spawn start method, each DataLoader worker receives the pickled ShardDataset
  (19.6 MiB, mostly the seven class maps of 190 k species) through a 64 KiB pipe, which the child drains only after importing
  torch. So the 32 workers start serially, about 2.7 s each. mt reproduced this locally: 4 workers take 6.07 s with a 20 MiB
  dataset, against 0.01 s with a 10 KiB one.
- (B) The summary.csv append on /work, in `MultiLogger.update`. Between the progress bar and that write, the code computes only
  running-sum means (I checked `mini_trainer/logging/core.py`). The wait is therefore in the file system; the mechanism is
  unknown.

Workers restart every phase by design, because they receive the phase plan. The bundle has two virtual epochs per pass, so
about 60 phase starts over 15 passes. At 8 ranks that is roughly 2-5 % of a 60-hour run (10-25 B200-h), depending on workers
per rank. Fixes applied (owner's go; mt, mini_trainer 77f8177..2a44941 on feature/lepinet-integration, 1,017 tests pass):
- (A) the dataset's large static state goes to a side file in the temporary directory. The pickle each worker
  receives is about 1 KiB, so the workers start in parallel. Locally, with 4 workers, iter() fell from 5.1 to 0.0 s.
- (B) summary.csv rows are appended in order by one background thread. finish() waits for them, and queued rows are
  written at exit, also after an error, but not after a SIGTERM or SIGKILL.
- An opt-in stack dump every 10 s until the first batch (`MT_PHASE_TRACEBACK_SECONDS=10`) will show in the smoke
  test what (B) was waiting on. On the node I expect a few seconds per phase start; to be confirmed there.

**First field check (mt, J1, 2026-11-24 22:36; mini_trainer 2a44941, fork workers, one process per GPU).** After the
first train-evaluation-train cycle, no GPU was idle for 30 s or more. Each phase start shows at most one or two 10 s dmon
samples below 25 % SM, so under about 20 s, against 180-200 s on polysh (same testbed, old code). The stack dumps never
fired, so no loader took over 10 s to its first batch. Attribution is open:
- Polysh also used fork workers, so fix (A) is not the difference.
- The whole wait vanished, not only (B)'s 60-80 s; the roughly 120 s nobody had explained went with it.
- J1 also ran on a different node, with three runs instead of two.
A free split, asked of mt: time the background summary.csv write on J1. If it still takes a minute, (B) moved the wait off
the training thread and earns the credit; if it takes about a second, the /work slowness was specific to that node or
its load. The DDP smoke test (383f978's worker lines) still has to show fix (A) under spawn.
Answer (mt, end of J1): no idle stretch of 30 s or more in the whole 4.7 h run, and the evaluation row's summary.csv append
finished within 6 s of the evaluation's end. So the write is fast on this node and fix (B) is not shown to be the cause.
Polysh's 180-200 s did not recur, and its cause remains unknown (that node, or /work's load at the time). The 8-GPU smoke
test is still the check that matters.

**Augmentation test scored (mt, fit-aug1/2/3, 2026-11-24; `/data/au761367/lepinet/fit/results_aug`, `plans/fit/README.md`).**
DINOv3 ViT-L/16 on the 1 % subset, 30 epochs, the same 293,880 validation images as fit-vitl (augmentation off at
validation). One seed per arm and no per-image validation predictions kept, so there are no confidence intervals and a
difference under about 0.01 F1 is not established. Validation species macro-F1 at epochs 10 / 20 / 30:

| arm | e10 | e20 | e30 | top-1 e30 | top-1 <100 / >=1000 | training top-1 (eval mode) |
|---|---|---|---|---|---|---|
| none (fit-vitl) | 0.179 | 0.196 | 0.210 | 26.1 | 11.7 / 52.2 | 1.000 at e18 |
| LIGHT (tol-full's) | 0.189 | 0.206 | **0.216** | 27.0 | 11.4 / 54.5 | 1.000 at e21 |
| H1 = LIGHT + TrivialAugmentWide | 0.183 | 0.203 | 0.211 | 26.5 | 10.7 / 55.1 | 1.000 at e24 |
| H2 = H1 + erasing + Mixup/CutMix | 0.128 | 0.197 | 0.212 | 27.2 | 11.6 / 55.3 | plateau 0.948 |

Every arm peaks at epochs 28-30. Throughput is within 1 % across arms (788-795 img/s).

Predictions scored:
- LIGHT above none, 0.215-0.235: right on the number (0.216), but the +0.006 is under the noise.
- H1 above LIGHT by 1-3 points: wrong (-0.5).
- H2 within 2 points of H1: right (+0.1).

Reading: universal augmentation buys nothing measurable on in-domain validation, and no arm moves the rarest bin (top-1
10.7-11.7). Is the 1 % proxy valid for a 15-pass full-scale run with sqrt oversampling? It is, for both halves of the
bundle's distribution:
- The rare half. Over 15 passes, sqrt oversampling shows each image of a species under 100 images about 48 times on
  average (76 at 20 images, 152 at 5; 57.2 M training rows, 190,166 species). That is the memorising regime of the 1 %
  runs, which see each image 30 times. There, augmentation did not help the tail.
- The common half. Species of 1,000 images or more get about 8 exposures per image, an under-fitting regime. In that
  regime AG measured TrivialAugmentWide at -1.7 points on the test fold.

TrivialAugmentWide's one measured benefit is shift robustness (AG, +8.6 on the trap probe), which in-domain validation
cannot see. Since AG, the trap deficit has been traced mostly to the label prior (TL11 check 4, Z23). TL8 already applies
TrivialAugmentWide when it trains a specialisation head.

Recommendation (supervisor, for the owner's decision): the bundle keeps LIGHT, tol-full's augmentation. That also takes
one factor out of the bundle. Drop Mixup/CutMix for three reasons:
- it is new code;
- it starts slowly (0.128 at e10);
- its mixed targets change what the logits mean, an untested effect on open-set scores.
The case for H1 is a generalist made robust to shifts nobody has named. It is untested at ToL scale, and AG's -1.7 is the
nearest estimate of its in-domain cost.

**Decided (owner, 2026-11-25): keep tol-full's augmentation.**

**What I got wrong here.** As supervisor I tracked job states and checkpoints through s1 and s2 (27 hours) and never read
the validation curve against tol-full's at matched epochs; the gap was visible at virtual epoch 4 (0.494 against 0.531),
about 5 hours into s1. For any long run, compare its curve with the reference run's after the first few evaluations.

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
