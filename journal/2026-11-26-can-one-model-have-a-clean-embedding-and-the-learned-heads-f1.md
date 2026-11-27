# Can one model have a clean embedding and the learned head's F1?

**Kind:** research · **Status:** RESOLVED (2026-11-26): mostly yes. Without the leaky ReLU before the normalisation, the
learned head (J4b) has the best species F1 (0.903, +0.7 over the learned head with the ReLU, +2.3 over the frozen),
prototype rows (+0.70) and the best open set (0.933), and recovers 40-100 % of its embedding gap to the frozen head;
a gap remains on placing new taxa in their family (-0.7 new species, -1.9 new genera). Reading f (J4f) does not help;
the frozen head without the ReLU (J4a) buys the same F1 but spends its embedding. Owner decides the ToL head.

## The question

The owner's goal for the next Tree of Life run's head (2026-11-26, the deadline for that run lifted to make room for
this): "the best of the two worlds: a very clean embedding space and a very good F1 score". J1 to J3
([[2026-11-24-should-the-cosine-heads-scale-and-bias-be-learned]]) left a trade-off on V2-S with sqrt oversampling:
- the learned head (per-species scale and bias) has the best species F1 (0.896; 0.885 under 100 images), but an
  embedding that is worse at placing new species and genera near their relatives (-1.3 to -3.3 points against the
  frozen head) and rows that point away from the images (open set by maximum cosine at chance);
- the frozen head has the cleanest embedding and prototype rows, and 1.6 points less F1 (2.1 in the tail);
- J3: a learned head trained afterwards on the frozen model recovers at most 28 % of the F1 gap. The lead lies in what
  the learned head trains into the backbone and the hidden layer;
- dev/122: no linear map (centring, within-class whitening) and no undoing of the leaky ReLU on the saved embeddings
  closes the embedding gap. The difference is in what the representation encodes.

## Two routes, both testable now

1. **Read the embedding one layer earlier.** The embedding is the hidden layer's output, the layer nearest the head.
   Several results say the layer before a projection head keeps more of what transfers: SimCLR; Wang et al. (CVPR
   2022), an MLP projector in supervised training; Kornblith et al. (NeurIPS 2021), where losses that raise accuracy
   make the penultimate layer transfer worse and the differences appear only in the last layers. If the learned
   head's damage stops at the hidden layer, its pooled trunk feature f is clean, and one model gives both worlds: its
   head for species, f for prototypes and open set. No retraining.
2. **Let the frozen head use the whole sphere.** The frozen head's code is sparse because the leaky ReLU keeps it in
   the positive orthant, where only non-overlapping codes are near-perpendicular: about 84 of 1,280 coordinates per
   image, the hidden layer driving the rest well below zero. With 12,632 species, a non-negative code is crowded;
   signed codes are not (random directions in 1,280 dimensions already have cosines of about +-0.03). If the crowding
   is what costs the frozen head its tail F1, removing the ReLU raises its F1 and keeps its geometry. Against it:
   Kini et al. (ICLR 2024) add a final ReLU on purpose, because under imbalance it keeps class means symmetric, which
   would favour the tail. The test is informative either way.
   The same change on the learned head settles the owner's question whether the ReLU handicapped the default model.

## J4 design

J2's testbed throughout (EfficientNetV2-S, marginal supervision, global_lepi_shards-v1, seed 42, sqrt oversampling,
J2's schedule), one factor per arm, each scored as J2 was (test fold, frequency bins, traps, calibration, open set),
and dev/122 on the saved embeddings.
- **J4f**, no training: re-embed J2's three models (ctl-os, frz-os, gs-os) at the pooled trunk feature f (the
  classifier's input, before dropout and the hidden layer), same images as e-j2-os, then dev/122 on f.
- **J4a**: frz-os with the hidden layer's activation removed (h = W f + c, normalised; nothing else changes).
- **J4b**: ctl-os with the same change.
mini_trainer needs an opt-in option for the activation, recorded in the checkpoint's metadata, default unchanged.
Cost: J4f about 1 B200-h; J4a and J4b about 5-6 B200-h each with evaluation; one 1-GPU job each, in parallel.

## Predictions (before any number)

1. J4f: the learned head's f is closer to the frozen head's f than its embedding is to the frozen embedding: the
   frozen-minus-learned gap on dev/122's new-species and new-genera measures at f is at most half the gap at the
   embedding (new genera, family: at most 1.6 points against 3.3). Confidence about even.
2. J4f: the frozen head's f is no better than its embedding (within 1 point on every dev/122 measure); its embedding
   is already its cleanest layer.
3. J4a: the frozen head without the ReLU gains species F1 (zero) over frz-os, +0.3 to +1.0, most of it under 100
   images; its embedding measures stay within 1 point of frz-os's and its rows stay prototypes (median cosine to the
   predicted class above +0.2). Confidence low: the imbalance argument points the other way.
4. J4b: the learned head without the ReLU lands within 0.3 of ctl-os in species F1, its embedding measures within 1
   point of ctl-os's, and its rows stay anti-prototypes (median cosine to the predicted class below 0): the scale, not
   the ReLU, shapes its geometry (J2).
5. Decision rule for the Tree of Life run's head: if prediction 1 holds with the learned head's f within 0.5 of the
   frozen embedding on the new-species and new-genera measures, the learned head with f exported as the embedding;
   else, if J4a reaches within 0.5 of ctl-os in species F1 with frz-os's embedding quality, the frozen head without
   the ReLU; else the trade-off stands and the next test is a deeper projector (embedding read before it).

**Submitted 2026-11-26 13:44-13:46 (mt; mini_trainer `.agents/local/campaign/plans/j4/README.md`).** j4a 12481289 and
j4b 12481291, one B200 each; e-j4f-ctl 12481293, e-j4f-frz 12481294, e-j4f-gs 12481297; e-j4a, e-j4b and e-j4-os
queued after them; about 11-12 B200-h. mini_trainer 3271c3c (on 7ef4442, neither pushed) adds `hidden_activation`
(`leaky_relu` or `none`, default `leaky_relu`, recorded in the checkpoint only when not the default). Tests: the
default is bit-identical to the code before the option (metadata, state-dict keys, outputs), `none` is exactly
normalise(W f + c), and a reload rebuilds it. J4a's and J4b's configs differ from frz-os's and ctl-os's in that one key.
J4f stores the head's input (a pre-hook: pooled f before dropout and the hidden layer, fp16) in e-j2-os's image order,
and checks that f through the model's own hidden layer reproduces the saved embedding (CPU smoke on frz-os: maximum
difference 2.4e-4, fp16). dev/122 `--layer pooled` scores it.

## J4f result (scored 2026-11-26)

mt's jobs (12481293/94/97, exit 0, about 0.5 B200-h) passed three checks: the rows match e-j2-os, the recomputed embeddings
are bitwise equal to e-j2-os's, and f through each checkpoint's own hidden layer and normalisation reproduces the saved
embedding (maximum difference 1.2e-4 to 2.4e-4, fp16 rounding), so f is exactly the head's input. dev/122 `--layer pooled
--transform none,center,whiten` (CPU, 4 cores) scored f L2-normalised; `/data/au761367/lepinet/j2/openset/
embedding_quality_pooled_none_center_whiten.json`. A first run labelled the centred variant with the bare arm name, so it
overwrote the raw one in the paired table; fixed (only `none` keeps the bare name, and `center` when it is the only
transform) and rerun before any number was used.

Measures (as in dev/122): macro accuracy over species, each image assigned the label of the nearest class mean
(cosine; the mean of the reference images per known species) or of the nearest reference image (1-NN); known species
by species, genus and family; never-trained species of known genera placed by genus and by family; never-trained
genera placed by family.

| layer, arm | class mean species | genus | family | 1-NN species | new species, genus | new species, family | new genera, family | effective dims |
|---|---|---|---|---|---|---|---|---|
| embedding, learned | .8825 | .9540 | .9896 | .8524 | .5132 | .8576 | .6391 | 142 |
| embedding, frozen | .8876 | .9576 | .9906 | .8650 | .5260 | .8777 | .6718 | 496 |
| embedding, global scale | .8402 | .9266 | .9817 | .8043 | .4785 | .8427 | .6271 | 168 |
| f, learned | .8766 | .9521 | .9887 | .8462 | .5106 | .8517 | .6321 | 149 |
| f, frozen | .8802 | .9543 | .9904 | .8603 | .5244 | .8783 | .6788 | 182 |
| f, global scale | .8400 | .9280 | .9814 | .7980 | .4875 | .8343 | .6115 | 115 |
| f whitened, learned | .8683 | .9471 | .9878 | .8328 | .5177 | .8618 | .6470 | 903 |
| f whitened, frozen | .8803 | .9545 | .9905 | .8592 | .5293 | .8824 | .6893 | 735 |

Frozen minus learned, points (95 % bootstrap over species in brackets for f):

| | class mean species | 1-NN | new species, genus | new species, family | new genera, family |
|---|---|---|---|---|---|
| embedding | +0.51 | +1.27 | +1.28 | +2.01 | +3.27 |
| f | +0.35 [0.13, 0.56] | +1.41 [1.22, 1.65] | +1.39 [0.78, 1.99] | +2.66 [2.20, 3.09] | +4.67 [4.18, 5.36] |
| embedding, whitened | +0.95 | +1.83 | +1.35 | +2.15 | +3.07 |
| f, whitened | +1.21 [1.00, 1.42] | +2.64 [2.41, 2.90] | +1.16 [0.51, 1.81] | +2.06 [1.55, 2.47] | +4.23 [3.56, 5.02] |

**Predictions scored.**
1. **Wrong.** At f the learned model is as far or further behind the frozen one as at the embedding, not half as far:
   new genera by family +4.7 against +3.3 (whitened +4.2 against +3.1), new species by family +2.7 against +2.0. Its
   learned head's hidden layer does not add the damage; it slightly repairs it (embedding minus f, learned: +0.1 to
   +0.7 on every measure).
2. **Right.** The frozen model's f is within 1 point of its embedding on all seven measures (largest: class mean
   species -0.74; new genera by family +0.69).
5. Decision rule, first branch: **no.** The learned model's f is 4.0 points below the frozen embedding on new genera by
   family (needs within 0.5), 2.6 on new species by family.

**What it means.** The learned head's cost to the embedding is in the trunk: no layer of the learned-head model places
new taxa as well as the frozen-head model, and its one-layer projector absorbs none of the damage. With J3 (its F1
lead also lies in the trunk), the per-species scale changes what the trunk learns, in both directions, so one model
read at two layers cannot give both worlds. This also lowers my expectation for a deeper projector (decision rule,
third branch): the literature's "damage stays in the projector" did not hold for a one-layer projector here. The
remaining route in J4 is J4a, the frozen head without the ReLU.

Side results. Whitening f within class helps the frozen model's new-taxon placement by 0.4 to 1.1 points (new genera
by family 0.6893 [+0.61, +1.46 over raw f]), the best value of any representation so far, level with the frozen
embedding after undoing the ReLU and whitening (0.6888); it costs the learned model 0.8 to 1.4 points on known species
and gains it 0.7 to 1.5 on new taxa. Centring f changes nothing (at most 0.23 points), because f's images are already
spread around the origin: mean pairwise cosine |mean f|^2 0.18 learned, 0.17 frozen, 0.07 global scale, with 66, 76
and 59 % of coordinates negative carrying 53, 46 and 50 % of the squared length. My earlier note that f is "almost
non-negative" (11-24 entry) was assumed from the SiLU and is wrong; corrected there.

## J4a interim (validation only, 2026-11-26 evening)

j4a 12481289 finished (14:28-18:57). Validation species macro-F1 (mt's validation split, not the test fold), J4a against
frz-os by epoch: .7462/.7258, .8075/.7789, .8531/.8242, .8946/.8715, .9131/.8939; ahead after every epoch, +1.9 points at
the end and above ctl-os's .9076; top-1 accuracy 93.48 (frz-os 92.28, ctl-os 93.18). One seed. Not scored against
prediction 3 until the test fold (e-j4a) and the embedding measures (e-j4-os, dev/122) are in: prediction 3 also requires
the embedding within 1 point of frz-os and the rows to stay prototypes. If both hold, the ReLU did cost the frozen
head accuracy in training, which the 11-24 entry's post-hoc transforms could not see: they tested whether the trained
embedding was handicapped, not whether training through the ReLU was.

## J4 result (scored 2026-11-26)

All jobs exit 0, about 11 B200-h. mt scored the test fold, traps, calibration and open set (mini_trainer
`.agents/local/campaign/plans/j4/README.md`, "J4 result"); dev/122 (embedding measures, `embedding_quality_none_whiten.json`,
`_pooled_none_whiten.json`, `_j4b_vs_frz_{embeddings,pooled}.json`) and dev/119 (class-mean open set,
`classmean_auroc_j4.json`) here, all under `/data/au761367/lepinet/j2/openset/`. Every arm seed 42, EfficientNetV2-S, sqrt
oversampling; J4a and J4b differ from frz-os and ctl-os in `hidden_activation: none` only. Checks: e-j4-os's rows equal
e-j2-os's; f through each J4 checkpoint's hidden layer (no activation) reproduces its saved embedding to 6.1e-5.

**Test fold, species macro-F1** (zero-shot threshold / calibrated) and by training-image bin:

| arm | species F1 | under 100 | 100-200 | 200-500 | 500-1,000 | 1,000 and over | genus | family | top-1 |
|---|---|---|---|---|---|---|---|---|---|
| ctl-os (learned, leaky ReLU) | .8961 / .8998 | .885 | .903 | .919 | .936 | .942 | .9559 | .9678 | .9314 |
| frz-os (frozen, leaky ReLU) | .8798 / .8850 | .864 | .889 | .908 | .927 | .935 | .9483 | .9746 | .9223 |
| J4a (frozen, no activation) | .9033 / .9043 | .892 | .909 | .924 | .939 | .945 | .9598 | .9770 | .9345 |
| J4b (learned, no activation) | .9030 / .9040 | .892 | .908 | .924 | .939 | .945 | .9595 | .9697 | .9345 |

Genus and family are zero-threshold macro-F1. J4a - frz-os +2.35
(+2.8 under 100), J4a - ctl-os +0.72, J4b - ctl-os +0.69, J4a - J4b +0.03.

**Geometry** (median over known test images; mt's `j4a.json`, `j4b.json`): cosine to the predicted class's row, and the
share of classes whose g cos is clamped at -1 (the floor, where every class gets the same logit):

| arm | scale g, median | bias b, median | cosine to predicted row | classes at the floor |
|---|---|---|---|---|
| ctl-os | 1.52 | -0.35 | -0.58 | 99 % |
| frz-os | 1 (fixed) | 0 | +0.33 | 0 |
| J4a | 1 (fixed) | 0 | +0.52 | 0 |
| J4b | 0.68 | -0.86 | +0.70 | 0 |

**Embedding** (dev/122, macro accuracy over species; measures as in the J4f section), raw embedding, and J4b - frz-os
with a 95 % bootstrap over species:

| arm | class mean, species | 1-NN | new species, genus | new species, family | new genera, family |
|---|---|---|---|---|---|
| ctl-os | .8825 | .8524 | .5132 | .8576 | .6391 |
| frz-os | .8876 | .8650 | .5260 | .8777 | .6718 |
| J4a | .8751 | .8408 | .5175 | .8602 | .6438 |
| J4b | .8898 | .8621 | .5285 | .8705 | .6527 |
| J4b - frz-os | +0.22 [0.02, 0.44] | -0.29 [-0.55, -0.05] | +0.25 [-0.35, 0.94] | -0.72 [-1.25, -0.26] | -1.92 [-2.69, -1.12] |
| J4b - ctl-os | +0.73 [0.55, 0.91] | +0.97 [0.79, 1.18] | +1.53 [0.99, 2.18] | +1.30 [0.82, 1.71] | +1.36 [0.65, 2.23] |
| J4a - frz-os | -1.25 [-1.44, -1.05] | -2.42 [-2.69, -2.16] | -0.85 [-1.48, -0.18] | -1.75 [-2.29, -1.16] | -2.80 [-3.64, -2.01] |
| J4a - J4b | -1.47 [-1.67, -1.30] | -2.13 [-2.35, -1.89] | -1.10 [-1.62, -0.58] | -1.03 [-1.48, -0.64] | -0.89 [-1.53, -0.26] |

At the pooled trunk feature the order is the same (J4b - frz-os on new genera by family -3.0 [-3.7, -2.4]), and whitening
helps neither J4 arm (J4a loses 1.6 to 3.0 points on known species).

**Open set**, AUROC, known_bal against all unknown strata (mt, paired with J2; class mean from dev/119):

| arm | entropy | max logit | max cosine to a row | class mean | 1-NN | 10-NN mean |
|---|---|---|---|---|---|---|
| ctl-os | .903 | .913 | .542 | .893 | .876 | .851 |
| frz-os | .912 | .812 | .812 | .912 | .902 | .889 |
| J4a | .924 | .928 | .928 | .894 | .878 | .854 |
| J4b | .927 | .933 | .923 | .906 | .891 | .871 |

Best on known_unif: ctl .929, frz .945, J4a .950, J4b .953.

**Traps** (tta3bm; probe calibrated / zero, held-out calibrated / zero): ctl .630/.387, .472/.224; frz .608/.384,
.468/.215; J4a .662/.390, .522/.217; J4b .660/.406, .556/.227. **Calibration**, debiased L2 (test fold, trap vocabulary,
trap list): ctl .057, .062, .034; frz .037, .089, .082; J4a .024, .097, .061; J4b .045, .118, .073.

**Predictions scored.**
3. **Partly wrong.** J4a gains species F1 over frz-os, mostly in the tail as predicted, but far more than predicted
   (+2.35 against +0.3 to +1.0; under 100 images +2.8), and its rows stay prototypes (+0.52 > +0.2). The embedding
   clause fails: 1-NN -2.4, new species by family -1.8, new genera by family -2.8, k-NN open set .878 against .902.
4. **Wrong on all three counts.** J4b beats ctl-os by +0.69 (not within 0.3); its embedding is better by 0.7 to 1.5
   points, not within 1; its rows are the strongest prototypes of the four (+0.70), not anti-prototypes. The scale did
   not shape ctl-os's geometry alone: with the leaky ReLU the learned scale grew to 1.5 and pushed 99 % of the classes
   onto the floor; without it, it settled at 0.68 and none reached the floor.
5. **The rule, read literally, says the trade-off stands** (first branch no, J4f; second branch no, J4a lacks frz-os's
   embedding) and a deeper projector is next. It did not foresee J4b, which beats ctl-os on every measure here except
   calibration under shift.

**What it means.**
- The leaky ReLU before the normalisation did handicap both heads, in training, which the 11-24 entry's post-hoc
  transforms could not see (they asked whether the trained embedding was handicapped, and it was not). The supervisor's
  answer to the owner then ("no handicap from the ReLU") was wrong for this reason.
- With the ReLU, the learned head's F1 lead came with anti-prototype rows; without it, the same head is more accurate
  (+0.7), its rows are prototypes, and its embedding recovers between 40 % (new genera by family) and all (new species
  by genus, class mean) of its gap to frz-os. The trade-off J1 to J3 measured was mostly the ReLU's.
- A gap to frz-os remains on placing never-trained taxa in their family (-0.7 new species, -1.9 new genera) and on k-NN
  open set (-1.1); J4b has 2.3 points more F1 and a better open set by its head.
- Removing the ReLU from the frozen head (J4a) buys the same F1 but spends the embedding, so the frozen head's
  embedding advantage was tied to the orthant; the learned head without the ReLU keeps more of it.
- Calibration under shift is worse for both J4 arms (trap vocabulary .118 J4b against .062); post-hoc temperature
  scaling is the obvious test.
- Face recognition had removed this ReLU for the reason found here (SphereFace's appendix: it confines features to
  the non-negative orthant); the 11-24 entry cited it and still concluded no handicap.

Limits: one seed; EfficientNetV2-S only (the ToL bundle is ViT-L); one dataset.

**Recommendation (supervisor; the owner decides):** the ToL bundle's head is J4b's, the learned per-class scale and
bias with `hidden_activation: none`. It needs mini_trainer 3271c3c (with 7ef4442), not pushed.

## Addendum (2026-11-27): J4b against frz-os on the embedding, the owner's first priority

The owner ranks the embedding's cleanness above species F1 and asked whether J4b is as good as frz-os at separating
known species, genera and families, at placing and at detecting new species. Paired against frz-os (bootstrap over
species, 95 %; dev/122 `embedding_quality_j4b_vs_frz_embeddings.json`, dev/119 `classmean_auroc_j4b_vs_frz.json`, which
now pairs against its first arm):

| what | measure (embedding only unless marked) | frz-os | J4b | J4b - frz-os, points |
|---|---|---|---|---|
| separating known taxa | class mean, species / genus / family | .8876 / .9576 / .9906 | .8898 / .9577 / .9905 | +0.22 [0.02, 0.44] / 0.00 / 0.00 |
| | 1-NN, species | .8650 | .8621 | -0.29 [-0.55, -0.05] |
| | class-mean cosine: same genus / same family, other genus / other family | .41 / .06 / .02 | .42 / .03 / .00 | |
| | nearest class mean is a congener | .764 | .748 | |
| placing new taxa | new species, genus | .5260 | .5285 | +0.25 [-0.35, 0.94] |
| | new species, family | .8777 | .8705 | -0.72 [-1.25, -0.26] |
| | new genera, family | .6718 | .6527 | -1.92 [-2.69, -1.12] |
| detecting new species (AUROC, known_bal, all strata) | nearest class mean | .912 | .906 | -0.7 [-0.8, -0.5] |
| | nearest reference image | .902 | .891 | -1.2 [-1.3, -1.0] |
| | nearest head row (head as prototypes) | .812 | .923 | +11.2 [10.7, 11.7] |
| head rows | cosine to the predicted row, median | +0.33 | +0.70 | |
| species F1, test fold (head) | | .8798 | .9030 | +2.32 |

By stratum (known_bal), class mean: near -0.7, mid -0.6, far -0.1 (interval includes 0); 1-NN -1.3, -1.0, -1.4; head
rows +11.6, +11.0, +3.5. On known_unif the embedding-only gaps are -0.4 to -1.9.

Reading: J4b equals frz-os on separating known taxa and on placing new species in their genus, and has the same
taxonomic ladder of class-mean cosines; it is behind on placing never-trained taxa in their family (-0.7 new species,
-1.9 new genera) and on detecting new species from the embedding alone (-0.7 to -1.2); it is far ahead when the head's
rows are used as the prototypes (+11), the paper's "open-set model trained closed-set" in its literal form. So the
embedding-only lead of frz-os survives at 0.7 to 1.9 points on the coarse and open-set measures, against 2.3 points of
F1 and rows that are usable prototypes. **Owner (2026-11-27): J4b's head for the ToL bundle.**
