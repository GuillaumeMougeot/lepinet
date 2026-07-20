# Is there a way to build a fast, small app quickly for lepinet inference

Goal: prediction app using the camera or the photo galery of a phone or desktop
and giving a species, genus and family names of a lepidoptera photo. 


Additional goals: The app also gives model confidence and highlight the levels
above a minimum confidence threshold, the other levels are greyed. If no family
is above threshold, then the app question the presence of a lepidopter or state
that the displayed species may not be part of the training dataset. For the
lowel level above threshold, the app display three pictures within its training
database. For each level, the app also gives the hyperlink toward the GBIF page.

The app should be light and fast and run offline.

## User interface

The app could look like a Google translate app. 

When launched, you get a home page with text explaining what this app is and in
the middle, a big camera button that opens the camera of the user and a small
galery button on its left side that open the phone galery. This could also be
just a camera button and when the camera opens, there is a galery botton on the
side. Ideally, rely on the default camera app, so the app does not have to
include camera management such as rotation, zoom etc. At the bottom of this home
page a link redirect a user who would like to learn more or give feedback to the
GitHub page.

After a picture has been taken, it shrinks down at the top tier of the screen
and the prediction appears almost instantenously at the bottom of the screen. A
loading symbol could be necessery in the future while waiting for the
prediction, but it would be great to not need it. 

In the middle tier, the predictions family, genus and species names are
presented on top of each others. There names can be copied to be searched. The
user can either highlight the text or a little copy symbol can be used. A
confidence score is displayed next to each level with visual cues. For instance,
this confidence number between 00 and 99 could be included inside a progress bar
circling around it that goes from 00 to 99 and from red to green. The taxa name
is greyed if the level of confidence is below a certain threshold. This
threshold could be 0.5, but this may change in the future. Next to each taxa
name, a hyperlink symbol redirect toward GBIF page associated with this taxa.
This 3x3 informations should be cleaning displayed in a structured fashion.

In the bottom tier of the display, three sample pictures of the lowest
highlighted level are displayed. They can be clicked to be displayed bigger. A
text below inform about the taxa name.

The user can then use the return button of their phone to go back to the home
page.

## Roadmap

This section describes the roadmap of the development process of this app.

The app should be as fast and as small as possible and run locally on the user
phone or desktop. Fast has higher priority than small. But both are important.

### Shrinking the model down to the fastest and smallest possible size

This section tackles the difficult problem of making a trained deep learning
model production-ready: fast and light-weight. 

The current model is a fastai-trained efficientnetv2_s with ~12000 classes. The
model heads is quite large. The model file size is currently ~170 Mb. The goal
would be to reduce it by x10 or more.

Repo used to train the model: https://github.com/GuillaumeMougeot/lepinet and
stored locally in ~/codes/lepinet. The script used is dev/030.

Here are some potential techniques to explore: To reduce the size of the last
layer:
- Low-rank factorization
- PCA on the logits -> Potential risk of strong accuracy loss.

To reduce the size of the entire model:
- Knowledge distillation
- Quantization 
- SVD?

Techniques with less potential or more substantial work/exploration yet: 
- Trained other architecture?
- Exploit the hierarchical structure even more by having a series of models?
- Learned embeddings: this is probably not going to reduce the size of the final
  app file.

More?

The part of the project is allowed to have time and space, meaning that the goal
has to be reached. Training new models is allowed. New scripts can be created
for this purpose in the dev folder, following the numbering structure. This is
to be seen as an engineering challenge and the app can go down in size as much
as possible. Less than a Mb is the dream. Go down to assembler if required
(well, this could also obfusticate the code so this could also be parallel dev).

### Creating the app

The app is intentionally simple and light-weight to be able to be easily
downloaded and run locally (offline).

The goal is to use most of the modern tools and techniques to reach this goal.

To avoid having to set the app on a "store" and be platform-agnostic, the
current avenue is to use PWA. This is strongly open to improvement if needed.

The app would be hosted on GitHub Pages, created with CI with GitHub Actions.
When opening the GitHub page, the user will be prompted to "download" the app,
like "'Add to Home Screen' to install this app for offline use." 

I don't know if this app code belongs to the same lepinet repo or to another
"lepinet-app" repo?

Here is an example of list of potential tools to use: 

| Purpose    | Recommendation                    | Why                                              |
| ---------- | --------------------------------- | ------------------------------------------------ |
| ML         | PyTorch                           | Already there                                    |
| Export     | ONNX                              | Browser standard                                 |
| Runtime    | ONNX Runtime Web                  | Fastest mature solution                          |
| Language   | TypeScript                        | Modern standard                                  |
| Runtime    | Bun                               | Yes                                              |
| Bundler    | Bun                               | Bun increasingly replaces Vite for many projects |
| UI         | Svelte 5                          | Small runtime, simple                            |
| PWA        | Bun                               | Offline install                                  |
| Hosting    | GitHub Pages                      | Free                                             |
| CI         | GitHub Actions                    | Automatic deployment                             |
| Formatting | Biome?                            | Replaces Prettier + ESLint                       |

This list is strongly amenable and must be changed if a row is wrong, too much
or missing. I can think of tools such as WebGPU, WASM or Rust lang.

### Future avenues of development

- Storing capture images locally or having some sort of identification history.

## Development rules

- Journal all developments: anyone reopening the code must be able to understand
  both the final product and the journey that led to it with all the important
  lesson. Keep an exhaustive but efficient writing style. If done within lepinet
  repo, follows the journaling method: create a journal entry per problem and
  document each new script in the dev/readme. 
- For dev, use all modern methods, uv, bun etc.

---

# Review

*Written 2026-07-20 by Claude (Opus 4.8), on request, as an objective critique
of the proposal above. Grounded in the repo's own numbers: 5,669,317 images,
12,041 species / 4,333 genera / 102 families, best test species macro-F1
**0.9148** (`20260716-154156`, sqrt-oversampling, `dev/032` on fold `set ==
'0'`). A companion document with the detailed plan and the ideas not in this
proposal is in [`2026-07-lepi-app-claude.md`](2026-07-lepi-app-claude.md).*

## Verdict up front

The project is well-scoped and the two halves are genuinely separable:
**compression is a solved-by-engineering problem** with a predictable outcome,
and **the app is a small frontend** with a handful of real but known
constraints. The 10× size reduction is achievable with a large margin — the
honest target is ~**5–8 MB**, not 17 MB. The "<1 MB dream" is not reachable at
12,041 classes without abandoning the class set (see the information floor
below).

The proposal's weakest points are not in the ML. They are three product-level
assumptions that are stated as if settled and are not: **the sample-image
gallery is incompatible with "offline"**, **the 0.5 confidence threshold is not
a meaningful quantity on this model**, and **"not a lepidopteran" is an open-set
problem the current model cannot answer**. Each is fixable, but each needs a
decision before UI work starts, because each changes the UI.

## Where the 170 MB actually is

Measured, not guessed:

| part | params | fp32 |
|---|---|---|
| effnetv2_s backbone (no classifier) | 20.18 M | 80.7 MB |
| heads: 1280 × (12041 + 4333 + 102) | 21.09 M | 84.4 MB |
| **total** | **41.3 M** | **165 MB** |

The proposal says "the model head is quite large". It is more than that: **the
heads are 51% of the model**, slightly larger than the entire backbone. This
reframes the roadmap. Attacking only the last layer caps the win at 2×;
attacking only the backbone also caps at 2×. Both must be done, and they need
different techniques.

Second, and more useful: the head is a **cosine (normalized) classifier**
(`mini_trainer.modeling.classifier.Classifier`, `normalized=True`). Every row of
the weight matrix is a unit-norm prototype. That means all 16,476 rows share an
identical dynamic range — which is the single best case that exists for
post-training quantization. An int8 per-tensor quantization of a unit-normalized
matrix loses almost nothing; 4-bit with per-row scales is plausible. The
proposal lists quantization as one option among many; it is in fact the
highest-yield, lowest-risk lever on the larger half of the model, and it should
be tried **first**, before any retraining.

## The compression roadmap, critiqued item by item

**Low-rank factorization on the head — right idea, but for the wrong reason and
in the wrong direction.** The head matrix is 1280 × 16476; its rank is already
capped at 1280. Post-hoc SVD to rank *r* helps only if *r* ≪ 1280, and
factorizing an already-normalized prototype matrix breaks the normalization the
model was trained under. The clean version of this idea is to **train with a
bottleneck**: set the classifier's `preclassification_size` (the `hidden`
argument) to 256 instead of the backbone's 1280. The head then costs 256 × 16476
= 4.2 M params (16.9 MB fp32, **4.2 MB at int8**), a 20× head reduction, and it
is trained rather than approximated so the accuracy cost is measured once and
paid honestly. This is the single most important change and it is not in the
proposal.

**PCA on the logits — drop it.** It is strictly worse than the bottleneck (it is
the same low-rank idea applied after the fact instead of during training), it
destroys calibration, and 12k-class logits from a cosine head have high
effective rank. The proposal already flags the accuracy risk; the risk is real
and the technique is redundant.

**Knowledge distillation — keep, but frame it correctly.** Distillation is not a
compression technique here, it is the *training method* for whatever smaller
backbone gets chosen. The valuable, under-appreciated property: the 0.9148 model
can produce soft targets for **unlabelled** images, so the student can be
trained on far more data than the labelled set, and soft targets over 12k
classes carry the hierarchy implicitly. Budget it as a proper training run, not
a post-processing step.

**Quantization — promote to first position.** See above.

**SVD — same thing as low-rank factorization, listed twice.**

**"Other architecture" is in the low-priority bucket and should be in the
high-priority one.** Swapping effnetv2_s (20.2 M) for mobilenetv4-conv-small
(~3.8 M) or effnetv2-b0 (~7 M) is a bigger, more certain backbone win than
anything post-hoc, and `dev/030` already takes `model_arch_name` as a config key
— the experiment costs one config file. It is listed as "less potential"; it is
the opposite. The relevant question is how much of the 0.9148 survives, and that
is exactly what the existing ledger + `dev/032` pipeline is built to answer.

**"A series of models exploiting the hierarchy" — reject for v1.** It multiplies
the artifact count, breaks the single-forward-pass latency story, and the
current model already predicts all three levels in one pass. Revisit only if a
family-conditional refinement stage is needed for look-alikes.

**"Go down to assembler" — this is a category error and it should be struck.**
The app runs in a browser. The floor is WASM SIMD (via ONNX Runtime Web) or
WebGPU shaders; there is no level below that you can reach, and hand-written
kernels would lose to ORT's. The parallel remark about obfuscation also
contradicts the project's own stance — the model is served from a public GitHub
Pages deployment built by public CI from a public repo, so it is downloadable
regardless. Decide whether the model is open (it should be, given the GBIF
provenance) and stop spending effort on the question.

### The information floor

Worth stating plainly so the "<1 MB" target can be retired with a number rather
than a feeling. A classifier over 12,041 species must store 12,041
distinguishable prototypes. At a 64-dim embedding and 4-bit weights — already an
aggressive, accuracy-costly setting — the species prototypes alone are 385 KB,
before the genus and family heads, before the backbone, before the taxonomy
strings, before the runtime. **Sub-1 MB is not reachable while keeping 12k
classes.** It becomes reachable only by cutting the class set (e.g. a regional
~2,000-species build), which is a legitimate product decision but a different
product. Recommendation: retire "<1 MB", adopt **≤ 8 MB total download** as the
v1 target, and keep the regional-build idea as a deliberate variant.

## Three unexamined assumptions in the app design

**1. The sample-image gallery cannot be offline.** The UI spec requires three
example photos of the predicted taxon. At 12,041 species × 3 thumbnails × ~8 KB
that is ~289 MB — 36× the entire model budget, and that is at thumbnail quality.
This is the largest contradiction in the proposal and it is currently invisible
because the images are described as if they were free. Options, in order of
preference: (a) fetch from GBIF on demand and show a graceful "needs connection"
state — offline inference, online illustration; (b) bundle **one** 96 px WebP
per species (~12 MB, doubles the app size); (c) bundle thumbnails for
families/genera only (102 + 4,333 entries ≈ 4 MB) and link out for species.
Related and equally unaddressed: GBIF images carry heterogeneous licences
(CC-BY, CC-BY-NC, some rights-reserved). Redistributing them inside an installed
app is a licensing act that needs per-image attribution and a licence filter.
Option (a) sidesteps this entirely, which is a strong argument for it.

**2. The 0.5 threshold is not a meaningful number on this model.** Softmax over
12k cosine-derived logits is systematically overconfident, and the value 0.5 has
no interpretation the user would recognise ("50% sure" is not what it means).
Greying a correct answer at 0.49 and highlighting a wrong one at 0.51 is worse
than showing no confidence at all, because the UI is making an explicit
reliability claim. The threshold must be **derived, not chosen**: calibrate on
the validation fold (temperature scaling is one parameter and costs nothing),
then pick per-level thresholds that hit a stated target precision on the
held-out `set == '0'` fold. Then the UI's claim is a measured claim. The
existing `dev/032` already emits per-prediction scores, so the data needed for
this is one script away.

**3. "Is this even a lepidopteran?" is open-set recognition, not a low
softmax.** The model was trained on moths and butterflies only; shown a beetle,
a leaf, or a coffee cup, it will confidently return a species, because softmax
normalizes over the classes it has. A low max-probability is a weak proxy at
best. Doing this properly needs either an out-of-distribution score (energy /
max-logit, thresholds set against an explicit negative image set) or a cheap
binary "lepidopteran / not" gate in front of the classifier. The negative set
has to be built — it does not exist in the repo today. This is the most
scientifically interesting open item in the proposal and the one most likely to
be underestimated.

## Smaller but real gaps

- **Hierarchy consistency.** The best model uses **independent** heads, which
  can return a species whose genus disagrees with the genus head. The UI stacks
  all three levels, so every inconsistency is visible to the user as an obvious
  error. Fix is free and should be done regardless: compute genus and family by
  marginalizing the species distribution up the taxonomy instead of reading
  their heads. Guaranteed-consistent, usually *more* accurate at the upper
  levels, and it deletes two-thirds of the head weights from the shipped
  artifact (~5 MB of the fp32 head).
- **Preprocessing parity.** The classic accuracy-loss bug in browser deployment:
  PyTorch's resize kernel and the browser's canvas `drawImage` do not agree, and
  a model trained at 460→256 that is fed a differently-resampled 256 px crop can
  silently lose several points. This must be tested numerically (same image →
  PyTorch logits vs ORT Web logits) and not assumed. Add EXIF orientation
  handling — phone photos are frequently rotated.
- **"Predictions appear almost instantaneously"** is optimistic. Expect ~200–600
  ms per inference on a mid-range phone via WASM SIMD, plus a **multi-second
  first-run cost** for graph initialization. Build the loading state now, and
  warm the model up at app launch so the first real prediction is not the slow
  one.
- **iOS is the binding storage constraint.** Safari's eviction policy is the
  reason to care about size, more than download time. It is a much stronger
  justification for the compression work than "fast and small" and is worth
  stating as such.
- **Metric mismatch.** macro-F1 is the right *research* metric (it is why
  oversampling won) but it does not describe app experience, which is dominated
  by common species. Report top-1 accuracy and top-5 recall alongside it — a
  user shown five candidates can disambiguate, so top-5 is arguably the product
  metric.
- **Geography is the missing free accuracy.** A purely visual model over 12k
  global species must separate allopatric look-alikes it will never see together
  in reality. A coarse per-species occurrence prior from GBIF (which is already
  the data source) is a few hundred KB compressed and is likely a larger
  accuracy gain than anything in the compression roadmap. Optional, opt-in,
  degrades gracefully offline. This is the single best idea not currently in the
  proposal.
- **Repo split: yes, separate `lepinet-app`.** Different toolchain, different
  CI, different release cadence, and mixing bun/TS into a research repo makes
  both harder to navigate. The clean seam: `lepinet` gains an export step that
  emits a **versioned artifact bundle** (ONNX + taxonomy + thresholds +
  calibration + metrics), published as a GitHub Release; `lepinet-app` pins a
  bundle version. That seam also gives the app reproducibility for free — every
  deployed app states which model it ships.
- **Disclaimer.** A confident wrong species used for a conservation, pest, or
  toxicity decision is a real harm. One line in the UI.

## What is good and should be preserved

- The compression/app split is the right decomposition, and the ordering
  (compress first) is correct — the app's constraints all follow from artifact
  size.
- PWA over app stores is the right call for this audience and this budget. No
  review process, no signing, one deployment, works on desktop.
- "Rely on the default camera app" is the correct instinct and saves the largest
  chunk of frontend work.
- Offline inference is a genuine privacy property — no photo leaves the device —
  and is worth stating as a feature rather than an implementation detail.
- The tool table is sound. Two amendments: ONNX Runtime Web should be configured
  WebGPU-with-WASM-fallback rather than either alone (WebGPU is fast where it
  works and unreliable on iOS Safari), and the PWA service worker should be
  `vite-plugin-pwa` or hand-written Workbox rather than "Bun", which is a
  runtime and bundler, not a PWA layer.
- The journaling rule is the project's real asset. It is why this review could
  be written from numbers instead of recollection.

## Recommended sequencing

The proposal implies compression research, then app. A better order front-loads
the unknowns that could invalidate the UI:

1. **Export the current 0.9148 model to ONNX and get it running in a browser,
   unmodified.** 165 MB is unshippable but it is testable, and it resolves
   preprocessing parity, real device latency, and ORT behaviour with a cosine
   head — the three things that would force a redesign if discovered late.
2. **Quantize (int8, post-training).** Highest yield per unit of effort, no
   retraining.
3. **Calibrate and derive thresholds** on the existing folds, in parallel with
   (2).
4. **Retrain with a 256-dim bottleneck + a smaller backbone**, distilled from
   the current model. This is the real compression work and it fits the repo's
   existing config/ledger loop.
5. **Build the app** against the artifact bundle from (4).
6. **Open-set detection and the geo prior** as v1.1 — both are research-shaped
   and should not block a first release.

The detailed plan, with per-step action items, size/accuracy budgets, script
numbering, and the open questions that need a decision before work starts, is in
[`2026-07-lepi-app-claude.md`](2026-07-lepi-app-claude.md).