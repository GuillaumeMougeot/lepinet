# lepi-app — detailed plan (Claude's companion to the proposal)

**Status:** decisions RESOLVED (§7), **Phases A and B are done and measured** — results in
[`2026-07-lepi-app-compression.md`](2026-07-lepi-app-compression.md), scripts `dev/040`–`dev/044`.
Phase C (retraining) and Phase D (the app) are open. Note that §1's size table and §8's
predictions were written *before* those measurements and are deliberately left unedited so the
compression journal can score them; the measured numbers supersede them. Companion to
[`2026-07-lepi-app.md`](2026-07-lepi-app.md) (the proposal + the Review section at its end).
Written 2026-07-20. This document holds the *how*: budgets, action items, script numbers,
and the ideas that were not in the proposal. The proposal remains the statement of intent;
where the two disagree, the proposal is the owner's and wins until edited.

---

## 0. The numbers this plan is built on

Measured from the repo, not assumed:

| quantity | value | source |
|---|---|---|
| images (global, filtered) | 5,669,317 | `data/global/models/…lepinet.parquet` |
| species / genera / families | 12,041 / 4,333 / 102 | same |
| best test species macro-F1 | **0.9148** | `20260716-154156`, `RESULTS.md` |
| micro-accuracy at that run | 0.9476 | `RESULTS.md` |
| backbone params (effnetv2_s, headless) | 20.18 M → 80.7 MB fp32 | timm |
| head params (1280 × 16,476) | 21.09 M → 84.4 MB fp32 | arithmetic |
| **total** | **41.3 M → 165 MB fp32** | matches the ~170 MB observed |
| head architecture | cosine / L2-normalized prototypes, `normalized=True` | `mini_trainer/modeling/classifier.py` |
| best head type | **independent** (beats hierarchical) | `RESULTS.md` |
| median images/species | 177; 31.8% of species have <100 | [[2026-07-does-longtail-help]] |

Two consequences drive everything below:

1. **The heads are 51% of the model.** Compressing only one half caps the win at 2×.
2. **The head is cosine-normalized**, so every prototype row has identical dynamic range.
   This is the ideal case for post-training quantization — int8 should be nearly free, and
   4-bit per-row is worth measuring.

---

## 1. Size budget: where 165 MB becomes ~6 MB

Each row is a separate, independently-verifiable step. Cumulative, in the order they should
be applied.

| step | technique | backbone | heads | total | accuracy risk |
|---|---|---|---|---|---|
| 0 | baseline fp32 | 80.7 | 84.4 | **165 MB** | — |
| 1 | drop genus+family heads, marginalize from species | 80.7 | 61.6 | **142 MB** | none (expect *gain* at genus/family) |
| 2 | int8 PTQ (per-channel weights) | 20.2 | 15.4 | **36 MB** | low — cosine head is the best case |
| 3 | 256-dim bottleneck (retrain) | 20.2 | 3.1 | **23 MB** | small, measured |
| 4 | smaller backbone (effnetv2-b0 / mnv4-conv-small), distilled | 4–7 | 3.1 | **7–10 MB** | the real cost; the experiment |
| 5 | 4-bit head prototypes (per-row scales) | 4–7 | 1.6 | **6–8 MB** | measure; abandon if it costs >0.5 F1 |

**Target: ≤ 8 MB total download** including runtime and taxonomy. That is a **20×
reduction**, double the proposal's stated 10× goal.

**Not reachable: <1 MB.** 12,041 species prototypes at 64 dims and 4 bits are already 385 KB
before genus, family, backbone, taxonomy, or runtime. Sub-1 MB requires cutting the class
set. Kept as a deliberate variant, not a target — see §6.

### The bottleneck idea, concretely

`mini_trainer`'s `Classifier` takes a `hidden` argument that sets
`preclassification_size`; today it defaults to the backbone's `in_features` = 1280. Setting
it to 256 inserts a trained `nn.Linear(1280, 256)` and shrinks every head 5×. This is
strictly better than post-hoc SVD or PCA-on-logits, both of which approximate a matrix that
was trained to be full-rank and break the L2 normalization the model was trained under.
Sweep `hidden ∈ {512, 256, 128}` and take the knee.

### Why marginalization (step 1) is free money

The winning model uses **independent** heads, so its three predictions are not guaranteed
consistent — it can name a species whose genus contradicts the genus head. The UI stacks
all three levels vertically, which makes every such inconsistency directly visible to the
user as an obvious error. Computing `P(genus) = Σ P(species | species ∈ genus)` instead:

- is guaranteed consistent by construction,
- is usually *more* accurate at genus and family than a separately-trained head,
- deletes 4,435 prototype rows (~5.1 M params, 20.5 MB fp32) from the shipped artifact,
- costs one sparse matmul against a fixed 12,041-long parent-index vector.

The parent index arrays already exist as the `mask_i` buffers on `HierarchicalClassifier`.
**Verify the accuracy claim on the test fold before shipping it** — it is an expectation,
not a measurement.

---

## 2. The three product decisions that block UI work

These change what the UI is, so they need answers before any Svelte is written.

### D1 — Sample images vs. offline

The proposal requires three example photos of the predicted taxon, and requires the app to
work offline. **These conflict.** 12,041 species × 3 thumbs × ~8 KB ≈ **289 MB**, 36× the
whole model budget. Additionally, GBIF images carry mixed licences (CC-BY, CC-BY-NC, some
rights-reserved); bundling them into an installed app is redistribution and needs per-image
attribution plus a licence filter.

| option | size | offline? | licence work |
|---|---|---|---|
| **A. fetch from GBIF on demand** (recommended) | 0 | inference offline, gallery online | none — hotlinking with attribution |
| B. one 96 px WebP per species | ~12 MB | yes | per-image, substantial |
| C. genus/family thumbs only, species links out | ~4 MB | partial | moderate |

**Recommendation: A.** "Identification works offline; example photos need a connection" is
an honest and understandable contract, it costs nothing, and it removes the licensing
problem entirely. Design the empty state deliberately rather than treating it as an error.

### D2 — What the confidence number means

The proposal fixes a 0.5 threshold. Softmax over 12k cosine logits is systematically
overconfident and 0.5 has no user-facing interpretation. Greying a correct answer at 0.49
while highlighting a wrong one at 0.51 is worse than showing nothing, because the UI is
making an explicit reliability claim on the model's behalf.

**Plan:** temperature-scale on the validation fold (one scalar, minutes of work), then
choose per-level thresholds that achieve a *stated* precision on the held-out `set == '0'`
fold — e.g. "greyed below the point where 80% of shown species are correct". Ship the
thresholds inside the artifact bundle so the app never hardcodes them. The UI stays exactly
as designed; only the number's provenance changes.

### D3 — "Not a lepidopteran" is open-set recognition

Trained on moths and butterflies only, the model will confidently classify a beetle, a leaf,
or a coffee cup, because softmax normalizes over the classes it has. A low max-probability
is a weak proxy. Doing this properly needs one of:

- **energy / max-logit OOD score**, with the threshold set against an explicit negative
  image set (other insect orders, plants, indoor scenes, human faces) — that set does not
  exist in the repo and must be built;
- a **small binary gate** in front of the classifier (cheap at inference, needs the same
  negative set);
- **defer to v1.1** and ship v1 with honest copy: "low confidence — this may not be a
  moth or butterfly, or may be a species outside the training set", without claiming to
  distinguish those two cases.

**Recommendation: defer, with honest copy.** It is the most scientifically interesting item
here and the one most likely to be underestimated; it should not block a first release.

---

## 3. Ideas not in the proposal

### 3.1 Geographic prior — probably the largest available accuracy gain

A purely visual model over 12k *global* species must separate look-alikes that never
co-occur in reality. GBIF — already the data source — has the occurrence records to build a
coarse per-species spatial prior. Multiply the visual posterior by it, and a large class of
confusions disappears for free.

- Storage: 12,041 species × a coarse cell grid (e.g. ~5° lat/lon, or H3 res-2), sparse,
  quantized to int8 log-probabilities. Estimate **a few hundred KB to ~2 MB** compressed —
  cheaper than any single compression step is expensive.
- Fully offline once downloaded (the grid ships with the app; the device supplies the
  coordinate).
- Must be **opt-in** (location permission) and must **degrade gracefully** — no location
  means the uniform prior, i.e. today's behaviour.
- Risk to state clearly: a strong prior suppresses genuinely interesting records — vagrants,
  new introductions, range shifts. Use a **tempered** prior (`P_visual · P_geo^α`, α < 1)
  and never let it hide a high-confidence visual prediction outright.

Precedent: this is essentially iNaturalist's geo model, and it is the largest single
accuracy lever they found. Recommend as v1.1, planned from the start so the bundle format
has a slot for it.

### 3.2 Ship a versioned artifact bundle, not a loose ONNX

The cleanest seam between the two repos. `lepinet` emits, as a GitHub Release:

```
lepinet-artifact-v1.2.0/
  model.onnx              # quantized, bottlenecked, marginalizing
  taxonomy.json           # 12041 species → genus → family, + GBIF taxonKeys
  thresholds.json         # per-level, derived (D2), with the precision they target
  calibration.json        # temperature
  metrics.json            # test F1/top1/top5 of *this exact artifact*
  MANIFEST.json           # source run id, git sha, date, input preprocessing spec
```

`lepinet-app` pins a version. Consequences: the deployed app can always state which model it
ships; a model regression cannot silently reach users; the preprocessing spec travels with
the weights, which is what prevents §4's parity bug from recurring.

### 3.3 Top-5, not top-1, is the product metric

macro-F1 is the right research metric — it is why oversampling won ([[2026-07-does-longtail-help]]).
But it does not describe app experience, which is dominated by common species, and a user
shown five candidates can disambiguate themselves. Report **top-1 accuracy and top-5
recall** alongside macro-F1 for every compression step, and consider whether the UI should
offer "other possibilities" under a low-confidence species.

### 3.4 Warm-up inference at launch

ORT Web's first inference includes graph initialization and can take seconds. Run one
inference on a dummy tensor at app start, so the user's first real prediction is a warm one.
Costs three lines; removes the worst latency the user would ever see.

### 3.5 Store the model in Cache Storage / OPFS, downloaded explicitly

Do not let the model be an implicit page asset. An explicit "Download model (7 MB)" step
with a progress bar makes the offline contract legible, survives Safari's eviction behaviour
more gracefully (the user can re-download knowingly), and makes the size budget visible to
the user — which is the honest thing to do.

---

## 4. Preprocessing parity — the bug to prevent, not to discover

The classic silent accuracy loss in browser deployment: PyTorch's resize kernel and the
browser's canvas `drawImage` do not produce the same pixels, and a model trained at 460→256
fed a differently-resampled crop can lose several points with nothing in the UI indicating
anything is wrong.

**Test, numerically, before building any UI:** take 100 held-out images, compute logits in
PyTorch and in ORT Web on the same files, and assert max absolute logit difference and
top-1 agreement. Any disagreement is a preprocessing bug, not a quantization artifact.
Freeze the resulting preprocessing spec into `MANIFEST.json` (§3.2).

Also: **handle EXIF orientation.** Phone photos are frequently rotated, and browsers vary in
whether they auto-apply the tag.

---

## 5. Action items

Numbering continues the `dev/` sequence (last used: 039). Each script gets a `dev/README`
row; each question that produces a result gets a journal entry, per the repo's rules.

### Phase A — de-risk (do this first; it can invalidate the UI design)

| # | action | script | done when |
|---|---|---|---|
| A1 | Export the current 0.9148 model to ONNX **unmodified**. Independent head exports cleanly; confirm the cosine `F.linear` + `cosine_to_zscore` path traces. | `dev/040_onnx_export.py` | `.onnx` produced, logits match PyTorch to <1e-3 |
| A2 | Preprocessing-parity harness (§4), PyTorch vs ORT Web on 100 images | `dev/041_ort_parity.py` | top-1 agreement ≥ 99%, spec written down |
| A3 | Real-device latency: 165 MB model, WASM SIMD and WebGPU, one mid-range Android + one iPhone | throwaway page | ms/inference and cold-start numbers recorded |
| A4 | Decide D1 / D2 / D3 (§2) | — | written into the proposal |

*A1–A3 are the whole point of Phase A: they are the things that would force a redesign if
found late. 165 MB is unshippable but perfectly testable.*

### Phase B — cheap compression (no retraining)

| # | action | script | done when |
|---|---|---|---|
| B1 | Marginalize genus/family from species; drop those heads | `dev/042_marginalize.py` | genus/family F1 ≥ the independent heads' on `set == '0'` |
| B2 | int8 post-training quantization; measure per-level F1 + top-5 | `dev/043_quantize.py` | ≤ 0.5 macro-F1 lost vs 0.9148 |
| B3 | 4-bit per-row on head prototypes only | (extend 043) | keep only if ≤ 0.5 F1 lost |
| B4 | Temperature calibration + derived thresholds (D2) | `dev/044_calibrate.py` | `thresholds.json` with a stated target precision |

*Every measurement goes through `dev/032` on `set == '0'` and lands in the ledger, so
compressed variants sit in the same table as the training runs. This is the reason the
compression work is cheap to evaluate — the infrastructure already exists.*

### Phase C — the real compression (retraining)

| # | action | config | done when |
|---|---|---|---|
| C1 | Bottleneck sweep: `hidden ∈ {512, 256, 128}`, everything else fixed at the 0.9148 recipe | `configs/2026MMDD_bottleneck_*.yaml` | knee identified, F1 cost quantified |
| C2 | Backbone sweep: effnetv2-b0, mobilenetv4-conv-small, at the chosen bottleneck | `configs/2026MMDD_smallbackbone_*.yaml` | size/accuracy Pareto front |
| C3 | Distillation: 0.9148 model → chosen student, soft targets over 12k classes | `dev/045_distill.py` | student beats its from-scratch equivalent |
| C4 | Re-run B1–B4 on the student; emit artifact bundle v1 | `dev/046_bundle.py` | ≤ 8 MB, F1 documented |

*Held fixed throughout: sqrt-oversampling (`oversample_power: 0.5`), Muon, one_cycle,
warmup 0.5, grad_clip 5.0, bs 64, 460→256 — the 0.9148 recipe. Change one thing at a time;
that discipline is what makes `RESULTS.md` readable.*

**Accuracy floor to agree on before starting:** what F1 is the app allowed to ship at? A
defensible answer is **≥ 0.87 test species macro-F1** (within ~0.045 of the research model)
in exchange for a 20× size reduction. Write the number down now, so C2's Pareto front has a
decision rule instead of an argument.

### Phase D — the app (`lepinet-app`, separate repo)

| # | action |
|---|---|
| D1 | Scaffold: bun + Svelte 5 + TS + Biome, `vite-plugin-pwa` for the service worker |
| D2 | ORT Web with WebGPU-preferred / WASM-SIMD fallback; warm-up inference at launch (§3.4) |
| D3 | Model fetch into Cache Storage with explicit progress (§3.5) |
| D4 | Home → capture (native camera via `<input capture>`) → result, per the proposal's UI |
| D5 | Result view: three levels, copy affordance, confidence ring, greyed below threshold, GBIF deep link from the stored `taxonKey` |
| D6 | Sample images per D1's decision |
| D7 | GitHub Actions → Pages, pinning an artifact bundle version |
| D8 | Disclaimer: not for conservation, pest, or toxicity decisions |

### Phase E — v1.1

- Geographic prior (§3.1)
- Open-set / OOD detection with a purpose-built negative set (D3)
- Identification history (from the proposal's future avenues)

---

## 6. The regional-build variant

Worth keeping visible because it is the only route to the proposal's "<1 MB dream", and
because it may be the better *product* for many users. Filtering to a region — Denmark, or
Northern Europe — cuts the class set from 12,041 to a few thousand or a few hundred. That:

- shrinks the head roughly proportionally (a 500-species build's head is trivial),
- **raises accuracy substantially** — most confusions in a global model are between species
  that never co-occur,
- and makes sub-1 MB genuinely reachable.

The cost is a build matrix instead of one artifact, and a "which region?" question in the
UI. Note this is the same insight as the geographic prior (§3.1), applied at build time
rather than inference time — the prior is the softer, single-artifact version. Do the prior
first; treat regional builds as an optimization if size becomes binding.

---

## 7. Decisions — RESOLVED 2026-07-20 (Guillaume)

All seven are closed. Recorded here as the binding answers; the discussion above is kept for
the reasoning but the decisions win where they differ.

| # | question | decision |
|---|---|---|
| D1 | sample images | **No thumbnails at all.** Link out to GBIF instead — keeps the app light and copyright-free. A "fetch example photos from GBIF" button is a possible later addition, not v1. |
| D2 | confidence threshold | **Derive it.** `dev/044` is the threshold estimator; no hardcoded 0.5. |
| D3 | open-set / "not a lepidopteran" | **Dropped.** Not v1, not v1.1. |
| — | accuracy floor | **0.87 test species macro-F1.** |
| — | model licence | **GPL.** The obfuscation idea in the proposal is dropped. |
| — | repo split | **Confirmed.** `lepinet` emits the versioned artifact bundle as a GitHub Release; `lepinet-app` (https://github.com/GuillaumeMougeot/lepinet-app, created 2026-07-20) pins a version. The app must **show the user the model version it is running**, and offer the latest. |
| — | geographic prior (§3.1) | **Deferred to v1.1**, and noted as harder than the plan implied: it needs co-occurrence data the project does not currently have. |

Two further steers on the plan above:

- **§3.3 (top-5 / micro-F1 as the product metric)** — agreed in principle, measured later, not
  a Phase B/C gate.
- **Tiny vision transformers** are added to the C2 backbone sweep as a candidate alongside the
  CNNs (see §5 Phase C).
- **EXIF orientation** is *not* needed for this model — a lepidopteran can sit in any
  orientation, and the training augmentation includes `flip_vert`. Worth revisiting only if the
  pipeline is reused for taxa with a canonical orientation.

---

## 8. What this plan asserts that has not been measured

Stated explicitly so that a future reader can tell the predictions from the results — the
same commitment-before-results discipline as [[2026-07-does-longtail-help]]:

- int8 PTQ on the cosine head costs **< 0.2** macro-F1. *(Reasoning: unit-norm rows share
  one dynamic range.)*
- Marginalizing genus/family from species **matches or beats** the independent heads at
  those levels.
- A 256-dim bottleneck costs **< 1.0** macro-F1.
- A distilled effnetv2-b0 student lands at **≥ 0.88** test species macro-F1.
- The final bundle lands at **6–8 MB**, not 17 MB and not <1 MB.

If these hold, the target is met with margin. If the bottleneck or the student costs much
more than predicted, the fallback is a larger backbone at int8 (~23 MB) — still a 7×
reduction, still shippable, just less comfortable on iOS.


---

# Review - Guillaume

	- App review-review:
		- 8 Mb could be an achievable goal.
		- The head is more than 50% of the model weights.
		- Using quantization is the number one idea.
		- Give up on the samples for now. Let's keep it light and copyright-free. Put links towards gbif instead.
		- Create a proper threshold estimator script.
		- Give up on the open-set problem.
		- Hierarchy consistency: can we measure if it is true that using the species level to guess genus and family levels is more accurate than reading their heads? This needs proper experiment to prove it.
		- Make a comparison between Pytorch resizing and ORT resizing.
		- I don't think that EXIF orientation handling is needed because lepidopteran can seat in all three directions. But maybe it is worth doing if this classification workflow is meant to be use for other species.
		- Agreed with loading the model state when loading the app so the first prediction is not the slowest one.
		- iOS has a size constraint on PWA it seems. Confirming the need for light app.
		- Micro f1 Top5 could be a better metric. I think this could wait later to verify.
		- Geography handling requires cooccurrence data, which do not have.
		- Agreed with the repo separation: lepinet emits the versioned artifact bundle as GitHub release and lepinet-app pins a bundle version (the latest could be downloaded and user should be informed of model weight number/version).
		- Yes, add a disclaimer about model mistakes.
		- Agreed with the "Recommended sequencing" section.
	- Review of lepi-claude:
		- About different classifier: could tiny vision transform models be better?
		- Agreed with the bottleneck size reduction before the classification layer.
		- Marginalization needs to be proven.
		- D1 - do not display thumbnails and link to GBIF. Maybe later we'll add a button to image fetch from GBIF.
		- D2 - let's run experiment to guess good thresholds.
		- D3 - forget about the open-set problem.
		- 3.1 is v1.1
		- 3.2 is a beautiful idea.
		- 3.3. will be measured later.
		- 3.4. yes
		- 3.5 sure, let's do that.
		- 4. to be tested.
		- 7. open questions: for D1-3 see above. acc floor could be macro f1 species 0.87. model license is gpl. repo is split. geo prior is for later.
	- Misc:
		- I created an empty repo for the app here: https://github.com/GuillaumeMougeot/lepinet-app
		- Phase A to D can be started whenever. You do it.