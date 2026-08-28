# Two directions: a regional checklist at inference, and our objective on Tree-of-Life

**Kind:** research · **Status:** **OPEN (2026-08-28).** Two owner-proposed directions, scoped with
hypotheses committed before any run. **D1** restricts the classifier's label space to a regional
species checklist at inference. **D2** trains our cosine z-score objective on Tree-of-Life data to
ask whether the *objective*, not the data, is what a better backbone needs.

A fact discovered while scoping D2 changes it substantially, and is the most useful thing in this
entry, so it is stated first.

## The fact that reshapes D2: ToL-10M ships images, ToL-200M does not

| repo | what is actually hosted | size |
|---|---|---|
| `imageomics/TreeOfLife-200M` | **metadata only** — `catalog.parquet` (16.8 GB) plus provenance and text embeddings. No image data | 0.05 TB |
| `imageomics/TreeOfLife-10M` | **63 image tarballs**, `dataset/EOL/image_set_NN.tar.gz` at ~32 GB each | **1.99 TB** |

So the two options are not variations of the same job:

- **ToL-200M requires a crawler.** The catalog gives URLs; the images must be fetched from GBIF, EOL
  and BioScan. At our policy that is **~84 M HTTP requests** against rate-limited providers, which is
  exactly the problem `gbifxdl` exists to solve, and the bottleneck is requests, not bytes.
- **ToL-10M requires a download.** 2.0 TB of tarballs from a CDN, sequential and resumable, no
  retry-against-rate-limit logic, no per-image quality control — the corpus is already curated.
  It fits the 7 TB budget with room for the decompressed copy.

**And ToL-10M gives the better experiment anyway.** BioCLIP-1 was trained on TreeOfLife-10M. Training
our objective on ToL-10M is therefore a **matched-corpus comparison**: same images, same taxa, one
variable — CLIP contrastive against cosine z-score with hierarchical marginal supervision. Training
on ToL-200M instead would confound the objective with a 20x data change and answer a question we did
not ask. The controlled experiment is the cheap one, which is not the usual direction for that trade.

**Recommendation: D2 runs on ToL-10M.** The 200M crawler stays available if the 10M result justifies
it, but building it first would be spending the expensive engineering before the cheap measurement
that tells us whether to bother.

### What D2 actually tests

The project has a strong prior *against* more source-domain data — our own factorial says scaling the
source domain does not buy shift robustness, and §4.14 found a fine-tuned BioCLIP-2 already beats
anything we could train. **That prior does not apply here**, and it is worth being precise about why:
every one of those results varied the *data*. None varied the **objective**. P3 compared our recipe
on their backbone; it never compared their recipe and ours on the same corpus.

The specific hypothesis: a CLIP contrastive objective aligns images to *text*, which is why frozen
BioCLIP features are unreadable by a cosine head (−16.14 pt, and §4.14.3's whole mechanism). An
objective that arranges features for a **cosine classifier with an angular margin** should produce a
trunk that is better *as a classifier trunk*, even if worse for retrieval or zero-shot. If true, the
payoff is not accuracy but **transferability of the frozen representation** — the thing the paper
currently has to caveat.

**Prediction (committed).** Our objective on ToL-10M, evaluated on Lepidoptera:

- in-distribution species macro-F1 **within ±1.5 pt of BioCLIP-2 fine-tuned (0.9146)** — i.e. the
  objective is not the lever for accuracy, consistent with everything else in this project;
- **frozen-probe** transfer **better than BioCLIP-2's by >3 pt** (BioCLIP-2 frozen: 0.8444
  in-distribution, 0.5901 probe). This is the real prediction. Falsified if the frozen probe lands
  within 1 pt of BioCLIP-2's, which would say the readout mismatch is about scale or architecture
  rather than the training objective;
- zero-shot capability lost entirely, which is not a cost we care about but should be stated.

If the frozen-probe prediction holds, §4.14.3's boundary condition stops being a caveat and becomes a
**design recommendation**: train foundation encoders with the head geometry that downstream users
will re-fit.

**Cost.** 2.0 TB download, then ~10 M images at ~1100 img/s is ~2.5 h/epoch at 20 M parameters. The
download pipeline is the only new engineering: parallel ranged HTTP with resume, checksum
verification per tarball, and streaming decompression into the existing parquet+images layout. That
is a day of work, not a week, because there is no rate limiting to design around.

## D1: restrict the label space to a regional checklist

**The observation is the owner's and it is a good one.** Every number in the paper comes from a
**12,041-class global head**, and no deployment looks like that. A Danish moth trap meets the ~500
species that occur in Denmark, and that list is knowable in advance — national checklists are
published and curated. The trap corpus's 486 species *are* essentially that list.

Constraining the classifier to a checklist is free at inference and removes 96 % of the ways the
model can be wrong. It is also squarely on the paper's stated subject, which is deployment, and it is
the one lever in the project that a practitioner can apply without touching the model.

### The design, and the two traps in it

`dev/081_restricted_label_set.py`. Masking is applied to the **species logits before the softmax**,
which is the only correct place: the posterior renormalises over the allowed set, and because every
coarser rank is derived from it by log-sum-exp over children, genus and family marginals sum over
retained children automatically. Masking after the softmax, or masking coarse heads separately,
would break the coherence that makes marginalisation worth using.

**Trap 1 — this must not become eval-set filtering.** We restrict the *label space*; the eval
parquet, its rows, its species and therefore the macro-F1 denominator are byte-identical across arms.
This project has been burned by the other thing (`--min-img-per-spc` on a test fold inflated a macro
average by ~3 pt), so the script asserts the scored frame's row count and species count do not move
between arms and refuses to continue otherwise.

**Trap 2 — the checklist must not come from the fold being scored.** Restricting to exactly the 368
species in the probe fold would leak the answer set. The checklist is the **486 species of the full
trap corpus** — a regional artefact, strictly larger than the eval set, and not chosen by looking at
it. `--checklist-from` takes the full corpus, never the fold.

### The measurement is a curve, not a point

A single restricted number is uninteresting; of course it is higher. The question is the **shape**:
how does accuracy scale with checklist size, and how much slack can a checklist carry before the
benefit vanishes? A real national checklist over-covers any single trap, so `--pad-to` grows the
checklist with random non-regional species to simulate that. Arms: unrestricted (12,041) / checklist
(486) / padded to 1,000 / 2,000 / 4,000.

**And the cost must be reported with it.** A restricted head *cannot* predict outside the checklist,
so every genuinely novel taxon becomes a guaranteed error rather than a possible one. This trades
open-set recall for closed-set accuracy, which is precisely the tension the paper is about, so the
held-out-species fold is scored too — the arm where it should hurt most.

### Predictions (committed)

- **Checklist (486) on probe: 0.72–0.78**, against the unrestricted 0.6270 for the baseline model.
  A 26x reduction in label space should be worth a great deal, because most confusions in a
  12,041-class fine-grained head are with species that are absent from the region entirely.
  Falsified below 0.68, which would say the model's errors are dominated by *locally* confusable
  congeners and the checklist cannot help.
- **The curve is strongly concave**: padding to 1,000 keeps most of the gain, padding to 4,000 keeps
  under half. A checklist that is 8x over-inclusive is still worth having.
- **Held-out species pays for it.** The restricted head should lose on the probe-held-out-species
  fold relative to its own unrestricted score, because those taxa are the ones most likely to fall
  outside a checklist built from a different capture period. This is the prediction I am least sure
  of and most want measured.
- I explicitly decline to predict whether restriction helps *more* or *less* on the better models
  (B8, P5). A stronger model has fewer confusions to remove, so the gain should shrink — but its
  errors may also be concentrated in the genuinely hard local congeners, where a checklist does
  nothing. Either result is informative.

### Why this is a paper section and not a trick

Two reasons. It is the only intervention in the project that requires **no training, no labels and no
data** — a practitioner applies it in one line. And it inverts the usual framing of open-set: the
paper spends §4.3–§4.5 asking how to *detect* taxa outside the label set, while D1 asks how much is
gained by *shrinking the label set to what is actually possible*. Those are complementary halves of
the same deployment question, and the second is much cheaper.

## Order of work

1. **D1 first** — inference-only, hours not days, and it feeds the paper's own subject.
2. **D2's download pipeline** — but for ToL-10M, and only after D1 lands, because D1 is cheap and
   D2 is a week of wall-clock even when nothing goes wrong.
