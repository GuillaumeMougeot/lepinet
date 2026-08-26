# BioCLIP-2 has already seen 65 % of our test fold, by exact occurrence id

**Kind:** research · **Status:** **RESOLVED (2026-08-26).** The owner's overlap hypothesis is
correct and larger than expected. **93.3 % of our species** and **65.4 % of our images** — including
**413,865 of our 629,742 test-fold images** — are inside TreeOfLife-200M, which is BioCLIP-2's
training set. Any in-distribution comparison against BioCLIP-2 is contaminated in its favour, and
**P1a's number will not mean what its committed prediction says it means.**

## Result

`dev/076_tol_overlap.py`, over the 21,166,328 Lepidoptera rows of
`imageomics/TreeOfLife-200M-Embeddings` (shards 100–161).

| | |
|---|---|
| ToL Lepidoptera images | 21,166,328 |
| ToL Lepidoptera species / genera / families | 70,453 / 12,404 / 172 |
| our vocabulary (>= 50 images) | 12,772 species |
| **our species also in ToL** | **11,916 — 93.3 %** |
| **our images in ToL, by exact GBIF occurrence id** | **4,141,385 — 65.4 %** |
| **our TEST-FOLD images in ToL** | **413,865 — 65.4 %** |

The image-level join is `gbifID` against ToL's `source_id` where `source_dataset == 'gbif'`. That is
**occurrence identity, not a name match** — the same photograph, not merely the same species.

## Why it was cheap, and how

The cache is 350 GB across 666 parquet shards, and none of it was downloaded. The dataset is grouped
`bioscan -> eol -> gbif` and then by taxonomy, so Lepidoptera was located at shards 100–161 **from
row-group statistics in the file footers alone**. Reading only the taxonomy and provenance columns
skips `emb` (768 x fp16 = 1536 bytes/row, ~99 % of the bytes), turning the job into a few GB of
strings streamed over HTTP. Whole thing runs on the laptop in about twenty minutes with the cluster
untouched.

## What this changes, in order of urgency

**1. P1a's in-distribution number is compromised as a comparison, and P1a is queued right now.**
It fits a head on frozen BioCLIP-2 features and evaluates on our global test fold — two thirds of
which is BioCLIP-2's training data. The committed range (0.86–0.91 against our 0.9148) and the
falsification line (0.80) were written assuming a fair contest. They do not describe one.

The number is still *interpretable* as "what a frozen BioCLIP-2 trunk can do here", and it is still
worth having. What it cannot support is the sentence "BioCLIP-2 is within N points of our backbone".

**The fix is a decontaminated fold**, and the same denominator rule that has bitten this project
three times applies: a clean-subset score cannot be compared against 0.9148 on the full fold either.
So it needs **both** arms re-scored on the same 34.6 % subset — our baseline and P1a. `dev/076` now
writes `test_ids_clean_of_tol.parquet` for exactly that.

**2. It reframes the 44 %.** BioCLIP-2 has seen 93 % of our species and two thirds of our images, and
still scores 44 % on trap photographs. That is not a weak representation. It is a representation that
fits the source domain extremely well and transfers poorly to a different camera — **finding 13, at
43x our data, by someone else**. It is evidence *against* training our own ToL backbone, not for it:
scaling the source domain 43x is the axis our own factorial says does not buy shift robustness.

**3. P1b is unaffected, and is now clearly the decisive run.** The probe benchmark is Flemming's trap
imagery, which is not in ToL at all — no GBIF occurrence ids, a different camera, a different
collection process. So the shifted comparison is clean on both sides. **P1b, not P1a, is the run that
answers whether the recipe needs our backbone.** That was already the more interesting question; it
is now the only uncontaminated one.

**4. The paper needs a contamination statement.** Any future comparison against a foundation model
trained on GBIF-derived data has this problem, and almost nobody checks it. Our benchmark is
GBIF-derived; so is ToL-200M; so is most of what the field will compare against. This is
[[2026-08-02-the-shifted-benchmark-is-also-the-adaptation-set]] again — a corpus quietly serving two
roles — and the response is the same: write the exclusivity down in the split, not in the pipeline.

## The thing worth generalising

**"Pretrained on a public archive" and "evaluated on a public archive" are the same sentence more
often than anyone checks.** We found this by joining on occurrence ids, which took twenty minutes and
no GPU, and it is checkable for any GBIF/iNat-derived benchmark against any of the current biological
foundation models. The reason it goes unnoticed is that the usual check is *taxonomic* overlap —
"does the model know these species?" — which is the weaker question and reassuringly answerable at
93 % without anyone realising the images are literally the same.

## Caveat on the denominator

The 12,772-species figure is the >= 50-image vocabulary recomputed here from the raw parquet; the
shipped models carry 12,041. The difference is downstream filtering and does not affect the image-level
join, which is over all 6,329,994 rows regardless of vocabulary.
