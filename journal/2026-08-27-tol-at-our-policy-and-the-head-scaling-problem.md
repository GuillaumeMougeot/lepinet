# ToL-200M under our own data policy is 88 M images and 204 K species — and that head fits

**Kind:** research · **Status:** **RESOLVED (2026-08-27).** All 666 shards scanned. Applying our own
`min_img_per_spc = 50` and a 2,000/species cap to TreeOfLife-200M leaves **88,107,628 images across
203,878 species**. The second number matters more than the first: **a 1280 x 203,878 prototype matrix
is 1.04 GB, or 3.13 GB with Adam state — it fits on one device.** Our own tail-filtering policy
defuses most of the head-scaling problem that Group H spent five failed experiments on.

## The corpus

233,055,986 rows, **884,662 species**.

| rank | distinct values |
|---|---|
| kingdom | 191 |
| phylum | 1,265 |
| class | 4,169 |
| order | 8,357 |
| family | 27,166 |
| genus | 138,171 |
| species | 884,662 |

It is much flatter than Lepidoptera: the top 10 species hold **1.77 %** of images against
Lepidoptera's 7.8 %, and the top 1,000 hold 28.7 % against 64.3 %.

**But the cap bites just as hard.** 40.7 % of images survive a 2,000/species cap corpus-wide, against
42.8 % for Lepidoptera alone. **My prediction of 50–70 % was wrong**, and the reasoning behind it was
wrong in an instructive way: I argued "flatter distribution -> less lost to the cap", which is true
per species and false in aggregate. With 884,662 species there are simply far more species sitting
above 2,000 in absolute terms, even though no single one dominates.

| cap | images | with min 50 |
|---|---|---|
| 250 | 40.3 M | 33.6 M |
| 500 | 55.2 M | 48.6 M |
| 1,000 | 73.4 M | 66.8 M |
| **2,000** | **94.7 M** | **88.1 M** |
| 5,000 | 126.8 M | 120.2 M |

## What it would cost

At min 50 / cap 2,000 — **15.5x our images, 16.1x our species**:

| | |
|---|---|
| compute | **22.2 h/epoch**; 67 h for 3 epochs, 111 h for 5 |
| storage | 6.8 TB at our current 77 KB/image, **2.2 TB at 256 px** |
| download | **~84 M new images** (we already hold 4.1 M) |
| prototype matrix | **1.04 GB, 3.13 GB with Adam — fits** |

**The download remains the binding constraint**, and reducing image size does not help it: the
bottleneck is 84 M HTTP requests against rate-limited providers, not bytes.

## The Group H consequence, which is the real find

Group H closed with five measured failures because a 1 M-species head needs 5.12 GB of parameters
plus 10.24 GB of Adam state, and the optimiser state is what does not fit
([[2026-08-09-can-centroids-be-trained-against]]).

**A min-50 floor turns 884,662 species into 203,878** — a 4.3x reduction — and with it 15.36 GB into
**3.13 GB**. The problem Group H could not solve by architecture is solved by a data policy we
already apply for unrelated reasons.

That does not retract Group H's findings; low-rank, sampled softmax, taxonomy codes and proxy-free
centroids all still fail on their own terms. It reframes the *question*: at Tree-of-Life scale the
honest answer is **"apply an image floor, then train an ordinary matrix"**, and the interesting
residue is what the floor costs in coverage — 680,784 species dropped, but only ~7 % of images.

## Caveat on the upper ranks

The 191 kingdoms and 1,265 phyla are not credible as taxonomy — there are about seven kingdoms. Four
sampled GBIF shards are perfectly clean (2 kingdoms, 3 phyla, 100 % coverage by the top two), so the
noise is concentrated elsewhere, most likely in the `eol` and `bioscan` portions whose taxonomy is
less curated.

**This matters because "seven taxonomic levels" is the strongest argument for the whole ToL
direction**, and it is currently unverified above family. Before relying on kingdom/phylum/class for
a hierarchical claim, someone has to measure what fraction of rows carry a *valid* value at each
rank. That is a cheap follow-up on data already scanned, and it should happen before, not after, any
download.

## L7's first arm, for context

`cap 250` — 37 % of our training data — against the uncapped baseline:

| | cap 250 | uncapped | Δ |
|---|---|---|---|
| in-distribution | 0.8783 | 0.9148 | **−3.65** |
| probe | 0.5776 | 0.6270 | **−4.94** |
| held-out species | 0.6248 | 0.6412 | −1.64 |

**Predicted 0.885–0.900 in-distribution; landed 0.8783, below the range** by 0.7 pt — the curve is
steeper than I expected. And the loss is **larger under shift than in-distribution** (−4.94 vs
−3.65), which is consistent with P1b: representation quality dominates off-distribution, and images
are what buy a representation. `cap 500` and `cap 1000` will say whether it is still climbing at our
current ~2,000.
