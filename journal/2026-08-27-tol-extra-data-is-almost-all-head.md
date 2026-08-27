# ToL's "3.1x more data" is almost entirely head, and at our cap policy it is worth 1.2x

**Kind:** research · **Status:** **RESOLVED (2026-08-27).** Measured on ToL's own per-species counts.
The owner's recollection was accurate: **the top 10 Lepidoptera species hold 1,628,035 images**, 7.8 %
of the corpus. Once our ~2,000/species cap is applied to ToL, the extra data available for species we
already model drops from **+12.96 M (3.1x)** to **+1.26 M (1.2x)**. This weakens the ToL-data
direction considerably and **corrects a framing I gave earlier the same day**.

## The distribution

20,880,218 images over 70,453 species.

| rank | species | images |
|---|---|---|
| 1 | *Danaus plexippus* | **300,632** |
| 2 | *Vanessa atalanta* | 237,866 |
| 3 | *Pieris rapae* | 192,365 |
| 4 | *Vanessa cardui* | 153,676 |
| 5 | *Pararge aegeria* | 142,674 |

| | share of corpus |
|---|---|
| top 10 species | **7.8 %** |
| top 100 | 26.9 % |
| top 1,000 (1.4 % of taxa) | **64.3 %** |
| top 5,000 | 88.4 % |

The comparison that matters for a **macro** metric: uncapped, the **top 10 species would take 7.8 %
of every epoch while the 65,453 rarest species combined take 11.6 %.** Ten taxa drawing almost as
much gradient as sixty-five thousand is not a long tail with a head, it is a head with a tail
attached.

Our own capped distribution, for contrast: top 10 = **0.7 %** of images. The cap is doing a great
deal of work.

## The correction

Earlier today I wrote that ToL holds "19.1 M images against our 6.2 M for the same 12,494 species --
3.1x more", and offered that as the case for revisiting the head cut. **That number is true and
misleading**, because it silently compares our *capped* corpus against their *uncapped* one. The
like-for-like figures:

| available for our 12,494 species | images | vs ours |
|---|---|---|
| **we currently have** | **6,152,933** | — |
| ToL capped at 2,000 (our policy) | 7,408,185 | **1.2x** (+1.26 M) |
| ToL capped at 3,000 | 8,804,849 | 1.4x (+2.65 M) |
| ToL capped at 5,000 | 10,667,951 | 1.7x (+4.52 M) |
| ToL uncapped | 19,116,737 | 3.1x (+12.96 M) |

So **~90 % of the apparent surplus is head images beyond 2,000 per species** -- exactly the images our
construction discarded on purpose, and exactly the ones a macro-F1 metric values least, since the
species holding them are already the easiest in the set.

## What this changes

**The ToL-data direction is much weaker than the 3.1x suggested.** At our policy the offer is 20 %
more images, concentrated in species we already classify well. That is not nothing, but it is not a
reason to build a data pipeline.

**"Restore the head" is now clearly the wrong framing, and it was mine.** Importing ToL's
distribution would hand 1.4 % of every epoch to *Danaus plexippus* alone. The defensible question is
narrower: **where is the optimal cap, and is 2,000 near it?**

**L7 is unchanged and is now the more valuable experiment**, because locating that optimum is the
only thing that decides whether +1.26 M / +2.65 M / +4.52 M is worth acquiring. Its sweep (250 / 500 /
1,000 against uncapped ~2,000) brackets the range we can test with local data.

**But its extrapolation must be stated carefully.** A slope measured between 250 and 2,000 says
nothing about caps of 10^5. If L7 is still climbing at 1,000, that justifies acquiring up to perhaps
3,000-5,000 per species -- not restoring the full head. I will score the committed prediction as
written rather than revise it: it concerns *our* curve, and this measurement concerns *ToL's*
distribution, so it is not evidence about the thing predicted.

## Method note

Nothing was downloaded for this. It reuses `tol_species_counts.csv`, written by `dev/076` during the
contamination scan, which came from taxonomy columns read out of parquet footers. The whole question
was answered from a 1.5 MB CSV already on disk.
