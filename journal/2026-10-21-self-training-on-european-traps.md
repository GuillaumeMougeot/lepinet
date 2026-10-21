# Does self-training on traps across Europe beat our Danish traps? (U)

**Kind:** research · **Status:** **OPEN** (2026-10-21). Design and predictions; waiting for the crop
pool's location and metadata.

## The question (the owner's)

We have AMI trap crops from all over Europe, more than 50 M of them. B3rep5x used 12,230 Danish
images, each copied 11 times (2.6 % of training rows). Can a pseudo-labelled set drawn from many
traps, with no copies, bring more than that? Should the teacher be our best self-trained model? And
should the selection be as balanced as possible?

## What we already know

* **Copies were the problem, not the share.** The dose sweep reused one image set (1x / 11x / 27x /
  58x), so the "2.6 % optimum" mixes share with repetition ([[2026-08-04-replication-sweep]]).
* **A better teacher with a global confidence gate shrank coverage and hurt** (R2: labels 99.84 %
  correct, 55 % fewer species, probe -3.80). Per-species top-K selection keeps coverage (R3: +4.24).
  `dev/065 --per-species-k` does this ([[2026-08-08-self-training-does-not-iterate]]).
* **Balancing failed only by copying.** End-to-end balancing (B9) copied rare species' two or three
  images about 75 times: probe -0.71, held-out species -3.62. With 50 M crops, balance can come from
  K *different* images per species.
* **A new trap keeps the gain for species pseudo-labelled at some other trap (109 %) and loses it
  for the rare tail; a new site keeps 23 % of the macro-F1 gain** (X1, X2,
  [[2026-10-12-self-training-on-a-trap-it-never-saw]]). Many traps should spread the tail.

## Design

**Teacher: B8** (198 M, best self-trained model on probe, 0.7798), at T = 1.91. Its Danish prior is
the risk: southern look-alikes can get Danish names with high confidence. Two guards: drop
predictions outside the country's GBIF checklist, and require a species to be predicted at two or
more traps.

**Selection.** No crop from the 12 Danish probe traps, so the Danish probe stays a new site. One
crop per individual per night (consecutive-frame or `dev/088`-style twin check). Each image used
once.

**Arms, one factor each, at 20 M (scale rule), scored on the Danish probe as in X1** (macro-F1,
held-out species, accuracy, recall by species group):

| ID | pseudo-labels | share | compares with |
|---|---|---|---|
| U1 | EU, natural: top crops by confidence | 2.6 % | X2 (new site, 0.7105), B3rep5x (0.7719) |
| U2 | EU, balanced: top K per species, at most a few per trap | 2.6 % | U1 |
| U3 | winner of U1/U2 | 5, 10, 20 % | U1/U2 at 2.6 % |
| U4 | winner plus the Danish B3rep5x set | 2.6 % + 2.6 % | B3rep5x |

## Predictions (written 2026-10-21, before any run)

* **U1 lands between X2 and B3rep5x: probe 0.72-0.75.** Many traps beat the two other Danish sites
  of X2, but the fauna and hardware are not Danish.
* **U2 beats U1 by more than 1 point on macro-F1 and held-out species, and ties on accuracy.**
  Falsified if U2 loses on probe by more than the 0.0041 floor.
* **With unique images the optimum share moves above 2.6 %, to 5-10 %.** If 2.6 % still wins, the
  optimum is a property of the share, not of the copies.
* **U4 beats B3rep5x** (local images plus European breadth).

## Open before running

* Where the crops live, and whether each carries trap id, timestamp, country and detector score.
* 50 M crops are about 1.5 TB, more than UCloud has free. Run the teacher where the crops are and
  move only the selection (30-40 GB for the 20 % arm). B8 inference over 50 M crops is about half a
  day on one GPU.
