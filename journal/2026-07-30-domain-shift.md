# Closing the 23-point domain gap — is augmentation a fix or a treadmill?

**Kind:** research · **Status:** OPEN (2026-07-30). B1 (`domain_aug: trap`) has trained: in-distribution species macro-F1
**0.8999** vs A1's 0.9035, i.e. the expected small cost (−0.36 pp) for training on deliberately
degraded images. **That number decides nothing** — the whole hypothesis is about the *shifted*
benchmark, queued as `lepi-B1-shift`. Recorded here only so the cost is on the books before the
benefit is known. Hypotheses before experiments. Feeds the same paper as
[[DIRECTIONS]] — a new *section*, not a new project: the flemming result showed
novelty detection degrades under shift, so robustness sits **upstream** of the open-set contribution
rather than beside it.

## The gap, measured

| | in-distribution | flemming (different source) |
|---|---|---|
| species macro-F1 (ConvNeXtV2-L) | 0.9316 | 0.6950 |
| open-set AUROC (ArcFace × z-score) | 0.9115 | 0.7272 |

Both halves of the system degrade together, and the logits say *why*: under shift the **known**
max-logit mean falls 32.6 → 20.6 while novel sits at 14.1. Shift makes *familiar* species look
unfamiliar. So a domain-shifted embedding does not merely lose accuracy — it corrupts the very score
the open-set mechanism reads.

## The owner's question: is augmentation the long-term answer?

**No, and it is worth being explicit about why**, because "add more augmentation" is the default
reflex and it has a specific failure mode: **it only removes nuisances you can name.** Motion blur,
low light, JPEG artefacts, background clutter — each is a hand-authored guess about what differs
between GBIF photographs and trap frames. That is:

- **Not generalisable.** The next deployment (a different trap, a different country, a phone camera)
  has a different nuisance set, and the tuning restarts. The effort scales with the number of
  domains, which is exactly the wrong asymptotics.
- **Bounded by imagination.** It cannot fix a shift nobody thought of, and the residual is invisible
  — you only learn what you missed when the numbers stay low.
- **Real but shallow.** It should still be measured first, because it is nearly free and it
  *calibrates the problem*: if hand-authored augmentation recovers most of 23 points, the gap was
  mostly nuisance variation; if it recovers little, the shift is semantic (different species mix,
  different pose/scale distribution, labelling conventions) and no amount of pixel jitter will help.

**That diagnostic value is the real reason to run it first** — not because it is expected to be the
solution.

**Where hand-authored preprocessing *is* the right answer:** as an explicit, documented **user-facing
knob**, not a hidden default. A user specialising the general classifier to their own trap should be
able to say "my images look like *this*" (e.g. background removal with something like flatbug, a
fixed crop, a colour profile). That reframes manual tweaking from a fragile internal hack into a
*deployment interface* — which is defensible, and honest about who does the work.

## The ladder (cheapest → most general)

1. **Domain-mimicking augmentation.** No OOD data at all. Diagnostic first, fix second.
2. **Background suppression / subject cropping** (flatbug-style). Attacks the single biggest
   *nameable* difference — GBIF specimens are centred on plain backgrounds, trap frames are cluttered
   — and, unlike colour jitter, it removes a whole *class* of nuisance rather than one instance of it.
   Also a natural user-facing knob.
3. **Self-training / UDA on unlabelled OOD images.** Pseudo-label trap images with confidence
   gating, mix into training, keep the full 12 k-species head. Uses OOD *images* but no OOD
   *labels* — respecting the owner's constraint. This is the first rung that adapts to shifts nobody
   named, so it is the first genuinely general one.
4. **Robust pretraining.** The teacher is the ceiling for the distilled student, and DINOv3 features
   are already trained for exactly this kind of invariance. Cheapest general lever we have not pulled:
   compare backbones *on the shifted benchmark*, not just in-distribution — we have never once ranked
   backbones by their flemming score.

Ordering rationale: 1–2 are cheap and diagnostic; 3 is where generality starts; 4 may already be
free (we have the checkpoints) and is the most likely to transfer to the *next* domain.

## Evaluation protocol (the part most likely to produce a wrong answer)

Flemming is **timelapse**, so near-duplicate frames abound. Getting this wrong inflates everything:

- **Grouped splits only** — by capture event / night / location. A random split leaks near-duplicates
  across train and test and can manufacture double-digit phantom gains.
- **Validate on held-out *species*, not just held-out images.** The stated risk is the model
  specialising on the 500 flemming species; only unseen species can detect that.
- **Report the triple every time**: in-distribution F1 (did robustness cost accuracy?), flemming F1
  (did it help?), and open-set AUROC (did it help the score we actually deploy?). Optimising any one
  alone is how this direction goes wrong.

## Predictions (committed)

- **H1** Augmentation recovers **3–8 points** of the 23 (partial, not decisive). If it recovers > 15,
  I am wrong about the shift being partly semantic — and that would be excellent news.
- **H2** Background suppression beats generic augmentation per unit of effort, because it removes a
  nameable *category* rather than sampling one nuisance dimension.
- **H3** Self-training helps more than either but risks confirmation bias on the 500 species;
  the held-out-species check is what makes this claim falsifiable.
- **H4 (the cheap one)** Backbone choice matters *more* under shift than in-distribution — the
  in-distribution spread across our backbones is ~2 points, and I expect the flemming spread to be
  larger. Testable today with checkpoints we already have.

**Dead-end criterion:** if 1–3 each yield < 2 points and H4 shows no backbone spread either, the gap
is not "domain shift" in the adaptable sense but a label/definition mismatch, and the honest paper
sentence becomes "cross-source generalisation is an open problem", with the measurement as the
contribution.
