# P1: does the recipe need *our* backbone, or any strong encoder?

**Kind:** research · **Status:** **RESOLVED (2026-08-27). The recipe is NOT trunk-agnostic — the
backbone stays in the contribution list.** P1b was falsified by 11 points: BioCLIP-2 frozen and
adapted reaches probe **0.5901** against T2b's **0.7515** on the identical procedure, a **16.14 pt**
deficit. The gap that is 5.77 pt in-distribution **triples under domain shift**.

*(earlier status)* On a decontaminated
fold, a frozen BioCLIP-2 trunk with a fitted classifier scores **0.8444 against our baseline's
0.9021 — a 5.77 pt gap**. BioCLIP-2 as a frozen
trunk with our two classifier stages on top. If it works, the contribution is the *procedure*, not
the backbone.

## Why this is the right experiment now

T2b is the entry that forces it. Classifier-only adaptation recovered the full gain from **A1's**
trunk — a representation trained with no domain augmentation, which had never seen the target domain
in any form, real or imitated. The conclusion recorded there:

> If adaptation needs nothing from the representation, **it does not need *ours*.**

That sentence has been sitting in the backlog since 6 August with the owner's "we'll do that later"
against it. It is now the most load-bearing untested claim in the project, because everything the
paper says about staged classifier work implicitly assumes our trunk is doing something a different
trunk would not.

**BioCLIP-2 is the encoder that makes the test sharp.** `imageomics/bioclip-2` is a ViT-L/14 CLIP
model trained on TreeOfLife-200M — this exact domain, at a scale we cannot approach. If our two cheap
classifier stages work on top of it, the pipeline reduces to *"take the best available encoder, fit a
classifier, adapt it"*, which is both a stronger claim and a more useful one than anything about our
backbone. If they do not, the backbone earns its place in the contribution list.

## Design

Two stages, trunk frozen throughout, mirroring F2 with the trunk swapped:

| | what | epochs |
|---|---|---|
| **P1a** | classifier fitted from scratch on the global data | 3 |
| **P1b** | that classifier adapted on target-domain pseudo-labels | 2 |

`dev/075_pretrained_trunk.py`. Head is `independent` — the plain 3-level cosine head of the 0.9148
baseline — so the comparison is against a number we know well and no dev-registered head is involved.
`oversample_power: 0.0`, because L5 showed resampling costs 1.5–2.9 pt under shift and rebalancing
belongs in the classifier stage; this keeps P1a a clean probe of the representation.

**It attaches without touching the package.** `build_learner` constructs a ViT backbone as
`ViTBody(arch_spec, pretrained=True)`, and the test/export rebuild path calls the same class, so
`ViTBody` is the only seam needed: `dev/075` swaps it for a factory returning a `BioCLIPBody` when the
arch name is the sentinel `bioclip2`. `resolve_arch` / `arch_is_vit` / `arch_body_features` are
patched alongside. Nothing in `src/lepinet` changes, so no published number can drift.

## Two things that would have silently handicapped the encoder

**Normalisation.** `make_dls` hard-codes ImageNet statistics; CLIP models use their own
(0.481/0.458/0.408, 0.269/0.261/0.276). Feeding ImageNet-normalised pixels to a CLIP tower does not
crash — it is a quiet few-point loss that would have read as "BioCLIP-2 is not that good". So
`BioCLIPBody` **undoes** the ImageNet normalisation and applies CLIP's, exactly, inside `forward`,
with the round-trip asserted in the selftest to 1e-5.

**Which feature.** open_clip's `visual(x)` returns the *projected* 768-d embedding, trained for
alignment with text. For transfer we want the pooled **pre-projection** feature, width 1024, so `proj`
is set to `None`. Both widths are asserted in the selftest so the choice cannot silently flip.

## The preflight, and what it caught

The encoder download and the `open_clip` install were the only genuinely untested steps, so they got
their own short job ahead of the real chain. It earned its keep twice:

1. **`uv pip install open_clip_torch` started pulling a 508 MB PyPI torch**, which would have replaced
   the image's pinned `2.12.1+cu130` build and changed the CUDA stack under the run. Fixed with
   `--no-deps`; the job now prints `torch.__version__` and `cuda_available` before the selftest so a
   regression is visible in the log rather than in a number three hours later.
2. `--no-deps` then meant `ftfy`'s own dependency was missing (`ModuleNotFoundError: wcwidth`), which
   `--no-deps` obliges us to name explicitly.

Neither would have been a mystery for long, but both would have been discovered inside a multi-hour
job instead of a five-minute one.

## Predictions (committed)

**P1a — in-distribution species macro-F1 0.86–0.91**, against our end-to-end baseline's 0.9148. The
encoder saw ~200 M organism images including Lepidoptera, so the features should be strong; but it
was trained for text alignment rather than for separating 12,041 species, and our baseline's trunk was
trained for exactly this task. **Falsified below 0.80**, which would say a frozen foreign encoder is
not a viable starting point and P1b is not worth running.

**P1b — probe 0.72–0.78.** The direct analogue is T2b (our trunk, frozen, classifier adapted) at
probe **0.7515**. Matching or beating that makes the recipe trunk-agnostic. **Falsified below 0.70**,
which would mean the adaptation recipe does depend on properties of our representation after all —
itself a useful result, and the one that keeps the backbone in the contribution list.

**Cost is unmeasured.** Throughput for a frozen ViT-L/14 at 224 on a B200 is not something this
project has measured; 1100 img/s is effnetv2-s at 256 and 480 img/s is the 198 M ConvNeXt. It will be
read from the first epoch rather than extrapolated from either — the 2.3x costing error on B10 came
from exactly that kind of extrapolation.

## What each outcome changes

- **P1b matches T2b (~0.75)** — the paper's claim becomes "two cheap classifier stages on any strong
  encoder", the backbone leaves the contribution list, and P2 (centroids instead of a trained
  classifier) becomes the obvious follow-up: encoder + centroids + 2 epochs, no trained head at all.
- **P1b clearly below T2b** — the representation matters after all, and the honest statement is that
  the staged recipe is tied to a trunk trained on the task. Also worth knowing, and it protects the
  paper from a reviewer running this experiment for us.
- **P1a below 0.80** — stop; a frozen foreign encoder is not a viable starting point at this
  granularity, and the fine-grained gap between 200 M-image pretraining and task-specific training is
  itself reportable.


---

## P1a: 0.8444 vs 0.9021 on a decontaminated fold (2026-08-27)

**Predicted in-distribution 0.86–0.91, falsified below 0.80. Landed 0.8629 on the full fold —
inside the range.** But the full fold is 65.4 % BioCLIP-2's own training data
([[2026-08-26-bioclip2-has-seen-two-thirds-of-our-test-fold]]), so both arms were re-scored on the
219,048-image / 11,998-species clean subset:

| | full fold | clean fold | change |
|---|---|---|---|
| our baseline | 0.9148 | **0.9021** | −1.27 |
| **P1a** (BioCLIP-2 frozen + fitted head) | 0.8629 | **0.8444** | −1.85 |
| **gap** | −5.19 | **−5.77** | |

**Contamination was worth about 0.58 pt to P1a**, not the several points I expected from a 65 %
overlap. Most of the −1.27 / −1.85 drop is the *denominator*, not the leak: the clean subset has a
median of 8 images per species against 20, which makes it a harder benchmark for everyone. The
contamination-specific effect is the **difference** of the two drops, and it is small.

That is a more interesting result than a large leak would have been. It says a fitted classifier on
frozen features does not memorise its trunk's training images the way a fine-tuned network would —
consistent with the trunk being frozen and only 12,041 prototypes being fitted.

**The 5.77 pt gap is the headline, and it is real** — far outside any plausible spread on an
in-distribution metric (measured spread 0.0010). A frozen BioCLIP-2 trunk is meaningfully behind a
task-trained one at this granularity, even though BioCLIP-2 saw 93 % of our species and 200 M
organism images.

Coarse levels tell the same story: genus 0.9125 vs 0.9525, family 0.9511 vs 0.9683.

**What it does not decide.** P1a is the *in-distribution* arm, which was never the interesting one.
The question is whether our classifier stages close that gap on the **shifted** benchmark, where the
comparison is uncontaminated and where the project's subject actually lives. That is P1b, running now
after a mount fix.


---

## P1b: falsified by 11 points, and it answers the question (2026-08-27)

**Predicted probe 0.72–0.78, falsified below 0.70. Landed 0.5901.**

| | trunk | procedure | probe | held-out |
|---|---|---|---|---|
| **T2b** | ours (A1, 20 M, no domain aug) | frozen, classifier adapted | **0.7515** | — |
| **P1b** | BioCLIP-2 (ViT-L/14, 303 M, 200 M images) | frozen, classifier adapted | **0.5901** | 0.5599 |
| Δ | | | **−16.14** | |

Same stage, same pseudo-labels, same head, same 2 epochs, same frozen-trunk protocol. **One factor
changed: the representation.** And the trunk trained on 6.3 M images of our own task beats the trunk
trained on 200 M organism images by sixteen points.

## The shape of the result is the interesting part

| axis | BioCLIP-2 frozen vs ours |
|---|---|
| in-distribution (P1a vs baseline, decontaminated fold) | **−5.77** |
| under domain shift (P1b vs T2b, probe) | **−16.14** |

**The deficit triples when the camera changes.** In-distribution, BioCLIP-2's features are merely
behind; under shift they fall apart. That is not a small-data-vs-big-data story — BioCLIP-2 has 33x
more images and had seen 93 % of our species and 65 % of our exact photographs.

## What it retracts, and what it establishes

**T2b's extrapolation was wrong, and it was mine.** That entry concluded:

> If adaptation needs nothing from the representation, **it does not need *ours*.**

The first clause is still true — adaptation recovers 83 % of the gain from a frozen trunk. The
inference to "any trunk" does not follow, and P1b is the counterexample. T2b showed adaptation works
from *a trunk trained on this task without domain augmentation*; it never showed adaptation works
from *an arbitrary trunk*. I generalised one step too far and it took a run to catch.

**The backbone earns its place in the contribution list.** The paper cannot claim "take the best
available encoder, fit a cheap classifier, adapt it". The correct claim is narrower and better
supported: *given a representation trained on the task, the remaining work is cheap and lives in the
classifier.*

**And it protects the paper from a reviewer running this experiment for us**, which was one of the
two reasons for doing it.

## What it does NOT establish

**Whether BioCLIP-2's representation is bad, or merely badly read while frozen.** P1b freezes the
trunk. A fine-tuned BioCLIP-2 may recover most or all of the gap, in which case the honest statement
becomes "their representation is fine but must be adapted, not probed". **P3** (three LR arms,
running) tests exactly this, and until it lands the interpretation of P1b is limited to frozen use.

This is the same confound that limited P1a, and it is worth stating that both P1 arms share it: the
whole of P1 measures *frozen* BioCLIP-2, which is a deployment mode, not a verdict on the encoder.
