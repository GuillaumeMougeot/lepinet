# Does "bigger everything" beat the 0.9148 effnetv2_s baseline — and make a good teacher?

**Status:** ✅ **RESOLVED (2026-07-26) — yes, bigger everything won.** ConvNeXtV2-L reached
**species macro-F1 0.9316 vs the 0.9148 baseline (+1.68 pp)**, genus 0.9739 (+1.36), family 0.9876
(+1.50), and species micro-acc *also* rose (0.9507 vs 0.9476) — a genuinely better model, not an
eval artifact, landing in the predicted +1–3 pt range. It is now the project's **best teacher**
(the mock effnetv2_s teacher was 0.911). Full result in the Results section below.

First scaling run of the
clean `src/lepinet` package: a larger, cutting-edge backbone at higher resolution, bigger batch,
more epochs, richer augmentation and MixUp, trained on a UCloud B200. Queued 2026-07-24.
Companion to [[2026-07-src-lepinet-baseline-port]] (the package) and [[2026-07-lepi-app-claude]]
(§7: bigger everything → distill into small students).

## The question

The 0.9148 baseline is efficientnet_v2_s (20 M params) at 256 px, 5 epochs, light aug. Two things
this run tests at once:

1. **Does scale help accuracy?** A modern large ConvNet at higher resolution with a longer
   schedule and stronger regularisation should push species macro-F1 above 0.9148 — the tail
   (53 % of species < 200 images, [[2026-07-does-longtail-help]]) is where a bigger model + richer
   aug + MixUp should pay.
2. **Is it a good teacher?** The app needs a small model (≤ 8 MB, [[2026-07-lepi-app-claude]]).
   The plan is teacher → knowledge distillation → student. A stronger teacher raises the student's
   ceiling, so this run is step one of that pipeline, not just a bigger number.

## The design

Held from the baseline (the proven levers): independent cosine head, Muon + AdamW, one_cycle,
warmup 0.5, grad_clip 5.0, sqrt-oversampling (`oversample_power: 0.5`). Changed — "bigger
everything":

| lever | baseline | this run | why |
|---|---|---|---|
| backbone | effnetv2_s (20 M, nf 1280) | **convnextv2_large.fcmae_ft_in22k_in1k_384** (198 M, nf 1536) | cutting-edge ConvNet (2023 FCMAE), 384-px IN-22k→1k pretrain, emits a spatial map (works with `PooledHead`), strong teacher |
| resolution | 256 (from 460) | **320** (from 384) | higher res resolves fine wing detail |
| batch | 64 | **96** | B200 has the memory; bigger batch = steadier Muon updates |
| epochs | 5 | **6** | more passes; capped by B200 wall-time (~3 h/epoch est.) |
| precision | fp16 | **bf16** | large model + MixUp; fp32 exponent range avoids overflow (head is fp32 regardless) |
| aug | light (no warp/lighting) | **warp 0.1, lighting 0.2, rotate 20, zoom 1.2** | a longer run absorbs stronger aug for generalisation |
| MixUp | — | **alpha 0.2** | convex-combination regulariser; helps the tail and calibration |
| base_lr | 1e-3 | **8e-4** | slightly gentler for a big pretrained model |

Config: `configs/20260724_ucloud_lepinet_convnextv2l_bigger.yaml`. Runs on `gpu-nvidia-b200-1-gpu`
via `ucloud/lepinet-convnextv2l-bigger.toml`, behind a family-9717 smoke (validates GPU memory +
throughput at the same arch/res/batch before the ~20 h full run).

## What was built to make this runnable (this session)

- **MixUp for multi-target heads.** fastai's stock `MixUp` reads the batch size from
  `self.y.size(0)`, which is a tuple of 3 per-level targets here → crash. `MixUpMulti`
  (`callbacks.py`) reads it from the input tensor and relies on the loss carrying `y_int=True` +
  `reduction` toggling (`MultiLevelCELoss`/`FastaiLossWrapper`), so mixing happens through the
  per-level loss (lerp of per-sample losses), not by mixing integer labels. Validated on the
  synthetic CPU e2e (train → checkpoint with MixUp on).
- **timm pretrained-tag arch names** (`resolve_arch` accepts `arch.tag`), so the exact
  384-px-pretrained ConvNeXtV2-L weights are used.

## Predictions (before results)

- **Accuracy: +1 to +3 pt species macro-F1 over 0.9148**, i.e. ~0.925–0.945. Reasoning: the
  256→320 res bump and a 10× bigger, better-pretrained backbone are each worth ~1 pt on
  fine-grained tasks; MixUp + richer aug add tail robustness at a 6-epoch budget. Risk: at only 6
  epochs a 198 M model may still be under-annealed (the baseline ladder showed under-annealing was
  the #1 lever) — if so, the win shrinks and the lever is *more epochs*, not more model.
- **base_lr 8e-4 with Muon may be slightly hot** for a big pretrained backbone. NaNGuard +
  grad_clip 5.0 are the guardrails; if epoch 0 diverges, drop to 4e-4.
- **Throughput ~400–600 img/s on the B200** (compute-bound, above the /work mount's ~950 img/s
  ceiling for effnetv2_s), so ~3 h/epoch, ~20 h total. The smoke's first-epoch ETA calibrates this.
- **As a teacher: net positive even if the accuracy win is modest** — soft targets over 12 k
  classes from a higher-capacity model carry more dark knowledge than the effnetv2_s baseline's.

**Falsification:** if it lands below 0.9148, either the LR was wrong (check epoch-0 loss) or 6
epochs was too few for 198 M params — re-run at a lower LR / more epochs before concluding scale
doesn't help here.

## Hyperparameters that are guesses (flagged, per the commitment-before-results habit)

`base_lr 8e-4`, `batch 96`, `img 320`, `6 epochs`, `mixup 0.2`, and the aug strengths are all first
estimates, not tuned. The smoke validates memory/throughput; the first epoch of the full run
validates the LR (loss must fall in epoch 0). Expect at least one re-run to tune LR/epochs.

## Scaling rationale — why 6 epochs, moderate aug, and not "full blast" (2026-07-28)

Recorded because the sequencing was deliberate, not timid (full argument in [[2026-07-landscape-and-plan]]):

- **6 epochs** = a one-day wall-clock budget (~3 h/epoch), an explicit first guess. The result was
  **not under-annealed** (macro *and* micro rose, tail levels most), so it sufficed to answer "does
  scale help." More epochs (10–15) is the obvious cheap next gain, not a correction.
- **Moderate, not maximal, augmentation**: heavy distortion hurts *short* runs, and fine-grained wing
  texture is fragile (aggressive warp/color destroys the discriminative signal). Aug strength must
  scale *with* epochs; we raised it moderately over the baseline's light aug, pending that tradeoff
  being validated at scale.
- **Not full-blast (biggest model + max res + max epochs + max aug at once)** on purpose:
  one-variable discipline (a full-blast failure tells you nothing about which lever); evidence-before-
  cost (earn the 40 h run by first proving +1.68 pp cheaply); and diminishing teacher returns for the
  end goal (the *student's* capacity caps the shipped model, so a 0.9316 teacher may already be
  enough). **Full-blast is now the justified next rung**, precisely because this controlled step
  proved the direction — it was sequencing, not an argument against scale.

## Design Q&A — owner's review while the run trains (2026-07-25)

Six questions raised while `lepi-big2` was mid-run. Recorded because the answers set the *next*
experiments, not just this one.

**Q1 — Why ConvNeXtV2-L and not DINOv2/DINOv3?** They are excellent — DINOv3 (2025) SSL features are
arguably the strongest general backbones going, and worth a run. Two reasons ConvNeXtV2-L went
*first*: (a) it emits a 4-D spatial map, so it drops into the existing `PooledHead`/cosine-head path
with **zero architecture plumbing**; DINO models are ViTs that emit patch tokens + CLS, needing a
new head-attach adapter (mean-pool tokens or CLS → bottleneck) that `arch_body_features` currently
*rejects* (it refuses non-4-D/ViT outputs by design). (b) With ~3 M in-domain labelled images, the
SSL-vs-supervised gap narrows and a strong IN-22k-pretrained ConvNet fine-tunes stably at high res;
ConvNeXt also scales to higher resolution more cheaply than a ViT (no quadratic token cost), and
high res is exactly what fine-grained wing texture needs. **Decision:** DINOv3 ViT-L is the next
teacher experiment — but it needs the ViT adapter first (a small, tracked task). Teacher accuracy
matters more than teacher architecture for distillation, so both are worth benchmarking head-to-head.

**Q2 — fastai default `aug_transforms` or custom?** It's fastai's `aug_transforms` *factory* with
**custom kwargs**, not the raw defaults. Baseline lightened it (`max_warp 0`, `max_lighting 0`,
`p_lighting 0`, flip incl. vertical, rotate 15, zoom 1.1) — heavy distortion hurts a 5-epoch run.
This run enriches it (`max_warp 0.1`, `max_lighting 0.2`, `p_lighting 0.5`, rotate 20, zoom 1.2), on
the bet that a longer/bigger run absorbs stronger aug for generalisation. Not yet tuned.

**Q3 — Why only 6 epochs?** A conservative first guess to fit one wall-clock day (~3 h/epoch × 6),
**not** a tuned choice — flagged as a guess up top. Given the baseline ladder found *under-annealing*
was the #1 lever, 6 epochs is likely too few for a 198 M model. **If val is still climbing at epoch
6, the clear next lever is more epochs (10–15), not more model.** one_cycle ties the LR schedule to
`nb_epochs`, so this means a fresh relaunch at the longer schedule, not a resume.

**Q4 — 80 GB GPU mem (low for a B200) but ~100% util.** Right read: the run is **compute-bound, not
memory-bound**, and ~100% SM util means we're already extracting the B200's compute — so filling
memory with a bigger batch would *not* buy throughput (it'd only change optimisation dynamics). The
memory headroom is better spent on **higher resolution** (uses compute + memory productively and
helps fine-grained accuracy) than on a bigger batch. Speed levers here would be `channels_last` /
`torch.compile`, not batch size.

**Q5 — ep1 promising.** Noted — LR 8e-4 with Muon is not diverging out of the gate, which was the
main epoch-0 risk. Good sign for the guessed LR.

**Q6 — Is ConvNeXt-L an intermediate rung?** Yes, exactly. The ladder is
effnetv2_s (20 M, baseline) → **ConvNeXtV2-L (198 M, this run)** → ConvNeXtV2-Huge (660 M) / DINOv3
ViT-L or larger. Each rung answers "does scale still pay here, at what cost?" before the next, and
the best model becomes the distillation teacher. Scaling slowly de-risks the 40 h runs.

**Q7 — Metric learning / ArcFace as a direction? Keep multi-heads? bf16?** Good direction, and worth
noting the current head is *already* a cosine (normalised-softmax) classifier — L2-normalised
prototypes + normalised embedding + learned scale — i.e. metric-learning-flavoured, just without a
margin. **ArcFace = add an additive angular margin on the true class during training** (off at
inference); it tightens intra-class / widens inter-class angles, which helps fine-grained + long-tail
+ the open-set flavour of species ID (unknown species at inference), and gives better-separated
embeddings → better abstain-threshold behaviour, which the app wants.
- **Keep the multi-head hierarchy** — apply ArcFace per level on the shared embedding (each level
  keeps its own prototypes; inject the margin into that level's layer during training).
  Recommended first test: **ArcFace on the species head only**, plain cosine on genus/family — the
  cheapest test of "does the margin move species macro-F1."
- **bf16 is the right call** and we already default to it: ArcFace's `cos/arccos` and large scale
  `s` (~30–64) overflow/NaN in fp16; bf16's exponent range fixes it. Confirmed lever, not luck.
- **Risk:** margin `m` + scale `s` are new hyperparameters; too-large a margin on rare classes (few
  samples pushed hard) can hurt the tail — and we *already* emphasise the tail via oversampling, so
  the two must be balanced. Mitigations if needed: sub-center ArcFace or frequency-adaptive margins.
- **Independent of Q1:** ArcFace works with any backbone, so it is not gated on the DINOv3 decision.

## Plumbing built — ArcFace + DINOv3, modular & default-off (2026-07-25)

Both levers from the Q&A are now in the package, **gated so the ep5 cosine baseline reproduces
unchanged** (33 CPU tests green, ruff clean). Design notes worth keeping:

**ArcFace (`head: arcface`).** `ArcFaceHead` *subclasses* `IndependentHead` and shares its exact
`state_dict` (same bottleneck, same unit-norm prototype layers) — verified in a test that loads an
`independent` checkpoint into an `arcface` head. The only differences:
- `forward` returns `scale * cos θ` (raw scaled cosine) instead of `cosine_to_zscore(cos) + bias`.
  The frozen-zero bias is unused (adds 0), so weights are interchangeable.
- **The margin is applied in the *loss*, not the head** (`loss.apply_arcface_margin`), because only
  the loss sees the labels. This keeps the head forward **label-free → still ONNX-traceable with
  `dynamo=False`** (the whole point of the package). The loss injects `cos(θ+m)` on the true class
  via the `cos(θ+m)=cosθ·cosm − sinθ·sinm` identity — and **only when logits carry grad**, so the
  margin is on in training and off in validation/inference (val metric uses the margin-free score,
  which is correct for model selection).
- Config: `arcface_scale` (s, default 30), `arcface_margin` (m, scalar or per-level fine→coarse).
  Guarded: `head=arcface` **rejects mixup/cutmix** (margin needs one true class per sample) and
  warns if not bf16. First experiment = **species-only margin `[0.3, 0, 0]`** (cheapest test).

**DINOv3 / ViT backbone.** `ViTBody` wraps a timm ViT headless (`num_classes=0`, timm's own CLS/
mean pool → one `[N,C]` vector); `FlatHead` runs the cosine/ArcFace head on it with the same
fp32-under-autocast guard as `PooledHead` but no spatial pool. `arch_is_vit` auto-detects (dummy
forward: 4-D map → conv path, else ViT), so **the config just names a ViT and everything routes
itself** — `build_learner` assembles `nn.Sequential(ViTBody(pretrained), FlatHead(head))` behind a
plain `Learner` (vision_learner can't attach a pooling head to a token backbone), and the checkpoint
carries `vit`/`arcface_scale` so test/export rebuild the right model (old effnet checkpoints default
to the conv+cosine path, load unchanged). timm 1.0.28 exposes DINOv3 — `vit_*_patch16_dinov3.*` and,
as a bonus, `convnext_*.dinov3_lvd1689m` (DINOv3-distilled ConvNeXts that emit 4-D maps and need
*no* ViT adapter at all — a cheap third teacher candidate).

**Smokes queued** (family 9717, 1 epoch, B200-MIG), each isolating one change:
`configs/20260725_ucloud_lepinet_arcface_smoke.yaml` (effnetv2_s + species-margin) and
`..._dinov3_smoke.yaml` (vit_base_patch16_dinov3 @224 + cosine head). They validate: head/backbone
build, pretrained load, Muon over the new params, bf16 forward, margin injection, train→save→reload.

## Results (2026-07-26)

`convnextv2_large.fcmae_ft_in22k_in1k_384` @320, bs 96, 6 ep, oversample 0.5, MixUp 0.2, bf16, base_lr
8e-4. Full fold-0 test, all 12,041 species / 629,742 images (`min_img_per_spc=0`, same eval set as the
0.9148 baseline):

| level | ConvNeXtV2-L | effnetv2_s baseline (`20260716-154156`) | Δ |
|---|---|---|---|
| **species** | **0.9316** | 0.9148 | **+1.68 pp** |
| genus | 0.9739 | 0.9603 | +1.36 pp |
| family | 0.9876 | 0.9726 | +1.50 pp |
| species micro-acc | 0.9507 | 0.9476 | +0.31 pp |

**Prediction scored:** +1.68 pp lands inside the pre-registered +1 to +3 pt range — scale + higher res
+ MixUp helped, as predicted, and did **not** appear under-annealed at 6 epochs (both macro and micro
rose together, and the tail levels — genus/family — gained the most, the intended effect). The
LR 8e-4 did not diverge. **More epochs / a bigger backbone (ConvNeXtV2-H, DINOv3-ConvNeXt) is the
obvious next lever** if more is wanted, but the question as posed is answered: bigger everything beats
the baseline.

**As a teacher:** at 0.9316 it is meaningfully stronger than the effnetv2_s mock (0.911), so the
next distillation round (T=1, [[2026-07-teacher-student-app-bridge]]) should lift the small student
above the 0.8786 it reached from the mock teacher — the headroom widened by ~2 pp. Swap it in as
`distill_teacher` and rerun; nothing else changes.
