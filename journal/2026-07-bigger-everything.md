# Does "bigger everything" beat the 0.9148 effnetv2_s baseline — and make a good teacher?

**Status:** OPEN (written before results, per the journal convention). First scaling run of the
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

## Results

_(pending — the smoke + full run are queued on UCloud; the queue daemon will report. Fill in
species/genus/family macro-F1 on the fold-0 test, throughput, and whether it beat 0.9148.)_
