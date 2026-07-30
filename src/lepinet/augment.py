"""Optional **domain-mimicking** augmentation — off by default, additive when on.

The training images (GBIF: specimens, mostly clean backgrounds, good light) look nothing like the
deployment images (camera traps: clutter, motion blur, night-time noise, heavy JPEG). That mismatch
costs ~23 points of macro-F1 (``journal/2026-07-30-domain-shift.md``), and the cheapest thing to try is
to make training images look more like trap frames.

**Design constraint: this must not touch the existing recipe.** Every published number in this repo
was produced with fastai's ``aug_transforms(**aug_kwargs)`` and nothing else. So these transforms are
a *separate, opt-in list* appended after the standard pipeline, selected by name:

```yaml
train:
  aug_kwargs: {...}      # unchanged, still the geometric/photometric baseline
  domain_aug: trap       # NEW; absent or null => byte-identical behaviour to before
```

A preset is just a list of transforms, so a ``dev/`` experiment can register its own
(``DOMAIN_AUG_REGISTRY["mine"] = [...]``) without editing this module — the same seam
:data:`~lepinet.heads.HEAD_REGISTRY` provides for heads.

All transforms are **batch** transforms operating on GPU tensors (``[N,C,H,W]``, float, post-
normalisation ordering handled by the caller) and are **train-only** (``split_idx=0``): validation
and test must see the untouched distribution, or the metric stops being comparable to every earlier
run.
"""
from __future__ import annotations

import torch
from fastai.vision.all import RandTransform, TensorImage


class RandomMotionBlur(RandTransform):
    """Directional blur — trap frames catch moths mid-movement; GBIF specimens are still.

    A separable box blur along a random direction, applied per batch. Cheap (one conv) and the
    dominant nuisance in a timelapse frame.
    """

    split_idx, order = 0, 90

    def __init__(self, p=0.25, max_kernel=9):
        super().__init__(p=p)
        self.max_kernel = max_kernel

    def encodes(self, x: TensorImage):
        k = int(torch.randint(3, self.max_kernel + 1, (1,)).item()) | 1  # odd
        horizontal = bool(torch.rand(1).item() < 0.5)
        kernel = torch.zeros(1, 1, k, k, device=x.device, dtype=x.dtype)
        if horizontal:
            kernel[..., k // 2, :] = 1.0 / k
        else:
            kernel[..., :, k // 2] = 1.0 / k
        c = x.shape[1]
        return TensorImage(torch.nn.functional.conv2d(
            x, kernel.expand(c, 1, k, k), padding=k // 2, groups=c))


class RandomLowLight(RandTransform):
    """Darken and add sensor noise — night captures at high ISO.

    Gamma up (darker) plus Gaussian noise, which together mimic the *joint* degradation of a
    low-light frame; brightness jitter alone does not, because it leaves the noise floor untouched.
    """

    split_idx, order = 0, 91

    def __init__(self, p=0.25, max_gamma=2.2, max_noise=0.05):
        super().__init__(p=p)
        self.max_gamma, self.max_noise = max_gamma, max_noise

    def encodes(self, x: TensorImage):
        gamma = 1.0 + torch.rand(1, device=x.device) * (self.max_gamma - 1.0)
        sigma = torch.rand(1, device=x.device) * self.max_noise
        lo, hi = x.min(), x.max()
        unit = (x - lo) / (hi - lo + 1e-6)
        out = unit.clamp_min(0).pow(gamma) + torch.randn_like(x) * sigma
        return TensorImage(out.clamp(0, 1) * (hi - lo) + lo)


class RandomJPEGish(RandTransform):
    """Approximate heavy JPEG: quantise to a coarse grid after a small blur.

    True JPEG needs a CPU round-trip per image, which would halve throughput; this reproduces the
    visible effect (loss of fine texture, banding) on GPU. Fine texture is exactly what fine-grained
    wing patterns rely on, so this is a *relevant* nuisance, not a generic corruption.
    """

    split_idx, order = 0, 92

    def __init__(self, p=0.2, min_levels=12):
        super().__init__(p=p)
        self.min_levels = min_levels

    def encodes(self, x: TensorImage):
        levels = int(torch.randint(self.min_levels, 64, (1,)).item())
        lo, hi = x.min(), x.max()
        unit = (x - lo) / (hi - lo + 1e-6)
        return TensorImage((unit * levels).round() / levels * (hi - lo) + lo)


#: Named presets. ``dev/`` may register more without editing this file.
DOMAIN_AUG_REGISTRY: dict[str, list] = {
    # The first hypothesis (journal 2026-07-30-domain-shift, B1): the three nuisances we can actually
    # name for camera traps. Probabilities are deliberately modest — the point is to broaden the
    # training distribution, not to replace it with corrupted images.
    "trap": [RandomMotionBlur(p=0.25), RandomLowLight(p=0.25), RandomJPEGish(p=0.20)],
    # Ablation handle: is the win (if any) from blur alone?
    "blur": [RandomMotionBlur(p=0.35)],
    "lowlight": [RandomLowLight(p=0.35)],
}


def build_domain_aug(spec) -> list:
    """``None`` -> ``[]`` (default, nothing changes); a preset name -> its transform list.

    Kept deliberately dumb: anything cleverer (per-transform probabilities from YAML) invites a
    config that silently differs from a published run.
    """
    if not spec:
        return []
    if isinstance(spec, str):
        if spec not in DOMAIN_AUG_REGISTRY:
            raise ValueError(f"Unknown domain_aug {spec!r}; known: {sorted(DOMAIN_AUG_REGISTRY)}")
        return list(DOMAIN_AUG_REGISTRY[spec])
    raise TypeError(f"domain_aug must be a preset name or null, got {type(spec).__name__}")
