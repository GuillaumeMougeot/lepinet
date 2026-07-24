"""Single-image / folder inference with test-time augmentation (TTA).

This is the deployment-style prediction path (the ``evaluate`` path in :mod:`lepinet.test` uses
fastai's ``test_dl`` to reproduce validation numbers exactly). Preprocessing here is the simple,
explicit contract the ONNX export also bakes in (``resize → [0,1] → ImageNet normalize``); resize
kernel differences were measured to be a non-issue (``journal/2026-07-lepi-app-compression.md``).

TTA averages per-level softmax over the four flips the model was trained to be invariant to
(identity / hflip / vflip / hflip+vflip — training used ``flip_vert=True`` plus the default
hflip). It is four cheap forward passes and is the natural, near-free accuracy bump for this model.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class LevelPrediction:
    """Top-k predictions for one taxonomic level."""

    level: str
    labels: list[str]
    confidences: list[float]

    @property
    def top(self) -> tuple[str, float]:
        return self.labels[0], self.confidences[0]


@dataclass
class ImagePrediction:
    """Per-level predictions for one image."""

    path: str
    levels: list[LevelPrediction]

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "predictions": {
                lp.level: [{"label": lab, "confidence": c} for lab, c in zip(lp.labels, lp.confidences)]
                for lp in self.levels
            },
        }


def _load_image_tensor(path: str | Path, img_size: int) -> torch.Tensor:
    """Load an image → normalized CHW float tensor ``[3, img_size, img_size]`` (RGB, ImageNet norm)."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def _tta_views(x: torch.Tensor) -> list[torch.Tensor]:
    """The four flip views (identity, hflip, vflip, hflip+vflip) of a batch ``[N,3,H,W]``."""
    return [x, torch.flip(x, [-1]), torch.flip(x, [-2]), torch.flip(x, [-2, -1])]


@torch.no_grad()
def predict_tensors(model, x: torch.Tensor, levels: Sequence[str], vocabs: dict,
                    tta: bool = True, topk: int = 5, device=None) -> list[ImagePrediction]:
    """Predict on a preprocessed batch ``x`` ``[N,3,H,W]``. Returns one :class:`ImagePrediction` per row.

    With ``tta`` the per-level softmax is averaged over the four flips before top-k.
    """
    device = device or next(model.parameters()).device
    x = x.to(device)
    views = _tta_views(x) if tta else [x]

    probs_sum = None
    for v in views:
        out = model(v)  # list of per-level logits
        p = [torch.softmax(o.float(), dim=1) for o in out]
        probs_sum = p if probs_sum is None else [a + b for a, b in zip(probs_sum, p)]
    probs = [p / len(views) for p in probs_sum]

    vocab_arrays = [np.array([str(v) for v in vocabs[level]]) for level in levels]
    n = x.shape[0]
    results: list[ImagePrediction] = []
    for r in range(n):
        lp = []
        for i, level in enumerate(levels):
            k = min(topk, probs[i].shape[1])
            conf, idx = probs[i][r].topk(k)
            lp.append(LevelPrediction(
                level=level,
                labels=vocab_arrays[i][idx.cpu().numpy()].tolist(),
                confidences=conf.cpu().numpy().tolist(),
            ))
        results.append(ImagePrediction(path="", levels=lp))
    return results


def _iter_images(images: str | Path | Sequence[str | Path]) -> list[Path]:
    """Normalize an image / folder / list argument to a flat list of image paths."""
    if isinstance(images, (str, Path)):
        p = Path(images)
        if p.is_dir():
            return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
        return [p]
    return [Path(i) for i in images]


def predict(
    model,
    images: str | Path | Sequence[str | Path],
    img_size: int = 256,
    tta: bool = True,
    topk: int = 5,
    batch_size: int = 32,
    device=None,
) -> list[ImagePrediction]:
    """Predict taxonomy for an image, a list of images, or a folder (recursive).

    ``model`` may be a loaded model or a checkpoint path (``.pt``). Returns a list of
    :class:`ImagePrediction`, one per input image, each carrying top-``k`` per-level labels +
    confidences (TTA-averaged by default).
    """
    from .test import load_model

    levels = vocabs = None
    if isinstance(model, (str, Path)):
        import torch as _t

        checkpoint = _t.load(model, map_location="cpu", weights_only=False)
        model, meta = load_model(checkpoint, img_size=img_size)
        levels, vocabs = meta["levels"], meta["vocabs"]
    else:
        if not hasattr(model, "_lepinet_meta"):
            raise ValueError("Pass a checkpoint path, or attach model._lepinet_meta = {'levels','vocabs'}.")
        levels, vocabs = model._lepinet_meta["levels"], model._lepinet_meta["vocabs"]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    paths = _iter_images(images)
    if not paths:
        return []

    out: list[ImagePrediction] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        x = torch.stack([_load_image_tensor(p, img_size) for p in chunk])
        preds = predict_tensors(model, x, levels, vocabs, tta=tta, topk=topk, device=device)
        for p, pred in zip(chunk, preds):
            pred.path = str(p)
            out.append(pred)
    return out
